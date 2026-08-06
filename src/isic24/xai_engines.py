"""Explainability engines for the modality-attendance study.

Moved verbatim out of the notebooks (training cells 26/28/33) so that SHAP, Grad-CAM and
LORE can be run against ANY saved checkpoint, not only during the run that trained it.
The bodies are unchanged; see `xai_driver` for the orchestration that used to live in
the training notebook's shared runner.

Where a helper existed in two cells, the version kept here is the one that ACTUALLY RAN:
cell 33 executes after cell 26 and shadows its plot helpers, so every published figure
used cell 33's (coarser `fmt="{:.2f}"`) barplot.
"""

import os
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import tensorflow as tf
import keras
from keras import Model
from keras.callbacks import Callback

PLOT_SHOW = True

# The Grad-CAM grid is `image_size / 32`: 8x8 at 256 px. Trimming a 1-pixel border to avoid
# padding artifacts costs 44% of the cells at that size, and gives the engine a different map
# from the one the faithfulness path scores. One map, untrimmed, is the default.
TRIM_CAM_BORDER = False


def append_epoch_row(path, row, key="epoch"):
    """Append one per-epoch row, REPLACING any existing row for the same epoch.

    These two CSVs (`shap_summary.csv`, `lore_fidelity_per_epoch.csv`) were opened in
    plain append mode. That was harmless while they were only ever written once, during
    the run that trained the model. It stops being harmless now that the whole point is
    to re-run the engines post-hoc against an existing `run_dir`: a second pass appended
    a duplicate epoch, and `plot_lore_fidelity_vs_val_auc_by_epoch` reads the second file.
    """
    path = Path(path)
    new = pd.DataFrame([row])
    if path.exists():
        try:
            old = pd.read_csv(path)
            if key in old.columns:
                old = old[old[key] != row[key]]
            new = pd.concat([old, new], ignore_index=True)
            if key in new.columns:
                new = new.sort_values(key).reset_index(drop=True)
        except Exception:
            pass          # unreadable/corrupt file: fall back to writing just this row
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False)



# --------------------------------------------------------------------------------------
# Shared plotting helpers (cell 33 -- the executed variant)
# --------------------------------------------------------------------------------------

def _finalize_plot(out_dir=None, fname=None, show=None):
    plt.tight_layout()
    if out_dir and fname:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
    should_show = PLOT_SHOW if show is None else bool(show)
    if should_show:
        plt.show()
    plt.close()


def encoded_indices_for_raw(encoded_feature_names, feat_name):
    idxs = []
    for j, name in enumerate(encoded_feature_names):
        if "__" in name:
            if name.split("__", 1)[0] == feat_name:
                idxs.append(j)
        elif name.startswith(f"{feat_name}_bin_"):
            idxs.append(j)
        elif name == feat_name:
            idxs.append(j)
    return np.array(idxs, dtype=int)


def select_top_numeric_features(global_importance_raw, raw_batch_df, k=4):
    """Return top-k raw features among those that are numeric-only."""
    numeric_feats = [
        f for f in global_importance_raw.keys()
        if f in raw_batch_df.columns and raw_batch_df[f].dtype.kind in "if"  # int or float
    ]

    ordered = sorted(
        numeric_feats,
        key=lambda name: global_importance_raw[name],
        reverse=True,
    )

    return ordered[:k]


def plot_topk_grouped_raw_global_pos_neg(
    global_imp, pos_imp, neg_imp, k=10, epoch_num=None, figsize=(10, 5), out_path=None
):
    # Ranking
    items = sorted(global_imp.items(), key=lambda kv: kv[1], reverse=True)[:k]
    names = [k for k, _ in items]

    df = pd.DataFrame({
        "feature": names,
        "Global":   [global_imp.get(n, 0.0) for n in names],
        "Pos preds":[pos_imp.get(n,   0.0) for n in names],
        "Neg preds":[neg_imp.get(n,   0.0) for n in names],
    })

    df_melt = df.melt(id_vars="feature", var_name="Class", value_name="Importance")

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_melt,
        y="feature", x="Importance", hue="Class",
        palette=(
            sns.color_palette("bright")[4],  # Global
            sns.color_palette("bright")[0],  # Pos
            sns.color_palette("bright")[3],  # Neg
        ),
        edgecolor="black",
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="{:.2f}", padding=4, fontsize=7)

    plt.ylabel("Mean |SHAP|")
    plt.xlabel("")
    title = f"Feature Importance (top {k}) – Global vs Pos vs Neg"
    if epoch_num is not None:
        title += f" – {epoch_num}"
    plt.title(title)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_shap_scatter_top4_numeric(
    shap_vals,
    preds,
    encoded_feature_names,
    global_importance_raw,
    raw_batch_df,
    explained_indices,
    epoch_num=None,
    positive_threshold=0.5,
    out_path=None,
):
    top_feats = select_top_numeric_features(global_importance_raw, raw_batch_df, k=4)

    pred_scores = preds[:, 0]
    pos_mask = pred_scores >= positive_threshold
    neg_mask = ~pos_mask

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    title = "SHAP values vs. top numeric features (pos vs. neg predictions)"
    if epoch_num is not None:
        title += f" – {epoch_num}"
    fig.suptitle(title, fontsize=14)

    colors = sns.color_palette("bright", 5)
    pos_color, neg_color = colors[0], colors[3]

    for i, (ax, feat_name) in enumerate(zip(axes, top_feats)):
        enc_idx = encoded_indices_for_raw(encoded_feature_names, feat_name)
        if enc_idx.size == 0:
            ax.set_title(f"{feat_name} (no encoded dims)")
            ax.axis("off")
            continue

        shap_feat = shap_vals[:, enc_idx].sum(axis=1)

        raw_vals_batch = raw_batch_df[feat_name].values
        raw_vals = raw_vals_batch[explained_indices]

        x_pos = raw_vals[pos_mask]
        y_pos = shap_feat[pos_mask]
        x_neg = raw_vals[neg_mask]
        y_neg = shap_feat[neg_mask]

        ax.scatter(
            x_pos, y_pos,
            s=18, marker="o", alpha=0.7,
            label="Positive", color=pos_color, edgecolors="none",
        )
        ax.scatter(
            x_neg, y_neg,
            s=18, marker="o", alpha=0.8,
            label="Negative", color=neg_color, edgecolors="none",
        )

        ax.axhline(0.0, color="grey", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_title(feat_name)

        if i in (0, 2):
            ax.set_ylabel("SHAP value")

    for i in range(len(top_feats), 4):
        axes[i].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------------------
# SHAP (cell 26)
# --------------------------------------------------------------------------------------

def aggregate_raw_importance(encoded_feature_names, mean_abs_vec):
    """Somma le dimensioni encoded sulla feature RAW da cui vengono.

    La mappatura sta in `xai_metrics.raw_name_for`, un posto solo. Qui c'era
    `name.startswith("age_approx_bin_")` scritto a mano: su ISIC24 e' equivalente (l'unica
    feature discretizzata e' `age_approx`), ma il notebook costruisce i nomi con
    `f"{name}_bin_{b}"` per OGNI `Discretization`. Su un dataset con una seconda feature
    discretizzata questa funzione avrebbe mappato `foo_bin_0` sul raw `"foo_bin_0"` invece
    che su `"foo"` — in silenzio, e in disaccordo con `encoded_indices_for_raw` qui sopra,
    che il `_bin_` lo gestisce generico. Trovato il 2026-08-01.
    """
    from .xai_metrics import raw_name_for
    agg = {}
    for j, val in enumerate(mean_abs_vec):
        raw = raw_name_for(encoded_feature_names[j])
        agg.setdefault(raw, 0.0)
        agg[raw] += float(abs(val))
    return agg


class ShapEngine:
    """
    SHAP engine for the tabular 'features' input of the multimodal model.

    - Original model signature: model({"images": images, "features": features}) -> (batch, 1)
    - Internal wrapper: wrapped_model([features, images]) -> (batch, 1)
    - Explainer: shap.GradientExplainer on wrapped_model
    - We keep SHAP values only for the 'features' input (index 0 in [features, images]).
    """

    def __init__(
        self,
        background_size: int = 128,
        max_explain_samples: int = 64,
        top_k: int = 10,
        positive_threshold: float = 0.5,
        use_gradient_explainer: bool = True,
    ):
        self.background_size = background_size
        self.max_explain_samples = max_explain_samples
        self.top_k = top_k
        self.positive_threshold = positive_threshold
        self.use_gradient_explainer = use_gradient_explainer

        self._explainer = None
        self._wrapped_model = None   # Keras model: [features, images] -> original_model(...)
        self._feature_input_index = 0

    # ----------------- internal utils ----------------- #
    def _build_wrapped_model(
        self,
        base_model: tf.keras.Model,
        feature_shape,
        image_shape,
    ):
        """
        Build a Keras model that takes [features, images] and calls the original
        dict-based model internally.
        """
        feat_in = tf.keras.Input(shape=feature_shape[1:], name="shap_features")
        img_in = tf.keras.Input(shape=image_shape[1:], name="shap_images")

        out = base_model({"features": feat_in, "images": img_in})
        self._wrapped_model = tf.keras.Model(
            inputs=[feat_in, img_in],
            outputs=out,
            name="wrapped_for_shap",
        )

    def _build_explainer(
        self,
        base_model: tf.keras.Model,
        bg_features: np.ndarray,
        bg_images: np.ndarray,
    ):
        # Build wrapped model lazily
        if self._wrapped_model is None:
            self._build_wrapped_model(
                base_model,
                feature_shape=bg_features.shape,
                image_shape=bg_images.shape,
            )

        bg_inputs = [bg_features, bg_images]  # [features, images]

        if self.use_gradient_explainer:
            self._explainer = shap.GradientExplainer(self._wrapped_model, bg_inputs)
        else:
            self._explainer = shap.explainers.Deep(self._wrapped_model, bg_inputs)

    def _select_background_and_expl_set(
        self,
        features: np.ndarray,
        images: np.ndarray,
    ):
        n = features.shape[0]
        bg_idx = np.random.choice(n, size=min(self.background_size, n), replace=False)

        ex_idx = np.arange(n)
        if n > self.max_explain_samples:
            ex_idx = np.random.choice(n, size=self.max_explain_samples, replace=False)

        bg_features = features[bg_idx]
        bg_images = images[bg_idx]
        ex_features = features[ex_idx]
        ex_images = images[ex_idx]

        return (bg_features, bg_images), (ex_features, ex_images), ex_idx

    # ----------------- public API ----------------- #
    def explain(
        self,
        validation_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        model: tf.keras.Model,
        class_index: int = 0,  # irrelevant for scalar output
    ) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
        """
        validation_data: (features, images, labels) (all numpy arrays)

        Returns:
            stats: dict with
                - mean_abs_all: (n_features,)
                - mean_abs_pos: (n_features,)
                - mean_abs_neg: (n_features,)
                - topk_all_indices, topk_pos_indices, topk_neg_indices
                - explained_indices: indices into the original batch
            shap_values_features: (n_explain, n_features)
            preds: (n_explain, 1)
        """

        features, images, labels = validation_data

        (bg_features, bg_images), (ex_features, ex_images), ex_idx = \
            self._select_background_and_expl_set(features, images)

        # Lazy explainer init on wrapped model
        if self._explainer is None:
            self._build_explainer(model, bg_features, bg_images)

        ex_inputs = [ex_features, ex_images]  # [features, images]
        preds = self._wrapped_model(ex_inputs).numpy()  # (n_ex, 1)

        # SHAP values: list [phi_features, phi_images]
        shap_values_all_inputs = self._explainer.shap_values(ex_inputs)
        shap_values_features = shap_values_all_inputs[self._feature_input_index]
        # (n_explain, n_features)

        shap_values_features = np.array(shap_values_features)

        # Handle shapes:
        # - expected: (n_ex, n_features)
        # - you currently get: (n_ex, n_features, 1)
        if shap_values_features.ndim == 3:
          # If last dim is 1 → (n_ex, n_features, 1) → squeeze last
          if shap_values_features.shape[-1] == 1:
              shap_values_features = shap_values_features[..., 0]
          # If second dim is 1 → (n_ex, 1, n_features) → squeeze middle
          elif shap_values_features.shape[1] == 1:
              shap_values_features = shap_values_features[:, 0, :]
        # Now: (n_ex, n_features)

        # Global mean |SHAP|
        mean_abs_all = np.mean(np.abs(shap_values_features), axis=0)  # (n_features,)

        # Positive / negative groups
        pred_scores = preds[:, 0]
        pos_mask = pred_scores >= self.positive_threshold
        neg_mask = ~pos_mask

        if pos_mask.any():
            mean_abs_pos = np.mean(np.abs(shap_values_features[pos_mask]), axis=0)
        else:
            mean_abs_pos = np.zeros_like(mean_abs_all)

        if neg_mask.any():
            mean_abs_neg = np.mean(np.abs(shap_values_features[neg_mask]), axis=0)
        else:
            mean_abs_neg = np.zeros_like(mean_abs_all)

        def topk_indices(x: np.ndarray, k: int):
            # robust 1D top-k
            x = x.reshape(-1)
            k = min(k, x.shape[0])
            return np.argsort(x)[::-1][:k]

        topk_all_idx = topk_indices(mean_abs_all, self.top_k)
        topk_pos_idx = topk_indices(mean_abs_pos, self.top_k)
        topk_neg_idx = topk_indices(mean_abs_neg, self.top_k)

        stats = {
            "mean_abs_all": mean_abs_all,
            "mean_abs_pos": mean_abs_pos,
            "mean_abs_neg": mean_abs_neg,
            "topk_all_indices": topk_all_idx,
            "topk_pos_indices": topk_pos_idx,
            "topk_neg_indices": topk_neg_idx,
            "explained_indices": ex_idx,
        }

        return stats, shap_values_features, preds


class ShapCallback(Callback):
    """
    Runs ShapEngine every N epochs on a frozen validation subset and produces:
      - grouped barplot (Global vs Pos vs Neg) for top-k raw features
      - 2x2 scatter for top-4 numeric raw features (pos vs neg predictions)

    Explanation set is built at on_train_begin by concatenating multiple
    batches from `data_source`, up to `max_samples` samples (and optionally
    up to `max_batches` batches).
    """

    def __init__(
        self,
        data_source: tf.data.Dataset,
        validation_df,
        feat_cols: List[str],
        encoded_feature_names: List[str],
        base_dir: str,
        background_size: int = 256,
        max_explain_samples: int = 256,
        top_k: int = 15,
        positive_threshold: float = 0.5,
        every_n_epochs: int = 1,
        max_samples: int = 512,   # total samples in explanation set
        max_batches: int | None = None,  # limit num batches to read, or None
        rng_seed: int = 0,
        subset_idx=None,           # indici POSIZIONALI in validation_df; None = prefisso
        subset_fingerprint=None,   # stampigliato nell'npz: senza, i file non sono
                                   # riutilizzabili sotto un campione diverso (guard in
                                   # xai_faithfulness._load_shap_values)
    ):
        super().__init__()
        self.subset_idx = None if subset_idx is None else np.asarray(subset_idx, int)
        self.subset_fingerprint = subset_fingerprint
        self.data_source = data_source
        self.validation_df = validation_df
        self.feat_cols = feat_cols
        self.encoded_feature_names = encoded_feature_names

        self.background_size = int(background_size)
        self.max_explain_samples = int(max_explain_samples)
        self.top_k = int(top_k)
        self.positive_threshold = float(positive_threshold)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.max_samples = int(max_samples)
        self.max_batches = max_batches
        self.rng_seed = int(rng_seed)

        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # frozen explanation set
        self._images = None
        self._features = None
        self._labels = None
        self._raw_batch_df = None

        # ShapEngine instance
        self._engine = ShapEngine(
            background_size=self.background_size,
            max_explain_samples=self.max_explain_samples,
            top_k=self.top_k,
            positive_threshold=self.positive_threshold,
        )

    # ---------- helpers ---------- #
    def _freeze_batch_from_dataset(self, ds: tf.data.Dataset):
        """
        Build explanation set by concatenating multiple batches from ds.

        Assumes ds yields (inputs, labels) where inputs is {'images','features'}.
        Also builds the corresponding raw_batch_df from validation_df (shuffle=False).
        """
        X_img_list, X_feat_list, y_list = [], [], []
        n_collected = 0

        for b_idx, (x, y) in enumerate(ds):
            if self.max_batches is not None and b_idx >= self.max_batches:
                break

            if not isinstance(x, dict):
                raise ValueError("Expected dataset elements like ({'images','features'}, y).")

            images_b = x["images"].numpy()
            features_b = x["features"].numpy()
            labels_b = y.numpy()

            remaining = self.max_samples - n_collected
            if remaining <= 0:
                break

            if images_b.shape[0] > remaining:
                images_b = images_b[:remaining]
                features_b = features_b[:remaining]
                labels_b = labels_b[:remaining]

            X_img_list.append(images_b)
            X_feat_list.append(features_b)
            y_list.append(labels_b)

            n_collected += images_b.shape[0]
            if n_collected >= self.max_samples:
                break

        if n_collected == 0:
            raise RuntimeError("No samples collected for SHAP explanation set.")

        self._images = np.concatenate(X_img_list, axis=0)
        self._features = np.concatenate(X_feat_list, axis=0)
        self._labels = np.concatenate(y_list, axis=0)

        # assume validation_ds built from validation_df with shuffle=False
        self._raw_batch_df = (
            self.validation_df.iloc[:n_collected][self.feat_cols].reset_index(drop=True)
        )

        if self.subset_idx is not None:
            if n_collected <= int(self.subset_idx.max()):
                raise RuntimeError(
                    f"il subset chiede la riga {int(self.subset_idx.max())} ma ne sono state "
                    f"lette {n_collected}: alza max_samples/max_batches di ShapCallback.")
            self._images = self._images[self.subset_idx]
            self._features = self._features[self.subset_idx]
            self._labels = self._labels[self.subset_idx]
            self._raw_batch_df = (self._raw_batch_df.iloc[self.subset_idx]
                                  .reset_index(drop=True))

    # ---------- Keras hooks ---------- #
    def on_train_begin(self, logs=None):
        self._freeze_batch_from_dataset(self.data_source)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.every_n_epochs) != 0:
            return
        if self._images is None or self._features is None:
            return

        np.random.seed(self.rng_seed)

        # SHAP on frozen set with current weights
        stats, shap_vals, preds = self._engine.explain(
            validation_data=(self._features, self._images, self._labels),
            model=self.model,
            class_index=0,
        )

        mean_abs_all = stats["mean_abs_all"]
        mean_abs_pos = stats["mean_abs_pos"]
        mean_abs_neg = stats["mean_abs_neg"]

        global_importance_raw = aggregate_raw_importance(
            self.encoded_feature_names, mean_abs_all
        )
        pos_importance_raw = aggregate_raw_importance(
            self.encoded_feature_names, mean_abs_pos
        )
        neg_importance_raw = aggregate_raw_importance(
            self.encoded_feature_names, mean_abs_neg
        )

        epoch_name = f"epoch_{epoch:03d}"
        epoch_dir = self.base_dir / epoch_name
        epoch_dir.mkdir(parents=True, exist_ok=True)

        # Persist arrays + tabular share for post-hoc analysis
        np.savez_compressed(
            epoch_dir / f"shap_arrays_{epoch_name}.npz",
            shap_values=shap_vals,                          # (n_ex, n_features_encoded)
            preds=preds,
            explained_indices=stats["explained_indices"],
            mean_abs_all=mean_abs_all,
            mean_abs_pos=mean_abs_pos,
            mean_abs_neg=mean_abs_neg,
            # L'impronta del campione: `explained_indices` sono POSIZIONI, quindi senza
            # questa un npz non e' distinguibile da uno prodotto su altre righe.
            **({} if self.subset_fingerprint is None
               else {"subset_fingerprint": str(self.subset_fingerprint)}),
            # La POSIZIONE IN `validation_df` di ogni campione spiegato — la stessa chiave,
            # con lo stesso significato, che scrivono Grad-CAM (`gradcam_arrays_*.npz`) e
            # LORE (colonna `val_index` nel per-sample CSV).
            #
            # Serve perche' `explained_indices` NON e' quella cosa: sono posizioni DENTRO il
            # campione (0..n-1), quindi due motori sullo stesso campione producono indici
            # che si somigliano e non sono confrontabili. E' esattamente il modo in cui
            # l'accordo SHAP<->LORE usciva NaN. Con `val_index` i tre motori si riallineano
            # per costruzione, senza dover sapere come e' stato costruito il campione.
            **({} if self.subset_idx is None
               else {"val_index": np.asarray(self.subset_idx, int)[
                   np.asarray(stats["explained_indices"], int)]}),
        )

        # Tabular share = sum |SHAP| over tabular dims / total. Here all encoded features
        # ARE tabular (image branch is separate input), so share = mean_abs_all.sum() over all dims.
        # More useful: per-raw-feature share aggregated into "categorical vs numeric vs interaction" buckets,
        # but simplest paper-ready signal is the raw importance vector itself (already saved).

        total_abs = float(mean_abs_all.sum())
        top1_share = float(mean_abs_all.max() / (total_abs + 1e-12))
        top5_share = float(np.sort(mean_abs_all)[::-1][:5].sum() / (total_abs + 1e-12))

        # Metriche di FORMA (leggibilita'): compattezza per entropia e k90 sparsity, dalle
        # stesse attribuzioni che sono gia' in memoria. Escono qui, con tutto il resto,
        # invece che da una cella da lanciare dopo. Sono un COMPLEMENTO: `shap_mass` accanto
        # dice se c'e' qualcosa da spiegare — con il ramo tabellare spento la compattezza
        # sarebbe 1.000 (il massimo) e `xai_metrics` la mette a NaN apposta.
        from .xai_metrics import shap_shape_from_values
        shape = shap_shape_from_values(shap_vals, stats["explained_indices"])

        shap_csv = self.base_dir / "shap_summary.csv"
        append_epoch_row(shap_csv, {
            "epoch": int(epoch), "total_abs_shap": total_abs,
            "top1_share": top1_share, "top5_share": top5_share,
            "n_explained": int(shap_vals.shape[0]),
            "n_pos": int((preds[:, 0] >= self.positive_threshold).sum()),
            "n_neg": int((preds[:, 0] <  self.positive_threshold).sum()),
            **{k: v for k, v in shape.items()
               if k in ("shap_compactness", "shap_k90", "shap_k90_sparsity",
                        "shap_mass", "frac_zero_shap", "branch_off")},
        })

        # 1) grouped barplot: global vs pos vs neg (top-k)
        plot_topk_grouped_raw_global_pos_neg(
            global_imp=global_importance_raw,
            pos_imp=pos_importance_raw,
            neg_imp=neg_importance_raw,
            epoch_num = epoch_name,
            k=self.top_k,
            figsize=(10, 5),
            out_path=str(
                epoch_dir
                / f"shap_grouped_raw_top{self.top_k:02d}_glob_pos_neg_{epoch_name}.png"
            ),
        )

        # 2) scatter grid: top-4 numeric features
        plot_shap_scatter_top4_numeric(
            shap_vals=shap_vals,
            preds=preds,
            encoded_feature_names=self.encoded_feature_names,
            global_importance_raw=global_importance_raw,
            raw_batch_df=self._raw_batch_df,
            explained_indices=stats["explained_indices"],
            positive_threshold=self.positive_threshold,
            epoch_num = epoch_name,
            out_path=str(
                epoch_dir / f"shap_top4_numeric_grid_{epoch_name}.png"
            ),
        )


# --------------------------------------------------------------------------------------
# Grad-CAM (cell 28)
# --------------------------------------------------------------------------------------

class MultiModalGradCAMEngine:
    """
    Grad-CAM engine for multimodal models (images + tabular features).

    Features:
    - Robust target selection: defaults to the 4D tensor feeding the first GlobalAveragePooling2D.
    - Two modes:
        * 'context'     → uses real tabular features.
        * 'image_only'  → replaces features with a baseline (zeros or provided vector).
    - Output:
        * grid: single grid image (HxWx3, float [0..1]) of horizontally stacked triptychs
        * triptychs: list of per-sample images (orig+pred | heatmap | overlay), each Hx(3W)x3 in [0..1]
        * preds: model predictions for the batch (numpy array)
    - Visualization:
        * Accepts a separate display_images batch or a deprocess_fn to convert model-input images to [0,1].
    """

    def __init__(
        self,
        use_guided_grads: bool = True,
        mode: str = 'context',              # 'context' or 'image_only'
        baseline: Optional[np.ndarray] = None,
        deprocess_fn: Optional[callable] = None,  # fn(img_float) -> float [0,1]
    ):
        assert mode in ('context', 'image_only'), "mode must be 'context' or 'image_only'"
        self.use_guided_grads = use_guided_grads
        self.mode = mode
        self.baseline = baseline
        self.deprocess_fn = deprocess_fn

    def explain(
        self,
        validation_data: Tuple[dict, np.ndarray],                            # ({'images', 'features'}, labels)
        model: tf.keras.Model,
        class_index: int,
        layer_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Compute Grad-CAMs for a batch and return (grid, triptychs, preds).

        validation_data is # ({'images', 'features'}, labels)

        """
        # --- Unpack ---
        x, labels = validation_data
        images = x['images']
        features = x['features']

        # --- Build grad model ---
        conv_target_tensor = self._resolve_target_tensor(model, layer_name)
        grad_model = Model(inputs=model.inputs, outputs=[conv_target_tensor, model.output])

        # --- Build feeds in model.inputs order ---
        # feeds = [features, images]
        # if self.mode == 'image_only':Fif se
        #     feeds = self._apply_baseline_to_feeds(model, feeds, images, features)

        feeds = self._ordered_feeds(model, images=images, features=features)
        if self.mode == 'image_only':
            feeds = self._apply_baseline_to_feeds(model, feeds, images, features)

        # --- Forward + gradients ---
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(feeds, training=False)

            # target score
            if preds.shape.rank == 2 and preds.shape[-1] == 1:
                # binary sigmoid -> use logit(p) to avoid saturation artifacts
                p = tf.clip_by_value(preds[:, 0], 1e-6, 1.0 - 1e-6)
                target = tf.math.log(p) - tf.math.log(1.0 - p)  # (B,)
            else:
                # multi-class / multi-label (keep prob target; optionally log it)
                target = preds[:, class_index]  # (B,)

            loss = tf.reduce_sum(target)  # scalar

        grads = tape.gradient(loss, conv_out)  # (B, Hc, Wc, C)

        if self.use_guided_grads:
            grads = (
                tf.cast(conv_out > 0, grads.dtype)
                * tf.cast(grads > 0, grads.dtype)
                * grads
            )

        # --- CAMs (B, Hc, Wc) ---
        cams = self._generate_cams(conv_out, grads)  # list of (Hc, Wc)
        cams = np.stack(cams, axis=0)
        # Le heatmap GREZZE, esposte perche' il callback possa SALVARLE. Fino al 01/08 questa
        # funzione restituiva solo il collage RGB: le uniche tracce del lato immagine erano
        # 106 PNG con la colormap gia' applicata, e da li' non si ricava nessuna metrica
        # (entropia, area attivata, componenti connesse, IoU fra configurazioni). Non e' nel
        # valore di ritorno per non rompere i chiamanti che spacchettano tre elementi.
        self.last_cams = cams
        self.last_preds = preds.numpy() if isinstance(preds, tf.Tensor) else preds

        # --- Triptychs with display images ---
        triptychs = []
        preds_np = self.last_preds
        # Il collage cresce come il batch: a n=144 sarebbe un PNG enorme che nessuno guarda,
        # e comporlo costa piu' del Grad-CAM stesso. Oltre `max_grid` si salvano solo gli
        # array, che sono il pezzo che serve.
        n_vis = min(images.shape[0], int(getattr(self, "max_grid", 8)))
        for i in range(n_vis):
            pred_text = self._format_pred_text(preds_np[i])
            vis_img = images[i]
            trip = self._compose_triptych(vis_img, cams[i], pred_text)
            triptychs.append(trip)

        grid = self._make_grid(triptychs) if triptychs else np.zeros((1, 1, 3))
        return grid, triptychs, preds_np

    # ------------------------ Saving helpers ------------------------
    def save_grid(self, grid: np.ndarray, output_dir: str, output_name: str) -> None:
        """Save the grid image to disk (RGB)."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_path = str(Path(output_dir) / output_name)
        img_u8 = np.uint8(255 * np.clip(grid, 0, 1))
        cv2.imwrite(out_path, cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))

    def save_triptychs(
        self,
        triptychs: List[np.ndarray],
        base_dir: str,
        epoch_name: str,
        prefix: str = "sample",
    ) -> None:
        """
        Save each triptych as:
          base_dir/epoch_name/{prefix}_{i:03d}.png
        """
        out_dir = Path(base_dir) / epoch_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(triptychs):
            img_u8 = np.uint8(255 * np.clip(img, 0, 1))
            path = out_dir / f"{prefix}_{i:03d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))

    # ------------------------ Internal helpers ------------------------
    def _make_grid(self, triptychs: List[np.ndarray]) -> np.ndarray:
        """Create a grid image; uses tf-explain grid_display if available, else a simple grid."""
        try:
            from tf_explain.utils.display import grid_display
            thumbs = np.stack([np.uint8(255 * np.clip(t, 0, 1)) for t in triptychs], axis=0)
            canvas = grid_display(thumbs)  # returns uint8 RGB
            return canvas.astype(np.float32) / 255.0
        except Exception:
            return self._naive_grid(triptychs)


    def _resolve_target_tensor(self, model: tf.keras.Model, layer_name: Optional[str]):
        """
        Prefer the 4D tensor feeding the first GlobalAveragePooling2D.
        If layer_name is provided, use that layer's output; else fallback to last 4D layer.
        """
        if layer_name is not None:
            return model.get_layer(layer_name).output
        # Prefer the tensor feeding the first GAP
        for l in model.layers:
            if isinstance(l, tf.keras.layers.GlobalAveragePooling2D):
                return l.input
        # Fallback to last 4D tensor in the graph
        for l in reversed(model.layers):
            try:
                shp = l.output_shape
            except Exception:
                continue
            if isinstance(shp, tuple) and len(shp) == 4:
                return l.output
        raise ValueError("No suitable 4D conv tensor found for Grad-CAM target")

    def _ordered_feeds(self, model: tf.keras.Model, images, features):
        """Create a list of inputs respecting model.inputs order (rank-4 → images, rank-2 → features)."""
        feeds = []
        for kt in model.inputs:
            if len(kt.shape) == 4:
                feeds.append(images)
            elif len(kt.shape) == 2:
                feeds.append(features)
            else:
                raise ValueError(f"Unexpected input rank for {kt.name}: {kt.shape}")
        return feeds

    def _apply_baseline_to_feeds(self, model: tf.keras.Model, feeds, images, features):
        """Replace the features tensor in feeds with a baseline (zeros by default)."""
        feat_inp = next(inp for inp in model.inputs if len(inp.shape) == 2)
        F = int(feat_inp.shape[-1])
        if self.baseline is None:
            baseline = tf.zeros((tf.shape(images)[0], F), dtype=getattr(features, 'dtype', tf.float32))
        else:
            base = tf.convert_to_tensor(self.baseline)
            if base.shape.rank == 1:
                baseline = tf.broadcast_to(base[None, :], [tf.shape(images)[0], F])
            else:
                baseline = base
        feeds_out = []
        for kt, fed in zip(model.inputs, feeds):
            feeds_out.append(baseline if len(kt.shape) == 2 else fed)
        return feeds_out

    def _generate_cams(self, conv_outputs: tf.Tensor, grads: tf.Tensor) -> List[np.ndarray]:
        """Compute per-sample weighted sum of conv activations; returns list of 2D CAMs normalized to [0,1]."""
        cams = []
        conv_np = conv_outputs.numpy() if isinstance(conv_outputs, tf.Tensor) else conv_outputs
        grads_np = grads.numpy() if isinstance(grads, tf.Tensor) else grads
        B = conv_np.shape[0]
        # Trimming the border removes the outer ring of an already tiny grid — at 256 px the CAM
        # is 8x8, so 28 of 64 cells go — and the faithfulness path scores the untrimmed map.
        sl = slice(1, -1) if TRIM_CAM_BORDER else slice(None)
        for i in range(B):
            gr = grads_np[i][sl, sl, :]
            co = conv_np[i][sl, sl, :]
            weights = gr.mean(axis=(0, 1))  # (C,)
            cam = (co * weights[None, None, :]).sum(axis=-1)
            cam = np.maximum(cam, 0.0)
            cam_min, cam_max = cam.min(), cam.max()
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            cams.append(cam)
        return cams

    def _format_pred_text(self, pred_row):
        """Format prediction text for display: binary → p=..., multi-class → ŷ=k (p=...)."""
        try:
            arr = np.array(pred_row)
        except Exception:
            return f"p={float(pred_row):.3f}"
        if arr.ndim == 0:
            return f"p={float(arr):.3f}"
        arr = arr.ravel()
        if arr.size == 1:
            return f"p={float(arr[0]):.3f}"
        else:
            k = int(np.argmax(arr))
            p = float(arr[k])
            return f"ŷ={k} (p={p:.3f})"

    def _compose_triptych(self, image_like: np.ndarray, cam_2d: np.ndarray, pred_text: str, colormap=cv2.COLORMAP_JET, image_weight=0.8) -> np.ndarray:
        """Create a 3-panel image (orig+text | heatmap | overlay), each panel HxWx3 in [0,1]."""
        img = self._to_display_rgb(image_like)
        H, W = img.shape[:2]
        # Heatmap
        heat = cv2.applyColorMap(np.uint8(255 * cv2.resize(cam_2d, (W, H))),colormap)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB) / 255.0
        # Overlay
        overlay = image_weight * img + (1.0 - image_weight) * heat
        overlay = np.clip(overlay, 0, 1)
        # Original with prediction text
        orig = (img.copy() * 255).astype(np.uint8)
        txt = pred_text if isinstance(pred_text, str) else str(pred_text)
        cv2.putText(orig, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(orig, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        orig = orig.astype(np.float32) / 255.0
        return np.hstack([orig, heat, overlay])

    # ---------------- Display helpers ----------------
    def _to_display_rgb(self, image: np.ndarray) -> np.ndarray:
        """
        Convert a model-input image to float RGB in [0,1] for visualization.
        Applies custom deprocess_fn if provided; else uses heuristics.
        """
        arr = image.numpy() if tf.is_tensor(image) else np.asarray(image)

        if self.deprocess_fn is not None:
            out = self.deprocess_fn(arr)
            return np.clip(self._ensure_rgb(out), 0.0, 1.0)

        # 1) uint8 → [0,1]
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
            return np.clip(self._ensure_rgb(arr), 0, 1)

        # 2) Float ranges
        arr = arr.astype(np.float32)
        amin, amax = float(arr.min()), float(arr.max())

        # true [-1,1] only if there are negatives
        if amin < 0.0 and amax <= 1.1:
            arr = (arr + 1.0) / 2.0
            return np.clip(self._ensure_rgb(arr), 0, 1)

        # already [0,1]
        if 0.0 <= amin and amax <= 1.1:
            return np.clip(self._ensure_rgb(arr), 0, 1)

        # [0,255] floats
        if amin >= -1.0 and amax > 1.5 and amax <= 260:
            arr = arr / 255.0
            return np.clip(self._ensure_rgb(arr), 0, 1)

        # fallback: per-image min-max
        rng = (amax - amin) if (amax - amin) > 1e-8 else 1.0
        arr = (arr - amin) / rng
        return np.clip(self._ensure_rgb(arr), 0, 1)

    def _ensure_rgb(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr

    def _naive_grid(self, triptychs: List[np.ndarray], cols: int = 2) -> np.ndarray:
        if not triptychs:
            return np.zeros((32, 32, 3), dtype=np.float32)
        H, W, C = triptychs[0].shape
        rows = int(np.ceil(len(triptychs) / cols))
        grid = np.ones((rows * H, cols * W, C), dtype=np.float32)
        for idx, img in enumerate(triptychs):
            r = idx // cols
            c = idx % cols
            grid[r*H:(r+1)*H, c*W:(c+1)*W] = np.clip(img, 0, 1)
        return grid


class MultiModalGradCAMCallback(Callback):
    def __init__(self, engine, probe_batch, class_index=0, layer_name=None,
                 save_dir=None, tag="GradCAM/grid", subset_idx=None,
                 subset_fingerprint=None, modes=None, batch_size=None):
        super().__init__()
        self.engine = engine
        self.probe_batch = probe_batch
        self.class_index = class_index
        self.layer_name = layer_name
        self.tag = tag
        # DUE MODI, uno stesso caricamento. `image_only` azzera le feature tabellari e
        # chiede "dove guarderebbe SENZA il tabellare"; `context` usa le feature vere e
        # chiede "dove guarda davvero". Per uno studio sull'attendance servono entrambi, e
        # la loro DIFFERENZA (IoU fra le due maschere) e' essa stessa un segnale: quanto il
        # ramo tabellare sposta l'attenzione visiva.
        #
        # Costano poco insieme perche' il caro e' altrove: caricare il checkpoint e leggere
        # il dataset fino alle 144 lesioni si paga UNA volta per run, il passo di gradiente
        # e' l'unica cosa che raddoppia.
        #
        # `modes=None` = comportamento storico: un modo solo, quello del motore.
        self.modes = tuple(modes) if modes else (engine.mode,)
        # Le POSIZIONI in validation_df delle immagini nel probe batch, e l'impronta del
        # campione. Servono a rendere le heatmap riallineabili con SHAP e LORE: fino al
        # 01/08 Grad-CAM girava sulle prime 8 righe e non lasciava traccia di quali fossero.
        self.subset_idx = subset_idx
        self.subset_fingerprint = subset_fingerprint
        # Quante immagini per passo di gradiente. None = tutte insieme (comportamento
        # storico, che a n=8 andava bene e a n=144 esaurisce la GPU).
        self.batch_size = batch_size
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        from .xai_metrics import gradcam_shape_from_cams, cam_masks, cam_mode_contrast
        single = len(self.modes) == 1
        by_mode, preds_by_mode = {}, {}

        for mode in self.modes:
            self.engine.mode = mode
            # A CHUNK, non tutto insieme. Con 144 immagini in un solo `GradientTape` e i
            # guided gradients servono ~22 GiB solo di attivazioni (misurato 01/08:
            # ResourceExhaustedError su un tensore (144, 8, 8, 1248) con 23.9 GiB di GPU).
            # Grad-CAM e' per campione: non c'e' nessuna dipendenza fra immagini, quindi
            # spezzare il batch da' lo STESSO risultato con memoria costante.
            xb, yb = self.probe_batch
            n_tot = int(len(next(iter(xb.values()))))
            step = max(1, int(self.batch_size or n_tot))
            cams_parts, preds_parts, grid = [], [], None
            for start in range(0, n_tot, step):
                sl = slice(start, min(start + step, n_tot))
                g, _, p = self.engine.explain(
                    validation_data=({k: v[sl] for k, v in xb.items()}, yb[sl]),
                    model=self.model,
                    class_index=self.class_index,
                    layer_name=self.layer_name,
                )
                cams_parts.append(np.asarray(self.engine.last_cams))
                preds_parts.append(np.asarray(p))
                if grid is None:
                    grid = g          # il collage viene dal primo chunk: e' un'anteprima
            self.engine.last_cams = np.concatenate(cams_parts, axis=0)
            preds = np.concatenate(preds_parts, axis=0)
            if not self.save_dir:
                continue
            # Nome storico con un modo solo (`epoch_003.png`), suffissato con due: le run
            # canoniche restano leggibili dagli script che le cercano per quel nome.
            sfx = "" if single else f"_{mode}"
            out = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(self.save_dir / f"epoch_{epoch:03d}{sfx}.png"),
                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

            cams = getattr(self.engine, "last_cams", None)
            if cams is None:
                continue
            by_mode[mode] = np.asarray(cams, np.float32)
            preds_by_mode[mode] = np.asarray(preds, np.float32).reshape(len(cams), -1)

            # LE HEATMAP GREZZE: precondizione di ogni metrica sul lato immagine. Da un PNG
            # con la colormap applicata non si ricava ne' entropia ne' area ne' IoU.
            # `val_index` le accompagna per lo stesso motivo per cui accompagna le regole
            # LORE: senza, non si sa QUALI lesioni sono state spiegate.
            np.savez_compressed(
                self.save_dir / f"gradcam_arrays{sfx}_epoch_{epoch:03d}.npz",
                cams=by_mode[mode], preds=np.asarray(preds, np.float32), mode=mode,
                **({} if self.subset_idx is None
                   else {"val_index": np.asarray(self.subset_idx, int)[:len(cams)]}),
                **({} if self.subset_fingerprint is None
                   else {"subset_fingerprint": str(self.subset_fingerprint)}))

            # Le metriche vanno nella TABELLA e basta. La figura per-run che c'era qui
            # (heatmap media, frazione sopra soglia, bar chart delle metriche) e' stata
            # tolta il 01/08: le prime due non dicono niente e la terza duplica in
            # pixel una riga di CSV. Su 110 run x 2 modi erano 220 PNG di ingombro.
            shape = gradcam_shape_from_cams(by_mode[mode])
            append_epoch_row(self.save_dir / "gradcam_summary.csv",
                             {"epoch_mode": f"{epoch:03d}:{mode}", "epoch": int(epoch),
                              "mode": mode, "n_cams": int(len(cams)), **shape},
                             key="epoch_mode")   # <- due righe per epoca: la chiave e' la coppia

        # Il segnale che nasce dall'avere ENTRAMBI: quanto il ramo tabellare sposta
        # l'attenzione visiva. IoU 1.0 = il tabellare non cambia dove il modello guarda.
        #
        # ⚠️ LA IoU DA SOLA NON BASTA, ed e' il motivo per cui questa riga usciva quasi
        # vuota. Su cinque famiglie su sei la testa e' `Dense(1)` su una `Concatenate`,
        # quindi il logit e' ADDITIVO nei due rami e d(logit)/d(conv_out) non dipende
        # affatto dalle feature: la heatmap e' identica per costruzione, non per risultato,
        # e la IoU vale 1.000 ovunque. Intanto la PREDIZIONE cambia fino a 0.53 — cioe' la
        # riga di confronto diceva «non e' successo niente» proprio mentre l'unica cosa che
        # si muoveva non era registrata da nessuna parte. Ora la riga porta anche la
        # differenza sulle heatmap grezze e quella sulle predizioni.
        if self.save_dir and {"image_only", "context"} <= set(by_mode):
            contrast = cam_mode_contrast(by_mode["context"], by_mode["image_only"],
                                         preds_by_mode.get("context"),
                                         preds_by_mode.get("image_only"))
            append_epoch_row(self.save_dir / "gradcam_summary.csv",
                             {"epoch_mode": f"{epoch:03d}:context_vs_image_only",
                              "epoch": int(epoch), "mode": "context_vs_image_only",
                              "n_cams": int(len(by_mode["context"])), **contrast},
                             key="epoch_mode")


# --------------------------------------------------------------------------------------
# LORE (cell 33)
# --------------------------------------------------------------------------------------

def plot_lore_feature_frequency(
    freq_all: dict, freq_pos: dict, freq_neg: dict,
    k: int = 10, epoch_num=None, mode: str = '', figsize=(10, 5), out_path=None,
):
    """Barplot of rule-feature frequencies: Global / Pos / Neg predictions."""
    items = sorted(freq_all.items(), key=lambda kv: kv[1], reverse=True)[:k]
    if not items:
        return
    names = [n for n, _ in items]
    df = pd.DataFrame({
        'feature': names,
        'Global': [freq_all.get(n, 0) for n in names],
        'Pos preds': [freq_pos.get(n, 0) for n in names],
        'Neg preds': [freq_neg.get(n, 0) for n in names],
    })
    df_melt = df.melt(id_vars='feature', var_name='Class', value_name='Rule frequency')
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_melt, y='feature', x='Rule frequency', hue='Class',
        palette=(sns.color_palette('bright')[4], sns.color_palette('bright')[0], sns.color_palette('bright')[3]),
        edgecolor='black',
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=4, fontsize=7)
    title = f'LORE Feature Frequency in Rules (top {k})'
    if mode:
        title += f' [{mode}]'
    if epoch_num is not None:
        title += f' – {epoch_num}'
    plt.title(title)
    plt.xlabel('Rule frequency (# instances)')
    plt.ylabel('')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_lore_rules_text(rules: list, epoch_num=None, mode: str = '', out_path=None) -> str:
    """Write per-instance LORE rules to a .txt file."""
    header = 'LORE Rules'
    if mode:
        header += f' [{mode}]'
    if epoch_num is not None:
        header += f' – {epoch_num}'
    lines = [header, '=' * len(header), '']

    def _fmt(subset, label):
        lines.append(f'=== {label} ({len(subset)} instances) ===')
        for i, r in subset:
            lines.append(f'[inst {i:02d}] pred={r["pred"]:.3f} label={r["label"]}')
            lines.append(f'  RULE: {r["rule_str"]}')
            for cr in r.get('crules_str', []):
                lines.append(f'  COUNTERFACTUAL: {cr}')
            lines.append('')

    pos = [(i, r) for i, r in enumerate(rules) if r['pred'] >= 0.5]
    neg = [(i, r) for i, r in enumerate(rules) if r['pred'] < 0.5]
    _fmt(pos, 'POSITIVE PREDICTIONS')
    _fmt(neg, 'NEGATIVE PREDICTIONS')

    text = "\n".join(lines)
    if out_path:
        with open(out_path, 'w') as f:
            f.write(text)
    return text


class LoreEngine:
    """
    LORE on raw (pre-FeatureSpace) tabular features.
    Categoricals are label-encoded before LORE, decoded back to strings
    in the predictor before FeatureSpace.
    """

    def __init__(
        self,
        feature_space,
        feat_cols: List[str],
        max_explain_samples: int = 32,
        positive_threshold: float = 0.5,
        lore_config: Optional[Dict] = None,
        lore_random_state: int = 0,
        oos_samples: int = 1000,
    ):
        self.feature_space = feature_space
        self.feat_cols = feat_cols
        self.max_explain_samples = int(max_explain_samples)
        self.positive_threshold = float(positive_threshold)
        # NOTE: 'random_state' inside lore_config is SILENTLY IGNORED by the xailib wrapper --
        # LoreTabularExplainer.fit() passes random_state=self.random_state (a class attribute,
        # default 0) and never reads the config dict. It is set explicitly in _build_explainer.
        self.lore_random_state = int(lore_random_state)
        self.oos_samples = int(oos_samples)
        self.lore_config = lore_config or {
            'neigh_type': 'random',
            'size': 300,
            'verbose': False,
        }
        self._image_encoder = None
        self._head_model = None
        self._predict_fn = None
        # Label encoding maps — built per explain() call from actual data
        self._label_maps = {}   # col → {string: int}
        self._inv_maps = {}     # col → {int: string}
        self._cat_cols = []

    # ── Model splitting ──────────────────────────────────────────────────
    def _build_sub_models(self, model):
        if self._image_encoder is not None:
            return
        # Cut the graph right after the (expensive) image backbone, at the
        # GlobalAveragePooling2D output. This pooled image feature is the single
        # trunk through which *all* image-side computation flows, so the head can
        # rebuild any downstream fusion scheme (plain concat, gamma fusion,
        # learnable scaling, adaptive gating) from [pooled_image, features].
        # The backbone stays in image_encoder (run once, cached); the head holds
        # only the cheap dense/fusion layers, so predicting over the LORE
        # neighbourhood is fast and works for every architecture.
        gap_layers = [l for l in model.layers
                      if isinstance(l, keras.layers.GlobalAveragePooling2D)]
        if not gap_layers:
            raise RuntimeError(
                "LORE: no GlobalAveragePooling2D found in "
                f"'{model.name}' to split the image branch."
            )
        img_emb = gap_layers[0].output            # (None, C) pooled image feature
        feat_in = model.input["features"]
        self._image_encoder = keras.Model(
            inputs=model.input["images"], outputs=img_emb, name="image_encoder",
        )
        # Reuse the intermediate pooled tensor directly as a head input (Keras
        # functional API): architecture-agnostic, no hardcoded layer names.
        self._head_model = keras.Model(
            inputs=[img_emb, feat_in], outputs=model.output, name="head_model",
        )
        self._predict_fn = tf.function(
            lambda embs, enc: self._head_model([embs, enc], training=False),
            reduce_retracing=True,
        )

    # ── Label encoding ───────────────────────────────────────────────────
    def _label_encode_df(self, df):
        """Label-encode categorical columns to integers. Returns encoded copy."""
        # self._cat_cols = [c for c in df.columns if df[c].dtype == 'object']
        self._cat_cols = [c for c in df.columns if c != 'target'
                          and not pd.api.types.is_numeric_dtype(df[c])]
        self._label_maps, self._inv_maps = {}, {}
        out = df.copy()
        for col in self._cat_cols:
            uniq = sorted(df[col].dropna().unique())
            lm = {v: i for i, v in enumerate(uniq)}
            self._label_maps[col] = lm
            self._inv_maps[col] = {i: v for v, i in lm.items()}
            out[col] = df[col].map(lm).astype('float64')
        return out

    # ── Predictor ────────────────────────────────────────────────────────
    def _make_predictor(self, image_embedding, feat_cols_no_target):
        head = self._head_model
        fs = self.feature_space
        emb_const = tf.constant(image_embedding[np.newaxis], dtype=tf.float32)
        cat_cols = self._cat_cols
        inv_maps = self._inv_maps
        predict_fn = self._predict_fn

        def predict(X) -> np.ndarray:
            if isinstance(X, pd.DataFrame):
                df = X.copy()
                cols = [c for c in df.columns if c != 'target']
            else:
                arr = np.asarray(X, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                df = pd.DataFrame(arr, columns=feat_cols_no_target)
                cols = feat_cols_no_target

            # Decode label-encoded categoricals back to strings
            feat_dict = {}
            for col in cols:
                if col in cat_cols:
                    ints = df[col].values.round().astype(int).clip(0, len(inv_maps[col]) - 1)
                    strings = [inv_maps[col].get(i, inv_maps[col][0]) for i in ints]
                    feat_dict[col] = tf.constant(strings)
                else:
                    feat_dict[col] = tf.constant(df[col].values.astype(np.float32))

            encoded = fs(feat_dict)
            n = encoded.shape[0]
            embs = tf.repeat(emb_const, n, axis=0)
            return predict_fn(embs, encoded).numpy()[:, 0]

        return predict

    # ── LORE wrappers ────────────────────────────────────────────────────
    def _build_explainer(self, predictor, background_df):
        # Imported lazily: xailib/lore_explainer is the most fragile dependency in the
        # stack and is being migrated (see LoreEngineSA), so importing it at module level
        # would take SHAP and Grad-CAM down with it whenever LORE breaks.
        from xailib.explainers.lore_explainer import LoreTabularExplainer

        class _BB:
            def __init__(self, fn, thr):
                self.fn, self.thr = fn, thr
            def predict(self, X):
                return (self.fn(X) >= self.thr).astype(int)
            def predict_proba(self, X):
                p = self.fn(X)
                return np.column_stack([1 - p, p])

        bb = _BB(predictor, self.positive_threshold)
        df_bg = background_df.copy().reset_index(drop=True)
        df_bg['target'] = bb.predict(df_bg).astype(np.int64)
        present = set(df_bg['target'].dropna().astype(int).unique().tolist())
        missing = [cls for cls in (0, 1) if cls not in present]
        if missing and len(df_bg) > 0:
            base_row = df_bg.iloc[0].copy()
            filler_rows = []
            for cls in missing:
                row = base_row.copy()
                row['target'] = cls
                filler_rows.append(row)
            df_bg = pd.concat([df_bg, pd.DataFrame(filler_rows)], ignore_index=True)
        exp = LoreTabularExplainer(bb=bb)
        exp.random_state = self.lore_random_state   # FIX: the config key is ignored by the wrapper
        exp.fit(df_bg, 'target', self.lore_config)
        self._predictor = predictor                 # kept for the out-of-sample fidelity below
        return exp

    # def _explain_one(self, lore_exp, x_series):
    #     try:
    #         exp = lore_exp.explain(x_series.values)
    #         rule = exp.getRules()
    #         crules = exp.getCounterfactualRules()
    #         premise = rule.get('premise', [])
    #
    #         def fmt(p):
    #             att, op, thr = p['att'], p['op'], float(p['thr'])
    #             if att in self._cat_cols:
    #                 idx = int(round(thr))
    #                 idx = max(0, min(idx, len(self._inv_maps[att]) - 1))
    #                 label = self._inv_maps[att][idx]
    #                 readable_op = '=' if op in ('>', '>=') else '≠'
    #                 return f"{att} {readable_op} '{label}'"
    #             return f"{att} {op} {thr:.4g}"
    #
    #         rule_str = ' AND '.join(fmt(p) for p in premise)
    #         rule_features = [p['att'] for p in premise]
    #         crules_str = [
    #             ' AND '.join(fmt(p) for p in cr.get('premise', []))
    #             for cr in (crules or [])
    #         ]
    #         return {'rule_str': rule_str, 'rule_features': rule_features, 'crules_str': crules_str}
    #     except Exception as e:
    #         return {'rule_str': f'[error: {e}]', 'rule_features': [], 'crules_str': []}

    def _explain_one(self, lore_exp, x_series):
        try:
            exp = lore_exp.explain(x_series.values)
            rule = exp.getRules()
            crules = exp.getCounterfactualRules()
            premise = rule.get('premise', [])

            def fmt(p):
                att, op, thr = p['att'], p['op'], float(p['thr'])
                if att in self._cat_cols:
                    idx = int(round(thr))
                    idx = max(0, min(idx, len(self._inv_maps[att]) - 1))
                    label = self._inv_maps[att][idx]
                    readable_op = '=' if op in ('>', '>=') else '!='
                    return f"{att} {readable_op} '{label}'"
                return f"{att} {op} {thr:.4g}"

            rule_str = ' AND '.join(fmt(p) for p in premise)
            rule_features = [p['att'] for p in premise]
            crules_str = [
                ' AND '.join(fmt(p) for p in cr.get('premise', []))
                for cr in (crules or [])
            ]
            # ---- native fidelity: dt.score(Z, Yb) -- the surrogate scored on the SAME neighbourhood
            # it was fitted on, i.e. IN-SAMPLE. A decision tree evaluated on its own training data
            # saturates by construction: this is a second, independent reason for the ~0.97-0.98
            # ceiling (the first being n=6 explained samples).
            fid = exp.expDict.get('fidelity')

            # ---- repaired fidelity: redraw a FRESH neighbourhood with the same generator and
            # score the surrogate out-of-sample, on continuous outputs and balanced classes.
            oos = {'lore_fidelity_oos_acc': float('nan'), 'lore_fidelity_oos_bacc': float('nan'),
                   'lore_fidelity_oos_r2': float('nan'), 'neigh_pos_rate': float('nan')}
            try:
                from sklearn.metrics import balanced_accuracy_score, r2_score
                dt = exp.exp.dt                                             # the surrogate tree
                Z = lore_exp.lore_explainer.neighgen_fn(x_series.values, self.oos_samples)
                p_model = self._predictor(Z)                                # continuous model prob
                y_model = (p_model >= self.positive_threshold).astype(int)
                y_surr = dt.predict(Z)
                oos['lore_fidelity_oos_acc'] = float((y_surr == y_model).mean())
                oos['lore_fidelity_oos_r2'] = float(r2_score(p_model, dt.predict_proba(Z)[:, 1]))
                oos['neigh_pos_rate'] = float(y_model.mean())               # did Z cross the boundary?
                if len(set(y_model.tolist())) > 1:
                    oos['lore_fidelity_oos_bacc'] = float(balanced_accuracy_score(y_model, y_surr))
            except Exception as e:
                oos['oos_error'] = f'{type(e).__name__}: {e}'

            return {'rule_str': rule_str, 'rule_features': rule_features, 'crules_str': crules_str,
                    'fidelity': float(fid) if fid is not None else float('nan'),
                    'rule_len': len(premise), **oos}
        except Exception as e:
            return {'rule_str': f'[LORE skipped: {type(e).__name__}: {e}]', 'rule_features': [], 'crules_str': [],
                    'fidelity': float('nan'), 'rule_len': 0}

    # ── Public API ───────────────────────────────────────────────────────
    def explain(self, raw_df, images, labels, model, mode='context'):
        self._build_sub_models(model)

        # Label-encode categoricals → all-numeric for LORE
        encoded_df = self._label_encode_df(raw_df)
        feat_cols_no_target = [c for c in encoded_df.columns if c != 'target']

        n = len(encoded_df)
        ex_idx = np.arange(n)
        if n > self.max_explain_samples:
            ex_idx = np.sort(np.random.choice(n, self.max_explain_samples, replace=False))

        ex_df = encoded_df.iloc[ex_idx].reset_index(drop=True)
        ex_labels = labels[ex_idx].ravel()

        if mode == 'context':
            img = images.mean(axis=0, keepdims=True).astype(np.float32)
        elif mode == 'tabular_only':
            img = np.zeros((1, *images.shape[1:]), dtype=np.float32)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        image_embedding = self._image_encoder(img, training=False).numpy()[0]
        predictor = self._make_predictor(image_embedding, feat_cols_no_target)
        preds = predictor(ex_df)
        lore_exp = self._build_explainer(predictor, encoded_df)

        freq_all, freq_pos, freq_neg = defaultdict(int), defaultdict(int), defaultdict(int)
        rules = []

        for i in range(len(ex_df)):
            result = self._explain_one(lore_exp, ex_df.iloc[i])
            print(f"[LORE {i+1}/{len(ex_df)}] rule_features={result['rule_features']}")
            result['pred'] = float(preds[i])
            result['label'] = int(ex_labels[i])
            rules.append(result)
            for feat in result['rule_features']:
                raw_feat = feat.split("=")[0] if "=" in feat else feat
                freq_all[raw_feat] += 1
                (freq_pos if preds[i] >= self.positive_threshold else freq_neg)[raw_feat] += 1

        def _agg(key):
            v = [r[key] for r in rules
                 if isinstance(r.get(key), float) and not np.isnan(r[key])]
            return float(np.mean(v)) if v else float('nan')

        def _sem(key):
            v = [r[key] for r in rules
                 if isinstance(r.get(key), float) and not np.isnan(r[key])]
            return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan')

        _lens = [r['rule_len'] for r in rules if r.get('rule_len')]
        return {
            'feature_freq_all': dict(freq_all),
            'feature_freq_pos': dict(freq_pos),
            'feature_freq_neg': dict(freq_neg),
            'explained_indices': ex_idx,
            'n_explained': len(rules),
            'fidelity': _agg('fidelity'),                       # native, IN-SAMPLE (saturates)
            'fidelity_sem': _sem('fidelity'),                   # <- with n=6 this did not exist
            'fidelity_oos_acc': _agg('lore_fidelity_oos_acc'),  # out-of-sample accuracy
            'fidelity_oos_bacc': _agg('lore_fidelity_oos_bacc'),# balanced (kills class inflation)
            'fidelity_oos_r2': _agg('lore_fidelity_oos_r2'),    # continuous (kills 0.5-threshold)
            'fidelity_oos_r2_sem': _sem('lore_fidelity_oos_r2'),
            'neigh_pos_rate': _agg('neigh_pos_rate'),
            'rule_len': float(np.mean(_lens)) if _lens else float('nan'),
        }, rules, preds


    from pathlib import Path


class LoreCallback(keras.callbacks.Callback):
    """
    Runs LoreEngine every N epochs on a frozen validation subset.
    Operates on RAW features (pre-FeatureSpace) — XAI-Lib handles
    categorical encoding internally.
    """

    def __init__(
        self,
        data_source: tf.data.Dataset,
        raw_feature_df: pd.DataFrame,
        feat_cols: List[str],
        feature_space,
        base_dir: str,
        max_explain_samples: int = 12,
        positive_threshold: float = 0.5,
        top_k: int = 10,
        every_n_epochs: int = 2,
        max_samples: int = 256,
        max_batches: Optional[int] = None,
        lore_config: Optional[Dict] = None,
        rng_seed: int = 0,
        modes: tuple = ('context',),
    ):
        super().__init__()
        self.data_source = data_source
        self.raw_feature_df = raw_feature_df  # pre-FeatureSpace validation DataFrame
        self.feat_cols = feat_cols
        self.top_k = int(top_k)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.max_samples = int(max_samples)
        self.max_batches = max_batches
        self.rng_seed = int(rng_seed)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._images = None
        self._raw_df = None
        self._labels = None
        self.last_quality = {}   # aggregate LORE quality (fidelity); merged into run diagnostics

        self.modes = tuple(modes)
        self._engine = LoreEngine(
            feature_space=feature_space,
            feat_cols=feat_cols,
            max_explain_samples=int(max_explain_samples),
            positive_threshold=float(positive_threshold),
            lore_config=lore_config,
        )

    def _freeze_batch(self, ds: tf.data.Dataset):
        """Collect images and labels from dataset, raw features from DataFrame."""
        imgs, labels = [], []
        n = 0
        for b_idx, (x, y) in enumerate(ds):
            if self.max_batches and b_idx >= self.max_batches:
                break
            remaining = self.max_samples - n
            if remaining <= 0:
                break
            imgs.append(x['images'].numpy()[:remaining])
            labels.append(y.numpy()[:remaining])
            n += imgs[-1].shape[0]

        if n == 0:
            raise RuntimeError('No samples collected for LORE.')
        self._images = np.concatenate(imgs)
        self._labels = np.concatenate(labels)
        # Take corresponding rows from raw DataFrame (same order, no shuffle in val)
        self._raw_df = self.raw_feature_df.iloc[:n][self.feat_cols].reset_index(drop=True)

    def on_train_begin(self, logs=None):
        self._freeze_batch(self.data_source)

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.every_n_epochs != 0 or self._images is None:
            return

        np.random.seed(self.rng_seed)
        epoch_name = f'epoch{epoch:03d}'
        epoch_dir = self.base_dir / epoch_name
        epoch_dir.mkdir(parents=True, exist_ok=True)

        row = {'epoch': int(epoch)}
        # 'tabular_only' doubled the cost but never reached the report -- the published
        # lore_fidelity is the 'context' one. Dropping it pays for max_explain_samples 6 -> 32.
        for mode in self.modes:
            mode_dir = epoch_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)

            stats, rules, preds = self._engine.explain(
                raw_df=self._raw_df,
                images=self._images,
                labels=self._labels,
                model=self.model,
                mode=mode,
            )

            self.last_quality[f'lore_fidelity_{mode}'] = stats.get('fidelity')
            self.last_quality[f'lore_rule_len_{mode}'] = stats.get('rule_len')
            row[f'lore_fidelity_{mode}'] = stats.get('fidelity')

            plot_lore_feature_frequency(
                freq_all=stats['feature_freq_all'],
                freq_pos=stats['feature_freq_pos'],
                freq_neg=stats['feature_freq_neg'],
                k=self.top_k,
                epoch_num=epoch_name,
                mode=mode,
                out_path=str(mode_dir / f'lore_feature_freq_top{self.top_k:02d}_{mode}_{epoch_name}.png'),
            )

            save_lore_rules_text(
                rules=rules,
                epoch_num=epoch_name,
                mode=mode,
                out_path=str(mode_dir / f'lore_rules_{mode}_{epoch_name}.txt'),
            )

        # headline quality (context mode) + per-epoch trajectory CSV (one row/epoch when run each epoch)
        self.last_quality['lore_fidelity'] = self.last_quality.get('lore_fidelity_context')
        self.last_quality['lore_rule_len'] = self.last_quality.get('lore_rule_len_context')
        append_epoch_row(self.base_dir / 'lore_fidelity_per_epoch.csv', row)



# --------------------------------------------------------------------------------------
# LORE sweep plots (cell 37)
# --------------------------------------------------------------------------------------

def plot_param_vs_lore_fidelity(df, param_col, title=None, out_dir=None):
    """LORE fidelity per strategy (mirror of plot_param_vs_best_auc)."""
    if df.empty or param_col not in df.columns or "lore_fidelity" not in df.columns:
        return
    d = df.dropna(subset=["lore_fidelity"]).sort_values(param_col)
    if d.empty:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(d[param_col], d["lore_fidelity"], marker="o", color="#c44e52")
    plt.xlabel(param_col); plt.ylabel("lore_fidelity")
    plt.title(title or f"LORE fidelity vs {param_col}")
    plt.grid(alpha=0.3)
    _finalize_plot(out_dir, f"{param_col}_vs_lore_fidelity.png")


def plot_lore_fidelity_vs_val_auc(df, param_col=None, title=None, out_dir=None, fname=None):
    """Scatter (one point per run): x=best_val_auc (performance), y=lore_fidelity (explanation)."""
    if df.empty or not {"best_val_auc", "lore_fidelity"}.issubset(df.columns):
        return
    d = df.dropna(subset=["best_val_auc", "lore_fidelity"])
    if d.empty:
        return
    plt.figure(figsize=(7, 5))
    if param_col and param_col in d.columns:
        sc = plt.scatter(d["best_val_auc"], d["lore_fidelity"], c=d[param_col],
                         cmap="viridis", s=60, edgecolors="black")
        plt.colorbar(sc, label=param_col)
        for _, r in d.iterrows():
            plt.annotate(f"{r[param_col]:.3g}", (r["best_val_auc"], r["lore_fidelity"]),
                         fontsize=7, xytext=(3, 3), textcoords="offset points")
    else:
        plt.scatter(d["best_val_auc"], d["lore_fidelity"], s=60, edgecolors="black")
    plt.xlabel("best_val_auc (performance)"); plt.ylabel("lore_fidelity (explanation)")
    plt.title(title or "LORE fidelity vs validation AUC (per strategy)")
    plt.grid(alpha=0.3)
    _finalize_plot(out_dir, fname or "lore_fidelity_vs_val_auc.png")


def plot_lore_fidelity_vs_val_auc_by_epoch(df, out_dir=None, mode="context"):
    """Single strategy: aligns per-epoch lore_fidelity (xai/lore/lore_fidelity_per_epoch.csv)
    with per-epoch val_auc (history.json) and traces the (val_auc, lore_fidelity) trajectory.
    Meaningful when the LORE callback runs every epoch; with best-epoch-only it shows one point."""
    if df.empty or "run_dir" not in df.columns:
        return
    fcol = f"lore_fidelity_{mode}"
    for _, r in df.iterrows():
        run_dir = r.get("run_dir")
        if not isinstance(run_dir, str):
            continue
        fcsv = os.path.join(run_dir, "xai", "lore", "lore_fidelity_per_epoch.csv")
        hjson = os.path.join(run_dir, "history.json")
        if not (os.path.exists(fcsv) and os.path.exists(hjson)):
            continue
        fdf = pd.read_csv(fcsv)
        if fcol not in fdf.columns or fdf.empty:
            continue
        val_auc = json.load(open(hjson)).get("val_auc", [])
        ep = fdf["epoch"].astype(int).values
        keep = (ep >= 0) & (ep < len(val_auc))
        if not keep.any():
            continue
        x = np.asarray(val_auc)[ep[keep]]
        y = fdf[fcol].values[keep]
        tag = str(r.get("setting_tag", r.get("run_id", "run")))
        plt.figure(figsize=(6, 5))
        plt.plot(x, y, marker="o", color="#4c72b0")
        for xi, yi, e in zip(x, y, ep[keep]):
            plt.annotate(f"e{e}", (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
        plt.xlabel("val_auc"); plt.ylabel(f"lore_fidelity ({mode})")
        plt.title(f"LORE fidelity vs val_auc by epoch \u2014 {tag}")
        plt.grid(alpha=0.3)
        _finalize_plot(out_dir, f"lore_fid_vs_valauc_by_epoch_{tag}.png")


def plot_lore_xai_suite(df, param_col, out_dir=None, by_epoch=True):
    """One call: per-strategy LORE fidelity, fidelity-vs-val_auc scatter (all strategies),
    and (optional) per-strategy per-epoch fidelity-vs-val_auc trajectory."""
    plot_param_vs_lore_fidelity(df, param_col, out_dir=out_dir)
    plot_lore_fidelity_vs_val_auc(df, param_col, out_dir=out_dir)
    if by_epoch:
        plot_lore_fidelity_vs_val_auc_by_epoch(
            df, out_dir=os.path.join(out_dir, "lore_by_epoch") if out_dir else None)

