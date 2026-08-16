"""Explainability engines for the modality-attendance study.

Moved out of the notebooks so that SHAP and Grad-CAM can be run against ANY saved
checkpoint, not only during the run that trained it. See `xai_driver` for the orchestration
that used to live in the training notebook's shared runner.

LORE lives in `xai_lore_sa`, and only there. The engine that ran on `xailib` was removed
once the runs it produced had been archived: it was not the default since 2026-07-29, and
the reason it had been kept -- reproducing the `lore_fidelity` column of the registry --
went away with the finding that no canonical run carries that column.
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

    The per-epoch CSVs were opened in plain append mode. That was harmless while they were
    only ever written once, during the run that trained the model. It stops being harmless
    now that the whole point is to re-run the engines post-hoc against an existing
    `run_dir`: a second pass appended a duplicate epoch, and whoever read the file by epoch
    got two rows where the run has one.
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
    """Sum the encoded dimensions onto the RAW feature they come from.

    The mapping lives in `xai_metrics.raw_name_for`, in one place. There used to be a
    hand-written `name.startswith("age_approx_bin_")` here: on ISIC24 it is equivalent (the
    only discretized feature is `age_approx`), but the notebook builds the names with
    `f"{name}_bin_{b}"` for EVERY `Discretization`. On a dataset with a second discretized
    feature this function would have mapped `foo_bin_0` onto the raw `"foo_bin_0"` instead of
    onto `"foo"` — silently, and disagreeing with `encoded_indices_for_raw` above, which
    handles `_bin_` generically. Found on 2026-08-01.
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
        subset_idx=None,           # POSITIONAL indices in validation_df; None = prefix
        subset_fingerprint=None,   # stamped into the npz: without it the files are not
                                   # reusable under a different sample (guard in
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

        from . import xai_subset as xs
        # The explained validation rows and their identity: `explained_indices` are positions
        # INSIDE the sample, so on their own they do not say which lesions those are.
        vi = (None if self.subset_idx is None
              else np.asarray(self.subset_idx, int)[
                  np.asarray(stats["explained_indices"], int)])

        # Persist arrays + tabular share for post-hoc analysis
        np.savez_compressed(
            epoch_dir / f"shap_arrays_{epoch_name}.npz",
            shap_values=shap_vals,                          # (n_ex, n_features_encoded)
            preds=preds,
            explained_indices=stats["explained_indices"],
            mean_abs_all=mean_abs_all,
            mean_abs_pos=mean_abs_pos,
            mean_abs_neg=mean_abs_neg,
            # The sample fingerprint: `explained_indices` are POSITIONS, so without it an npz
            # cannot be told from one produced on other rows.
            **({} if self.subset_fingerprint is None
               else {"subset_fingerprint": str(self.subset_fingerprint)}),
            # The POSITION IN `validation_df` of every explained sample, and its row id — the
            # same two keys, with the same meaning, that Grad-CAM (`gradcam_arrays_*.npz`),
            # LORE (per-sample CSV) and faithfulness write.
            #
            # They are needed because `explained_indices` is NOT that thing: those are positions
            # INSIDE the sample (0..n-1), so two engines on the same sample produce indices that
            # look alike and are not comparable. That is exactly how the SHAP<->LORE agreement
            # came out NaN. And a position alone does not survive a rebuild of the fold: the
            # identity does, and it is `row_id`.
            **({} if vi is None else {"val_index": vi, **xs.stamp_row_ids(vi)}),
        )

        # Tabular share = sum |SHAP| over tabular dims / total. Here all encoded features
        # ARE tabular (image branch is separate input), so share = mean_abs_all.sum() over all dims.
        # More useful: per-raw-feature share aggregated into "categorical vs numeric vs interaction" buckets,
        # but simplest paper-ready signal is the raw importance vector itself (already saved).

        total_abs = float(mean_abs_all.sum())
        top1_share = float(mean_abs_all.max() / (total_abs + 1e-12))
        top5_share = float(np.sort(mean_abs_all)[::-1][:5].sum() / (total_abs + 1e-12))

        # SHAPE metrics (readability): entropy compactness and k90 sparsity, from the same
        # attributions already in memory. They come out here, with everything else, rather than
        # from a cell to launch afterwards. They are a COMPLEMENT: `shap_mass` beside them says
        # whether there is anything to explain — with the tabular branch off, compactness would
        # be 1.000 (the maximum) and `xai_metrics` sets it to NaN on purpose.
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
        # The RAW heatmaps, exposed so the callback can SAVE them. Until 01/08 this function
        # returned the RGB collage alone: the only traces of the image side were 106 PNGs with
        # the colormap already applied, and no metric can be recovered from those (entropy,
        # activated area, connected components, IoU between configurations). It is not in the
        # return value so as not to break callers that unpack three elements.
        self.last_cams = cams
        self.last_preds = preds.numpy() if isinstance(preds, tf.Tensor) else preds

        # --- Triptychs with display images ---
        triptychs = []
        preds_np = self.last_preds
        # The collage grows with the batch: at n=144 it would be an enormous PNG nobody looks
        # at, and composing it costs more than Grad-CAM itself. Beyond `max_grid` only the
        # arrays are saved, which are the piece that matters.
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
        # TWO MODES, one same loading. `image_only` zeroes the tabular features and asks
        # "where would it look WITHOUT the tabular branch"; `context` uses the real features and
        # asks "where does it actually look". An attendance study needs both, and their
        # DIFFERENCE (IoU between the two masks) is itself a signal: how much the tabular branch
        # moves visual attention.
        #
        # They cost little together because the expense is elsewhere: loading the checkpoint and
        # reading the dataset up to the 144 lesions is paid ONCE per run, and the gradient step
        # is the only thing that doubles.
        #
        # `modes=None` = the historical behaviour: one mode only, the engine's own.
        self.modes = tuple(modes) if modes else (engine.mode,)
        # The POSITIONS in validation_df of the images in the probe batch, and the sample
        # fingerprint. They make the heatmaps realignable with SHAP and LORE: until 01/08
        # Grad-CAM ran on the first 8 rows and left no trace of which they were.
        self.subset_idx = subset_idx
        self.subset_fingerprint = subset_fingerprint
        # How many images per gradient step. None = all at once (the historical behaviour,
        # fine at n=8 and out of GPU memory at n=144).
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
            # IN CHUNKS, not all at once. With 144 images in a single `GradientTape` the guided
            # gradients need some 22 GiB of activations alone (measured 01/08:
            # ResourceExhaustedError on a (144, 8, 8, 1248) tensor with 23.9 GiB of GPU).
            # Grad-CAM is per sample: there is no dependency between images, so splitting the
            # batch gives the SAME result at constant memory.
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
                    grid = g          # the collage comes from the first chunk: it is a preview
            self.engine.last_cams = np.concatenate(cams_parts, axis=0)
            preds = np.concatenate(preds_parts, axis=0)
            if not self.save_dir:
                continue
            # The historical name under a single mode (`epoch_003.png`), suffixed under two:
            # the canonical runs stay readable by whatever looks for them under that name.
            sfx = "" if single else f"_{mode}"
            out = (np.clip(grid, 0, 1) * 255).astype(np.uint8)
            cv2.imwrite(str(self.save_dir / f"epoch_{epoch:03d}{sfx}.png"),
                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

            cams = getattr(self.engine, "last_cams", None)
            if cams is None:
                continue
            by_mode[mode] = np.asarray(cams, np.float32)
            preds_by_mode[mode] = np.asarray(preds, np.float32).reshape(len(cams), -1)

            # THE RAW HEATMAPS: the precondition of every image-side metric. From a PNG with
            # the colormap applied neither entropy nor area nor IoU can be recovered.
            # `val_index` and `row_id` come with them for the same reason they come with the LORE
            # rules: without them, WHICH lesions are explained is unknown.
            from . import xai_subset as xs
            vi = (None if self.subset_idx is None
                  else np.asarray(self.subset_idx, int)[:len(cams)])
            np.savez_compressed(
                self.save_dir / f"gradcam_arrays{sfx}_epoch_{epoch:03d}.npz",
                cams=by_mode[mode], preds=np.asarray(preds, np.float32), mode=mode,
                **({} if vi is None else {"val_index": vi, **xs.stamp_row_ids(vi)}),
                **({} if self.subset_fingerprint is None
                   else {"subset_fingerprint": str(self.subset_fingerprint)}))

            # The metrics go into the TABLE and nowhere else. The per-run figure that used to
            # be here (mean heatmap, fraction above threshold, bar chart of the metrics) was
            # removed on 01/08: the first two say nothing and the third duplicates a CSV row in
            # pixels. Over 110 runs x 2 modes that was 220 PNGs of clutter.
            shape = gradcam_shape_from_cams(by_mode[mode])
            append_epoch_row(self.save_dir / "gradcam_summary.csv",
                             {"epoch_mode": f"{epoch:03d}:{mode}", "epoch": int(epoch),
                              "mode": mode, "n_cams": int(len(cams)), **shape},
                             key="epoch_mode")   # <- two rows per epoch: the key is the pair

        # The signal that comes from having BOTH: how much the tabular branch moves visual
        # attention. IoU 1.0 = the tabular branch does not change where the model looks.
        #
        # THE IoU ALONE IS NOT ENOUGH, and that is why this row used to come out almost empty.
        # On five families out of six the head is `Dense(1)` over a `Concatenate`, so the logit
        # is ADDITIVE in the two branches and d(logit)/d(conv_out) does not depend on the
        # features at all: the heatmap is identical by construction, not as a result, and the
        # IoU is 1.000 everywhere. Meanwhile the PREDICTION moves by up to 0.53 — that is, the
        # comparison row said "nothing happened" exactly while the only thing that moved was
        # recorded nowhere. The row now carries the difference on the raw heatmaps and the one
        # on the predictions.
        if self.save_dir and {"image_only", "context"} <= set(by_mode):
            contrast = cam_mode_contrast(by_mode["context"], by_mode["image_only"],
                                         preds_by_mode.get("context"),
                                         preds_by_mode.get("image_only"))
            append_epoch_row(self.save_dir / "gradcam_summary.csv",
                             {"epoch_mode": f"{epoch:03d}:context_vs_image_only",
                              "epoch": int(epoch), "mode": "context_vs_image_only",
                              "n_cams": int(len(by_mode["context"])), **contrast},
                             key="epoch_mode")
