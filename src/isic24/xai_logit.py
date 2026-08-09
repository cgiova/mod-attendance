"""Exact logit decomposition — the only attendance measure that is not an approximation.

Every fusion head in this study is additive, so the pre-activation logit splits into a
bias plus one term per modality, with no residual:

    z(x) = b + c_img(x) + c_tab(x)

* concatenation heads (``vanilla``, ``loss_regularization``, ``fixed_scaling``,
  ``learnable_scaling``) split the kernel: ``c_img = u_img . W[:d_img]``;
* additive heads (``controlled_fusion`` via ``FusionWithGamma``, ``adaptive_gating`` via
  ``Add``) share the kernel: ``c_img = gamma_img * u_img . W``.

``u_img`` / ``u_tab`` are the tensors as they enter the sum, so any per-branch scaling
(``ConstantScale``, ``ScalarMultiply``, the softmax gates) is already inside them.  For
``adaptive_gating`` the gates read both modalities: the split stays exact per sample, but
``c_img`` is not a function of the image alone.  That is a property of the architecture,
not of this code, and it is why the gate weights are reported alongside.

**Magnitude is not discriminative value.**  This says how much each branch moves the
logit, which is a different question from what an ablation (AUC drop) or an endpoint model
answers.  On ISIC-2024 the three disagree, and reporting one as "the" reliance is the error
this module exists to make impossible.

Centering: raw contributions are offset by whatever the branch emits at a declared
baseline (constant grey image, encoded-zero features), so a contribution of zero means
"this branch says nothing beyond the baseline".  Centred terms still sum to the centred
logit exactly.

Cost: one forward pass per run over the adopted sample.  No gradients, no perturbations.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from . import xai_faithfulness as xf
from . import xai_subset as xs
from .xai_driver import load_run_model, make_probe_batch

# Reconstruction tolerance: the split is algebraic, so anything above float noise means the
# head is not the additive one we think it is.
RECON_TOL = 1e-4

FUSION_CLASSES = ("Concatenate", "Add", "FusionWithGamma")


def _final_dense(model):
    """The output Dense: the last layer with a kernel, and the model's output producer."""
    producer = model.outputs[0]._keras_history.operation
    if getattr(producer, "kernel", None) is None:
        raise ValueError(f"{model.name}: the output comes from {producer.name}, which has no kernel")
    return producer


def _fusion_of(dense):
    """The layer feeding the output Dense, plus its two branch tensors."""
    inbound = dense.input
    if isinstance(inbound, (list, tuple)):
        raise ValueError(f"{dense.name}: expected a single fused input, got {len(inbound)}")
    fusion = inbound._keras_history.operation
    cls = type(fusion).__name__
    if cls not in FUSION_CLASSES:
        raise ValueError(f"{fusion.name} is a {cls}; expected one of {FUSION_CLASSES}")
    branches = fusion.input
    if not isinstance(branches, (list, tuple)) or len(branches) != 2:
        raise ValueError(f"{fusion.name}: expected 2 branch tensors, got {branches}")
    return fusion, list(branches)


def head_split(model):
    """Describe how this model's logit decomposes.

    Returns ``dict(kind, gamma_img, gamma_tab, w_img, w_tab, bias, branches)`` where the
    two weight vectors are already sliced for the concatenation case.
    """
    dense = _final_dense(model)
    fusion, branches = _fusion_of(dense)
    W = np.asarray(dense.kernel).reshape(-1)
    bias = float(np.asarray(dense.bias).reshape(-1)[0]) if dense.use_bias else 0.0

    kind = type(fusion).__name__
    if kind == "Concatenate":
        d_img = int(branches[0].shape[-1])
        w_img, w_tab = W[:d_img], W[d_img:]
        g_img = g_tab = 1.0
    else:
        # The gammas of FusionWithGamma are non-trainable WEIGHTS, and get_config() reports
        # the value they were built with -- the post-hoc gamma sweep changes only the weight.
        w_img = w_tab = W
        g_img = float(np.asarray(getattr(fusion, "gamma_img", 1.0)))
        g_tab = float(np.asarray(getattr(fusion, "gamma_tab", 1.0)))

    return dict(kind=kind, fusion_name=fusion.name, dense_name=dense.name,
                gamma_img=g_img, gamma_tab=g_tab, w_img=w_img, w_tab=w_tab,
                bias=bias, branches=branches)


def _branch_model(model, split):
    import keras
    return keras.Model(model.inputs, split["branches"])


def assert_branch_order(model, split, batch):
    """Branch 0 must respond to the image input, branch 1 to the features.

    Every builder emits ``[image, tabular]``, but the order is an assumption until it is
    measured -- and under ``adaptive_gating`` both branches also move with the other
    modality, so only this direction of the test is meaningful.
    """
    bm = _branch_model(model, split)
    x = {k: np.asarray(v)[:2] for k, v in batch.items()}
    base = [np.asarray(t) for t in bm.predict(x, verbose=0)]

    perturbed = []
    for key in ("images", "features"):
        y = {k: v.copy() for k, v in x.items()}
        y[key] = np.zeros_like(y[key]) if key == "images" else y[key] * 0.0 + 1.0
        perturbed.append([np.asarray(t) for t in bm.predict(y, verbose=0)])

    moved_img = np.abs(perturbed[0][0] - base[0]).max()
    moved_tab = np.abs(perturbed[1][1] - base[1]).max()
    if moved_img == 0 or moved_tab == 0:
        raise AssertionError(
            f"{model.name}: branch order not confirmed (image branch moved {moved_img:.2e}, "
            f"tabular branch moved {moved_tab:.2e}); both must be non-zero")


def _baseline_batch(batch):
    """Constant grey image, encoded-zero features -- the June convention."""
    return {"images": np.full_like(np.asarray(batch["images"])[:1], 0.5),
            "features": np.zeros_like(np.asarray(batch["features"])[:1])}


def contributions(model, batch, check_order=True):
    """Per-sample logit decomposition for one model. Raises if it does not reconstruct."""
    split = head_split(model)
    if check_order:
        assert_branch_order(model, split, batch)

    bm = _branch_model(model, split)
    x = {k: np.asarray(v) for k, v in batch.items()}
    u_img, u_tab = [np.asarray(t) for t in bm.predict(x, verbose=0)]

    c_img = split["gamma_img"] * (u_img @ split["w_img"])
    c_tab = split["gamma_tab"] * (u_tab @ split["w_tab"])

    b_img, b_tab = [np.asarray(t) for t in bm.predict(_baseline_batch(batch), verbose=0)]
    off_img = float(split["gamma_img"] * (b_img @ split["w_img"])[0])
    off_tab = float(split["gamma_tab"] * (b_tab @ split["w_tab"])[0])

    pred = np.asarray(model.predict(x, verbose=0)).reshape(-1)
    recon = 1.0 / (1.0 + np.exp(-(split["bias"] + c_img + c_tab)))
    delta = float(np.abs(recon - pred).max())
    if delta > RECON_TOL:
        raise AssertionError(
            f"{model.name}: logit split does not reconstruct the prediction "
            f"(max |delta| = {delta:.2e} > {RECON_TOL:.0e}); the head is not additive")

    df = pd.DataFrame({
        "prediction": pred,
        "image_logit_raw": c_img,
        "tabular_logit_raw": c_tab,
        "image_logit_centered": c_img - off_img,
        "tabular_logit_centered": c_tab - off_tab,
    })
    df["total_logit_centered"] = df["image_logit_centered"] + df["tabular_logit_centered"]
    denom = df["image_logit_centered"].abs() + df["tabular_logit_centered"].abs()
    df["image_abs_share_descriptive"] = np.where(denom > 0,
                                                 df["image_logit_centered"].abs() / denom, np.nan)
    meta = dict(head_kind=split["kind"], bias=split["bias"],
                image_baseline_logit=off_img, tabular_baseline_logit=off_tab,
                recon_max_abs_delta=delta,
                image_head_dim=int(u_img.shape[-1]), tabular_head_dim=int(u_tab.shape[-1]))
    return df, meta


def _target_aligned(df, labels):
    """Signed towards the sample's own class: positive means "pushes towards the truth"."""
    sign = np.where(np.asarray(labels).reshape(-1) > 0.5, 1.0, -1.0)
    df["target"] = np.asarray(labels).reshape(-1).astype(int)
    df["image_target_aligned"] = sign * df["image_logit_centered"]
    df["tabular_target_aligned"] = sign * df["tabular_logit_centered"]
    return df


OUT_DIRNAME = "logit_decomposition"
PER_SAMPLE_CSV = "logit_contributions_per_sample.csv"


def decompose_run(run_dir, art_path, subset_idx=None, batch=None, labels=None,
                  overwrite=False, write=True, probe_fn=None):
    """Decompose one run over the adopted sample and write its per-sample CSV.

    ``batch``/``labels`` (or ``probe_fn``, called only if the run has to be recomputed) let
    a caller build the probe once and reuse it: the images are decoded once, not 110 times.
    """
    abs_dir = xs.run_dir_abs(run_dir, art_path)
    out_dir = os.path.join(abs_dir, "xai", OUT_DIRNAME)
    out_csv = os.path.join(out_dir, PER_SAMPLE_CSV)
    if write and os.path.exists(out_csv) and not overwrite:
        return pd.read_csv(out_csv)

    if batch is None:
        batch, labels = probe_fn() if probe_fn else make_probe_batch(subset_idx=subset_idx)
    model, cfg = load_run_model(abs_dir)
    df, meta = contributions(model, batch)
    df = _target_aligned(df, labels)

    idx = np.arange(len(df)) if subset_idx is None else np.asarray(subset_idx, int)
    df.insert(0, xs.ROW_ID_KEY, xs.stamp_row_ids(idx)[xs.ROW_ID_KEY])
    df.insert(1, "val_index", idx)
    df.insert(0, "run_id", cfg.get("run_id", os.path.basename(abs_dir)))
    for k, v in meta.items():
        df[k] = v

    if write:
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(out_csv, index=False)
        json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return df


def _run_row(df):
    """One row per run: the by-seed summary of a per-sample table."""
    img, tab = df["image_logit_centered"], df["tabular_logit_centered"]
    pos, neg = df["target"] == 1, df["target"] == 0
    return dict(
        run_id=df["run_id"].iat[0], head_kind=df["head_kind"].iat[0],
        n_samples=len(df), n_positive_rows=int(pos.sum()),
        # `read_explained_ids` e non la colonna a mano: i CSV gia' su disco portano
        # `isic_id`, quelli nuovi `row_id`, e la lettura dei due sta in un posto solo.
        n_unique_lesions=int(np.unique(xs.read_explained_ids(df)).size),
        image_head_dim=int(df["image_head_dim"].iat[0]),
        tabular_head_dim=int(df["tabular_head_dim"].iat[0]),
        mean_abs_image_logit_centered=img.abs().mean(),
        mean_abs_tabular_logit_centered=tab.abs().mean(),
        mean_image_abs_share_descriptive=df["image_abs_share_descriptive"].mean(),
        median_image_abs_share_descriptive=df["image_abs_share_descriptive"].median(),
        frac_image_dominated=float((df["image_abs_share_descriptive"] > 0.5).mean()),
        mean_image_target_aligned=df["image_target_aligned"].mean(),
        mean_tabular_target_aligned=df["tabular_target_aligned"].mean(),
        # Aggregating over both classes hides the asymmetry: alignment is spent mostly on
        # the benign side, so a single mean reads as "the image is aligned" on 50% negatives.
        image_target_aligned_pos=df.loc[pos, "image_target_aligned"].mean(),
        image_target_aligned_neg=df.loc[neg, "image_target_aligned"].mean(),
        tabular_target_aligned_pos=df.loc[pos, "tabular_target_aligned"].mean(),
        tabular_target_aligned_neg=df.loc[neg, "tabular_target_aligned"].mean(),
        image_share_pos=df.loc[pos, "image_abs_share_descriptive"].mean(),
        image_share_neg=df.loc[neg, "image_abs_share_descriptive"].mean(),
        mean_image_logit_positive=img[pos].mean(), mean_tabular_logit_positive=tab[pos].mean(),
        mean_image_logit_negative=img[neg].mean(), mean_tabular_logit_negative=tab[neg].mean(),
        recon_max_abs_delta=df["recon_max_abs_delta"].iat[0],
    )


AGG_BY_RUN = "logit_decomposition_by_run.csv"
AGG_BY_FAMILY = "logit_decomposition_by_family.csv"

SHARE_COLS = ["mean_image_abs_share_descriptive", "median_image_abs_share_descriptive",
              "frac_image_dominated", "mean_image_target_aligned",
              "mean_tabular_target_aligned"]

CLASS_COLS = ["image_share_pos", "image_share_neg", "image_target_aligned_pos",
              "image_target_aligned_neg", "tabular_target_aligned_pos",
              "tabular_target_aligned_neg"]


def family_table(by_run):
    """One row per family: the shares with their between-seed sd, then the class split.

    The two halves are aggregated differently — `mean`+`std` against `mean` alone — so the
    first arrives with two column levels and the second with one. They are flattened here
    before being put side by side: joining them as they come raises, which is why the class
    columns existed in the code and never reached the artifact.
    """
    share = by_run.groupby("impl_name")[SHARE_COLS].agg(["mean", "std"]).round(4)
    share.columns = [f"{c}__{stat}" for c, stat in share.columns]
    cls = by_run.groupby("impl_name")[CLASS_COLS].mean().round(4)
    return share.join(cls).reset_index()


def decompose_study(master, art_path, subset_idx=None, central=None, overwrite=False,
                    out_dir=None, verbose=True):
    """Decompose every run in ``master`` (optionally restricted to the central settings).

    Writes the per-run and per-family tables next to the other cross-run summaries, and
    returns both.  The image probe is built once and shared by every run.
    """
    m = xs.drop_ablations(master, "logit decomposition", verbose)
    if central:
        m = m[[bool(central.get(i) == s) for i, s in zip(m["impl_name"], m["setting_tag"])]]

    # Built on first use: re-reading finished runs from disk then costs no image decoding.
    probe = {}

    def _probe():
        if not probe:
            probe["b"], probe["y"] = make_probe_batch(subset_idx=subset_idx)
        return probe["b"], probe["y"]

    rows, per_sample = [], []
    for k, (_, r) in enumerate(m.iterrows(), 1):
        df = decompose_run(r["run_dir"], art_path, subset_idx=subset_idx, probe_fn=_probe,
                           overwrite=overwrite)
        df["impl_name"], df["setting_tag"], df["seed"] = r["impl_name"], r["setting_tag"], r["seed"]
        row = _run_row(df)
        row.update(impl_name=r["impl_name"], setting_tag=r["setting_tag"], seed=r["seed"])
        rows.append(row)
        per_sample.append(df)
        if verbose:
            print(f"[{k}/{len(m)}] {r['impl_name']:>20} {r['setting_tag']:>10} "
                  f"share {row['mean_image_abs_share_descriptive']:.3f}  "
                  f"delta {row['recon_max_abs_delta']:.1e}")

    by_run = pd.DataFrame(rows)
    by_family = family_table(by_run)

    out_dir = out_dir or os.path.join(art_path, "runs", xs.AGGREGATE_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    by_run.to_csv(os.path.join(out_dir, AGG_BY_RUN), index=False)
    by_family.to_csv(os.path.join(out_dir, AGG_BY_FAMILY), index=False)
    return by_run, by_family, pd.concat(per_sample, ignore_index=True)
