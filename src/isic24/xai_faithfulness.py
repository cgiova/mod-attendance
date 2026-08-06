"""Explanation-faithfulness metrics for the modality-attendance study.

Two halves:

* the **stratified re-integration** of already-computed per-sample curves -- pure
  arithmetic over the saved ``*_curves.npz``, no model, no GPU;
* the **perturbation machinery** (``explanation_faithfulness`` and friends), moved
  verbatim out of the post-hoc notebook so it can score any saved checkpoint.

The second half needs the notebook's data objects.  Rather than reading them out of the
caller's globals -- which is what tied it to one notebook -- it takes them from an
injected :class:`XaiContext`; call :func:`configure` once per session.
"""

from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from .xai_engines import _finalize_plot, encoded_indices_for_raw


# --------------------------------------------------------------------------------------
# Injected context: the notebook state the perturbation code needs
# --------------------------------------------------------------------------------------

@dataclass
class XaiContext:
    """The seven notebook objects the faithfulness code used to read from globals.

    ``validation_df``/``validation_ds`` must stay in the SAME order: the tabular
    perturbation aligns encoded batches to raw rows positionally
    (``raw_full.iloc[:got]``), so a shuffled validation set silently mismatches rows.
    """
    feat_cols: Sequence[str]
    training_df: Any
    validation_df: Any
    validation_ds: Any
    feature_space: Any
    to_feature_dict: Callable
    encoded_feature_names: Sequence[str]


class _Unconfigured:
    """Stands in for the context until `configure` runs, so the moved bodies fail with a
    usable message instead of `AttributeError: 'NoneType' object has no attribute ...`."""

    def __getattr__(self, name):
        raise RuntimeError(
            f"xai_faithfulness.configure(XaiContext(...)) has not been called, but the "
            f"perturbation code just asked for {name!r}. Build the context from the "
            f"notebook's FEAT_COLS / training_df / validation_df / validation_ds / "
            f"feature_space / _to_feature_dict / encoded_feature_names.")


_CTX = _Unconfigured()


def configure(ctx: XaiContext, **overrides) -> None:
    """Bind the session context and optionally override any ``FAITH_*`` knob.

    Overrides land in this module's globals, which is exactly where the moved bodies
    read them from (they kept their verbatim ``globals().get(...)`` lookups).
    """
    global _CTX, _FAITH_TAB_BASELINE
    _CTX = ctx
    # The baseline is a lazily-cached function of training_df; a new context (different
    # eval set, different resampling) must not silently reuse the previous one.
    _FAITH_TAB_BASELINE = None
    for k, v in overrides.items():
        globals()[k] = v



# --------------------------------------------------------------------------------------
# Reportability floors.  Measured motivation (vanilla/seed_47, 128 samples): the four
# confidence x correctness cells hold 61 / 45 / 18 / **4** samples -- "confidently wrong"
# is single-digit per run.  Below these floors we emit the count and withhold the mean,
# rather than publishing an average over a handful of samples.
MIN_CELL_N = 10          # per-run
MIN_CELL_N_POOLED = 25   # pooled within a strategy family

# `_strata_ci` is a percentile bootstrap, which is anti-conservative at small n.
# We never attach an interval to a cell that is itself below the reporting floor.
N_BOOTSTRAP = 2000


def _trapz(y, x, axis=-1):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return fn(y, x=x, axis=axis)


# --------------------------------------------------------------------------------------
# Strata definitions
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Strata:
    """Boolean masks over the samples of one run, plus the confidence threshold used."""
    masks: dict
    conf_thresh: float
    n_total: int


def build_strata(labels, preds, conf_split="per_run", conf_thresh=None) -> Strata:
    """Split samples by class, by model confidence, and by correctness.

    ``confidence = |p - 0.5|``.  The split point is the **median of this run**
    (``conf_split="per_run"``, default) so that "sure vs unsure" is a well-defined
    *within-run* contrast; pass an explicit ``conf_thresh`` for a fixed global cut.
    A quantile split is used rather than fixed bands such as [0.4, 0.6] because the
    models are not calibrated (family ECE runs to ~0.099) and fixed bands would leave
    the uncertain cell nearly empty.
    """
    labels = np.asarray(labels).ravel()
    preds = np.asarray(preds).ravel()

    conf = np.abs(preds - 0.5)
    if conf_thresh is None:
        if conf_split == "per_run":
            conf_thresh = float(np.median(conf))
        else:
            raise ValueError("conf_split='pooled' requires an explicit conf_thresh")

    pos = labels > 0.5
    sure = conf >= conf_thresh
    correct = (preds >= 0.5) == pos

    masks = {
        "all": np.ones_like(pos, dtype=bool),
        # class
        "class_pos": pos,
        "class_neg": ~pos,
        # confidence
        "sure": sure,
        "unsure": ~sure,
        # correctness
        "correct": correct,
        "wrong": ~correct,
        # the four requested cells: confidence x correctness
        "sure_correct": sure & correct,
        "sure_wrong": sure & ~correct,
        "unsure_correct": ~sure & correct,
        "unsure_wrong": ~sure & ~correct,
        # class x confidence -- needed to tell a genuine confidence effect apart from the
        # class effect: the model is confident mostly on negatives, so the marginal
        # "sure" cell is largely the "class_neg" cell wearing a different hat.
        "pos_sure": pos & sure,
        "pos_unsure": pos & ~sure,
        "neg_sure": ~pos & sure,
        "neg_unsure": ~pos & ~sure,
    }
    return Strata(masks=masks, conf_thresh=float(conf_thresh), n_total=int(pos.size))


# --------------------------------------------------------------------------------------
# Per-sample gains out of a saved curves npz
# --------------------------------------------------------------------------------------

def persample_gains(npz, method, direction, q_max=None):
    """Per-sample ``AUC(explanation-ordered) - AUC(random-ordered)`` from saved curves.

    Returns ``None`` when the requested method/direction is absent from the file.

    The aggregate published elsewhere is ``trapz(exp.mean(0)) - trapz(rand.mean(0))``,
    which equals the mean of these per-sample values exactly (the integral is linear) --
    that identity is what `verify_reintegration` checks.

    ``q_max`` re-integrates over the clean low-budget regime only (see REPORT sec. 7.4:
    at q >= 0.4 so much of the image is destroyed that the metric is dominated by
    out-of-distribution behaviour).
    """
    ke, kr = f"{direction}_exp_{method}", f"{direction}_rand_{method}"
    if ke not in npz.files or kr not in npz.files:
        return None

    q = np.asarray(npz["q_grid"], dtype=float)
    exp_, rand_ = np.asarray(npz[ke], dtype=float), np.asarray(npz[kr], dtype=float)

    if q_max is not None:
        keep = q <= q_max + 1e-12
        q, exp_, rand_ = q[keep], exp_[:, keep], rand_[:, keep]
        if q.size < 2:
            return None

    return _trapz(exp_ - rand_, q, axis=1)


def gain_window(npz, q_max=None):
    """Width of the budget window `persample_gains` integrates over, for normalisation.

    A gain is an integral over q, so a window of half the width carries mechanically
    about half the area: `q<=0.2` is ~1/3 of `q<=0.6` before any explanation quality
    enters.  Dividing by this width puts the two on a common *rate* scale (mean logit
    gap per unit of budget).  The saved grids start at q=0, so the width equals `q_max`,
    but it is read off the grid rather than assumed.
    """
    q = np.asarray(npz["q_grid"], dtype=float)
    if q_max is not None:
        q = q[q <= q_max + 1e-12]
    if q.size < 2:
        return np.nan
    return float(q[-1] - q[0])


# The sign each ranking convention implies for a gain, given the sample's label.  Only
# `target` is per-sample: deleting evidence for the TRUE class lowers the positive logit
# on positives and raises it on negatives, so a single aggregate flip cannot express it --
# in any stratum holding both classes, `mean(g*sgn) != +-mean(g)`.  Applying the sign to
# the per-sample vector BEFORE the stratum mean is the only correct order.
def mode_sign(mode, labels=None):
    """Return ``+1``/``-1``, or a per-sample vector for ``target``. ``None`` -> ``+1``."""
    if mode in (None, "", "abs", "posev"):
        return 1.0
    if mode == "negev":
        return -1.0
    if mode == "target":
        if labels is None:
            raise ValueError("mode='target' needs labels")
        return np.where(np.asarray(labels).ravel() > 0.5, 1.0, -1.0)
    raise ValueError(f"unknown mode {mode!r}")


def _strata_ci(values, n_boot=N_BOOTSTRAP, alpha=0.05, seed=0):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


# --------------------------------------------------------------------------------------
# Run discovery
# --------------------------------------------------------------------------------------

SIGNED_MODES = ("abs", "posev", "negev", "target")


def discover_curve_files(art_path, master_rows_csv=None, include_signed=False):
    """Map every CANONICAL run to its available per-sample curve files.

    The canonical registry is ``summaries/master_results_rows.csv`` (110 runs).  This
    filter matters: ``runs/`` holds 130 ``faithfulness_curves.npz`` files, the extra 20
    being superseded study dirs and seed_49 -- analysing them unfiltered shifts the
    family means (see STATUS: controlled_fusion +0.070 vs +0.149).

    ``include_signed`` adds the ``aligned_signed`` runs, which carry FOUR npz per run
    (one per ranking convention), so a row is keyed by (run, source, **mode**) -- the
    ``mode`` column is what keeps abs/posev/negev/target from being pooled together
    downstream.  Off by default: the signed protocol covers 6 runs, one per family,
    picked for convenience.

    Returns a DataFrame with one row per (run, source, mode).
    """
    runs_path = os.path.join(art_path, "runs")
    if master_rows_csv is None:
        master_rows_csv = os.path.join(art_path, "summaries", "master_results_rows.csv")

    master = pd.read_csv(master_rows_csv)
    aligned_root = os.path.join(runs_path, "_robustness", "stage2_aligned_attr")
    signed_root = os.path.join(runs_path, "_robustness", "aligned_signed")

    rows = []
    for _, r in master.iterrows():
        # run_dir in the CSV is an absolute path from another machine/session; re-root it.
        tail = str(r.run_dir).split("artifacts/enhanced/")[-1]
        run_dir = os.path.join(art_path, tail)

        canonical = os.path.join(run_dir, "xai", "faithfulness", "faithfulness_curves.npz")
        aligned = os.path.join(aligned_root, str(r.run_id), "faithfulness_aligned_attr_curves.npz")

        found = [("canonical", "abs", canonical), ("aligned", "abs", aligned)]
        if include_signed:
            # DOVE stanno le curve signed. Dal 01/08 le scrive `aligned_signed_faithfulness`
            # dentro la run (`xai/shap_tab/`), sul campione a lesioni distinte: 110 run x 4
            # modi = 440 file. La vecchia radice `_robustness/aligned_signed/` ne ha 24, di
            # 6 run sul PREFISSO, scelte per convenienza — leggere quella significa
            # pubblicare medie di famiglia con n_runs=1 su un campione superato.
            # Per-run PRIMA, radice vecchia come fallback: cosi' una run gia' rifatta non
            # viene mai letta dalla copia vecchia, e le run che non lo sono restano visibili.
            sdir_run = os.path.join(run_dir, "xai", "shap_tab")
            sdir_old = os.path.join(signed_root, str(r.run_id))
            for m in SIGNED_MODES:
                fname = f"faithfulness_aligned_signed_{m}_curves.npz"
                p_run = os.path.join(sdir_run, fname)
                found.append(("aligned_signed", m,
                              p_run if os.path.exists(p_run)
                              else os.path.join(sdir_old, fname)))

        for source, mode, path in found:
            if os.path.exists(path):
                rows.append({
                    "impl_name": r.impl_name, "run_id": r.run_id, "seed": r.seed,
                    "setting_tag": r.setting_tag, "source": source, "mode": mode,
                    "npz_path": path,
                })
    out = pd.DataFrame(rows)

    # Il fallback qui sopra e' comodo e ha un buco: se UNA run non avesse il file per-run,
    # verrebbe pescata dalla radice vecchia — cioe' dal PREFISSO — e finirebbe nella stessa
    # media di famiglia di 109 run sul campione a lesioni distinte, senza un errore. E' la
    # forma di difetto che e' gia' costata due volte (`combined_faithfulness.csv` a 130
    # righe, `lore_sa_quality` con due `pos_rate`). Qui non si indovina: si dice.
    if len(out) and (out.source == "aligned_signed").any():
        sg = out[out.source == "aligned_signed"]
        per_run = sg.npz_path.str.contains(os.sep + "shap_tab" + os.sep)
        if per_run.any() and (~per_run).any():
            mixed = sorted(sg.loc[~per_run, "run_id"].unique())
            warnings.warn(
                f"[CURVE FILES] le curve `aligned_signed` vengono da DUE campioni: "
                f"{int(per_run.sum())} file dalle run (lesioni distinte) e "
                f"{int((~per_run).sum())} da _robustness/aligned_signed (prefisso).\n"
                f"  Run ancora sul prefisso: {mixed[:6]}{' ...' if len(mixed) > 6 else ''}\n"
                f"  Mediarle insieme mescola due popolazioni. Rifare quelle run, oppure "
                f"filtrare su `npz_path.str.contains('shap_tab')`.", RuntimeWarning,
                stacklevel=2)
    return out


# --------------------------------------------------------------------------------------
# The stratified table
# --------------------------------------------------------------------------------------

# (source, method) pairs worth reading, and the budget cutoffs to integrate to.
# The aligned attributor was only ever run with methods=("tabular",), but it DID save the
# insertion curves -- REPORT_S sec.2's "deletion only" is inaccurate, and `ins` below
# surfaces the insertion column at no cost.
DEFAULT_METHODS = {
    "canonical": ("tabular", "image", "joint"),
    "aligned": ("tabular",),
    "aligned_signed": ("tabular",),
}
DEFAULT_Q_MAX = (0.20, None)   # None = full budget (0.60), the pre-specified metric


def stratified_table(art_path, methods=None, q_maxes=DEFAULT_Q_MAX,
                     conf_split="per_run", conf_thresh=None, files=None, verbose=True,
                     include_signed=False):
    """Build the per-run x stratum faithfulness table. Pure re-integration, no GPU.

    Two columns per cell: ``gain`` is the raw integral, unchanged and directly
    comparable with everything published; ``gain_norm`` is the same quantity divided by
    the budget window ``q_width``, which is the only one of the two that may be compared
    ACROSS ``q_max`` values (see `gain_window`).  Confidence bounds are reported on the
    raw scale -- divide by ``q_width`` for the normalised ones.

    For signed sources the ranking convention's sign is applied to the per-sample vector
    before the stratum mean (`mode_sign`); ``target`` is not recoverable otherwise.
    """
    methods = methods or DEFAULT_METHODS
    if files is None:
        files = discover_curve_files(art_path, include_signed=include_signed)

    out = []
    for _, f in files.iterrows():
        mode = f.get("mode", "abs")
        with np.load(f.npz_path) as npz:
            if "labels" not in npz.files or "preds_orig" not in npz.files:
                continue
            labels = npz["labels"]
            st = build_strata(labels, npz["preds_orig"],
                              conf_split=conf_split, conf_thresh=conf_thresh)
            sgn = mode_sign(mode, labels)

            for method in methods.get(f.source, ()):
                for direction in ("del", "ins"):
                    for q_max in q_maxes:
                        g = persample_gains(npz, method, direction, q_max=q_max)
                        if g is None:
                            continue
                        g = g * sgn
                        width = gain_window(npz, q_max)
                        for stratum, m in st.masks.items():
                            v = g[m]
                            n = int(v.size)
                            ok = n >= MIN_CELL_N
                            lo, hi = _strata_ci(v) if ok else (np.nan, np.nan)
                            out.append({
                                "impl_name": f.impl_name, "run_id": f.run_id,
                                "seed": f.seed, "setting_tag": f.setting_tag,
                                "source": f.source, "mode": mode, "method": method,
                                "direction": direction,
                                "q_max": 0.60 if q_max is None else q_max,
                                "q_width": width,
                                "stratum": stratum, "n": n,
                                "gain": float(v.mean()) if ok else np.nan,
                                "gain_norm": float(v.mean()) / width if ok else np.nan,
                                "gain_all_n": float(v.mean()) if n else np.nan,
                                "ci_lo": lo, "ci_hi": hi,
                                "conf_thresh": st.conf_thresh,
                            })
        if verbose and len(out) % 5000 < 200:
            print(f"  ... {len(out)} rows", flush=True)

    return pd.DataFrame(out)


def pool_by_family(df):
    """Pool per-run cells within a strategy family (5 runs x 128 samples).

    Sample-weighted, so a run contributing 4 samples to a thin cell does not count as
    much as one contributing 61.  `gain_all_n` is used as the input because per-run cells
    below MIN_CELL_N have a withheld `gain` but a perfectly usable sample mean -- the
    floor applies to what we *report*, and pooling is exactly how a thin cell becomes
    reportable.
    """
    g = df.dropna(subset=["gain_all_n"]).copy()
    g["_w"] = g["n"] * g["gain_all_n"]

    # `mode` MUST be a grouping key: a signed run contributes four rankings per cell, and
    # pooling abs with posev/negev/target averages quantities of opposite sign into a
    # number that looks perfectly reasonable.  Older tables without the column are all
    # `abs` by construction.
    keys = ["impl_name", "source", "method", "direction", "q_max", "stratum"]
    if "mode" in g.columns:
        keys.insert(2, "mode")
    if "q_width" in g.columns:
        keys.append("q_width")

    agg = (g.groupby(keys)
             .agg(n=("n", "sum"), _wsum=("_w", "sum"), n_runs=("run_id", "nunique"))
             .reset_index())
    agg["gain"] = agg["_wsum"] / agg["n"]
    agg.loc[agg["n"] < MIN_CELL_N_POOLED, "gain"] = np.nan
    if "q_width" in agg.columns:
        agg["gain_norm"] = agg["gain"] / agg["q_width"]
    return agg.drop(columns="_wsum")


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------

def verify_reintegration(npz_path, json_path=None, tol=1e-9):
    """Check the per-sample re-integration reproduces the aggregate saved in the JSON.

    Guards the whole module: if this holds, every stratified number below is the same
    quantity as the published one, merely restricted to a subset of samples.
    """
    import json
    if json_path is None:
        json_path = npz_path.replace("_curves.npz", ".json")
    meta = json.load(open(json_path))

    problems = []
    with np.load(npz_path) as npz:
        for key, val in meta.items():
            if not key.startswith("faithfulness_") or not key.endswith("_gain"):
                continue
            parts = key.split("_")          # faithfulness_<method>_<del|ins>_gain
            if len(parts) != 4:
                continue
            method, direction = parts[1], parts[2]
            g = persample_gains(npz, method, direction)
            if g is None:
                continue
            delta = abs(float(g.mean()) - float(val))
            if delta > tol:
                problems.append((key, float(val), float(g.mean()), delta))
    return problems


# ======================================================================================
# Perturbation machinery -- moved verbatim from the post-hoc notebook (cell 24, the
# hardened variant).  Bodies are unchanged except that the seven notebook globals it read
# now come from the injected XaiContext; see `configure`.
# ======================================================================================

# =====================================================================
# Explanation-Guided Deletion / Insertion Faithfulness AUC
# ---------------------------------------------------------------------
# Primary explanation-quality metric, comparable across strategies.
#   s(x)            = logit(p_model(x))            (positive-class logit)
#   C_exp(q)        = mean_i[ s(x_i) - s(x_i with top-q% evidence removed) ]
#   C_rand(q)       = same budget removed at random (mean over n_random)
#   deletion_auc    = trapz_q C(q)   (q=0 -> 0 anchor)
#   faithfulness_gain = deletion_auc_exp - deletion_auc_rand   (higher = better)
# Insertion is the symmetric secondary metric (start from neutral baseline,
# re-insert top-q% evidence). Computed for tabular (SHAP), image (Grad-CAM)
# and joint (both modalities together). Reuses saved SHAP arrays so no
# GradientExplainer re-fit is needed.
# =====================================================================
import os, json, glob, time, gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2

# ---- Config (tweak to control cost) ---------------------------------
FAITH_N_SAMPLES    = 128
FAITH_BUDGETS      = (0.05, 0.10, 0.20, 0.40, 0.60)
FAITH_METHODS      = ("tabular", "image", "joint")
FAITH_DO_INSERTION = True          # set False if too heavy
FAITH_N_RANDOM     = 3             # random-baseline repetitions
FAITH_IMG_GRID     = None          # None = the native Grad-CAM grid; an int overrides it
FAITH_CAM_STRIDE   = 32            # backbone total stride: the CAM is image_size / stride
FAITH_IMG_REPLACE  = "blur"        # 'blur' | 'mean'
FAITH_RNG_SEED     = 0
FAITH_EVAL_BATCH   = 64
FAITH_SAVE_PLOTS   = False         # per-run deletion/insertion curve PNGs (off: keep only the aggregate per-strategy plots)
ENABLE_FAITHFULNESS_IN_RUN = True  # hook inside run_experiment for future runs

_FAITH_TAB_BASELINE = None         # lazily built {col: median/mode}
_TRAPZ = getattr(np, "trapezoid", np.trapz)  # numpy>=2 renamed trapz -> trapezoid


def _logit_np(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _build_tabular_baseline():
    """Median for numeric raw features, most-frequent token for categoricals."""
    base = {}
    for col in _CTX.feat_cols:
        s = _CTX.training_df[col]
        if pd.api.types.is_numeric_dtype(s):
            base[col] = float(s.median())
        else:
            base[col] = s.astype(str).mode().iloc[0]
    return base


def faith_subsets(k=5, n=FAITH_N_SAMPLES, seed=0, n_val=None):
    """`k` DISJOINT random subsets of `n` validation samples each.

    Subset 0 is the canonical first-`n` (so its numbers must reproduce the published
    faithfulness.json); subsets 1..k-1 are disjoint random draws from the rest. This is
    what answers "would a DIFFERENT batch of 128 give the same answer?" -- the published
    numbers rest on a single batch of 128 out of 1091 (11.7%)."""
    n_val = n_val or len(_CTX.validation_df)
    assert k * n <= n_val, f"{k} x {n} > {n_val} validation samples: subsets cannot be disjoint"
    rest = np.setdiff1d(np.arange(n_val), np.arange(n))
    rest = np.random.default_rng(seed).permutation(rest)
    return [np.arange(n)] + [np.sort(rest[i * n:(i + 1) * n]) for i in range(k - 1)]


# Campione ATTIVO di default. `None` = comportamento storico (le prime N righe). Lo imposta
# `xai_subset.adopt(...)` una volta per sessione, ed e' cosi' che la scelta del campione
# diventa un DEFAULT del framework invece di un argomento da ricordarsi a ogni chiamata:
# ogni call site che non passa `subset_idx` eredita questo.
FAITH_SUBSET_IDX = None


def _build_faith_subset(n, subset_idx=None):
    """Validation samples for the faithfulness run.

    subset_idx=None  -> `FAITH_SUBSET_IDX` se impostato, altrimenti le prime `n` righe.
    subset_idx=array -> quelle righe di validation (usato dal check a 5 sottoinsiemi).

    NB `faith_subsets` costruisce apposta il sottoinsieme 0 come `np.arange(n)`, cioe' il
    prefisso: quel check confronta batch diversi FRA LORO e deve restare sul prefisso anche
    quando il default e' un altro. Passa sempre `subset_idx` esplicito, quindi non eredita.
    """
    if subset_idx is None and FAITH_SUBSET_IDX is not None:
        subset_idx = FAITH_SUBSET_IDX
    raw_full = _CTX.validation_df.reset_index(drop=True)
    take_n = n if subset_idx is None else int(np.max(subset_idx)) + 1
    imgs, labs, got = [], [], 0
    for x, y in _CTX.validation_ds:
        b = int(x["images"].shape[0])
        take = min(b, take_n - got)
        imgs.append(x["images"][:take].numpy())
        labs.append(np.asarray(y)[:take].reshape(-1))
        got += take
        if got >= take_n:
            break
    imgs = np.concatenate(imgs, 0).astype(np.float32)
    labs = np.concatenate(labs, 0)
    if subset_idx is not None:
        subset_idx = np.asarray(subset_idx, int)
        return (raw_full.iloc[subset_idx][_CTX.feat_cols].reset_index(drop=True),
                imgs[subset_idx], labs[subset_idx])
    raw_df = raw_full.iloc[:got][_CTX.feat_cols].reset_index(drop=True)
    return raw_df, imgs, labs


# Impronta del campione ATTIVO in questa sessione. `None` = comportamento storico (le prime
# N righe). La imposta il notebook quando lavora su un campione stratificato, e da quel
# momento `_load_shap_values` RIFIUTA gli npz che non la portano.
ACTIVE_SUBSET_FINGERPRINT = None


def _load_shap_values(run_dir, best_epoch, model, raw_df, imgs):
    """Reuse the per-sample SHAP arrays saved by ShapCallback; fallback to recompute.

    GUARD SULL'IDENTITA' DEL CAMPIONE (2026-07-31). Gli npz salvano `explained_indices` come
    POSIZIONI (0..127), non come ISIC id. Con un campione diverso dal prefisso quelle
    posizioni indicano altre lesioni: `keep = ei < len(raw_df)` passa lo stesso, la
    reindicizzazione riesce, e ne escono valori SHAP di una run applicati alle feature di
    altri campioni. Nessun errore, nessun warning, numeri nell'ordine di grandezza giusto.
    E' il modo piu' facile di ottenere un risultato plausibile e falso, quindi qui
    l'assenza di stampigliatura e' un ERRORE, non un avviso: tutti gli npz canonici sono
    senza, e sono esattamente quelli che non vanno riusati fuori dal prefisso.
    """
    # DOVE cercare: con un campione adottato gli array giusti stanno in `xai/shap_tab/`
    # (li scrive `ShapCallback` sotto lo stesso campione); `xai/shap/` tiene i canonici di
    # giugno, che sono sul PREFISSO e senza stampigliatura — cioe' esattamente quelli che il
    # guard qui sotto deve rifiutare. Cercare solo li' significava rifiutare sempre, anche
    # dopo aver rigenerato SHAP correttamente. La scelta della cartella e' di `xai_driver`,
    # in un posto solo, cosi' scrittura / protezione / lettura non possono divergere.
    from .xai_driver import shap_subdir
    shap_dir = os.path.join(run_dir, "xai", shap_subdir())
    cand = os.path.join(shap_dir, f"epoch_{int(best_epoch):03d}",
                        f"shap_arrays_epoch_{int(best_epoch):03d}.npz")
    if not os.path.exists(cand):
        hits = sorted(glob.glob(os.path.join(shap_dir, "epoch_*", "shap_arrays_epoch_*.npz")))
        cand = hits[-1] if hits else None
    if cand is None and shap_dir.endswith("shap_tab"):
        # niente ancora sul campione: si ricade sul canonico, che il guard rifiutera' con
        # il messaggio giusto invece di far finta che SHAP non esista e ricalcolarlo.
        hits = sorted(glob.glob(os.path.join(run_dir, "xai", "shap", "epoch_*",
                                             "shap_arrays_epoch_*.npz")))
        cand = hits[-1] if hits else None
    if cand is not None and os.path.exists(cand) and globals().get("EVAL_IS_CANONICAL", True):
        d = np.load(cand)
        if ACTIVE_SUBSET_FINGERPRINT is not None:
            stamp = str(d["subset_fingerprint"]) if "subset_fingerprint" in d.files else None
            if stamp != ACTIVE_SUBSET_FINGERPRINT:
                raise RuntimeError(
                    f"[SUBSET GUARD] {cand}\n"
                    f"  impronta dell'npz: {stamp or 'ASSENTE (= prefisso, artefatto canonico)'}\n"
                    f"  impronta attiva:   {ACTIVE_SUBSET_FINGERPRINT}\n"
                    f"  `explained_indices` sono POSIZIONI: riusarle su un campione diverso\n"
                    f"  accoppia i valori SHAP di queste righe alle feature di ALTRE lesioni,\n"
                    f"  senza errore e con numeri plausibili. Due uscite legittime:\n"
                    f"    * l'attributore ALLINEATO, che non legge nulla da disco:\n"
                    f"      xf.aligned_signed_faithfulness(..., subset_idx=..., out_dir=...)\n"
                    f"      — 67 s/run a n=128, misurato su 110 run;\n"
                    f"    * rigenerare SHAP sul campione nuovo scrivendo gli npz con\n"
                    f"      subset_fingerprint={ACTIVE_SUBSET_FINGERPRINT!r} (fuori dalle run canoniche).\n"
                    f"  Per tornare al prefisso: xf.ACTIVE_SUBSET_FINGERPRINT = None.")
        sv = np.asarray(d["shap_values"])
        if sv.ndim == 3:
            sv = sv[..., 0] if sv.shape[-1] == 1 else sv[:, 0, :]
        ei = np.asarray(d["explained_indices"]).astype(int).ravel()
        keep = ei < len(raw_df)
        return sv[keep], ei[keep]
    # ---- saved SHAP arrays unavailable / not applicable ----
    # This lean notebook REUSES the per-sample SHAP arrays saved at training time, keyed to the
    # CANONICAL fold-0 ordering. They are skipped when EVAL_IS_CANONICAL is False (a custom eval
    # set would misalign them). Recomputing SHAP needs ShapEngine (training notebook, cell 24),
    # intentionally not imported here.
    raise FileNotFoundError(
        f"tabular SHAP faithfulness unavailable for {shap_dir} (missing arrays, or non-canonical "
        f"EVAL_SET). Use methods=('image',) on a custom set, or regenerate SHAP in the training notebook.")


def _persample_raw_importance(shap_values):
    """(n_explain, n_encoded) |SHAP| -> (n_explain, n_raw_features) summed per raw feature."""
    n = shap_values.shape[0]
    imp = np.zeros((n, len(_CTX.feat_cols)), dtype=np.float64)
    for ci, col in enumerate(_CTX.feat_cols):
        idx = encoded_indices_for_raw(_CTX.encoded_feature_names, col)
        if idx.size:
            imp[:, ci] = np.abs(shap_values[:, idx]).sum(axis=1)
    return imp


def _encode_raw(df):
    return np.asarray(_CTX.feature_space(_CTX.to_feature_dict(df)))


def _score_logit(model, enc_feats, imgs, batch=None):
    batch = batch or FAITH_EVAL_BATCH
    enc = tf.convert_to_tensor(enc_feats)
    out = []
    for i in range(0, len(imgs), batch):
        p = model({"images": tf.convert_to_tensor(imgs[i:i + batch]),
                   "features": enc[i:i + batch]}, training=False)
        out.append(tf.squeeze(p, -1).numpy())
    return _logit_np(np.concatenate(out))


def _perturb_tabular_encoded(raw_df, order, k, deletion=True):
    """deletion: top-k raw features -> training baseline.
       insertion: start all-baseline, restore top-k real values. Returns encoded (n,128)."""
    global _FAITH_TAB_BASELINE
    if _FAITH_TAB_BASELINE is None:
        _FAITH_TAB_BASELINE = _build_tabular_baseline()
    base = _FAITH_TAB_BASELINE
    n = len(raw_df)
    sel = np.zeros((n, len(_CTX.feat_cols)), dtype=bool)
    if k > 0:
        rows = np.arange(n)[:, None]
        sel[rows, order[:, :k]] = True
    df = raw_df.copy()
    if deletion:
        for ci, col in enumerate(_CTX.feat_cols):
            m = sel[:, ci]
            if m.any():
                df.loc[m, col] = base[col]
    else:
        for col in _CTX.feat_cols:
            df[col] = base[col]
        for ci, col in enumerate(_CTX.feat_cols):
            m = sel[:, ci]
            if m.any():
                df.loc[m, col] = raw_df.loc[m, col].values
    return _encode_raw(df)


def _gradcam_maps(model, imgs, enc_feats, batch=None):
    """Per-sample Grad-CAM saliency upsampled to image size, in [0,1]."""
    batch = batch or FAITH_EVAL_BATCH
    conv = None
    for l in model.layers:
        if isinstance(l, keras.layers.GlobalAveragePooling2D):
            conv = l.input
            break
    if conv is None:
        for l in reversed(model.layers):
            shp = getattr(l, "output_shape", None)
            if isinstance(shp, tuple) and len(shp) == 4:
                conv = l.output
                break
    grad_model = keras.Model(model.inputs, [conv, model.output])
    enc = tf.convert_to_tensor(enc_feats)
    maps = []
    H, W = imgs.shape[1], imgs.shape[2]
    for i in range(0, len(imgs), batch):
        ib = tf.convert_to_tensor(imgs[i:i + batch])
        fb = enc[i:i + batch]
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model({"images": ib, "features": fb}, training=False)
            p = tf.clip_by_value(preds[:, 0], 1e-6, 1.0 - 1e-6)
            loss = tf.reduce_sum(tf.math.log(p) - tf.math.log(1.0 - p))
        grads = tape.gradient(loss, conv_out)
        grads = tf.cast(conv_out > 0, grads.dtype) * tf.cast(grads > 0, grads.dtype) * grads
        w = tf.reduce_mean(grads, axis=(1, 2))                       # (b, C)
        cam = tf.nn.relu(tf.reduce_sum(conv_out * w[:, None, None, :], axis=-1))  # (b,Hc,Wc)
        cam = cam.numpy()
        native = cam.shape[1]
        for j in range(cam.shape[0]):
            c = cam[j] - cam[j].min()
            mx = c.max()
            c = c / mx if mx > 0 else c
            maps.append(cv2.resize(c, (W, H)))
    return np.stack(maps, 0), int(native)


def _patch_saliency_order(cams, grid):
    n, H, W = cams.shape
    ph, pw = H // grid, W // grid
    s = cams[:, :grid * ph, :grid * pw].reshape(n, grid, ph, grid, pw).mean(axis=(2, 4))
    return np.argsort(-s.reshape(n, grid * grid), axis=1)           # desc, (n, grid*grid)


FAITH_BLUR_SIGMA = 8   # production value; Stage 3 sweeps this to test OOD sensitivity


def _image_replacement(imgs, sigma=None):
    if FAITH_IMG_REPLACE == "mean":
        m = imgs.mean(axis=(1, 2), keepdims=True)
        return np.broadcast_to(m, imgs.shape).astype(imgs.dtype).copy()
    sigma = FAITH_BLUR_SIGMA if sigma is None else sigma
    repl = np.empty_like(imgs)
    for i in range(len(imgs)):
        repl[i] = cv2.GaussianBlur(imgs[i], (0, 0), sigmaX=sigma)
    return repl


def _perturb_image(imgs, repl, order, k, grid, deletion=True):
    n, H, W, C = imgs.shape
    ph, pw = H // grid, W // grid
    out = imgs.copy() if deletion else repl.copy()
    src = repl if deletion else imgs
    for i in range(n):
        for pidx in order[i, :k]:
            r, c = divmod(int(pidx), grid)
            y0, x0 = r * ph, c * pw
            out[i, y0:y0 + ph, x0:x0 + pw, :] = src[i, y0:y0 + ph, x0:x0 + pw, :]
    return out


def _bootstrap_ci(vals, n_boot=2000, seed=0):
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = rng.choice(vals, size=(n_boot, vals.size), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def explanation_faithfulness(model, run_dir, best_epoch, n_samples=None, budgets=None,
                             methods=None, do_insertion=None, n_random=None, grid=None,
                             rng_seed=None, save=True, verbose=True,
                             subset_idx=None, blur_sigma=None, tab_imp_override=None,
                             out_name="faithfulness", out_dir=None):
    """Defaults reproduce the original behaviour exactly.

    subset_idx       : validation rows to explain (default: first FAITH_N_SAMPLES).
    blur_sigma       : image-removal blur strength (default: FAITH_BLUR_SIGMA = 8).
    tab_imp_override : (n, 39) per-sample raw importance from ANY attributor, replacing the
                       |SHAP| ranking loaded from disk. This is how Stage 2 feeds in the
                       aligned Permutation attributor.
    out_name         : basename prefix for every file written, so variants never collide.
    out_dir          : WHERE to write. Default (None) = run_dir/xai/faithfulness/, i.e. next to the
                       published artifacts. The robustness stages pass a directory under
                       runs/_robustness/ instead, so the new checks stay physically separate from
                       the published results and can be deleted wholesale without touching them.
    """
    """Compute deletion/insertion faithfulness for a trained model. Saves artifacts
    under run_dir/xai/faithfulness/ and returns a flat dict of headline metrics."""
    t0 = time.time()
    n_samples = n_samples or FAITH_N_SAMPLES
    budgets = tuple(budgets or FAITH_BUDGETS)
    methods = tuple(methods or FAITH_METHODS)
    do_insertion = FAITH_DO_INSERTION if do_insertion is None else do_insertion
    n_random = FAITH_N_RANDOM if n_random is None else n_random
    grid = grid or FAITH_IMG_GRID          # None here: resolved from the CAM below
    rng = np.random.default_rng(FAITH_RNG_SEED if rng_seed is None else rng_seed)
    q_grid = np.array((0.0,) + budgets, dtype=np.float64)

    need_tab = ("tabular" in methods) or ("joint" in methods)
    need_img = ("image" in methods) or ("joint" in methods)

    raw_df, imgs, labs = _build_faith_subset(n_samples, subset_idx=subset_idx)
    tab_imp = None
    if need_tab:
        if tab_imp_override is not None:
            tab_imp = np.asarray(tab_imp_override, float)
            assert tab_imp.shape == (len(raw_df), len(_CTX.feat_cols)), (
                f"tab_imp_override must be ({len(raw_df)}, {len(_CTX.feat_cols)}), got {tab_imp.shape}")
        else:
            sv, ei = _load_shap_values(run_dir, best_epoch, model, raw_df, imgs)
            raw_df = raw_df.iloc[ei].reset_index(drop=True)
            imgs, labs = imgs[ei], labs[ei]
            tab_imp = _persample_raw_importance(sv)[:len(raw_df)]
    n = len(raw_df)
    n_raw = len(_CTX.feat_cols)

    enc_orig = _encode_raw(raw_df)
    s_orig = _score_logit(model, enc_orig, imgs)
    preds_orig = 1.0 / (1.0 + np.exp(-s_orig))

    if need_img:
        cams, cam_grid = _gradcam_maps(model, imgs, enc_orig)
        grid = cam_grid if grid is None else int(grid)
        if grid > cam_grid:
            raise ValueError(
                f"[FAITH IMG GRID] grid={grid} over a {cam_grid}x{cam_grid} Grad-CAM: ranking "
                f"{grid * grid} patches from {cam_grid ** 2} numbers is nominal precision, not "
                f"measured. Leave FAITH_IMG_GRID=None to follow the CAM, or raise image_size "
                f"(the CAM is image_size / {FAITH_CAM_STRIDE}).")
        img_order_exp = _patch_saliency_order(cams, grid)
        repl = _image_replacement(imgs, sigma=blur_sigma)
    n_patches = (grid or 0) ** 2

    def _tab_order(exp):
        if exp:
            return np.argsort(-tab_imp, axis=1)
        return np.argsort(rng.random((n, n_raw)), axis=1)

    def _img_order(exp):
        if exp:
            return img_order_exp
        return np.argsort(rng.random((n, n_patches)), axis=1)

    def _drops(method, exp, deletion):
        tab_o = _tab_order(exp) if method in ("tabular", "joint") else None
        img_o = _img_order(exp) if method in ("image", "joint") else None
        mat = np.zeros((n, len(q_grid)))
        for qi, q in enumerate(q_grid):
            enc, ims = enc_orig, imgs
            if method in ("tabular", "joint"):
                k = int(round(q * n_raw))
                enc = _perturb_tabular_encoded(raw_df, tab_o, k, deletion=deletion)
            if method in ("image", "joint"):
                k = int(round(q * n_patches))
                ims = _perturb_image(imgs, repl, img_o, k, grid, deletion=deletion)
            mat[:, qi] = _score_logit(model, enc, ims)
        return mat

    results = {"n_faith_samples": int(n), "faith_budgets": list(budgets),
               "blur_sigma": float(FAITH_BLUR_SIGMA if blur_sigma is None else blur_sigma),
               "tab_ranking": "shap_gradient" if tab_imp_override is None else "override"}
    npz = {"q_grid": q_grid, "s_orig": s_orig, "labels": labs, "preds_orig": preds_orig}
    curves_json = {}

    for method in methods:
        # ---- deletion: C(q) = s_orig - s_perturbed ----
        del_exp = s_orig[:, None] - _drops(method, exp=True, deletion=True)
        del_rand = np.zeros((n, len(q_grid)))
        for _ in range(n_random):
            del_rand += s_orig[:, None] - _drops(method, exp=False, deletion=True)
        del_rand /= max(1, n_random)
        auc_de = float(_TRAPZ(del_exp.mean(0), q_grid))
        auc_dr = float(_TRAPZ(del_rand.mean(0), q_grid))
        results[f"deletion_auc_{method}_exp"] = auc_de
        results[f"deletion_auc_{method}_rand"] = auc_dr
        results[f"faithfulness_{method}_del_gain"] = auc_de - auc_dr
        npz[f"del_exp_{method}"] = del_exp
        npz[f"del_rand_{method}"] = del_rand
        curves_json[f"del_exp_{method}"] = del_exp.mean(0).tolist()
        curves_json[f"del_rand_{method}"] = del_rand.mean(0).tolist()

        if do_insertion:
            # ---- insertion: recovery relative to all-baseline (q=0) ----
            ins_exp = _drops(method, exp=True, deletion=False)
            ins_exp = ins_exp - ins_exp[:, [0]]
            ins_rand = np.zeros((n, len(q_grid)))
            for _ in range(n_random):
                m = _drops(method, exp=False, deletion=False)
                ins_rand += m - m[:, [0]]
            ins_rand /= max(1, n_random)
            auc_ie = float(_TRAPZ(ins_exp.mean(0), q_grid))
            auc_ir = float(_TRAPZ(ins_rand.mean(0), q_grid))
            results[f"insertion_auc_{method}_exp"] = auc_ie
            results[f"insertion_auc_{method}_rand"] = auc_ir
            results[f"faithfulness_{method}_ins_gain"] = auc_ie - auc_ir
            npz[f"ins_exp_{method}"] = ins_exp
            npz[f"ins_rand_{method}"] = ins_rand
            curves_json[f"ins_exp_{method}"] = ins_exp.mean(0).tolist()
            curves_json[f"ins_rand_{method}"] = ins_rand.mean(0).tolist()

    # ---- primary headline (joint deletion if available) + per-sample bootstrap CI ----
    primary = "joint" if "joint" in methods else methods[0]
    results["faithfulness_primary_method"] = primary
    results["faithfulness_gain"] = results[f"faithfulness_{primary}_del_gain"]
    ps_gain = _TRAPZ(npz[f"del_exp_{primary}"] - npz[f"del_rand_{primary}"], q_grid, axis=1)
    m, lo, hi = _bootstrap_ci(ps_gain)
    results["faithfulness_gain_ci_lo"] = lo
    results["faithfulness_gain_ci_hi"] = hi
    npz[f"persample_gain_{primary}"] = ps_gain

    if save:
        fdir = out_dir or os.path.join(run_dir, "xai", "faithfulness")
        # GUARD — non scrivere un artefatto calcolato su un campione adottato dentro
        # `xai/faithfulness/`, che e' la cartella dei numeri PUBBLICATI (prefisso storico).
        # `xs.adopt()` imposta il globale FAITH_SUBSET_IDX e `_build_faith_subset` ci ricade
        # a ogni `subset_idx=None`: il loop post-hoc, che passa out_dir=None e
        # out_name="faithfulness", sovrascriverebbe il canonico con numeri di un altro
        # campione, senza errore e con valori plausibili.
        #
        # 2026-08-01: il guard NON e' piu' "out_dir deve stare fuori dalle run". Gli
        # artefatti del campione vivono ora dentro la run, in `xai/shap_tab/`
        # (`xai_subset.out_dir_for`), quindi la condizione si e' spostata su cosa e' la
        # cartella di destinazione — che e' anche piu' robusta di prima: adesso becca anche
        # chi passa `out_dir=<run>/xai/faithfulness` a mano.
        if FAITH_SUBSET_IDX is not None and os.path.basename(
                os.path.normpath(fdir)) == "faithfulness":
            raise RuntimeError(
                f"\n  SUBSET GUARD: un campione XAI e' adottato in questa sessione "
                f"(n={len(FAITH_SUBSET_IDX)}, impronta {ACTIVE_SUBSET_FINGERPRINT}),\n"
                f"  ma questa chiamata scriverebbe '{out_name}' nella cartella dei numeri "
                f"PUBBLICATI:\n"
                f"    {fdir}\n"
                f"  I numeri sarebbero di un campione diverso da quello pubblicato.\n"
                f"  Rimedi: out_dir=xs.out_dir_for(run_dir, 'shap_tab'), oppure disarmare\n"
                f"  il campione con xs.adopt(validation_df, balance=None).")
        os.makedirs(fdir, exist_ok=True)
        # Traccia del campione dentro l'artefatto stesso: un npz senza questa chiave e' del
        # prefisso storico, e non c'e' altro modo di distinguerli a posteriori.
        _eff_subset = subset_idx if subset_idx is not None else FAITH_SUBSET_IDX
        results["subset_fingerprint"] = ACTIVE_SUBSET_FINGERPRINT
        results["n_subset"] = int(len(_eff_subset)) if _eff_subset is not None else None
        np.savez_compressed(os.path.join(fdir, f"{out_name}_curves.npz"), **npz)
        with open(os.path.join(fdir, f"{out_name}.json"), "w") as f:
            json.dump({**results, "curves": curves_json, "q_grid": q_grid.tolist()}, f, indent=2)
        for method in (methods if FAITH_SAVE_PLOTS else ()):
            plt.figure(figsize=(6, 4))
            plt.plot(q_grid, curves_json[f"del_exp_{method}"], marker="o", label="explanation")
            plt.plot(q_grid, curves_json[f"del_rand_{method}"], marker="s", label="random")
            plt.xlabel("budget q removed"); plt.ylabel("mean logit drop C(q)")
            plt.title(f"Deletion faithfulness ({method}) gain={results[f'faithfulness_{method}_del_gain']:+.3f}")
            plt.legend(); plt.grid(alpha=0.3)
            _finalize_plot(fdir, f"{out_name}_deletion_curve_{method}.png")
            if do_insertion:
                plt.figure(figsize=(6, 4))
                plt.plot(q_grid, curves_json[f"ins_exp_{method}"], marker="o", label="explanation")
                plt.plot(q_grid, curves_json[f"ins_rand_{method}"], marker="s", label="random")
                plt.xlabel("budget q inserted"); plt.ylabel("mean logit recovery")
                plt.title(f"Insertion faithfulness ({method}) gain={results[f'faithfulness_{method}_ins_gain']:+.3f}")
                plt.legend(); plt.grid(alpha=0.3)
                _finalize_plot(fdir, f"{out_name}_insertion_curve_{method}.png")

    results["faith_seconds"] = round(time.time() - t0, 2)
    if verbose:
        print(f"[FAITH] n={n} | primary={primary} | gain={results['faithfulness_gain']:+.4f} "
              f"| {results['faith_seconds']}s")
    return results


# ======================================================================================
# Aligned tabular attribution -- SIGNED and class-conditioned
# ======================================================================================
#
# WHY THIS EXISTS.  The original SHAP path used `shap.GradientExplainer` on a wrapper
# taking BOTH inputs, with a background drawn from the very batch being explained.
# Expected gradients therefore integrated a path along which the IMAGE also varied,
# while `_perturb_tabular_encoded` holds the real image fixed and only replaces raw
# tabular columns.  The attributor explained a different function from the one the test
# perturbs (REPORT sec.7.5).  This function closes that gap on four axes at once:
#
#   1. raw 39-feature space          -- the space the perturbation acts in
#   2. this sample's image frozen    -- backbone run once, embedding held constant
#   3. background disjoint, from training
#   4. the score is the LOGIT        -- `_score_logit` integrates logit(p), so an f()
#                                       returning the sigmoid explains the wrong scale
#
# Promoted from the post-hoc notebook's Stage 2 cell, which already had (1)-(3).
#
# SIGN CONVENTION -- read this before interpreting any *_negev_* column.
# Splitting evidence by sign gives two orders with OPPOSITE expected effects:
#   posev  (phi > 0, descending): removing it should LOWER  the positive logit
#   negev  (phi < 0, ascending):  removing it should RAISE  the positive logit
# `explanation_faithfulness` always scores "removal lowers the logit" as good, so the
# negev curve is anti-faithful BY CONSTRUCTION unless its gain is negated.  We negate it
# and say so in the column name (`..._negev_signflipped`).  This is precisely the defect
# REPORT sec.7.5 diagnoses: "rimuovere evidenza contro la classe puo' aumentare
# correttamente il logit e venire contato come anti-fedelta'".

ALIGNED_MAX_EVALS_PER_FEATURE = 2   # Permutation: exact antithetic pass -> 2*n_feat + 1


def _split_cat_cols(training_df, feat_cols):
    return [c for c in feat_cols if not pd.api.types.is_numeric_dtype(training_df[c])]


def _label_maps(training_df, bg_raw, cat_cols):
    """SHAP needs an all-numeric matrix, so categoricals travel as integer codes.

    Sound for Permutation specifically: it only ever SUBSTITUTES whole values taken from
    background rows, never interpolating between them, so the code is an opaque label and
    no arithmetic meaning is imputed to it.
    """
    lmap, imap = {}, {}
    for c in cat_cols:
        levels = sorted(pd.concat([training_df[c], bg_raw[c]]).astype(str).unique())
        lmap[c] = {v: i for i, v in enumerate(levels)}
        imap[c] = {i: v for v, i in lmap[c].items()}
    return lmap, imap


def _find_gap_layer(model):
    import keras
    gap = [l for l in model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D)]
    if not gap:
        raise ValueError(
            "this model has no GlobalAveragePooling2D layer, so the image/head split the "
            "aligned attributor relies on is undefined. Pass a model whose image branch "
            "ends in a GAP, or supply attributions via tab_imp_override.")
    return gap[0]


def aligned_tab_attributions(model, raw_df, imgs, background="baseline", bg_size=100,
                             bg_seed=0, max_evals=None, verbose=False):
    """Per-sample SIGNED attributions over the raw features, aligned to the deletion test.

    ``background``:
      * ``"baseline"`` (default) -- a SINGLE row: exactly the median/mode row
        `_build_tabular_baseline` substitutes during deletion.  Each phi_i is then
        literally "the contribution of moving feature i from the baseline to its observed
        value", i.e. the quantity the perturbation measures.  Efficiency then gives
        ``sum_i phi_i = logit(p(x)) - logit(p(baseline))``.
      * ``"training"`` -- `bg_size` rows sampled from training.  This is what produced the
        already-published `faithfulness_aligned_attr.csv`; kept as the robustness variant.

    Returns ``(phi, meta)`` with ``phi`` of shape ``(n_samples, n_raw_features)``.
    """
    import shap
    import keras
    import tensorflow as tf

    ctx = _CTX
    feat_cols = list(ctx.feat_cols)
    n_feat = len(feat_cols)

    if background == "baseline":
        global _FAITH_TAB_BASELINE
        if _FAITH_TAB_BASELINE is None:
            _FAITH_TAB_BASELINE = _build_tabular_baseline()
        bg_raw = pd.DataFrame([_FAITH_TAB_BASELINE])[feat_cols].reset_index(drop=True)
    elif background == "training":
        bg_raw = ctx.training_df[feat_cols].sample(bg_size, random_state=bg_seed).reset_index(drop=True)
    else:
        raise ValueError(f"background must be 'baseline' or 'training', got {background!r}")

    cat_cols = _split_cat_cols(ctx.training_df, feat_cols)
    lmap, imap = _label_maps(ctx.training_df, bg_raw, cat_cols)

    def encode(df):
        out = df[feat_cols].copy()
        for c in feat_cols:
            if c in cat_cols:
                out[c] = out[c].astype(str).map(lmap[c]).fillna(0)
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce")
        return out.to_numpy(dtype=np.float64)

    def decode(X):
        df = pd.DataFrame(np.atleast_2d(np.asarray(X, dtype=np.float64)), columns=feat_cols)
        for c in feat_cols:
            if c in cat_cols:
                idx = df[c].round().fillna(0).astype(int).clip(0, len(imap[c]) - 1)
                df[c] = [imap[c][i] for i in idx]
            else:
                df[c] = df[c].astype("float32")
        return df

    # Split the model at the GAP so the backbone runs ONCE per sample and only the cheap
    # head is re-evaluated for each of the thousands of Permutation coalitions.
    gap = _find_gap_layer(model)
    img_emb, feat_in = gap.output, model.input["features"]
    img_enc = keras.Model(model.input["images"], img_emb)
    head = keras.Model([img_emb, feat_in], model.output)
    head_fn = tf.function(lambda e, f: head([e, f], training=False), reduce_retracing=True)

    embs = img_enc.predict(imgs, batch_size=32, verbose=0)

    def make_f(emb_i):
        e = tf.constant(emb_i[np.newaxis], dtype=tf.float32)

        def f(X):
            enc = ctx.feature_space(ctx.to_feature_dict(decode(X)))
            n = int(enc.shape[0])
            p = head_fn(tf.repeat(e, n, axis=0), enc).numpy()[:, 0]
            return _logit_np(p)      # the deletion curve integrates logits, not sigmoids
        return f

    X = encode(raw_df)
    BG = encode(bg_raw)
    max_evals = max_evals or (ALIGNED_MAX_EVALS_PER_FEATURE * n_feat + 1)

    phi = np.zeros((len(raw_df), n_feat), dtype=np.float64)
    for i in range(len(raw_df)):
        ex = shap.explainers.Permutation(make_f(embs[i]), BG)
        sv = ex(X[i:i + 1], max_evals=max_evals, silent=True)
        phi[i] = np.asarray(sv.values).reshape(-1)          # SIGNED, no np.abs here
        if verbose and (i + 1) % 32 == 0:
            print(f"    aligned attributions {i+1}/{len(raw_df)}", flush=True)

    meta = {"background": background, "bg_rows": int(len(bg_raw)),
            "max_evals": int(max_evals), "score_space": "logit",
            "n_features": n_feat, "signed": True}
    return phi, meta


def attribution_order_key(phi, labels=None, mode="abs"):
    """Turn signed attributions into the ranking key `explanation_faithfulness` sorts on.

    It ranks by ``argsort(-key)``, so the sign convention is expressed entirely through
    the key -- no change to the perturbation code:

      ``abs``    ``|phi|``   -- magnitude only; comparable with the published column
      ``posev``  ``phi``     -- most positive first  (evidence FOR the positive class)
      ``negev``  ``-phi``    -- most negative first  (evidence AGAINST it)
      ``target`` class-conditioned: ``phi`` for positives, ``-phi`` for negatives, i.e.
                 evidence for the sample's TRUE class first.  Needs ``labels``.

    A per-sample count of same-sign features is returned as ``n_eff``: where a sample has
    fewer than k features of the requested sign the tail is filled by the residual order,
    and that is recorded rather than passed over in silence.
    """
    phi = np.asarray(phi, dtype=np.float64)
    if mode == "abs":
        key, n_eff = np.abs(phi), np.full(len(phi), phi.shape[1])
    elif mode == "posev":
        key, n_eff = phi, (phi > 0).sum(axis=1)
    elif mode == "negev":
        key, n_eff = -phi, (phi < 0).sum(axis=1)
    elif mode == "target":
        if labels is None:
            raise ValueError("mode='target' needs labels")
        sgn = np.where(np.asarray(labels).ravel() > 0.5, 1.0, -1.0)[:, None]
        signed = phi * sgn
        key, n_eff = signed, (signed > 0).sum(axis=1)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return key, n_eff.astype(int)


# Orders whose gain must be NEGATED wholesale: every sample's explanation predicts the
# logit goes UP, while `explanation_faithfulness` scores a drop as faithful.
#   posev  -> every sample expects a drop      -> no flip
#   negev  -> every sample expects a rise      -> flip the aggregate
#   target -> depends on the sample's LABEL    -> NOT expressible as one flip; see
#             `_target_gain_persample`.  Deleting evidence for the true class lowers the
#             positive logit on positives and raises it on negatives, so the sign has to
#             be applied per sample BEFORE averaging.  Flipping the aggregate instead
#             would just re-mix the two populations -- the very error this whole section
#             exists to remove.
SIGN_FLIPPED_MODES = {"negev"}
PERSAMPLE_SIGNED_MODES = {"target"}


def _target_gain_persample(curves_npz_path, labels, direction="del", q_max=None):
    """Label-conditioned gain: per-sample gain times the direction the true class implies.

    Positive result = the explanation moved the logit AWAY from the true class when its
    supporting evidence was removed, which is what faithfulness means once the target is
    the sample's own class rather than the fixed positive one.

    ``q_max`` restricts the integral to the low-budget regime, exactly as it does for
    every other gain in this module; ``None`` is the full budget.
    """
    sgn = mode_sign("target", labels)
    with np.load(curves_npz_path) as npz:
        g = persample_gains(npz, "tabular", direction, q_max=q_max)
    if g is None:
        return None
    return float(np.mean(g * sgn))


def aligned_signed_faithfulness(model, run_dir, best_epoch, phi=None, labels=None,
                                raw_df=None, imgs=None, modes=SIGNED_MODES,
                                background="baseline", out_dir=None,
                                out_prefix="faithfulness_aligned_signed",
                                save=True, verbose=True, q_maxes=(0.20,),
                                subset_idx=None, n_samples=None,
                                allow_clobber=False, **kw):
    """Deletion + insertion under each sign convention, from ONE attribution pass.

    Scrive in `<run_dir>/xai/shap_tab/` — cartella propria, dentro la run, che nessun
    motore canonico tocca. Nomi `out_prefix`-ati, quindi non sfiora `faithfulness.json` /
    `faithfulness_curves.npz` ne' `faithfulness_aligned_attr.csv`.

    Produce anche la FIGURA per-run (`plots/`): e' il motore a farla, non una cella a valle.
    """
    from . import xai_subset as _xs
    if out_dir is None:
        out_dir = _xs.out_dir_for(run_dir, "shap_tab")
        # Il write guard vale anche qui, ora che si scrive DENTRO la run: rieseguire su una
        # run che ha gia' `xai/shap_tab/` la sovrascriverebbe. Solo sulla destinazione di
        # default — con `out_dir` esplicito (la sonda, i confronti) la scrittura e' altrove
        # e non c'e' niente da proteggere.
        if save:
            from . import xai_driver as _xd
            _xd.assert_writable(run_dir, engines=("shap_tab",), allow_clobber=allow_clobber)
    # `subset_idx` deve arrivare a ENTRAMBI i passi: le attribuzioni si calcolano su queste
    # righe e la perturbazione deve girare sulle STESSE. Prima non era un parametro, quindi
    # passarlo solo in **kw dava attribuzioni sul prefisso e cancellazioni sul campione
    # stratificato -- disallineate, senza errore.
    if raw_df is None or imgs is None or labels is None:
        raw_df, imgs, labels = _build_faith_subset(n_samples or FAITH_N_SAMPLES,
                                                   subset_idx=subset_idx)
    if subset_idx is not None:
        kw["subset_idx"] = subset_idx
    elif n_samples is not None:
        kw["n_samples"] = n_samples
    if phi is None:
        phi, meta = aligned_tab_attributions(model, raw_df, imgs,
                                             background=background, verbose=verbose)
    else:
        meta = {"background": background, "signed": True, "score_space": "logit"}

    results = {"aligned_meta": meta}
    for mode in modes:
        key, n_eff = attribution_order_key(phi, labels=labels, mode=mode)
        out = explanation_faithfulness(
            model, run_dir, best_epoch, methods=("tabular",), tab_imp_override=key,
            out_name=f"{out_prefix}_{mode}", out_dir=out_dir, save=save, verbose=False, **kw)

        flip = -1.0 if mode in SIGN_FLIPPED_MODES else 1.0
        suffix = "_signflipped" if flip < 0 else ""
        fdir = out_dir or os.path.join(run_dir, "xai", "faithfulness")
        npz_path = os.path.join(fdir, f"{out_prefix}_{mode}_curves.npz")
        if mode in PERSAMPLE_SIGNED_MODES:
            # The per-sample curves the call just wrote are the only place the label-wise
            # sign can be applied before averaging.
            for direction in ("del", "ins"):
                g = _target_gain_persample(npz_path, labels, direction) if save else None
                if g is not None:
                    results[f"aligned_{mode}_{direction}_gain_labelsigned"] = g
            if not save:
                raise ValueError("mode='target' needs save=True: the label-wise sign is "
                                 "applied to the per-sample curves that call writes.")
        else:
            for direction in ("del", "ins"):
                k = f"faithfulness_tabular_{direction}_gain"
                if k in out:
                    results[f"aligned_{mode}_{direction}_gain{suffix}"] = flip * float(out[k])

        # Low-budget re-integration of the very curves just written.  Without it every
        # signed number is full-budget by construction, and full budget is the regime
        # REPORT sec.7.4 showed to be OOD-dominated.  Free: the curves are already on disk.
        if save and q_maxes:
            # Same sign-convention suffix as the full-budget keys above, so a reader never
            # has to wonder whether these are already sign-corrected.  They are.
            sfx = ("_labelsigned" if mode in PERSAMPLE_SIGNED_MODES else suffix)
            with np.load(npz_path) as npz:
                sgn = mode_sign(mode, labels)
                for q_max in q_maxes:
                    tag = f"q{q_max:.2f}".replace(".", "p")
                    width = gain_window(npz, q_max)
                    for direction in ("del", "ins"):
                        g = persample_gains(npz, "tabular", direction, q_max=q_max)
                        if g is None:
                            continue
                        v = float(np.mean(g * sgn))
                        results[f"aligned_{mode}_{direction}_gain{sfx}_{tag}"] = v
                        results[f"aligned_{mode}_{direction}_gainnorm{sfx}_{tag}"] = v / width
        results[f"aligned_{mode}_n_eff_mean"] = float(np.mean(n_eff))
        results[f"aligned_{mode}_n_eff_min"] = int(np.min(n_eff))
        if verbose:
            # The full-budget deletion key only: the low-budget ones added below share the
            # prefix, and a prefix search would report whichever landed first in the dict.
            key = f"aligned_{mode}_del_gain" + (
                "_labelsigned" if mode in PERSAMPLE_SIGNED_MODES else suffix)
            g = results.get(key)
            shown = f"{g:+.4f}" if g is not None else "n/a"
            print(f"  {mode:7s} {key.replace(f'aligned_{mode}_', ''):24s} = {shown}"
                  f"   (n_eff mean {np.mean(n_eff):.1f})", flush=True)
    if save:
        _write_aligned_signed_summary(out_dir, out_prefix, modes, labels)
    return results, phi


def _write_aligned_signed_summary(out_dir, out_prefix, modes, labels):
    """UNA riga per modo, in un CSV leggibile — il gemello di `shap_summary.csv`.

    Prima qui c'era una figura delle curve, per ogni run. Su 110 run erano 110 PNG che
    nessuno apre e 34 MB, e le curve di UNA run non si confrontano con niente: il confronto
    fra famiglie lo fa `xs.report()`. Quello che serve per run e' invece un punto d'ingresso
    leggibile senza aprire quattro JSON — che e' esattamente cio' che `xai/shap/` aveva e
    `xai/shap_tab/` no.

    I gain sono RICALCOLATI dalle curve con `mode_sign` + `persample_gains`: mai letti dallo
    scalare grezzo del JSON, che e' PRE-convenzione di segno.
    """
    import pandas as pd
    rows = []
    for mode in modes:
        p = os.path.join(out_dir, f"{out_prefix}_{mode}_curves.npz")
        if not os.path.exists(p):
            continue
        with np.load(p) as z:
            sgn = mode_sign(mode, labels)
            rec = {"mode": mode, "n": int(len(z["labels"])),
                   "pos_rate": float(np.mean(z["labels"]))}
            for direction in ("del", "ins"):
                g = persample_gains(z, "tabular", direction, q_max=None)
                rec[f"{direction}_gain"] = float(np.mean(g * sgn))
                gq = persample_gains(z, "tabular", direction, q_max=0.20)
                rec[f"{direction}_gain_q20norm"] = (float(np.mean(gq * sgn))
                                                    / gain_window(z, 0.20))
            rows.append(rec)
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "faithfulness_summary.csv"), index=False)


def check_efficiency(model, raw_df, imgs, phi, tol=0.05):
    """Permutation efficiency: does sum_i phi_i equal logit(p(x)) - logit(p(baseline))?

    This is THE test that we explain the function the perturbation test evaluates.  It
    only holds with a single-row masker and an f() that returns the logit -- in
    probability space it fails while everything is in fact correct.
    """
    global _FAITH_TAB_BASELINE
    if _FAITH_TAB_BASELINE is None:
        _FAITH_TAB_BASELINE = _build_tabular_baseline()
    ctx = _CTX
    feat_cols = list(ctx.feat_cols)

    base_df = pd.DataFrame([_FAITH_TAB_BASELINE] * len(raw_df))[feat_cols].reset_index(drop=True)
    s_x = _score_logit(model, _encode_raw(raw_df[feat_cols]), imgs)
    s_b = _score_logit(model, _encode_raw(base_df), imgs)

    lhs, rhs = phi.sum(axis=1), (s_x - s_b)
    err = np.abs(lhs - rhs)
    return {"max_abs_err": float(err.max()), "mean_abs_err": float(err.mean()),
            "corr": float(np.corrcoef(lhs, rhs)[0, 1]), "passes": bool(err.max() <= tol)}
