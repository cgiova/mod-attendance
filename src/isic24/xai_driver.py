"""Orchestration for the XAI engines -- the part that used to live in the training
notebook's shared runner (cell 36).

The key fact that made this move possible: the XAI callbacks were **never passed to
``model.fit``**.  ``run_experiment`` always reloaded the best checkpoint afterwards and
drove the Keras hooks by hand.  So this was already a post-hoc driver sitting in the
training notebook; ``run_posthoc_xai`` is the same code with the checkpoint discovery
that the training loop used to supply implicitly.

Training keeps the ability to call it at the end of ``fit``, behind a knob that now
defaults to OFF (``ENABLE_INTRAINING_XAI``).
"""

from __future__ import annotations

import glob
import json
import os

import keras

from . import paths as _paths
from . import xai_faithfulness as xf
from .xai_engines import (
    MultiModalGradCAMCallback,
    MultiModalGradCAMEngine,
    ShapCallback,
)

# --------------------------------------------------------------------------------------
# The load-bearing kwargs.
#
# These are the values `make_xai_callbacks` passes, NOT the engines' constructor defaults.
# Every published number was produced with the values below; changing one silently makes
# the artifacts non-comparable.
# --------------------------------------------------------------------------------------
SHAP_KWARGS = dict(
    background_size=64, max_explain_samples=128,
    top_k=10, positive_threshold=0.5, every_n_epochs=1,
    max_samples=128, max_batches=5, rng_seed=0,
)
GRADCAM_KWARGS = dict(use_guided_grads=True, mode="image_only")
GRADCAM_CB_KWARGS = dict(class_index=0)
# The TWO Grad-CAM modes, run within the same model load (2026-08-01).
#   `image_only`  zeroes the tabular features -> "where it would look WITHOUT the table"
#   `context`     uses the real features      -> "where it actually looks"
# Until today only `image_only` ran: with the tabular branch off, i.e. with the fusion
# disabled. For an attendance study that is precisely the configuration that does not answer
# the question ("does the modality weight move visual attention?"), and the difference
# between the two modes (`cam_iou_modes`) is itself a measure of how much the table moves it.
# None = a single mode, the one in GRADCAM_KWARGS (historical behaviour).
GRADCAM_MODES = ("image_only", "context")
# Images per gradient step. With all 144 at once the GPU runs out of memory: guided
# gradients keep the activations of the WHOLE network for every image (measured 01/08:
# ~22 GiB on a (144, 8, 8, 1248) tensor, OOM on a 24 GB GPU). Grad-CAM is per sample, so
# splitting the batch gives the same result at constant memory. Raise it when the GPU is
# free, lower it if it still OOMs.
GRADCAM_BATCH = 16
# --------------------------------------------------------------------------------------
# `make_xai_callbacks` builds `LoreSACallback` + `LoreEngineSA` with LORE_SA_KWARGS, and
# there is no other LORE engine. The one that ran on `xailib` was the default until
# 2026-07-29 -- its `random` neighbourhood produces explanations of little use, 1.7%
# opposite-class evidence -- and was removed once the runs it produced had been archived.
# The two did NOT explain the same function: the old one shared one image embedding across
# samples (`images.mean(axis=0)`), this one takes one per sample. So the `lore_fidelity`
# column of the June registry stays an artifact of that path and is never compared row by
# row with these numbers.
#
# WHICH NUMBER TO REPORT. Not `fidelity`: it is `dt.score(Z, Yb)` in-sample, 1.0 by
# construction on an unpruned tree. Report:
#   `fid_cv_bacc`               fidelity on held-out folds (the only discriminating one)
#   `lore_sa_coverage`          fraction of samples with a NON-degenerate neighbourhood
#   `neigh_unique_frac`, `neigh_pos_rate`   without which the fidelity cannot be read
# Deletion/insertion do not belong here: they are the metric of tabular SHAP and Grad-CAM,
# not of LORE, which is a rule surrogate measured by agreement with the black box.
# --------------------------------------------------------------------------------------

# Configuration chosen 2026-07-29 by measuring on vanilla seed_47 (n=2 samples):
# `mutpb=.5` + `init_mutpb=.3` gives 2.9x upstream's diversity, the longest rules (3.0) and
# 4x the opposite-class evidence of `mutpb=.8`, at +19% cost. Details and the full table:
# `ISIC24_REPORT.md`, LORE generator configuration.
LORE_SA_KWARGS = dict(
    max_explain_samples=32, every_n_epochs=1, max_samples=256, max_batches=2, rng_seed=0,
    neigh_type="genetic", num_instances=1000, mutpb=0.5, init_mutpb=0.3, threshold=0.5,
)

# Where LORE runs. Its hot loop is the dense HEAD (~1000 evaluations per instance) plus the
# genetic algorithm and the tree, which are pure CPU; the CNN runs once per run. So it keeps
# the GPU busy 23 min/run while barely using it: on a shared machine, 42 hours of idle GPU.
#
# `"/CPU:0"` moves LORE to the CPU **while leaving the GPU visible and free** for the other
# engines: `run_posthoc_xai` passes the device to `load_run_model`, so the variables are BORN
# on the CPU. That is the only point at which the choice takes effect.
#
# ⚠️ A `tf.device` around the computation alone (the one in `xai_lore_sa.on_epoch_end`) is
# NOT enough and fails on its own — tried 02/08, ~20 s per run:
#     InvalidArgumentError: Trying to access resource stem_conv/kernel/... located in
#     device /...GPU:0 from device /...CPU:0
# because `tf.device` does not move variables already allocated, and Keras 3's `predict` is
# XLA-compiled, which refuses the remote variable instead of copying it.
#
# Cost MEASURED 02/08 over 6 runs: **20.0 min/run on CPU** against 22 min on GPU. The CPU
# costs nothing and is in fact slightly faster — consistent with the GPU only ever serving
# the CNN. So `"/CPU:0"` is the right choice for the long pass, not a fallback.
LORE_SA_DEVICE = None

ALL_ENGINES = ("shap", "gradcam", "lore")


# `xai_status.json` records outcomes keyed by CALLBACK CLASS NAME, not by engine name.
# This maps back, so the "already complete" shortcut can ask about the engines actually
# requested instead of about whatever happens to be in the file.
#
# The bug this closes (found 2026-07-30 in dry-run): the 20 `learnable_scaling` runs whose
# SHAP arrays were filled in post-hoc on 29/07 carry `{"ShapCallback": "ok"}`.  The old test
# was `all(v == "ok" for v in prev["engines"].values())` -- vacuously true for any engine
# NOT in the file -- so `engines=("lore",)` skipped all 20 and reported them complete.  A
# full LORE pass over the 110 would have silently produced 90.
def _status_key_for(engine):
    if engine == "shap":
        return "ShapCallback"
    if engine == "gradcam":
        return "MultiModalGradCAMCallback"
    if engine == "lore":
        return "LoreSACallback"
    return engine


# --------------------------------------------------------------------------------------
# Canonical-artifact write guard  (added 2026-07-29: canonical runs that have already
# produced their artifacts must never be overwritten)
#
# The trap this closes: `run_posthoc_xai` skips only when `xai/xai_status.json` says every
# engine is "ok" -- and that file is written by THIS driver, so none of the 110 canonical
# runs has one.  They were produced by the old in-training path.  Pointed at a canonical
# run dir, the driver therefore sails past the skip check even with `overwrite=False` and
# rewrites `xai/shap/epoch_*`, `xai/gradcam/epoch_*.png`, `xai/lore/epoch*` in place.
#
# So the guard keys on the ARTIFACTS, not on the status file, and it is PER ENGINE: it
# refuses only for the engines whose own outputs are already there.  Filling in what a run
# is missing stays allowed -- that is additive, and it is what `run_posthoc_xai_batch` is
# for (e.g. ~20 runs have `faithfulness.json` but no SHAP arrays).
#
# Scope: this covers the paths THIS driver writes.  `xai/faithfulness/` is written by
# `xai_faithfulness.explanation_faithfulness`, not from here, and is governed by the
# post-hoc notebook's own `FAITH_OVERWRITE` knob (its loop reuses an existing
# `faithfulness.json` unless you flip it).
# --------------------------------------------------------------------------------------
CANONICAL_WRITE_GUARD = True

# Everything each engine's callback writes under `run_dir/xai/`, per engine.
def shap_subdir():
    """`shap` or `shap_tab`, according to whether an XAI sample is adopted.

    ONE place decides where SHAP lives, because three pieces have to agree: the writer
    (`make_xai_callbacks`), the protector (`_CLOBBERABLE`) and the reader
    (`xai_faithfulness._load_shap_values`). Should they diverge, the guard defends an empty
    folder while the reader finds another one — and June's canonical npz, which are on the
    PREFIX and unstamped, go back to being the only ones found.

    `xai/shap/`      canonical artifacts, prefix, never rewritten
    `xai/shap_tab/`  everything computed on the adopted sample: SHAP arrays + signed
                     faithfulness. One folder, one sample.
    """
    return "shap_tab" if SHAP_KWARGS.get("subset_fingerprint") else "shap"


def gradcam_subdir():
    """`gradcam` or `gradcam_img`, according to whether an XAI sample is adopted.

    Same logic as `shap_subdir`. `xai/gradcam/` holds the 106 historical PNGs: EIGHT images
    taken off the head of `validation_ds`, with no arrays and no trace of which they were.
    `xai/gradcam_img/` is the image side redone on the distinct-lesion sample, with the RAW
    heatmaps saved — which are the precondition of any metric on the image.
    """
    return "gradcam_img" if SHAP_KWARGS.get("subset_fingerprint") else "gradcam"


_CLOBBERABLE = {
    # depends on the adopted sample, as `lore` depends on the engine: see `_patterns_for`
    "shap": {"shap":     ("xai/shap/epoch_*", "xai/shap/*.csv"),
             "shap_tab": ("xai/shap_tab/epoch_*", "xai/shap_tab/shap_summary.csv")},
    "gradcam": {"gradcam":     ("xai/gradcam/epoch_*",),
                "gradcam_img": ("xai/gradcam_img/gradcam_arrays*_epoch_*.npz",
                                "xai/gradcam_img/gradcam_summary.csv")},
    # LORE-SA has a folder of its own since 2026-08-01. Whatever the removed engine left
    # under `xai/lore/` is historical and no new write can land on it.
    "lore": ("xai/lore_sa/epoch*", "xai/lore_sa/lore_sa_per_epoch.csv"),
    # Not engines of `make_xai_callbacks`: `xai_faithfulness` writes both. They are here
    # because `existing_xai_artifacts` is the only place that knows what a run has already
    # produced, and the faithfulness loop asks it for the same reason the others do.
    "shap_tab": ("xai/shap_tab/faithfulness_aligned_signed_*.json",),
    "faithfulness": ("xai/faithfulness/*.json", "xai/faithfulness/*_curves.npz"),
}


def _patterns_for(engine):
    pats = _CLOBBERABLE.get(engine, ())
    if isinstance(pats, dict):
        if engine == "shap":            # they depend on the adopted sample
            return pats.get(shap_subdir(), ())
        return pats.get(gradcam_subdir(), ())
    return pats


def _is_artifact(path):
    """An EMPTY directory is not an artifact.

    The callbacks create `epoch<NNN>/<engine>/` before computing, so an interrupted run
    (Ctrl-C, OOM, dead kernel) leaves an empty folder. Counting it as an artifact, the write
    guard blocked forever the re-execution of a run that had produced nothing, and the only
    way out would have been `allow_clobber=True` -- that is, disarming the protection exactly
    where it is needed. Found on 2026-07-30 on `fixed_scaling/a1p000`, blocked by a
    `tabular_sa/` emptied 30 seconds earlier.
    """
    return bool(os.listdir(path)) if os.path.isdir(path) else True


def existing_xai_artifacts(run_dir, engines=ALL_ENGINES):
    """Outputs of `engines` already on disk -- exactly what a re-run would replace."""
    hits = []
    for eng in engines:
        for pat in _patterns_for(eng):
            hits += [p for p in glob.glob(os.path.join(run_dir, pat)) if _is_artifact(p)]
    return sorted(hits)


def assert_writable(run_dir, engines=ALL_ENGINES, allow_clobber=False):
    """Refuse to write an engine's artifacts over a run that already has them."""
    if allow_clobber or not CANONICAL_WRITE_GUARD:
        return []
    hits = existing_xai_artifacts(run_dir, engines)
    if hits:
        busy = sorted(e for e in engines if existing_xai_artifacts(run_dir, (e,)))
        free = sorted(set(engines) - set(busy))
        shown = "\n    ".join(os.path.relpath(h, run_dir) for h in hits[:6])
        more = f"\n    ... and {len(hits) - 6} more" if len(hits) > 6 else ""
        raise RuntimeError(
            f"[XAI WRITE GUARD] {run_dir}\n  already holds {len(hits)} artifacts on disk for "
            f"{busy}:\n    {shown}{more}\n"
            f"  Re-running here OVERWRITES them. `overwrite=False` does not protect you: it "
            f"skips only when xai/xai_status.json exists, which the in-training path never "
            f"wrote.\n"
            + (f"  Engines {free} have no artifacts here: engines={tuple(free)} adds them "
               f"without touching anything.\n" if free else "")
            + f"  **overwrite=True is NOT enough to pass**: it says 'recompute', not 'you may "
              f"destroy'. The only consent is allow_clobber=True (or "
              f"xai_driver.CANONICAL_WRITE_GUARD = False).\n"
              f"  Otherwise copy config.json + checkpoints into a temporary directory and point "
              f"the driver there.")
    return hits


# --------------------------------------------------------------------------------------
# Probe batch
# --------------------------------------------------------------------------------------

def make_probe_batch(n=8, subset_idx=None):
    """The validation images Grad-CAM runs on.

    ``subset_idx`` -- positions in ``validation_df``: this is how Grad-CAM explains the SAME
    lesions as SHAP and LORE.  Without it, the historical behaviour returns: the first ``n``
    rows of ``validation_ds`` (built with ``shuffle=False``), that is **eight** images of the
    prefix.  Eight samples are not enough for any of the image-side metrics, and that is why
    the image side was redone on 2026-08-01.

    Used to be a module-level global mutated on first use and cached across every run in
    the study; it is now an explicit argument so a caller can score a different model on
    a different probe.
    """
    ctx = xf._CTX
    if subset_idx is None:
        xb, yb = next(iter(ctx.validation_ds))
        return {k: v[:n] for k, v in xb.items()}, yb[:n]

    # Read the dataset up to the highest position asked for, then index: the same strategy as
    # `LoreSACallback._freeze_batch`, and it does not assume the sample is a prefix -- which is
    # exactly the assumption that has already deceived us twice.
    import numpy as _np
    import tensorflow as _tf
    idx = _np.asarray(subset_idx, int)
    need = int(idx.max()) + 1
    xs_, ys_, seen = [], [], 0
    for xb, yb in ctx.validation_ds:
        xs_.append(xb); ys_.append(yb)
        seen += int(_tf.shape(yb)[0])
        if seen >= need:
            break
    x_all = {k: _tf.concat([b[k] for b in xs_], axis=0) for k in xs_[0]}
    y_all = _tf.concat(ys_, axis=0)
    return ({k: _tf.gather(v, idx) for k, v in x_all.items()}, _tf.gather(y_all, idx))


# --------------------------------------------------------------------------------------
# Callback construction and driving
# --------------------------------------------------------------------------------------

def make_xai_callbacks(run_dir, model_name, probe_batch=None, engines=ALL_ENGINES,
                       xai_dir=None):
    """Build the SHAP / Grad-CAM / LORE callbacks for one run directory.

    ``xai_dir`` redirects ALL writing away from ``run_dir/xai``: this is how the stratified
    sample produces artifacts without touching the canonical runs (and without needing
    ``allow_clobber``, which would disarm the write guard exactly where it is needed).
    """
    ctx = xf._CTX
    xai_dir = xai_dir or os.path.join(run_dir, "xai")

    out = []
    if "shap" in engines:
        d = os.path.join(xai_dir, shap_subdir())
        os.makedirs(d, exist_ok=True)
        out.append(ShapCallback(
            data_source=ctx.validation_ds, validation_df=ctx.validation_df,
            feat_cols=ctx.feat_cols, encoded_feature_names=ctx.encoded_feature_names,
            base_dir=d, **SHAP_KWARGS))
    if "gradcam" in engines:
        d = os.path.join(xai_dir, gradcam_subdir())
        os.makedirs(d, exist_ok=True)
        # The probe batch follows the adopted sample: with `subset_idx` Grad-CAM explains the
        # SAME lesions as SHAP and LORE, and the heatmaps become realignable with the other two
        # explanations. Without it, the historical behaviour stands (the first 8 rows).
        _sub = SHAP_KWARGS.get("subset_idx")
        if probe_batch is None:
            probe_batch = make_probe_batch(subset_idx=_sub)
        out.append(MultiModalGradCAMCallback(
            engine=MultiModalGradCAMEngine(**GRADCAM_KWARGS),
            probe_batch=probe_batch, save_dir=d,
            subset_idx=_sub, subset_fingerprint=SHAP_KWARGS.get("subset_fingerprint"),
            modes=(GRADCAM_MODES if _sub is not None else None),
            batch_size=GRADCAM_BATCH,
            tag=f"{model_name}/GradCAM", **GRADCAM_CB_KWARGS))
    if "lore" in engines:
        from .xai_lore_sa import _lore_sa_callback_cls
        d = os.path.join(xai_dir, "lore_sa")
        os.makedirs(d, exist_ok=True)
        out.append(_lore_sa_callback_cls()(
            data_source=ctx.validation_ds, raw_feature_df=ctx.validation_df,
            feat_cols=ctx.feat_cols, feature_space=ctx.feature_space,
            to_feature_dict=ctx.to_feature_dict, training_df=ctx.training_df,
            base_dir=d, **LORE_SA_KWARGS))
    return out


def run_xai_on_best_epoch(best_model, xai_callbacks, best_epoch, best_val_auc, run_dir=None):
    """Drive the Keras hooks by hand against an already-trained model.

    Returns ``{EngineName: "ok" | "failed: ..."}``.  The previous version returned a
    single ``any_done`` boolean that was ``True`` when just one of the three engines had
    succeeded, printing the other two failures as warnings -- a batch loop over 110 runs
    then looked busy while producing partial artifacts.  That footgun cost a silent
    partial once already (``seed_47__a0p125``), so the per-engine outcome is returned and,
    when ``run_dir`` is given, persisted to ``xai/xai_status.json``.
    """
    logs = {"val_auc": float(best_val_auc)}
    status = {}
    for cb in xai_callbacks:
        name = cb.__class__.__name__
        try:
            if hasattr(cb, "every_n_epochs"):
                cb.every_n_epochs = 1
            cb.set_model(best_model)
            cb.set_params({"epochs": int(best_epoch + 1), "steps": None, "verbose": 0})
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin(logs={})
            cb.on_epoch_end(int(best_epoch), logs=logs)
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(logs={})
            status[name] = "ok"
        except Exception as e:
            status[name] = f"failed: {type(e).__name__}: {e}"
            print(f"[XAI WARN] {name} failed at epoch {best_epoch}: {type(e).__name__}: {e}")

    if run_dir:
        os.makedirs(os.path.join(run_dir, "xai"), exist_ok=True)
        p = os.path.join(run_dir, "xai", "xai_status.json")
        # MERGE, don't replace: running one engine on a run that already recorded another
        # used to wipe the older entry, and the skip shortcut reads this file -- so the
        # engine that got erased would silently be re-run later, or (worse) be reported as
        # never done. Adding an engine is additive on disk; the ledger must match.
        merged = {}
        if os.path.exists(p):
            try:
                merged = json.load(open(p)).get("engines", {})
            except (ValueError, OSError):
                merged = {}
        merged.update(status)
        with open(p, "w") as f:
            json.dump({"best_epoch": int(best_epoch), "engines": merged}, f, indent=2)
    return status


# --------------------------------------------------------------------------------------
# Post-hoc entry points
# --------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------
# FeatureSpace binding -- this checkpoint against the space this session encodes with.
#
# Nothing here knows a dataset. The identities are read off the artifacts in play: the run
# records the space it was trained with (`config.json`), the space reports its own width and
# adapt state, and the session declares the space it loaded. A reference committed to the
# repository would have to carry a width, and a width belongs to one dataset only.
# --------------------------------------------------------------------------------------
def feature_space_fingerprint(fs):
    """(encoded width, sha256) of a FeatureSpace's ADAPT STATE.

    Feature order, encoded width, categorical vocabularies and every adapt weight. The
    width alone cannot identify a space: the adapt is not deterministic, so two spaces of
    equal width and different statistics encode the same row differently.
    """
    import hashlib

    import numpy as np

    h = hashlib.sha256()
    order = list(fs.features.keys())
    dim = int(fs.get_encoded_features().shape[-1])
    h.update(("ORDER:" + "|".join(order)).encode())
    h.update(f"DIM:{dim}".encode())
    for name in order:                       # vocabularies are order-sensitive
        pp = fs.preprocessors.get(name)
        if hasattr(pp, "get_vocabulary"):
            h.update(f"VOCAB:{name}:".encode())
            h.update("|".join(map(str, pp.get_vocabulary())).encode())
    for w in fs.get_weights():               # normalisation statistics
        w = np.asarray(w)
        h.update(str(w.shape).encode())
        h.update(str(w.dtype).encode())
        h.update(w.tobytes())
    return dim, h.hexdigest()


def _model_feature_dim(model):
    """The tabular input width of a checkpoint, read off the model."""
    for t in getattr(model, "inputs", []):
        if "features" in (getattr(t, "name", "") or "") and len(t.shape) == 2:
            return int(t.shape[-1])
    rank2 = [t for t in getattr(model, "inputs", []) if len(t.shape) == 2]
    if len(rank2) == 1:                      # the image input is rank 4
        return int(rank2[0].shape[-1])
    raise RuntimeError(f"could not locate the tabular input of {model.name}")


def assert_fs_matches_model(model, fs, run_id=None):
    """The last resort: the checkpoint's tabular width against the live encoded width."""
    mdim, fdim = _model_feature_dim(model), int(fs.get_encoded_features().shape[-1])
    if mdim != fdim:
        raise RuntimeError(
            f"[FS GUARD] {run_id or '<run>'}: the checkpoint expects {mdim} encoded features "
            f"and this FeatureSpace emits {fdim}. It was not trained with this space.")
    return mdim


def assert_run_feature_space(cfg, run_dir, declared_file, declared_fp=None):
    """Refuse a checkpoint trained against another FeatureSpace, BEFORE loading it.

    Returns the level the check reached -- ``"fingerprint"``, ``"file"`` or ``"width"`` --
    so a caller can say what was verified instead of implying more than it checked. A run
    that records no space falls back to the width, which is all its config affords.
    """
    run_file = cfg.get("feature_space_file")
    if run_file is None:
        return "width"
    if run_file != declared_file:
        raise RuntimeError(
            f"[FS GUARD] {run_dir}\n  was trained against {run_file}, and this session "
            f"encodes with {declared_file}.\n  Every number the run produces here would be "
            f"computed on features it has never seen.")
    run_fp = cfg.get("feature_space_fingerprint")
    if run_fp is None or declared_fp is None:
        return "file"
    if run_fp != declared_fp:
        raise RuntimeError(
            f"[FS GUARD] {run_dir}\n  records the same path but another adapt state:\n"
            f"  run  {run_fp}\n  live {declared_fp}\n  Same width, different statistics: "
            f"the encoding differs row by row and nothing else would report it.")
    return "fingerprint"


def load_run_model(run_dir, device=None):
    """Reload a run's best checkpoint together with its recorded best epoch.

    ``safe_mode=False`` plus the five custom layers being registered (import
    ``isic24.mod_attendance`` anywhere in the session) is what lets the 70/110
    fixed_scaling / learnable_scaling / controlled_fusion / adaptive_gating checkpoints
    deserialize at all.

    ``device`` (e.g. ``"/CPU:0"``) is the device the VARIABLES are created on. It is the
    difference between running an engine on the CPU and not: a `tf.device` around the
    computation alone does not move weights already allocated on the GPU, and Keras 3's
    `predict` (XLA) dies instead of copying them. Around the LOADING, weights are born where
    they are wanted and from there on the whole graph is coherent. See `LORE_SA_DEVICE`.
    """
    ck = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "best_*.keras")))
    if not ck:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/checkpoints/")

    cfg_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    # Before the checkpoint reaches memory: a run paired with another FeatureSpace costs
    # nothing to refuse here and produces silently wrong numbers if it is loaded.
    ctx = xf._CTX
    assert_run_feature_space(cfg, run_dir, ctx.feature_space_file,
                             ctx.feature_space_fingerprint)

    import contextlib
    import tensorflow as tf
    with (tf.device(device) if device else contextlib.nullcontext()):
        model = keras.models.load_model(ck[0], compile=False, safe_mode=False)
    assert_fs_matches_model(model, ctx.feature_space, cfg.get("run_id", run_dir))
    return model, cfg


def run_posthoc_xai(run_dir, engines=ALL_ENGINES, best_epoch=None, best_val_auc=None,
                    probe_batch=None, overwrite=False, dry_run=False, verbose=True,
                    allow_clobber=False, xai_dir=None):
    """Run the XAI engines against a saved checkpoint, long after it was trained.

    This is the same pipeline the training notebook ran at the end of ``fit`` -- same
    engines, same kwargs, same hook sequence -- with the checkpoint loaded from disk
    instead of handed over in memory.

    ``overwrite`` and ``allow_clobber`` are deliberately NOT the same knob:

    * ``overwrite=True`` means "don't skip, recompute" -- it disables the
      ``xai_status.json`` shortcut.
    * ``allow_clobber=True`` means "you may destroy artifacts that are already there" --
      it disables :func:`assert_writable`.

    So ``overwrite=True`` on its own still raises when artifacts exist.  That is the point:
    recomputing into an empty run dir is routine, while rewriting a run that already
    produced its artifacts is not, and the two should not share a spelling.
    """
    model_cfg = {}
    if best_epoch is None or best_val_auc is None:
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"{cfg_path} missing: pass best_epoch/best_val_auc explicitly")
        model_cfg = json.load(open(cfg_path))
        best_epoch = model_cfg["best_epoch"] if best_epoch is None else best_epoch
        best_val_auc = model_cfg.get("best_val_auc", 0.0) if best_val_auc is None else best_val_auc

    # An external `xai_dir` writes nothing inside run_dir: neither the status file nor the
    # write guard has jurisdiction there, and resuming is the caller's job (`xai_subset.is_done`).
    external = xai_dir is not None
    status_path = os.path.join(xai_dir or os.path.join(run_dir, "xai"), "xai_status.json")
    if not overwrite and os.path.exists(status_path):
        prev = json.load(open(status_path)).get("engines", {})
        # Skip only if EVERY requested engine is recorded ok -- see `_status_key_for`.
        if all(prev.get(_status_key_for(e)) == "ok" for e in engines):
            if verbose:
                print(f"[skip] {run_dir}: {list(engines)} already done (overwrite=False)")
            return prev

    if dry_run:
        if verbose:
            n = len(existing_xai_artifacts(run_dir, engines))
            blocked = "" if (allow_clobber or not n) else \
                f" -- BUT the write guard would block it: {n} XAI artifacts already present"
            print(f"[dry-run] would run {engines} on {run_dir} @ epoch {best_epoch}{blocked}")
        return {e: "dry-run" for e in engines}

    if not external:
        assert_writable(run_dir, engines=engines, allow_clobber=allow_clobber)

    # WHERE the weights are born. With `LORE_SA_DEVICE="/CPU:0"` the checkpoint is loaded with
    # its variables on the CPU, and from there on LORE runs entirely on the CPU while the GPU
    # stays visible and free for the other engines. A `tf.device` around `explain` alone
    # (`xai_lore_sa`) is not enough and is not an alternative: that was the mistake of 02/08.
    # It holds for LORE ALONE: loading a model on the CPU for SHAP or Grad-CAM would make them
    # silently very slow, so the combination is an error rather than a surprise.
    _dev = LORE_SA_DEVICE if "lore" in engines else None
    if _dev and tuple(engines) != ("lore",):
        raise RuntimeError(
            f"LORE_SA_DEVICE={LORE_SA_DEVICE!r} holds for the WHOLE model: with "
            f"engines={tuple(engines)} the other engines would land on it too.\n"
            "  Run LORE on its own (engines=('lore',)), or LORE_SA_DEVICE=None.")

    model, cfg = load_run_model(run_dir, device=_dev)
    model_name = (model_cfg or cfg).get("model_name", os.path.basename(run_dir))
    cbs = make_xai_callbacks(run_dir, model_name, probe_batch=probe_batch, engines=engines,
                             xai_dir=xai_dir)
    return run_xai_on_best_epoch(model, cbs, best_epoch, best_val_auc,
                                 run_dir=(xai_dir and os.path.dirname(xai_dir)) or run_dir)


def run_posthoc_xai_batch(art_path, engines=ALL_ENGINES, only_impl=None, max_runs=None,
                          overwrite=False, dry_run=False, verbose=True, allow_clobber=False):
    """Apply :func:`run_posthoc_xai` across the canonical run registry.

    Deliberately opt-in: nothing here runs unless called, and ``overwrite=False`` means a
    re-invocation resumes rather than redoing work.  Runs that already carry XAI artifacts
    are refused by :func:`assert_writable` and reported as ``failed: RuntimeError`` rather
    than silently rewritten; the loop moves on to the next run.
    """
    import gc

    import pandas as pd

    master = pd.read_csv(os.path.join(art_path, "summaries", "master_results_rows.csv"))
    if only_impl:
        master = master[master.impl_name.isin(list(only_impl))]
    if max_runs:
        master = master.head(int(max_runs))

    results = {}
    for i, r in master.reset_index(drop=True).iterrows():
        run_dir = _paths.run_dir_abs(r.run_dir, art_path)
        try:
            results[r.run_id] = run_posthoc_xai(
                run_dir, engines=engines, best_epoch=int(r.best_epoch),
                best_val_auc=float(r.best_val_auc), overwrite=overwrite,
                dry_run=dry_run, verbose=verbose, allow_clobber=allow_clobber)
        except Exception as e:
            results[r.run_id] = {"_run": f"failed: {type(e).__name__}: {e}"}
            print(f"[FAIL] {r.run_id}: {type(e).__name__}: {e}")
        finally:
            if not dry_run:
                keras.backend.clear_session()
                gc.collect()
        if verbose:
            print(f"[{i + 1}/{len(master)}] {r.run_id}: {results[r.run_id]}", flush=True)
    return results
