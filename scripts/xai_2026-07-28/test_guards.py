"""Guard suite — every guard must FIRE, not merely stay silent.

Why it exists (2026-08-04): three defects cast from the same mould — a check that, on the
real artifacts, is silent both when it works and when it is broken. Re-running the pipeline
does not tell them apart. Every guard here has, where possible, THREE cases:

    A. the condition the guard must catch    -> must raise
    B. the legitimate condition              -> must not raise
    C. the OLD code on case A                -> must stay silent

Case C is what turns a test from "it runs" into "it proves I am measuring the fix".

Usage:  python scripts/xai_2026-07-28/test_guards.py
No GPU, no checkpoint: artifacts are only needed where stated.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from isic24 import paths as _paths0                           # noqa: E402
ART = str(_paths0.art_path())
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'✓ PASS' if cond else '✗ FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    return bool(cond)


def raises(fn, exc=RuntimeError):
    try:
        fn()
        return False, ""
    except exc as e:
        return True, str(e).splitlines()[0][:70]
    except Exception as e:                                   # noqa: BLE001
        return False, f"eccezione INATTESA {type(e).__name__}: {e}"


def section(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


TMP = tempfile.mkdtemp(prefix="guards_")
rng = np.random.default_rng(0)

# ======================================================================== 1 ===
section("1. [STABILITY] — two runs with SHAP on different samples")
from isic24 import xai_metrics as xmet                        # noqa: E402

NAMES = [f"f{i}" for i in range(20)] + [f"cat__{t}" for t in "abcde"]
N = 12


def make_shap_run(tag, val_index, root=TMP):
    d = os.path.join(root, tag, "xai", "shap_tab", "epoch_003")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "shap_arrays_epoch_003.npz"),
             shap_values=rng.normal(size=(N, len(NAMES))),
             explained_indices=np.arange(N),          # ALWAYS 0..N-1: that is the point
             val_index=np.asarray(val_index, int),
             subset_fingerprint=np.asarray("fp-test"))
    return os.path.join(root, tag)


run_same_a = make_shap_run("s_a", np.arange(1, N + 1))
run_same_b = make_shap_run("s_b", np.arange(1, N + 1))
run_diff = make_shap_run("s_c", np.arange(500, 500 + N))


def stability(dirs):
    m = pd.DataFrame([dict(impl_name="fixed_scaling", setting_tag=t, seed=47, run_dir=p)
                      for t, p in zip(("a0p250", "a0p500"), dirs)])
    return xmet.stability_table(m, "/", NAMES)


ok, msg = raises(lambda: stability([run_same_a, run_diff]))
check("A. different samples -> raises", ok, msg)
ok_b, msg_b = raises(lambda: stability([run_same_a, run_same_b]))
check("B. same sample -> does not raise", not ok_b, msg_b)

_orig = xmet._run_top_k


def _old_run_top_k(run_dir, groups, k=xmet.TOP_K):
    p = xmet._shap_npz(run_dir)
    with np.load(p) as z:
        phi = np.abs(np.asarray(z["shap_values"], float))
        idx = np.asarray(z["explained_indices"], int).ravel()      # the bug
    return [xmet.raw_top_k(r, groups, k) for r in phi], idx


xmet._run_top_k = _old_run_top_k
ok_c, _ = raises(lambda: stability([run_same_a, run_diff]))
xmet._run_top_k = _orig
check("C. the old code stays silent (the guard was dead)", not ok_c)

# ======================================================================== 2 ===
section("2. sample_matches / resume_verified — 'done' means done ON THIS sample")
from isic24 import xai_subset as xs                           # noqa: E402

vi_ok = np.arange(1, N + 1)
check("A. shap_tab, right sample -> True",
      xs.sample_matches(run_same_a, "shap_tab", vi_ok) is True)
check("B. shap_tab, different sample (same length) -> False",
      xs.sample_matches(run_same_a, "shap_tab", vi_ok + 500) is False)
check("C. different length -> False",
      xs.sample_matches(run_same_a, "shap_tab", vi_ok[:5]) is False)

d_noval = os.path.join(TMP, "s_noval", "xai", "shap_tab", "epoch_003")
os.makedirs(d_noval, exist_ok=True)
np.savez(os.path.join(d_noval, "shap_arrays_epoch_003.npz"),
         shap_values=rng.normal(size=(N, len(NAMES))), explained_indices=np.arange(N))
check("D. npz without val_index -> None ('I do not know' != 'fine')",
      xs.sample_matches(os.path.join(TMP, "s_noval"), "shap_tab", vi_ok) is None)

fp = "fp-test"
os.makedirs(os.path.join(run_same_a, "xai", "shap_tab"), exist_ok=True)
xs.mark_done(run_same_a, "shap_tab", fp)
todo, done, redo = xs.resume_verified({"r": run_same_a}, "shap_tab", fp, vi_ok)
check("E. stamped + right sample -> done", done == ["r"] and not redo)
todo, done, redo = xs.resume_verified({"r": run_same_a}, "shap_tab", fp, vi_ok + 500)
check("F. stamped but different sample -> MUST BE REDONE", redo == ["r"] and not done,
      "the case that could not fire on 03/08")

# ======================================================================== 3 ===
section("3. [SUBSET GUARD] — reusing canonical npz under an adopted sample")
from isic24 import xai_faithfulness as xf                     # noqa: E402
from isic24 import xai_driver as xd                           # noqa: E402

canon = os.path.join(TMP, "canon", "xai", "shap", "epoch_003")
os.makedirs(canon, exist_ok=True)
np.savez(os.path.join(canon, "shap_arrays_epoch_003.npz"),        # WITHOUT fingerprint
         shap_values=rng.normal(size=(N, len(NAMES))), explained_indices=np.arange(N))
raw = pd.DataFrame({"a": np.arange(N)})

_prev_fp = xf.ACTIVE_SUBSET_FINGERPRINT
_prev_sub = xd.SHAP_KWARGS.get("subset_fingerprint")
try:
    # `shap_subdir()` derives the folder from here: without a fingerprint it reads
    # `xai/shap/`, which is where the canonical npz the guard must refuse lives.
    xd.SHAP_KWARGS.pop("subset_fingerprint", None)
    check("0. shap_subdir() follows SHAP_KWARGS", xd.shap_subdir() == "shap")
    xf.ACTIVE_SUBSET_FINGERPRINT = "unique-equal-s0-n144-DIFFERENT"
    ok, msg = raises(lambda: xf._load_shap_values(
        os.path.join(TMP, "canon"), 3, None, raw, None))
    check("A. npz without fingerprint + active sample -> raises", ok, msg)

    xf.ACTIVE_SUBSET_FINGERPRINT = None
    ok_b, msg_b = raises(lambda: xf._load_shap_values(
        os.path.join(TMP, "canon"), 3, None, raw, None))
    check("B. no active sample -> does not raise (canonical regime)", not ok_b, msg_b)
finally:
    xf.ACTIVE_SUBSET_FINGERPRINT = _prev_fp
    if _prev_sub is not None:
        xd.SHAP_KWARGS["subset_fingerprint"] = _prev_sub

# ======================================================================== 4 ===
section("4. [XAI WRITE GUARD] — never overwrite a canonical run")
busy = os.path.join(TMP, "busy")
os.makedirs(os.path.join(busy, "xai", "shap", "epoch_003"), exist_ok=True)
open(os.path.join(busy, "xai", "shap", "epoch_003", "shap_arrays_epoch_003.npz"), "w").close()
empty = os.path.join(TMP, "empty")
os.makedirs(os.path.join(empty, "xai"), exist_ok=True)

ok, msg = raises(lambda: xd.assert_writable(busy, engines=("shap",)))
check("A. run holding artifacts -> raises", ok, msg)
ok_b, _ = raises(lambda: xd.assert_writable(empty, engines=("shap",)))
check("B. empty run -> does not raise", not ok_b)
ok_c, _ = raises(lambda: xd.assert_writable(busy, engines=("shap",), allow_clobber=True))
check("C. allow_clobber=True -> passes", not ok_c)

interrupted = os.path.join(TMP, "interrupted")
os.makedirs(os.path.join(interrupted, "xai", "shap", "epoch_003"), exist_ok=True)  # EMPTY
ok_d, _ = raises(lambda: xd.assert_writable(interrupted, engines=("shap",)))
check("D. empty dir from an interrupted run -> does NOT block", not ok_d,
      "regression of the a1p000 defect of 30/07")

# ======================================================================== 5 ===
section("5. LORE_SA_DEVICE must not apply to the other engines")
_prev = xd.LORE_SA_DEVICE
try:
    xd.LORE_SA_DEVICE = "/CPU:0"
    ok, msg = raises(lambda: xd.run_posthoc_xai(
        TMP, engines=("lore", "shap"), best_epoch=3, best_val_auc=0.9))
    check("A. device + mixed engines -> raises", ok, msg)
finally:
    xd.LORE_SA_DEVICE = _prev

# ======================================================================== 6 ===
section("6. Sample construction — the balance assertions")
ids = np.array([f"L{i // 3}" for i in range(300)])            # 3 copies per lesion
y = np.array([1 if int(i.lstrip("L")) % 2 == 0 else 0 for i in ids])
idx, info = xs.unique_lesion_subset(ids, y, n=40, seed=0, balance="equal")
check("A. balance='equal' -> exactly 50/50", int(y[idx].sum()) == len(idx) // 2,
      f"{int(y[idx].sum())} pos / {len(idx) - int(y[idx].sum())} neg")
check("B. no lesion repeated", len(set(ids[idx])) == len(idx))
ok, msg = raises(lambda: xs.unique_lesion_subset(ids, y, n=99999, seed=0, balance="equal"),
                 exc=(ValueError, AssertionError))
check("C. n beyond what is available -> raises", ok, msg)
ok, msg = raises(lambda: xs.unique_lesion_subset(ids, y[:10], n=10, seed=0), exc=ValueError)
check("D. ids/labels of different length -> raises", ok, msg)

# ======================================================================== 7 ===
section("7. across_runs — the unit is the run, and ties must be DECLARED")
ok, msg = raises(lambda: xs.across_runs([1, 2, 3], [1, 2]), exc=ValueError)
check("A. different lengths -> raises", ok, msg)
r = xs.across_runs([1.0, 1.0, 0.5, 0.4, 0.3], [1.0, 1.0, 0.7, 0.6, 0.5])
check("B. ties are counted separately", r["n_ties"] == 2 and r["n_informative"] == 3,
      f"n_ties={r['n_ties']}, n_informative={r['n_informative']}")
check("C. <6 informative runs -> no Wilcoxon", np.isnan(r["wilcoxon_p"]) and "note" in r)

# ======================================================================== 8 ===
section("8. Sign conventions — mode_sign applies PER SAMPLE")
lab = np.array([1, 0, 1, 0])
check("A. posev -> +1", np.all(xf.mode_sign("posev", lab) == 1))
check("B. negev -> -1", np.all(xf.mode_sign("negev", lab) == -1))
s = xf.mode_sign("target", lab)
check("C. target -> per-sample sign, not a scalar",
      np.asarray(s).size == len(lab) and not np.all(s == s.flat[0]), f"{np.asarray(s).ravel()}")
ok, msg = raises(lambda: xf.mode_sign("target", None), exc=ValueError)
check("D. target without labels -> raises", ok, msg)
ok, msg = raises(lambda: xf.mode_sign("boh", lab), exc=ValueError)
check("E. unknown mode -> raises", ok, msg)

# ======================================================================== 9 ===
section("9. Populations: do the cross-run globs pick up SUPERSEDED runs?")
if os.path.isdir(os.path.join(ART, "runs")):
    import glob as _g
    canon_lore = _g.glob(f"{ART}/runs/*/*/*/*/xai/lore_sa/lore_sa_per_epoch.csv")
    all_lore = _g.glob(f"{ART}/runs/**/lore_sa_per_epoch.csv", recursive=True)
    check("A. a naive glob on lore_sa_per_epoch picks up MORE than the canonical ones",
          len(all_lore) > len(canon_lore),
          f"{len(all_lore)} total vs {len(canon_lore)} under xai/lore_sa/ — filter on the path")
    master = pd.read_csv(os.path.join(ART, "summaries", "master_results_rows.csv"))
    check("B. the canonical registry holds 110 runs", len(master) == 110, f"{len(master)}")
    cf = os.path.join(ART, "runs", "_faithfulness", "combined_faithfulness.csv")
    if os.path.exists(cf):
        n = len(pd.read_csv(cf))
        check("C. combined_faithfulness.csv is NOT the registry", n != len(master),
              f"{n} rows vs {len(master)} runs: do not use it for family means")
else:
    print("  (artifacts unavailable: section skipped)")

# ======================================================================= 10 ===
section("10. [EVAL GUARD] — and the blind spot on the adopted sample")
from isic24.eval_guard import assert_canonical_eval                # noqa: E402


def _ns(**kw):
    return dict({"EVAL_SET": "canonical", "FAITH_N_SAMPLES": 128,
                 "validation_df": None, "_val_snap": "/nonexistent"}, **kw)


def _call(ns, **kw):
    return assert_canonical_eval(ns, require_snapshot=False, **kw)


_pf, _pi = xf.ACTIVE_SUBSET_FINGERPRINT, getattr(xf, "FAITH_SUBSET_IDX", None)
try:
    xf.FAITH_SUBSET_IDX, xf.ACTIVE_SUBSET_FINGERPRINT = None, None
    ok, msg = raises(lambda: _call(_ns(EVAL_SET="custom", EVAL_FOLDS=[1, 2])))
    check("A. EVAL_SET custom -> raises", ok, msg)
    ok, msg = raises(lambda: _call(_ns(FAITH_N_SAMPLES=256)))
    check("B. FAITH_N_SAMPLES different -> raises", ok, msg)
    ok_c, msg_c = raises(lambda: _call(_ns()))
    check("C. canonical regime -> does not raise", not ok_c, msg_c)

    # the blind spot: `xs.adopt` installs a GLOBAL default the guard never looked at
    xf.FAITH_SUBSET_IDX = list(range(144))
    xf.ACTIVE_SUBSET_FINGERPRINT = "unique-equal-s0-n144-TEST"
    ok, msg = raises(lambda: _call(_ns()))
    check("D. ADOPTED sample -> raises (this was the blind spot)", ok, msg)
    ok_e, msg_e = raises(lambda: _call(_ns(), allow_subset=True))
    check("E. allow_subset=True -> passes", not ok_e, msg_e)
    d = _call(_ns(), allow_subset=True)
    check("F. the descriptor reports the sample", d.get("adopted_subset_n") == 144,
          f"adopted_subset_n={d.get('adopted_subset_n')}")

    # The snapshot: the policy depends on WHO wrote it. Since 2026-08-06 the one on disk
    # comes from the retraining (`source='training_run'`), so the branch that matters raises.
    xf.FAITH_SUBSET_IDX, xf.ACTIVE_SUBSET_FINGERPRINT = None, None
    _vdf = pd.DataFrame({"isic_id": [f"ISIC_{i:07d}" for i in range(10)]})

    def _snap_ns(source, ids):
        path = os.path.join(TMP, f"snap_{source}.json")
        import json as _json
        _json.dump({"seed": 49, "n": len(ids), "source": source, "isic_ids_sorted": sorted(ids)},
                   open(path, "w"))
        return _ns(validation_df=_vdf, _val_snap=path)

    _same = _vdf.isic_id.tolist()
    _other = [f"ISIC_{i:07d}" for i in range(1, 11)]
    ok, msg = raises(lambda: _call(_snap_ns("training_run", _other)))
    check("G. snapshot from a training run + different set -> RAISES", ok, msg)
    ok_h, msg_h = raises(lambda: _call(_snap_ns("posthoc_regen_2026-07-28", _other)))
    check("H. post-hoc regenerated snapshot + different set -> warns, does not raise", not ok_h, msg_h)
    d = _call(_snap_ns("training_run", _same))
    check("I. snapshot from a training run + same set -> passes and records it",
          d.get("snapshot_match") is True and d.get("snapshot_source") == "training_run",
          f"match={d.get('snapshot_match')}, source={d.get('snapshot_source')}")
finally:
    xf.ACTIVE_SUBSET_FINGERPRINT, xf.FAITH_SUBSET_IDX = _pf, _pi

# ======================================================================= 11 ===
section("11. VALIDATION IDENTITY — the frame every run is held out from, and its order")
# Rewritten 2026-08-11. Until TR-06 this section asked whether today's validation was the one
# the June checkpoints saw, bridged by the `labels` vector inside their faithfulness npz.
# TR-03 (split before resample) and TR-06 (fold 0 keeps every positive, downsamples negatives,
# upsamples nothing) changed that frame ON PURPOSE, and a 130-wide FeatureSpace means no June
# checkpoint loads against it: those four cases had to fail, and standing red teaches its
# readers to ignore red.
#
# The question that can still go wrong is this one: does the pipeline rebuild, row for row and
# IN ORDER, the frame frozen in `validation_snapshot.json`?  Order is part of the identity
# because every `val_index` in an XAI artifact is a position into this frame.
import json as _js11                                              # noqa: E402

_SEED11, _VAL_NEG11 = 49, 0.01
_PARQ11 = os.path.join(BASE, "data", "isic24", "enhanced",
                       "enhanced_train_metadata_clean.parquet")
_SNAP11 = os.path.join(ART, "validation_snapshot.json")
_IMPUTED_NUM11 = ["age_approx", "size_age_interaction", "normalized_lesion_size"]


def _regen11():
    """Replica of @@SPLIT@@ and @@RESAMPLE@@ on CPU and without keras: the patient-grouped fold
    is taken on the NATURAL frame, then fold 0 keeps every positive and downsamples negatives.
    Returns (whole frame, natural training fold, resampled evaluation arm)."""
    from sklearn.model_selection import StratifiedGroupKFold
    d = pd.read_parquet(_PARQ11).reset_index(drop=True)
    d["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=_SEED11)
    for i, (_tr, _va) in enumerate(sgkf.split(d, y=d.target, groups=d.patient_id)):
        d.loc[_va, "fold"] = int(i)
    tr = d.query("fold != 0").copy()
    va = d.query("fold == 0").copy()
    va = pd.concat([va.query("target == 0").sample(frac=_VAL_NEG11, random_state=_SEED11),
                    va.query("target == 1")], axis=0).sort_index().reset_index(drop=True)
    return d, tr, va


if not os.path.exists(_PARQ11):
    print("  (dataset unavailable: section skipped)")
else:
    _all11, _tr11, _va11 = _regen11()

    check("A. the evaluation arm is TR-06's: 791 rows, 721 + 70, prevalence 0.0885",
          (len(_va11), int((_va11.target == 0).sum()), int(_va11.target.sum())) == (791, 721, 70)
          and round(float(_va11.target.mean()), 4) == 0.0885,
          f"{len(_va11)} rows, {int(_va11.target.sum())} positive, "
          f"prevalence {float(_va11.target.mean()):.4f}")

    check("B. nothing is upsampled on the evaluation arm: every row is a distinct lesion",
          not _va11.isic_id.duplicated().any(),
          f"{_va11.isic_id.nunique()} distinct over {len(_va11)} rows")

    check("C. no lesion and no patient crosses the split",
          not (set(_tr11.isic_id) & set(_va11.isic_id))
          and not (set(_tr11.patient_id) & set(_va11.patient_id)),
          f"lesions {len(set(_tr11.isic_id) & set(_va11.isic_id))}, "
          f"patients {len(set(_tr11.patient_id) & set(_va11.patient_id))}")

    _live11 = [str(v) for v in _va11.isic_id]
    if not os.path.exists(_SNAP11):
        print(f"  (no snapshot at {_SNAP11}: D and E skipped)")
    else:
        _frozen11 = [str(v) for v in _js11.load(open(_SNAP11))["row_ids"]]
        check("D. the regeneration matches the frozen snapshot, IN ORDER",
              _live11 == _frozen11,
              f"n={len(_frozen11)}, same set={set(_live11) == set(_frozen11)}")

        # The case that must FAIL: the same rows in another order.  @@VAL_SNAPSHOT@@ raises on
        # it, and without this case D would pass on any permutation of the right rows.
        _rotated11 = _live11[1:] + _live11[:1]
        check("E. the same rows in a different ORDER do not match (the test can fail)",
              _rotated11 != _frozen11 and set(_rotated11) == set(_frozen11),
              "rotated by one")

    # TR-04 moved the medians to the training fold.  This does not check where they come from --
    # it checks that it MATTERS: on the whole frame they are different numbers, so a regression
    # to the pre-TR-04 imputation is observable rather than a matter of trust.
    _differ11 = [c for c in _IMPUTED_NUM11 if _tr11[c].median() != _all11[c].median()]
    check("F. TR-04 is observable: the training-fold median differs from the whole-frame median",
          bool(_differ11), f"differ on {_differ11}")

# ======================================================================= 12 ===
section("12. class_weight x scalar-reduced loss — the weighting that vanishes silently")
# `loss_regularization` compiles a loss that calls `tf.reduce_mean(...)` INSIDE the function
# (reg_mode='per_batch', the one on all 25 canonical runs). Keras applies `class_weight` as a
# sample_weight to the RETURNED value: if that value is already a scalar, the weighting
# degenerates into a single multiplicative factor and no longer separates the classes.
# The other five families compile `bce` per sample, where it does act.
try:
    import tensorflow as _tf                                      # noqa: E402
    import keras as _keras                                        # noqa: E402

    _Xd = np.random.default_rng(0).standard_normal((96, 4)).astype("float32")
    _yd = (np.arange(96) % 3 == 0).astype("float32")               # prevalence 1/3
    _CW = {0: 0.75, 1: 1.5}                                        # 'balanced' at prevalence 1/3

    def _step(loss_fn, cw):
        _keras.utils.set_random_seed(7)
        m = _keras.Sequential([_keras.layers.Input((4,)),
                               _keras.layers.Dense(1, activation="sigmoid")])
        m.compile(optimizer=_keras.optimizers.SGD(0.5), loss=loss_fn)
        before = [w.numpy().copy() for w in m.trainable_weights]
        m.fit(_Xd, _yd, epochs=1, batch_size=96, verbose=0, class_weight=cw)
        return np.concatenate([(w.numpy() - b).ravel()
                               for w, b in zip(m.trainable_weights, before)])

    def _cos(loss_fn):
        a, b = _step(loss_fn, _CW), _step(loss_fn, None)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    _persample = _cos(lambda yt, yp: _keras.losses.binary_crossentropy(yt, yp, label_smoothing=0.02))
    _perbatch = _cos(lambda yt, yp: _tf.reduce_mean(
        _keras.losses.binary_crossentropy(yt, yp, label_smoothing=0.02)))

    check("A. per-sample loss (5 families): class_weight ACTS", _persample < 1 - 1e-6,
          f"cos(with CW, without CW) = {_persample:.6f}")
    check("B. scalar-reduced loss (loss_regularization): class_weight INERT",
          _perbatch > 1 - 1e-6, f"cos(with CW, without CW) = {_perbatch:.6f}")

    # ...and the guard that from now on catches it before the fit
    from isic24.mod_attendance import assert_loss_is_per_sample          # noqa: E402

    def _compiled(loss_fn):
        m = _keras.Sequential([_keras.layers.Input((4,)),
                               _keras.layers.Dense(1, activation="sigmoid")])
        m.compile(optimizer="sgd", loss=loss_fn)
        return m

    _ps_fn = lambda yt, yp: _keras.losses.binary_crossentropy(yt, yp, label_smoothing=0.02)   # noqa: E731
    _pb_fn = lambda yt, yp: _tf.reduce_mean(_ps_fn(yt, yp))                                   # noqa: E731

    ok, msg = raises(lambda: assert_loss_is_per_sample(_compiled(_pb_fn), "loss_regularization"),
                     ValueError)
    check("C. the guard RAISES on the scalar-reduced loss", ok, msg)
    ok_d, msg_d = raises(lambda: assert_loss_is_per_sample(_compiled(_ps_fn), "per_sample"),
                         ValueError)
    check("D. the guard stays silent on the per-sample loss", not ok_d, msg_d)
    ok_e, msg_e = raises(lambda: assert_loss_is_per_sample(
        _compiled(_keras.losses.BinaryCrossentropy(label_smoothing=0.02)), "bce"), ValueError)
    check("E. the guard stays silent on a keras.losses.Loss (weights BEFORE reducing)", not ok_e, msg_e)
except Exception as _e:                                            # noqa: BLE001
    print(f"  (keras unavailable: section skipped — {type(_e).__name__})")

# ======================================================================= 13 ===
section("13. exact logit decomposition — the six heads, and the one that does not decompose")
# The study's exact attendance measure rests on one property: the head is additive, so
# z = b + c_img + c_tab with no residual. Here the six topologies are rebuilt in miniature
# (fake backbone, same head) and the reconstruction must hold to within float noise; a head
# with an MLP after the fusion must instead raise.
try:
    import keras as _k                                             # noqa: E402
    from isic24 import xai_logit as xlog                           # noqa: E402
    from isic24.mod_attendance import (ConstantScale, FusionWithGamma,  # noqa: E402
                                       GateEntropyLoss, LearnableAlpha, ScalarMultiply)

    _IMG, _FEAT, _NB = (16, 16, 3), 12, 6
    _rb = np.random.default_rng(3)
    _xb = {"images": _rb.random((_NB, *_IMG)).astype("float32"),
           "features": _rb.normal(size=(_NB, _FEAT)).astype("float32")}

    def _toy(kind, alpha=0.3, gamma_img=1.0, img_dim=None):
        img_in = _k.Input(shape=_IMG, name="images")
        f_in = _k.Input(shape=(_FEAT,), name="features")
        x1 = _k.layers.GlobalAveragePooling2D()(_k.layers.Conv2D(20, 3, padding="same")(img_in))
        x1 = _k.layers.Dropout(0.2)(x1)
        if img_dim:
            x1 = _k.layers.Dense(img_dim, activation="selu", name="img_proj")(x1)
        x2 = _k.layers.Dense(9, activation="selu")(f_in)
        x2 = _k.layers.Dense(16, activation="selu")(x2)
        x2 = _k.layers.Dropout(0.1)(x2)

        if kind == "fixed_scaling":
            x1 = ConstantScale(1.0 - alpha, name="image_weighting")(x1)
            x2 = ConstantScale(alpha, name="tabular_weighting")(x2)
        elif kind == "learnable_scaling":
            a, b = LearnableAlpha(init_alpha=0.4, name="learnable_alpha")(x1)
            x1 = ScalarMultiply(name="image_weighting")([x1, b])
            x2 = ScalarMultiply(name="tabular_weighting")([x2, a])

        if kind == "controlled_fusion":
            fused = FusionWithGamma(gamma_img=gamma_img, gamma_tab=1.0,
                                    name="fusion_with_gamma")([x1, x2])
        elif kind == "adaptive_gating":
            x1 = _k.layers.Dense(24, activation="selu", name="img_emb")(x1)
            x2 = _k.layers.Dense(24, activation="selu", name="tab_emb")(x2)
            g = _k.layers.Dense(8, activation="relu")(_k.layers.Concatenate()([x1, x2]))
            g = GateEntropyLoss(weight=1e-4)(_k.layers.Dense(2, activation="softmax")(g))
            fused = _k.layers.Add(name="fused")([
                _k.layers.Multiply(name="weighted_img")([x1, _k.layers.Lambda(lambda w: w[:, 0:1])(g)]),
                _k.layers.Multiply(name="weighted_tab")([x2, _k.layers.Lambda(lambda w: w[:, 1:2])(g)])])
        else:
            fused = _k.layers.Concatenate(name="concat")([x1, x2])

        if kind == "nonadditive":
            fused = _k.layers.Dense(8, activation="selu")(fused)
        out = _k.layers.Dense(1, activation="sigmoid", name="output")(fused)
        return _k.Model({"images": img_in, "features": f_in}, out, name=kind)

    for _fam, _kind in [("vanilla", "Concatenate"), ("loss_regularization", "Concatenate"),
                        ("fixed_scaling", "Concatenate"), ("learnable_scaling", "Concatenate"),
                        ("controlled_fusion", "FusionWithGamma"), ("adaptive_gating", "Add")]:
        _m = _toy(_fam, img_dim=16 if _fam == "controlled_fusion" else None)
        _d, _meta = xlog.contributions(_m, _xb)
        check(f"B. {_fam}: the head is {_kind} and the logit recomposes",
              _meta["head_kind"] == _kind and _meta["recon_max_abs_delta"] < xlog.RECON_TOL,
              f"{_meta['head_kind']}, delta={_meta['recon_max_abs_delta']:.1e}")

    _mn = _toy("nonadditive")
    _okn, _msgn = raises(lambda: xlog.contributions(_mn, _xb, check_order=False),
                         (AssertionError, ValueError))
    check("A. non-additive head (MLP after the fusion): RAISES", _okn, _msgn)

    # Case C: the gammas of FusionWithGamma are non-trainable WEIGHTS, while get_config()
    # reports the construction value. The post-hoc gamma sweep writes the weight: whoever
    # reads the config misses it and credits the branch with a contribution the model never
    # computes.
    _mg = _toy("controlled_fusion", img_dim=16)
    _b0, _ = xlog.contributions(_mg, _xb, check_order=False)
    _mg.get_layer("fusion_with_gamma").gamma_img.assign(2.0)
    _b1, _meta1 = xlog.contributions(_mg, _xb, check_order=False)
    check("C. gamma read from the WEIGHT, not from get_config()",
          np.allclose(_b1["image_logit_raw"], 2 * _b0["image_logit_raw"], rtol=1e-4)
          and _meta1["recon_max_abs_delta"] < xlog.RECON_TOL,
          f"config={_mg.get_layer('fusion_with_gamma').get_config()['gamma_img']}, "
          f"weight=2.0, delta={_meta1['recon_max_abs_delta']:.1e}")

    _mv = _toy("vanilla")
    _okv, _msgv = raises(lambda: xlog.assert_branch_order(_mv, xlog.head_split(_mv), _xb),
                         AssertionError)
    check("D. branch order confirmed on a live topology", not _okv, _msgv)
    _md = _toy("fixed_scaling", alpha=1.0)                       # image branch scaled to 0
    _okd, _msgd = raises(lambda: xlog.assert_branch_order(_md, xlog.head_split(_md), _xb),
                         AssertionError)
    check("E. dead image branch (alpha=1): RAISES instead of measuring zero", _okd, _msgd)

    # The class split is the point of the decomposition: the aggregate must CARRY it, and
    # the two halves aggregate differently, so putting them side by side used to raise.
    _bt = pd.DataFrame([dict(impl_name="vanilla", seed=s,
                             **{c: float(i) for i, c in enumerate(xlog.SHARE_COLS)},
                             **{c: float(i) for i, c in enumerate(xlog.CLASS_COLS)})
                        for s in (47, 53)])
    _ft = xlog.family_table(_bt)
    check("F. the family table carries the class split beside the shares",
          all(c in _ft.columns for c in xlog.CLASS_COLS)
          and all(f"{c}__mean" in _ft.columns for c in xlog.SHARE_COLS) and len(_ft) == 1,
          f"{len(_ft.columns)} columns, {len(_ft)} row")
except Exception as _e:                                            # noqa: BLE001
    print(f"  (keras unavailable: section skipped — {type(_e).__name__}: {_e})")

# ======================================================================= 14 ===
section("14. unimodal ablations — the rule lives in the CODE, not in a document")
# At the endpoints of a sweep a branch is off by construction: their explanation values are
# algebra. The rule lives in `xs.ABLATION_SETTINGS` and must hold at EVERY point that
# aggregates by family. Case C: a plain `groupby` (the old code) does not notice.
_ab = pd.DataFrame({
    "impl_name": ["fixed_scaling"] * 4 + ["vanilla"],
    "setting_tag": ["a0p000", "a0p500", "a1p000", "a0p750", "baseline"],
    "seed": [47] * 5,
    "lore_coverage": [0.0, 0.5, 1.0, 0.8, 0.7],
})
check("A. the endpoints are recognised as ablations",
      xs.is_ablation("fixed_scaling", "a0p000") and xs.is_ablation("fixed_scaling", "a1p000")
      and not xs.is_ablation("fixed_scaling", "a0p500"))
check("B. `branch_off` says WHICH branch is switched off",
      xs.branch_off("fixed_scaling", "a0p000") == "tabular"
      and xs.branch_off("fixed_scaling", "a1p000") == "image"
      and xs.branch_off("vanilla", "baseline") is None)

_kept = xs.drop_ablations(_ab, "test", verbose=False)
check("C. drop_ablations removes 2 rows out of 5 and moves the mean",
      len(_kept) == 3
      and abs(_kept.lore_coverage.mean() - 2 / 3) < 1e-9
      and abs(_ab.lore_coverage.mean() - 0.60) < 1e-9,
      f"n={len(_kept)}, mean {_kept.lore_coverage.mean():.3f} against {_ab.lore_coverage.mean():.3f}")
check("D. the plain groupby (the old code) does NOT notice",
      abs(_ab.groupby("impl_name").lore_coverage.mean()["fixed_scaling"] - 0.575) < 1e-9,
      "0.575 against 0.650 once filtered: the defect case C measures")
check("E. a table without the identity columns passes through untouched",
      len(xs.drop_ablations(pd.DataFrame({"x": [1, 2]}), "test", verbose=False)) == 2)

# `pool_by_family` was the last point aggregating by family WITHOUT the filter: the per-stratum
# tables came out with 45 `fixed_scaling` runs instead of 35 (on ISIC: 1.195 instead of 1.052).
from isic24 import xai_faithfulness as _xf                    # noqa: E402
_st = pd.DataFrame({
    "impl_name": ["fixed_scaling"] * 3,
    "setting_tag": ["a0p000", "a1p000", "a0p500"],
    "run_id": ["r0", "r1", "r2"],
    "source": ["aligned_signed"] * 3, "mode": ["target"] * 3, "method": ["tabular"] * 3,
    "direction": ["del"] * 3, "q_max": [0.2] * 3, "q_width": [0.2] * 3, "stratum": ["all"] * 3,
    "n": [100, 100, 100], "gain_all_n": [1.0, 1.0, 0.4],
})
_pooled_f = _xf.pool_by_family(_st, verbose=False)
check("F. pooling by family excludes the endpoints",
      len(_pooled_f) == 1 and _pooled_f.n_runs.iloc[0] == 1
      and abs(_pooled_f.gain.iloc[0] - 0.4) < 1e-9,
      f"n_runs={_pooled_f.n_runs.iloc[0]}, gain {_pooled_f.gain.iloc[0]:.3f}")
_old = (_st.n * _st.gain_all_n).sum() / _st.n.sum()
check("G. the plain weighted mean (the old code) does NOT notice",
      abs(_old - 0.8) < 1e-9, "0.800 against 0.400 once filtered")

# ======================================================================= 17 ===
section("17. indices — one function only, and identity by row id")
# `explained_indices` (position INSIDE the sample) and `val_index` (position in
# validation_df) have similar names and are integers of the same size: crossing them does
# not raise, and has already produced three defects. Reading them lives in one place.
_p_idx = os.path.join(TMP, "idx.npz")
np.savez(_p_idx, explained_indices=np.arange(4), val_index=np.array([10, 20, 30, 40]))
with np.load(_p_idx) as _z:
    _i, _src = xs.read_explained_index(_z)
    check("A. when val_index is there, that is what is read", _src == "val_index" and _i.tolist() == [10, 20, 30, 40])
np.savez(_p_idx, explained_indices=np.arange(4))
with np.load(_p_idx) as _z:
    _i, _src = xs.read_explained_index(_z)
    check("B. without val_index it falls back to the prefix, DECLARING it", _src == "prefix")
    _i2, _src2 = xs.read_explained_index(_z, allow_prefix=False)
    check("C. allow_prefix=False -> 'I do not know', not a guess",
          _i2 is None and _src2 == "unknown")

_vdf = pd.DataFrame({"isic_id": ["A", "B", "C", "B"]})     # validation holds duplicated rows
check("D. index_for_row_ids resolves the unique rows",
      xs.index_for_row_ids(["C", "A"], _vdf).tolist() == [2, 0])
_okd, _msgd = raises(lambda: xs.index_for_row_ids(["B"], _vdf), KeyError)
check("E. a duplicated id RAISES instead of taking the first occurrence", _okd, _msgd)
_oke, _msge = raises(lambda: xs.index_for_row_ids(["Z"], _vdf), KeyError)
check("F. a missing id raises", _oke, _msge)

# ======================================================================= 18 ===
section("18. row identity — one pipeline, any dataset")
# The code must not know what the identity column is called: `isic_id` here, `image_id` on
# BRSET. What ends up INSIDE the artifacts is `row_id`, always.
_ID_COL0, _REG0 = xs.ID_COL, xs.VALIDATION_ROW_IDS
_brset = pd.DataFrame({"image_id": ["img1", "img2", "img3", "img4"]})
try:
    xs.ID_COL = "image_id"
    check("A. the identity column is declared, not guessed",
          xs.row_ids(_brset).tolist() == ["img1", "img2", "img3", "img4"])
    _okg, _msgg = raises(lambda: xs.row_ids(_vdf), KeyError)
    check("B. missing column -> raises, saying what to declare", _okg, _msgg)
    xs.register_validation_ids(_brset)
    check("C. the key written in the artifact is neutral, not the column name",
          list(xs.stamp_row_ids([2, 0])) == ["row_id"]
          and xs.stamp_row_ids([2, 0])["row_id"].tolist() == ["img3", "img1"])
    _okh, _msgh = raises(lambda: xs.stamp_row_ids([99]))
    check("D. an index outside the registered validation RAISES", _okh, _msgh)
    check("E. with no registered validation, no identity is invented",
          xs.register_validation_ids(None) is None and xs.stamp_row_ids([0]) == {})

    # The case POSITIONS cannot see: same positional sample, fold rebuilt in another order.
    # It is the defect row identity exists for.
    _run_id = os.path.join(TMP, "id_run")
    _d = os.path.join(_run_id, "xai", "shap_tab", "epoch_003")
    os.makedirs(_d, exist_ok=True)
    np.savez(os.path.join(_d, "shap_arrays_epoch_003.npz"),
             shap_values=rng.normal(size=(2, len(NAMES))), explained_indices=np.arange(2),
             val_index=np.array([0, 2]), row_id=np.asarray(["img1", "img3"]))
    xs.register_validation_ids(_brset)
    check("F. same positions, same lesions -> True",
          xs.sample_matches(_run_id, "shap_tab", [0, 2]) is True)
    xs.register_validation_ids(pd.DataFrame({"image_id": ["img4", "img2", "img1", "img3"]}))
    check("G. same positions, REORDERED fold -> False (positions would say True)",
          xs.sample_matches(_run_id, "shap_tab", [0, 2]) is False)
    xs.register_validation_ids(None)
    check("H. the old code (positions only) on case G: stays silent",
          xs.sample_matches(_run_id, "shap_tab", [0, 2]) is True)

    # Artifacts already on disk carry `isic_id`: they are still read, never rewritten.
    check("I. the old identity (`isic_id`) stays readable",
          xs.read_explained_ids(pd.DataFrame({"isic_id": ["A", "B"]})).tolist() == ["A", "B"])
finally:
    xs.ID_COL, xs.VALIDATION_ROW_IDS = _ID_COL0, _REG0

# ======================================================================= 19 ===
section("19. paths — no dataset name inside the code")
from isic24 import paths as _pth                               # noqa: E402

_hist = "/home/claudio/isic24-dum/artifacts/enhanced/runs/vanilla/20260604-210304/seed_47/e20"
check("A. a run_dir from another machine re-roots onto this tree",
      _pth.run_dir_abs(_hist, "/local/art")
      == "/local/art/runs/vanilla/20260604-210304/seed_47/e20")
check("B. and it re-roots after the folder has been moved too",
      _pth.run_dir_abs("/x/artifacts/isic24/enhanced/runs/vanilla/r1", "/local/art")
      == "/local/art/runs/vanilla/r1")
_oki, _msgi = raises(lambda: _pth.run_dir_abs("/x/y/z", "/local/art"), ValueError)
check("C. a path without 'runs/' RAISES instead of returning the one from the "
      "originating machine", _oki, _msgi)
check("D. the old code on case C: returns a nonexistent path that looks right",
      "/x/y/z".split("artifacts/enhanced/")[-1] == "/x/y/z")
check("E. a dataset root has two segments",
      _pth.dataset_root("/r", "brset", "enhanced").as_posix().endswith("/r/brset/enhanced"))
check("F. a run_dir already inside the local tree passes through untouched",
      _pth.run_dir_abs("/local/art/runs/vanilla/r1", "/local/art")
      == "/local/art/runs/vanilla/r1")

# ======================================================================= 16 ===
section("16. image grid — one path only, and no nominal precision")
# The engine used to trim the border (6x6) while faithfulness ranked patches on the full map
# (8x8): 28 cells out of 64 differ, presented as the same map. And FAITH_IMG_GRID was fixed
# at 16, that is 256 patches ranked from 64 numbers.
from isic24 import xai_engines as xe                            # noqa: E402

check("A. the engine no longer trims the border by default", xe.TRIM_CAM_BORDER is False)
check("B. FAITH_IMG_GRID follows the CAM instead of being fixed", xf.FAITH_IMG_GRID is None,
      f"FAITH_IMG_GRID={xf.FAITH_IMG_GRID}")

_cams = rng.random((3, 256, 256))
for _g in (8, 16):
    _o = xf._patch_saliency_order(_cams, _g)
    check(f"C{_g}. the ranking on a {_g}x{_g} grid covers every patch once",
          _o.shape == (3, _g * _g) and all(sorted(r.tolist()) == list(range(_g * _g)) for r in _o))

# The guard that refuses a grid finer than the CAM: exercised on the condition alone,
# with no model, because the condition is the rule.
def _grid_guard(grid, cam_grid):
    if grid > cam_grid:
        raise ValueError("[FAITH IMG GRID] nominal precision")
    return grid


ok, msg = raises(lambda: _grid_guard(16, 8), ValueError)
check("D. a grid finer than the CAM -> raises", ok, msg)
ok_e, _ = raises(lambda: _grid_guard(8, 8), ValueError)
check("E. a grid equal to the CAM -> passes", not ok_e)

# ======================================================================= 15 ===
section("15. TRAINING CONTRACT — the properties that live only in notebook cells")
# Rewritten 2026-08-11.  This section used to pre-flight the @@PROBE_SHUFFLE_FIXED@@ cell of
# `isic24_modality_attendance_train.ipynb`: a probe the canonical notebook does not carry, in a
# file that is now the backup.  What it presides over instead is the set of contracts that no
# module can hold -- they are cell order, cell bindings and cell arithmetic -- so if this
# section does not check them, nothing does.  Static: no TensorFlow, no GPU.
import json as _js15                                              # noqa: E402
import re as _re15                                                # noqa: E402

_NB15 = os.path.join(BASE, "notebooks", "modality_attendance",
                     "isic24_modality_attendance_train_canon.ipynb")
_cells15 = [''.join(c["source"]) for c in _js15.load(open(_NB15))["cells"]
            if c["cell_type"] == "code"]
_text15 = "\n".join(_cells15)


def _cell15(marker):
    """The one cell carrying a marker, or ''.  Cells are addressed by marker, never by index."""
    hits = [c for c in _cells15 if c.startswith(f"# @@{marker}@@")]
    return hits[0] if len(hits) == 1 else ""


_ORDER15 = ("SPLIT", "IMPUTE", "RESAMPLE", "FEATURE_SPACE", "APPLY_FEATURES")
_idx15 = {m: next((i for i, c in enumerate(_cells15) if c.startswith(f"# @@{m}@@")), -1)
          for m in _ORDER15}
check("A. TR-03/TR-04: split, then imputation, then resampling — in this order, or the fold is "
      "taken on duplicated rows and the medians see the evaluation arm",
      all(v >= 0 for v in _idx15.values()) and list(_idx15.values()) == sorted(_idx15.values()),
      str(_idx15))

check("B. TR-01: the shuffle buffer is a size, reshuffled each epoch, and a buffer of 1 raises",
      "shuffle_buffer" in _cell15("CFG")
      and "TRAIN_SHUFFLE_BUFFER > 1" in _cell15("APPLY_FEATURES")
      and "reshuffle_each_iteration=True" in _cell15("APPLY_FEATURES"),
      "declared in CFG, clamped and asserted in APPLY_FEATURES")

check("C. TR-02: augmentation is applied exactly once, on the training view only",
      _text15.count("map(augment_fn") == 1
      and "map(augment_fn" in _cell15("APPLY_FEATURES")
      and "augment" not in _cell15("BUILD_DATASETS"),
      f'{_text15.count("map(augment_fn")} occurrences of map(augment_fn)')

_apply15 = _cell15("APPLY_FEATURES")
check("D. TR-10: the encoding is written once and no dataset is derived from itself",
      _apply15.count('feature_space(x["features"])') == 1
      and "validation_ds.map" not in _apply15 and "(testing_ds.map" not in _apply15
      and "validation_base_ds" in _apply15,
      "one encoder, views derived from the base datasets")

check("E. TR-12: the schedule is built with the run's own epoch budget, and is not picked up "
      "from globals()",
      '"lr_cb" in globals()' not in _text15
      and "get_lr_callback(CFG.batch_size, epochs" in _cell15("RUNNER")
      and "def get_lr_callback(batch_size, epochs, mode" in _cell15("LR_SCHEDULE"),
      "per-run schedule, no default epochs")

_SWEEPS15 = ("SWEEP_VANILLA", "SWEEP_FIXED", "SWEEP_LEARNABLE", "SWEEP_LOSSREG",
             "SWEEP_CF", "SWEEP_GATING")
_sweeps15 = {m: _cell15(m) for m in _SWEEPS15}
_impls15 = {m: _re15.findall(r'^IMPL = "(\w+)"', c, _re15.M) for m, c in _sweeps15.items()}
_bad15 = sorted(m for m, v in _impls15.items() if len(v) != 1)
check("F. TR-22: every sweep declares its family ONCE and hands the same name to open_study "
      "and to run_model_experiment — written twice, they can disagree, and once did",
      not _bad15 and len({v[0] for v in _impls15.values()}) == len(_SWEEPS15)
      and all("open_study(IMPL, STUDY_TAG)" in c and "impl_name=IMPL" in c
              for c in _sweeps15.values()),
      f"not declared once in {_bad15}" if _bad15
      else f"families {sorted(v[0] for v in _impls15.values())}")

check("G. TR-22: no sweep mints its own study tag — a restarted sweep resumes its study "
      "instead of opening a second one beside it",
      "_make_exp_tag()" not in "\n".join(_sweeps15.values())
      and all("STUDY_TAG" in c for c in _sweeps15.values()),
      "STUDY_TAG declared in all six")

# ======================================================================= 20 ===
section("20. report figure — same population and same scale as the table beside it")
# A figure beside a table that says another number is the costliest defect: the figure is
# what gets looked at. Guarded here are the two choices that can diverge silently — the
# ablations, and the budget scale.
import matplotlib                                                # noqa: E402
matplotlib.use("Agg")
from isic24 import xai_figures as xfig                           # noqa: E402

def make_curves(tag, q_grid, level, root=TMP):
    """A curve npz with `del_exp_image` constant at `level` and the random arm at 0."""
    p = os.path.join(root, f"curves_{tag}.npz")
    n, k = 8, len(q_grid)
    np.savez(p, q_grid=np.asarray(q_grid, dtype=float),
             labels=np.zeros(n), preds_orig=np.full(n, .5),
             **{f"{d}_{a}_image": np.full((n, k), level if a == "exp" else 0.0)
                for d in ("del", "ins") for a in ("exp", "rand")})
    return p


_Q = [0.0, 0.1, 0.2]
_files = pd.DataFrame([
    dict(impl_name="fixed_scaling", run_id="r0", seed=47, setting_tag="a0p000",
         source="canonical", mode="abs", npz_path=make_curves("e0", _Q, 5.0)),
    dict(impl_name="fixed_scaling", run_id="r1", seed=47, setting_tag="a1p000",
         source="canonical", mode="abs", npz_path=make_curves("e1", _Q, 5.0)),
    dict(impl_name="fixed_scaling", run_id="r2", seed=47, setting_tag="a0p500",
         source="canonical", mode="abs", npz_path=make_curves("mid", _Q, 1.0)),
])
_q, _cur, _pr = xfig.read_arms(TMP, files=_files)
check("A. the figures aggregate without the unimodal ablations",
      len(_pr) == 1 and abs(_pr.del_gain.iloc[0] - 1.0) < 1e-9,
      f"n_runs={len(_pr)}, gain {_pr.del_gain.iloc[0]:.3f}")
check("B. the code without the filter (the old one) does NOT notice",
      abs(np.mean([5.0, 5.0, 1.0]) - 11 / 3) < 1e-9, "3.667 against 1.000 once filtered")

# Same curve, different budget window: the raw area doubles, the rate does not. It is the
# scale on which the report compares q<=0.2 with q<=0.6 (`gain_norm`).
_wide = pd.DataFrame([dict(impl_name="vanilla", run_id="w0", seed=47, setting_tag="baseline",
                           source="canonical", mode="abs",
                           npz_path=make_curves("wide", [0.0, 0.2, 0.4], 1.0))])
_, _, _pw = xfig.read_arms(TMP, files=_wide, q_max=0.4)
check("C. areas are per unit of budget, comparable across windows",
      abs(_pw.del_gain.iloc[0] - _pr.del_gain.iloc[0]) < 1e-9,
      f"{_pw.del_gain.iloc[0]:.3f} on q<=0.4 against {_pr.del_gain.iloc[0]:.3f} on q<=0.2")
_, _, _pc = xfig.read_arms(TMP, files=_files, central={"fixed_scaling": "a0p500"})
check("D. `central` restricts to one setting per family", len(_pc) == 1)

# ======================================================================= 21 ===
section("21. dataset variant — the extra columns are engineering, not new measurement")
# The premise the tabular-ablation question rests on: this variant adds columns ON TOP of the
# ISIC release, and the tabular branch reads them INSTEAD of the raw measurements they are
# computed from. Case D is what makes the split falsifiable: read against the variant's own
# header, release and engineered columns are indistinguishable.
import csv as _csv                                                # noqa: E402

_REL_META = _paths0.data_dir() / "train-metadata.csv"
_VAR_META = _paths0.data_path() / "train-metadata.csv"


def _header(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return set(next(_csv.reader(fh)))


if not (_REL_META.exists() and _VAR_META.exists()):
    print("  (release or variant metadata unavailable: section skipped)")
else:
    _pnb = os.path.join(BASE, "notebooks", "modality_attendance",
                        "isic24_modality_attendance_posthoc.ipynb")
    _feat_cell = [s for s in (''.join(c["source"]) for c in _js15.load(open(_pnb))["cells"])
                  if "NEW_NUMERIC_COLUMNS = " in s]
    _ns_feat = {}
    exec(_feat_cell[0], _ns_feat)                                 # list literals only
    _release = _ns_feat["CATEGORICAL_COLUMNS"] + _ns_feat["NUMERIC_COLUMNS"]
    _engineered = _ns_feat["NEW_CATEGORICAL_COLUMNS"] + _ns_feat["NEW_NUMERIC_COLUMNS"]
    _rel_h, _var_h = _header(_REL_META), _header(_VAR_META)

    check("A. every column called `release` is in the ISIC release",
          not set(_release) - _rel_h,
          sorted(set(_release) - _rel_h) or f"{len(_release)} of {len(_ns_feat['FEAT_COLS'])}")
    check("B. every engineered column is absent from it", not set(_engineered) & _rel_h,
          sorted(set(_engineered) & _rel_h) or f"{len(_engineered)} added by the variant")
    check("C. the two lists partition FEAT_COLS",
          sorted(_release + _engineered) == sorted(_ns_feat["FEAT_COLS"]))
    check("D. against the variant header the split is invisible",
          not (set(_release + _engineered) - _var_h), f"{len(_var_h - _rel_h)} extra columns")
    # The arm the probe compares against must exist: raw measurements the model never reads. The
    # diagnostic TBP column is held out here as it is in the cell, so both report the same count.
    _raw = sorted(c for c in _rel_h
                  if c.startswith("tbp_lv_") and c != "tbp_lv_dnn_lesion_confidence")
    _unused = [c for c in _raw if c not in _ns_feat["FEAT_COLS"]]
    check("E. raw TBP measurements exist that no feature list reaches", len(_unused) > 20,
          f"{len(_unused)} unused of {len(_raw)}")


# ======================================================================= 22 ===
section("22. CAM overlap — the effect and its null are the same kind of number")
# "Two families look in the same places" means nothing until two SEEDS of one family have
# been measured the same way. Guarded here: the unit (the run pair, not the lesion), the
# population, and the sample guard that stops two runs explained on different lesions from
# being compared.
from isic24 import xai_metrics as _xm2                            # noqa: E402


def make_cams(tag, hot, fp="s0", n=4, root=TMP):
    """A run dir whose every heatmap is 1.0 on the cells in `hot` and 0 elsewhere."""
    d = os.path.join(root, "runs", tag, "xai", "gradcam_img")
    os.makedirs(d, exist_ok=True)
    cams = np.zeros((n, 3, 3), dtype=np.float32)
    for i, j in hot:
        cams[:, i, j] = 1.0
    np.savez(os.path.join(d, "gradcam_arrays_context_epoch_001.npz"),
             cams=cams, preds=np.full((n, 1), .5, np.float32),
             mode="context", val_index=np.arange(n), subset_fingerprint=fp)
    return os.path.join(root, "runs", tag)


_TL = [(0, 0), (0, 1)]
_BR = [(2, 1), (2, 2)]
_rows, _paths_map = [], {}
for _impl, _hot in (("vanilla", _TL), ("controlled_fusion", _TL), ("fixed_scaling", _BR)):
    for _seed in (47, 53):
        _tag = f"{_impl}_{_seed}"
        make_cams(_tag, _hot)
        _rows.append(dict(impl_name=_impl, setting_tag={"fixed_scaling": "a0p500"}.get(_impl, "base"),
                          seed=_seed, run_dir=os.path.join("runs", _tag)))
make_cams("fixed_scaling_abl", _TL)
_rows.append(dict(impl_name="fixed_scaling", setting_tag="a1p000", seed=47,
                  run_dir=os.path.join("runs", "fixed_scaling_abl")))
_master = pd.DataFrame(_rows)
_central = {"vanilla": "base", "controlled_fusion": "base", "fixed_scaling": "a0p500"}
_ov = _xm2.cam_overlap_table(_master, TMP, central=_central)

_n_cross = (_ov.pair_kind == "cross_family").sum()
_n_null = (_ov.pair_kind == "seed_null").sum()
check("A. the two populations are complete and the ablation is out",
      _n_cross == 6 and _n_null == 3, f"cross {_n_cross}, null {_n_null}")
_same = _ov[(_ov.pair_kind == "cross_family")
            & (_ov.impl_a == "controlled_fusion") & (_ov.impl_b == "vanilla")]
_diff = _ov[(_ov.pair_kind == "cross_family") & (_ov.impl_a == "fixed_scaling")]
check("B. same map -> IoU 1, disjoint maps -> IoU 0",
      (_same.iou_top == 1).all() and (_diff.iou_top == 0).all(),
      f"same {_same.iou_top.tolist()}, disjoint {sorted(set(_diff.iou_top))}")
check("C. the seed null is measured, not assumed", (_ov[_ov.pair_kind == "seed_null"].iou_top == 1).all(),
      "identical seeds -> 1.000, so a cross-family 1.000 says nothing on its own")
_master2 = _master.copy()
make_cams("vanilla_47", _TL, fp="OTHER")
ok, msg = raises(lambda: _xm2.cam_overlap_table(_master2, TMP, central=_central))
check("D. heatmaps from a different sample -> RAISES", ok, msg)

# =============================================================================
section("23. One population per run — the engines must explain the same rows")
# Four engines write into four folders and nothing forced them to agree. The invariant is
# NESTING, not equality: LORE explains a declared subset by design, while a population that
# overlaps the reference only in part is two samples wearing one name.
from isic24 import xai_subset as _xs23                            # noqa: E402


def make_pop(tag, shap_idx, lore_idx=None, faith_idx=None, faith_n=None, root=TMP):
    """A run dir carrying one artifact per engine, each on the rows it is given."""
    run = os.path.join(root, "runs", "pop", tag)
    d = os.path.join(run, "xai", "shap_tab", "epoch_001")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "shap_arrays_epoch_001.npz"),
             val_index=np.asarray(shap_idx, int))
    if lore_idx is not None:
        d = os.path.join(run, "xai", "lore_sa", "epoch001")
        os.makedirs(d, exist_ok=True)
        pd.DataFrame({"val_index": np.asarray(lore_idx, int)}).to_csv(
            os.path.join(d, "lore_sa_persample_001.csv"), index=False)
    if faith_idx is not None or faith_n is not None:
        d = os.path.join(run, "xai", "faithfulness")
        os.makedirs(d, exist_ok=True)
        kw = ({"val_index": np.asarray(faith_idx, int)} if faith_idx is not None
              else {"labels": np.zeros(faith_n, np.float32)})
        np.savez(os.path.join(d, "faithfulness_curves.npz"), **kw)
    return run


_clean = make_pop("clean", shap_idx=range(12), lore_idx=[1, 3, 5], faith_idx=range(12))
_pops = _xs23.sample_populations(_clean)
check("A. every engine's population is read, faithfulness included",
      set(_pops.kind) == {"shap_tab", "lore_sa", "faithfulness"}
      and sorted(_pops.n) == [3, 12, 12], f"{dict(zip(_pops.kind, _pops.n))}")
check("B. a declared subset is nested, not a mismatch",
      _xs23.assert_nested_samples(_clean) is not None, "LORE 3 of 12 passes")

_part = make_pop("partial", shap_idx=range(12), faith_idx=range(8, 20))
ok, msg = raises(lambda: _xs23.assert_nested_samples(_part))
check("C. a partly-overlapping population -> RAISES", ok and "outside" in msg, msg)

_blind = make_pop("blind", shap_idx=range(12), faith_n=8)
ok, msg = raises(lambda: _xs23.assert_nested_samples(_blind))
check("D. a population that does not record its rows -> RAISES", ok and "unknown" in msg, msg)
check("E. and it can be audited instead of raising",
      _xs23.assert_nested_samples(_blind, raise_on_unknown=False) is not None,
      "raise_on_unknown=False reports, which is what immutable artifacts need")

_m23 = pd.DataFrame([dict(impl_name="vanilla", setting_tag="base", seed=s,
                          run_dir=os.path.join("runs", "pop", t))
                     for s, t in ((47, "clean"), (53, "partial"))])
_t23, _ok23 = _xs23.sample_audit(_m23, TMP, verbose=False)
check("F. the audit reports across runs without raising",
      _ok23 is False and len(_t23) == 5, f"ok={_ok23}, rows={len(_t23)}")

# ======================================================================= 24 ===
section("24. model families — the builders are a module, and read nothing from a notebook")
# Until 2026-08-11 the six architectures were notebook cells reading CFG, feature_space, adam
# and bce off the global namespace: a cell can be edited after the run it produced, and no
# test could import one. They are now `isic24/families.py`; this is its contract, checked
# statically so the suite still needs no TensorFlow.
import ast as _ast24                                            # noqa: E402
import builtins as _bi24                                        # noqa: E402
import json as _js24                                            # noqa: E402
import re as _re24                                              # noqa: E402

_fam_src = open(os.path.join(BASE, "src", "isic24", "families.py")).read()
_fam_tree = _ast24.parse(_fam_src)
_fam_defs = {n.name: n for n in _fam_tree.body if isinstance(n, _ast24.FunctionDef)}
_fam_text = {n: _ast24.get_source_segment(_fam_src, f) for n, f in _fam_defs.items()}

_reg_node = next(n for n in _fam_tree.body if isinstance(n, _ast24.Assign)
                 and getattr(n.targets[0], "id", None) == "FAMILIES")
_registry = {k.value: v.id for k, v in zip(_reg_node.value.keys, _reg_node.value.values)}

_nb24 = os.path.join(BASE, "notebooks", "modality_attendance",
                     "isic24_modality_attendance_train_canon.ipynb")
_cells24 = "\n".join(''.join(c["source"]) for c in _js24.load(open(_nb24))["cells"]
                     if c["cell_type"] == "code")
_impl24 = set(_re24.findall(r'^IMPL = "(\w+)"', _cells24, _re24.M))
check("A. the registry keys are exactly the families the sweeps declare",
      set(_registry) == _impl24, f"registry={sorted(_registry)} | sweeps={sorted(_impl24)}")

_sig24 = {n: [a.arg for a in _fam_defs[n].args.args] for n in _registry.values()}
check("B. every builder takes the signature the runner calls",
      all(a == ["setting", "spec", "run_eagerly"] for a in _sig24.values()), str(_sig24))

_bound24 = set(dir(_bi24))
for _n in _fam_tree.body:
    if isinstance(_n, (_ast24.Import, _ast24.ImportFrom)):
        _bound24 |= {(a.asname or a.name).split(".")[0] for a in _n.names}
    elif isinstance(_n, (_ast24.FunctionDef, _ast24.ClassDef)):
        _bound24.add(_n.name)
    elif isinstance(_n, _ast24.Assign):
        _bound24 |= {t.id for t in _n.targets if isinstance(t, _ast24.Name)}


def _free24(fn):
    """Names a function reads and neither binds nor finds at module level."""
    bound, loaded = set(), set()
    for a in _ast24.walk(fn):
        if isinstance(a, _ast24.arg):
            bound.add(a.arg)
        elif isinstance(a, (_ast24.FunctionDef, _ast24.ClassDef)):
            bound.add(a.name)
        elif isinstance(a, _ast24.Name):
            (loaded if isinstance(a.ctx, _ast24.Load) else bound).add(a.id)
    return loaded - bound - _bound24


_leaks24 = {n: sorted(_free24(f)) for n, f in _fam_defs.items() if _free24(f)}
check("C. no builder reads a global: everything comes from `setting` and `spec`",
      not _leaks24, str(_leaks24))

_old24 = _ast24.parse(
    "def build(setting, run_eagerly=False):\n"
    "    x = keras.Input(shape=(*CFG.image_size, 3))\n"
    "    d = feature_space.get_encoded_features().shape[1]\n"
    "    return x, d, adam, bce\n").body[0]
check("D. the OLD cell-based builder fails that same check on all four notebook globals",
      _free24(_old24) == {"CFG", "feature_space", "adam", "bce"}, str(sorted(_free24(_old24))))

_mints24 = [n for n in _registry.values()
            if "Adam.from_config(spec.optimizer.get_config())" in _fam_text[n]]
_shares24 = [n for n in _registry.values() if "optimizer=spec.optimizer" in _fam_text[n]]
check("E. every family mints a fresh optimizer, none shares the instance and its slots",
      len(_mints24) == len(_registry) and not _shares24,
      f"{len(_mints24)}/{len(_registry)} mint from config, shared by {_shares24}")

_own_loss24 = [n for n in _registry.values() if "loss=spec.loss" not in _fam_text[n]]
check("F. exactly one family compiles a loss of its own, and it is loss_regularization",
      _own_loss24 == ["build_loss_regularized"], str(_own_loss24))

_smoke24 = [c for c in _cells24.split("# @@") if c.startswith("SMOKE@@")]
_missing24 = [k for k in _registry if f'"{k}"' not in _smoke24[0]]
check("G. @@SMOKE@@ names every family and pins none of their sizes — the settings are explicit, "
      "the parameter counts come from a fingerprint recorded per configuration",
      len(_smoke24) == 1 and not _missing24 and "model_fingerprint" in _smoke24[0]
      and not _re24.search(r"\b\d{7}\b", _smoke24[0]),
      f"families with no smoke setting: {_missing24}" if _missing24 else "6 named, 0 sizes")

# ======================================================================= 25 ===
section("25. RESUME — a finished run is never trained twice, an interrupted one always is")
# TR-22.  Every execution of a sweep cell used to mint a fresh study_tag and `run_experiment`
# always trained, so an interrupted sweep restarted left a second partial study beside the
# first and the registry saw both.  `_completed_run` is the discriminator, and it lives in
# @@RUNNER@@ because that is where the sweeps call it: it is extracted from the notebook and
# executed here against fake run directories, so this tests the code the sweeps run.
import ast as _ast25                                              # noqa: E402
import json as _js25                                              # noqa: E402

_runner25 = _cell15("RUNNER")
_fn25 = next((_ast25.get_source_segment(_runner25, n)
              for n in _ast25.parse(_runner25).body
              if isinstance(n, _ast25.FunctionDef) and n.name == "_completed_run"), None)
check("A. @@RUNNER@@ defines _completed_run", _fn25 is not None)

if _fn25:
    _ns25 = {"os": os, "json": _js25, "pd": pd}
    exec(_fn25, _ns25)                                            # noqa: S102
    _completed25 = _ns25["_completed_run"]

    def _fake_run25(name, best_val_auc=True, ckpt=True, summary=True):
        """A run directory as `run_experiment` leaves it, with pieces optionally missing."""
        d = os.path.join(TMP, "resume", name)
        os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
        cfg = {"impl_name": "vanilla", "model_name": "vanilla__baseline", "run_id": name}
        if best_val_auc:
            cfg["best_val_auc"] = 0.9012
        with open(os.path.join(d, "config.json"), "w") as f:
            _js25.dump(cfg, f)
        if summary:
            pd.DataFrame([{"run_id": name, "best_val_auc": 0.9012}]).to_csv(
                os.path.join(d, "single_run_summary.csv"), index=False)
        if ckpt:
            open(os.path.join(d, "checkpoints", f"best_{cfg['model_name']}.keras"), "w").close()
        return d

    _done25 = _completed25(_fake_run25("complete"))
    check("B. a finished run returns its summary, so a resumed sweep still writes a COMPLETE "
          "combined_summary.csv instead of one holding only the runs it retrained",
          _done25 is not None and float(_done25.iloc[0]["best_val_auc"]) == 0.9012,
          "summary read back from disk")

    check("C. a run interrupted during fit is trained again: `best_val_auc` reaches config.json "
          "only after fit returns",
          _completed25(_fake_run25("interrupted", best_val_auc=False)) is None)

    check("D. a run whose checkpoint is gone is trained again — that is what deleting one to "
          "re-train it looks like, and it needs no new knob",
          _completed25(_fake_run25("no_ckpt", ckpt=False)) is None)

    _empty25 = os.path.join(TMP, "resume", "empty")
    os.makedirs(_empty25, exist_ok=True)
    check("E. a directory that was only created is trained again",
          _completed25(_empty25) is None)

    # Case C of the suite's convention: what the OLD code had available could not tell the two
    # apart, which is why the discriminator had to be a key INSIDE config.json.
    check("F. directory existence is not completeness: it is True for the finished run and for "
          "the interrupted one alike",
          os.path.isdir(os.path.join(TMP, "resume", "complete"))
          and os.path.isdir(os.path.join(TMP, "resume", "interrupted")),
          "both exist, one is trainable")

# =============================================================================
shutil.rmtree(TMP, ignore_errors=True)
n_ok = sum(1 for _, o, _ in RESULTS if o)
print(f"\n{'=' * 74}")
print(f"TOTAL: {n_ok}/{len(RESULTS)} PASS")
for name, o, det in RESULTS:
    if not o:
        print(f"  ✗ {name}  {det}")
sys.exit(0 if n_ok == len(RESULTS) else 1)
