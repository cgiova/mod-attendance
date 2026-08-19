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
import inspect as _inspect28
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
        return False, f"UNEXPECTED exception {type(e).__name__}: {e}"


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
    m = pd.DataFrame([dict(impl_name="fixed_scaling", setting_tag=t, seed=47,
                           run_id=f"r_{t}", run_dir=p)
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
_prev_idx = xf.FAITH_SUBSET_IDX
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

    # The other half of the same defect: with no sample adopted, `_build_faith_subset` used to
    # take the first `n` validation rows, which is how one run came to hold two populations
    # under one name -- the prefix and the sample, overlapping in 41 rows of 128.
    xf.FAITH_SUBSET_IDX = None
    ok_c, msg_c = raises(lambda: xf._build_faith_subset(), RuntimeError)
    check("C. no adopted sample -> the rows to explain are not invented", ok_c, msg_c)
finally:
    xf.ACTIVE_SUBSET_FINGERPRINT = _prev_fp
    xf.FAITH_SUBSET_IDX = _prev_idx
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

# The faithfulness curves are here too, and this is what protects them: the folder name no
# longer decides anything, what a run already HOLDS does. June refuses, the canonical runs --
# empty by construction -- accept.
faith = os.path.join(TMP, "faith")
os.makedirs(os.path.join(faith, "xai", "faithfulness"), exist_ok=True)
open(os.path.join(faith, "xai", "faithfulness", "faithfulness.json"), "w").close()
ok_e, msg_e = raises(lambda: xd.assert_writable(faith, engines=("faithfulness",)))
check("E. a run that already holds curves refuses a second faithfulness pass", ok_e, msg_e)
ok_f, _ = raises(lambda: xd.assert_writable(empty, engines=("faithfulness",)))
check("F. a run with no curves accepts one", not ok_f)

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

# The LORE subset: one definition, and the prefix it replaces. `unique_lesion_subset` keeps
# the first occurrence of each lesion, so in the real sample the positives sit at the front.
_yL = np.array([1] * 40 + [0] * 60)
_sampL = np.arange(100)
_loreL = xs.lore_subset(_yL, _sampL, 20)
check("E. lore_subset is half per class, distinct, and inside the adopted sample",
      len(_loreL) == 20 and int(_yL[_loreL].sum()) == 10
      and len(set(_loreL)) == 20 and set(_loreL) <= set(_sampL),
      f"{int(_yL[_loreL].sum())} pos / {20 - int(_yL[_loreL].sum())} neg")
check("F. the prefix definition it replaces stays SILENT on the same sample: 20 positives, "
      "no negative, and a different set of rows",
      int(_yL[_sampL[:20]].sum()) == 20 and set(_sampL[:20]) != set(_loreL),
      f"{len(set(_sampL[:20]) & set(_loreL))}/20 rows in common")
ok, msg = raises(lambda: xs.lore_subset(_yL, _sampL, 100), exc=ValueError)
check("G. a class too small to fill half the subset -> raises, instead of returning fewer",
      ok, msg)

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
import glob as _g                                                 # noqa: E402

# The property is about the PATTERN, so it is measured on a tree built here. Read off the
# live artifacts it was silently conditional on superseded runs still lying in `runs/`, and
# it stopped holding the day they were archived — a guard that a cleanup can switch off was
# never guarding the glob.
_LORE9 = "lore_sa_per_epoch.csv"
_r9 = os.path.join(TMP, "pop9", "runs")
os.makedirs(os.path.join(_r9, "vanilla", "20260812-000000", "seed_8", "baseline__e20",
                         "xai", "lore_sa"), exist_ok=True)
os.makedirs(os.path.join(_r9, "_robustness", "stage1_lore", "20260604-210304"), exist_ok=True)
open(os.path.join(_r9, "vanilla", "20260812-000000", "seed_8", "baseline__e20",
                  "xai", "lore_sa", _LORE9), "w").close()
open(os.path.join(_r9, "_robustness", "stage1_lore", "20260604-210304", _LORE9), "w").close()
_canon9 = _g.glob(f"{_r9}/*/*/*/*/xai/lore_sa/{_LORE9}")
_all9 = _g.glob(f"{_r9}/**/{_LORE9}", recursive=True)
check("A. a naive glob on lore_sa_per_epoch picks up MORE than the canonical ones",
      len(_all9) == len(_canon9) + 1 and len(_canon9) == 1,
      f"{len(_all9)} total vs {len(_canon9)} under xai/lore_sa/ — filter on the path")

if os.path.isdir(os.path.join(ART, "runs")):
    canon_lore = _g.glob(f"{ART}/runs/*/*/*/*/xai/lore_sa/{_LORE9}")
    all_lore = _g.glob(f"{ART}/runs/**/{_LORE9}", recursive=True)
    check("A2. and on the live tree the two agree: no superseded LORE artifact is left "
          "under runs/ for the naive one to find",
          len(all_lore) == len(canon_lore),
          f"{len(all_lore)} total, {len(canon_lore)} canonical")
    master = pd.read_csv(os.path.join(ART, "summaries", "master_results_rows.csv"))
    # Not a fixed count: a registry is stale exactly when it names runs that are not there,
    # or when the studies it names hold runs it never registered.
    _rd9 = [xs.run_dir_abs(p, ART) for p in master.run_dir]
    _gone9 = [p for p in _rd9 if not os.path.isdir(p)]
    _studies9 = sorted({os.path.dirname(os.path.dirname(p)) for p in _rd9})
    _disk9 = sum(len(_g.glob(os.path.join(s, "*", "*", "single_run_summary.csv")))
                 for s in _studies9)
    check("B. the registry is the population on disk: every run it names exists, and the "
          "studies it names hold no run it leaves out",
          not _gone9 and _disk9 == len(master),
          f"{len(master)} runs over {len(_studies9)} studies, {len(_gone9)} missing")
    cf = os.path.join(ART, "runs", "_faithfulness", "combined_faithfulness.csv")
    if os.path.exists(cf):
        n = len(pd.read_csv(cf))
        check("C. combined_faithfulness.csv is NOT the registry", n != len(master),
              f"{n} rows vs {len(master)} runs: do not use it for family means")
else:
    print("  (artifacts unavailable: section skipped)")

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

    # The post-hoc rebuilds nothing: it LOADS `evaluation_frame.parquet`.  D compares the
    # regeneration with the id list; these two compare it with the file that is actually read,
    # which is where a frame and its snapshot could drift apart unseen.
    _FRAMES11 = {n: os.path.join(ART, f"{n}_frame.parquet") for n in ("evaluation", "training")}
    if not all(os.path.exists(p) for p in _FRAMES11.values()):
        print("  (frozen frames unavailable: G and H skipped)")
    else:
        _fe11 = pd.read_parquet(_FRAMES11["evaluation"])
        _ft11 = pd.read_parquet(_FRAMES11["training"])
        check("G. the frozen evaluation frame IS the regenerated arm, in order",
              [str(v) for v in _fe11.isic_id] == _live11,
              f"{len(_fe11)} frozen rows against {len(_va11)} regenerated")
        check("H. the two frozen frames share no lesion and no patient",
              not (set(_fe11.isic_id) & set(_ft11.isic_id))
              and not (set(_fe11.patient_id) & set(_ft11.patient_id)),
              f"lesions {len(set(_fe11.isic_id) & set(_ft11.isic_id))}, "
              f"patients {len(set(_fe11.patient_id) & set(_ft11.patient_id))}")

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
section("16. image grid — one map, trimmed, and the patches it measures")
# The backbone pads with `same` at five stride-2 stages, so the outer ring of the 8x8 grid is
# partly computed from zeros. With the trim switched off, the bottom-right cell reached a median
# 0.515 over the 132 canonical runs and won the argmax on 31% of the maps — and, being the
# maximum, it normalized every other cell. Two Grad-CAM implementations made it worse: the
# engine could trim while faithfulness ranked patches on the untrimmed map, the two presented
# as the same heatmap.
from isic24 import xai_engines as xe                            # noqa: E402

# A conv tensor whose border is the loudest thing in it: the artifact, in the shape it has.
_B, _G, _C = 2, 8, 4
_conv = np.full((_B, _G, _G, _C), 0.1, np.float32)
_conv[:, 3:5, 3:5, :] = 1.0                                     # the signal, at the centre
_conv[:, -1, :, :] = _conv[:, :, -1, :] = 9.0                   # the padded ring
_grads = np.full_like(_conv, 0.5)
_cam = xe.cams_from_conv(_conv, _grads)

check("A. the CAM drops one cell per side of the native grid",
      _cam.shape == (_B, _G - 2 * xe.CAM_TRIM, _G - 2 * xe.CAM_TRIM), f"shape {_cam.shape}")
check("B. the padded ring reaches neither the map nor the channel weights: the peak is the "
      "centre, not the border", all(np.unravel_index(c.argmax(), c.shape) in
                                    ((2, 2), (2, 3), (3, 2), (3, 3)) for c in _cam),
      f"argmax {[np.unravel_index(c.argmax(), c.shape) for c in _cam]}")

# ONE implementation, called by both paths: the drift that produced two maps under one name
# cannot come back by editing a copy, because there is no copy.
check("C. the faithfulness path computes its maps through the engine's function",
      "cams_from_conv(conv_out, tape.gradient(loss, conv_out))" in
      _inspect28.getsource(xf._gradcam_maps))

# Patch <-> pixel: cell (r, c) of the trimmed map perturbs exactly the pixels it measures.
_side = _G * xe.CAM_STRIDE
_img = np.zeros((1, _side, _side, 3), np.float32)
_repl = np.ones_like(_img)
_g6 = _G - 2 * xe.CAM_TRIM
for _r, _c in ((0, 0), (2, 3), (_g6 - 1, _g6 - 1)):
    _out = xf._perturb_image(_img, _repl, np.array([[_r * _g6 + _c]]), 1, _g6, deletion=True)
    _y0, _x0 = xe.CAM_ORIGIN + _r * xe.CAM_STRIDE, xe.CAM_ORIGIN + _c * xe.CAM_STRIDE
    _want = np.zeros_like(_img)
    _want[0, _y0:_y0 + xe.CAM_STRIDE, _x0:_x0 + xe.CAM_STRIDE, :] = 1.0
    check(f"D({_r},{_c}). the patch touches the pixels its cell measures",
          np.array_equal(_out, _want))

check("E. the patches cover the image minus the ring the trim leaves unmeasured — the "
      "identity holds at any image_size, so the next dataset inherits it",
      xe.CAM_ORIGIN + _g6 * xe.CAM_STRIDE == _side - xe.CAM_ORIGIN,
      f"{xe.CAM_ORIGIN} + {_g6}x{xe.CAM_STRIDE} != {_side - xe.CAM_ORIGIN}")

# The two arms must rank the SAME patches: pinning the unmeasured cells to the bottom of the
# explained ranking alone would raise `faithfulness_gain` by construction.
_src16 = _inspect28.getsource(xf.explanation_faithfulness)
check("F. the explained arm ranks the CAM's own cells, no finer",
      "np.argsort(-cams.reshape(len(cams), -1), axis=1)" in _src16 and
      "n_patches = grid * grid" in _src16)
check("G. the random arm draws from the same patch count",
      "rng.random((n, n_patches))" in _src16)

# ======================================================================= 15 ===
section("15. TRAINING CONTRACT — the properties that live only in notebook cells")
# Rewritten 2026-08-11.  This section used to pre-flight the @@PROBE_SHUFFLE_FIXED@@ cell of
# `isic24_modality_attendance_train.ipynb`: a probe the canonical notebook does not carry, in a
# file that is now the backup.  What it presides over instead is the set of contracts that no
# module can hold -- they are cell order, cell bindings and cell arithmetic -- so if this
# section does not check them, nothing does.  Static: no TensorFlow, no GPU.
import glob as _glob15                                            # noqa: E402
import json as _js15                                              # noqa: E402
import re as _re15                                                # noqa: E402


def _notebook15(marker):
    """The active dataset's notebook carrying a marker.  Every dataset brings its own
    train/post-hoc pair under its own file name, so a notebook is addressed by the dataset it
    belongs to and a marker only it holds, never by name."""
    hits = [p for p in sorted(_glob15.glob(os.path.join(
                BASE, "notebooks", "modality_attendance", f"*{_paths0.DATASET}*.ipynb")))
            if f"# @@{marker}@@" in open(p, encoding="utf-8").read()]
    if len(hits) != 1:
        raise FileNotFoundError(f"@@{marker}@@ is in {len(hits)} {_paths0.DATASET} notebooks: "
                                f"{[os.path.basename(h) for h in hits]}")
    return hits[0]


_NB15 = _notebook15("FAMILIES")
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
    _ns_feat = {}
    exec(_cell15("FEAT_COLS"), _ns_feat)                          # list literals only
    _release = _ns_feat["BASE_CAT_COLS"] + _ns_feat["BASE_NUM_COLS"]
    _engineered = _ns_feat["NEW_CAT_COLS"] + _ns_feat["NEW_NUM_COLS"]
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

# `open_study` is the one door of all six sweeps — section 15 case F proves each hands it
# `(IMPL, STUDY_TAG)` — so the resume contract is tested once and holds for every family.
import re as _re25                                              # noqa: E402
from isic24.run_ids import make_exp_tag as _mk25                # noqa: E402

_open25 = "\n".join(
    _ast25.get_source_segment(_runner25, n) for n in _ast25.parse(_runner25).body
    if (isinstance(n, _ast25.FunctionDef) and n.name == "open_study")
    or (isinstance(n, _ast25.Assign) and getattr(n.targets[0], "id", "") == "_STUDY_TAG_RE"))
_ns25o = {"os": os, "re": _re25, "RUNS_PATH": os.path.join(TMP, "studies"),
          "_make_exp_tag": lambda: "20260812-000000"}
exec(_open25, _ns25o)                                           # noqa: S102

_tag25, _dir25 = _ns25o["open_study"]("vanilla", None)
check("G. STUDY_TAG=None mints a tag, creates the directory, and prints a tag that is a valid "
      "resume tag", os.path.isdir(_dir25) and _ns25o["_STUDY_TAG_RE"].match(_mk25()) is not None)

_okh, _msgh = raises(lambda: _ns25o["open_study"]("vanilla", 20260811 - 231745), ValueError)
check("H. an UNQUOTED timestamp is a subtraction: it RAISES instead of opening "
      "runs/vanilla/20029066, which str() used to accept", _okh, _msgh)

_oki, _msgi = raises(lambda: _ns25o["open_study"]("vanilla", "20260101-120000"),
                     FileNotFoundError)
check("I. resuming a tag that is not on disk RAISES: makedirs(exist_ok=True) used to open an "
      "empty study and train the whole sweep into it", _oki, _msgi)

# ======================================================================= 26 ===
section("26. THE REGISTRY — a summary the others do not carry never enters it")
# The registry decides which runs every post-hoc loop sees. Its source used to be the
# study-level `combined_summary.csv`, an aggregate: one of the canonical five is written
# with padded headers, and pandas reads `" seed"` and `"impl_name        "` as columns of
# their own — the family lands in the registry with NaN keys and vanishes from the groupby.
from isic24 import xai_subset as xs26                            # noqa: E402

_ROOT26 = os.path.join(TMP, "registry")
_COLS26 = list(xs26.REGISTRY_COLS)


def _run26(study, impl, seed, tag, padded=False):
    """One run on disk: its own summary, in the schema the runner writes."""
    rd = os.path.join(study, f"seed_{seed}", f"{tag}__e20")
    os.makedirs(rd, exist_ok=True)
    row = dict(impl_name=impl, exp_tag=os.path.basename(study), setting_tag=tag,
               run_id=f"{os.path.basename(study)}/seed_{seed}__{tag}__e20", seed=seed,
               best_epoch=3, best_val_auc=0.9, run_dir=rd, model_name=f"{impl}__{tag}")
    if padded:                       # the real defect: `to_csv` after a formatted read
        with open(os.path.join(rd, "single_run_summary.csv"), "w") as fh:
            fh.write(", ".join(f"{c:<18}" for c in _COLS26) + "\n")
            fh.write(", ".join(f"{str(row[c]):<18}" for c in _COLS26) + "\n")
    else:
        pd.DataFrame([row]).to_csv(os.path.join(rd, "single_run_summary.csv"), index=False)
    return row


_studies26 = {}
for _impl26, _padded26 in (("vanilla", False), ("learnable_scaling", True)):
    _st26 = os.path.join(_ROOT26, _impl26, "20260101-000000")
    _rows26 = [_run26(_st26, _impl26, 8, "baseline", padded=_padded26)]
    _studies26[_impl26] = _st26
    # the study-level aggregate the OLD registry read, padded exactly like the per-run one
    with open(os.path.join(_st26, "combined_summary.csv"), "w") as _fh26:
        _w26 = (lambda v: f"{v:<18}") if _padded26 else str
        _fh26.write(",".join(_w26(c) for c in _COLS26) + "\n")
        for _r26 in _rows26:
            _fh26.write(",".join(_w26(str(_r26[c])) for c in _COLS26) + "\n")

_okA26, _msgA26 = raises(lambda: xs26.build_registry(_studies26))
check("A. a per-run summary with padded headers RAISES: `\" seed\"` is not `seed`, and the "
      "run enters the registry with NaN keys", _okA26, _msgA26)

_clean26 = {"vanilla": _studies26["vanilla"]}
_reg26 = xs26.build_registry(_clean26, n_expected=1)
check("B. the clean study builds: one row per run, every registry column present",
      len(_reg26) == 1 and not [c for c in _COLS26 if c not in _reg26.columns],
      f"{_reg26.shape} columns ok")

# C. the OLD source on case A: concat of the study aggregates, exactly as @@MASTER_TABLE@@ did.
_old26 = pd.concat([pd.read_csv(os.path.join(d, "combined_summary.csv")).assign(impl_name=i)
                    for i, d in _studies26.items()], ignore_index=True)
_seen26 = set(_old26.groupby("impl_name")[["best_val_auc"]].mean().index)
check("C. the OLD code stays silent on the same defect: it concatenates, and the padded "
      "family reaches the aggregate with setting_tag/run_dir at NaN",
      len(_old26) == 2 and _old26.setting_tag.isna().sum() == 1
      and _old26.run_dir.isna().sum() == 1,
      f"families in the groupby: {sorted(_seen26)}")

_empty26 = os.path.join(_ROOT26, "controlled_fusion", "20260101-000000")
os.makedirs(_empty26, exist_ok=True)
_okD26, _msgD26 = raises(lambda: xs26.build_registry({**_clean26,
                                                      "controlled_fusion": _empty26}))
check("D. a study with no run RAISES instead of dropping its family out of the registry",
      _okD26, _msgD26)

_okE26, _msgE26 = raises(lambda: xs26.build_registry(_clean26, n_expected=132))
check("E. a run count below the declared one RAISES, naming the count per family",
      _okE26, _msgE26)

_wrong26 = os.path.join(_ROOT26, "adaptive_gating", "20260101-000000")
_run26(_wrong26, "vanilla", 8, "base")
_okF26, _msgF26 = raises(lambda: xs26.build_registry({"adaptive_gating": _wrong26}))
check("F. a run filed under another family's study RAISES: the study directory is not "
      "evidence of what the run is", _okF26, _msgF26)

# ======================================================================= 27 ===
section("27. POST-HOC CONTRACT — the properties that live only in notebook cells")
# The counterpart of section 15 for the post-hoc notebook: who is explained and who is
# stamped as done are cell-level facts, and no module can hold them.
import json as _js27                                              # noqa: E402

_NB27 = _notebook15("FAITH_LOOP")
_cells27 = [''.join(c["source"]) for c in _js27.load(open(_NB27))["cells"]
            if c["cell_type"] == "code"]
_text27 = "\n".join(_cells27)


def _cell27(marker):
    """The one cell whose header starts with this marker, numbering included, or ''."""
    hits = [c for c in _cells27 if c.startswith(f"# @@{marker}")]
    return hits[0] if len(hits) == 1 else ""


_c1, _c3 = _cell27("SUBSET@@ 1/3"), _cell27("SUBSET@@ 3/3")
check("A. PH-15: the LORE subset has ONE definition and, since 16/08, ONE producer — "
      "xs.lore_subset in @@SUBSET@@ 3/3, no cell building an index of its own, and "
      "@@SHAPE@@ no longer guessing one",
      _text27.count("xs.lore_subset(") == 1 and "xs.lore_subset(" in _c3
      and "np.linspace(" not in _c1 + _c3 and "LORE_IDX_BY_N" not in _text27,
      f"{_text27.count('xs.lore_subset(')} call site, no local construction")

check("B. PH-05: a failed engine is never stamped as done — BOTH engine loops inspect the "
      "result and skip the run before mark_done",
      all('!= "ok"' in c and c.find("[FAIL]") < c.find("xs.mark_done") for c in (_c1, _c3)),
      "the failure path continues before the stamp, in each loop")

# ======================================================================= 28 ===
section("28. LORE reference pool — the neighbourhood is drawn from the TRAINING rows")
# `lore_sa` generates a neighbour by sampling `uniform(min, max)` per numeric feature and a
# level per categorical one, both read off the descriptor of the dataset it is handed. Handed
# the rows being explained, it can only resample their own box.
try:
    from isic24.xai_lore_sa import LoreEngineSA as _LE28                # noqa: E402

    _tr28 = pd.DataFrame({"x": np.linspace(0.0, 100.0, 200),
                          "site": ["A", "B", "C", "D"] * 50,
                          "target": [0, 1] * 100})
    _expl28 = _tr28[(_tr28.x >= 40) & (_tr28.x <= 45) & (_tr28.site == "A")]
    _eng28 = _LE28(feature_space=None, feat_cols=["x", "site"], to_feature_dict=None,
                   training_df=_tr28)

    _ref28 = _eng28._make_dataset(_eng28.training_df, _eng28.training_df["target"])
    _num28, _cat28 = _ref28.descriptor["numeric"]["x"], _ref28.descriptor["categorical"]["site"]
    check("A. the reference descriptor is the training population: the full numeric range and "
          "every level",
          _num28["min"] == 0.0 and _num28["max"] == 100.0
          and set(_cat28["distinct_values"]) == set("ABCD"),
          f"x in [{_num28['min']:.0f}, {_num28['max']:.0f}], levels "
          f"{sorted(_cat28['distinct_values'])}")

    _old28 = _eng28._make_dataset(_expl28, _expl28["target"])
    _on28, _oc28 = _old28.descriptor["numeric"]["x"], _old28.descriptor["categorical"]["site"]
    check("B. the explained rows as the reference stay SILENT and box the neighbourhood in: "
          "a level they do not carry cannot be generated, nor exist as a column",
          _on28["max"] - _on28["min"] < 10.0 and set(_oc28["distinct_values"]) == {"A"},
          f"x in [{_on28['min']:.1f}, {_on28['max']:.1f}], levels "
          f"{sorted(_oc28['distinct_values'])}")

    _src28 = _inspect28.getsource(_LE28.explain)
    check("C. explain() builds it from the training frame, not from the rows it explains",
          "self._make_dataset(self.training_df" in _src28
          and "_make_dataset(raw_df" not in _src28)
except ImportError as _e28:                                             # noqa: BLE001
    print(f"  (lore_sa unavailable: section skipped — {_e28})")

# ======================================================================= 29 ===
section("29. ALIGNED ATTRIBUTOR — one permutation, drawn the same way for every model")
# `max_evals = 2d+1` buys a single antithetic permutation. That is a budget decision, but the
# DRAW must not be free: `shap.explainers.Permutation` reseeds numpy's global generator at
# every construction, so without a seed each run ranked with its own order and left the
# process reseeded from entropy behind it.
try:
    import shap as _shap29                                            # noqa: E402
    from isic24 import xai_faithfulness as _xf29                      # noqa: E402

    # A THREE-way interaction. A pairwise one is not enough: the antithetic pair averages the
    # two orders of a couple exactly, so any permutation returns the exact Shapley value and
    # the case cannot fail. Order-dependence starts where a term needs three features at once.
    def _f29(X):
        X = np.atleast_2d(X)
        return X[:, 0] + 2.0 * X[:, 1] * X[:, 2] * X[:, 3]

    _bg29 = np.zeros((1, 4))
    _x29 = np.array([[1.0, 2.0, 1.5, 2.5]])
    _ev29 = 2 * 4 + 1

    def _phi29(seed):
        return np.asarray(_shap29.explainers.Permutation(_f29, _bg29, seed=seed)(
            _x29, max_evals=_ev29, silent=True).values).reshape(-1)

    check("A. the same seed gives the same attributions",
          np.allclose(_phi29(7), _phi29(7)), "reproducible")

    _phis29 = np.array([_phi29(s) for s in range(8)])
    check("B. the draw moves them: one antithetic permutation IS order-dependent, which is why "
          "it cannot be left to chance",
          len(np.unique(_phis29.round(6), axis=0)) > 1,
          f"{len(np.unique(_phis29.round(6), axis=0))} distinct attributions over 8 seeds, "
          f"spread {(_phis29.max(0) - _phis29.min(0)).max():.3f}")
    check("E. and efficiency holds all the same: the marginals telescope along ANY single "
          "permutation, so the check that could have caught this is silent by construction",
          np.abs(_phis29.sum(1) - (_f29(_x29)[0] - _f29(_bg29)[0])).max() < 1e-9,
          "sum(phi) = f(x) - f(baseline) for every draw")

    np.random.seed(123)
    _st29 = np.random.get_state()[1][:4].copy()
    _shap29.explainers.Permutation(_f29, _bg29)
    check("C. constructed without a seed it RESEEDS the global generator, silently — the old "
          "call site did this once per sample",
          not np.array_equal(np.random.get_state()[1][:4], _st29), "global state moved")

    _src29 = _inspect28.getsource(_xf29.aligned_tab_attributions)
    check("D. the call site passes the seed and puts the global state back",
          "seed=perm_seed + i" in _src29 and "np.random.set_state(np_state)" in _src29)
except ImportError as _e29:                                            # noqa: BLE001
    print(f"  (shap unavailable: section skipped — {_e29})")

# ======================================================================= 30 ===
section("30. LOGIT DECOMPOSITION — the cached table is checked against the rows asked for")
# `decompose_run` caches its per-sample table inside the run folder, and a folder name carries
# no population: a table computed on another sample is reused under the same path, and the
# aggregate then spans two populations with `n_samples` as its only trace.
from isic24 import xai_logit as _xlog                                   # noqa: E402


def _cached30(tag, val_index, root=TMP):
    """A run folder holding only the per-sample table, no checkpoint: loading one would fail."""
    rd = os.path.join(root, "runs", "study30", tag)
    d = os.path.join(rd, "xai", _xlog.OUT_DIRNAME)
    os.makedirs(d, exist_ok=True)
    v = np.asarray(val_index, int)
    pd.DataFrame({"run_id": tag, "row_id": [f"L{i:04d}" for i in v], "val_index": v,
                  "target": v % 2, "head_kind": "additive",
                  "image_head_dim": 8, "tabular_head_dim": 8,
                  "image_logit_centered": 0.50, "tabular_logit_centered": -0.25,
                  "image_abs_share_descriptive": 0.60,
                  "image_target_aligned": 0.10, "tabular_target_aligned": 0.20,
                  "recon_max_abs_delta": 1e-9}).to_csv(
        os.path.join(d, _xlog.PER_SAMPLE_CSV), index=False)
    return rd


_ask30 = np.arange(10)
_rd30a = _cached30("other_sample", _ask30 + 5)
_rd30b = _cached30("same_sample", _ask30)

_ok30, _m30 = raises(lambda: _xlog.decompose_run(_rd30a, TMP, subset_idx=_ask30))
check("A. a table decomposed on other rows is refused, not reused", _ok30, _m30)

check("B. the table decomposed on the rows asked for is reused, with no model loaded",
      np.array_equal(_xlog.decompose_run(_rd30b, TMP, subset_idx=_ask30)["val_index"], _ask30))

_master30 = pd.DataFrame({"impl_name": "vanilla", "setting_tag": "base", "seed": [0, 1],
                          "run_dir": [_rd30b, _rd30a]})
_ok30c, _m30c = raises(lambda: _xlog.decompose_study(_master30, TMP, subset_idx=None,
                                                     out_dir=TMP, verbose=False))
check("C. with no `subset_idx` to compare against, two populations are caught run to run "
      "before they reach one aggregate", _ok30c, _m30c)

_src30 = _inspect28.getsource(xf.explanation_faithfulness)
check("D. the faithfulness stamp names the rows the curves hold: when the saved SHAP covers "
      "part of the sample, `val_index` is narrowed with it",
      "eff_subset = eff_subset[ei]" in _src30 and 'npz["val_index"] = eff_subset' in _src30
      and "assert len(eff_subset) == n" in _src30)

# ======================================================================= 31 ===
section("31. FEATURE SPACE BINDING — this checkpoint against the space of this session")
# Nothing here is a constant: the run records the space it was trained with, the space
# reports its own width and adapt state, and the session declares the one it loaded. Any
# width written into the code or into a committed reference belongs to one dataset only.
from isic24 import xai_driver as _xd31                                  # noqa: E402

_DECL31 = "feature_space/20260810-192622/feature_space.keras"
_OTHER31 = "feature_space/20260810-163712/feature_space.keras"
_FP31 = "a" * 64


def _bind31(cfg, fp=_FP31):
    return _xd31.assert_run_feature_space(cfg, "run31", _DECL31, fp)


_ok31, _m31 = raises(lambda: _bind31({"feature_space_file": _OTHER31}))
check("A. a run trained against another space is refused before the checkpoint is loaded",
      _ok31, _m31)

check("B. the run that declares this space passes, and says how far the check reached",
      _bind31({"feature_space_file": _DECL31, "feature_space_fingerprint": _FP31})
      == "fingerprint" and _bind31({"feature_space_file": _DECL31}) == "file")

_ok31c, _m31c = raises(lambda: _bind31({"feature_space_file": _DECL31,
                                        "feature_space_fingerprint": "b" * 64}))
check("C. the same path re-adapted is caught too: same file, another adapt state", _ok31c, _m31c)

check("D. a run that records no space falls back to the width, and DECLARES that it did",
      _bind31({}) == "width")

# The two spaces of case E differ only in their adapt statistics.
from types import SimpleNamespace as _SNS31                             # noqa: E402
import keras as _keras31                                                # noqa: E402
import tensorflow as _tf31                                              # noqa: E402

_fs31 = []
for _scale in (1.0, 7.0):
    _f31 = _keras31.utils.FeatureSpace(
        features={"x": _keras31.utils.FeatureSpace.float_normalized(),
                  "c": _keras31.utils.FeatureSpace.string_categorical(num_oov_indices=0)},
        output_mode="concat")
    _f31.adapt(_tf31.data.Dataset.from_tensor_slices(
        {"x": np.arange(20, dtype="float32") * _scale,
         "c": np.array(["p", "q"] * 10)}).batch(4))
    _fs31.append(_f31)

_d31a, _fp31a = _xd31.feature_space_fingerprint(_fs31[0])
_d31b, _fp31b = _xd31.feature_space_fingerprint(_fs31[1])
_stub31 = _SNS31(inputs=[_SNS31(name="features", shape=(None, _d31a))])
check("E. the width check CANNOT see it: two spaces of equal width and different adapt "
      "state pass it, and only the fingerprint separates them",
      _d31a == _d31b and _fp31a != _fp31b
      and _xd31.assert_fs_matches_model(_stub31, _fs31[1]) == _d31b,
      f"width {_d31a} both, fingerprints {_fp31a[:8]} vs {_fp31b[:8]}")

_src31 = _inspect28.getsource(_xd31.load_run_model)
check("F. every loop inherits the check: load_run_model runs it BEFORE keras loads the "
      "checkpoint, so a mispaired run costs no GPU",
      _src31.index("assert_run_feature_space") < _src31.index("keras.models.load_model")
      and "ctx.feature_space_file" in _src31)

# ======================================================================= 32 ===
section("32. PAIR TABLES — the unit of inference is the seed, not the pair row")
# A pair table is a graph: runs are the nodes, pairs are the edges. A test over its rows
# counts each model once per neighbour, and on the canonical grid it turns six seeds into
# thirty-odd "observations". Guarded here: the reduction, what it does to the p-value, and
# the invariant that makes it definable at all.
from scipy import stats as _stats32                                        # noqa: E402


def pair_table(effect, null, impl="fixed_scaling"):
    """`effect` {seed: [values]} inside one seed, `null` {(seed_a, seed_b): value}."""
    r = [dict(impl_name=impl, pair_kind="adjacent_param", seed_a=s, seed_b=s, v=v)
         for s, vs in effect.items() for v in vs]
    r += [dict(impl_name=impl, pair_kind="same_param_seed", seed_a=a, seed_b=b, v=v)
          for (a, b), v in null.items()]
    return pd.DataFrame(r)


_SEEDS32 = [8, 47, 53, 71, 90, 101]
# The canonical shape: the knob moves the top-k features less than the seed does, on every
# seed — six settings per seed, and the complete set of seed pairs as the null.
_eff32 = {s: [0.86 + 0.01 * i for i in range(6)] for s in _SEEDS32}
_null32 = {(a, b): 0.70 for i, a in enumerate(_SEEDS32) for b in _SEEDS32[i + 1:]}
_t32 = pair_table(_eff32, _null32)
_c32 = _xm2.seed_level_contrast(_t32, "v", "adjacent_param", "same_param_seed", by="impl_name")
_row32 = _c32.iloc[0]

check("A. the unit reported is the seed, not the row",
      _row32.n_seeds == 6 and len(_t32) == 51,
      f"{int(_row32.n_seeds)} seeds out of {len(_t32)} pair rows")

_mwu32 = _stats32.mannwhitneyu(_t32.v[_t32.pair_kind == "adjacent_param"],
                            _t32.v[_t32.pair_kind == "same_param_seed"]).pvalue
check("C. the old inference is optimistic exactly where the new one CANNOT be: Mann-Whitney "
      "over the rows against the paired test over the seeds",
      _mwu32 < 1e-6 and _row32.sign_p >= _row32.p_floor
      and abs(_row32.p_floor - 0.03125) < 1e-12,
      f"rows p = {_mwu32:.2e}, seeds p = {_row32.sign_p:.4f} at the floor {_row32.p_floor:.5f}")

# Four times the pairs at UNCHANGED per-seed means: a row-level test reads them as evidence,
# a seed-level one does not move. That is what "the unit is the seed" has to mean.
_t32b = pair_table({s: [v for v in vs for _ in range(4)] for s, vs in _eff32.items()}, _null32)
_c32b = _xm2.seed_level_contrast(_t32b, "v", "adjacent_param", "same_param_seed", by="impl_name")
_mwu32b = _stats32.mannwhitneyu(_t32b.v[_t32b.pair_kind == "adjacent_param"],
                             _t32b.v[_t32b.pair_kind == "same_param_seed"]).pvalue
check("D. four times the pairs, the same six seeds -> the seed test is identical and the row "
      "test is not",
      abs(_c32b.iloc[0].sign_p - _row32.sign_p) < 1e-12
      and abs(_c32b.iloc[0].diff_mean - _row32.diff_mean) < 1e-12 and _mwu32b < _mwu32,
      f"rows {_mwu32:.2e} -> {_mwu32b:.2e}, seeds {_row32.sign_p:.4f} unchanged")

# A seed's null is the mean over the pairs that TOUCH it. On the complete graph that equals
# the mean over rows; on the chain of consecutive seeds — the null the producer used to
# build — it does not, because the two end seeds carry one pair and the middle ones two.
_t32c = pair_table({s: [0.9] for s in (8, 47, 53, 71)},
                   {(8, 47): 0.9, (47, 53): 0.3, (53, 71): 0.3})
_c32c = _xm2.seed_level_contrast(_t32c, "v", "adjacent_param", "same_param_seed")
_rowmean32 = _t32c.v[_t32c.pair_kind == "same_param_seed"].mean()
check("B. a seed's null is the mean over the pairs that touch it, not the mean over rows",
      abs(_c32c.iloc[0].null_mean - 0.525) < 1e-12 and abs(_rowmean32 - 0.5) < 1e-12,
      f"by seed {_c32c.iloc[0].null_mean:.4f}, by row {_rowmean32:.4f}")

_t32d = _t32.copy()
_t32d.loc[_t32d.pair_kind == "adjacent_param", "seed_b"] = 999
ok, msg = raises(lambda: _xm2.seed_level_contrast(_t32d, "v", "adjacent_param",
                                                  "same_param_seed", by="impl_name"))
check("E. an effect arm spanning two seeds -> RAISES instead of reducing", ok, msg)

_t32e = pd.concat([_t32, pair_table({}, _null32, impl="vanilla")], ignore_index=True)
_c32e = _xm2.seed_level_contrast(_t32e, "v", "adjacent_param", "same_param_seed",
                                 by="impl_name")
check("F. a family with no effect arm is dropped and NAMED, never averaged into the table",
      set(_c32e.impl_name) == {"fixed_scaling"},
      f"{sorted(set(_t32e.impl_name))} in, {sorted(set(_c32e.impl_name))} out")

# The producer has to emit the seed columns for both arms, or nothing above is computable.
_grid32 = [(t, s) for t in ("a0p250", "a0p500") for s in (47, 53, 71)]
_st32 = xmet.stability_table(
    pd.DataFrame([dict(impl_name="fixed_scaling", setting_tag=t, seed=s, run_id=f"{t}_{s}",
                       run_dir=make_shap_run(f"st32_{t}_{s}", np.arange(1, N + 1)))
                  for t, s in _grid32]), "/", NAMES)
check("G. stability_table keys both arms the same way: seed_a/seed_b, no mixed column",
      {"seed_a", "seed_b"} <= set(_st32.columns) and "seed" not in _st32.columns
      and _st32.seed_a.dtype.kind == "i"
      and (_st32[_st32.pair_kind == "adjacent_param"].seed_a
           == _st32[_st32.pair_kind == "adjacent_param"].seed_b).all(),
      f"columns {[c for c in _st32.columns if 'seed' in c]}")
check("H. the seed null is the COMPLETE set of seed pairs, not the consecutive ones: on 3 "
      "seeds that is 3 pairs per setting and not 2",
      (_st32.pair_kind == "same_param_seed").sum() == 6
      and (_st32.pair_kind == "adjacent_param").sum() == 3,
      f"{(_st32.pair_kind == 'same_param_seed').sum()} null pairs on 2 settings x 3 seeds")

# ======================================================================= 33 ===
section("33. AGGREGATE NAMES — a table never takes the name another population holds")
# The files of `_faithfulness/` are named after what a table IS, never after the runs it is
# over, so re-running a section writes over the previous study's table under its own name and
# nothing fails. Guarded here: what identifies a population, what does not, and the one case
# that must still pass — a study rewriting its own table.
_AGG = os.path.join(TMP, "agg33")
os.makedirs(_AGG, exist_ok=True)


def agg_csv(name, run_ids, cols=("run_id",)):
    p = os.path.join(_AGG, name)
    d = {c: list(run_ids) for c in cols}
    pd.DataFrame(dict(d, impl_name="fixed_scaling", setting_tag="a0p500",
                      seed=list(range(len(list(run_ids)))))).to_csv(p, index=False)
    return p


_june = agg_csv("june.csv", [f"j{i}" for i in range(110)])
_canon = [f"c{i}" for i in range(132)]
ok, msg = raises(lambda: xs.assert_same_population(_june, _canon))
check("A. another study's table under the name this one takes -> RAISES", ok, msg)

_own = agg_csv("own.csv", _canon)
ok_b, msg_b = raises(lambda: xs.assert_same_population(_own, _canon))
check("B. a study rewriting its OWN table passes", not ok_b, msg_b)
ok_b2, _ = raises(lambda: xs.assert_same_population(_own, _canon[:5]))
check("B2. and so does a partial redo: one run in common is enough", not ok_b2)
ok_b3, _ = raises(lambda: xs.assert_same_population(os.path.join(_AGG, "absent.csv"), _canon))
check("B3. a name nobody holds is free", not ok_b3)

# identity, precisely: the two studies share every family, setting and seed.
_shared = pd.read_csv(_june)[["impl_name", "setting_tag", "seed"]]
_shared_c = pd.read_csv(_own)[["impl_name", "setting_tag", "seed"]].head(len(_shared))
check("C. the OLD code has nothing to notice with: family, setting and seed are IDENTICAL "
      "across the two studies, and only run_id separates them",
      _shared.equals(_shared_c) and not (set(pd.read_csv(_june).run_id)
                                         & set(pd.read_csv(_own).run_id)),
      "same keys, disjoint run ids")

_anon = os.path.join(_AGG, "anon.csv")
pd.DataFrame(dict(impl_name=["fixed_scaling"], setting_tag=["a0p500"], seed=[8],
                  stability_at_k=[0.9])).to_csv(_anon, index=False)
ok, msg = raises(lambda: xs.assert_same_population(_anon, _canon))
check("D. a table that cannot name its runs is refused, not trusted", ok, msg)

_pairs = os.path.join(_AGG, "pairs.csv")
pd.DataFrame(dict(run_id_a=["c1"], run_id_b=["c2"], stability_at_k=[0.9])).to_csv(_pairs,
                                                                                 index=False)
ok_e, msg_e = raises(lambda: xs.assert_same_population(_pairs, _canon))
check("E. a PAIR table names its population in two columns, and both are read",
      not ok_e, msg_e)
ok_e2, _ = raises(lambda: xs.assert_same_population(_pairs, [f"j{i}" for i in range(110)]))
check("E2. and the same pair table against another population still raises", ok_e2)

_srcF = _inspect28.getsource(xs.report)
check("F. every aggregate xs.report writes is guarded, and the shape tables are cleared "
      "before either of them is written",
      _srcF.count("stamp_aggregate") == 3
      and _srcF.rindex("stamp_aggregate") < _srcF.index("df.to_csv(p, index=False)"),
      f"{_srcF.count('stamp_aggregate')} call sites in xs.report")

_srcG = _inspect28.getsource(xmet.stability_table)
check("G. the pair table the guard has to read now carries run ids",
      "run_id_a=" in _srcG and "run_id_b=" in _srcG
      and {"run_id_a", "run_id_b"} <= set(_st32.columns),
      f"{[c for c in _st32.columns if c.startswith('run_id')]}")

# PH-11. `run_id` says WHICH runs a table holds. It cannot say under which SAMPLE and which
# knobs their numbers were produced -- that lived only in notebook output, recoverable by
# opening the runs one by one. The provenance travels with the guard because the two are one
# act: a table that names its population and not its origin is half-identified.
import json as _js33                                                     # noqa: E402

_ctx0_33, _fp0_33, _idx0_33 = xf._CTX, xf.ACTIVE_SUBSET_FINGERPRINT, xf.FAITH_SUBSET_IDX
_snap0_33, _vids0_33 = xs.ACTIVE_SUBSET_SNAPSHOT, xs.VALIDATION_ROW_IDS
try:
    _vdf33 = pd.DataFrame({"isic_id": [f"ISIC_{i:07d}" for i in range(8)],
                           "target": [0, 1] * 4})
    xf.configure(xf.XaiContext(
        feat_cols=["a", "b"], training_df=pd.DataFrame({"a": range(20)}),
        validation_df=_vdf33, validation_ds=None, feature_space=None,
        to_feature_dict=None, encoded_feature_names=["a", "b"],
        feature_space_file="feature_space/20260810-192622", feature_space_fingerprint="ab12cd34"))
    xf.ACTIVE_SUBSET_FINGERPRINT, xf.FAITH_SUBSET_IDX = "unique-equal-s0-n144", list(range(144))
    xs.ACTIVE_SUBSET_SNAPSHOT = "/art/subset.unique-equal.n144.seed0.json"

    _tbl33 = agg_csv("provenance.csv", _canon)
    _prov33 = xs.stamp_aggregate(_tbl33, _canon, aggregate="provenance.csv", central=None)
    _side33 = os.path.splitext(_tbl33)[0] + ".provenance.json"
    _read33 = _js33.load(open(_side33))
    check("H. the aggregate says what produced it, beside the table and under its own name: "
          "the sample, its snapshot and the feature space — the three things run_id cannot say",
          os.path.exists(_side33)
          and _read33["subset_fingerprint"] == "unique-equal-s0-n144"
          and _read33["subset_n"] == 144
          and _read33["subset_snapshot"].endswith("n144.seed0.json")
          and _read33["feature_space_fingerprint"] == "ab12cd34"
          and _read33["n_runs"] == 132,
          f"{os.path.basename(_side33)}: {_read33['subset_fingerprint']}, "
          f"n={_read33['subset_n']}, {_read33['n_runs']} runs")

    check("I. the knobs the caller passes are part of it, so two tables computed the same way "
          "over the same runs are still told apart by what was asked of them",
          _read33["aggregate"] == "provenance.csv" and "central" in _read33
          and _read33["q_clean"] == xs.Q_CLEAN,
          f"aggregate={_read33['aggregate']}, q_clean={_read33['q_clean']}")

    xf._CTX = xf._Unconfigured()
    _okj, _msgj = raises(lambda: xs.aggregate_provenance())
    check("J. an unconfigured session RAISES instead of writing a table nobody can trace back "
          "to a sample — the sentinel answers, no branch of our own",
          _okj, _msgj)
finally:
    xf._CTX, xf.ACTIVE_SUBSET_FINGERPRINT, xf.FAITH_SUBSET_IDX = _ctx0_33, _fp0_33, _idx0_33
    xs.ACTIVE_SUBSET_SNAPSHOT, xs.VALIDATION_ROW_IDS = _snap0_33, _vids0_33

_srcK = "".join(_inspect28.getsource(f) for f in (_xlog.decompose_study, _xlog.gamma_curve))
check("K. the other two producers go through the same door: no aggregate anywhere is guarded "
      "without being stamped, which is what keeps the two from drifting apart",
      _srcK.count("stamp_aggregate") == 2 and "assert_same_population" not in _srcK,
      f"{_srcK.count('stamp_aggregate')} call sites over decompose_study and gamma_curve")

# ======================================================================= 34 ===
section("34. GATE DIAGNOSTICS — the population is the validation set, not its first batch")
# The gate is Dense(2, softmax) over the fused embeddings, so its weights are a per-sample
# quantity; the cell read them from `next(iter(ds))` and saved the mean as a whole-validation
# number. Guarded here: the pass covers every row, the one-batch mean is shown differing from
# it, and the feature ablation no longer draws from the global RNG.
import keras as _k34                                                    # noqa: E402
import tensorflow as _tf34                                              # noqa: E402
from isic24 import xai_logit as _xl34                                   # noqa: E402

_N34, _B34 = 200, 32


def gate_model34():
    """A model whose gate weight is a monotone function of the single feature.

    Its weights are pinned, not initialised: the point of the fixture is that the mean over a
    prefix and the mean over the whole set differ by construction, and a random draw could
    make that difference vanish.
    """
    img = _k34.Input(shape=(4, 4, 3), name="images")
    tab = _k34.Input(shape=(1,), name="features")
    e_img = _k34.layers.Dense(2, activation="selu", name="img_emb")(_k34.layers.Flatten()(img))
    e_tab = _k34.layers.Dense(2, activation="selu", name="tab_emb")(tab)
    g = _k34.layers.Dense(2, activation="softmax", name="gate_weights")(
        _k34.layers.Concatenate()([e_img, e_tab]))
    w_i = _k34.layers.Lambda(lambda w: w[:, 0:1])(g)
    w_t = _k34.layers.Lambda(lambda w: w[:, 1:2])(g)
    fused = _k34.layers.Add()([_k34.layers.Multiply()([e_img, w_i]),
                               _k34.layers.Multiply()([e_tab, w_t])])
    m = _k34.Model([img, tab], _k34.layers.Dense(1, activation="sigmoid", name="out")(fused))
    m.get_layer("img_emb").set_weights([np.full((48, 2), 1 / 48, "float32"),
                                        np.zeros(2, "float32")])
    m.get_layer("tab_emb").set_weights([np.array([[1.0, 0.0]], "float32"),
                                        np.zeros(2, "float32")])
    # The gate reads the tabular embedding only — the two image columns are zeroed — so the
    # weights stay a function of `features` while the image branch still moves the logit.
    m.get_layer("gate_weights").set_weights(
        [np.array([[0, 0], [0, 0], [2.0, -2.0], [0, 0]], "float32"), np.zeros(2, "float32")])
    m.get_layer("out").set_weights([np.ones((2, 1), "float32"), np.zeros(1, "float32")])
    return m


# `features` ramps from 0 to 1 down the frame, so the first batch is the bottom of the ramp
# and any per-sample gate must read differently there than over the whole set.
_feat34 = np.linspace(0.0, 1.0, _N34, dtype="float32").reshape(-1, 1)
_ds34 = _tf34.data.Dataset.from_tensor_slices((
    {"images": np.zeros((_N34, 4, 4, 3), "float32"), "features": _feat34},
    np.r_[np.zeros(_N34 // 2), np.ones(_N34 - _N34 // 2)].astype("float32"))).batch(_B34)

_m34 = gate_model34()
_df34 = _xl34.gate_per_sample(_m34, _ds34)
_sum34 = _xl34.gate_summary(_df34)

check("A. the pass covers every row, and the summary says how many",
      len(_df34) == _N34 and _sum34["n"] == _N34
      and list(_df34.val_index) == list(range(_N34)),
      f"n={_sum34['n']} of {_N34}, val_index stamped")

# C. the defect, measured: what the old code reported against what the row set says.
_gate34 = _k34.Model(_m34.inputs, _m34.get_layer("gate_weights").output)
_first34 = float(np.asarray(_gate34(next(iter(_ds34))[0], training=False))[:, 0].mean())
check("C. the old one-batch mean is NOT the whole-validation mean it was saved as",
      abs(_first34 - _sum34["gate_weight_mean_img"]) > 1e-4,
      f"batch 0 ({_B34} rows) {_first34:.6f} vs all {_N34} rows "
      f"{_sum34['gate_weight_mean_img']:.6f}")

check("B. the two gate weights are a softmax and still sum to one per sample",
      np.allclose(_df34.w_img + _df34.w_tab, 1.0, atol=1e-5))

# D. the ablation is reproducible: it used to draw the feature mask from the global stream.
_x34 = next(iter(_ds34))[0]
_a34 = [np.asarray(_xl34.ablate_batch(_x34, "tab", 0.5,
                                      _tf34.random.Generator.from_seed(7))["features"])
        for _ in range(2)]
_b34 = np.asarray(_xl34.ablate_batch(_x34, "tab", 0.5,
                                     _tf34.random.Generator.from_seed(9))["features"])
check("D. a declared seed makes the feature ablation reproducible, and a different seed "
      "moves it — the mask is no longer drawn from the global stream",
      np.array_equal(_a34[0], _a34[1]) and not np.array_equal(_a34[0], _b34),
      "same seed identical, other seed different")

check("E. at strength 1.0 the features are fully zeroed, which is why the unseeded draw "
      "was inert and the defect dormant",
      np.array_equal(np.asarray(_xl34.ablate_batch(_x34, "tab", 1.0,
                                                   _tf34.random.Generator.from_seed(0))["features"]),
                     np.zeros_like(np.asarray(_x34["features"]))),
      "mask all False -> zeros, whatever the draw")

# The images are identical and the gate ignores them, so greying one out moves the logit by
# exactly `selu(0.5) * 2 * w_img`: the impact of removing the image IS its gate weight, up to a
# constant. A correct rank correlation over all 200 rows can only be 1.
check("F. the rank correlation is the gate's own claim put to the test: on a model where the "
      "image impact is proportional to the image weight, it must come out at 1",
      abs(_sum34["spearman_w_img_delta_img"] - 1.0) < 1e-9 and len(_df34) == _N34,
      f"rho_img {_sum34['spearman_w_img_delta_img']:+.6f} over {len(_df34)} rows")

# ======================================================================= 35 ===
section("35. THE DECLARED ENVIRONMENT — what the pipeline imports, requirements declares")
# `xailib` was installed on this machine from a requirements line that was COMMENTED OUT, so
# the import worked here and nowhere else: the environment reproduced only because the machine
# remembered an install the declaration no longer described. The engine that imported it is
# gone; what stays is the check, walking the imports rather than trusting the file.
import ast as _ast35                                                    # noqa: E402
import glob as _glob35                                                  # noqa: E402
import json as _js35                                                    # noqa: E402

# module name -> the distribution that provides it, where the two differ.
_DIST35 = {"sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow",
           "tf_explain": "tf-explain", "keras_cv": "keras-cv", "skimage": "scikit-image"}
# The notebook runtime itself, which requirements does not declare any more than it declares
# jupyter: an import of it is not a dependency of the pipeline.
_RUNTIME35 = {"IPython"}
# The pipeline: the modules, the two canonical notebooks, and the dataset build the status
# names as the source of truth. The Colab exploration notebooks are not part of it.
_NB35 = ["notebooks/modality_attendance/isic24_modality_attendance_train_canon.ipynb",
         "notebooks/modality_attendance/isic24_modality_attendance_posthoc canon.ipynb",
         "notebooks/data_preproc/isic24_create_dataset.ipynb"]


def imports35(src):
    out = set()
    src = "\n".join(("#" + l if l.lstrip().startswith(("!", "%")) else l)
                     for l in src.splitlines())
    for n in _ast35.walk(_ast35.parse(src)):
        if isinstance(n, _ast35.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, _ast35.ImportFrom) and n.level == 0 and n.module:
            out.add(n.module.split(".")[0])
    return out


def nb_code35(path):
    nb = _js35.load(open(os.path.join(BASE, path)))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


_used35 = set()
for _f in sorted(_glob35.glob(os.path.join(BASE, "src", "isic24", "*.py"))):
    _used35 |= imports35(open(_f).read())
for _p in _NB35:
    _used35 |= imports35(nb_code35(_p))
_third35 = sorted(m for m in _used35
                  if m not in sys.stdlib_module_names and m not in {"isic24", "src"}
                  and m not in _RUNTIME35)

_req35 = open(os.path.join(BASE, "requirements.txt")).read()
_declared35 = {l.split("==")[0].split(">=")[0].split("<")[0].split("@")[0].split(";")[0]
               .split("[")[0].strip().lower()
               for l in _req35.splitlines() if l.strip() and not l.startswith("#")}
_missing35 = [m for m in _third35
              if _DIST35.get(m, m).lower() not in _declared35 and m.lower() not in _declared35]
check("A. every third-party module the pipeline imports is declared in requirements.txt",
      not _missing35, f"{len(_third35)} modules, undeclared: {_missing35 or 'none'}")

_bare35 = [l.strip() for l in _req35.splitlines()
           if l.strip() and not l.startswith("#")
           and not any(t in l for t in ("==", ">=", "<=", "<", ">", "@"))]
check("B. no requirement is left unversioned — a bare name resolves to whatever the day gives",
      not _bare35, f"unversioned: {_bare35 or 'none'}")

# C. the backend and the visible device are chosen BEFORE tensorflow is imported, or the
# import has already read the defaults and the assignment is decoration.
_env35 = ("KERAS_BACKEND", "CUDA_VISIBLE_DEVICES", "TF_CPP_MIN_LOG_LEVEL")
for _p in _NB35[:2]:
    _code35 = nb_code35(_p)
    _tf35 = min((_code35.index(t) for t in ("import tensorflow",) if t in _code35), default=-1)
    _late35 = [v for v in _env35
               if v not in _code35 or (_tf35 >= 0 and _code35.index(v) > _tf35)]
    check(f"C. {os.path.basename(_p)[:34]}: the environment is set before tensorflow is "
          f"imported", not _late35 and _tf35 > 0, f"set after the import: {_late35 or 'none'}")

_bad35 = "import tensorflow as tf\nos.environ['KERAS_BACKEND'] = 'tensorflow'\n"
_tf_bad = _bad35.index("import tensorflow")
check("D. the reversed order fails the same check, so it is the ORDER being measured and not "
      "the presence of the two lines",
      _bad35.index("KERAS_BACKEND") > _tf_bad,
      "assignment after the import is what the check catches")

# Cases A and B read the FILE. Neither can see the two halves of one gap: a pin that does not
# match what is on the machine (`tensorflow-metadata` declared 1.16.1, installed 1.21.0, which
# is what made `@@ENV@@` walk the whole TFDS tree before raising), and a dependency reached by
# NAME and never imported. Both are answered by the installed metadata, not by the source.
from importlib.metadata import PackageNotFoundError as _PNF35   # noqa: E402
from importlib.metadata import version as _ver35                # noqa: E402


def pins35(text):
    """{distribution: version} for the `==` lines alone -- a range is a deliberate choice."""
    out = {}
    for l in text.splitlines():
        l = l.split("#")[0].split(";")[0].strip()       # markers are dropped, not evaluated:
        if "==" in l:                                   # the two platform lines pin one version
            name, ver = l.split("==", 1)
            out[name.split("[")[0].strip().lower()] = ver.strip()
    return out


def installed35(dist):
    try:
        return _ver35(dist)
    except _PNF35:
        return None


_pins35 = pins35(_req35)
_drift35 = {d: (w, installed35(d)) for d, w in _pins35.items() if installed35(d) != w}
check("E. every pinned requirement is the version actually INSTALLED — a declaration the "
      "machine does not match describes an environment nobody has",
      not _drift35,
      "; ".join(f"{d}: declared {w}, installed {g}" for d, (w, g) in _drift35.items())
      or f"{len(_pins35)} pinned, all matching")

# Reached by NAME and never imported: nobody writes `import pyarrow`, it is pandas' parquet
# engine, resolved from a string -- so the import walk of case A is blind to it by construction,
# while every frame the pipeline freezes or reads goes through it.
_BY_NAME35 = ("pyarrow",)
_byname35 = [d for d in _BY_NAME35
             if installed35(d) is None or d.lower() not in _declared35]
check("F. the dependencies reached by name are declared and installed all the same — an import "
      "walk cannot reach them, so they are named here",
      not _byname35,
      f"missing: {_byname35}" if _byname35 else f"{len(_BY_NAME35)}: {', '.join(_BY_NAME35)}")

_probe35 = pins35("guard-probe-not-a-package==0.0.1\nnumpy>=2.0,<2.1\n")
check("G. it is the COMPARISON being measured: a range stays out of the pinned set, and a pin "
      "nothing satisfies is caught — otherwise E and F would pass over a broken check too",
      list(_probe35) == ["guard-probe-not-a-package"]
      and installed35("guard-probe-not-a-package") != "0.0.1",
      f"pinned set from a mixed file: {list(_probe35)}")


# ======================================================================= 36 ===
section("36. FEATURE PROVENANCE — the engineered columns against the code that creates them")
# The premise the study rests on: switching off the tabular branch does not remove the image's
# information, because most tabular features are computed FROM the image measurements. Written
# beside the formulas that claim can only be read; parsed off the functions that assign the
# columns, and recomputed through them, it is measured. The SEQUENCE is part of it — `clean_df`
# reassigns a column `feature_engineering` has already written, with another formula.
import json as _js36                                                    # noqa: E402
from isic24.feature_provenance import (engineered_provenance,           # noqa: E402
                                       verify_engineered)

_FUNCS36 = ("feature_engineering", "clean_df")
_FE36 = '''def feature_engineering(df):
    df["ratio"] = df["a"] / df["b"]
    df["shifted"] = df["a"] - df["c"]
    df["combo"] = df["ratio"] + df["shifted"]
    df["tag"] = df["s"] + "_" + df["t"]
    return df, ["ratio", "shifted", "combo"], ["tag"]
'''
_CLEAN36 = '''def clean_df(df):
    out = df.copy()
    out["ratio"] = out["ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
'''
_NB36 = os.path.join(TMP, "build36.ipynb")


def _nbcell36(src):
    return {"cell_type": "code", "source": src.splitlines(keepends=True),
            "metadata": {}, "outputs": [], "execution_count": None}


_js36.dump({"cells": [_nbcell36(_FE36),
                      _nbcell36("df = pd.read_csv('/nonexistent/metadata.csv')\n"),
                      _nbcell36(_CLEAN36)],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, open(_NB36, "w"))

_src36, _roots36 = engineered_provenance(_NB36, _FUNCS36)
check("A. a column built from other engineered columns resolves to the raw ones, so the map is "
      "expressed in columns the build never writes",
      _roots36["combo"] == {"a", "b", "c"} and _roots36["ratio"] == {"a", "b"},
      f"combo <- {sorted(_roots36['combo'])}")

check("B. only the functions are taken from the notebook: the cell between them reads a file "
      "that does not exist, and nothing executes it",
      "read_csv" not in _src36 and _src36.count("def ") == 2,
      f"{len(_src36.splitlines())} lines, the two functions alone")

# b=0 on the second row: the ratio is infinite out of the first function and 0.0 after the
# second, so the two disagree on a value that IS on disk.
_df36 = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan], "b": [2.0, 0.0, 5.0, 1.0],
                      "c": [0.5, 0.5, 1.0, 1.0], "s": ["x", "y", "x", "y"],
                      "t": ["p", "p", "q", "q"]})
_ns36 = {"np": np, "pd": pd}
exec(_src36, _ns36)
_built36 = _ns36["clean_df"](_ns36["feature_engineering"](_df36.copy())[0])

_partial36 = verify_engineered(_built36, _src36, _roots36, ("feature_engineering",))
check("C. the sequence is part of the transformation: checked against the first function alone, "
      "the column the second one rewrites reports a difference",
      _partial36["ratio"] > 0,
      f"ratio differs on {_partial36['ratio']} of {len(_built36)} rows")

_full36 = verify_engineered(_built36, _src36, _roots36, _FUNCS36)
check("D. through both, nothing differs — the substituted value and the NaN included",
      int(_full36.sum()) == 0, f"0 differing rows over {len(_built36)}")

_edited36 = _built36.copy()
_edited36.loc[0, "shifted"] = 99.0
_edited36.loc[2, "tag"] = "wrong"
_dev36 = verify_engineered(_edited36, _src36, _roots36, _FUNCS36)
check("E. a column that is not what it reads is caught in both kinds, numeric and categorical, "
      "and the one computed FROM it stays clean",
      _dev36["shifted"] == 1 and _dev36["tag"] == 1 and _dev36["ratio"] == 0
      and _dev36["combo"] == 0, f"differing rows {_dev36[_dev36 > 0].to_dict()}")

_trimmed36 = {k: v - {"b"} for k, v in _roots36.items()}
_ok36, _det36 = raises(lambda: verify_engineered(_built36, _src36, _trimmed36, _FUNCS36), KeyError)
check("F. a map naming too few sources cannot quietly agree: the build is handed the columns the "
      "map names, and the one left out is missing", _ok36, _det36)

_literal36 = ["a", "b", "shifted"]
_feat36 = ["a", "b", "shifted", "combo", "tag"]
check("G. the release half is the complement of what the build writes, so a column the build "
      "starts computing changes side on its own — the hand-typed list still calls it release",
      [c for c in _feat36 if c not in _roots36] == ["a", "b"] and "shifted" in _literal36,
      f"derived {[c for c in _feat36 if c not in _roots36]}, hand-typed {_literal36}")

_sneaky36 = _ns36["feature_engineering"](_df36.copy())[0].assign(extra=1.0)
check("H. a column created by a form the parse does not read stays out of the map, and would be "
      "filed as a release column; comparing the headers is what sees it, and that check does not "
      "depend on having read the code right",
      "extra" not in _roots36
      and (set(_sneaky36.columns) - set(_df36.columns)) - set(_roots36) == {"extra"},
      "a column the parse cannot see is exactly what the header difference reports")

# The live tree. The frozen evaluation frame carries the build's own metadata header plus the
# `fold` column the split adds, and it is the small one.
_NBL36 = os.path.join(BASE, "notebooks", "data_preproc", "isic24_create_dataset.ipynb")
_FRAME36 = os.path.join(ART, "evaluation_frame.parquet")
_NOTCAND36 = {"isic_id", "patient_id", "lesion_id", "target", "image_type", "copyright_license",
              "tbp_lv_dnn_lesion_confidence", "iddx_full",
              "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5"}

if not os.path.exists(_FRAME36):
    print("  (frozen evaluation frame unavailable: the live cases are skipped)")
else:
    _srcL36, _rootsL36 = engineered_provenance(_NBL36, _FUNCS36)
    _nsL36 = {}
    exec(_cell15("FEAT_COLS"), _nsL36)                             # list literals only
    _featL36 = _nsL36["FEAT_COLS"]
    _engL36 = [c for c in _featL36 if c in _rootsL36]
    _reachL36 = set().union(*(_rootsL36[c] for c in _engL36))
    _hdr36 = [c for c in pd.read_parquet(_FRAME36).columns if c != "fold"]
    _used36 = [c for c in _hdr36 if c in _featL36]
    _drop36 = [c for c in _hdr36 if c not in _featL36 and c not in _NOTCAND36]
    check("I. the metadata header partitions into the columns nothing may read, the ones the "
          "model reads and the ones it drops, with nothing left over",
          _NOTCAND36 <= set(_hdr36) and not set(_used36) & _NOTCAND36
          and len(_NOTCAND36) + len(_used36) + len(_drop36) == len(_hdr36),
          f"{len(_hdr36)} = {len(_NOTCAND36)} not candidates + {len(_used36)} used + "
          f"{len(_drop36)} dropped")

    _reex36 = [c for c in _drop36 if c in _reachL36]
    _abs36 = [c for c in _drop36 if c not in _reachL36]
    check("J. what the dropped columns become is a measurement and not the sentence beside them: "
          "part survives inside an engineered feature, part reaches the model in no form at all",
          bool(_reex36) and bool(_abs36),
          f"{len(_reex36)} re-expressed, {len(_abs36)} absent: {_abs36}")

    if _REL_META.exists():
        _outside36 = sorted(_reachL36 - _header(_REL_META))
        check("K. every engineered feature the model reads resolves to columns of the ISIC "
              "release: the variant computes, it does not measure",
              not _outside36,
              _outside36 or f"{len(_reachL36)} source columns, all in the release")

        _added36 = set(_hdr36) - _header(_REL_META)
        check("L. and the map is complete because the data says so: the columns the build ADDS "
              "to the source header are exactly the ones it names, so nothing it creates can be "
              "mistaken for a column the release already shipped",
              _added36 == set(_rootsL36),
              f"added {len(_added36)}, named {len(_rootsL36)}, unnamed "
              f"{sorted(_added36 - set(_rootsL36)) or 'none'}")


# ======================================================================= 37 ===
section("37. RULE ZERO — the repository speaks English, printed messages included")
# The audit is the single file written in Italian, by decision. Everything else that stays in
# the repository is English: comments, docstrings, raised messages, notebook markdown. The tree
# was walked once, on 16/08, and held 195 Italian lines across nine modules; this is here so
# the next one is caught by the suite instead of by reading.
import glob as _glob37                                                  # noqa: E402
import json as _js37                                                    # noqa: E402
import re as _re37                                                      # noqa: E402

# Words that cannot occur in English prose or inside an identifier. Four short ones are left
# out on purpose: they live inside `None`, `non-trainable`, `come out` and `Cordei`, which is
# how a language check turns into noise nobody trusts. The boundaries are letters and digits
# only: not the apostrophe, or every elided and truncated form would walk through, and not the
# underscore, which is what `_nuovi` and `n_campioni` hid behind until 16/08.  # noqa: lang
_WORDS37 = ("che", "della", "delle", "degli", "dei", "dal", "dalla", "viene", "vengono",     # noqa: lang
            "senza", "perche", "perché", "quindi", "nessun", "nessuna", "nessuno",           # noqa: lang
            "invece", "gia", "già", "piu", "più", "cioe", "cioè", "questo", "questa",        # noqa: lang
            "sulla", "nella", "nello", "ogni", "qui", "dove", "dentro", "stanno", "hanno",   # noqa: lang
            "scrive", "chiede", "lette", "alza", "esterna", "esterno", "nuovi", "nuovo",     # noqa: lang
            "storici", "storico", "visti", "attributore", "fedelta", "qualita", "identita",  # noqa: lang
            "vicinato", "campione", "campioni", "impronta")                                  # noqa: lang
_RE37 = _re37.compile(r"(?<![A-Za-z0-9])(" + "|".join(_WORDS37) + r")(?![A-Za-z0-9])", _re37.I)

# The vocabulary above is Italian by necessity, and it sits inside the population case A walks.
# It carries a marker rather than being excused by line number, so the section can be moved or
# extended without opening a hole behind it.
_SKIP37 = "noqa: lang"


def italian37(text):
    return [l.strip()[:72] for l in text.splitlines()
            if _SKIP37 not in l and _RE37.search(l)]


_NBS37 = ["notebooks/modality_attendance/isic24_modality_attendance_train_canon.ipynb",
          "notebooks/modality_attendance/isic24_modality_attendance_posthoc canon.ipynb",
          "notebooks/data_preproc/isic24_create_dataset.ipynb"]

_bad37 = {}
for _pat37 in ("src/isic24/**/*.py", "scripts/**/*.py"):
    for _f in sorted(_glob37.glob(os.path.join(BASE, _pat37), recursive=True)):
        if (_hits := italian37(open(_f, encoding="utf-8").read())):
            _bad37[os.path.relpath(_f, BASE)] = _hits
for _p in _NBS37:
    _nb37 = _js37.load(open(os.path.join(BASE, _p), encoding="utf-8"))
    if (_hits := italian37("\n".join("".join(c["source"]) for c in _nb37["cells"]))):
        _bad37[_p] = _hits

check("A. no module, no script and no canonical notebook carries Italian — comments, "
      "docstrings and raised messages alike, this suite included",
      not _bad37,
      "; ".join(f"{k}: {v[0]}" for k, v in list(_bad37.items())[:3]) or
      f"{len(_NBS37)} notebooks and every .py under src/ and scripts/")

check("B. and it is the LANGUAGE being measured, not an empty walk: a line of the prose "
      "deleted on 16/08 is caught",
      italian37("# Il collage cresce come il batch: sarebbe un PNG che nessuno guarda"),  # noqa: lang
      "the sentence that opened the Grad-CAM collage comment")

check("C. English that reads Italian to a careless matcher stays clean — `None`, "
      "`non-trainable`, `come out` and a truncated token buried in an English word are not "
      "the check firing",
      not italian37("x = None  # non-trainable weights come out of the solo branch, "
                    "named after Cordei and codeine"),
      "no false positive on the shapes that made a naive word list unusable")

check("D. the marker is what lets the vocabulary itself live inside the population A walks: "
      "without it, that same line is Italian and is caught",
      italian37("_WORDS37 = ('che', 'della', 'senza')"),                       # noqa: lang
      "the exemption is one marked line, not a rule about where the section sits")

check("E. an underscore does not shield an identifier: `_nuovi` and `n_campioni` reached a "  # noqa: lang
      "written CSV column because `\\w` counted `_` as part of the word",
      italian37("_nuovi = sorted(glob.glob(p))") and                           # noqa: lang
      italian37('fam = L.agg(n_campioni=("i", "size"))'),                      # noqa: lang
      "the two spellings that walked past the first version of this check")

check("F. and a word carrying the same trap in English does not fire: `palette` ends in a "
      "vocabulary entry and `_del` is a prefixed English abbreviation",
      not italian37("palette = _del_curves(z)  # deletion arm"),
      "widening the vocabulary did not widen the false positives")


# ======================================================================= 38 ===
section("38. [NOTEBOOKS] — a canonical notebook is addressable and runs in its own order")
# The two notebooks are the only executable entry points of the study, and they are the one
# place the suite could not see: a name defined three cells below its use, a marker naming two
# cells, a marker pointing at a cell that was deleted. All three were there on 16/08, and the
# first cost a NameError 132 runs into the faithfulness loop.
import ast as _ast38                                                    # noqa: E402
import builtins as _bi38                                                # noqa: E402
import json as _js38                                                    # noqa: E402
import re as _re38                                                      # noqa: E402

_NB38 = {"train": "notebooks/modality_attendance/isic24_modality_attendance_train_canon.ipynb",
         "posthoc": "notebooks/modality_attendance/isic24_modality_attendance_posthoc canon.ipynb"}
_BUILT38 = set(dir(_bi38)) | {"__name__", "__file__", "get_ipython", "display", "In", "Out"}
# A marker is the name AND its part: `@@SUBSET@@ 2/3` and `@@SUBSET@@ 3/3` are two
# addresses, and a bare `@@SUBSET@@` in a printed instruction is none.
_MARK38 = _re38.compile(r"@@[A-Z0-9_]+@@(?:\s+\d+/\d+)?")


def cells38(path):
    nb = _js38.load(open(os.path.join(BASE, path), encoding="utf-8"))
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


def _bind38(tree):
    out = set()
    for n in _ast38.walk(tree):
        if isinstance(n, (_ast38.FunctionDef, _ast38.AsyncFunctionDef, _ast38.ClassDef)):
            out.add(n.name)
        elif isinstance(n, _ast38.Name) and isinstance(n.ctx, _ast38.Store):
            out.add(n.id)
        elif isinstance(n, (_ast38.Import, _ast38.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, _ast38.arg):
            out.add(n.arg)
        elif isinstance(n, _ast38.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (_ast38.Global, _ast38.Nonlocal)):
            out.update(n.names)
    return out


def free38(cells):
    """Names loaded before any cell binds them, walking the notebook in its own order."""
    known, bad = set(), []
    for i, (src, kind) in enumerate(cells):
        if kind != "code":
            continue
        src = "\n".join(("#" + l if l.lstrip().startswith(("!", "%")) else l)
                        for l in src.splitlines())
        try:
            tree = _ast38.parse(src)
        except SyntaxError as e:
            bad.append((i, f"SYNTAX: {e}"))
            continue
        b = _bind38(tree)
        loads = {n.id for n in _ast38.walk(tree)
                 if isinstance(n, _ast38.Name) and isinstance(n.ctx, _ast38.Load)}
        bad += [(i, n) for n in sorted(loads - b - known - _BUILT38)]
        known |= b
    return bad


def markers38(cells):
    """marker -> how many cells it addresses. A cell's own marker is the FIRST on its header
    line; the ones after it are references, which is how a header can point at another cell."""
    out = {}
    for src, _ in cells:
        head = src.splitlines()[0] if src.splitlines() else ""
        if (m := _MARK38.search(head)):
            out[m.group(0)] = out.get(m.group(0), 0) + 1
    return out


_CELLS38 = {k: cells38(v) for k, v in _NB38.items()}
_free38 = {k: free38(v) for k, v in _CELLS38.items()}

check("A. no cell uses a name an earlier cell has not bound: the notebook runs in the order "
      "it is written in",
      not any(_free38.values()),
      "; ".join(f"{k} cell {i}: {n}" for k, v in _free38.items() for i, n in v[:3])
      or f"{sum(len(c) for c in _CELLS38.values())} cells over {len(_NB38)} notebooks")

check("B. and the walk is really linear: a cell calling a helper the NEXT cell defines is "
      "caught, which is the shape `assert_fs_matches_model` had",
      free38([("y = assert_fs_matches_model(m)", "code"),
              ("def assert_fs_matches_model(m): return 1", "code")]),
      "definition-after-use, the class of defect nbcheck was written for")

_dup38 = {k: {m: n for m, n in markers38(v).items() if n > 1} for k, v in _CELLS38.items()}
check("C. no marker names two cells: a marker is the address of ONE cell, and PyCharm cannot "
      "search an index",
      not any(_dup38.values()),
      "; ".join(f"{k}: {sorted(v)}" for k, v in _dup38.items() if v)
      or f"{sum(len(markers38(v)) for v in _CELLS38.values())} distinct markers")

_dangling38 = {}
for _k, _cs in _CELLS38.items():
    _have = set(markers38(_cs))
    _named = {m for src, _ in _cs for m in _MARK38.findall(src)}
    if _named - _have:
        _dangling38[_k] = sorted(_named - _have)
check("D. every marker a cell NAMES exists as a cell: a printed instruction that sends the "
      "reader to a deleted cell is worse than no instruction",
      not _dangling38, "; ".join(f"{k}: {v}" for k, v in _dangling38.items())
      or "assert messages, run-order lists and cross-references all resolve")

_src38 = {k: "\n".join(s for s, _ in v) for k, v in _CELLS38.items()}

check("E. neither notebook re-declares CENTRAL_SETTINGS: it is study-level fact, it lives in "
      "xai_subset, and a second copy is a second study nobody compares",
      not _re38.search(r"CENTRAL_SETTINGS\s*=\s*\{", _src38["posthoc"] + _src38["train"]),
      "the notebook binds the module's, it does not write the six tags again")

check("F. no sweep starts a NEW study by default: with the canonical tags in place a re-run "
      "prints RUN SKIP, and STUDY_TAG=None retrains the sweep beside the canonical one",
      not _re38.search(r"STUDY_TAG\s*=\s*None", _src38["train"]),
      "six sweeps, six tags on disk")

_lore38 = [i for i, (src, _) in enumerate(_CELLS38["posthoc"]) if "LORE_SA_DEVICE" in src]
check("G. exactly one post-hoc cell arms the LORE engine: two launchers over two populations "
      "wrote the same xai/lore_sa/ and differed by 40 GPU hours",
      len(_lore38) == 1, f"cells {_lore38}")

check("H. the post-hoc reaches the aggregate door only through stamp_aggregate, never through "
      "assert_same_population alone: a guarded table with no provenance beside it",
      "assert_same_population" not in _src38["posthoc"],
      "one door, both duties")


# =============================================================================
shutil.rmtree(TMP, ignore_errors=True)
n_ok = sum(1 for _, o, _ in RESULTS if o)
print(f"\n{'=' * 74}")
print(f"TOTAL: {n_ok}/{len(RESULTS)} PASS")
for name, o, det in RESULTS:
    if not o:
        print(f"  ✗ {name}  {det}")
sys.exit(0 if n_ok == len(RESULTS) else 1)
