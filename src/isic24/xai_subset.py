"""The XAI evaluation sample.

⚠️ READ FIRST: THIS MODULE'S ORIGINAL MOTIVATION WAS WRONG
-----------------------------------------------------------
It was written (31/07) to correct a supposed prefix bias: "the first 32 samples are 40.6%
malignant against 9.1% in validation". **That 9.1% is wrong**: it came from
`neg_sample = 0.01` without reading the next line, `pos_sample = 5.0` -- positives are
upsampled 5x. True prevalence is **0.334**, and a prefix at 0.406 over n=32 gives
**P = 0.245**, i.e. noise. On top of that `df.sample(frac=1.0)` shuffles before the fold
split, so "the first N rows" are already a random sample.

**There is no prefix bias to correct.** The module stays because it still does two useful
things -- exact class quotas and a verifiable sample identity -- but it must not be sold
as a bias fix.

THE REAL PROBLEM, THAT ONE IS REAL
----------------------------------
`pos_sample=5.0` uses `replace=True`: 364 unique positives become 1820 rows. In the
stratified n=256 sample there are **222 unique lesions** over 256 rows, and among the 85
positives only **51 unique lesions** (multiplicity up to 3). The XAI explains the same
lesion several times: the samples are not independent and **every per-run confidence
interval is optimistic**.

The remedy is to deduplicate on row identity, not to stratify: `unique_lesion_subset`
(+ `adopt`), in production since 31/07 with n=144 distinct lesions, balance='equal'.

⚠️ AND THE PREFIX OF *THIS* SAMPLE REALLY IS IMBALANCED (measured 2026-08-01)
-----------------------------------------------------------------------------
This does not contradict the above: the prefix of `validation_df` really does sit at 0.336.
But `unique_lesion_subset` keeps the FIRST occurrence of each lesion, which for a positive
is the minimum of ~5 copies (`pos_sample=5.0`) and for a negative a single draw. Sorting
the indices pushes positives to the front: median index 160 on positives against 581 on
negatives. In the n=144 sample (50/50 by construction) the first 48 rows are **35 pos /
13 neg**.

So: to explain a subset, pass an EXPLICIT balanced index array; never let
`max_explain_samples` truncate. Comparisons between strategies stay valid (every run sees
the same samples); absolute levels and between-class contrasts do not.

WHERE THE ARTIFACTS GO (since 2026-08-01)
-----------------------------------------
Inside the run, in their own subfolders: `<run_dir>/xai/shap_tab/` and
`<run_dir>/xai/lore_sa/` (see `out_dir_for`). Only what is cross-run -- the between-family
tables and figures, `report()` -- lives in `runs/_faithfulness/`.

WHAT REMAINS VALID
------------------
The fingerprint (`subset_fingerprint`) and the guard that follows from it in
`xai_faithfulness`: the SHAP npz store `explained_indices` as POSITIONS, so reusing them
under a different sample pairs the SHAP values of some rows with the features of other
lesions, with no error and with plausible numbers. That risk is real whichever sample is
chosen.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from . import paths as _paths

# PER-RUN artifacts live inside the run (see `out_dir_for` below). What is left here is the
# destination of what is CROSS-RUN and therefore belongs to none of them: the between-family
# tables and figures.
AGGREGATE_DIRNAME = "_faithfulness"


def stratified_subset(labels, n, seed=0):
    """POSITIONAL indices into `validation_df` of a class-stratified sample.

    Proportional: the sample prevalence reproduces the pool's (up to rounding).
    Deterministic given `seed`. Returns SORTED indices, because `_build_faith_subset` reads
    the dataset in order and then indexes into it.

    NB it corrects no prefix bias -- there is none, see the module docstring. It guarantees
    EXACT class quotas rather than quotas in distribution, which removes one source of
    variance between samples without moving the expected value.
    """
    y = np.asarray(labels).reshape(-1)
    n = int(n)
    if n > len(y):
        raise ValueError(f"n={n} > {len(y)} samples available in the validation set")
    rng = np.random.default_rng(seed)
    out = []
    classes = np.unique(y)
    # Proportional quota, then the remainders are handed out from the largest class down:
    # deterministic instead of dependent on the class iteration order.
    quotas = {}
    for c in classes:
        idx_c = np.flatnonzero(y == c)
        quotas[c] = int(np.floor(len(idx_c) * n / len(y)))
    leftover = n - sum(quotas.values())
    for c in sorted(classes, key=lambda c: -int(np.sum(y == c))):
        if leftover <= 0:
            break
        quotas[c] += 1
        leftover -= 1
    for c in classes:
        idx_c = np.flatnonzero(y == c)
        k = min(quotas[c], len(idx_c))
        out.append(rng.choice(idx_c, size=k, replace=False))
    idx = np.sort(np.concatenate(out))
    assert len(idx) == n, f"quotas do not add up: {len(idx)} != {n}"
    return idx


def unique_lesion_subset(ids, labels, n=None, seed=0, balance="pool"):
    """POSITIONAL indices over **distinct lesions**: one row per identity, never two.

    This is the remedy to the real problem: `pos_sample=5.0` with `replace=True` makes the
    same lesion appear ~5 times, and the copies are EXACT replicas (same image, same
    features, same prediction). Explaining them repeatedly adds no information, it only
    narrows the intervals.

    ``balance``:
      * ``"pool"``  -- prevalence proportional to that of the UNIQUE lesions;
      * ``"equal"`` -- half per class. Maximizes the power of the between-class contrast at
        equal cost; the absolute level is no longer the population's, but stays comparable
        between strategies because the sample is the same for all of them;
      * ``"all"``   -- every distinct lesion, no sampling (``n`` ignored).
    ``n=None`` means the largest sample this split allows: every distinct lesion for
    ``"pool"``, twice the smaller class for ``"equal"``.

    Raises when ``n`` is not attainable, saying what the maximum is: the worst way to fail
    here would be to silently return fewer positives than asked for.
    """
    ids = np.asarray(ids)
    y = np.asarray(labels).reshape(-1)
    if len(ids) != len(y):
        raise ValueError(f"ids ({len(ids)}) and labels ({len(y)}) have different lengths")

    # One representative per lesion: the FIRST occurrence. Deterministic and independent of
    # the seed, so two samples with different seeds draw from the same lesions.
    _, first = np.unique(ids, return_index=True)
    rep = np.sort(first)
    y_rep = y[rep]
    classes = np.unique(y_rep)
    avail = {int(c): int(np.sum(y_rep == c)) for c in classes}

    if balance == "all":
        return rep, {"balance": "all", "n": len(rep), "per_class": avail,
                     "n_unique_available": {int(c): v for c, v in avail.items()}}
    
    if n is None:               # the largest sample this split allows, given the balance
        n = len(rep) if balance == "pool" else min(avail.values()) * len(classes)
    n = int(n)

    if balance == "equal":
        k = n // len(classes)
        want = {int(c): k for c in classes}
        for c in list(want)[:n - k * len(classes)]:
            want[c] += 1
    elif balance == "pool":
        want = {int(c): int(np.floor(avail[int(c)] * n / len(rep))) for c in classes}
        left = n - sum(want.values())
        for c in sorted(want, key=lambda c: -avail[c]):
            if left <= 0:
                break
            want[c] += 1
            left -= 1
    else:
        raise ValueError(f"balance={balance!r}: expected 'pool', 'equal' or 'all'")

    short = {c: (want[c], avail[c]) for c in want if want[c] > avail[c]}
    if short:
        # The attainable maximum depends on the balance:
        #   'pool'  -> proportional over unique lesions: reaches ALL of them (n<=len(rep));
        #   'equal' -> bounded by the smallest class, 2x that (here the unique positives).
        n_max = len(rep) if balance == "pool" else min(avail.values()) * len(classes)
        raise ValueError(
            f"n={n} not attainable over distinct lesions with balance={balance!r}.\n"
            f"  unique lesions per class: {avail}  (total {len(rep)})\n"
            f"  it would need {dict(want)}\n"
            f"  maximum n with this balance: {n_max}\n"
            f"  alternatives: n<={n_max}, balance='all' (all {len(rep)} lesions), "
            f"or balance='pool' if you are using 'equal'.")

    rng = np.random.default_rng(seed)
    out = []
    for c in classes:
        pool_c = rep[y_rep == c]
        out.append(rng.choice(pool_c, size=want[int(c)], replace=False))
    idx = np.sort(np.concatenate(out))
    assert len(np.unique(ids[idx])) == len(idx), "the sample holds repeated lesions"
    return idx, {"balance": balance, "n": len(idx),
                 "per_class": {int(c): int(np.sum(y[idx] == c)) for c in classes},
                 "n_unique_available": avail}


def lore_subset(labels, subset_idx, n):
    """The `n` rows of the adopted sample LORE explains: half per class, evenly spaced.

    LORE costs about 30 minutes a run, so it covers a subset of the sample every other
    engine explains whole. That subset must not be a prefix: `unique_lesion_subset` keeps
    the FIRST occurrence of each lesion and the positives are upsampled, so the front of the
    sample is where they sit -- in the n=144 sample the first 48 rows are 35 positive
    against 13 negative. Evenly spaced inside each class instead: balanced by construction,
    deterministic, and identical for every run.

    `labels`     the label of every validation row; `subset_idx` positions into it.
    Returns positions in the SAME space as `subset_idx`, sorted.
    """
    y = np.asarray(labels).reshape(-1)
    idx = np.asarray(subset_idx, int)
    if n % 2:
        raise ValueError(f"n={n}: the LORE subset is half per class, so it must be even")
    half, pick = n // 2, []
    for c in (0, 1):
        pool = idx[y[idx] == c]
        if len(pool) < half:
            raise ValueError(f"class {c}: {len(pool)} lesions in the adopted sample of "
                             f"{len(idx)}, {half} needed for n={n}")
        pick.append(pool[np.linspace(0, len(pool) - 1, half).round().astype(int)])
    return np.sort(np.concatenate(pick))


def subset_fingerprint(ids_all, idx, seed, n, kind="stratified"):
    """The sample's identity. This is what stops two different orderings being mixed.

    It includes the row ids IN THE SAMPLE'S ORDER, not just the count: two samples of the
    same size over the same validation set must have different fingerprints.
    """
    ids = [str(ids_all[i]) for i in np.asarray(idx, int)]
    h = hashlib.sha256(("|".join(ids) + f"|{kind}|{seed}|{n}").encode()).hexdigest()[:16]
    return f"{kind}-s{seed}-n{n}-{h}"


# The fingerprint of the HISTORICAL behaviour: the first N rows. Every canonical artifact
# was produced that way and carries no stamp, so "no stamp" == this.
PREFIX_FINGERPRINT = "prefix"


def save_snapshot(path, ids_all, idx, seed, n, kind="stratified", extra=None):
    """Write the sample to disk. The IDs, not only the indices: indices alone do not
    survive a recomputation of the fold."""
    fp = subset_fingerprint(ids_all, idx, seed, n, kind=kind)
    snap = {
        "fingerprint": fp, "kind": kind, "seed": int(seed), "n": int(n),
        "positional_idx": [int(i) for i in idx],
        "row_ids": [str(ids_all[i]) for i in np.asarray(idx, int)],
        **(extra or {}),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(snap, f, indent=1)
    return fp


def load_snapshot(path):
    with open(path) as f:
        return json.load(f)


def verify_snapshot(snap, ids_all):
    """Does the saved sample STILL describe this validation set?

    If the fold is rebuilt differently, the positional indices point at other lesions: this
    raises instead of producing plausible, wrong numbers.
    """
    idx = np.asarray(snap["positional_idx"], int)
    if int(np.max(idx)) >= len(ids_all):
        raise RuntimeError(
            f"snapshot out of range: max index {int(np.max(idx))} over {len(ids_all)} "
            f"validation rows. The fold has changed: regenerate the sample.")
    have = [str(ids_all[i]) for i in idx]
    # `isic_ids`: the key of snapshots written before row identity had a neutral name.
    want = list(snap.get("row_ids", snap.get("isic_ids", [])))
    if have != want:
        n_bad = sum(a != b for a, b in zip(have, want))
        raise RuntimeError(
            f"snapshot does NOT match the current validation set: {n_bad}/{len(idx)} ids "
            f"differ at the same positions. The artifacts already produced under fingerprint "
            f"{snap['fingerprint']} are NOT comparable with this session.")
    return snap["fingerprint"]


# --------------------------------------------------------------------------------------
# WHERE THE SAMPLE'S ARTIFACTS GO  (redone 2026-08-01: "every artifact of a run must live
# in that run's own folder")
#
# There used to be a parallel root `runs/_subset/<run_id__>/`, mirroring the run tree and
# keeping these artifacts away from the canonical ones. It did keep them apart, but it
# spread one run over two places and was not the structure to carry to new datasets.
#
# Now they are written INSIDE the run, in subfolders no canonical engine uses:
#
#     <run_dir>/xai/shap_tab/   aligned tabular attribution + signed faithfulness
#     <run_dir>/xai/lore_sa/    LORE-SA (per-sample rules)
#
# Separation from the canonical artifacts comes from the folder NAME, not from distance:
# `xai/shap/`, `xai/gradcam/`, `xai/lore/` and `xai/faithfulness/` stay untouched and no new
# write can land in them (cf. the SUBSET GUARD in `xai_faithfulness`, which keeps refusing
# `xai/faithfulness/` while a sample is adopted).
#
# The resume stamp lives in the same subfolder as the artifact, so the two resumes are
# independent by construction and the `lore-` prefix on the fingerprint -- which only served
# to separate them when they shared a directory -- is no longer needed.
# --------------------------------------------------------------------------------------
SUBSET_KINDS = ("shap_tab", "lore_sa", "gradcam_img")

# The per-engine folders, and the "tasks" each one hosts. `shap_tab` holds TWO of them (the
# SHAP arrays and the signed faithfulness), so its stamp has to say which.
SUBSET_TASKS = {"shap_tab": ("faith", "shap"), "lore_sa": ("lore",),
                "gradcam_img": ("gradcam",)}


# Re-rooting a `run_dir` lives in `paths`, the paths module: what stays here is the name the
# rest of the code already calls it by. It used to be the same line copied six times.
run_dir_abs = _paths.run_dir_abs


def out_dir_for(run_dir, kind):
    """Where the `kind` artifacts of this run go. `kind` in SUBSET_KINDS."""
    if kind not in SUBSET_KINDS:
        raise ValueError(f"kind={kind!r}: expected {SUBSET_KINDS}")
    return os.path.join(run_dir, "xai", kind)


def stamp_path(run_dir, kind, fp, task=None):
    """The file saying 'this task is already done UNDER THIS fingerprint'.

    ⚠️ `task=None` must produce EXACTLY `DONE.<fp>.json`, the name already carried by the
    110 faithfulness stamps and the 8 LORE ones. Changing it would mean redoing six hours of
    faithfulness on the next launch, with nobody noticing until the GPU is busy. `task`
    exists only for the second task of `shap_tab` (the SHAP arrays), which shares the folder
    with the first.
    """
    name = f"DONE.{fp}.json" if task is None else f"DONE.{task}.{fp}.json"
    return os.path.join(out_dir_for(run_dir, kind), name)


def is_done(run_dir, kind, fp, task=None):
    return os.path.exists(stamp_path(run_dir, kind, fp, task))


def mark_done(run_dir, kind, fp, payload=None, task=None):
    """Written ONLY once the computation is finished.

    This is why interrupting is safe: a run cut in half leaves no stamp, so on restart it is
    redone from scratch. The granularity of a resume is the RUN, not the sample."""
    p = stamp_path(run_dir, kind, fp, task)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"run_dir": run_dir, "kind": kind, "task": task or "default",
                   "fingerprint": fp, **(payload or {})}, f, indent=1)
    return p


def resume_report(run_dirs, kind, fp, task=None):
    """(todo, done) — to be printed BEFORE launching, so the cost is known in advance.

    `run_dirs` is a dict {run_id: run_dir} (or an iterable of pairs): resuming works per run,
    and the keys come back so the caller prints run_ids rather than paths.
    """
    items = list(run_dirs.items() if hasattr(run_dirs, "items") else run_dirs)
    done = [rid for rid, d in items if is_done(d, kind, fp, task)]
    todo = [rid for rid, d in items if not is_done(d, kind, fp, task)]
    return todo, done


# The work plan: which engine writes where, under which stamp, and what it costs.
# `sec_per_sample` are measurements, not estimates: SHAP 30/07, LORE 02/08 over 6 runs on CPU
# (20.0 min at n=48; on GPU it was 22, see `xai_driver.LORE_SA_DEVICE`). Grad-CAM stays None
# until it is measured on the first run.
JOBS = {
    "shap":    dict(kind="shap_tab",    task="shap",  engines=("shap",),    sec_per_sample=1.2),
    "lore":    dict(kind="lore_sa",     task=None,    engines=("lore",),    sec_per_sample=25.0),
    "gradcam": dict(kind="gradcam_img", task=None,    engines=("gradcam",), sec_per_sample=None),
}


# --------------------------------------------------------------------------------------
# Statistics with the RUN as the unit
#
# Every XAI run sees the SAME validation sample. So 10 runs x 32 samples are NOT 320
# observations: they are 32 lesions seen by 10 models. Pooling rows across runs violates
# independence in the worst way and the resulting p-value is optimistic.
#
# It cost dearly on 31/07: class-wise LORE gave a pooled chi2 p=0.0016 ("the most solid
# result"); with the run as the unit it becomes Wilcoxon p=0.047 and a sign test p=0.75 --
# suggestive, not established.
# --------------------------------------------------------------------------------------

def across_runs(per_run_a, per_run_b, name_a="A", name_b="B", drop_ties=True):
    """Paired contrast between two quantities measured on every run.

    One value per RUN, not per sample. Returns a sign test and a Wilcoxon: the first uses
    the direction only, the second the magnitudes as well, and they are read together --
    when they diverge, the effect lives in a few runs with large differences. `p_floor` is
    the smallest p the number of units allows, so a p sitting on it means every unit agreed
    and the test has nothing left to give.

    ``drop_ties`` removes the runs whose difference is exactly zero. They must be removed
    AND DECLARED: typically they are saturated by construction (coverage 0 vs 0, or 1 vs 1)
    and cannot show the effect, yet the sign test would count them as non-positive.
    """
    from scipy import stats

    a = np.asarray(per_run_a, float)
    b = np.asarray(per_run_b, float)
    if a.shape != b.shape:
        raise ValueError(f"paired per-run values are required: {a.shape} vs {b.shape}")
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = b - a

    inf = d != 0 if drop_ties else np.ones_like(d, bool)
    n_inf = int(inf.sum())
    res = {
        "n_run": int(d.size), "n_informative": n_inf, "n_ties": int(d.size - n_inf),
        "n_positive": int((d > 0).sum()),
        "diff_median": float(np.median(d)) if d.size else float("nan"),
        "diff_mean": float(np.mean(d)) if d.size else float("nan"),
        "name_a": name_a, "name_b": name_b,
    }
    res["sign_p"] = (float(stats.binomtest(int((d[inf] > 0).sum()), n_inf, 0.5).pvalue)
                     if n_inf else float("nan"))
    res["wilcoxon_p"] = (float(stats.wilcoxon(d[inf]).pvalue) if n_inf >= 6 else float("nan"))
    # The smallest p this many units can reach, sign test and Wilcoxon alike: the extreme
    # outcome has probability 2^-n in each tail. It belongs beside the p-value, not in a
    # sentence written by hand next to the table.
    res["p_floor"] = float(min(1.0, 2.0 ** (1 - n_inf))) if n_inf else float("nan")
    if n_inf < 6:
        # With fewer than 6 informative runs Wilcoxon cannot go below 0.05 even when every
        # difference agrees in sign: returning it would suggest a test that is not there.
        res["note"] = f"only {n_inf} informative runs: Wilcoxon not applicable (>=6 needed)"
    return res


def print_across_runs(res, indent="  "):
    """The same figures in readable form, with the caveats attached rather than footnoted."""
    print(f"{indent}{res['name_b']} vs {res['name_a']} — unit = run, not sample")
    print(f"{indent}  runs: {res['n_run']} total, {res['n_informative']} informative, "
          f"{res['n_ties']} at zero difference (saturated/degenerate, excluded from the tests)")
    print(f"{indent}  in favour of {res['name_b']}: {res['n_positive']}/{res['n_run']}")
    print(f"{indent}  median difference {res['diff_median']:+.3f} (mean {res['diff_mean']:+.3f})")
    print(f"{indent}  sign test       p = {res['sign_p']:.4f}"
          if res['sign_p'] == res['sign_p'] else f"{indent}  sign test       n/a")
    if res.get("note"):
        print(f"{indent}  Wilcoxon        n/a — {res['note']}")
    else:
        print(f"{indent}  Wilcoxon        p = {res['wilcoxon_p']:.4f}")
    if res['p_floor'] == res['p_floor']:
        print(f"{indent}  floor at n = {res['n_informative']}: p = {res['p_floor']:.4f}, "
              "unreachable from below whatever the effect size")
    if res['n_run'] < 8:
        print(f"{indent}  ⚠ {res['n_run']} runs: any p here is fragile. More seeds, not more samples.")


# --------------------------------------------------------------------------------------
# The single injection point
# --------------------------------------------------------------------------------------

def adopt(validation_df, n=None, balance="pool", seed=0, snapshot_dir=None,
          apply_to=("faithfulness", "lore", "shap"), verbose=True):
    """Pick the XAI sample **once per session** and make it everyone's default.

    Call it right after `xf.configure(...)`, at the SAME two points in the two notebooks.
    From there on no cell passes `subset_idx`: faithfulness inherits it from
    `xf.FAITH_SUBSET_IDX`, LORE and SHAP from their kwargs in the driver. That is what makes
    the choice a framework default instead of an argument to remember.

    It samples over **distinct lesions** (`unique_lesion_subset`): `pos_sample` with
    `replace=True` makes the same lesion appear several times, and the copies are exact
    replicas that narrow the intervals without adding information.

    Returns ``(idx, info)``. ``adopt(validation_df, balance=None)`` disarms everything and
    restores the historical prefix, which is how the published numbers are reproduced.
    """
    from . import xai_driver as xd
    from . import xai_faithfulness as xf

    global ACTIVE_SUBSET_SNAPSHOT
    if balance is None:                                  # restore the historical behaviour
        xf.FAITH_SUBSET_IDX = None
        xf.ACTIVE_SUBSET_FINGERPRINT = None
        ACTIVE_SUBSET_SNAPSHOT = None
        xd.LORE_SA_KWARGS.pop("subset_idx", None)
        xd.SHAP_KWARGS.pop("subset_idx", None)
        xd.SHAP_KWARGS.pop("subset_fingerprint", None)
        if verbose:
            print("[sample] DISARMED — back to the first N rows (historical behaviour).")
        return None, {"balance": None}

    ids = register_validation_ids(validation_df)     # artifacts take their identity from here
    y = validation_df.target.values
    idx, info = unique_lesion_subset(ids, y, n=n, seed=seed, balance=balance)

    kind = f"unique-{balance}"
    snapshot_dir = snapshot_dir or "."
    path = os.path.join(snapshot_dir, f"subset.{kind}.n{len(idx)}.seed{seed}.json")
    if os.path.exists(path):
        snap = load_snapshot(path)
        fp = verify_snapshot(snap, ids)          # raises if the fold is no longer the same
        idx = np.asarray(snap["positional_idx"], int)
    else:
        fp = save_snapshot(path, ids, idx, seed, len(idx), kind=kind, extra={"info": info})
    ACTIVE_SUBSET_SNAPSHOT = path

    if "faithfulness" in apply_to:
        xf.FAITH_SUBSET_IDX = idx
        xf.ACTIVE_SUBSET_FINGERPRINT = fp
    if "lore" in apply_to:
        xd.LORE_SA_KWARGS["subset_idx"] = idx
    if "shap" in apply_to:
        # max_explain_samples MUST cover the sample, otherwise `_subsample` redraws a random
        # subset of it and the npz no longer matches the fingerprint stamped on it.
        xd.SHAP_KWARGS["subset_idx"] = idx
        xd.SHAP_KWARGS["subset_fingerprint"] = fp
        xd.SHAP_KWARGS["max_explain_samples"] = len(idx)
        xd.SHAP_KWARGS["max_samples"] = int(idx.max()) + 1
        xd.SHAP_KWARGS["max_batches"] = None

    if verbose:
        n_u = len(set(ids[idx]))
        print(f"[sample] {len(idx)} DISTINCT lesions (balance={balance!r}, seed={seed})")
        print(f"         per class: {info['per_class']}   prevalence {float(np.mean(y[idx])):.3f}")
        print(f"         unique available: {info['n_unique_available']}")
        print(f"         check: {n_u}/{len(idx)} distinct ids "
              f"{'OK' if n_u == len(idx) else '** DUPLICATES **'}")
        print(f"         fingerprint: {fp}")
        print(f"         active on: {', '.join(apply_to)}")
        print(f"         snapshot: {path}")
        print(f"         restore the historical prefix: xs.adopt(validation_df, balance=None)")
    return idx, {**info, "fingerprint": fp, "snapshot": path}


# --------------------------------------------------------------------------------------
# CROSS-RUN reporting — the between-family tables and figures
#
# It lives here, not in a notebook cell, because of the 2026-08-01 request: "I want the XAI
# to draw the plots directly, not to have to add more cells". The faithfulness and LORE
# loops call it at the end; no extra cell to remember to run.
#
# PER-RUN figures are drawn by the engines (`xai_faithfulness._plot_aligned_signed_run`,
# `xai_lore_sa._plot_lore_sa_run`) inside the run. What lands here is only what by
# definition belongs to no single run.
# --------------------------------------------------------------------------------------
SIGNED_MODES_REPORT = ("posev", "negev", "target")   # `abs` is the artifact mode: ~0, noise
Q_CLEAN = 0.20                                       # the full budget is OOD-dominated


# --------------------------------------------------------------------------------------
# Unimodal ablations — the rule, not a note in a document
# --------------------------------------------------------------------------------------
# At the endpoints of a sweep a branch is OFF by construction: their explanation values are
# algebra, not measurement (`lore_coverage` exactly 0.000 and 1.000, SHAP mass exactly 0).
# They are ablations, not sweep points, and they enter no family mean. Whoever adds a family
# with endpoints declares it HERE: it is the only place the aggregations consult.
ABLATION_SETTINGS = {
    ("fixed_scaling", "a0p000"): "tabular",     # alpha=0 -> tabular branch off
    ("fixed_scaling", "a1p000"): "image",       # alpha=1 -> image branch off
}

# The one setting per family the report compares across families: the centre of each sweep.
# Declared here, next to the ablations, because it is the same kind of study-level fact and
# because the notebook, the tables and the figures must all read one definition.
CENTRAL_SETTINGS = {
    "vanilla": "baseline", "fixed_scaling": "a0p500",
    "learnable_scaling": "init_a0p500", "loss_regularization": "lam1e-03",
    "controlled_fusion": "base", "adaptive_gating": "base",
}


def branch_off(impl_name, setting_tag):
    """Which branch this ablation switches off, or ``None`` if the setting is not one."""
    return ABLATION_SETTINGS.get((str(impl_name), str(setting_tag)))


def is_ablation(impl_name, setting_tag):
    return branch_off(impl_name, setting_tag) is not None


def drop_ablations(df, where="", verbose=True):
    """Remove the unimodal ablations from a per-run table before aggregating it.

    To be called at every point that does `groupby("impl_name")`. On ISIC it moves the AUC
    mean of `fixed_scaling` from 0.8974 to 0.9014, and its family `lore_coverage` from a
    value that contains 0.000 and 1.000 to one that does not.
    """
    if not len(df) or "impl_name" not in df.columns or "setting_tag" not in df.columns:
        return df
    keep = [not is_ablation(i, t) for i, t in zip(df["impl_name"], df["setting_tag"])]
    n = len(keep) - sum(keep)
    if n and verbose:
        print(f"  [ablations] {where}: {n} unimodal endpoint runs excluded from the aggregate")
    return df[keep]


# The columns the registry is entitled to. A summary that does not carry them is not a
# summary of one of these runs, and it enters the registry as NaN rather than as an error.
REGISTRY_COLS = ("impl_name", "exp_tag", "setting_tag", "run_id", "seed", "best_epoch",
                 "best_val_auc", "run_dir", "model_name")


def _alpha_final(run_dir, best_epoch):
    """The learned alpha the best checkpoint holds, or None where the run logged no trajectory.

    The runner writes `single_run_summary.csv` before anything reads the trained layer back, so
    the learned alpha reaches disk only in `alpha_trajectory.csv` and in the study-level
    `combined_summary.csv` -- the aggregate this registry refuses. The row to take is
    `best_epoch`: `restore_best_weights` lets training run past it, and the last row is then the
    alpha of an epoch the checkpoint does not hold.
    """
    import pandas as pd

    traj = os.path.join(run_dir, "alpha_trajectory.csv")
    if not os.path.exists(traj):
        return None
    a = pd.read_csv(traj).set_index("epoch")["alpha"]
    if best_epoch not in a.index:
        raise RuntimeError(
            f"[REGISTRY] {traj}: epochs {a.index.min()}..{a.index.max()} do not cover "
            f"best_epoch={best_epoch}. Falling back to the last row would register the alpha "
            f"of an epoch the checkpoint does not hold.")
    return float(a.loc[best_epoch])


def build_registry(canonical, n_expected=None):
    """One row per run of the declared studies, read from the per-run summaries.

    `canonical`  {impl_name: study_dir}. Every run under it must be there: a study whose
                 glob is empty raises instead of leaving the family out of the registry, and
                 with it out of every table the registry feeds.

    The study-level `combined_summary.csv` is deliberately NOT the source. It is an
    aggregate written once per sweep -- one of the canonical five carries whitespace-padded
    headers (`"impl_name        "`, `" seed"`), which concatenate into a second set of
    columns and leave that family's `setting_tag` and `run_dir` at NaN, invisible to every
    `groupby`. The per-run summary is the artifact each run writes for itself, so a schema
    that is not the one the others carry is a fact about a single run and is named as such.

    A family that logs an `alpha_trajectory.csv` also gets `alpha_final`, the value its best
    checkpoint holds: the per-run summary is written before it exists.
    """
    import glob as _glob
    import pandas as pd

    frames = []
    for impl, study in canonical.items():
        paths = sorted(_glob.glob(os.path.join(study, "*", "*", "single_run_summary.csv")))
        if not paths:
            raise RuntimeError(
                f"[REGISTRY] {impl}: no run under {study}. A missing study used to drop out "
                f"of the registry in silence, taking its family with it.")
        for f in paths:
            df = pd.read_csv(f)
            missing = [c for c in REGISTRY_COLS if c not in df.columns]
            if len(df) != 1 or missing:
                raise RuntimeError(
                    f"[REGISTRY] {f}: {len(df)} rows, missing columns {missing}. A summary "
                    f"whose schema is not the one the others carry joins the registry with "
                    f"NaN keys and its family disappears from every groupby.")
            if df.impl_name.iloc[0] != impl:
                raise RuntimeError(
                    f"[REGISTRY] {f}: declares impl_name={df.impl_name.iloc[0]!r} under the "
                    f"study of {impl!r}")
            # `os.path.dirname(f)`, not the recorded `run_dir`: that one is the path of the
            # machine that trained, and the artifact being read is here.
            alpha = _alpha_final(os.path.dirname(f), int(df.best_epoch.iloc[0]))
            if alpha is not None:
                df["alpha_final"] = alpha
            frames.append(df)

    master = pd.concat(frames, ignore_index=True)
    if n_expected is not None and len(master) != n_expected:
        raise RuntimeError(
            f"[REGISTRY] {len(master)} runs over {len(canonical)} studies, {n_expected} "
            f"declared: " + ", ".join(f"{i}={sum(master.impl_name == i)}" for i in canonical))
    return master


ACTIVE_SUBSET_SNAPSHOT = None   # set by `adopt`; the sample every aggregate is computed under


def aggregate_provenance(**knobs):
    """What a cross-run aggregate was computed FROM: population, sample, encoding, knobs.

    The post-hoc twin of the training notebook's `run_provenance()`, and deliberately the same
    kind of content — data and parameters, not a code version, which neither half records.
    `run_id` says WHICH runs a table holds; nothing said under which sample and which knobs
    their numbers were produced, and that is recoverable only by opening the runs one by one.

    Everything here is session state already declared: `XaiContext` for the frames and the
    encoding, `adopt` for the sample. An unconfigured session raises through `_CTX` itself,
    which is the right answer: a table that cannot say what produced it is the defect.
    """
    from . import paths as pth
    from . import xai_faithfulness as xf

    ctx = xf._CTX
    idx = getattr(xf, "FAITH_SUBSET_IDX", None)
    return {
        "dataset": f"{pth.DATASET}/{pth.VARIANT}",
        "n_train": int(len(ctx.training_df)),
        "n_val": int(len(ctx.validation_df)),
        "n_features": len(ctx.feat_cols),
        "subset_fingerprint": getattr(xf, "ACTIVE_SUBSET_FINGERPRINT", None),
        "subset_n": None if idx is None else int(len(idx)),
        "subset_snapshot": ACTIVE_SUBSET_SNAPSHOT,
        "feature_space_file": ctx.feature_space_file,
        "feature_space_fingerprint": ctx.feature_space_fingerprint,
        "faith_budgets": list(getattr(xf, "FAITH_BUDGETS", ())),
        "q_clean": Q_CLEAN,
        **knobs,
    }


def stamp_aggregate(path, run_ids, **knobs):
    """Guard the name, then record what produced the table — one door, both duties.

    They are separable only in the sense that a bug could separate them, which is how a
    guarded aggregate would end up with no provenance beside it. Called BEFORE the write, so
    `assert_same_population` still sees the file the previous population left.
    """
    assert_same_population(path, run_ids)
    prov = {"n_runs": len(set(str(r) for r in run_ids)),
            **aggregate_provenance(**knobs)}
    with open(os.path.splitext(path)[0] + ".provenance.json", "w") as f:
        json.dump(prov, f, indent=1, default=str)
    return prov


def assert_same_population(path, run_ids):
    """A cross-run aggregate never takes a name another population still holds.

    The file names of `AGGREGATE_DIRNAME` say what a table is and not which runs it is over,
    so re-running a section writes over the previous study's table under its own name. Nothing
    fails, and afterwards there is no way to tell which population a number came from: it is
    the two-populations-one-name defect at the aggregate level, where it is worse than at the
    run level because a run directory at least names itself.

    `impl_name`, `setting_tag` and `seed` cannot separate them -- every study here carries the
    same families, the same settings and overlapping seeds. Only `run_id` can, so a table that
    holds none is refused rather than trusted: an aggregate that cannot name its runs is not
    evidence about any of them. Pair tables carry `run_id_a`/`run_id_b`, and any column whose
    name starts with `run_id` counts.

    An overlap of one run is enough to pass: extending a study or redoing part of it writes
    over its own table, which is the intended use. What is refused is a table with nothing in
    common with the population being written.
    """
    import pandas as pd

    if not os.path.exists(path):
        return
    cols = [c for c in pd.read_csv(path, nrows=0).columns if c.startswith("run_id")]
    if not cols:
        raise RuntimeError(
            f"[AGGREGATE] {path} carries no run identity, so it cannot say which runs it "
            f"describes — family, setting and seed are shared by every study run here. "
            f"Archive it before this one takes its name.")
    prev = set()
    for c in cols:
        prev |= set(pd.read_csv(path, usecols=[c])[c].dropna().astype(str))
    if not (prev & {str(r) for r in run_ids}):
        raise RuntimeError(
            f"[AGGREGATE] {path} describes {len(prev)} runs and not one of them is in this "
            f"population: it is another study's table under the name this one is about to "
            f"take. Archive it before this one takes its name.")


def _run_gains(run_dir, prefix="faithfulness_aligned_signed", modes=("abs",) + SIGNED_MODES_REPORT):
    """Every gain of a run, WITH the sign conventions already applied.

    Never from the JSON scalar: `faithfulness_tabular_del_gain` is PRE-convention (`negev`
    must be negated, `target` signed per sample before the mean). It goes through the npz,
    which carry the `labels` as well — so validation_df is not needed.
    """
    from . import xai_faithfulness as xf
    d = out_dir_for(run_dir, "shap_tab")
    rec = {}
    for mode in modes:
        p = os.path.join(d, f"{prefix}_{mode}_curves.npz")
        if not os.path.exists(p):
            return None
        with np.load(p) as z:
            labels = z["labels"]; sgn = xf.mode_sign(mode, labels)
            rec["n"] = len(labels); rec["pos_rate"] = float(labels.mean())
            for direction in ("del", "ins"):
                g = xf.persample_gains(z, "tabular", direction, q_max=None)
                rec[f"{mode}_{direction}"] = float(np.mean(g * sgn))
                gq = xf.persample_gains(z, "tabular", direction, q_max=Q_CLEAN)
                rec[f"{mode}_{direction}_q{int(Q_CLEAN * 100)}norm"] = (
                    float(np.mean(gq * sgn)) / xf.gain_window(z, Q_CLEAN))
    return rec


def _lore_quality(run_dir):
    """LORE-SA quality of a run, with the per-class breakdown.

    `*_5050` = mean of the two class means. It is the column to quote when the sample is
    unbalanced: it neutralizes the composition without throwing samples away.
    """
    import glob as _glob
    import pandas as pd
    hits = sorted(_glob.glob(os.path.join(out_dir_for(run_dir, "lore_sa"), "epoch*",
                                          "lore_sa_persample_*.csv")))
    if not hits:
        return None
    d = pd.read_csv(hits[-1])
    d["covf"] = (d.neigh_n_classes > 1).astype(float)
    d["rule_len_covered"] = d["rule_len"].where(d["covf"].astype(bool))
    g = d.groupby("true_label").agg(c=("covf", "mean"), r=("rule_len_covered", "mean"))
    return dict(n=len(d), pos_rate=float(d.true_label.mean()),
                n_pos=int(d.true_label.sum()), n_neg=int((1 - d.true_label).sum()),
                rng_protocol="seeded-v2" if "rng_seed" in d else "unseeded-v1",
                lore_coverage=d["covf"].mean(), lore_coverage_5050=g["c"].mean(),
                rule_len=d["rule_len_covered"].mean(),
                rule_len_5050=g["r"].reindex([0, 1]).mean(skipna=False),
                cov_y0=g["c"].get(0, np.nan), cov_y1=g["c"].get(1, np.nan),
                rl_y0=g["r"].get(0, np.nan), rl_y1=g["r"].get(1, np.nan),
                neigh_pos_rate=d["neigh_pos_rate"].mean(),
                fid_insample=d["fid_insample"].mean(),
                n_warning=int(d.warning.notna().sum()), n_error=int(d.error.notna().sum()))


def report(master, art_path, kinds=("shap_tab", "lore_sa"), central=None, tag="",
           out_dir=None, verbose=True, encoded_feature_names=None, lore_idx=None,
           shape_metrics=True, shape_top_k=None):
    """Between-family tables and figures, from the artifacts already on disk. Zero GPU.

    `master`  the `master_results_rows.csv` DataFrame (columns impl_name, setting_tag,
              seed, run_id, run_dir).
    `central` {impl_name: setting_tag} of the representative settings. **Needed**: the
              master holds 45 `fixed_scaling` rows, `a0p000` (degenerate, all zeros)
              included, and a plain `groupby("impl_name")` blends them all into the family
              mean. None = every run is used, and the warning is printed.
    `tag`     forces the file suffix. Default: derived from each kind's own `n` (`_n144` for
              faithfulness, `_n48` for LORE), so the two cannot label themselves with the
              other's n.
    `shape_top_k` selects raw features for stability and agreement; defaults to TOP_K.
    `kinds=()` still reads per-run faithfulness and LORE for the shape-family summary.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from .xai_engines import _finalize_plot

    do_shape = shape_metrics and encoded_feature_names is not None
    if do_shape:
        from . import xai_metrics as xmet
        shape_top_k = xmet.TOP_K if shape_top_k is None else shape_top_k
        n_raw = len(xmet.raw_groups(encoded_feature_names))
        if not isinstance(shape_top_k, (int, np.integer)) or not 1 <= shape_top_k < n_raw:
            raise ValueError(f"shape_top_k must be an integer in [1, {n_raw}); got {shape_top_k}.")

    out_dir = out_dir or os.path.join(art_path, "runs", AGGREGATE_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)

    def _sfx(df):
        """A suffix DERIVED from the data, not from the caller.

        `shap_tab` and `lore_sa` run on different n (144 and 48): a single `tag` passed from
        outside labels one of them with the other's n, and the wrong file stays on disk when
        the next pass writes under the right name. If n is not unique across the runs it
        falls back to the caller's `tag`, which at least does not lie about a single value.
        """
        if tag:
            return f"_{tag}"
        ns = sorted(set(int(v) for v in df["n"])) if len(df) else []
        return f"_n{ns[0]}" if len(ns) == 1 else ""
    # the master carries the paths of the machine that wrote it: re-anchor them on art_path
    rd = {r.run_id: run_dir_abs(r.run_dir, art_path) for _, r in master.iterrows()}
    written, out = [], {}

    if "shap_tab" in kinds or do_shape:
        rows = []
        for _, r in master.iterrows():
            rec = _run_gains(rd[r.run_id])
            if rec is not None:
                rows.append(dict(impl_name=r.impl_name, setting_tag=r.setting_tag,
                                 seed=r.seed, run_id=r.run_id, **rec))
        faith = pd.DataFrame(rows)
        out["faith"] = faith
        if "shap_tab" in kinds and len(faith):
            sfx = _sfx(faith)
            p = os.path.join(out_dir, f"faithfulness_aligned_signed{sfx}.csv")
            stamp_aggregate(p, faith.run_id, aggregate="faithfulness_aligned_signed",
                            central=central, modes=list(SIGNED_MODES_REPORT))
            faith.to_csv(p, index=False); written.append(p)
            if central:
                sel = faith[[central.get(i) == t
                             for i, t in zip(faith.impl_name, faith.setting_tag)]]
            else:
                sel = drop_ablations(faith, "faithfulness", verbose)
                if verbose:
                    print("  ⚠ central=None: the family mean includes every sweep setting "
                          "(the unimodal ablations are excluded regardless)")
            col = f"target_del_q{int(Q_CLEAN * 100)}norm"
            order = sel.groupby("impl_name")[col].mean().sort_values().index.tolist()
            out["central"], out["order"] = sel, order

            cmap = dict(zip(order, plt.cm.viridis(np.linspace(.05, .85, len(order)))))
            fig, axes = plt.subplots(1, len(SIGNED_MODES_REPORT),
                                     figsize=(4.2 * len(SIGNED_MODES_REPORT), 4.4),
                                     sharey=True, squeeze=False)
            for ax, mo in zip(axes[0], SIGNED_MODES_REPORT):
                c = f"{mo}_del_q{int(Q_CLEAN * 100)}norm"
                for x, fam in enumerate(order):
                    v = sel.loc[sel.impl_name == fam, c].values
                    ax.scatter(np.full(len(v), x) + np.linspace(-.12, .12, max(len(v), 1)),
                               v, s=22, color=cmap[fam], alpha=.75, zorder=3)
                    if len(v):
                        ax.hlines(v.mean(), x - .3, x + .3, color=cmap[fam], lw=2.5, zorder=4)
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels([f.replace("_", "\n") for f in order], fontsize=8)
                ax.set_title(f"`{mo}`"); ax.grid(axis="y", alpha=.25)
            axes[0][0].set_ylabel(f"deletion gain_norm (q$\\leq${Q_CLEAN})")
            n_seed = sel.groupby("impl_name").size().min() if len(sel) else 0
            fig.suptitle(f"Aligned tabular faithfulness — {len(sel)} runs, "
                         f"{sel.impl_name.nunique()} families, min {n_seed} seeds", y=1.0)
            _finalize_plot(out_dir, f"faithfulness_by_family{sfx}.png")
            written.append(os.path.join(out_dir, f"faithfulness_by_family{sfx}.png"))

    if "lore_sa" in kinds or do_shape:
        rows = []
        for _, r in master.iterrows():
            rec = _lore_quality(rd[r.run_id])
            if rec is not None:
                rows.append(dict(impl_name=r.impl_name, setting_tag=r.setting_tag,
                                 seed=r.seed, run_id=r.run_id, **rec))
        lore = pd.DataFrame(rows)
        # LORE runs can share the SAME n over different samples (on 01/08 `vanilla` had
        # three at n=48, one on the prefix of the 144 and two on the balanced sample). If the
        # prevalence is not unique, the family mean blends different populations.
        if len(lore) and "pos_rate" in lore:
            for _f, _g in lore.groupby("impl_name"):
                if _g.pos_rate.round(3).nunique() > 1 and verbose:
                    print(f"  ⚠ {_f}: LORE runs on DIFFERENT samples "
                          f"(pos_rate {sorted(_g.pos_rate.round(3).unique())}) — "
                          f"the family mean blends populations, redo the old ones")
        out["lore"] = lore
        if "lore_sa" in kinds and len(lore):
            sfx = _sfx(lore)
            p = os.path.join(out_dir, f"lore_sa_quality{sfx}.csv")
            stamp_aggregate(p, lore.run_id, aggregate="lore_sa_quality", central=central)
            lore.to_csv(p, index=False); written.append(p)
            lo = (drop_ablations(lore, "lore", verbose)
                  .groupby("impl_name")[["lore_coverage", "lore_coverage_5050",
                                         "rule_len", "rule_len_5050"]].mean())
            lo = lo.reindex([f for f in out.get("order", lo.index) if f in lo.index])
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
            for ax, (raw, bal, name) in zip(axes, [
                    ("lore_coverage", "lore_coverage_5050", "lore_coverage"),
                    ("rule_len", "rule_len_5050", "rule_len")]):
                x = np.arange(len(lo))
                ax.bar(x - .2, lo[raw], .4, color="#b0b0b0",
                       label=f"raw (pos_rate {lore.pos_rate.mean():.2f})")
                ax.bar(x + .2, lo[bal], .4, color="#2b7bba", label="reweighted 50/50")
                ax.set_xticks(x)
                ax.set_xticklabels([f.replace("_", "\n") for f in lo.index], fontsize=8)
                ax.set_title(name); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
            unbal = abs(lore.pos_rate.mean() - .5) > .05
            fig.suptitle("LORE-SA by family" + (
                f" — ⚠ UNBALANCED sample ({int(lore.n_pos.mean())} pos / "
                f"{int(lore.n_neg.mean())} neg): ordering yes, levels no" if unbal
                else " — balanced sample"), y=1.0)
            _finalize_plot(out_dir, f"lore_sa_quality{sfx}.png")
            written.append(os.path.join(out_dir, f"lore_sa_quality{sfx}.png"))

    # ---- SHAPE metrics that are CROSS-RUN by nature -------------------------------------
    # Compactness and rule compactness are written by the engines as they compute
    # (`ShapCallback`, `LoreSACallback`). These two are not, and cannot be: stability
    # compares TWO runs and the SHAP<->LORE agreement crosses two explainers. They run here,
    # at the end of the loops, the first moment at which every run exists.
    if do_shape:
        shape = xmet.collect(master, art_path,
                             encoded_feature_names=encoded_feature_names, lore_idx=lore_idx,
                             k=shape_top_k)
        stab = xmet.stability_table(master, art_path, encoded_feature_names, k=shape_top_k)
        # Guarded before the loop writes anything: the pooled tables that follow are derived
        # from these two, so a population that is refused here never reaches them either.
        for df, name in ((shape, "xai_shape_metrics.csv"), (stab, "xai_stability.csv")):
            if len(df):
                stamp_aggregate(os.path.join(out_dir, name), master.run_id,
                                aggregate=name, lore_idx=lore_idx, shape_top_k=shape_top_k,
                                encoded_feature_names=list(encoded_feature_names))
        for df, name in ((shape, "xai_shape_metrics.csv"), (stab, "xai_stability.csv")):
            if len(df):
                p = os.path.join(out_dir, name)
                df.to_csv(p, index=False)
                written.append(p)
        out["shape"], out["stability"] = shape, stab
        out["by_family"] = xmet.summary_by_family(art_path, shape_df=shape, stab_df=stab,
                                                  central=central, out_dir=out_dir,
                                                  verbose=False, faith_df=faith, lore_df=lore)
        written.append(os.path.join(out_dir, "xai_summary_by_family.csv"))

    if verbose:
        print(f"[report] {len(written)} files in {out_dir}")
        for p in written:
            print("   ", os.path.basename(p))
    out["written"] = written
    return out


# --------------------------------------------------------------------------------------
# The indices of the explained rows — ONE function, two spaces kept apart
# --------------------------------------------------------------------------------------
# `explained_indices` is the position INSIDE the sample (0..n-1), whatever the sample is.
# `val_index` is the position in `validation_df`. The names are similar, they are integers
# of the same size, and crossing them does not raise: it has already produced three distinct
# defects. Whoever wants the indices asks HERE, and is told which space they live in.

def _artifact_column(npz_or_df, key, dtype):
    """One column from an npz or an already-read CSV, without knowing which it is."""
    keys = (npz_or_df.files if hasattr(npz_or_df, "files")
            else list(getattr(npz_or_df, "columns", [])))
    if key not in keys:
        return None
    return np.asarray(npz_or_df[key], dtype).ravel()


def read_explained_index(npz_or_df, allow_prefix=True):
    """``(indices into validation_df, provenance)`` from an npz or a per-sample CSV.

    ``provenance`` is ``"val_index"`` when the artifact records it, and ``"prefix"`` when it
    falls back on ``explained_indices``: those values coincide with validation positions
    ONLY on the historical artifacts, where the sample was the prefix. Returns
    ``(None, "unknown")`` when the artifact does not say which rows it explained — "I do not
    know" is not "does not match", and the caller decides.
    """
    vi = _artifact_column(npz_or_df, "val_index", int)
    if vi is not None:
        return vi, "val_index"
    if allow_prefix:
        ei = _artifact_column(npz_or_df, "explained_indices", int)
        if ei is not None:
            return ei, "prefix"
    return None, "unknown"


def read_explained_ids(npz_or_df):
    """The row ids an artifact explained, or ``None`` when it does not record them.

    This is the identity that survives a rebuild of the fold; positional indices do not.
    ``isic_id`` is the key of artifacts written before row identity had a dataset-independent
    name; it is still read, never written.
    """
    ids = _artifact_column(npz_or_df, ROW_ID_KEY, str)
    return ids if ids is not None else _artifact_column(npz_or_df, "isic_id", str)


# --------------------------------------------------------------------------------------
# Row identity — one pipeline, any dataset
# --------------------------------------------------------------------------------------
# `ID_COL` is the column identifying a row in the validation dataframe, and it changes with
# the dataset: `isic_id` on ISIC-2024, `image_id` on BRSET. It is declared ONCE per session,
# next to the other parameters; from there on no function names it.
#
# `ROW_ID_KEY` never changes: it is the name under which identity is written inside the
# artifacts, so an npz is readable without knowing which dataset it comes from.
ID_COL = "isic_id"
ROW_ID_KEY = "row_id"

# The ACTIVE validation of this session: the only source of identity when writing. It is
# filled by `adopt` and `xai_faithfulness.configure`, the two points at which a session
# declares which eval set it works on; the last declaration wins.
VALIDATION_ROW_IDS = None


def row_ids(df):
    """The identity column of a dataframe, whatever the dataset is."""
    if ID_COL not in getattr(df, "columns", []):
        raise KeyError(
            f"identity column {ID_COL!r} missing: the columns are {list(df.columns)[:8]}. "
            f"Declare xs.ID_COL next to the other session parameters.")
    return df[ID_COL].to_numpy()


def register_validation_ids(validation_df):
    """Declare which validation is active. Artifacts take their identity from here."""
    global VALIDATION_ROW_IDS
    VALIDATION_ROW_IDS = (None if validation_df is None
                          else np.asarray(row_ids(validation_df), str))
    return VALIDATION_ROW_IDS


def stamp_row_ids(val_index):
    """``{ROW_ID_KEY: ...}`` to write into an artifact, ``{}`` when no validation is known.

    The single point translating positions -> identity when writing. An out-of-range index
    raises: it means the sample and the registered validation come from two different folds,
    which is exactly the case where the written ids would be plausible and wrong.
    """
    if VALIDATION_ROW_IDS is None or val_index is None:
        return {}
    idx = np.asarray(val_index, int)
    if idx.size and int(idx.max()) >= len(VALIDATION_ROW_IDS):
        raise RuntimeError(
            f"stamp_row_ids: index {int(idx.max())} over a validation of "
            f"{len(VALIDATION_ROW_IDS)} rows. The sample does not belong to the registered "
            f"validation: regenerate it, or re-declare the context.")
    return {ROW_ID_KEY: VALIDATION_ROW_IDS[idx]}


def index_for_row_ids(ids, validation_df):
    """Positions in `validation_df` of these ids, with both ways of being wrong closed.

    Validation is resampled: 291 rows are duplicates, so id -> position is not injective. A
    duplicated or missing id raises instead of silently resolving to the first occurrence.
    """
    col = row_ids(validation_df)
    pos = {}
    dup = set()
    for i, v in enumerate(col):
        (dup.add(v) if v in pos else pos.setdefault(v, i))
    want = list(ids)
    missing = [v for v in want if v not in pos]
    ambiguous = [v for v in want if v in dup]
    if missing or ambiguous:
        raise KeyError(f"ids missing from validation: {missing[:5]} | "
                       f"duplicated (position undefined): {ambiguous[:5]}")
    return np.asarray([pos[v] for v in want], int)


# Where each engine leaves the artifact declaring which rows it explained, inside
# `out_dir_for(run_dir, kind)`. The most recent one is always the one read.
SUBSET_ARTIFACT_GLOB = {
    "lore_sa": os.path.join("epoch*", "lore_sa_persample_*.csv"),
    "shap_tab": os.path.join("epoch_*", "shap_arrays_epoch_*.npz"),
    "gradcam_img": "gradcam_arrays*_epoch_*.npz",
}

# The faithfulness curves explain rows too, but they live outside `SUBSET_KINDS`: that tuple
# drives the DONE stamps, whose file names must not move. So the populations are read from a
# map of their own, which extends the one above instead of replacing it.
POPULATION_KINDS = SUBSET_KINDS + ("faithfulness",)
_POPULATION_GLOB = {**SUBSET_ARTIFACT_GLOB, "faithfulness": "faithfulness_curves.npz"}


def _population_dir(run_dir, kind):
    if kind == "faithfulness":
        return os.path.join(run_dir, "xai", "faithfulness")
    return out_dir_for(run_dir, kind)


def read_sample_identity(run_dir, kind):
    """``(indices, row ids)`` of the most recent `kind` artifact; ``None`` where absent.

    `allow_prefix=False`: `explained_indices` lives in another space (position INSIDE the
    sample), and comparing it with validation indices always returned False on healthy
    artifacts.
    """
    import glob as _glob
    pat = _POPULATION_GLOB.get(kind)
    if pat is None:
        return None, None
    hits = sorted(_glob.glob(os.path.join(_population_dir(run_dir, kind), pat)))
    if not hits:
        return None, None
    if hits[-1].endswith(".csv"):
        import pandas as pd
        art = pd.read_csv(hits[-1])
        return read_explained_index(art, allow_prefix=False)[0], read_explained_ids(art)
    with np.load(hits[-1]) as z:
        return read_explained_index(z, allow_prefix=False)[0], read_explained_ids(z)


def _population_size(run_dir, kind):
    """How many rows the `kind` artifact explains, even when it does not say WHICH."""
    import glob as _glob
    pat = _POPULATION_GLOB.get(kind)
    hits = sorted(_glob.glob(os.path.join(_population_dir(run_dir, kind), pat or "")))
    if not hits:
        return None
    if hits[-1].endswith(".csv"):
        import pandas as pd
        return int(len(pd.read_csv(hits[-1])))
    with np.load(hits[-1]) as z:
        for key in ("labels", "val_index", ROW_ID_KEY):
            if key in z.files:
                return int(np.asarray(z[key]).ravel().size)
    return None


def sample_populations(run_dir, kinds=POPULATION_KINDS):
    """One row per explanation artifact of a run: WHICH rows it explains, and how many.

    The four engines write into four places and nothing forced them to agree, so a run can
    carry a Grad-CAM on one sample and a faithfulness curve on another. That is not
    hypothetical: the canonical faithfulness curves predate `adopt()` and sit on the first
    128 validation rows, which share 41 rows with the 144-lesion sample the
    other engines use. Nothing said so, because no artifact was ever read side by side.

    ``index`` is ``None`` for artifacts written before `val_index` existed: "unknown" is not
    "matches", and `assert_nested_samples` treats it as unknown.
    """
    import pandas as pd
    rows = []
    for kind in kinds:
        idx, ids = read_sample_identity(run_dir, kind)
        n = _population_size(run_dir, kind)
        if idx is None and n is None:
            continue
        rows.append(dict(kind=kind, n=len(idx) if idx is not None else n,
                         index=None if idx is None else np.sort(np.asarray(idx, int)),
                         has_ids=ids is not None))
    return pd.DataFrame(rows, columns=["kind", "n", "index", "has_ids"])


def assert_nested_samples(run_dir, kinds=POPULATION_KINDS, raise_on_unknown=True):
    """Every artifact of a run must explain a SUBSET of the largest population.

    Nesting, not equality: LORE explains 48 of the 144 lesions by design, and that is
    legitimate. What is not legitimate is a population that overlaps the reference only in
    part — two samples wearing one name, which is how a class split ends up computed on rows
    the rest of the study never explains.

    An artifact that does not record which rows it explained cannot be shown to be nested.
    With ``raise_on_unknown=False`` it is reported instead of raising, which is the only way
    to audit artifacts produced before `val_index` existed.
    """
    pops = sample_populations(run_dir, kinds)
    if pops.empty:
        return pops
    ref = pops.loc[pops["n"].idxmax()]
    problems, unknown = [], []
    for _, r in pops.iterrows():
        if r["index"] is None or ref["index"] is None:
            if r["n"] != ref["n"]:
                unknown.append(f"{r['kind']}: n={r['n']} vs {ref['n']}, rows unknown"
                               f" — it predates `val_index`, so nesting cannot be shown")
            continue
        extra = np.setdiff1d(r["index"], ref["index"])
        if extra.size:
            shared = len(np.intersect1d(r["index"], ref["index"]))
            problems.append(f"{r['kind']}: {extra.size} rows outside {ref['kind']}"
                            f" — {r['n']} explained, {shared} shared")
    bad = problems + (unknown if raise_on_unknown else [])
    if bad:
        raise RuntimeError(f"[SAMPLE] not one population — {bad[0]}\n  "
                           + "\n  ".join(bad[1:] + [f"reference: {ref['kind']} n={ref['n']}",
                                                    f"run_dir={run_dir}"]))
    return pops


def sample_audit(master, art_path, kinds=POPULATION_KINDS, verbose=True):
    """`sample_populations` over every run, plus the verdict — the audit form of the guard.

    Returns ``(table, ok)``. It reports instead of raising because the canonical artifacts
    are immutable: the mismatch they carry is a fact to state in the report, not a failure
    to fix in place. On a dataset explained in one session it is expected to come back clean.
    """
    import pandas as pd
    rows, failures = [], {}
    for _, r in master.iterrows():
        d = _paths.run_dir_abs(r["run_dir"], art_path)
        try:
            assert_nested_samples(d, kinds)
        except RuntimeError as e:
            failures[str(e).splitlines()[0]] = failures.get(str(e).splitlines()[0], 0) + 1
        for _, p in sample_populations(d, kinds).iterrows():
            rows.append(dict(impl_name=r["impl_name"], setting_tag=r["setting_tag"],
                             seed=r["seed"], kind=p["kind"], n=p["n"],
                             knows_rows=p["index"] is not None))
    t = pd.DataFrame(rows)
    ok = not failures
    if verbose:
        if not t.empty:
            print(t.groupby("kind").agg(runs=("n", "size"), n_min=("n", "min"), n_max=("n", "max"),
                                        knows_rows=("knows_rows", "sum")).to_string())
        for msg, n in sorted(failures.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>4} runs  {msg}")
        print(f"\n[SAMPLE AUDIT] every population nested in the sample: "
              f"{'PASS' if ok else 'FAIL'}")
    return t, ok


def sample_matches(run_dir, kind, expected_idx):
    """Were the `kind` artifacts computed ON THESE samples?  (True/False/None)

    The comparison goes through row ids when the artifact carries them and the active
    validation is registered: two folds rebuilt in a different order have the SAME positions
    and different ids, so only ids can tell them apart. Without ids on either side it falls
    back to positions, which is all that can be said about the older artifacts.

    The fingerprint in the stamp identifies the PARENT sample (the 144 lesions), not the
    subset an engine actually explained. LORE explains 48 of them: on 31/07 they were the
    first 48 of the sorted 144 (35 pos / 13 neg), since 01/08 they are an explicit 24+24.
    Same fingerprint, two different samples — so "already done" alone is not enough, or the
    redo would skip exactly the runs that need redoing.

    Returns ``None`` when the artifact does not record which rows it explained (every one
    produced before `val_index` existed): "I do not know" is different information from
    "does not match", and the caller decides whether to redo.
    """
    got, got_ids = read_sample_identity(run_dir, kind)
    exp_ids = stamp_row_ids(expected_idx).get(ROW_ID_KEY)
    if got_ids is not None and exp_ids is not None:
        return len(got_ids) == len(exp_ids) and bool(
            np.array_equal(np.sort(got_ids), np.sort(np.asarray(exp_ids, str))))
    if got is None:
        return None
    exp = np.asarray(expected_idx, int)
    return len(got) == len(exp) and bool(np.array_equal(np.sort(got), np.sort(exp)))


def resume_verified(run_dirs, kind, fp, expected_idx, task=None):
    """Like `resume_report`, but "done" means **done on THIS sample**.

    Returns ``(todo, done, redo)``: runs that are stamped but on different samples (or on
    unrecorded ones) land in `redo`, not in `done`. It is the difference between "do not
    redo what is done" and "do not redo anything".
    """
    items = list(run_dirs.items() if hasattr(run_dirs, "items") else run_dirs)
    todo, done, redo = [], [], []
    for rid, d in items:
        if not is_done(d, kind, fp, task):
            todo.append(rid)
            continue
        ok = sample_matches(d, kind, expected_idx)
        (done if ok is True else redo).append(rid)
    return todo, done, redo
