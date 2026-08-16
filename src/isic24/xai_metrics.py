"""Explanation SHAPE metrics — a complement, never the focus.

What they measure, and why the distinction is not pedantry
----------------------------------------------------------
Faithfulness (`xai_faithfulness`) measures whether an explanation tells the truth about the
model. Attendance (`lore_coverage`) measures how much the model looks at a modality. These
measure something else again: how **readable** an explanation is — concentrated on few
features, short, stable.

⚠️ **READABILITY AND ATTENDANCE ARE ANTI-CORRELATED BY THIS DESIGN.** Measured 2026-08-01 on
`fixed_scaling` seed 47, where `a0p000` has the tabular branch **off by construction**:

    setting   |SHAP| mass   SHAP compactness   rule_len   LORE compactness
    a0p000        0.0000            1.000          0.00         1.0079  (out of range!)
    a0p500       22.9674            0.272          1.38         0.9970
    a1p000      203.8043            0.282          9.47         0.9328

The model that **explains nothing** scores highest on both. The less a model uses a modality,
the shorter its explanation: that is mechanical, not a coincidence.

Hence the two rules of this module, which are structural and not comments:

1. **When the branch is off, the shape metrics are `NaN`, not a high score.** A footnote in
   a CSV does not survive the next reading; a `NaN` does. The threshold (`MIN_SHAP_MASS`)
   separates cleanly: `a0p000` has mass **exactly 0.0**, the next configuration 15.5.
2. **Every row carries its own `shap_mass` / `rule_len`**, so the reader sees what the score
   comes from without going back to the artifacts.

They are not aggregated with faithfulness into a single scalar (the `Q_XAI` proposal of the
external sheet, 01/08): adding readability to correctness hides exactly the trade-off the
study means to show. They are reported **beside** it, and plotted **against**
`lore_coverage`, never inside it.

Zero GPU: it only reads the SHAP npz and LORE CSVs already on disk, and writes CSVs into
`runs/_faithfulness/` next to the other summaries.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

from . import paths as _paths

# Below this total |SHAP| mass the tabular branch is off and "compactness" is an artifact of
# dividing by zero. `a0p000` measures exactly 0.0, its neighbour 15.5: any threshold in
# between separates them, and 1.0 is conservative without being arbitrary at the margin.
MIN_SHAP_MASS = 1.0
# T6 below this number of shared samples is not a measurement, it is noise.
MIN_OVERLAP = 20
N_PERM_AGREEMENT = 200         # permutations of the SHAP<->LORE agreement null
TOP_K = 10                      # top-k RAW features, not encoded: see `raw_top_k`


# --------------------------------------------------------------------------------------
# encoded -> raw
#
# `xai_engines` already has this mapping, but in TWO directions and with a discrepancy:
# `encoded_indices_for_raw` accepts `{feat}_bin_{b}` for any feature, while
# `aggregate_raw_importance` has `age_approx_bin_` written by hand. On ISIC24 they agree (the
# only discretized feature is `age_approx`), but the notebook builds the names with
# `f"{name}_bin_{b}"` for EVERY `Discretization` — so on a dataset with two discretized
# features `aggregate_raw_importance` would map `foo_bin_0` onto the raw `"foo_bin_0"`
# instead of `"foo"`, silently. Here the mapping lives in ONE place and is generic.
# --------------------------------------------------------------------------------------
_BIN_RE = re.compile(r"^(?P<raw>.+)_bin_\d+$")


def raw_name_for(encoded_name):
    """Name of the RAW feature this encoded dimension comes from."""
    if "__" in encoded_name:                      # one-hot categorical: `feat__token`
        return encoded_name.split("__", 1)[0]
    m = _BIN_RE.match(encoded_name)               # discretized: `feat_bin_3`
    if m:
        return m.group("raw")
    return encoded_name                           # numeric: the dimension IS the feature


def raw_groups(encoded_feature_names):
    """{raw feature: the encoded indices belonging to it}."""
    out = {}
    for j, name in enumerate(encoded_feature_names):
        out.setdefault(raw_name_for(name), []).append(j)
    return {k: np.asarray(v, int) for k, v in out.items()}


def raw_top_k(phi_row, groups, k=TOP_K):
    """The k most important RAW features for one sample.

    RAW and not encoded on purpose: with one-hot encoding the top 10 encoded dimensions can
    be 10 levels of the SAME categorical feature — an encoded top-10 would say "the
    explanation uses 10 things" where it uses one. The importance of a raw feature is the sum
    of the |values| of its dimensions, the same convention as `aggregate_raw_importance`.
    """
    imp = {r: float(np.abs(phi_row[idx]).sum()) for r, idx in groups.items()}
    order = sorted(imp, key=lambda r: -imp[r])
    return {r for r in order[:k] if imp[r] > 0}


# --------------------------------------------------------------------------------------
# T1 + T3 — explanation shape, per run
# --------------------------------------------------------------------------------------
def _shap_npz(run_dir, prefer_subset=True):
    """The SHAP npz to read: the adopted-sample one first, then the canonical one.

    `xai/shap_tab/` holds the arrays computed on the distinct-lesion sample, `xai/shap/` the
    canonical ones on the prefix. The order matters for the SHAP<->LORE agreement: only with
    SHAP and LORE on the SAME sample does the overlap go from 14/48 to 48/48.
    """
    for sub in (("shap_tab", "shap") if prefer_subset else ("shap",)):
        hits = sorted(glob.glob(os.path.join(run_dir, "xai", sub, "epoch_*",
                                             "shap_arrays_epoch_*.npz")))
        if hits:
            return hits[-1]
    return None


def shap_shape_from_values(shap_values, explained_indices=None):
    """T1 — compactness (entropy) and k90 sparsity, from the IN-MEMORY ATTRIBUTIONS.

    This is the function `ShapCallback` calls while computing: the metrics come out together
    with everything else, in the same `shap_summary.csv`, with no cell to run afterwards.
    `shap_shape_metrics` below is only the version that re-reads from disk, for artifacts
    produced before this hook existed.

    Both are per sample, then averaged. Zero-mass samples are EXCLUDED from the mean and
    counted in `frac_zero_shap`: on a zero-mass sample `p_j = 0` everywhere, the entropy is 0
    and compactness would come out at 1 — the top score for "I explained nothing".
    """
    phi = np.abs(np.asarray(shap_values, float))
    if phi.ndim == 3:                                  # (n, 1, d) or (n, d, 1)
        phi = phi[:, 0, :] if phi.shape[1] == 1 else phi[..., 0]
    idx = (np.asarray(explained_indices, int).ravel()
           if explained_indices is not None else np.arange(phi.shape[0]))
    d = phi.shape[1]
    mass = phi.sum(axis=1)
    live = mass > 0
    rec = {"n_shap": int(phi.shape[0]), "d_encoded": int(d),
           "shap_mass": float(mass.sum()), "shap_mass_per_sample": float(mass.mean()),
           "frac_zero_shap": float((~live).mean()),
           "shap_sample_hash": int(idx.sum())}      # to check the runs share the same sample
    if not live.any() or rec["shap_mass"] < MIN_SHAP_MASS:
        # tabular branch off: there is no shape to measure
        rec.update(shap_compactness=np.nan, shap_k90=np.nan, shap_k90_sparsity=np.nan,
                   branch_off=True)
        return rec
    P = phi[live] / mass[live, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    H = -(P * logP).sum(axis=1) / np.log(d)
    srt = -np.sort(-phi[live], axis=1)
    cum = np.cumsum(srt, axis=1) / srt.sum(axis=1, keepdims=True)
    k90 = (cum < 0.90).sum(axis=1) + 1               # first k reaching 90%
    rec.update(shap_compactness=float(np.mean(1.0 - H)),
               shap_k90=float(np.mean(k90)),
               shap_k90_sparsity=float(np.mean(1.0 - (k90 - 1) / (d - 1))),
               branch_off=False)
    return rec


def shap_shape_metrics(run_dir, encoded_feature_names=None, k=TOP_K):
    """As above, but re-reading the npz already written — for the historical artifacts."""
    p = _shap_npz(run_dir)
    if p is None:
        return None
    with np.load(p) as z:
        return shap_shape_from_values(z["shap_values"], z["explained_indices"])


def _lore_persample(run_dir):
    from . import xai_subset as xs
    hits = sorted(glob.glob(os.path.join(xs.out_dir_for(run_dir, "lore_sa"), "epoch*",
                                         "lore_sa_persample_*.csv")))
    if not hits:      # historical artifacts (up to 31/07), a different folder
        hits = sorted(glob.glob(os.path.join(run_dir, "xai", "lore", "epoch*", "tabular_sa",
                                             "lore_sa_persample_*.csv")))
    return pd.read_csv(hits[-1]) if hits else None


def lore_shape_from_df(d, d_raw=None):
    """T3 — rule compactness, with the REPAIRED formula, from the IN-MEMORY DataFrame.

    `LoreSACallback` calls it as soon as it finishes explaining: the columns land in
    `lore_sa_per_epoch.csv` together with `lore_coverage` and the rest, with no extra cell.

    The sheet proposes `1 - (|R|-1)/(d-1)`, which on an empty rule (`|R| = 0`, which LORE
    writes as "IF <no premise>") gives **1.0079**: outside [0,1] and maximal. An empty rule
    is not a compact explanation, it is the absence of one. Here empty rules are excluded
    from the mean and counted in `frac_empty_rules`; if all of them are empty the result is
    `NaN`.
    """
    if d is None or not len(d):
        return None
    rl = d["rule_len"].astype(float)
    n_raw = int(d_raw) if d_raw else int(max(rl.max(), 2))
    rec = {"n_lore": int(len(d)), "lore_rule_len": float(rl.mean()),
           "frac_empty_rules": float((rl < 1).mean()),
           "lore_pos_rate": float(d["true_label"].mean()) if "true_label" in d else np.nan,
           # The sample's IDENTITY, not just its size. `n` is not enough: on 01/08
           # `vanilla` ended up with three runs all at n=48 but over TWO different samples —
           # seed 47 from the 31/07 pass (prefix of the 144, pos_rate 0.729), seeds 53 and 71
           # on the balanced sample (0.500). Averaging them gives a family `lore_coverage`
           # belonging to no population. New runs carry `val_index` and the key is exact;
           # for the old ones (n, prevalence) separates them anyway.
           "lore_sample_key": (f"h{abs(hash(tuple(d['val_index'].tolist()))) % 10**8:08d}"
                               if "val_index" in d.columns else
                               f"n{len(d)}-p{float(d['true_label'].mean()):.3f}"
                               if "true_label" in d.columns else f"n{len(d)}")}
    live = rl >= 1
    if not live.any():
        rec.update(lore_rule_compactness=np.nan, branch_off_lore=True)
        return rec
    comp = 1.0 - (rl[live] - 1) / max(n_raw - 1, 1)
    rec.update(lore_rule_compactness=float(np.clip(comp, 0.0, 1.0).mean()),
               branch_off_lore=False)
    return rec


def lore_shape_metrics(run_dir, d_raw=None):
    """As above, but re-reading the per-sample CSV — for the historical artifacts."""
    return lore_shape_from_df(_lore_persample(run_dir), d_raw)


# --------------------------------------------------------------------------------------
# T6 — SHAP <-> LORE agreement
# --------------------------------------------------------------------------------------
# A name can start with a DIGIT: ISIC24 has `3d_lesion_orientation` and
# `3d_position_distance`. With a leading `[A-Za-z_]` — the Python identifier — those 14
# conditions disappeared silently and the agreement with SHAP came out lower than it is.
_COND_RE = re.compile(r"^\s*([A-Za-z0-9_][A-Za-z0-9_.]*)\s*(<=|>=|!=|<|>|=)")


def rule_features(rule_text):
    """The RAW features named in a LORE rule. Empty for "IF <no premise>".

    It is a SET: a rule can constrain the same feature twice (`x > a AND x <= b`), so
    `len(rule_features) <= rule_len` — measured, 33/48 coincide and the rest have two-sided
    intervals. That is the right semantics for a Jaccard between feature sets, which is what
    both sec. 2.4 and sec. 3.5 ask for.
    """
    if not isinstance(rule_text, str) or "nessuna premessa" in rule_text:
        return set()
    body = rule_text.split("THEN")[0].replace("IF", "", 1)
    out = set()
    for cond in body.split(" AND "):
        m = _COND_RE.match(cond.strip())
        if m:
            out.add(m.group(1))
    return out


def shap_lore_agreement(run_dir, encoded_feature_names, k=TOP_K, lore_idx=None,
                        n_perm=N_PERM_AGREEMENT, seed=0):
    """T6 — Jaccard between SHAP's top-k RAW features and the features in the LORE rule.

    ⚠️ TWO OBSTACLES, both measured, both reported as columns rather than hidden.

    1. **Different spaces.** SHAP lives on 127 ENCODED dimensions, LORE rules on RAW
       features. The encoded->raw aggregation is `raw_top_k`, not a direct Jaccard.
    2. **The two explainers index into different spaces, and the two names look alike.**
       `explained_indices` in the SHAP npz is the position INSIDE the sample (`0..143`);
       `val_index` is the position in `validation_df` (`1..1059`), and that is what LORE
       writes. Pairing one with the other does not raise: it crosses the two spaces and
       returns the accidental collisions — **15/48** on the 03/08 runs, below `MIN_OVERLAP`,
       i.e. a `NaN` that looks like a wrong sample and was a wrong comparison. With
       `val_index` on both sides the overlap is **48/48**. The canonical npz in `xai/shap/`
       predate `val_index`: there the sample IS the prefix, so `explained_indices` already
       lives in `validation_df` space and is the correct fallback. Below `MIN_OVERLAP` the
       result stays `NaN`.
    """
    from . import xai_subset as xs

    p = _shap_npz(run_dir)
    lore = _lore_persample(run_dir)
    if p is None or lore is None or "rule_text" not in lore:
        return None
    with np.load(p) as z:
        phi = np.abs(np.asarray(z["shap_values"], float))
        # `val_index` for the new runs, `explained_indices` for the canonical ones on the
        # prefix: LORE indexes into `validation_df`, so SHAP must do the same.
        shap_idx, _ = xs.read_explained_index(z)

    # ⚠️ The `i` column of the LORE CSV is the position INSIDE the LORE sample, NOT in
    # `validation_df`. The two coincide only when the LORE sample was the prefix — and since
    # 01/08 it is not. So the real indices are either passed by the caller, or read from the
    # `val_index` column `LoreSACallback` writes on the new runs; if neither is there, this
    # function does NOT guess: it returns NaN and says why. Assuming the prefix here would
    # give an agreement over 48 wrong samples with plausible numbers.
    if "val_index" in lore.columns:                      # new runs: they carry it
        lore_idx = lore["val_index"].to_numpy()
    elif isinstance(lore_idx, dict):                     # {n_samples: indices}
        lore_idx = lore_idx.get(len(lore))
    elif lore_idx is not None and len(lore_idx) != len(lore):
        # One array applied to runs with DIFFERENT samples would pair the wrong rows
        # silently: the 6 runs of 31/07 have n=48, the historical ones n=32 on the raw
        # prefix. If the lengths do not match, nothing is guessed.
        return {"n_overlap": np.nan, "agreement_shap_lore": np.nan,
                "agreement_note": f"lore_idx has {len(lore_idx)} indices but the run "
                                  f"explained {len(lore)}: different samples, not pairable"}
    if lore_idx is None:
        return {"n_overlap": np.nan, "agreement_shap_lore": np.nan,
                "agreement_note": "validation indices unknown: pass lore_idx="
                                  "<positional indices> or {n: indices}"}
    lore_idx = np.asarray(lore_idx, int)[:len(lore)]

    pos_in_shap = {int(v): j for j, v in enumerate(shap_idx)}
    pairs = [(pos_in_shap[int(v)], r) for r, v in enumerate(lore_idx)
             if int(v) in pos_in_shap]
    rec = {"n_overlap": len(pairs)}
    if len(pairs) < MIN_OVERLAP:
        rec["agreement_shap_lore"] = np.nan
        rec["agreement_note"] = (f"only {len(pairs)} samples shared between SHAP and LORE "
                                 f"(minimum {MIN_OVERLAP})")
        return rec
    g = raw_groups(encoded_feature_names)
    A_sets, B_sets = [], []
    for si, li in pairs:
        A = raw_top_k(phi[si], g, k=k)
        B = rule_features(lore["rule_text"].iloc[li])
        if not A and not B:
            continue
        A_sets.append(A)
        B_sets.append(B)

    def _j(a, b):
        return len(a & b) / max(len(a | b), 1)

    rec["agreement_shap_lore"] = (float(np.mean([_j(a, b) for a, b in zip(A_sets, B_sets)]))
                                  if A_sets else np.nan)
    rec["agreement_n_used"] = len(A_sets)
    if A_sets:
        rec.update(_agreement_null(A_sets, B_sets, sorted(g), n_perm=n_perm, seed=seed))
    return rec


def _agreement_null(A_sets, B_sets, raw_features, n_perm=N_PERM_AGREEMENT, seed=0):
    """The null of the SHAP<->LORE agreement, on TWO scales. Without it the number is unreadable.

    A Jaccard between a top-k and a rule set grows **mechanically** with rule length, and
    `rule_len` is exactly what orders the families: the raw value therefore produces an
    ordering that looks like a result and is not. Measured 2026-08-04 over 110 runs: rho with
    `mean_rule_feats` is +0.937 on the observed and **+0.945 on the null**, excess -0.008.
    That is why these columns come out TOGETHER with the agreement, not "later, if needed".

    * `agreement_null_*` — permutation INSIDE the run: SHAP row *i* is paired with a shuffled
      rule. It preserves the set sizes on both sides and destroys only the **per-sample**
      correspondence.
    * `agreement_rho_global` — and that alone is not enough: preserving the marginals, a
      null-level value is compatible with "they agree on which features matter globally, and
      not on which matter for THIS lesion". That is told apart by correlating the two
      per-raw-feature frequencies over the whole run. Its null shuffles feature identity.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)

    def _j(a, b):
        return len(a & b) / max(len(a | b), 1)

    obs = float(np.mean([_j(a, b) for a, b in zip(A_sets, B_sets)]))
    null = np.array([np.mean([_j(A_sets[i], B_sets[j]) for i, j in
                              enumerate(rng.permutation(len(B_sets)))])
                     for _ in range(n_perm)])
    sd = float(null.std())
    out = {
        "agreement_null_mean": float(null.mean()),
        "agreement_null_sd": sd,
        "agreement_z": float((obs - null.mean()) / sd) if sd > 0 else np.nan,
        "agreement_p_perm": float((1 + int((null >= obs).sum())) / (1 + len(null))),
        "agreement_mean_rule_feats": float(np.mean([len(b) for b in B_sets])),
    }
    fa = np.array([sum(f in s for s in A_sets) for f in raw_features], float)
    fb = np.array([sum(f in s for s in B_sets) for f in raw_features], float)
    if fa.std() > 0 and fb.std() > 0:
        out["agreement_rho_global"] = float(stats.spearmanr(fa, fb)[0])
        gn = [stats.spearmanr(fa, rng.permutation(fb))[0] for _ in range(max(n_perm // 4, 10))]
        out["agreement_rho_global_null"] = float(np.median(gn))
        ta = {raw_features[i] for i in np.argsort(-fa)[:TOP_K]}
        tb = {raw_features[i] for i in np.argsort(-fb)[:TOP_K]}
        out["agreement_top_global_jaccard"] = len(ta & tb) / max(len(ta | tb), 1)
    return out


# --------------------------------------------------------------------------------------
# T2 — stability between configurations
# --------------------------------------------------------------------------------------
def setting_value(impl_name, setting_tag):
    """The NUMERIC value of a family's parameter axis, or None when it has none.

    Three families out of six have an axis (`fixed_scaling` 9 points, `learnable_scaling` 5,
    `loss_regularization` 5). `vanilla`, `controlled_fusion` and `adaptive_gating` have a
    single setting: for them "adjacent configurations" is NOT defined, and the metric is not
    computed rather than computed over an invented pair.
    """
    t = str(setting_tag)
    m = re.search(r"a(\d)p(\d{3})$", t)                    # a0p500 / init_a0p500
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"lam(\d)e([+-]\d\d)$", t)               # lam1e-03 / lam0e+00
    if m:
        return float(f"{m.group(1)}e{m.group(2)}")
    return None


def _run_top_k(run_dir, groups, k=TOP_K):
    from . import xai_subset as xs

    p = _shap_npz(run_dir)
    if p is None:
        return None, None
    with np.load(p) as z:
        phi = np.abs(np.asarray(z["shap_values"], float))
        # `val_index` and not `explained_indices`: the latter is `0..n-1` for ANY sample,
        # so the identity guard in `stability_table` could never fire — two different samples
        # would give identical indices. With `val_index` the comparison is over the real
        # lesions. On the canonical ones (prefix) the two coincide.
        idx, _ = xs.read_explained_index(z)
    return [raw_top_k(r, groups, k) for r in phi], idx


def _jaccard_runs(a, b):
    js = [len(x & y) / len(x | y) for x, y in zip(a, b) if (x | y)]
    return float(np.mean(js)) if js else np.nan


def stability_table(master, art_path, encoded_feature_names, k=TOP_K):
    """T2 — Stability@k, with its NULL beside it.

    Two kinds of pair, over the SAME samples:

    * ``adjacent_param``  same seed, consecutive settings on the axis -> the weight's effect;
    * ``same_param_seed`` same setting, every pair of seeds -> **the seed noise**.

    Without the second, "Stability@5 = 0.6 between alpha=0.5 and alpha=0.625" is not
    interpretable: it could be initialization noise entirely. The number that matters is the
    difference between the two, which is why the null is computed here and not "later, if
    needed".

    The unit of INFERENCE over this table is the seed and not the row -- `seed_level_contrast`
    -- which is why the null takes the COMPLETE set of seed pairs and not the consecutive ones
    only: reduced by seed, a null built on consecutive pairs gives the two extreme seeds one
    pair each and the middle ones two, so how well a seed's null is estimated would depend on
    where its number falls in a sort, which means nothing. The top-k sets are cached per run,
    so the extra pairs cost set operations and no I/O.

    NB `lam0e+00` stays on the axis ordered by VALUE (0 < 1e-5): no logarithm is taken, but
    the spacing between pairs is very uneven and the `param_a`/`param_b` columns make that
    explicit instead of leaving it to be inferred.
    """
    from . import xai_subset as xs

    g = raw_groups(encoded_feature_names)
    m = master.copy()
    m["param"] = [setting_value(i, t) for i, t in zip(m.impl_name, m.setting_tag)]
    m["run_dir_abs"] = [_paths.run_dir_abs(d, art_path) for d in m.run_dir]

    cache, ref_idx = {}, None
    def topk(run_dir):
        nonlocal ref_idx
        if run_dir not in cache:
            tk, idx = _run_top_k(run_dir, g, k)
            if tk is not None:
                if ref_idx is None:
                    ref_idx = idx
                elif not np.array_equal(ref_idx, idx):
                    raise RuntimeError(
                        "[STABILITY] two runs have SHAP on different samples: the Jaccard "
                        "would compare explanations of different lesions.")
            cache[run_dir] = tk
        return cache[run_dir]

    rows = []
    for impl, grp in m.groupby("impl_name"):
        # (a) adjacent pairs on the axis, at equal seed
        if grp["param"].notna().all() and grp.setting_tag.nunique() > 1:
            for seed, gs in grp.groupby("seed"):
                gs = gs.sort_values("param")
                for (_, ra), (_, rb) in zip(gs.iloc[:-1].iterrows(), gs.iloc[1:].iterrows()):
                    # A pair touching an ablation crosses an architectural boundary: the
                    # model at one end lacks the branch the metric measures.
                    if xs.is_ablation(impl, ra.setting_tag) or xs.is_ablation(impl, rb.setting_tag):
                        continue
                    A, B = topk(ra.run_dir_abs), topk(rb.run_dir_abs)
                    if A is None or B is None:
                        continue
                    rows.append(dict(impl_name=impl, pair_kind="adjacent_param",
                                     seed_a=seed, seed_b=seed,
                                     run_id_a=ra.run_id, run_id_b=rb.run_id,
                                     tag_a=ra.setting_tag, tag_b=rb.setting_tag,
                                     param_a=ra.param, param_b=rb.param,
                                     stability_at_k=_jaccard_runs(A, B), k=k))
        # (b) NULL: same setting, different seeds. Ablations drop out here too, and for the
        # same reason as in (a): an effect and its null have to be measured on the same
        # population, or their difference is partly a difference of population. Excluding
        # only the branch-off setting left the tabular-only end in the null and out of the
        # effect — 0.005 on ISIC, and no reason to expect it to stay that small elsewhere.
        for tag, gt in grp.groupby("setting_tag"):
            if xs.is_ablation(impl, tag):
                continue
            gt = list(gt.sort_values("seed").itertuples())
            for i, ra in enumerate(gt):
                for rb in gt[i + 1:]:
                    A, B = topk(ra.run_dir_abs), topk(rb.run_dir_abs)
                    if A is None or B is None:
                        continue
                    rows.append(dict(impl_name=impl, pair_kind="same_param_seed",
                                     seed_a=ra.seed, seed_b=rb.seed,
                                     run_id_a=ra.run_id, run_id_b=rb.run_id,
                                     tag_a=tag, tag_b=tag,
                                     param_a=ra.param, param_b=rb.param,
                                     stability_at_k=_jaccard_runs(A, B), k=k))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Inference over a pair table — the unit is the SEED, not the pair
# --------------------------------------------------------------------------------------
def seed_level_contrast(pairs, value_col, effect_kind, null_kind, by=None):
    """The effect arm against its null, reduced to one value per seed before any test.

    A table of run pairs is a graph: the runs are the nodes and the pairs are the edges, so
    a test over its rows counts a model once for every neighbour it has. On the canonical
    grid that turns six seeds into some thirty-five rows per arm, and the p-value then
    reports an n that does not exist — the same error as pooling samples inside a run, one
    level up.

    The two arms do not reduce the same way, and the asymmetry is structural rather than an
    inconvenience to hide: an effect pair lives INSIDE one seed, a null pair spans two. A
    seed's effect is therefore the mean over its own pairs, its null the mean over the pairs
    that TOUCH it, so every null pair counts for both of its seeds and the null values stay
    correlated across seeds. The reduction fixes the count, not that correlation, and the
    correlation is what keeps this a paired test between two arms rather than a comparison of
    two independent groups.

    The test is `xs.across_runs` over the per-seed differences `effect - null`, and `p_floor`
    is the smallest p that many seeds allow: with six it is 0.031, and no effect size reaches
    below it. The spreads returned are across seeds, which is the level the test works at; the
    spread across pair rows describes the table and cannot be read as uncertainty.
    """
    from . import xai_subset as xs

    p = pairs[pairs.pair_kind.isin((effect_kind, null_kind))]
    span = p[(p.pair_kind == effect_kind) & (p.seed_a != p.seed_b)]
    if len(span):
        raise RuntimeError(
            f"[SEED UNIT] {len(span)} '{effect_kind}' pairs span two seeds: the arm does not "
            "reduce to one value per seed, and pairing it against the null is undefined.")

    rows, no_effect = [], []
    for key, g in ([(None, p)] if by is None else p.groupby(by)):
        e = g[g.pair_kind == effect_kind].groupby("seed_a")[value_col].mean()
        u = g[g.pair_kind == null_kind]
        n = pd.concat([u.set_index("seed_a")[value_col],
                       u.set_index("seed_b")[value_col]]).groupby(level=0).mean()
        seeds = sorted(set(e.index) & set(n.index))
        if not seeds:
            no_effect.append(str(key))
            continue
        ev, nv = e.loc[seeds].to_numpy(float), n.loc[seeds].to_numpy(float)
        res = xs.across_runs(nv, ev, name_a=null_kind, name_b=effect_kind)
        rows.append(dict({by: key} if by else {}, n_seeds=len(seeds),
                         effect_mean=float(ev.mean()), effect_sd=float(ev.std(ddof=1)),
                         null_mean=float(nv.mean()), null_sd=float(nv.std(ddof=1)),
                         diff_mean=res["diff_mean"], n_positive=res["n_positive"],
                         sign_p=res["sign_p"], wilcoxon_p=res["wilcoxon_p"],
                         p_floor=res["p_floor"]))
    if no_effect:
        print(f"  [seed unit] no '{effect_kind}' arm, so no contrast: {', '.join(no_effect)}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Collection + per-family summary
# --------------------------------------------------------------------------------------
def collect(master, art_path, encoded_feature_names=None, k=TOP_K, d_raw=None,
            lore_idx=None):
    """One row per run with every shape metric available.

    `d_raw` — the number of RAW features, the denominator of `lore_rule_compactness`. Derived
    from `encoded_feature_names` when not passed: deriving it from the maximum rule length,
    as the first draft did, makes the denominator differ from run to run and the values
    incomparable between families.
    `lore_idx` — POSITIONAL indices into validation_df of the samples LORE explained, for T6.
    """
    if d_raw is None and encoded_feature_names is not None:
        d_raw = len(raw_groups(encoded_feature_names))
    rows = []
    for _, r in master.iterrows():
        rd = _paths.run_dir_abs(r.run_dir, art_path)
        rec = dict(impl_name=r.impl_name, setting_tag=r.setting_tag, seed=r.seed,
                   run_id=r.run_id, param=setting_value(r.impl_name, r.setting_tag))
        for f in (shap_shape_metrics(rd, encoded_feature_names, k),
                  lore_shape_metrics(rd, d_raw)):
            if f:
                rec.update(f)
        if encoded_feature_names is not None:
            a = shap_lore_agreement(rd, encoded_feature_names, k, lore_idx=lore_idx)
            if a:
                rec.update(a)
        rows.append(rec)
    return pd.DataFrame(rows)


def summary_by_family(art_path, shape_df=None, stab_df=None, central=None, out_dir=None,
                      verbose=True):
    """The SINGLE per-family table: faithfulness + attendance + shape, side by side.

    The column order is the thesis: the PRIMARY metrics first (faithfulness, attendance),
    then the shape ones, marked `shape__`. There is no aggregate scalar merging them: adding
    readability to correctness hides the trade-off instead of showing it.
    """
    from . import xai_subset as xs
    out_dir = out_dir or os.path.join(art_path, "runs", xs.AGGREGATE_DIRNAME)
    parts = []

    f = os.path.join(out_dir, "faithfulness_aligned_signed_n144.csv")
    if os.path.exists(f):
        d = xs.drop_ablations(pd.read_csv(f), "faithfulness", verbose)
        if central:
            d = d[[central.get(i) == t for i, t in zip(d.impl_name, d.setting_tag)]]
        parts.append(d.groupby("impl_name").agg(
            n_runs=("run_id", "nunique"),
            faith_target_q20=("target_del_q20norm", "mean"),
            faith_posev_q20=("posev_del_q20norm", "mean")))

    l = os.path.join(out_dir, "lore_sa_quality_n48.csv")
    if os.path.exists(l):
        d = xs.drop_ablations(pd.read_csv(l), "lore", verbose)
        if central:
            d = d[[central.get(i) == t for i, t in zip(d.impl_name, d.setting_tag)]]
        parts.append(d.groupby("impl_name").agg(
            lore_coverage=("lore_coverage_5050", "mean"),
            lore_rule_len=("rule_len_5050", "mean")))

    if shape_df is not None and len(shape_df):
        d = xs.drop_ablations(shape_df, "shape", verbose)
        if central:
            d = d[[central.get(i) == t for i, t in zip(d.impl_name, d.setting_tag)]]
        agg = {"shape__shap_compactness": ("shap_compactness", "mean"),
               "shape__shap_k90_sparsity": ("shap_k90_sparsity", "mean"),
               "shape__shap_mass": ("shap_mass", "mean")}
        for src, dst in [("lore_rule_compactness", "shape__lore_rule_compactness"),
                         ("agreement_shap_lore", "shape__agreement_shap_lore"),
                         ("n_overlap", "shape__agreement_n_overlap")]:
            if src in d:
                agg[dst] = (src, "mean")
        fam = d.groupby("impl_name").agg(**agg)
        # ⚠️ The LORE columns can be averaged across runs only when the runs explained the
        # SAME number of samples. `controlled_fusion base` has seed 47 on the new sample (48)
        # and seed 8 on the historical one (32): the mean gave `n_overlap = 36.5`, an
        # agreement computed over two different samples and presented as a single number. It
        # is the same double-counting error as in the LORE section, one level up. Here it
        # comes out NaN and the `shape__lore_n` column shows why.
        key = "lore_sample_key" if "lore_sample_key" in d else "n_lore"
        if key in d:
            mix = d.groupby("impl_name")[key].agg(
                lambda s: sorted({str(v) for v in s.dropna()}))
            n_mix = (d.groupby("impl_name")["n_lore"]
                     .agg(lambda s: sorted({int(v) for v in s.dropna()})))
            fam["shape__lore_n"] = [",".join(map(str, v)) or None for v in n_mix]
            fam["shape__lore_samples"] = [len(v) for v in mix]
            mixed = [len(v) > 1 for v in mix]
            for c in ("shape__lore_rule_compactness", "shape__agreement_shap_lore",
                      "shape__agreement_n_overlap"):
                if c in fam:
                    fam.loc[mixed, c] = np.nan
        parts.append(fam)

    if stab_df is not None and len(stab_df):
        piv = (stab_df.pivot_table(index="impl_name", columns="pair_kind",
                                   values="stability_at_k", aggfunc="mean")
               .rename(columns={"adjacent_param": "shape__stability_adjacent",
                                "same_param_seed": "shape__stability_seed_null"}))
        if {"shape__stability_adjacent", "shape__stability_seed_null"} <= set(piv.columns):
            # THIS is the reading: stability between adjacent weights means something only
            # when it falls below the seed noise. Positive = weights move more than the seed.
            piv["shape__stability_effect"] = (piv["shape__stability_seed_null"]
                                              - piv["shape__stability_adjacent"])
        parts.append(piv)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1)
    p = os.path.join(out_dir, "xai_summary_by_family.csv")
    out.to_csv(p)
    if verbose:
        print(f"[metrics] {os.path.basename(p)} — {len(out)} families, "
              f"{len(out.columns)} columns ({sum(c.startswith('shape__') for c in out.columns)} shape)")
    return out


# --------------------------------------------------------------------------------------
# Grad-CAM — heatmap shape metrics (T4)
#
# The same two rules as on the tabular side: they are computed INSIDE the engine
# (`MultiModalGradCAMCallback` calls them as soon as it has the heatmaps) and they are a
# COMPLEMENT, never a single score.
#
# ⚠️ And the same inversion trap. A concentrated heatmap does not mean a good explanation: a
# model that ignores the image can come out as the most "focused" one. They are read beside
# faithfulness, not in its place.
# --------------------------------------------------------------------------------------
CAM_TAU = 0.5        # binary mask threshold, on the heatmap normalized to [0,1]
CAM_Q = 0.10         # top-q% of the pixels for the energy concentration


def gradcam_shape_from_cams(cams, tau=CAM_TAU, q=CAM_Q):
    """Compactness, focus, energy concentration and connectedness, averaged over samples.

    `cams` (n, Hc, Wc), the RAW heatmaps. Negative values are clipped to zero before
    normalizing: Grad-CAM produces them, and including them in the mass would stop `p_j`
    being a distribution.
    """
    c = np.maximum(np.asarray(cams, float), 0.0)
    if c.ndim == 2:
        c = c[None]
    n, H, W = c.shape
    flat = c.reshape(n, -1)
    tot = flat.sum(axis=1)
    live = tot > 0
    rec = {"cam_frac_empty": float((~live).mean())}
    if not live.any():
        # Image branch OFF (`fixed_scaling` a1p000: beta=0 zeroes the gradient, the heatmap
        # is all zeros). The metrics are NaN — rule 1 of this module — but every KEY is
        # there: returning four keys out of six made these runs write a CSV with a DIFFERENT
        # SCHEMA from the other 105, and every cross-run concat broke on `cam_area`. A mode
        # with no result is not a mode with fewer columns.
        rec.update(cam_compactness=np.nan, cam_focus=np.nan,
                   cam_energy_top10=np.nan, cam_connectedness=np.nan,
                   cam_area=np.nan, cam_n_components=np.nan)
        return rec
    P = flat[live] / tot[live, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    H_norm = -(P * logP).sum(axis=1) / np.log(H * W)
    srt = -np.sort(-P, axis=1)
    k = max(1, int(np.ceil(q * H * W)))

    # binary mask on the heatmap rescaled to [0,1] per sample
    mn = flat[live].min(axis=1, keepdims=True)
    mx = flat[live].max(axis=1, keepdims=True)
    B = ((flat[live] - mn) / (mx - mn + 1e-12) >= tau).reshape(-1, H, W)
    area = B.reshape(len(B), -1).mean(axis=1)

    # connected components (4-neighbourhood) without depending on scipy/cv2
    conn = []
    for b in B:
        conn.append(_n_components(b))
    conn = np.asarray(conn, float)
    frag = conn / (B.reshape(len(B), -1).sum(axis=1) + 1e-12)

    rec.update(cam_compactness=float(np.mean(1.0 - H_norm)),
               cam_focus=float(np.mean(1.0 - area)),
               cam_energy_top10=float(np.mean(srt[:, :k].sum(axis=1))),
               cam_connectedness=float(np.mean(1.0 / (1.0 + frag))),
               cam_area=float(np.mean(area)),
               cam_n_components=float(conn.mean()))
    return rec


def _n_components(mask):
    """4-neighbourhood connected components of a boolean mask. Iterative flood fill."""
    m = np.asarray(mask, bool).copy()
    n = 0
    H, W = m.shape
    for i in range(H):
        for j in range(W):
            if not m[i, j]:
                continue
            n += 1
            stack = [(i, j)]
            m[i, j] = False
            while stack:
                a, b = stack.pop()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < H and 0 <= y < W and m[x, y]:
                        m[x, y] = False
                        stack.append((x, y))
    return n


def cam_masks(cams, tau=CAM_TAU):
    """Per-sample binary masks — the basis of the between-configuration IoU (Stability_CAM)."""
    c = np.maximum(np.asarray(cams, float), 0.0).reshape(len(cams), -1)
    mn, mx = c.min(axis=1, keepdims=True), c.max(axis=1, keepdims=True)
    return (c - mn) / (mx - mn + 1e-12) >= tau


# Below this threshold the difference between two heatmaps is float32 noise, not an effect:
# on a grid normalized to [0,1] two reductions in different order diverge by a few ulp
# (measured: 2.4e-07 over 100 runs where the heatmap is identical by construction).
CAM_EPS_FLOAT32 = 1e-05


def cam_mode_contrast(cams_a, cams_b, preds_a=None, preds_b=None, tau=CAM_TAU):
    """How much two configurations (e.g. `context` vs `image_only`) REALLY differ.

    It returns three things that must be read together, because separately they mislead:

    * `cam_iou_modes` — overlap of the binary masks. It is the historical metric and it is
      COARSE: on a 6x6 grid (36 cells, 16 after the border trim) it cannot tell a small shift
      from no shift at all.
    * `cam_dmean` / `cam_dmax` — the difference on the continuous heatmaps, which the IoU
      does not see. `cam_modes_identical` says whether it sits below float32 noise: when it
      is true, the two modes were NOT really compared, they are the same computation twice.
    * `pred_dmean` / `pred_dmax` — how far the prediction moves. It is the only live channel
      when the head is additive in the two branches: without it, the row seems to say that
      zeroing the tabular branch does nothing.

    `pred_dcam_rho` (Spearman between |Δcam| and |Δpred| per sample) TESTS the hypothesis
    that the two differences are the same phenomenon — it does not confirm it. On
    `adaptive_gating`, the only family where the heatmaps really change, the answer is
    **weakly yes**: +0.17 on average, 4 seeds out of 5 positive, one at -0.10. A low value
    here does NOT mean the effect is absent, it means "where it looks" and "how sure it is"
    move on largely independent axes. If the heatmaps are identical it is NaN by definition.
    """
    a = np.asarray(cams_a, float).reshape(len(cams_a), -1)
    b = np.asarray(cams_b, float).reshape(len(cams_b), -1)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    ma, mb = cam_masks(a, tau), cam_masks(b, tau)
    inter, union = (ma & mb).sum(axis=1), (ma | mb).sum(axis=1)
    live = union > 0
    rec = {
        "cam_iou_modes": float(np.mean(inter[live] / union[live])) if live.any() else np.nan,
        "cam_iou_n_valid": int(live.sum()),   # 0 = image branch off, the IoU is rightly NaN
    }

    d = np.abs(a - b)
    d_per_sample = d.mean(axis=1)
    rec["cam_dmean"] = float(d_per_sample.mean())
    rec["cam_dmax"] = float(d.max()) if d.size else np.nan
    rec["cam_modes_identical"] = bool(rec["cam_dmax"] <= CAM_EPS_FLOAT32)

    if preds_a is None or preds_b is None:
        return rec
    pa = np.asarray(preds_a, float).reshape(len(preds_a), -1)[:n]
    pb = np.asarray(preds_b, float).reshape(len(preds_b), -1)[:n]
    dp = np.abs(pa - pb).mean(axis=1)
    rec["pred_dmean"] = float(dp.mean())
    rec["pred_dmax"] = float(dp.max()) if dp.size else np.nan
    rho = np.nan
    if not rec["cam_modes_identical"] and d_per_sample.std() > 0 and dp.std() > 0:
        from scipy.stats import spearmanr
        rho = float(spearmanr(d_per_sample, dp).correlation)
    rec["pred_dcam_rho"] = rho
    return rec


# --------------------------------------------------------------------------------------
# Grad-CAM — where two models look, and the null that question needs
# --------------------------------------------------------------------------------------
CAM_TOP_FRAC = 0.20      # second binarization: the top fifth of the cells of each map


def _run_cams(run_dir, mode="context"):
    """The stored heatmaps of a run and the fingerprint of the sample they were computed on."""
    g = sorted(glob.glob(os.path.join(run_dir, "xai", "gradcam_img",
                                      f"gradcam_arrays_{mode}_epoch_*.npz")))
    if not g:
        return None, None
    with np.load(g[-1]) as z:
        return np.asarray(z["cams"], float), str(z["subset_fingerprint"])


def _cam_top_masks(cams, frac=CAM_TOP_FRAC):
    c = np.maximum(np.asarray(cams, float), 0.0).reshape(len(cams), -1)
    k = max(1, int(round(frac * c.shape[1])))
    return c >= np.sort(c, axis=1)[:, -k][:, None]


def _iou_median(A, B):
    inter = (A & B).sum(axis=1)
    union = np.maximum((A | B).sum(axis=1), 1)
    return float(np.median(inter / union))


def cam_overlap_table(master, art_path, central=None, mode="context", frac=CAM_TOP_FRAC):
    """Do two FAMILIES look in the same places more than two SEEDS of one family do?

    The effect and its null are the same kind of number: cross-family pairs (two families,
    one seed) against seed pairs (one family, two seeds), over the same lesions, with the
    run PAIR as the unit — one median IoU per pair, not one per lesion. That unit describes
    the table; the unit the two arms are TESTED at is the seed, in `seed_level_contrast`,
    because a run enters as many pairs as it has partners.

    Two binarizations are carried side by side because a 6x6 map quantizes any overlap
    heavily and a single rule would not show it: `cam_masks` (half of each map's range,
    the convention the engine already uses) and the top `frac` cells of each map.
    """
    from . import xai_subset as xs

    m = master if central is None else master[
        [central.get(i) == t for i, t in zip(master.impl_name, master.setting_tag)]]
    m = xs.drop_ablations(m, "CAM overlap")

    store, fp0 = {}, None
    for r in m.itertuples():
        cams, fp = _run_cams(_paths.run_dir_abs(r.run_dir, art_path), mode)
        if cams is None:
            continue
        if fp0 is None:
            fp0 = fp
        elif fp != fp0:
            raise RuntimeError(
                "[CAM OVERLAP] two runs have Grad-CAM on different samples: the IoU would "
                f"compare heatmaps of different lesions ({fp0!r} vs {fp!r}).")
        store[(r.impl_name, r.seed)] = (cam_masks(cams), _cam_top_masks(cams, frac))

    fams = sorted({f for f, _ in store})
    seeds = sorted({s for _, s in store})
    rows = []
    for s in seeds:
        for i, fa in enumerate(fams):
            for fb in fams[i + 1:]:
                if (fa, s) in store and (fb, s) in store:
                    A, B = store[(fa, s)], store[(fb, s)]
                    rows.append(dict(pair_kind="cross_family", impl_a=fa, impl_b=fb,
                                     seed_a=s, seed_b=s, iou_half=_iou_median(A[0], B[0]),
                                     iou_top=_iou_median(A[1], B[1])))
    for f in fams:
        for i, sa in enumerate(seeds):
            for sb in seeds[i + 1:]:
                if (f, sa) in store and (f, sb) in store:
                    A, B = store[(f, sa)], store[(f, sb)]
                    rows.append(dict(pair_kind="seed_null", impl_a=f, impl_b=f,
                                     seed_a=sa, seed_b=sb, iou_half=_iou_median(A[0], B[0]),
                                     iou_top=_iou_median(A[1], B[1])))
    return pd.DataFrame(rows)
