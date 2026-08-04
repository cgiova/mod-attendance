"""Cantiere B: budget-comparable signed faithfulness, by re-integration only.

Two things were wrong with the signed protocol, both fixed in `xai_faithfulness`:

  B.1  every signed number was full-budget by construction -- `_target_gain_persample`
       called `persample_gains` without `q_max`, and `discover_curve_files` never
       globbed `_robustness/aligned_signed/**`, so the 24 signed curves could not reach
       `stratified_table` at all.
  B.2  a gain is an integral over q, so `q<=0.2` is mechanically ~1/3 of `q<=0.6`.
       `gain_norm = gain / q_width` is added ALONGSIDE the raw column, never replacing it.

No model, no GPU: everything below is arithmetic over curves already on disk.  Writes a
NEW file, `faithfulness_strata_signed.csv`; the canonical artifacts are immutable.

Run:  python scripts/xai_2026-07-28/run_strata_signed.py
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = "/mnt/disk1/home/cgiovannoni/isic24-dum"
sys.path.insert(0, BASE + "/src")

from isic24 import xai_faithfulness as xf  # noqa: E402

ART = os.path.join(BASE, "artifacts", "enhanced")
OUT = os.path.join(ART, "runs", "_faithfulness", "faithfulness_strata_signed.csv")
OUT_POOLED = os.path.join(ART, "runs", "_faithfulness", "faithfulness_strata_signed_pooled.csv")

VANILLA = os.path.join(ART, "runs", "_robustness", "aligned_signed",
                       "20260604-210304", "seed_47__baseline__e20")

fails = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""), flush=True)
    if not cond:
        fails.append(name)


# ----------------------------------------------------------------------------------
print("\n=== P1  the published full-budget numbers are reproduced, and the low-budget "
      "ones match STATUS ===", flush=True)

with np.load(os.path.join(VANILLA, "faithfulness_aligned_signed_abs_curves.npz")) as z:
    full = float(xf.persample_gains(z, "tabular", "del").mean())
    low = float(xf.persample_gains(z, "tabular", "del", q_max=0.20).mean())
    w_full, w_low = xf.gain_window(z, None), xf.gain_window(z, 0.20)

# -0.06983569991132296 is the vanilla `aligned_abs_del_gain` in faithfulness_aligned_signed.csv
check("abs del, full budget == published CSV", abs(full - (-0.06983569991132296)) < 1e-9,
      f"{full:+.6f}")
check("abs del, q<=0.20 == STATUS -0.0083", abs(low - (-0.0083)) < 5e-5, f"{low:+.6f}")
check("q_width read off the grid, not assumed", (w_full, w_low) == (0.60, 0.20),
      f"full={w_full} low={w_low}")
check("normalised abs del == STATUS -0.1164 / -0.0413",
      abs(full / w_full - (-0.1164)) < 5e-5 and abs(low / w_low - (-0.0413)) < 5e-5,
      f"{full / w_full:+.6f} / {low / w_low:+.6f}")

# ----------------------------------------------------------------------------------
print("\n=== P2  the raw per-sample re-integration still equals the saved aggregate ===",
      flush=True)

probs = []
for mode in xf.SIGNED_MODES:
    p = os.path.join(VANILLA, f"faithfulness_aligned_signed_{mode}_curves.npz")
    probs += [(mode, *x) for x in xf.verify_reintegration(p)]
check("verify_reintegration clean on all 4 signed modes (tol 1e-9)", not probs, str(probs[:2]))

# ----------------------------------------------------------------------------------
print("\n=== P3  target-mode sign: per-sample BEFORE the mean, not an aggregate flip ===",
      flush=True)

with np.load(os.path.join(VANILLA, "faithfulness_aligned_signed_target_curves.npz")) as z:
    labels = z["labels"]
    g = xf.persample_gains(z, "tabular", "del", q_max=0.20)
    st = xf.build_strata(labels, z["preds_orig"])
sgn = xf.mode_sign("target", labels)
pos, neg = st.masks["class_pos"], st.masks["class_neg"]

# Within a single class the label sign is constant, so the correct answer coincides with
# a wholesale flip -- that is what makes this a test rather than a tautology.
check("class_pos: per-sample sign == +mean (sign is constant there)",
      abs((g * sgn)[pos].mean() - g[pos].mean()) < 1e-12)
check("class_neg: per-sample sign == -mean (sign is constant there)",
      abs((g * sgn)[neg].mean() + g[neg].mean()) < 1e-12)
# In any mixed stratum the two differ, and no post-hoc flip recovers the right value.
mix = (g * sgn).mean()
check("stratum 'all': per-sample sign differs from BOTH aggregate flips",
      abs(mix - g.mean()) > 1e-6 and abs(mix + g.mean()) > 1e-6,
      f"correct={mix:+.4f}  +flip={g.mean():+.4f}  -flip={-g.mean():+.4f}")

# ----------------------------------------------------------------------------------
print("\n=== P4  the 24 signed curves are discovered and keyed by mode ===", flush=True)

files = xf.discover_curve_files(ART, include_signed=True)
sg = files[files.source == "aligned_signed"]
check("24 signed curve files (6 runs x 4 modes)", len(sg) == 24, f"{len(sg)} found")
check("legacy sources unchanged in count", len(files[files.source != "aligned_signed"]) == 220,
      f"{len(files[files.source != 'aligned_signed'])}")
check("mode column present and complete",
      set(sg["mode"]) == set(xf.SIGNED_MODES) and sg["mode"].notna().all())

# ----------------------------------------------------------------------------------
print("\n=== P5  build the table ===", flush=True)

tab = xf.stratified_table(ART, files=sg, verbose=False)
check("table is non-empty", len(tab) > 0, f"{len(tab)} rows")
check("both budgets present", set(tab.q_max) == {0.20, 0.60})
check("both directions present", set(tab.direction) == {"del", "ins"})
check("gain_norm == gain / q_width",
      np.allclose(tab.gain.dropna(), (tab.gain_norm * tab.q_width).dropna(), atol=1e-12))

pooled = xf.pool_by_family(tab)
check("pool_by_family keys on mode (no cross-mode averaging)", "mode" in pooled.columns)
check("every pooled cell comes from exactly 1 run", set(pooled.n_runs) == {1},
      "signed covers 1 run per family -- see the caveat below")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
tab.to_csv(OUT, index=False)
pooled.to_csv(OUT_POOLED, index=False)
print(f"\n  wrote {OUT}  ({len(tab)} rows)")
print(f"  wrote {OUT_POOLED}  ({len(pooled)} rows)")

# ----------------------------------------------------------------------------------
print("\n=== The numbers: deletion, stratum 'all', raw vs normalised ===", flush=True)

v = tab[(tab.stratum == "all") & (tab.direction == "del")]
piv = v.pivot_table(index="impl_name", columns=["mode", "q_max"], values="gain_norm")
piv = piv.reindex(columns=[(m, q) for m in xf.SIGNED_MODES for q in (0.20, 0.60)])
print("\n-- gain_norm (per unit of budget; the only cross-budget comparable column) --")
print(piv.round(3).to_string())

print("\n-- raw gain, for reference against the published CSV --")
print(v.pivot_table(index="impl_name", columns=["mode", "q_max"], values="gain")
       .reindex(columns=[(m, q) for m in xf.SIGNED_MODES for q in (0.20, 0.60)])
       .round(3).to_string())

print("\n-- does the low-budget window change the ranking? (target mode, deletion) --")
t = v[v["mode"] == "target"].pivot_table(index="impl_name", columns="q_max", values="gain_norm")
t["rank_q0.20"] = t[0.20].rank(ascending=False)
t["rank_q0.60"] = t[0.60].rank(ascending=False)
print(t.round(3).to_string())
print(f"\n  Spearman(q0.20, q0.60) over 6 families = {t[0.20].corr(t[0.60], method='spearman'):+.3f}")

# ----------------------------------------------------------------------------------
print("\n=== Caveats that must travel with this table ===", flush=True)
print("""  * 6 runs, ONE per family, picked for convenience -> n_runs=1 everywhere. The
    MIN_CELL_N_POOLED floor was calibrated on 5 runs x 128 samples; a pooled cell here
    is a single run. Cross-family comparison under the signed protocol is NOT defensible
    until the full 110-run pass is done (~5 h GPU, queued).
  * fixed_scaling is represented only by a0p000 -- the image-only endpoint where the
    tabular branch is scaled to zero and EVERY signed number is exactly 0.0 by
    construction (eff_corr is NaN there). Those zeros are not a measurement. The
    representative-alpha run is queued as scripts/xai_2026-07-28/run_signed_a05.py.
  * Same budget window != same unit of evidence: at q=0.05 the tabular branch drops 2
    features out of 39, the image branch 13 patches out of 256, with different
    substitutions (median/mode row vs blur sigma=8) and different insertion anchors.
    Normalisation makes windows comparable, not modalities.""")

print(f"\n{'ALL CHECKS PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
