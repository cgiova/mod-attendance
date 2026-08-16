"""Equivalence harness: notebook-inline faithfulness vs the extracted module.

Executes the post-hoc notebook's setup cells in a namespace (so the notebook itself is
the source of truth for both the data objects AND the reference implementation), then
runs `explanation_faithfulness` from both and diffs every returned metric and curve.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

BASE = "/mnt/disk1/home/cgiovannoni/isic24-dum"
os.chdir(BASE)
N_SAMPLES = int(os.environ.get("EQUIV_N", "16"))
METHODS = tuple(os.environ.get("EQUIV_METHODS", "tabular,image,joint").split(","))
RUNS = int(os.environ.get("EQUIV_RUNS", "1"))

nb = json.load(open("notebooks/modality_attendance/isic24_modality_attendance_posthoc.ipynb"))["cells"]
NB = {"__name__": "__nbmain__"}

SETUP = list(range(0, 26))          # title -> params -> data -> FS -> gradcam -> faithfulness core
t0 = time.time()
for i in SETUP:
    c = nb[i]
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    src = "\n".join(("#" + l if l.lstrip().startswith(("!", "%")) else l) for l in src.splitlines())
    try:
        exec(compile(src, f"<nb cell {i}>", "exec"), NB)
    except Exception as e:
        print(f"[setup cell {i}] {type(e).__name__}: {e}")
        raise
print(f"[setup] notebook cells 0-25 executed in {time.time()-t0:.0f}s", flush=True)

# ---- the module, configured from the very same objects -------------------------------
sys.path.insert(0, os.path.join(BASE, "src"))
from isic24 import xai_faithfulness as xf

xf.configure(xf.XaiContext(
    feat_cols=NB["FEAT_COLS"],
    training_df=NB["training_df"],
    validation_df=NB["validation_df"],
    validation_ds=NB["validation_ds"],
    feature_space=NB["feature_space"],
    to_feature_dict=NB["_to_feature_dict"],
    encoded_feature_names=NB["encoded_feature_names"],
    feature_space_file=NB["FEATURE_SPACE_FILE"],
    feature_space_fingerprint=NB["FEATURE_SPACE_FP"],
))

import keras, numpy as np, pandas as pd

# The notebook's own faithfulness code (cell 24) calls `encoded_indices_for_raw`, which the
# notebook only defines at cell 38 -- inside the ARCHIVED appendix, i.e. AFTER the cell-31
# driver that uses it.  Run the post-hoc notebook linearly today and the tabular method
# raises NameError.  Inject it so the reference implementation can run at all; it is
# byte-identical to the module's copy (verified against cells 26/33/38).
from isic24.xai_engines import encoded_indices_for_raw as _eifr
NB.setdefault("encoded_indices_for_raw", _eifr)

master = pd.read_csv(os.path.join(NB["SUMMARIES_PATH"], "master_results_rows.csv"))
# one run per family, so the custom-layer deserialization path is exercised too
# only runs that actually have SHAP arrays: the 20 learnable init!=0.5 runs never had XAI
# enabled, so the tabular method legitimately raises for them in BOTH implementations.
import glob as _g
_has = master.run_dir.map(lambda d: bool(_g.glob(os.path.join(
    os.environ.get("ART", "artifacts/enhanced"),
    str(d).split("artifacts/enhanced/")[-1], "xai", "shap", "epoch_*", "shap_arrays_*.npz"))))
picks = master[_has].groupby("impl_name").head(1)
if os.environ.get("EQUIV_ONLY"):
    picks = picks[picks.impl_name.isin(os.environ["EQUIV_ONLY"].split(","))]
picks = picks.head(RUNS)

TOL_DET = 1e-6      # deterministic quantities must match exactly
TOL_GAIN = 0.04     # measured per-run replay ceiling (REPORT sec.9)

allbad = []
for _, r in picks.iterrows():
    rd = os.path.join(NB["ART_PATH"], str(r.run_dir).split("artifacts/enhanced/")[-1])
    print(f"\n=== {r.impl_name} / {r.run_id} (epoch {r.best_epoch}) ===", flush=True)
    model = keras.models.load_model(
        os.path.join(rd, "checkpoints", f"best_{r.model_name}.keras"),
        compile=False, safe_mode=False)

    common = dict(n_samples=N_SAMPLES, methods=METHODS, save=False, verbose=False)

    def _call(fn):
        t = time.time()
        try:
            return fn(model, rd, int(r.best_epoch), **common), time.time() - t
        except Exception as e:
            return f"{type(e).__name__}: {e}", time.time() - t

    ref, t_ref = _call(NB["explanation_faithfulness"])
    got, t_mod = _call(xf.explanation_faithfulness)
    if isinstance(ref, str) or isinstance(got, str):
        same = ref == got
        print(f"  both raised identically: {same}\n    ref={str(ref)[:90]}")
        if not same:
            allbad.append((r.run_id, "exception", ref, got))
        keras.backend.clear_session()
        continue
    print(f"  notebook {t_ref:.0f}s | module {t_mod:.0f}s")

    keys = sorted(set(ref) | set(got))
    bad = []
    for k in keys:
        if k == "faith_seconds":      # wall clock, not a result
            continue
        a, b = ref.get(k), got.get(k)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            d = abs(float(a) - float(b))
            tol = TOL_DET if not k.endswith("_gain") else TOL_GAIN
            flag = "" if d <= tol else "  <-- MISMATCH"
            if d > 0 or flag:
                print(f"  {k:44s} {a:+.10f} vs {b:+.10f}  d={d:.2e}{flag}")
            if flag:
                bad.append((k, a, b, d))
        elif a != b:
            print(f"  {k:44s} differs (non-numeric)")
            bad.append((k, a, b, None))
    print("  " + ("ALL MATCH" if not bad else f"{len(bad)} MISMATCHES"))
    allbad += [(r.run_id,) + t for t in bad]
    keras.backend.clear_session()

print("\n" + "=" * 70)
print("EQUIVALENCE:", "PASS" if not allbad else f"FAIL ({len(allbad)})")
for x in allbad:
    print("  ", x)
