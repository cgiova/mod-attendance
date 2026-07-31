"""LORE benchmark: the stack we are on vs the batched rewrite, same checkpoint, same rows.

Replaces the "~27x slower" figure in REPORT sec.9 / REPORT_S sec.3.3, which rests on no
completed genetic run -- the only trace on disk is a Ctrl-C'd traceback at 1 sample of 4.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

BASE = "/mnt/disk1/home/cgiovannoni/isic24-dum"
os.chdir(BASE)
N_SAMPLES = int(os.environ.get("BENCH_N", "2"))
SUBSET = int(os.environ.get("BENCH_SUBSET", "64"))

nb = json.load(open("notebooks/modality_attendance/isic24_modality_attendance_posthoc.ipynb"))["cells"]
NB = {"__name__": "__nbmain__"}
sys.path.insert(0, os.path.join(BASE, "src"))
for i, c in enumerate(nb):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    src = "\n".join(("#" + l if l.lstrip().startswith(("!", "%")) else l) for l in src.splitlines())
    exec(compile(src, f"<nb {i}>", "exec"), NB)
    if "xf.configure" in src:
        break
print("[setup] done", flush=True)

from isic24 import xai_faithfulness as xf
from isic24.eval_guard import assert_canonical_eval

# Lo studio primario si misura SEMPRE sul validation canonico: i knob EVAL_SET /
# EVAL_FOLDS / FAITH_N_SAMPLES restano usabili, ma non da qui.
assert_canonical_eval(NB)

from isic24.xai_lore_sa import LoreEngineSA
import keras, numpy as np, pandas as pd

RUN = "artifacts/enhanced/runs/vanilla/20260604-210304/seed_47/baseline__e20"
cfg = json.load(open(os.path.join(RUN, "config.json")))
model = keras.models.load_model(
    os.path.join(RUN, "checkpoints", f"best_{cfg['model_name']}.keras"),
    compile=False, safe_mode=False)

raw_df, imgs, labels = xf._build_faith_subset(SUBSET)
print(f"[subset] {len(raw_df)} rows, {int(np.sum(labels))} positive "
      f"(neigh_pos_rate is the driver of the random generator's cost)", flush=True)

results = []

# ---------------- lore_sa, batched genetic + random ----------------------------------
for neigh in ("genetic", "random"):
    print(f"\n=== lore_sa / {neigh} ===", flush=True)
    eng = LoreEngineSA(
        feature_space=NB["feature_space"], feat_cols=NB["FEAT_COLS"],
        to_feature_dict=NB["_to_feature_dict"], training_df=NB["training_df"],
        neigh_type=neigh, num_instances=1000)
    try:
        stats, df = eng.explain(model, raw_df, imgs, labels, max_explain_samples=N_SAMPLES)
        print("   ", stats, flush=True)
        results.append(stats)
    except Exception as e:
        print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
        results.append({"engine": "lore_sa", "neigh_type": neigh,
                        "error": f"{type(e).__name__}: {e}"})

# ---------------- lore_explainer (the stack in production) ---------------------------
from isic24.xai_engines import LoreEngine

for neigh in ("random", "genetic"):
    print(f"\n=== lore_explainer / {neigh} (current stack) ===", flush=True)
    eng = LoreEngine(
        feature_space=NB["feature_space"], feat_cols=NB["FEAT_COLS"],
        max_explain_samples=N_SAMPLES, positive_threshold=0.5,
        lore_config={"neigh_type": neigh, "size": 300}, lore_random_state=0,
        oos_samples=200)
    t0 = time.time()
    try:
        stats, rules, preds = eng.explain(
            raw_df=raw_df.iloc[:N_SAMPLES], images=imgs[:N_SAMPLES],
            labels=labels[:N_SAMPLES], model=model, mode="context")
        dt = time.time() - t0
        row = {"engine": "lore_explainer", "neigh_type": neigh, "n_explained": N_SAMPLES,
               "min_per_sample": dt / 60.0 / N_SAMPLES,
               "fidelity_mean": stats.get("fidelity"),
               "rule_len_mean": stats.get("rule_len")}
        print("   ", row, flush=True)
        results.append(row)
    except Exception as e:
        dt = time.time() - t0
        print(f"    FAILED after {dt/60:.1f} min: {type(e).__name__}: {e}", flush=True)
        results.append({"engine": "lore_explainer", "neigh_type": neigh,
                        "min_per_sample": dt / 60.0 / max(1, N_SAMPLES),
                        "error": f"{type(e).__name__}: {e}"})

out = pd.DataFrame(results)
print("\n" + "=" * 78)
print(out.to_string(index=False))
out.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_lore.csv"),
           index=False)
