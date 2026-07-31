"""Workstream B verification: is the new attributor actually aligned to the test?

(i)   efficiency: sum_i phi_i == logit(p(x)) - logit(p(baseline))   [single-row masker]
(ii)  the same check in PROBABILITY space, to show it fails there -- i.e. that returning
      the sigmoid (what Stage 2 did) is a real misalignment, not a cosmetic one
(iii) posev > 0 and negev (sign-flipped) > 0
(iv)  correlation of the new `abs` gain with the published faithfulness_aligned_attr.csv
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

BASE = "/mnt/disk1/home/cgiovannoni/isic24-dum"
os.chdir(BASE)
N = int(os.environ.get("SIGNED_N", "16"))

nb = json.load(open("notebooks/modality_attendance/isic24_modality_attendance_posthoc.ipynb"))["cells"]
NB = {"__name__": "__nbmain__"}
for i, c in enumerate(nb):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if "run_posthoc_xai" in src and "xf.configure" in src:
        break                                  # stop right after the import/configure cell
    src = "\n".join(("#" + l if l.lstrip().startswith(("!", "%")) else l) for l in src.splitlines())
    exec(compile(src, f"<nb {i}>", "exec"), NB)
# execute the import/configure cell too
sys.path.insert(0, os.path.join(BASE, "src"))
exec(compile("".join(c["source"]), "<nb configure>", "exec"), NB)
print("[setup] done", flush=True)

from isic24 import xai_faithfulness as xf
import keras, numpy as np, pandas as pd

RUN = "artifacts/enhanced/runs/vanilla/20260604-210304/seed_47/baseline__e20"
cfg = json.load(open(os.path.join(RUN, "config.json")))
model = keras.models.load_model(
    os.path.join(RUN, "checkpoints", f"best_{cfg['model_name']}.keras"),
    compile=False, safe_mode=False)

raw_df, imgs, labels = xf._build_faith_subset(N)
print(f"[subset] {len(raw_df)} samples, {int(np.sum(labels))} positive", flush=True)

t = time.time()
phi, meta = xf.aligned_tab_attributions(model, raw_df, imgs, background="baseline")
print(f"[attr] baseline masker: {time.time()-t:.0f}s  meta={meta}", flush=True)

print("\n(i) EFFICIENCY in LOGIT space (the aligned claim)")
eff = xf.check_efficiency(model, raw_df, imgs, phi)
print("   ", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in eff.items()})

print("\n(ii) the same sum against the PROBABILITY-space difference (Stage 2's f())")
base_df = pd.DataFrame([xf._FAITH_TAB_BASELINE] * len(raw_df))[list(NB["FEAT_COLS"])].reset_index(drop=True)
p_x = 1 / (1 + np.exp(-xf._score_logit(model, xf._encode_raw(raw_df[NB["FEAT_COLS"]]), imgs)))
p_b = 1 / (1 + np.exp(-xf._score_logit(model, xf._encode_raw(base_df), imgs)))
err_p = np.abs(phi.sum(1) - (p_x - p_b))
print(f"    max_abs_err={err_p.max():.6f}  mean={err_p.mean():.6f}   <- large: the two "
      f"scales are NOT interchangeable")

print("\n(iii) sign conventions")
res, _ = xf.aligned_signed_faithfulness(
    model, RUN, int(cfg["best_epoch"]), phi=phi, labels=labels, raw_df=raw_df, imgs=imgs,
    n_samples=N, out_dir="/tmp/claude-1006/-mnt-disk1-home-cgiovannoni-isic24-dum/ca061703-cbca-4878-8a7b-c8b234e92ddb/scratchpad/signed_out",
    save=True, verbose=True)
print("\n   full result:")
for k, v in sorted(res.items()):
    if k != "aligned_meta":
        print(f"     {k:44s} {v}")

print("\n(iv) published aligned column, same run (background=training, prob space):")
pub = pd.read_csv("artifacts/enhanced/runs/_robustness/stage2_aligned_attr/faithfulness_aligned_attr.csv")
row = pub[pub.run_id == "20260604-210304/seed_47__baseline__e20"]
print(row[["tab_gain_aligned", "tab_gain_published"]].to_string(index=False))
print(f"    new abs del_gain (n={N}, baseline masker, logit) = {res.get('aligned_abs_del_gain'):+.4f}")
print("    -> expected to differ: masker AND score space both changed (plus n).")
