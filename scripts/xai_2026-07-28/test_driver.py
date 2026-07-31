"""Driver smoke test (plan verification A): does run_posthoc_xai actually work standalone,
and is a re-run idempotent?  Also checks make_probe_batch == the old _materialize_probe."""
import json, os, shutil, sys, tempfile, warnings
warnings.filterwarnings("ignore"); os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","3")
BASE="/mnt/disk1/home/cgiovannoni/isic24-dum"; os.chdir(BASE); sys.path.insert(0,BASE+"/src")

nb=json.load(open("notebooks/modality_attendance/isic24_modality_attendance_posthoc.ipynb"))["cells"]
NB={"__name__":"__nbmain__"}
for c in nb:
    if c["cell_type"]!="code": continue
    s="".join(c["source"])
    exec(compile("\n".join(("#"+l if l.lstrip().startswith(("!","%")) else l) for l in s.splitlines()),"<nb>","exec"),NB)
    if "xf.configure" in s: break
print("[setup] done",flush=True)

import numpy as np, pandas as pd, keras
from isic24 import xai_driver as drv

# ---- (0) probe equivalence: module's lazy default vs the notebook's old builder --------
xb, yb = drv.make_probe_batch(n=8)
probe_ds = NB["build_dataset"](
    NB["validation_ids"][:8], NB["training_validation_hdf5"],
    {k: v[:8] for k, v in NB["validation_features"].items()}, NB["validation_labels"][:8],
    batch_size=8, shuffle=False, augment=False, cache=False, drop_remainder=False)
probe_ds = probe_ds.map(lambda x, y: ({'images': x['images'],
                                       'features': NB["feature_space"](x['features'])}, y))
xo, yo = next(iter(probe_ds))
same_img = np.allclose(xb["images"].numpy(), xo["images"].numpy())
same_feat = np.allclose(xb["features"].numpy(), xo["features"].numpy())
print(f"(0) probe batch == old _materialize_probe(8): images={same_img} features={same_feat}")

# ---- run into a TEMP run_dir so no canonical artifact is touched ----------------------
SRC="artifacts/enhanced/runs/vanilla/20260604-210304/seed_47/baseline__e20"
tmp=tempfile.mkdtemp(prefix="posthoc_xai_")
os.makedirs(os.path.join(tmp,"checkpoints"))
cfg=json.load(open(os.path.join(SRC,"config.json")))
shutil.copy(os.path.join(SRC,"config.json"), os.path.join(tmp,"config.json"))
shutil.copy(os.path.join(SRC,"checkpoints",f"best_{cfg['model_name']}.keras"),
            os.path.join(tmp,"checkpoints",f"best_{cfg['model_name']}.keras"))
print(f"[tmp run_dir] {tmp}")

ENG=("gradcam","shap")     # LORE skipped: 32 samples of 'random' is ~1h, benchmarked separately

def rowcount(p): return len(pd.read_csv(p)) if os.path.exists(p) else 0
shap_csv=os.path.join(tmp,"xai","shap","shap_summary.csv")

print("\n(1) first run")
s1=drv.run_posthoc_xai(tmp, engines=ENG, verbose=True)
print("   ",s1)
n1=rowcount(shap_csv)
gc1=sorted(os.listdir(os.path.join(tmp,"xai","gradcam")))
print(f"    shap_summary rows={n1} | gradcam files={gc1}")

print("\n(2) re-run with overwrite=False -> should SKIP")
s2=drv.run_posthoc_xai(tmp, engines=ENG, verbose=True)
print("   ",s2)

print("\n(3) re-run with overwrite=True -> must NOT duplicate the epoch row")
s3=drv.run_posthoc_xai(tmp, engines=ENG, overwrite=True, verbose=True)
n3=rowcount(shap_csv)
print(f"    shap_summary rows: {n1} -> {n3}")

status=json.load(open(os.path.join(tmp,"xai","xai_status.json")))
print("\n(4) xai_status.json:", json.dumps(status))

ok = (all(v=="ok" for v in s1.values()) and n1==n3==1 and same_img and same_feat
      and all(v=="ok" for v in status["engines"].values()))
print("\nDRIVER SMOKE TEST:", "PASS" if ok else "FAIL")
print("  artifacts produced:", {d: sorted(os.listdir(os.path.join(tmp,"xai",d)))
                                for d in os.listdir(os.path.join(tmp,"xai"))
                                if os.path.isdir(os.path.join(tmp,"xai",d))})
shutil.rmtree(tmp)
