"""Signed/class-conditioned aligned faithfulness across one run per family, n=128.

Writes to runs/_faithfulness/faithfulness_aligned_signed.csv. Nothing here touches the
canonical faithfulness.* artifacts or the published faithfulness_aligned_attr.csv.

FRAGILE: the setup below execs the code cells of isic24_modality_attendance_posthoc.ipynb
in order and stops at the first cell containing `xf.configure`. It has no other contract
with that notebook -- reorder the cells, rename the configure call, or move the data-loading
cells after it and this stops working (silently, if the missing names happen to be optional).
This is a one-off verification harness kept for reproducibility, not a supported entry point.
"""
import json, os, sys, time, warnings
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

from isic24 import xai_faithfulness as xf
from isic24.eval_guard import assert_canonical_eval

# Lo studio primario si misura SEMPRE sul validation canonico: i knob EVAL_SET /
# EVAL_FOLDS / FAITH_N_SAMPLES restano usabili, ma non da qui.
assert_canonical_eval(NB)

import keras, numpy as np, pandas as pd, gc

ART=NB["ART_PATH"]; OUTDIR=os.path.join(ART,"runs","_robustness","aligned_signed")
os.makedirs(OUTDIR,exist_ok=True)
OUT=os.path.join(ART,"runs","_faithfulness","faithfulness_aligned_signed.csv")

master=pd.read_csv(os.path.join(NB["SUMMARIES_PATH"],"master_results_rows.csv"))
import glob as g
has=master.run_dir.map(lambda d: bool(g.glob(os.path.join(ART,str(d).split("artifacts/enhanced/")[-1],
                                                          "xai","shap","epoch_*","shap_arrays_*.npz"))))
picks=master[has].groupby("impl_name").head(int(os.environ.get("PER_FAMILY","1")))
done=set(pd.read_csv(OUT).run_id) if os.path.exists(OUT) else set()
picks=picks[~picks.run_id.isin(done)]
print(f"[runs] {len(picks)} to do",flush=True)

raw_df,imgs,labels=xf._build_faith_subset(xf.FAITH_N_SAMPLES)
print(f"[subset] {len(raw_df)} samples, {int(np.sum(labels))} positive",flush=True)

for i,r in picks.reset_index(drop=True).iterrows():
    rd=os.path.join(ART,str(r.run_dir).split("artifacts/enhanced/")[-1])
    t0=time.time()
    try:
        model=keras.models.load_model(os.path.join(rd,"checkpoints",f"best_{r.model_name}.keras"),
                                      compile=False,safe_mode=False)
        phi,meta=xf.aligned_tab_attributions(model,raw_df,imgs,background="baseline")
        eff=xf.check_efficiency(model,raw_df,imgs,phi)
        res,_=xf.aligned_signed_faithfulness(
            model,rd,int(r.best_epoch),phi=phi,labels=labels,raw_df=raw_df,imgs=imgs,
            out_dir=os.path.join(OUTDIR,str(r.run_id)),save=True,verbose=False)
        row={"impl_name":r.impl_name,"run_id":r.run_id,"seed":r.seed,"setting_tag":r.setting_tag,
             "eff_max_abs_err":eff["max_abs_err"],"eff_corr":eff["corr"],"eff_passes":eff["passes"],
             "seconds":time.time()-t0,
             **{k:v for k,v in res.items() if k!="aligned_meta"}}
        pd.DataFrame([row]).to_csv(OUT,mode="a",header=not os.path.exists(OUT),index=False)
        print(f"[{i+1}/{len(picks)}] {r.impl_name:20s} abs={row.get('aligned_abs_del_gain'):+.4f} "
              f"posev={row.get('aligned_posev_del_gain'):+.4f} "
              f"negev={row.get('aligned_negev_del_gain_signflipped'):+.4f} "
              f"target={row.get('aligned_target_del_gain_labelsigned'):+.4f} "
              f"| eff_err={eff['max_abs_err']:.2e} ({time.time()-t0:.0f}s)",flush=True)
    except Exception as e:
        print(f"[FAIL] {r.run_id}: {type(e).__name__}: {e}",flush=True)
    finally:
        keras.backend.clear_session(); gc.collect()
print("\n",pd.read_csv(OUT).to_string(index=False) if os.path.exists(OUT) else "no output")
