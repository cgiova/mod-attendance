# %load scripts/xai_2026-07-28/cell_fill_missing_shap.py
# ======================================================================================
# Riempie gli array SHAP dei 20 run canonici che non li hanno.
#
# Da incollare nel notebook POST-HOC, dopo la cella con `xf.configure` (la 23): usa il
# contesto gia' caricato -- stessa FeatureSpace, stesso validation canonico, stesso
# registro -- quindi non c'e' modo che produca artefatti su una base diversa dagli altri.
#
# Perche' mancano: nel notebook di training `XAI_INIT_ALPHA = {0.5}` limita la XAI al solo
# valore di riferimento della griglia learnable_scaling, mentre `XAI_ALPHAS = ALPHAS` e
# `XAI_LAMBDAS = LAMBDA_REGS` la danno a tutti i valori delle altre due. Non sono run
# falliti: hanno il checkpoint, e si rimedia post-hoc senza ri-allenare.
#
# DRY_RUN = True elenca cosa farebbe senza caricare un solo modello.
# ======================================================================================
DRY_RUN = False
ENGINES = ("shap",)          # LORE escluso: ~1 h/run. Grad-CAM: aggiungi "gradcam".

import gc, glob, os, time
import keras, pandas as pd
from isic24 import xai_driver as xd
from isic24.eval_guard import assert_canonical_eval

assert_canonical_eval(globals())          # gli array SHAP sono indicizzati sul fold 0

_master = pd.read_csv(os.path.join(SUMMARIES_PATH, "master_results_rows.csv"))
_rd = lambda d: os.path.join(ART_PATH, str(d).split("artifacts/enhanced/")[-1])
_has = lambda d: bool(glob.glob(os.path.join(_rd(d), "xai", "shap", "epoch_*",
                                             "shap_arrays_*.npz")))

_todo = _master[~_master.run_dir.map(_has)].reset_index(drop=True)
print(f"[runs] {len(_todo)}/{len(_master)} senza array SHAP")
if len(_todo):
    print(_todo.groupby(["impl_name", "setting_tag"]).size().to_string())

# Cintura di sicurezza indipendente dal write guard: nessuno dei selezionati deve avere
# gia' artefatti per questi engine, altrimenti non e' piu' un riempimento additivo.
_busy = [r.run_id for _, r in _todo.iterrows()
         if xd.existing_xai_artifacts(_rd(r.run_dir), ENGINES)]
assert not _busy, f"{len(_busy)} run hanno gia' artefatti per {ENGINES}: {_busy[:3]}"

for _i, _r in _todo.iterrows():
    _t0 = time.time()
    try:
        _out = xd.run_posthoc_xai(_rd(_r.run_dir), engines=ENGINES,
                                  best_epoch=int(_r.best_epoch),
                                  best_val_auc=float(_r.best_val_auc),
                                  dry_run=DRY_RUN, verbose=False)
        _ok = all(v in ("ok", "dry-run") for v in _out.values())
    except Exception as _e:
        _out, _ok = {"_run": f"failed: {type(_e).__name__}: {_e}"}, False
    finally:
        if not DRY_RUN:
            keras.backend.clear_session(); gc.collect()
    print(f"[{_i+1}/{len(_todo)}] {'ok  ' if _ok else 'FAIL'} {_r.run_id} "
          f"({time.time()-_t0:.0f}s) {_out}", flush=True)

print(f"\nrun con array SHAP: {int(_master.run_dir.map(_has).sum())}/{len(_master)}")
