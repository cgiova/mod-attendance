"""Genera gli array SHAP per i run canonici che non li hanno.

Sono **20** su 110, tutti `learnable_scaling` con `init_alpha != 0.5`
(`init_a0p100/300/700/900`, 5 seed ciascuno).

PERCHE' MANCANO -- non sono run falliti ne' saltati. Nel notebook di training:

    XAI_ALPHAS     = ALPHAS         # fixed_scaling      -> tutti e 9 i valori
    XAI_INIT_ALPHA = {0.5}          # learnable_scaling  -> SOLO il valore di riferimento
    XAI_LAMBDAS    = LAMBDA_REGS    # loss_regularization-> tutti e 5 i valori

e la cella della griglia passa
`enable_xai_fn=lambda s: round(float(s["init_alpha"]), 3) in XAI_INIT_ALPHA`.  Le due griglie
sorelle passano la lista completa, quindi hanno la XAI su 45/45 e 25/25 run.  E' una singola
riga asimmetrica rispetto alle altre due, non un guasto.

La conseguenza pratica e' che ogni analisi che parte dagli array salvati li salta -- per esempio
`run_signed.py`, che filtra sui run con `xai/shap/epoch_*/shap_arrays_*.npz` e quindi vede 90
run invece di 110.  Tutti e 20 hanno pero' il checkpoint, quindi si rimedia post-hoc senza
ri-allenare nulla.  (Hanno gia' `xai/faithfulness/`, prodotto dal loop post-hoc tramite il
fallback di ricalcolo di `_load_shap_values`: e' l'unica sottodirectory XAI che possiedono.)

**E' additivo e non sovrascrive niente.** Verificato che tutti e 20 hanno il checkpoint e
**zero** artefatti XAI (shap, gradcam e lore): `xai_driver.assert_writable` li consente proprio
perche' non c'e' niente da distruggere. Gli altri 90 restano intatti -- se per errore questo
script puntasse su uno di loro, il guard lo fermerebbe.

Costo: ~1.5 min per run, quindi ~30 min in totale. Interrompibile: rilanciandolo salta i run
che nel frattempo hanno gia' gli array.

    python scripts/xai_2026-07-28/fill_missing_shap.py
    ENGINES=shap,gradcam python scripts/xai_2026-07-28/fill_missing_shap.py
    DRY_RUN=1 python scripts/xai_2026-07-28/fill_missing_shap.py

LORE NON e' incluso di proposito: a 32 campioni costa ~1 h per run (~20 h in totale), e va
deciso dopo il cantiere A -- vedi `bench_lore_quality.py`.

FRAGILE: come gli altri harness qui, esegue le celle del post-hoc fino a `xf.configure`.
"""
import gc
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

BASE = "/mnt/disk1/home/cgiovannoni/isic24-dum"
os.chdir(BASE)
ENGINES = tuple(os.environ.get("ENGINES", "shap").split(","))
DRY_RUN = bool(int(os.environ.get("DRY_RUN", "0")))

nb = json.load(open("notebooks/modality_attendance/"
                    "isic24_modality_attendance_posthoc.ipynb"))["cells"]
NB = {"__name__": "__nbmain__"}
sys.path.insert(0, os.path.join(BASE, "src"))
for i, c in enumerate(nb):
    if c["cell_type"] != "code":
        continue
    src = "\n".join(("#" + l if l.lstrip().startswith(("!", "%")) else l)
                    for l in "".join(c["source"]).splitlines())
    exec(compile(src, f"<nb {i}>", "exec"), NB)
    if "xf.configure" in src:
        break
print("[setup] done", flush=True)

from isic24 import xai_driver as xd
from isic24.eval_guard import assert_canonical_eval

# Gli array SHAP sono indicizzati sull'ordinamento del fold 0: generarli su un eval set
# diverso li renderebbe incompatibili con tutto il resto senza che si veda.
assert_canonical_eval(NB)

import keras
import pandas as pd

ART = NB["ART_PATH"]
master = pd.read_csv(os.path.join(NB["SUMMARIES_PATH"], "master_results_rows.csv"))


def run_dir_of(d):
    return os.path.join(ART, str(d).split("artifacts/enhanced/")[-1])


def has_shap(d):
    return bool(glob.glob(os.path.join(run_dir_of(d), "xai", "shap", "epoch_*",
                                       "shap_arrays_*.npz")))


todo = master[~master.run_dir.map(has_shap)].reset_index(drop=True)
print(f"[runs] {len(todo)}/{len(master)} senza array SHAP", flush=True)
if len(todo):
    print(todo.groupby(["impl_name", "setting_tag"]).size().to_string(), flush=True)

# Cintura di sicurezza indipendente dal guard: nessuno di questi deve avere artefatti.
busy = [r.run_id for _, r in todo.iterrows()
        if xd.existing_xai_artifacts(run_dir_of(r.run_dir), ENGINES)]
if busy:
    raise SystemExit(f"[ABORT] {len(busy)} run hanno gia' artefatti per {ENGINES}: {busy[:3]}. "
                     "Non e' piu' un riempimento additivo, fermati e guarda cosa e' cambiato.")

results = {}
for i, r in todo.iterrows():
    rd = run_dir_of(r.run_dir)
    t0 = time.time()
    try:
        results[r.run_id] = xd.run_posthoc_xai(
            rd, engines=ENGINES, best_epoch=int(r.best_epoch),
            best_val_auc=float(r.best_val_auc), dry_run=DRY_RUN, verbose=False)
        ok = all(v in ("ok", "dry-run") for v in results[r.run_id].values())
    except Exception as e:
        results[r.run_id] = {"_run": f"failed: {type(e).__name__}: {e}"}
        ok = False
    finally:
        if not DRY_RUN:
            keras.backend.clear_session()
            gc.collect()
    print(f"[{i+1}/{len(todo)}] {'ok ' if ok else 'FAIL'} {r.run_id} "
          f"({time.time()-t0:.0f}s) {results[r.run_id]}", flush=True)

after = int(master.run_dir.map(has_shap).sum())
print(f"\nrun con array SHAP: {after}/{len(master)}")
failed = [k for k, v in results.items() if not all(x in ("ok", "dry-run") for x in v.values())]
print("falliti:", failed or "nessuno")
