"""Quale generatore di vicinato usare: velocita' E qualita' della spiegazione.

`bench_lore.py` ha misurato solo il TEMPO (genetic 113x piu' veloce). Quello non basta a
scegliere: con i default, `lore_sa` genetic produce `rule_len = 1` e `fidelity = 1.0` su ogni
campione, contro le 9.58 premesse del legacy. Le cause sono note (Addendum in
`BENCH_lore_2026-07-28.md`):

  * la fedelta' di `lore_sa` e' `dt.score(Z, Yb)` su un albero non potato, cioe' accuratezza
    di training: 1.0 per costruzione;
  * il vicinato genetico e' saturo di duplicati perche' `record_init` restituisce `x`
    invariato, quindi la popolazione iniziale e' l'istanza ripetuta.

Questo script mette a confronto le configurazioni sulle colonne che decidono:

    min_per_sample        costo
    neigh_unique_frac     frazione di righe DISTINTE nel vicinato   <- la diagnosi
    neigh_top_frac        peso della riga piu' frequente
    neigh_pos_rate        bilanciamento delle classi
    rule_len              lunghezza della regola
    fid_cv_bacc           fedelta' su fold tenuti fuori             <- discrimina davvero
    fid_insample          ~1.0 ovunque, tenuta come controllo

FRAGILE: come gli altri harness qui, esegue le celle del post-hoc fino a `xf.configure`.

    python scripts/xai_2026-07-28/bench_lore_quality.py
    BENCH_N=4 BENCH_SUBSET=64 python scripts/xai_2026-07-28/bench_lore_quality.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

BASE = "/mnt/disk1/home/cgiovannoni/isic24-dum"
os.chdir(BASE)
N_SAMPLES = int(os.environ.get("BENCH_N", "2"))
SUBSET = int(os.environ.get("BENCH_SUBSET", "64"))
RUN = os.environ.get("BENCH_RUN",
                     "artifacts/enhanced/runs/vanilla/20260604-210304/seed_47/baseline__e20")

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

from isic24 import xai_faithfulness as xf
from isic24.eval_guard import assert_canonical_eval

assert_canonical_eval(NB)

import keras
import numpy as np
import pandas as pd

from isic24.xai_lore_sa import LoreEngineSA

cfg = json.load(open(os.path.join(RUN, "config.json")))
model = keras.models.load_model(
    os.path.join(RUN, "checkpoints", f"best_{cfg['model_name']}.keras"),
    compile=False, safe_mode=False)

raw_df, imgs, labels = xf._build_faith_subset(SUBSET)
print(f"[subset] {len(raw_df)} righe, {int(np.sum(labels))} positive | "
      f"{N_SAMPLES} campioni spiegati per configurazione", flush=True)

# Le configurazioni.  Misurato su modello sintetico (2026-07-29), a `num_instances=300`:
#
#   config                 unique_frac  rule_len  cv_bacc
#   default                      0.244         1    0.983
#   init.3                       0.246         3    0.970
#   init.3 + mutpb .5            0.714         1    0.880
#   init.5 + mutpb .8            0.967         3    0.748
#
# Due cose che cambiano come si legge la tabella:
#   1. la leva che rompe la saturazione di duplicati e' `mutpb`, NON `init_mutpb` da solo:
#      diversificare la popolazione iniziale allunga la regola ma il GA ricollassa;
#   2. c'e' un COMPROMESSO, non un ottimo: piu' il vicinato e' diverso, piu' la fedelta'
#      scende. Un vicinato stretto e' banale da approssimare (fedelta' alta e vuota); uno
#      troppo largo smette di essere *locale*, che e' il presupposto di LORE. A mutpb .8 la
#      mutazione campiona uniformemente sull'intero range della feature.
# Quindi lo scopo di questo sweep non e' trovare "il migliore" ma scegliere un punto
# consapevolmente, su dati veri e non sintetici.
# Lo sweep ISOLA `mutpb`: le prime quattro righe cambiano solo quello, così la tabella può
# davvero attribuirgli l'effetto. La quinta aggiunge `init_mutpb` sopra il miglior `mutpb`
# per rispondere separatamente a "l'init diversificata aggiunge qualcosa?" -- sul sintetico
# da sola non aggiungeva nulla (unique_frac 0.244 -> 0.246).
CONFIGS = [
    ("genetic default (mutpb .2)", dict(neigh_type="genetic", num_instances=1000)),
    ("genetic mutpb .35",     dict(neigh_type="genetic", num_instances=1000,
                                   gen_kwargs={"mutpb": 0.35})),
    ("genetic mutpb .5",      dict(neigh_type="genetic", num_instances=1000,
                                   gen_kwargs={"mutpb": 0.5})),
    ("genetic mutpb .8",      dict(neigh_type="genetic", num_instances=1000,
                                   gen_kwargs={"mutpb": 0.8})),
    ("genetic mutpb .5 + init .3", dict(neigh_type="genetic", num_instances=1000,
                                        init_mutpb=0.3, gen_kwargs={"mutpb": 0.5})),
    ("random",                dict(neigh_type="random", num_instances=1000)),
]

COLS = ["min_per_sample", "neigh_unique_frac", "neigh_top_frac", "neigh_pos_rate",
        "rule_len", "fid_cv_bacc", "fid_insample", "n_degenerate"]

rows = []
for tag, kw in CONFIGS:
    print(f"\n=== {tag} ===", flush=True)
    try:
        eng = LoreEngineSA(feature_space=NB["feature_space"], feat_cols=NB["FEAT_COLS"],
                           to_feature_dict=NB["_to_feature_dict"], training_df=NB["training_df"],
                           **kw)
        stats, df = eng.explain(model, raw_df, imgs, labels,
                                max_explain_samples=N_SAMPLES, verbose=True)
        row = {"config": tag, "n_degenerate": stats["n_degenerate"]}
        for c in COLS:
            if c == "n_degenerate":
                continue
            row[c] = float(df[c].dropna().mean()) if c in df and df[c].notna().any() else None
        rows.append(row)
    except Exception as e:
        print(f"    FALLITA: {type(e).__name__}: {e}", flush=True)
        rows.append({"config": tag, "error": f"{type(e).__name__}: {e}"})

out = pd.DataFrame(rows)
print("\n" + "=" * 100)
print(out.round(4).to_string(index=False))
print("""
Come si legge:
  neigh_unique_frac  vicino a 1 = vicinato sano; sotto ~0.5 le regole sono corte per artefatto
  fid_cv_bacc        0.5 = il surrogato non generalizza; sopra = la regola dice qualcosa
  fid_insample       ~1.0 ovunque: e' il controllo, non un risultato

NON cercare il massimo di una colonna sola: unique_frac e fid_cv_bacc si muovono in DIREZIONI
OPPOSTE. Un vicinato stretto da' fedelta' alta e vuota, uno largo smette di essere locale.
Il punto da scegliere e' il piu' diverso fra quelli che tengono fid_cv_bacc ben sopra 0.5 e
neigh_pos_rate non collassato.

CAVEAT sul fid_cv_bacc, da tenere presente proprio nelle righe con unique_frac basso: con un
vicinato saturo di duplicati la cross-validation e' OTTIMISTICAMENTE DISTORTA, perche' lo
stesso punto ripetuto finisce su entrambi i lati dello split e il fold "tenuto fuori" e' in
parte memorizzato. Cioe' e' meno affidabile esattamente dove la diagnosi e' peggiore. Va letto
accanto a unique_frac, mai da solo. Una fedelta' fuori campione piena richiede un vicinato
ridisegnato (come fa il legacy) e costa un secondo giro del generatore per campione.

Riferimento legacy (random 1000, 10 run canonici): rule_len 9.58, fidelity in-sample 0.983.
NB: legacy e lore_sa non spiegano la stessa funzione (embedding immagine condiviso vs per-campione).
""")
dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_lore_quality.csv")
out.to_csv(dest, index=False)
print("scritto", dest)
