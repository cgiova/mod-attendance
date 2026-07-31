"""Build the smoke/probe notebook: short cells to run BEFORE the long real runs."""
import json, os

CELLS = []

def md(s):  CELLS.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(keepends=True)})
def co(s):  CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": s.strip("\n").splitlines(keepends=True)})

md(r"""
# Celle di prova — da eseguire PRIMA dei run veri

Ogni cella verifica **un** pezzo della catena e stampa `PASS` / `FAIL` con il motivo.
Sono tutte corte: le più lente sono P1 (setup) e P6, sui 2–3 minuti; le altre stanno sotto il minuto.
**Non scrivono nulla nelle run canoniche.** P5 scrive in una directory `_smoke/` separata e
verifica md5/mtime dei canonici prima e dopo; P6 lavora su una dir temporanea in `/tmp` e
controlla che il write guard rifiuti le run canoniche; P8 cancella `_smoke/`. Tutto è
misurato sul **subset canonico** (prefisso delle prime 128 righe di validation), e P1 fallisce
se il post-hoc è rimasto configurato su un eval set esplorativo.

Esegui dall'alto in basso e fermati alla prima che fallisce. I costi in tabella sono su GPU; su
CPU girano lo stesso, più lente.

| cella | cosa conferma | costo |
|---|---|---|
| **P0** | su cosa girano le celle (informativa, non blocca) | ~5 s |
| **P1** | il setup del notebook post-hoc ricostruisce i globals | 1–3 min |
| **P2** | i 5 layer custom sono registrati → i checkpoint si deserializzano | ~1 min |
| **P3** | probe batch + forward pass coerenti | ~10 s |
| **P4** | **efficienza di Permutation** — l'attributore è allineato al test di deletion | ~30 s |
| **P5** | pipeline signed completa a n=8, 4 modi, e i file canonici restano intatti | ~40 s |
| **P6** | il write guard rifiuta le run canoniche; poi skip/overwrite su dir temporanea | ~2 min |
| **P7** | `lore_sa` gira e cosa risponde su `rule_len` (domanda aperta n.3) | ~1 min |
| **P8** | riepilogo + i comandi dei run veri | — |

P0 non blocca: senza GPU si prosegue su CPU.
""")

# ------------------------------------------------------------------ P0
md(r"""## P0 — Su cosa girano le celle

Informativa, **non un cancello**: se la GPU non c'è, tutto quello che segue gira su CPU e
funziona lo stesso, solo più lento. Le celle P1–P7 verificano l'idraulica — che il setup
ricostruisca i globals, che i checkpoint si deserializzino, che l'attributore sia allineato,
che i guard scattino — e nessuna di queste cose dipende dal dispositivo. I numeri che ne escono
non sono comunque da pubblicare (n=8 invece di 128).

L'unica regola che resta è la solita sui **run veri**: le righe di
`faithfulness_aligned_signed.csv` sono state calcolate su Quadro RTX 6000, e aggiungerne una
calcolata su CPU cambierebbe il substrato a metà di una tabella che serve a confrontare
famiglie fra loro. Se lo fai comunque, annotalo nel memo.

Al 29/07 la GPU non era visibile per un mismatch fra modulo kernel caricato (`580.159.03`) e
librerie userspace (`580.173.02`), con `dkms status` che riporta `580.173.02` già compilato per
il kernel in esecuzione — cioè un riavvio in sospeso, non un pacchetto rotto.""")
co(r'''
import os, glob
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("tensorflow", tf.__version__)
print("GPU visibili:", gpus)

# Il confronto che discrimina: modulo kernel caricato vs libreria userspace.
try:
    krn = open("/proc/driver/nvidia/version").read().split("Kernel Module")[1].split()[0]
except Exception as e:
    krn = f"(illeggibile: {e})"
lib = [os.path.basename(p).split(".so.")[-1]
       for p in glob.glob("/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.*") if ".so.1" not in p]
print("modulo kernel   :", krn)
print("libreria userspace:", lib)

ON_GPU = bool(gpus)
print("\nP0: GPU disponibile — le celle seguenti girano su GPU." if ON_GPU else
      "\nP0: nessuna GPU, si prosegue su CPU. Non e' un blocco: P1-P7 verificano l'idraulica e\n"
      "    non dipendono dal dispositivo. Metti in conto tempi piu' lunghi (P4/P5/P6 soprattutto).\n"
      "    Se le due versioni sopra differiscono e' un mismatch driver, non un problema del codice.")
''')

# ------------------------------------------------------------------ P1
md(r"""## P1 — Setup

Esegue le celle di codice del notebook post-hoc **in ordine, fino alla prima che contiene
`xf.configure`** (cella 23), e tiene i globals risultanti in `NB`. È esattamente il contratto
che usano `run_signed.py`, `bench_lore.py` ed `equiv.py`: se questa cella passa, passa il loro
setup.

Include il guard anti-drift della FeatureSpace, quindi un fallimento qui può anche voler dire
che la FeatureSpace non è più quella canonica (dim 127) — non solo che manca un import.

E include `assert_canonical_eval`, che è l'altro asse lungo cui i numeri smettono di essere
confrontabili: **su quali righe** sono calcolati. `EVAL_SET`, `EVAL_FOLDS` e `FAITH_N_SAMPLES`
restano knob veri e usabili, ma tutto ciò che tocca lo studio primario deve dichiarare di
essere sul set canonico e fallire rumorosamente se non lo è. Se hai lasciato il post-hoc su
`EVAL_SET="custom"` per un esperimento esplorativo, questa cella se ne accorge qui invece di
lasciarti scoprire dopo che i numeri non erano confrontabili.""")
co(r'''
import json, sys, time
BASE = "/mnt/disk1/home/cgiovannoni/isic24-dum"
os.chdir(BASE); sys.path.insert(0, BASE + "/src")

t0 = time.time()
nb = json.load(open("notebooks/modality_attendance/isic24_modality_attendance_posthoc.ipynb"))["cells"]
NB = {"__name__": "__nbmain__"}
stopped_at = None
for i, c in enumerate(nb):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    src = "\n".join(("#" + l if l.lstrip().startswith(("!", "%")) else l) for l in src.splitlines())
    exec(compile(src, f"<nb {i}>", "exec"), NB)
    if "xf.configure" in src:
        stopped_at = i
        break

from isic24 import xai_faithfulness as xf
from isic24.eval_guard import assert_canonical_eval
import numpy as np, pandas as pd, keras

# Lo studio primario si misura sempre sul validation canonico. Alza RuntimeError se il
# post-hoc e' rimasto configurato per un esperimento esplorativo.
EVAL_DESC = assert_canonical_eval(NB)

need = ["ART_PATH", "SUMMARIES_PATH", "feature_space", "FEAT_COLS", "training_df",
        "validation_df", "_to_feature_dict"]
missing = [k for k in need if k not in NB]
# _CTX parte come _Unconfigured(), NON come None: qualunque `is not None` passerebbe sempre.
ctx_ok = not isinstance(xf._CTX, xf._Unconfigured)
enc_dim = len(xf._CTX.encoded_feature_names) if ctx_ok else None

print(f"setup fermato alla cella {stopped_at} in {time.time()-t0:.0f}s")
print("globals mancanti:", missing or "nessuno")
print("xf._CTX configurato:", ctx_ok, "| encoded dim:", enc_dim)
print("validation_df:", len(NB.get("validation_df", [])), "righe | FEAT_COLS:", len(NB.get("FEAT_COLS", [])))
if enc_dim not in (None, 127):
    print(f"ATTENZIONE: encoded dim {enc_dim} != 127 canonico. Il guard della cella 18 non ha "
          "protestato, quindi controlla il manifest prima di fidarti dei numeri.")

# Il kernel puo' essere piu' vecchio dei moduli: e' successo, e si manifesta molto piu'
# tardi come una colonna che "non esiste". Meglio saperlo qui.
import pathlib
proc_start = os.path.getmtime(f"/proc/{os.getpid()}") if os.path.exists(f"/proc/{os.getpid()}") else 0
stale = [os.path.basename(f) for f in glob.glob(os.path.join(BASE, "src", "isic24", "*.py"))
         if os.path.getmtime(f) > proc_start]
if stale:
    print(f"\nATTENZIONE: {stale} sono stati modificati dopo l'avvio del kernel.\n"
          "  Il codice in memoria e' quello vecchio. Riavvia il kernel prima di fidarti\n"
          "  dei risultati (P7 ricarica xai_lore_sa da sola, gli altri no).")

ok = not missing and ctx_ok and bool(enc_dim)
print("\nP1 PASS" if ok else
      f"\nP1 FAIL — mancanti={missing}, ctx={ctx_ok}, enc_dim={enc_dim}")
''')

# ------------------------------------------------------------------ P2
md(r"""## P2 — Deserializzazione dei checkpoint

70 checkpoint su 110 (`fixed_scaling`, `learnable_scaling`, `controlled_fusion`,
`adaptive_gating`) contengono layer custom: senza `import isic24.mod_attendance` da qualche
parte nella sessione falliscono in silenzio. Qui ne carico **uno per famiglia** e verifico
anche che gli array SHAP salvati esistano — è il filtro che `run_signed.py` usa per scegliere
i run.""")
co(r'''
import isic24.mod_attendance  # registra i 5 layer custom
import gc

ART = NB["ART_PATH"]
master = pd.read_csv(os.path.join(NB["SUMMARIES_PATH"], "master_results_rows.csv"))

def run_dir_of(row):
    return os.path.join(ART, str(row.run_dir).split("artifacts/enhanced/")[-1])

has_shap = master.run_dir.map(lambda d: bool(glob.glob(os.path.join(
    ART, str(d).split("artifacts/enhanced/")[-1], "xai", "shap", "epoch_*", "shap_arrays_*.npz"))))
PICKS = master[has_shap].groupby("impl_name").head(1).reset_index(drop=True)
print(f"{int(has_shap.sum())}/{len(master)} run hanno array SHAP; scelto 1 per famiglia\n")

rows = []
for _, r in PICKS.iterrows():
    rd = run_dir_of(r)
    try:
        m = keras.models.load_model(os.path.join(rd, "checkpoints", f"best_{r.model_name}.keras"),
                                    compile=False, safe_mode=False)
        rows.append({"impl_name": r.impl_name, "setting_tag": r.setting_tag, "seed": r.seed,
                     "params": int(m.count_params()), "err": None})
        del m
    except Exception as e:
        rows.append({"impl_name": r.impl_name, "setting_tag": r.setting_tag, "seed": r.seed,
                     "params": None, "err": f"{type(e).__name__}: {e}"})
    finally:
        keras.backend.clear_session(); gc.collect()

chk = pd.DataFrame(rows)
print(chk.to_string(index=False))
bad = chk.err.notna().sum()
print("\nP2 PASS" if bad == 0 else f"\nP2 FAIL — {bad} checkpoint non deserializzabili (vedi colonna err)")
''')

# ------------------------------------------------------------------ P3
md(r"""## P3 — Probe batch e forward pass

`make_probe_batch` sostituisce il vecchio `_materialize_probe`: prende le prime `n` righe di
`validation_ds` (costruito con `shuffle=False`). Qui controllo che il batch abbia le chiavi
attese, che il modello lo mangi, e che le predizioni siano probabilità sensate.""")
co(r'''
from isic24 import xai_driver as xd

# RUN e BEST_EPOCH devono venire dalla STESSA riga, altrimenti P5 spiega un checkpoint
# all'epoca di un altro.
PROW = (PICKS[PICKS.impl_name.eq("vanilla")].iloc[0] if (PICKS.impl_name == "vanilla").any()
        else PICKS.iloc[0])
RUN, BEST_EPOCH = run_dir_of(PROW), int(PROW.best_epoch)
cfg = json.load(open(os.path.join(RUN, "config.json")))
model = keras.models.load_model(os.path.join(RUN, "checkpoints", f"best_{cfg['model_name']}.keras"),
                                compile=False, safe_mode=False)
print("run di prova:", os.path.relpath(RUN, ART))

xb, yb = xd.make_probe_batch(8)
print("chiavi del batch:", sorted(xb.keys()))
print("images:", tuple(xb["images"].shape), "| features:", tuple(xb["features"].shape))

p = model.predict(xb, verbose=0).reshape(-1)
print("predizioni:", np.round(p, 4))

ok = (set(xb.keys()) == {"images", "features"} and xb["images"].shape[0] == 8
      and np.all(np.isfinite(p)) and p.min() >= 0.0 and p.max() <= 1.0)
print("\nP3 PASS" if ok else "\nP3 FAIL — batch o predizioni non conformi")
''')

# ------------------------------------------------------------------ P4
md(r"""## P4 — Efficienza di Permutation (la prova che conta)

È il test che dice se stiamo spiegando **la stessa funzione** che il test di deletion
perturba. Con masker a riga singola e punteggio in logit deve valere

$$\sum_i \varphi_i \;=\; \mathrm{logit}\,p(x) - \mathrm{logit}\,p(\mathrm{baseline})$$

Atteso: `max_abs_err ~ 1e-6`, `corr ~ 1.000`.

La cella misura anche il **controllo negativo**: lo stesso confronto in spazio di probabilità
— quello che usava Stage 2 — deve sbagliare di parecchio (nei run veri: 1.55). Se anche quello
passa, allora qualcosa non sta usando il logit e il numero non significa quello che crediamo.

Gira a **n=8** invece di 128: serve a vedere se la macchina funziona, non a produrre un numero
pubblicabile.""")
co(r'''
# PREFISSO STRETTO del subset canonico: _build_faith_subset(n) prende le prime n righe di
# validation, quindi qualunque n <= 128 sono le stesse righe nello stesso ordine, solo meno.
# Non viene alzato da solo: il set su cui si misura non cambia mai senza che tu lo decida.
N_PROBE = 8
raw_df, imgs, labels = xf._build_faith_subset(N_PROBE)
both_classes = len(np.unique(labels.astype(int))) == 2
print(f"subset: prime {len(raw_df)} righe delle {xf.FAITH_N_SAMPLES} canoniche, "
      f"{int(np.sum(labels))} positive | entrambe le classi: {both_classes}")
if not both_classes:
    print("NOTA: una classe sola in queste righe, quindi il modo 'target' (segno per campione)\n"
          "      non verra' esercitato da P5. Se lo vuoi coprire, alza N_PROBE a mano: qualunque\n"
          "      valore <= 128 resta un prefisso del subset canonico, e il costo scala lineare.")

t0 = time.time()
phi, meta = xf.aligned_tab_attributions(model, raw_df, imgs, background="baseline")
print(f"attribuzioni {phi.shape} in {time.time()-t0:.0f}s | meta: {meta}")

eff = xf.check_efficiency(model, raw_df, imgs, phi)
print("\nspazio logit (atteso):", {k: (round(v, 9) if isinstance(v, float) else v)
                                   for k, v in eff.items()})

# Controllo negativo: stessa somma confrontata con la differenza di PROBABILITA'.
base = pd.DataFrame([xf._FAITH_TAB_BASELINE] * len(raw_df))[list(xf._CTX.feat_cols)].reset_index(drop=True)
sig = lambda s: 1.0 / (1.0 + np.exp(-s))
d_prob = sig(xf._score_logit(model, xf._encode_raw(raw_df[list(xf._CTX.feat_cols)]), imgs)) \
       - sig(xf._score_logit(model, xf._encode_raw(base), imgs))
err_prob = float(np.abs(phi.sum(axis=1) - d_prob).max())
print(f"spazio probabilita' (controllo negativo): max_abs_err = {err_prob:.4f}")

ok = eff["passes"] and eff["max_abs_err"] < 1e-4 and eff["corr"] > 0.9999 and err_prob > 0.1
print("\nP4 PASS" if ok else
      f"\nP4 FAIL — logit err={eff['max_abs_err']:.2e} corr={eff['corr']:.6f} "
      f"prob err={err_prob:.4f} (il controllo negativo deve FALLIRE, cioe' err > 0.1)")
''')

# ------------------------------------------------------------------ P5
md(r"""## P5 — Pipeline signed completa, a n=8

Gira i 4 modi (`abs`, `posev`, `negev`, `target`) dentro `_smoke/`, e poi verifica che
**nessun artefatto canonico sia stato toccato**: md5 su tutti i CSV riassuntivi sotto
`runs/_faithfulness/` e `runs/_robustness/` (incluso `faithfulness_aligned_attr.csv`, che sta in
`runs/_robustness/stage2_aligned_attr/` e non dove verrebbe da cercarlo, e
`faithfulness_aligned_signed.csv`, che prova che `out_dir=SMOKE` ha davvero isolato la
scrittura), più dimensione+mtime su tutti i `faithfulness.json` e `faithfulness_curves.npz`.

Attenzione ai segni, che è il punto in cui è facile sbagliare a mano:
`posev` nessun flip · `negev` flip **globale** · `target` flip **per campione**, applicato
alle curve *prima* di mediare. Perché `target` venga davvero esercitato servono **entrambe le
classi** negli 8 campioni: `_build_faith_subset` prende le prime righe della validation, quindi
è una proprietà di quelle righe, non del codice — P4 lo controlla e alza `N_PROBE` finché
entrambe compaiono (fino a 64).

Con n=8 i valori non sono confrontabili con la tabella dei run veri: quello che conta è che i
4 modi escano tutti senza eccezioni. `n_eff_min` viene stampato ma **non** usato come criterio
— un campione senza feature a evidenza positiva dà legittimamente `posev n_eff_min = 0`.""")
co(r'''
import hashlib, shutil

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# Tutti i CSV riassuntivi, ovunque siano: faithfulness_aligned_attr.csv sta sotto
# runs/_robustness/stage2_aligned_attr/, NON in runs/_faithfulness/. Elencarli a mano
# significa scartarne uno in silenzio quando il path non e' quello che credi.
CSVS = sorted(set(glob.glob(os.path.join(ART, "runs", "_faithfulness", "*.csv"))
                  + glob.glob(os.path.join(ART, "runs", "_robustness", "**", "*.csv"),
                              recursive=True)))
CSVS = [p for p in CSVS if os.sep + "_smoke" + os.sep not in p]
FILES = sorted(glob.glob(os.path.join(ART, "runs", "*", "*", "*", "*", "xai", "faithfulness",
                                      "faithfulness.json")))
FILES += sorted(glob.glob(os.path.join(ART, "runs", "*", "*", "*", "*", "xai", "faithfulness",
                                       "faithfulness_curves.npz")))
stamp = lambda p: (os.path.getsize(p), os.stat(p).st_mtime_ns)
before_csv = {p: md5(p) for p in CSVS}
before_f = {p: stamp(p) for p in FILES}
print(f"sotto osservazione: {len(CSVS)} CSV (md5) + {len(FILES)} json/npz (size+mtime)")

SMOKE = os.path.join(ART, "runs", "_robustness", "_smoke")
shutil.rmtree(SMOKE, ignore_errors=True); os.makedirs(SMOKE, exist_ok=True)

t0 = time.time()
res, _ = xf.aligned_signed_faithfulness(
    model, RUN, BEST_EPOCH, phi=phi, labels=labels, raw_df=raw_df, imgs=imgs,
    out_dir=SMOKE, save=True, verbose=True, n_samples=N_PROBE)
print(f"\n4 modi in {time.time()-t0:.0f}s")

gains = {k: round(v, 4) for k, v in res.items() if k.endswith(("_gain", "_gain_signflipped",
                                                              "_gain_labelsigned"))}
n_effs = {k: v for k, v in res.items() if "n_eff_min" in k}
print("gain :", gains)
print("n_eff:", n_effs)

changed = [os.path.relpath(p, ART) for p in CSVS if md5(p) != before_csv[p]]
changed += [os.path.relpath(p, ART) for p in FILES if stamp(p) != before_f[p]]
n_modes = len({k.split("_")[1] for k in gains})
ok = n_modes == 4 and len(gains) == 8 and not changed
print("\nP5 PASS" if ok else
      f"\nP5 FAIL — modi={n_modes}/4, gain={len(gains)}/8, canonici modificati={changed}")
if ok and not both_classes:
    print("      (ma 'target' non e' stato esercitato: una classe sola nel subset — vedi P4)")
''')

# ------------------------------------------------------------------ P6
md(r"""## P6 — Driver post-hoc: write guard, skip, niente duplicati

**Prima di tutto il write guard.** Il driver saltava una run solo se trovava
`xai/xai_status.json` con tutti gli engine `ok` — ma quel file lo scrive il driver *nuovo*, e
nessuna delle 110 run canoniche ce l'ha: vengono dal vecchio percorso in-training. Quindi
`run_posthoc_xai` puntato su una run canonica passava oltre il controllo **anche con
`overwrite=False`** e riscriveva `xai/shap/epoch_*`, `xai/gradcam/epoch_*.png`,
`xai/lore/epoch*` sul posto. 135 run hanno artefatti XAI su disco.

Da oggi il guard si basa sugli **artefatti**, non sul file di stato, ed è **per engine**:
rifiuta solo per gli engine che hanno già output propri. Riempire quello che manca resta
permesso — è additivo, ed è a cosa serve `run_posthoc_xai_batch`. Fuori scope:
`xai/faithfulness/`, che non lo scrive il driver ma `explanation_faithfulness`, governato dal
knob `FAITH_OVERWRITE` del loop post-hoc.

La prima cella verifica il rifiuto **e** i due bypass (`allow_clobber=True`,
`CANONICAL_WRITE_GUARD=False`): un guard che si disattiva da solo sarebbe peggio di nessun
guard. Nessuno dei due scrive niente.

Poi la parte vera, su una **run dir temporanea**: copio config + checkpoint altrove e giro
tre volte:

1. prima esecuzione → tutti gli engine `ok`;
2. `overwrite=False` → **skip**, senza ricaricare il modello;
3. `overwrite=True` **da solo** → deve essere fermato lo stesso. I due flag non sono lo stesso
   knob: `overwrite` dice *ricalcola* e disattiva la scorciatoia di `xai_status.json`,
   `allow_clobber` dice *puoi distruggere quello che c'è* e disattiva il guard. Ricalcolare in
   una dir vuota è ordinaria amministrazione, riscrivere una run che ha già prodotto i suoi
   artefatti no, e le due cose non devono condividere la stessa parola;
4. `overwrite=True` + `allow_clobber=True` → rifà davvero, ma `shap_summary.csv` deve restare a
   **1 riga**: i due CSV sono aperti in append e il dedupe di `append_epoch_row` è recente.

LORE è escluso: 32 campioni di `random` sono ~1 h, ed è già misurato a parte.""")
co(r'''
import tempfile, shutil

# (a) il write guard: deve rifiutare, e i suoi bypass devono scattare SOLO se richiesti
print("artefatti gia' presenti su", os.path.relpath(RUN, ART) + ":",
      {e: len(xd.existing_xai_artifacts(RUN, (e,))) for e in xd.ALL_ENGINES})
n_art = len(xd.existing_xai_artifacts(RUN))
try:
    xd.run_posthoc_xai(RUN, engines=("gradcam",), verbose=False)
    guard_ok = False
    print("!! il guard NON ha bloccato: la run canonica sarebbe stata sovrascritta")
except RuntimeError as exc:
    guard_ok = "WRITE GUARD" in str(exc)
    print("(a1) rifiuto:", str(exc).splitlines()[0])

# I due bypass, che devono restare espliciti. Nessuno dei due scrive niente qui.
bypass_ok = xd.assert_writable(RUN, allow_clobber=True) == []
xd.CANONICAL_WRITE_GUARD = False
knob_ok = xd.assert_writable(RUN) == []
xd.CANONICAL_WRITE_GUARD = True
empty_ok = xd.assert_writable(tempfile.mkdtemp()) == []
print(f"(a2) allow_clobber=True consente: {bypass_ok} | CANONICAL_WRITE_GUARD=False consente: "
      f"{knob_ok} | dir vuota consentita: {empty_ok}")

# (b) dry_run non scrive e non solleva, ma segnala che sarebbe bloccato
out_dry = xd.run_posthoc_xai(RUN, dry_run=True, verbose=True)
print("dry-run:", out_dry)
ck = sorted(glob.glob(os.path.join(RUN, "checkpoints", "best_*.keras")))

TMP = tempfile.mkdtemp(prefix="posthoc_xai_smoke_")
os.makedirs(os.path.join(TMP, "checkpoints"))
shutil.copy(os.path.join(RUN, "config.json"), os.path.join(TMP, "config.json"))
shutil.copy(ck[0], os.path.join(TMP, "checkpoints", os.path.basename(ck[0])))
print("run dir temporanea:", TMP)

ENG = ("gradcam", "shap")
shap_csv = os.path.join(TMP, "xai", "shap", "shap_summary.csv")
rowcount = lambda p: len(pd.read_csv(p)) if os.path.exists(p) else 0

print("\n(1) prima esecuzione"); s1 = xd.run_posthoc_xai(TMP, engines=ENG, verbose=True)
n1 = rowcount(shap_csv); print("   ", s1, "| shap_summary righe:", n1)

print("\n(2) overwrite=False -> deve saltare")
t0 = time.time(); s2 = xd.run_posthoc_xai(TMP, engines=ENG, verbose=True); dt = time.time() - t0
print(f"    {s2} in {dt:.2f}s")

print("\n(3a) overwrite=True da solo -> deve comunque essere fermato dal write guard")
try:
    xd.run_posthoc_xai(TMP, engines=ENG, overwrite=True, verbose=False)
    overwrite_blocked = False
    print("    !! passato: overwrite=True sta bypassando il guard, non dovrebbe")
except RuntimeError as exc:
    overwrite_blocked = "WRITE GUARD" in str(exc)
    print("    fermato, come deve:", str(exc).splitlines()[0])

print("\n(3b) overwrite=True + allow_clobber=True -> rifa' senza duplicare")
s3 = xd.run_posthoc_xai(TMP, engines=ENG, overwrite=True, allow_clobber=True, verbose=True)
n3 = rowcount(shap_csv); print(f"    shap_summary righe: {n1} -> {n3}")

status = json.load(open(os.path.join(TMP, "xai", "xai_status.json")))
print("\nxai_status.json:", json.dumps(status))
shutil.rmtree(TMP, ignore_errors=True)

ok = (guard_ok and n_art > 0 and bypass_ok and knob_ok and empty_ok and overwrite_blocked
      and out_dry == {e: "dry-run" for e in xd.ALL_ENGINES}
      and len(ck) == 1 and all(v == "ok" for v in s1.values()) and s2 == s1 and dt < 5
      and n1 == n3 == 1 and all(v == "ok" for v in status["engines"].values()))
print("\nP6 PASS" if ok else
      f"\nP6 FAIL — write_guard={guard_ok} ({n_art} artefatti), bypass={bypass_ok}/{knob_ok}/"
      f"{empty_ok}, overwrite_da_solo_bloccato={overwrite_blocked}, dry_run={out_dry}, "
      f"checkpoint={len(ck)}, run1={s1}, skip={s2} ({dt:.1f}s), righe {n1}->{n3} (attese 1->1)")
''')

# ------------------------------------------------------------------ P7
md(r"""## P7 — LORE: il vicinato regge? e la fedeltà misura qualcosa?

Non è un pass/fail della pipeline: è il cantiere A. La diagnosi è chiusa e questa cella la
**riproduce sui nostri dati** invece di fidarsi di una repro sintetica.

Le due cifre che non si muovevano hanno due cause distinte:

- **`fidelity = 1.0` è vera per costruzione.** `lore_sa` la calcola come `dt.score(Z, Yb)` su
  un albero senza `max_depth`, `min_samples_leaf` né pruning: è l'accuratezza di training sul
  proprio training set. Ora accanto compare `fid_cv_bacc`, la stessa famiglia di albero valutata
  su fold tenuti fuori dallo stesso vicinato — costa **zero chiamate al black box**.
- **`rule_len` corto perché il vicinato genetico è saturo di duplicati.** La popolazione
  iniziale è l'istanza da spiegare ripetuta (`record_init` restituisce `x` invariato) e con
  `mutpb=0.2` resta tale. Il numero da guardare è `neigh_unique_frac`, non il tasso di positivi:
  un vicinato può avere entrambe le classi ed essere comunque inutilizzabile.

`num_instances` cambia la conclusione — a 300 il vicinato usciva mono-classe, a 1000 no — quindi
la cella gira **entrambi** e stampa le regole per esteso, consequente compreso. `N_LORE` è
indipendente da `N_PROBE`: `explain` fa `min(max_explain_samples, len(raw_df))`, quindi chiedere
50 campioni su un `raw_df` di 8 righe ne spiega 8.""")
co(r'''
# Il kernel puo' avere in cache una versione precedente di xai_lore_sa: e' l'unico modulo XAI
# senza stato (xai_faithfulness NO -- ricaricarlo azzererebbe xf._CTX), quindi qui si ricarica.
import importlib
import isic24.xai_lore_sa as lsa
lsa = importlib.reload(lsa)
from isic24.xai_lore_sa import LoreEngineSA
assert hasattr(lsa, "neighbourhood_stats"), \
    "xai_lore_sa e' ancora quello vecchio: riavvia il kernel ed esegui da P1."

N_LORE = 2                       # campioni da spiegare (indipendente da N_PROBE)
raw_L, imgs_L, lab_L = xf._build_faith_subset(max(N_LORE, N_PROBE))

def run_lore(tag, **kw):
    eng = LoreEngineSA(feature_space=NB["feature_space"], feat_cols=NB["FEAT_COLS"],
                       to_feature_dict=NB["_to_feature_dict"], training_df=NB["training_df"],
                       neigh_type=kw.pop("neigh_type", "genetic"), **kw)
    print(f"\n=== {tag} ===")
    st, d = eng.explain(model, raw_L, imgs_L, lab_L, max_explain_samples=N_LORE, verbose=True)
    return st, d

# I default di LoreEngineSA sono ORA i nostri (mutpb .5 + init .3), non quelli di lore_sa:
# il confronto utile e' quindi upstream contro tarato, non 300 contro 1000.
st300, d300 = run_lore("lore_sa upstream (mutpb .2, init off)",
                       num_instances=1000, mutpb=None, init_mutpb=None)
st1000, d1000 = run_lore("nostro default (mutpb .5 + init .3)", num_instances=1000)

# `neigh_pos_rate` NON va mediato fra campioni: un vicinato tutto-0 e uno tutto-1 danno
# media 0.5 e sembrano bilanciati mentre sono entrambi mono-classe. Serve `neigh_n_classes`.
cols = ["neigh_rows", "neigh_unique", "neigh_unique_frac", "neigh_top_frac",
        "neigh_n_classes", "neigh_pos_rate", "rule_len", "fid_insample", "fid_cv_bacc"]
# `.get` e non `d[c]`: una colonna mancante deve diventare NaN e restare visibile, non
# interrompere la cella dopo che i due giri sono gia' stati pagati.
avg = lambda d, c: (float(d[c].dropna().mean())
                    if c in d.columns and d[c].notna().any() else float("nan"))
cmp_ = pd.DataFrame([{"config": ni, **{c: avg(d, c) for c in cols}}
                     for ni, d in (("upstream", d300), ("tarato", d1000))])
print("\n" + cmp_.round(4).to_string(index=False))
# La riga che conta davvero: quanti vicinati sono mono-classe, per campione.
for ni, d in (("upstream", d300), ("tarato", d1000)):
    if "neigh_n_classes" in d.columns:
        mono = int((d.neigh_n_classes < 2).sum())
        print(f"  {ni}: {mono}/{len(d)} vicinati MONO-CLASSE "
              f"| pos_rate per campione: {[round(v,3) for v in d.neigh_pos_rate.tolist()]}")

missing = [c for c in cols if c not in d1000.columns]
if missing:
    print(f"\n!! colonne assenti: {missing} -- modulo non aggiornato, riavvia il kernel.")
for d in (d300, d1000):
    if "error" in d.columns and d.error.notna().any():
        print("errori:", d.error.dropna().tolist())

print("\n--- regole, consequente compreso ---")
for tag, d in (("upstream", d300), ("tarato", d1000)):
    for _, r in d.iterrows():
        print(f"[{tag}] campione {r['i']} (label vera {r['true_label']}, "
              f"classe della regola {r['rule_class']}): {r['rule_text']}")

# Il confronto che inquadra: il legacy, su 10 run canonici, dà rule_len 9.58 / fidelity 0.983.
print("\nlegacy (random, 1000):  rule_len 9.58   fidelity 0.983  [lore_fidelity_posthoc.csv]")
print(f"lore_sa tarato:          rule_len {d1000.rule_len.mean():.2f}   "
      f"in-sample {d1000.fid_insample.mean():.3f}   cv_bacc {d1000.fid_cv_bacc.mean():.3f}")

degen = int(d300.warning.notna().sum() + d1000.warning.notna().sum())
print(f"\nvicinati segnalati come degeneri: {degen}/{len(d300)+len(d1000)}")
print("P7 e' informativa. I due numeri da riportare nel cantiere A sono "
      "`neigh_unique_frac` e `fid_cv_bacc`.")
''')

# ------------------------------------------------------------------ P8
md(r"""## P8 — Pulizia e comandi dei run veri

Cancella `_smoke/` e stampa cosa lanciare se tutto è passato.""")
co(r'''
shutil.rmtree(SMOKE, ignore_errors=True)
keras.backend.clear_session(); gc.collect()
print("rimosso", os.path.relpath(SMOKE, ART), "\n")

print("""Se P0-P6 sono PASS, i run veri sono:

  # 1. il run in sospeso: fixed_scaling a alpha=0.5 (~3 min)
  #    la riga a0p000 gia' in tabella e' l'endpoint immagine-only, tutti i modi valgono 0
  python scripts/xai_2026-07-28/run_signed_a05.py

  # 2. (solo se decidi di promuovere la variante signed a primaria) tutti i 110 run, ~5 h
  #    serve per un ranking fra famiglie difendibile: le righe attuali sono run singoli
  PER_FAMILY=999 python scripts/xai_2026-07-28/run_signed.py

Entrambi scrivono in append su
  artifacts/enhanced/runs/_faithfulness/faithfulness_aligned_signed.csv
saltando i run_id gia' presenti, quindi sono interrompibili e riprendibili.
Nessuno dei due tocca faithfulness_aligned_attr.csv ne' i faithfulness.json canonici.""")
''')

nbj = {"cells": CELLS,
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                   "name": "python3"},
                    "language_info": {"name": "python"}},
       "nbformat": 4, "nbformat_minor": 5}

out = "/mnt/disk1/home/cgiovannoni/isic24-dum/notebooks/modality_attendance/isic24_modality_attendance_smoke.ipynb"
json.dump(nbj, open(out, "w"), indent=1, ensure_ascii=False)
print("scritto", out, "|", len(CELLS), "celle")
