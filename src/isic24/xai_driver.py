"""Orchestration for the XAI engines -- the part that used to live in the training
notebook's shared runner (cell 36).

The key fact that made this move possible: the XAI callbacks were **never passed to
``model.fit``**.  ``run_experiment`` always reloaded the best checkpoint afterwards and
drove the Keras hooks by hand.  So this was already a post-hoc driver sitting in the
training notebook; ``run_posthoc_xai`` is the same code with the checkpoint discovery
that the training loop used to supply implicitly.

Training keeps the ability to call it at the end of ``fit``, behind a knob that now
defaults to OFF (``ENABLE_INTRAINING_XAI``).
"""

from __future__ import annotations

import glob
import json
import os

import keras

from . import xai_faithfulness as xf
from .xai_engines import (
    LoreCallback,
    MultiModalGradCAMCallback,
    MultiModalGradCAMEngine,
    ShapCallback,
)

# --------------------------------------------------------------------------------------
# The load-bearing kwargs.
#
# These are the values `make_xai_callbacks` passed, NOT the engines' constructor defaults
# (LoreCallback's own defaults are every_n_epochs=2 / max_explain_samples=12).  Every
# published `lore_fidelity` number was produced with the values below; changing one
# silently makes the artifacts non-comparable.
# --------------------------------------------------------------------------------------
SHAP_KWARGS = dict(
    background_size=64, max_explain_samples=128,
    top_k=10, positive_threshold=0.5, every_n_epochs=1,
    max_samples=128, max_batches=5, rng_seed=0,
)
GRADCAM_KWARGS = dict(use_guided_grads=True, mode="image_only")
GRADCAM_CB_KWARGS = dict(class_index=0)
# I DUE MODI di Grad-CAM, girati nello stesso caricamento (2026-08-01).
#   `image_only`  azzera le feature tabellari -> "dove guarderebbe SENZA il tabellare"
#   `context`     usa le feature vere         -> "dove guarda davvero"
# Fino a oggi girava solo `image_only`: con il tabellare spento, cioe' con la fusione
# disattivata. Per uno studio sull'attendance e' proprio la configurazione che non risponde
# alla domanda ("il peso della modalita' sposta l'attenzione visiva?"), e la differenza fra
# i due modi (`cam_iou_modes`) e' essa stessa una misura di quanto il tabellare la sposta.
# None = un modo solo, quello di GRADCAM_KWARGS (comportamento storico).
GRADCAM_MODES = ("image_only", "context")
# Immagini per passo di gradiente. A 144 tutte insieme la GPU esaurisce la memoria: i
# guided gradients tengono le attivazioni di TUTTA la rete per ogni immagine (misurato il
# 01/08: ~22 GiB su un tensore (144, 8, 8, 1248), OOM su una GPU da 24 GB). Grad-CAM e' per
# campione, quindi spezzare il batch da' lo stesso risultato a memoria costante. Alzarlo se
# la GPU e' libera, abbassarlo se va ancora in OOM.
GRADCAM_BATCH = 16
# max_explain_samples was 6 -- lore_fidelity was the mean over SIX instances (the same 6
# for all 110 runs), which is why it saturated at ~0.97-0.98 and could not discriminate
# between strategies.  Raised to 32.  'random_state' is dropped because the xailib wrapper
# IGNORES it (LoreEngine sets it on the wrapper instead).
LORE_KWARGS = dict(
    max_explain_samples=32, positive_threshold=0.5, top_k=10,
    every_n_epochs=1, max_samples=256, max_batches=2,
    lore_config={"neigh_type": "random", "size": 300}, rng_seed=0,
)

# --------------------------------------------------------------------------------------
# Quale motore LORE costruisce `make_xai_callbacks`.
#
#   "lore_sa" -> `LoreSACallback` + `LoreEngineSA` con LORE_SA_KWARGS.  **Default dal
#                2026-07-29**, per decisione di Claudio: il vicinato `random` del legacy
#                produce spiegazioni poco utili (1.7% di evidenza di classe opposta), quindi
#                le spiegazioni vanno rigenerate con questo.
#   "legacy"   -> `LoreCallback` + `LoreEngine` (xailib/lore_explainer).  Resta selezionabile
#                perche' e' cio' che ha prodotto OGNI `lore_fidelity` in `master_results_*`:
#                serve per riprodurre quei numeri, non per produrne di nuovi.
#
# CONSEGUENZA DA NON PERDERE.  I due motori NON spiegano la stessa funzione -- il legacy usa
# un embedding immagine condiviso (`images.mean(axis=0)`), `LoreEngineSA` uno per campione --
# e non usano la stessa metrica riportabile (vedi sotto).  Quindi la colonna `lore_fidelity`
# dei 110 run resta un artefatto del percorso legacy: va tenuta come storia, non confrontata
# riga per riga con i numeri nuovi.
#
# QUALE NUMERO RIPORTARE.  Non `fidelity`: e' `dt.score(Z, Yb)` in-sample, 1.0 per costruzione
# su un albero non potato.  Vale sia per il legacy sia per lore_sa.  Riportare:
#   `fid_cv_bacc`               fedelta' su fold tenuti fuori (l'unica che discrimina)
#   `lore_sa_coverage`          frazione di campioni con vicinato NON degenere
#   `neigh_unique_frac`, `neigh_pos_rate`   perche' senza questi la fedelta' non si legge
# Deletion/insertion non entrano qui: sono la metrica di SHAP-tabellare e Grad-CAM, non di
# LORE, che e' un surrogato a regole e si misura per accordo col black box.
# --------------------------------------------------------------------------------------
LORE_ENGINE = "lore_sa"

# Configurazione scelta il 2026-07-29 misurando su vanilla seed_47 (n=2 campioni):
# `mutpb=.5` + `init_mutpb=.3` da' 2.9x la diversita' di upstream, le regole piu' lunghe
# (3.0) e 4x l'evidenza di classe opposta di `mutpb=.8`, a +19% di costo.  Dettagli e
# tabella completa: BENCH_lore_2026-07-28.md, "Scelta del generatore".
LORE_SA_KWARGS = dict(
    max_explain_samples=32, every_n_epochs=1, max_samples=256, max_batches=2, rng_seed=0,
    neigh_type="genetic", num_instances=1000, mutpb=0.5, init_mutpb=0.3, threshold=0.5,
)

# Dove gira LORE. Il suo ciclo caldo e' la TESTA densa (~1000 valutazioni per istanza) piu'
# l'algoritmo genetico e l'albero, che sono CPU puro; la CNN gira una volta sola per run.
# Quindi tiene la GPU occupata 23 min/run senza usarla quasi: su una macchina condivisa,
# 42 ore di GPU ferma.
#
# `"/CPU:0"` sposta LORE sulla CPU **lasciando la GPU visibile e libera** per gli altri
# motori: `run_posthoc_xai` passa il device a `load_run_model`, quindi le variabili NASCONO
# sulla CPU. E' l'unico punto in cui la scelta ha effetto.
#
# ⚠️ Il `tf.device` attorno al solo calcolo (quello in `xai_lore_sa.on_epoch_end`) NON basta
# e da solo fallisce — provato il 02/08, ~20 s per run:
#     InvalidArgumentError: Trying to access resource stem_conv/kernel/... located in
#     device /...GPU:0 from device /...CPU:0
# perche' `tf.device` non sposta le variabili gia' allocate e la `predict` di Keras 3 e'
# compilata con XLA, che rifiuta la variabile remota invece di copiarla.
#
# Costo MISURATO il 02/08 su 6 run: **20,0 min/run su CPU** contro i 22 min su GPU. La CPU
# non costa niente, e' anzi leggermente piu' veloce — coerente con il fatto che la GPU serviva
# solo alla CNN. Quindi `"/CPU:0"` e' la scelta giusta per il giro lungo, non un ripiego.
LORE_SA_DEVICE = None

ALL_ENGINES = ("shap", "gradcam", "lore")


# `xai_status.json` records outcomes keyed by CALLBACK CLASS NAME, not by engine name.
# This maps back, so the "already complete" shortcut can ask about the engines actually
# requested instead of about whatever happens to be in the file.
#
# The bug this closes (found 2026-07-30 in dry-run): the 20 `learnable_scaling` runs whose
# SHAP arrays were filled in post-hoc on 29/07 carry `{"ShapCallback": "ok"}`.  The old test
# was `all(v == "ok" for v in prev["engines"].values())` -- vacuously true for any engine
# NOT in the file -- so `engines=("lore",)` skipped all 20 and reported them complete.  A
# full LORE pass over the 110 would have silently produced 90.
def _status_key_for(engine):
    if engine == "shap":
        return "ShapCallback"
    if engine == "gradcam":
        return "MultiModalGradCAMCallback"
    if engine == "lore":
        # The two LORE engines are different callbacks writing to different subdirs: a run
        # carrying the legacy result must NOT count as done for `lore_sa`.
        return "LoreSACallback" if LORE_ENGINE == "lore_sa" else "LoreCallback"
    return engine


# --------------------------------------------------------------------------------------
# Canonical-artifact write guard  (added 2026-07-29, Claudio: "non voglio assolutamente
# sovrascrivere le run canoniche che hanno gia' prodotto i loro artefatti")
#
# The trap this closes: `run_posthoc_xai` skips only when `xai/xai_status.json` says every
# engine is "ok" -- and that file is written by THIS driver, so none of the 110 canonical
# runs has one.  They were produced by the old in-training path.  Pointed at a canonical
# run dir, the driver therefore sails past the skip check even with `overwrite=False` and
# rewrites `xai/shap/epoch_*`, `xai/gradcam/epoch_*.png`, `xai/lore/epoch*` in place.
#
# So the guard keys on the ARTIFACTS, not on the status file, and it is PER ENGINE: it
# refuses only for the engines whose own outputs are already there.  Filling in what a run
# is missing stays allowed -- that is additive, and it is what `run_posthoc_xai_batch` is
# for (e.g. ~20 runs have `faithfulness.json` but no SHAP arrays).
#
# Scope: this covers the paths THIS driver writes.  `xai/faithfulness/` is written by
# `xai_faithfulness.explanation_faithfulness`, not from here, and is governed by the
# post-hoc notebook's own `FAITH_OVERWRITE` knob (its loop reuses an existing
# `faithfulness.json` unless you flip it).
# --------------------------------------------------------------------------------------
CANONICAL_WRITE_GUARD = True

# Everything each engine's callback writes under `run_dir/xai/`, per engine.
#
# I due motori LORE scrivono nella STESSA `xai/lore/` ma in sottocartelle e CSV distinti
# (`epoch*/context` + `lore_fidelity_per_epoch.csv` contro `epoch*/tabular_sa` +
# `lore_sa_per_epoch.csv`).  Tenerli separati qui e' quello che permette di aggiungere il
# motore nuovo a una run che ha gia' quello legacy SENZA `allow_clobber` -- cioe' senza
# aprire anche la porta a sovrascrivere il legacy nella stessa chiamata.
def shap_subdir():
    """`shap` o `shap_tab`, secondo che un campione XAI sia adottato o no.

    UN SOLO posto decide dove vive SHAP, perche' tre pezzi devono essere d'accordo: chi
    scrive (`make_xai_callbacks`), chi protegge (`_CLOBBERABLE`) e chi rilegge
    (`xai_faithfulness._load_shap_values`). Se divergono, il guard difende una cartella
    vuota mentre il lettore ne trova un'altra — e gli npz canonici di giugno, che sono sul
    PREFISSO e senza stampigliatura, tornano a essere gli unici che si trovano.

    `xai/shap/`      artefatti canonici, prefisso, mai riscritti
    `xai/shap_tab/`  tutto cio' che e' calcolato sul campione adottato: array SHAP +
                     fedelta' con segno. Una cartella, un campione.
    """
    return "shap_tab" if SHAP_KWARGS.get("subset_fingerprint") else "shap"


def gradcam_subdir():
    """`gradcam` o `gradcam_img`, secondo che un campione XAI sia adottato o no.

    Stessa logica di `shap_subdir`. `xai/gradcam/` tiene i 106 PNG storici: OTTO immagini
    prese dalla testa di `validation_ds`, senza array e senza traccia di quali fossero.
    `xai/gradcam_img/` e' il lato immagine rifatto sul campione a lesioni distinte, con le
    heatmap GREZZE salvate — che sono la precondizione di qualunque metrica sull'immagine.
    """
    return "gradcam_img" if SHAP_KWARGS.get("subset_fingerprint") else "gradcam"


_CLOBBERABLE = {
    # dipende dal campione adottato, come `lore` dipende dal motore: vedi `_patterns_for`
    "shap": {"shap":     ("xai/shap/epoch_*", "xai/shap/*.csv"),
             "shap_tab": ("xai/shap_tab/epoch_*", "xai/shap_tab/shap_summary.csv")},
    "gradcam": {"gradcam":     ("xai/gradcam/epoch_*",),
                "gradcam_img": ("xai/gradcam_img/gradcam_arrays*_epoch_*.npz",
                                "xai/gradcam_img/gradcam_summary.csv")},
    "lore": {
        "legacy":  ("xai/lore/epoch*/context", "xai/lore/epoch*/tabular_only",
                    "xai/lore/lore_fidelity_per_epoch.csv"),
        # 2026-08-01: LORE-SA ha una cartella sua, `xai/lore_sa/`. Prima conviveva col
        # legacy dentro `xai/lore/` e i due si distinguevano per sottocartella
        # (`tabular_sa` contro `context`+`tabular_only`); ora li separa il livello sopra,
        # e il guard non deve piu' guardare dentro `xai/lore/` per questo motore.
        # I `xai/lore/epoch*/tabular_sa/` delle run di giugno-luglio restano dove sono:
        # sono artefatti storici e nessuna scrittura nuova puo' finirci sopra.
        "lore_sa": ("xai/lore_sa/epoch*", "xai/lore_sa/lore_sa_per_epoch.csv"),
    },
    # Non e' un engine di `make_xai_callbacks`: lo scrive
    # `xai_faithfulness.aligned_signed_faithfulness`. Sta qui perche' `existing_xai_artifacts`
    # e' l'unico posto che sa cosa una run ha gia' prodotto, e il loop della fedelta' lo
    # interroga per la stessa ragione degli altri.
    "shap_tab": ("xai/shap_tab/faithfulness_aligned_signed_*.json",),
}


def _patterns_for(engine):
    pats = _CLOBBERABLE.get(engine, ())
    if isinstance(pats, dict):
        if engine == "shap":            # dipendono dal campione adottato
            return pats.get(shap_subdir(), ())
        if engine == "gradcam":
            return pats.get(gradcam_subdir(), ())
        return pats.get(LORE_ENGINE, ())   # 'lore': dipende da quale motore e' selezionato
    return pats


def _is_artifact(path):
    """Una directory VUOTA non e' un artefatto.

    I callback creano `epoch<NNN>/<engine>/` prima di calcolare, quindi un run
    interrotto (Ctrl-C, OOM, kernel morto) lascia una cartella vuota. Contandola come
    artefatto il write guard bloccava per sempre la ri-esecuzione di una run che non
    aveva prodotto nulla, e l'unica uscita sarebbe stata `allow_clobber=True` -- cioe'
    disarmare la protezione proprio dove serve. Trovato il 2026-07-30 su
    `fixed_scaling/a1p000`, bloccata da una `tabular_sa/` vuota di 30 secondi prima.
    """
    return bool(os.listdir(path)) if os.path.isdir(path) else True


def existing_xai_artifacts(run_dir, engines=ALL_ENGINES):
    """Outputs of `engines` already on disk -- exactly what a re-run would replace."""
    hits = []
    for eng in engines:
        for pat in _patterns_for(eng):
            hits += [p for p in glob.glob(os.path.join(run_dir, pat)) if _is_artifact(p)]
    return sorted(hits)


def assert_writable(run_dir, engines=ALL_ENGINES, allow_clobber=False):
    """Refuse to write an engine's artifacts over a run that already has them."""
    if allow_clobber or not CANONICAL_WRITE_GUARD:
        return []
    hits = existing_xai_artifacts(run_dir, engines)
    if hits:
        busy = sorted(e for e in engines if existing_xai_artifacts(run_dir, (e,)))
        free = sorted(set(engines) - set(busy))
        shown = "\n    ".join(os.path.relpath(h, run_dir) for h in hits[:6])
        more = f"\n    ... e altri {len(hits) - 6}" if len(hits) > 6 else ""
        raise RuntimeError(
            f"[XAI WRITE GUARD] {run_dir}\n  ha gia' {len(hits)} artefatti su disco per "
            f"{busy}:\n    {shown}{more}\n"
            f"  Rieseguire qui li SOVRASCRIVE. Nota che overwrite=False non protegge: salta "
            f"solo se esiste xai/xai_status.json, che nessuna run canonica ha (le 110 vengono "
            f"dal vecchio percorso in-training).\n"
            + (f"  Gli engine {free} qui non hanno artefatti: engines={tuple(free)} li "
               f"aggiunge senza toccare niente.\n" if free else "")
            + f"  **overwrite=True NON basta a passare di qui**: dice 'ricalcola', non 'puoi "
              f"distruggere'. L'unico consenso e' allow_clobber=True (o "
              f"xai_driver.CANONICAL_WRITE_GUARD = False).\n"
              f"  In alternativa copia config.json + checkpoints in una dir temporanea e punta "
              f"il driver li'.")
    return hits


# --------------------------------------------------------------------------------------
# Probe batch
# --------------------------------------------------------------------------------------

def make_probe_batch(n=8, subset_idx=None):
    """Le immagini di validation su cui gira Grad-CAM.

    ``subset_idx`` -- posizioni in ``validation_df``: e' cosi' che Grad-CAM spiega le STESSE
    lesioni di SHAP e LORE.  Senza, si torna al comportamento storico: le prime ``n`` righe
    di ``validation_ds`` (costruito con ``shuffle=False``), cioe' **otto** immagini del
    prefisso.  Otto campioni non bastano per nessuna delle metriche sul lato immagine, ed e'
    il motivo per cui il lato immagine e' stato rifatto il 2026-08-01.

    Used to be a module-level global mutated on first use and cached across every run in
    the study; it is now an explicit argument so a caller can score a different model on
    a different probe.
    """
    ctx = xf._CTX
    if subset_idx is None:
        xb, yb = next(iter(ctx.validation_ds))
        return {k: v[:n] for k, v in xb.items()}, yb[:n]

    # Si legge il dataset fino alla posizione piu' alta richiesta e poi si indicizza: e' la
    # stessa strategia di `LoreSACallback._freeze_batch`, e non assume che il campione sia
    # un prefisso -- che e' esattamente l'assunzione che ci ha gia' ingannati due volte.
    import numpy as _np
    import tensorflow as _tf
    idx = _np.asarray(subset_idx, int)
    need = int(idx.max()) + 1
    xs_, ys_, seen = [], [], 0
    for xb, yb in ctx.validation_ds:
        xs_.append(xb); ys_.append(yb)
        seen += int(_tf.shape(yb)[0])
        if seen >= need:
            break
    x_all = {k: _tf.concat([b[k] for b in xs_], axis=0) for k in xs_[0]}
    y_all = _tf.concat(ys_, axis=0)
    return ({k: _tf.gather(v, idx) for k, v in x_all.items()}, _tf.gather(y_all, idx))


# --------------------------------------------------------------------------------------
# Callback construction and driving
# --------------------------------------------------------------------------------------

def make_xai_callbacks(run_dir, model_name, probe_batch=None, engines=ALL_ENGINES,
                       xai_dir=None):
    """Build the SHAP / Grad-CAM / LORE callbacks for one run directory.

    ``xai_dir`` reindirizza TUTTA la scrittura fuori da ``run_dir/xai``: e' cosi' che il
    campione stratificato produce artefatti senza sfiorare le run canoniche (e senza
    bisogno di ``allow_clobber``, che disarmerebbe il write guard proprio dove serve).
    """
    ctx = xf._CTX
    xai_dir = xai_dir or os.path.join(run_dir, "xai")

    out = []
    if "shap" in engines:
        d = os.path.join(xai_dir, shap_subdir())
        os.makedirs(d, exist_ok=True)
        out.append(ShapCallback(
            data_source=ctx.validation_ds, validation_df=ctx.validation_df,
            feat_cols=ctx.feat_cols, encoded_feature_names=ctx.encoded_feature_names,
            base_dir=d, **SHAP_KWARGS))
    if "gradcam" in engines:
        d = os.path.join(xai_dir, gradcam_subdir())
        os.makedirs(d, exist_ok=True)
        # Il probe batch segue il campione adottato: con `subset_idx` Grad-CAM spiega le
        # STESSE lesioni di SHAP e LORE, e le heatmap diventano riallineabili con le altre
        # due spiegazioni. Senza, resta il comportamento storico (le prime 8 righe).
        _sub = SHAP_KWARGS.get("subset_idx")
        if probe_batch is None:
            probe_batch = make_probe_batch(subset_idx=_sub)
        out.append(MultiModalGradCAMCallback(
            engine=MultiModalGradCAMEngine(**GRADCAM_KWARGS),
            probe_batch=probe_batch, save_dir=d,
            subset_idx=_sub, subset_fingerprint=SHAP_KWARGS.get("subset_fingerprint"),
            modes=(GRADCAM_MODES if _sub is not None else None),
            batch_size=GRADCAM_BATCH,
            tag=f"{model_name}/GradCAM", **GRADCAM_CB_KWARGS))
    if "lore" in engines:
        # Cartelle DISTINTE per i due motori (2026-08-01): `xai/lore/` resta del legacy,
        # LORE-SA ha `xai/lore_sa/`. Prima condividevano `xai/lore/` e si separavano solo
        # una sottocartella piu' in basso, il che rendeva impossibile leggere dal nome della
        # directory quale motore avesse prodotto cosa.
        d = os.path.join(xai_dir, "lore" if LORE_ENGINE == "legacy" else "lore_sa")
        os.makedirs(d, exist_ok=True)
        if LORE_ENGINE == "legacy":
            out.append(LoreCallback(
                data_source=ctx.validation_ds, raw_feature_df=ctx.validation_df,
                feat_cols=ctx.feat_cols, feature_space=ctx.feature_space,
                base_dir=d, **LORE_KWARGS))
        elif LORE_ENGINE == "lore_sa":
            from .xai_lore_sa import _lore_sa_callback_cls
            out.append(_lore_sa_callback_cls()(
                data_source=ctx.validation_ds, raw_feature_df=ctx.validation_df,
                feat_cols=ctx.feat_cols, feature_space=ctx.feature_space,
                to_feature_dict=ctx.to_feature_dict, training_df=ctx.training_df,
                base_dir=d, **LORE_SA_KWARGS))
        else:
            raise ValueError(f"LORE_ENGINE={LORE_ENGINE!r}: attesi 'legacy' o 'lore_sa'")
    return out


def run_xai_on_best_epoch(best_model, xai_callbacks, best_epoch, best_val_auc, run_dir=None):
    """Drive the Keras hooks by hand against an already-trained model.

    Returns ``{EngineName: "ok" | "failed: ..."}``.  The previous version returned a
    single ``any_done`` boolean that was ``True`` when just one of the three engines had
    succeeded, printing the other two failures as warnings -- a batch loop over 110 runs
    then looked busy while producing partial artifacts.  That footgun cost a silent
    partial once already (``seed_47__a0p125``), so the per-engine outcome is returned and,
    when ``run_dir`` is given, persisted to ``xai/xai_status.json``.
    """
    logs = {"val_auc": float(best_val_auc)}
    status = {}
    for cb in xai_callbacks:
        name = cb.__class__.__name__
        try:
            if hasattr(cb, "every_n_epochs"):
                cb.every_n_epochs = 1
            cb.set_model(best_model)
            cb.set_params({"epochs": int(best_epoch + 1), "steps": None, "verbose": 0})
            if hasattr(cb, "on_train_begin"):
                cb.on_train_begin(logs={})
            cb.on_epoch_end(int(best_epoch), logs=logs)
            if hasattr(cb, "on_train_end"):
                cb.on_train_end(logs={})
            status[name] = "ok"
        except Exception as e:
            status[name] = f"failed: {type(e).__name__}: {e}"
            print(f"[XAI WARN] {name} failed at epoch {best_epoch}: {type(e).__name__}: {e}")

    if run_dir:
        os.makedirs(os.path.join(run_dir, "xai"), exist_ok=True)
        p = os.path.join(run_dir, "xai", "xai_status.json")
        # MERGE, don't replace: running one engine on a run that already recorded another
        # used to wipe the older entry, and the skip shortcut reads this file -- so the
        # engine that got erased would silently be re-run later, or (worse) be reported as
        # never done. Adding an engine is additive on disk; the ledger must match.
        merged = {}
        if os.path.exists(p):
            try:
                merged = json.load(open(p)).get("engines", {})
            except (ValueError, OSError):
                merged = {}
        merged.update(status)
        with open(p, "w") as f:
            json.dump({"best_epoch": int(best_epoch), "engines": merged}, f, indent=2)
    return status


# --------------------------------------------------------------------------------------
# Post-hoc entry points
# --------------------------------------------------------------------------------------

def load_run_model(run_dir, device=None):
    """Reload a run's best checkpoint together with its recorded best epoch.

    ``safe_mode=False`` plus the five custom layers being registered (import
    ``isic24.mod_attendance`` anywhere in the session) is what lets the 70/110
    fixed_scaling / learnable_scaling / controlled_fusion / adaptive_gating checkpoints
    deserialize at all.

    ``device`` (es. ``"/CPU:0"``) e' il device su cui le VARIABILI vengono create. E' la
    differenza fra far girare un motore sulla CPU e non farlo: un `tf.device` attorno al
    solo calcolo non sposta pesi gia' allocati sulla GPU, e la `predict` di Keras 3 (XLA)
    muore invece di copiarli. Attorno al CARICAMENTO, invece, i pesi nascono dove si vuole
    e da li' in poi tutto il grafo e' coerente. Vedi `LORE_SA_DEVICE`.
    """
    ck = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "best_*.keras")))
    if not ck:
        raise FileNotFoundError(f"no checkpoint under {run_dir}/checkpoints/")

    cfg_path = os.path.join(run_dir, "config.json")
    cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}

    import contextlib
    import tensorflow as tf
    with (tf.device(device) if device else contextlib.nullcontext()):
        model = keras.models.load_model(ck[0], compile=False, safe_mode=False)
    return model, cfg


def run_posthoc_xai(run_dir, engines=ALL_ENGINES, best_epoch=None, best_val_auc=None,
                    probe_batch=None, overwrite=False, dry_run=False, verbose=True,
                    allow_clobber=False, xai_dir=None):
    """Run the XAI engines against a saved checkpoint, long after it was trained.

    This is the same pipeline the training notebook ran at the end of ``fit`` -- same
    engines, same kwargs, same hook sequence -- with the checkpoint loaded from disk
    instead of handed over in memory.

    ``overwrite`` and ``allow_clobber`` are deliberately NOT the same knob:

    * ``overwrite=True`` means "don't skip, recompute" -- it disables the
      ``xai_status.json`` shortcut.
    * ``allow_clobber=True`` means "you may destroy artifacts that are already there" --
      it disables :func:`assert_writable`.

    So ``overwrite=True`` on its own still raises when artifacts exist.  That is the point:
    recomputing into an empty run dir is routine, while rewriting a run that already
    produced its artifacts is not, and the two should not share a spelling.
    """
    model_cfg = {}
    if best_epoch is None or best_val_auc is None:
        cfg_path = os.path.join(run_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"{cfg_path} missing: pass best_epoch/best_val_auc explicitly")
        model_cfg = json.load(open(cfg_path))
        best_epoch = model_cfg["best_epoch"] if best_epoch is None else best_epoch
        best_val_auc = model_cfg.get("best_val_auc", 0.0) if best_val_auc is None else best_val_auc

    # Con `xai_dir` esterna non si scrive dentro run_dir: ne' lo status file ne' il write
    # guard hanno giurisdizione, e la ripresa la fa il chiamante (isic24.xai_subset.is_done).
    external = xai_dir is not None
    status_path = os.path.join(xai_dir or os.path.join(run_dir, "xai"), "xai_status.json")
    if not overwrite and os.path.exists(status_path):
        prev = json.load(open(status_path)).get("engines", {})
        # Skip only if EVERY requested engine is recorded ok -- see `_status_key_for`.
        if all(prev.get(_status_key_for(e)) == "ok" for e in engines):
            if verbose:
                print(f"[skip] {run_dir}: {list(engines)} gia' fatti (overwrite=False)")
            return prev

    if dry_run:
        if verbose:
            n = len(existing_xai_artifacts(run_dir, engines))
            blocked = "" if (allow_clobber or not n) else \
                f" -- MA il write guard lo bloccherebbe: {n} artefatti XAI gia' presenti"
            print(f"[dry-run] would run {engines} on {run_dir} @ epoch {best_epoch}{blocked}")
        return {e: "dry-run" for e in engines}

    if not external:
        assert_writable(run_dir, engines=engines, allow_clobber=allow_clobber)

    # DOVE nascono i pesi. Con `LORE_SA_DEVICE="/CPU:0"` il checkpoint viene caricato con le
    # variabili sulla CPU, e da li' in poi LORE gira tutto sulla CPU pur avendo la GPU
    # visibile e libera per gli altri motori. Il `tf.device` attorno al solo `explain`
    # (`xai_lore_sa`) non basta e non e' un'alternativa: e' l'errore del 02/08.
    # Vale per LORE DA SOLO: caricare su CPU un modello per SHAP o Grad-CAM li renderebbe
    # lentissimi in silenzio, quindi la combinazione e' un errore invece che una sorpresa.
    _dev = LORE_SA_DEVICE if "lore" in engines else None
    if _dev and tuple(engines) != ("lore",):
        raise RuntimeError(
            f"LORE_SA_DEVICE={LORE_SA_DEVICE!r} vale per il modello INTERO: con "
            f"engines={tuple(engines)} ci finirebbero sopra anche gli altri motori.\n"
            "  Lanciare LORE da solo (engines=('lore',)), oppure LORE_SA_DEVICE=None.")

    model, cfg = load_run_model(run_dir, device=_dev)
    model_name = (model_cfg or cfg).get("model_name", os.path.basename(run_dir))
    cbs = make_xai_callbacks(run_dir, model_name, probe_batch=probe_batch, engines=engines,
                             xai_dir=xai_dir)
    return run_xai_on_best_epoch(model, cbs, best_epoch, best_val_auc,
                                 run_dir=(xai_dir and os.path.dirname(xai_dir)) or run_dir)


def run_posthoc_xai_batch(art_path, engines=ALL_ENGINES, only_impl=None, max_runs=None,
                          overwrite=False, dry_run=False, verbose=True, allow_clobber=False):
    """Apply :func:`run_posthoc_xai` across the canonical run registry.

    Deliberately opt-in: nothing here runs unless called, and ``overwrite=False`` means a
    re-invocation resumes rather than redoing work.  Runs that already carry XAI artifacts
    are refused by :func:`assert_writable` and reported as ``failed: RuntimeError`` rather
    than silently rewritten; the loop moves on to the next run.
    """
    import gc

    import pandas as pd

    master = pd.read_csv(os.path.join(art_path, "summaries", "master_results_rows.csv"))
    if only_impl:
        master = master[master.impl_name.isin(list(only_impl))]
    if max_runs:
        master = master.head(int(max_runs))

    results = {}
    for i, r in master.reset_index(drop=True).iterrows():
        tail = str(r.run_dir).split("artifacts/enhanced/")[-1]
        run_dir = os.path.join(art_path, tail)
        try:
            results[r.run_id] = run_posthoc_xai(
                run_dir, engines=engines, best_epoch=int(r.best_epoch),
                best_val_auc=float(r.best_val_auc), overwrite=overwrite,
                dry_run=dry_run, verbose=verbose, allow_clobber=allow_clobber)
        except Exception as e:
            results[r.run_id] = {"_run": f"failed: {type(e).__name__}: {e}"}
            print(f"[FAIL] {r.run_id}: {type(e).__name__}: {e}")
        finally:
            if not dry_run:
                keras.backend.clear_session()
                gc.collect()
        if verbose:
            print(f"[{i + 1}/{len(master)}] {r.run_id}: {results[r.run_id]}", flush=True)
    return results
