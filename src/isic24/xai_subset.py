"""Campione stratificato per la XAI.

⚠️ LEGGERE PRIMA: LA MOTIVAZIONE ORIGINALE DI QUESTO MODULO ERA SBAGLIATA
--------------------------------------------------------------------------
Questo modulo era nato (31/07) per correggere una presunta distorsione del prefisso: «i primi
32 campioni hanno il 40.6% di maligni contro il 9.1% del validation». **Quel 9.1% e' sbagliato**:
veniva da `neg_sample = 0.01` senza vedere la riga dopo, `pos_sample = 5.0` — i positivi sono
upsamplati 5x. La prevalenza vera e' **0.334**, il prefisso a 0.406 su n=32 da' **P = 0.245**,
cioe' rumore. E `df.sample(frac=1.0)` rimescola prima dello split in fold, quindi «le prime N
righe» sono gia' un campione casuale.

**Non c'e' nessuna distorsione del prefisso da correggere.** Il modulo resta perche' fa comunque
due cose utili — quote di classe esatte e un'identita' verificabile del campione — ma non
spacciarlo per una correzione di bias.

IL PROBLEMA VERO, QUELLO SI'
----------------------------
`pos_sample=5.0` usa `replace=True`: 364 positivi unici diventano 1820 righe. Sul campione
stratificato n=256 ci sono **222 lesioni uniche** su 256 righe, e fra gli 85 positivi solo **51
lesioni uniche** (molteplicita' fino a 3). La XAI spiega la stessa lesione piu' volte: i campioni
non sono indipendenti e **ogni intervallo di confidenza per run e' ottimistico**.

Il rimedio e' deduplicare su `isic_id`, non stratificare: `unique_lesion_subset` (+ `adopt`),
usato in produzione dal 31/07 con n=144 lesioni distinte, balance='equal'.

⚠️ E IL PREFISSO DI *QUESTO* CAMPIONE E' SBILANCIATO SUL SERIO (misurato 2026-08-01)
------------------------------------------------------------------------------------
Non contraddice quanto sopra: il prefisso di `validation_df` sta davvero a 0.336. Ma
`unique_lesion_subset` tiene la PRIMA occorrenza di ogni lesione, che per un positivo e' il
minimo di ~5 copie (`pos_sample=5.0`) e per un negativo un'estrazione sola. Ordinando gli
indici i positivi finiscono davanti: mediana dell'indice 160 sui positivi contro 581 sui
negativi. Sul campione n=144 (50/50 per costruzione) i primi 48 sono **35 pos / 13 neg**.

Quindi: per spiegarne un sottoinsieme si passa un array di indici ESPLICITO e bilanciato,
non si lascia troncare `max_explain_samples`. Confronti fra strategie restano validi (tutte
le run vedono gli stessi campioni); livelli assoluti e contrasti fra classi no.

DOVE FINISCONO GLI ARTEFATTI (dal 2026-08-01)
---------------------------------------------
Dentro la run, in sottocartelle proprie: `<run_dir>/xai/shap_tab/` e
`<run_dir>/xai/lore_sa/` (vedi `out_dir_for`). Solo cio' che e' cross-run — tabelle e
figure di confronto fra famiglie, `report()` — sta in `runs/_faithfulness/`.

COSA RESTA VALIDO
-----------------
L'impronta (`subset_fingerprint`) e il guard che ne discende in `xai_faithfulness`: gli npz SHAP
salvano `explained_indices` come POSIZIONI, quindi riusarli sotto un campione diverso accoppia i
valori SHAP di certe righe alle feature di altre lesioni, senza errore e con numeri plausibili.
Quello e' un rischio reale a prescindere da quale campione si scelga.
"""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np

# Gli artefatti PER RUN vivono dentro la run (vedi `out_dir_for` piu' sotto). Qui resta
# solo la destinazione di cio' che e' CROSS-RUN e quindi non appartiene a nessuna: tabelle
# e figure di confronto fra famiglie.
AGGREGATE_DIRNAME = "_faithfulness"


def stratified_subset(labels, n, seed=0):
    """Indici POSIZIONALI in `validation_df` di un campione stratificato per classe.

    Proporzionale: la prevalenza del campione riproduce quella del pool (a meno
    dell'arrotondamento).  Deterministico dato `seed`.  Restituisce indici ORDINATI, perche'
    `_build_faith_subset` legge il dataset in ordine e poi indicizza.

    NB non corregge nessuna distorsione del prefisso -- quella non esiste, vedi il docstring
    del modulo.  Garantisce le quote di classe ESATTE invece che in distribuzione, il che
    toglie una fonte di varianza fra campioni ma non sposta il valore atteso.
    """
    y = np.asarray(labels).reshape(-1)
    n = int(n)
    if n > len(y):
        raise ValueError(f"n={n} > {len(y)} campioni disponibili nel validation set")
    rng = np.random.default_rng(seed)
    out = []
    classes = np.unique(y)
    # Quota proporzionale, poi si distribuiscono i resti dalla classe piu' grande in giu':
    # deterministico invece che dipendente dall'ordine di iterazione delle classi.
    quotas = {}
    for c in classes:
        idx_c = np.flatnonzero(y == c)
        quotas[c] = int(np.floor(len(idx_c) * n / len(y)))
    leftover = n - sum(quotas.values())
    for c in sorted(classes, key=lambda c: -int(np.sum(y == c))):
        if leftover <= 0:
            break
        quotas[c] += 1
        leftover -= 1
    for c in classes:
        idx_c = np.flatnonzero(y == c)
        k = min(quotas[c], len(idx_c))
        out.append(rng.choice(idx_c, size=k, replace=False))
    idx = np.sort(np.concatenate(out))
    assert len(idx) == n, f"quote non tornano: {len(idx)} != {n}"
    return idx


def unique_lesion_subset(isic_ids, labels, n=None, seed=0, balance="pool"):
    """Indici POSIZIONALI su **lesioni distinte**: una riga per `isic_id`, mai due.

    E' il rimedio al problema vero: `pos_sample=5.0` con `replace=True` fa comparire la stessa
    lesione ~5 volte, e le copie sono repliche ESATTE (stessa immagine, stesse feature, stessa
    predizione). Spiegarle piu' volte non aggiunge informazione, restringe gli intervalli e
    basta.

    ``balance``:
      * ``"pool"``  -- prevalenza proporzionale a quella delle LESIONI uniche;
      * ``"equal"`` -- meta' per classe. Massimizza la potenza del contrasto fra classi a
        parita' di costo; il livello assoluto non e' piu' quello della popolazione, ma resta
        confrontabile fra strategie perche' il campione e' lo stesso per tutte;
      * ``"all"``   -- ogni lesione distinta, nessun campionamento (``n`` ignorato).

    Solleva se ``n`` non e' ottenibile, dicendo qual e' il massimo: il modo peggiore di
    fallire qui sarebbe restituire in silenzio meno positivi di quelli chiesti.
    """
    ids = np.asarray(isic_ids)
    y = np.asarray(labels).reshape(-1)
    if len(ids) != len(y):
        raise ValueError(f"isic_ids ({len(ids)}) e labels ({len(y)}) di lunghezza diversa")

    # Un rappresentante per lesione: la PRIMA occorrenza. Deterministico e indipendente dal
    # seed, cosi' due campioni con seed diversi pescano fra le stesse lesioni.
    _, first = np.unique(ids, return_index=True)
    rep = np.sort(first)
    y_rep = y[rep]
    classes = np.unique(y_rep)
    avail = {int(c): int(np.sum(y_rep == c)) for c in classes}

    if balance == "all":
        return rep, {"balance": "all", "n": len(rep), "per_class": avail,
                     "n_unique_available": {int(c): v for c, v in avail.items()}}
    if n is None:
        raise ValueError("n e' obbligatorio se balance != 'all'")
    n = int(n)

    if balance == "equal":
        k = n // len(classes)
        want = {int(c): k for c in classes}
        for c in list(want)[:n - k * len(classes)]:
            want[c] += 1
    elif balance == "pool":
        want = {int(c): int(np.floor(avail[int(c)] * n / len(rep))) for c in classes}
        left = n - sum(want.values())
        for c in sorted(want, key=lambda c: -avail[c]):
            if left <= 0:
                break
            want[c] += 1
            left -= 1
    else:
        raise ValueError(f"balance={balance!r}: attesi 'pool', 'equal' o 'all'")

    short = {c: (want[c], avail[c]) for c in want if want[c] > avail[c]}
    if short:
        # Il massimo ottenibile dipende dal balance:
        #   'pool'  -> proporzionale sulle lesioni uniche: si arriva fino a TUTTE (n<=len(rep));
        #   'equal' -> lo limita la classe piu' piccola, 2x quella (qui i positivi unici).
        n_max = len(rep) if balance == "pool" else min(avail.values()) * len(classes)
        raise ValueError(
            f"n={n} non ottenibile su lesioni distinte con balance={balance!r}.\n"
            f"  lesioni uniche per classe: {avail}  (totale {len(rep)})\n"
            f"  servirebbero {dict(want)}\n"
            f"  n massimo con questo balance: {n_max}\n"
            f"  alternative: n<={n_max}, balance='all' (tutte le {len(rep)} lesioni), "
            f"oppure balance='pool' se stai usando 'equal'.")

    rng = np.random.default_rng(seed)
    out = []
    for c in classes:
        pool_c = rep[y_rep == c]
        out.append(rng.choice(pool_c, size=want[int(c)], replace=False))
    idx = np.sort(np.concatenate(out))
    assert len(np.unique(ids[idx])) == len(idx), "il campione contiene lesioni ripetute"
    return idx, {"balance": balance, "n": len(idx),
                 "per_class": {int(c): int(np.sum(y[idx] == c)) for c in classes},
                 "n_unique_available": avail}


def subset_fingerprint(isic_ids, idx, seed, n, kind="stratified"):
    """Identita' del campione. E' questa che impedisce di mescolare ordinamenti diversi.

    Include gli ISIC id NELL'ORDINE del campione, non solo il conteggio: due campioni della
    stessa dimensione sullo stesso validation set devono avere impronte diverse.
    """
    ids = [str(isic_ids[i]) for i in np.asarray(idx, int)]
    h = hashlib.sha256(("|".join(ids) + f"|{kind}|{seed}|{n}").encode()).hexdigest()[:16]
    return f"{kind}-s{seed}-n{n}-{h}"


# L'impronta del comportamento STORICO: le prime N righe. Ogni artefatto canonico e' stato
# prodotto cosi' e non porta stampigliatura, quindi "nessuna stampigliatura" == questo.
PREFIX_FINGERPRINT = "prefix"


def save_snapshot(path, isic_ids, idx, seed, n, kind="stratified", extra=None):
    """Scrive il campione su disco. Gli ID, non solo gli indici: gli indici da soli non
    sopravvivono a un ricalcolo del fold."""
    fp = subset_fingerprint(isic_ids, idx, seed, n, kind=kind)
    snap = {
        "fingerprint": fp, "kind": kind, "seed": int(seed), "n": int(n),
        "positional_idx": [int(i) for i in idx],
        "isic_ids": [str(isic_ids[i]) for i in np.asarray(idx, int)],
        **(extra or {}),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(snap, f, indent=1)
    return fp


def load_snapshot(path):
    with open(path) as f:
        return json.load(f)


def verify_snapshot(snap, isic_ids):
    """Il campione salvato descrive ANCORA questo validation set?

    Se il fold viene ricostruito diversamente, gli indici posizionali puntano ad altre
    lesioni: qui si solleva invece di produrre numeri plausibili e sbagliati.
    """
    idx = np.asarray(snap["positional_idx"], int)
    if int(np.max(idx)) >= len(isic_ids):
        raise RuntimeError(
            f"snapshot fuori range: indice max {int(np.max(idx))} su {len(isic_ids)} righe "
            f"di validation. Il fold e' cambiato: rigenera il campione.")
    have = [str(isic_ids[i]) for i in idx]
    if have != list(snap["isic_ids"]):
        n_bad = sum(a != b for a, b in zip(have, snap["isic_ids"]))
        raise RuntimeError(
            f"snapshot NON corrisponde al validation set corrente: {n_bad}/{len(idx)} id "
            f"diversi alle stesse posizioni. Gli artefatti gia' prodotti con l'impronta "
            f"{snap['fingerprint']} NON sono confrontabili con questa sessione.")
    return snap["fingerprint"]


# --------------------------------------------------------------------------------------
# DOVE VANNO GLI ARTEFATTI DEL CAMPIONE  (rifatto il 2026-08-01, richiesta di Claudio:
# "ogni artefatto relativo alla run deve stare nella cartella originale della run")
#
# Prima esisteva una radice parallela `runs/_subset/<run_id__>/`, che replicava l'albero
# delle run e teneva gli artefatti lontani dai canonici. Teneva davvero, ma spargeva la
# stessa run su due posti e non era la struttura da portarsi sui dataset nuovi.
#
# Ora si scrive DENTRO la run, in sottocartelle che nessun motore canonico usa:
#
#     <run_dir>/xai/shap_tab/   attribuzione tabellare allineata + fedelta' con segno
#     <run_dir>/xai/lore_sa/    LORE-SA (regole per campione)
#
# La separazione dai canonici la fa il NOME della cartella, non la distanza: `xai/shap/`,
# `xai/gradcam/`, `xai/lore/` e `xai/faithfulness/` restano intatti e nessuna scrittura
# nuova puo' finirci dentro (cfr. il SUBSET GUARD in `xai_faithfulness`, che continua a
# rifiutare `xai/faithfulness/` finche' un campione e' adottato).
#
# La stampigliatura di ripresa vive nella stessa sottocartella dell'artefatto, quindi le
# due riprese sono indipendenti per costruzione e il prefisso `lore-` sull'impronta --
# che serviva solo a separarle quando condividevano una directory -- non serve piu'.
# --------------------------------------------------------------------------------------
SUBSET_KINDS = ("shap_tab", "lore_sa", "gradcam_img")

# Le cartelle nuove per motore, e i "compiti" che ciascuna ospita. `shap_tab` ne ha DUE
# (gli array SHAP e la fedelta' con segno), quindi li' la stampigliatura deve dire quale.
SUBSET_TASKS = {"shap_tab": ("faith", "shap"), "lore_sa": ("lore",),
                "gradcam_img": ("gradcam",)}


def run_dir_abs(run_dir, art_path):
    """Ri-radica un `run_dir` letto da un CSV sull'albero degli artefatti LOCALE.

    I CSV di riepilogo (`master_results_rows.csv`, i `*_combined.csv` per famiglia) portano
    `run_dir` come percorso ASSOLUTO della macchina che ha fatto il training —
    `/home/claudio/isic24-dum/...` — che qui non esiste. Usato grezzo da' un
    `ValueError: File not found` su un percorso che *sembra* giusto, ed e' un errore che
    compare solo su una macchina diversa da quella di origine: cioe' sempre, tranne dove e'
    stato scritto.

    Questa e' la SESTA copia di `split("artifacts/enhanced/")[-1]` — le altre cinque stanno
    in `xai_faithfulness`, `xai_driver`, `xai_metrics` (x2) e piu' sotto in questo modulo.
    E' la voce «doppioni di mappatura» della lista di pulizia: le altre vanno tirate qui.
    """
    return os.path.join(str(art_path), str(run_dir).split("artifacts/enhanced/")[-1])


def out_dir_for(run_dir, kind):
    """Dove vanno gli artefatti di `kind` per questa run. `kind` in SUBSET_KINDS."""
    if kind not in SUBSET_KINDS:
        raise ValueError(f"kind={kind!r}: attesi {SUBSET_KINDS}")
    return os.path.join(run_dir, "xai", kind)


def stamp_path(run_dir, kind, fp, task=None):
    """Il file che dice 'questo compito e' gia' fatto CON QUESTA impronta'.

    ⚠️ `task=None` deve produrre ESATTAMENTE `DONE.<fp>.json`, il nome che hanno gia' le
    110 stampigliature della fedelta' e le 8 di LORE.  Cambiarlo significherebbe rifare
    sei ore di fedelta' al primo rilancio, senza che nessuno se ne accorga finche' la GPU
    non e' occupata.  Il `task` esiste solo per il secondo compito di `shap_tab` (gli array
    SHAP), che convive con il primo nella stessa cartella.
    """
    name = f"DONE.{fp}.json" if task is None else f"DONE.{task}.{fp}.json"
    return os.path.join(out_dir_for(run_dir, kind), name)


def is_done(run_dir, kind, fp, task=None):
    return os.path.exists(stamp_path(run_dir, kind, fp, task))


def mark_done(run_dir, kind, fp, payload=None, task=None):
    """Scritto SOLO a calcolo finito.

    E' la ragione per cui l'interruzione e' sicura: una run interrotta a meta' non lascia
    stampigliatura, quindi al riavvio viene rifatta da capo. La granularita' della ripresa
    e' la RUN, non il campione."""
    p = stamp_path(run_dir, kind, fp, task)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"run_dir": run_dir, "kind": kind, "task": task or "default",
                   "fingerprint": fp, **(payload or {})}, f, indent=1)
    return p


def resume_report(run_dirs, kind, fp, task=None):
    """(da_fare, gia_fatte) — da stampare PRIMA di lanciare, cosi' si sa cosa costera'.

    `run_dirs` e' un dict {run_id: run_dir} (o un iterabile di coppie): la ripresa e' per
    run, e le chiavi tornano indietro cosi' che il chiamante stampi i run_id e non i path.
    """
    items = list(run_dirs.items() if hasattr(run_dirs, "items") else run_dirs)
    done = [rid for rid, d in items if is_done(d, kind, fp, task)]
    todo = [rid for rid, d in items if not is_done(d, kind, fp, task)]
    return todo, done


# Il piano di lavoro: quale motore scrive dove, con quale stampigliatura, e quanto costa.
# `sec_per_sample` sono misure, non stime: SHAP 30/07, LORE 01/08 su 8 run (23.2 min a
# n=48). Grad-CAM e' None finche' non lo si misura sulla prima run.
JOBS = {
    "shap":    dict(kind="shap_tab",    task="shap",  engines=("shap",),    sec_per_sample=1.2),
    "lore":    dict(kind="lore_sa",     task=None,    engines=("lore",),    sec_per_sample=29.0),
    "gradcam": dict(kind="gradcam_img", task=None,    engines=("gradcam",), sec_per_sample=None),
}


# --------------------------------------------------------------------------------------
# Statistica con il RUN come unita'
#
# Tutte le run della XAI vedono lo STESSO campione di validation. Quindi 10 run x 32
# campioni NON sono 320 osservazioni: sono 32 lesioni viste da 10 modelli. Poolare le righe
# fra run viola l'indipendenza nel modo peggiore e il p-value che ne esce e' ottimistico.
#
# Costato caro il 31/07: il class-wise LORE dava chi2 poolato p=0.0016 ("il risultato piu'
# solido"); con il run come unita' diventa Wilcoxon p=0.047 e test dei segni p=0.75 --
# suggestivo, non stabilito.
# --------------------------------------------------------------------------------------

def across_runs(per_run_a, per_run_b, name_a="A", name_b="B", drop_ties=True):
    """Contrasto appaiato fra due quantita' misurate su ogni run.

    Un valore per RUN, non per campione. Restituisce test dei segni e Wilcoxon: il primo usa
    solo la direzione, il secondo anche le ampiezze, e vanno letti insieme -- se divergono,
    l'effetto sta in pochi run con differenze grandi.

    ``drop_ties`` toglie i run con differenza esattamente nulla. Vanno tolti E DICHIARATI:
    tipicamente sono saturi per costruzione (copertura 0 vs 0, o 1 vs 1) e non possono
    mostrare l'effetto, ma il test dei segni li conterebbe come non-positivi.
    """
    from scipy import stats

    a = np.asarray(per_run_a, float)
    b = np.asarray(per_run_b, float)
    if a.shape != b.shape:
        raise ValueError(f"servono valori appaiati per run: {a.shape} vs {b.shape}")
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    d = b - a

    inf = d != 0 if drop_ties else np.ones_like(d, bool)
    n_inf = int(inf.sum())
    res = {
        "n_run": int(d.size), "n_informativi": n_inf, "n_pari": int(d.size - n_inf),
        "n_positivi": int((d > 0).sum()),
        "diff_mediana": float(np.median(d)) if d.size else float("nan"),
        "diff_media": float(np.mean(d)) if d.size else float("nan"),
        "name_a": name_a, "name_b": name_b,
    }
    res["segni_p"] = (float(stats.binomtest(int((d[inf] > 0).sum()), n_inf, 0.5).pvalue)
                      if n_inf else float("nan"))
    res["wilcoxon_p"] = (float(stats.wilcoxon(d[inf]).pvalue) if n_inf >= 6 else float("nan"))
    if n_inf < 6:
        # Con meno di 6 run informativi il Wilcoxon non puo' scendere sotto 0.05 nemmeno con
        # tutte le differenze concordi: restituirlo darebbe l'idea di un test che non c'e'.
        res["nota"] = f"solo {n_inf} run informativi: Wilcoxon non applicabile (servono >=6)"
    return res


def print_across_runs(res, indent="  "):
    """Le stesse cifre in forma leggibile, coi caveat attaccati invece che in una nota a parte."""
    print(f"{indent}{res['name_b']} vs {res['name_a']} — unita' = run, non campione")
    print(f"{indent}  run: {res['n_run']} totali, {res['n_informativi']} informativi, "
          f"{res['n_pari']} a differenza nulla (saturi/degeneri, esclusi dai test)")
    print(f"{indent}  a favore di {res['name_b']}: {res['n_positivi']}/{res['n_run']}")
    print(f"{indent}  differenza mediana {res['diff_mediana']:+.3f} (media {res['diff_media']:+.3f})")
    print(f"{indent}  test dei segni  p = {res['segni_p']:.4f}"
          if res['segni_p'] == res['segni_p'] else f"{indent}  test dei segni  n/d")
    if res.get("nota"):
        print(f"{indent}  Wilcoxon        n/d — {res['nota']}")
    else:
        print(f"{indent}  Wilcoxon        p = {res['wilcoxon_p']:.4f}")
    if res['n_run'] < 8:
        print(f"{indent}  ⚠ {res['n_run']} run: qualunque p qui e' fragile. Piu' seed, non piu' campioni.")


# --------------------------------------------------------------------------------------
# Il punto di iniezione unico
# --------------------------------------------------------------------------------------

def adopt(validation_df, n=None, balance="pool", seed=0, snapshot_dir=None,
          apply_to=("faithfulness", "lore", "shap"), verbose=True):
    """Sceglie il campione XAI **una volta per sessione** e lo rende il default di tutti.

    Va chiamata subito dopo `xf.configure(...)`, negli STESSI due punti nei due notebook.
    Da li' in poi nessuna cella deve passare `subset_idx`: la fedelta' lo eredita da
    `xf.FAITH_SUBSET_IDX`, LORE e SHAP dai loro kwargs nel driver. E' questo che rende la
    scelta un default del framework invece di un argomento da ricordarsi.

    Campiona su **lesioni distinte** (`unique_lesion_subset`): `pos_sample` con
    `replace=True` fa comparire la stessa lesione piu' volte, e le copie sono repliche
    esatte che restringono gli intervalli senza aggiungere informazione.

    Ritorna ``(idx, info)``.  ``adopt(validation_df, balance=None)`` disarma tutto e
    ripristina il prefisso storico, che e' come si riproducono i numeri pubblicati.
    """
    from . import xai_driver as xd
    from . import xai_faithfulness as xf

    if balance is None:                                  # ripristino del comportamento storico
        xf.FAITH_SUBSET_IDX = None
        xf.ACTIVE_SUBSET_FINGERPRINT = None
        xd.LORE_SA_KWARGS.pop("subset_idx", None)
        xd.SHAP_KWARGS.pop("subset_idx", None)
        xd.SHAP_KWARGS.pop("subset_fingerprint", None)
        if verbose:
            print("[campione] DISARMATO — si torna alle prime N righe (comportamento storico).")
        return None, {"balance": None}

    ids = validation_df.isic_id.values
    y = validation_df.target.values
    idx, info = unique_lesion_subset(ids, y, n=n, seed=seed, balance=balance)

    kind = f"unique-{balance}"
    snapshot_dir = snapshot_dir or "."
    path = os.path.join(snapshot_dir, f"subset.{kind}.n{len(idx)}.seed{seed}.json")
    if os.path.exists(path):
        snap = load_snapshot(path)
        fp = verify_snapshot(snap, ids)          # solleva se il fold non e' piu' lo stesso
        idx = np.asarray(snap["positional_idx"], int)
    else:
        fp = save_snapshot(path, ids, idx, seed, len(idx), kind=kind, extra={"info": info})

    if "faithfulness" in apply_to:
        xf.FAITH_SUBSET_IDX = idx
        xf.ACTIVE_SUBSET_FINGERPRINT = fp
    if "lore" in apply_to:
        xd.LORE_SA_KWARGS["subset_idx"] = idx
    if "shap" in apply_to:
        # max_explain_samples DEVE coprire il campione, altrimenti `_subsample` ne ripesca a
        # caso un sottoinsieme e l'npz non corrisponde piu' all'impronta che porta stampata.
        xd.SHAP_KWARGS["subset_idx"] = idx
        xd.SHAP_KWARGS["subset_fingerprint"] = fp
        xd.SHAP_KWARGS["max_explain_samples"] = len(idx)
        xd.SHAP_KWARGS["max_samples"] = int(idx.max()) + 1
        xd.SHAP_KWARGS["max_batches"] = None

    if verbose:
        n_u = len(set(ids[idx]))
        print(f"[campione] {len(idx)} lesioni DISTINTE (balance={balance!r}, seed={seed})")
        print(f"           per classe: {info['per_class']}   prevalenza {float(np.mean(y[idx])):.3f}")
        print(f"           uniche disponibili: {info['n_unique_available']}")
        print(f"           verifica: {n_u}/{len(idx)} id distinti "
              f"{'OK' if n_u == len(idx) else '** DUPLICATI **'}")
        print(f"           impronta: {fp}")
        print(f"           attivo su: {', '.join(apply_to)}")
        print(f"           snapshot: {path}")
        print(f"           ripristino del prefisso storico: xs.adopt(validation_df, balance=None)")
    return idx, {**info, "fingerprint": fp, "snapshot": path}


# --------------------------------------------------------------------------------------
# Rendiconto CROSS-RUN — tabelle e figure di confronto fra famiglie
#
# Sta qui, e non in una cella del notebook, per la richiesta del 2026-08-01: "voglio che la
# XAI crei direttamente i plot, non che dobbiamo aggiungere ulteriori celle". I loop della
# fedelta' e di LORE la chiamano in coda; nessuna cella in piu' da ricordarsi di eseguire.
#
# Le figure PER RUN le fanno i motori (`xai_faithfulness._plot_aligned_signed_run`,
# `xai_lore_sa._plot_lore_sa_run`) dentro la run. Qui ci finisce solo cio' che per
# definizione non appartiene a una run sola.
# --------------------------------------------------------------------------------------
SIGNED_MODES_REPORT = ("posev", "negev", "target")   # `abs` e' il modo-artefatto: ~0, rumore
Q_CLEAN = 0.20                                       # full budget e' OOD-dominato


def _run_gains(run_dir, prefix="faithfulness_aligned_signed", modes=("abs",) + SIGNED_MODES_REPORT):
    """Tutti i gain di una run, CON le convenzioni di segno gia' applicate.

    Mai dallo scalare del JSON: `faithfulness_tabular_del_gain` e' PRE-convenzione
    (`negev` va negato, `target` firmato per campione prima della media). Si passa dagli
    npz, che portano dentro anche le `labels` — quindi non serve validation_df.
    """
    from . import xai_faithfulness as xf
    d = out_dir_for(run_dir, "shap_tab")
    rec = {}
    for mode in modes:
        p = os.path.join(d, f"{prefix}_{mode}_curves.npz")
        if not os.path.exists(p):
            return None
        with np.load(p) as z:
            labels = z["labels"]; sgn = xf.mode_sign(mode, labels)
            rec["n"] = len(labels); rec["pos_rate"] = float(labels.mean())
            for direction in ("del", "ins"):
                g = xf.persample_gains(z, "tabular", direction, q_max=None)
                rec[f"{mode}_{direction}"] = float(np.mean(g * sgn))
                gq = xf.persample_gains(z, "tabular", direction, q_max=Q_CLEAN)
                rec[f"{mode}_{direction}_q{int(Q_CLEAN * 100)}norm"] = (
                    float(np.mean(gq * sgn)) / xf.gain_window(z, Q_CLEAN))
    return rec


def _lore_quality(run_dir):
    """Qualita' LORE-SA di una run, con la scomposizione per classe.

    `*_5050` = media delle due medie di classe. E' la colonna da citare quando il campione
    non e' bilanciato: neutralizza la composizione senza buttare via campioni.
    """
    import glob as _glob
    import pandas as pd
    hits = sorted(_glob.glob(os.path.join(out_dir_for(run_dir, "lore_sa"), "epoch*",
                                          "lore_sa_persample_*.csv")))
    if not hits:
        return None
    d = pd.read_csv(hits[-1])
    d["covf"] = (d.neigh_n_classes > 1).astype(float)
    g = d.groupby("true_label").agg(c=("covf", "mean"), r=("rule_len", "mean"))
    return dict(n=len(d), pos_rate=float(d.true_label.mean()),
                n_pos=int(d.true_label.sum()), n_neg=int((1 - d.true_label).sum()),
                lore_coverage=d["covf"].mean(), lore_coverage_5050=g["c"].mean(),
                rule_len=d["rule_len"].mean(), rule_len_5050=g["r"].mean(),
                cov_y0=g["c"].get(0, np.nan), cov_y1=g["c"].get(1, np.nan),
                rl_y0=g["r"].get(0, np.nan), rl_y1=g["r"].get(1, np.nan),
                neigh_pos_rate=d["neigh_pos_rate"].mean(),
                fid_insample=d["fid_insample"].mean(),
                n_warning=int(d.warning.notna().sum()), n_error=int(d.error.notna().sum()))


def report(master, art_path, kinds=("shap_tab", "lore_sa"), central=None, tag="",
           out_dir=None, verbose=True, encoded_feature_names=None, lore_idx=None,
           shape_metrics=True):
    """Tabelle + figure di confronto fra famiglie, dagli artefatti gia' su disco. Zero GPU.

    `master`  il DataFrame di `master_results_rows.csv` (colonne impl_name, setting_tag,
              seed, run_id, run_dir).
    `central` {impl_name: setting_tag} dei setting rappresentativi. **Serve**: il master
              ha 45 righe di `fixed_scaling`, `a0p000` (degenere, tutti zeri) compresa, e
              un `groupby("impl_name")` secco le mescola tutte nella media di famiglia.
              None = si usano tutte le run, e viene stampato l'avviso.
    `tag`     forza il suffisso dei file. Default: derivato dall'`n` dei dati di ciascun
              kind (`_n144` per la fedelta', `_n48` per LORE), cosi' i due non possono
              etichettarsi con l'n dell'altro.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from .xai_engines import _finalize_plot

    out_dir = out_dir or os.path.join(art_path, "runs", AGGREGATE_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)

    def _sfx(df):
        """Suffisso DERIVATO dai dati, non dal chiamante.

        `shap_tab` e `lore_sa` girano su n diversi (144 e 48): un `tag` unico passato da
        fuori ne etichetta uno dei due con l'n dell'altro, e il file sbagliato resta su
        disco quando il prossimo giro scrive col nome giusto. Se l'n non e' unico fra le
        run si torna al `tag` del chiamante, che almeno non mente su un valore singolo.
        """
        if tag:
            return f"_{tag}"
        ns = sorted(set(int(v) for v in df["n"])) if len(df) else []
        return f"_n{ns[0]}" if len(ns) == 1 else ""
    # il master porta i path della macchina su cui e' stato scritto: si riancora su art_path
    rd = {r.run_id: os.path.join(art_path, str(r.run_dir).split("artifacts/enhanced/")[-1])
          for _, r in master.iterrows()}
    written, out = [], {}

    if "shap_tab" in kinds:
        rows = []
        for _, r in master.iterrows():
            rec = _run_gains(rd[r.run_id])
            if rec is not None:
                rows.append(dict(impl_name=r.impl_name, setting_tag=r.setting_tag,
                                 seed=r.seed, run_id=r.run_id, **rec))
        faith = pd.DataFrame(rows)
        out["faith"] = faith
        if len(faith):
            sfx = _sfx(faith)
            p = os.path.join(out_dir, f"faithfulness_aligned_signed{sfx}.csv")
            faith.to_csv(p, index=False); written.append(p)
            if central:
                sel = faith[[central.get(i) == t
                             for i, t in zip(faith.impl_name, faith.setting_tag)]]
            else:
                sel = faith
                if verbose:
                    print("  ⚠ central=None: la media di famiglia include TUTTI i setting, "
                          "anche quelli degeneri")
            col = f"target_del_q{int(Q_CLEAN * 100)}norm"
            order = sel.groupby("impl_name")[col].mean().sort_values().index.tolist()
            out["central"], out["order"] = sel, order

            cmap = dict(zip(order, plt.cm.viridis(np.linspace(.05, .85, len(order)))))
            fig, axes = plt.subplots(1, len(SIGNED_MODES_REPORT),
                                     figsize=(4.2 * len(SIGNED_MODES_REPORT), 4.4),
                                     sharey=True, squeeze=False)
            for ax, mo in zip(axes[0], SIGNED_MODES_REPORT):
                c = f"{mo}_del_q{int(Q_CLEAN * 100)}norm"
                for x, fam in enumerate(order):
                    v = sel.loc[sel.impl_name == fam, c].values
                    ax.scatter(np.full(len(v), x) + np.linspace(-.12, .12, max(len(v), 1)),
                               v, s=22, color=cmap[fam], alpha=.75, zorder=3)
                    if len(v):
                        ax.hlines(v.mean(), x - .3, x + .3, color=cmap[fam], lw=2.5, zorder=4)
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels([f.replace("_", "\n") for f in order], fontsize=8)
                ax.set_title(f"`{mo}`"); ax.grid(axis="y", alpha=.25)
            axes[0][0].set_ylabel(f"deletion gain_norm (q$\\leq${Q_CLEAN})")
            n_seed = sel.groupby("impl_name").size().min() if len(sel) else 0
            fig.suptitle(f"Fedelta' tabellare allineata — {len(sel)} run, "
                         f"{sel.impl_name.nunique()} famiglie, min {n_seed} seed", y=1.0)
            _finalize_plot(out_dir, f"faithfulness_by_family{sfx}.png")
            written.append(os.path.join(out_dir, f"faithfulness_by_family{sfx}.png"))

    if "lore_sa" in kinds:
        rows = []
        for _, r in master.iterrows():
            rec = _lore_quality(rd[r.run_id])
            if rec is not None:
                rows.append(dict(impl_name=r.impl_name, setting_tag=r.setting_tag,
                                 seed=r.seed, run_id=r.run_id, **rec))
        lore = pd.DataFrame(rows)
        # Le run LORE possono avere lo STESSO n ma campioni diversi (il 01/08 `vanilla` ne
        # aveva tre a n=48, una sul prefisso dei 144 e due sul campione bilanciato). Se la
        # prevalenza non e' unica, la media di famiglia mescola popolazioni diverse.
        if len(lore) and "pos_rate" in lore:
            for _f, _g in lore.groupby("impl_name"):
                if _g.pos_rate.round(3).nunique() > 1 and verbose:
                    print(f"  ⚠ {_f}: run LORE su campioni DIVERSI "
                          f"(pos_rate {sorted(_g.pos_rate.round(3).unique())}) — "
                          f"la media di famiglia mescola popolazioni, rifare le vecchie")
        out["lore"] = lore
        if len(lore):
            sfx = _sfx(lore)
            p = os.path.join(out_dir, f"lore_sa_quality{sfx}.csv")
            lore.to_csv(p, index=False); written.append(p)
            lo = lore.groupby("impl_name")[["lore_coverage", "lore_coverage_5050",
                                            "rule_len", "rule_len_5050"]].mean()
            lo = lo.reindex([f for f in out.get("order", lo.index) if f in lo.index])
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
            for ax, (raw, bal, name) in zip(axes, [
                    ("lore_coverage", "lore_coverage_5050", "lore_coverage"),
                    ("rule_len", "rule_len_5050", "rule_len")]):
                x = np.arange(len(lo))
                ax.bar(x - .2, lo[raw], .4, color="#b0b0b0",
                       label=f"grezzo (pos_rate {lore.pos_rate.mean():.2f})")
                ax.bar(x + .2, lo[bal], .4, color="#2b7bba", label="riponderato 50/50")
                ax.set_xticks(x)
                ax.set_xticklabels([f.replace("_", "\n") for f in lo.index], fontsize=8)
                ax.set_title(name); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.25)
            unbal = abs(lore.pos_rate.mean() - .5) > .05
            fig.suptitle("LORE-SA per famiglia" + (
                f" — ⚠ campione SBILANCIATO ({int(lore.n_pos.mean())} pos / "
                f"{int(lore.n_neg.mean())} neg): ordinamento si', livelli no" if unbal
                else " — campione bilanciato"), y=1.0)
            _finalize_plot(out_dir, f"lore_sa_quality{sfx}.png")
            written.append(os.path.join(out_dir, f"lore_sa_quality{sfx}.png"))

    # ---- metriche di FORMA che sono CROSS-RUN per natura --------------------------------
    # Compattezza e rule compactness le scrivono i motori mentre calcolano (`ShapCallback`,
    # `LoreSACallback`).  Queste due no, e non possono: la stabilita' confronta DUE run e
    # l'accordo SHAP<->LORE ne incrocia due explainer.  Girano qui, in coda ai loop, che e'
    # il primo momento in cui tutte le run esistono.
    if shape_metrics and encoded_feature_names is not None:
        from . import xai_metrics as xmet
        shape = xmet.collect(master, art_path,
                             encoded_feature_names=encoded_feature_names, lore_idx=lore_idx)
        stab = xmet.stability_table(master, art_path, encoded_feature_names)
        for df, name in ((shape, "xai_shape_metrics.csv"), (stab, "xai_stability.csv")):
            if len(df):
                p = os.path.join(out_dir, name)
                df.to_csv(p, index=False)
                written.append(p)
        out["shape"], out["stability"] = shape, stab
        out["by_family"] = xmet.summary_by_family(art_path, shape_df=shape, stab_df=stab,
                                                  central=central, out_dir=out_dir,
                                                  verbose=False)
        written.append(os.path.join(out_dir, "xai_summary_by_family.csv"))

    if verbose:
        print(f"[report] {len(written)} file in {out_dir}")
        for p in written:
            print("   ", os.path.basename(p))
    out["written"] = written
    return out


def sample_matches(run_dir, kind, expected_idx):
    """Gli artefatti di `kind` sono stati calcolati SU QUESTI campioni?  (True/False/None)

    L'impronta nella stampigliatura identifica il campione-PADRE (le 144 lesioni), non il
    sottoinsieme che un motore ha davvero spiegato.  LORE ne spiega 48: il 31/07 erano i
    primi 48 dei 144 ordinati (35 pos / 13 neg), dal 01/08 sono 24+24 espliciti.  Stessa
    impronta, due campioni diversi — quindi "gia' fatto" da solo non basta, o il rifacimento
    salterebbe proprio le run che vanno rifatte.

    Ritorna ``None`` quando l'artefatto non registra quali righe ha spiegato (tutti quelli
    prodotti prima che `val_index` esistesse): "non lo so" e' un'informazione diversa da
    "non corrisponde", e chi chiama decide se rifare.
    """
    import glob as _glob
    exp = np.asarray(expected_idx, int)
    d = out_dir_for(run_dir, kind)
    if kind == "lore_sa":
        hits = sorted(_glob.glob(os.path.join(d, "epoch*", "lore_sa_persample_*.csv")))
        if not hits:
            return None
        import pandas as pd
        df = pd.read_csv(hits[-1])
        if "val_index" not in df.columns:
            return None
        got = df["val_index"].to_numpy()
    elif kind == "shap_tab":
        hits = sorted(_glob.glob(os.path.join(d, "epoch_*", "shap_arrays_epoch_*.npz")))
        if not hits:
            return None
        with np.load(hits[-1]) as z:
            got = np.asarray(z["explained_indices"], int).ravel()
    elif kind == "gradcam_img":
        hits = sorted(_glob.glob(os.path.join(d, "gradcam_arrays*_epoch_*.npz")))
        if not hits:
            return None
        with np.load(hits[-1]) as z:
            if "val_index" not in z.files:
                return None
            got = np.asarray(z["val_index"], int).ravel()
    else:
        return None
    return len(got) == len(exp) and bool(np.array_equal(np.sort(got), np.sort(exp)))


def resume_verified(run_dirs, kind, fp, expected_idx, task=None):
    """Come `resume_report`, ma "fatto" significa **fatto su QUESTO campione**.

    Ritorna ``(todo, done, da_rifare)``: le run stampigliate ma su campioni diversi (o su
    campioni non registrati) finiscono in `da_rifare`, non in `done`.  E' la differenza fra
    "non rifare cio' che e' fatto" e "non rifare niente".
    """
    items = list(run_dirs.items() if hasattr(run_dirs, "items") else run_dirs)
    todo, done, redo = [], [], []
    for rid, d in items:
        if not is_done(d, kind, fp, task):
            todo.append(rid)
            continue
        ok = sample_matches(d, kind, expected_idx)
        (done if ok is True else redo).append(rid)
    return todo, done, redo
