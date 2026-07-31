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

Il rimedio e' deduplicare su `isic_id`, non stratificare. NON ancora implementato: e' il
pezzo da scrivere, e va deciso insieme a quale n serve davvero una volta noto l'n efficace.

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

# Radice di default per tutto cio' che nasce dal campione stratificato. Fuori dalle run
# canoniche di proposito: cancellarla per intero non deve perdere niente di pubblicato.
STRATIFIED_ROOT = os.path.join("runs", "_stratified")


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
# Ripresa — per la radice stratificata, dove ne' xai_status.json ne' il write guard arrivano
# --------------------------------------------------------------------------------------
def stamp_path(out_root, run_id, fp):
    """Il file che dice 'questa run e' gia' fatta CON QUESTA impronta'."""
    return os.path.join(out_root, run_id.replace("/", "__"), f"DONE.{fp}.json")


def is_done(out_root, run_id, fp):
    return os.path.exists(stamp_path(out_root, run_id, fp))


def mark_done(out_root, run_id, fp, payload=None):
    """Scritto SOLO a calcolo finito.

    E' la ragione per cui l'interruzione e' sicura: una run interrotta a meta' non lascia
    stampigliatura, quindi al riavvio viene rifatta da capo. La granularita' della ripresa
    e' la RUN, non il campione."""
    p = stamp_path(out_root, run_id, fp)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"run_id": run_id, "fingerprint": fp, **(payload or {})}, f, indent=1)
    return p


def out_dir_for(out_root, run_id):
    """Dove vanno gli artefatti di questa run sotto la radice stratificata."""
    return os.path.join(out_root, run_id.replace("/", "__"))


def resume_report(out_root, run_ids, fp):
    """(da_fare, gia_fatte) — da stampare PRIMA di lanciare, cosi' si sa cosa costera'."""
    done = [r for r in run_ids if is_done(out_root, r, fp)]
    todo = [r for r in run_ids if not is_done(out_root, r, fp)]
    return todo, done


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
