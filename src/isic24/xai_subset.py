"""Campione stratificato casuale per la XAI, al posto del prefisso di `validation_ds`.

PERCHE' ESISTE (misurato il 2026-07-31)
---------------------------------------
`_build_faith_subset` e `LoreSACallback._freeze_batch` prendono le PRIME N righe di
`validation_ds`, costruito con `shuffle=False`.  Non e' un campione casuale: nei primi 32
ci sono 13/32 maligni (40.6%) contro il 9.1% del validation set
(`neg_sample=0.01` -> 364 positivi su 3995 righe).  P(>=13 | p=0.091) = 2e-6: il prefisso e'
arricchito **4.5x** nella classe positiva.

E' un errore SISTEMATICO, non rumore: il prefisso e' lo STESSO per tutte le 110 run, quindi
poolare i seed non lo riduce di un centesimo.  I confronti FRA strategie reggono (campione
identico per tutti = confronto appaiato); sono i LIVELLI ASSOLUTI a essere distorti, insieme
a ogni frase del tipo "sul validation set".

Il rimedio non e' "piu' campioni" ma "un campione stratificato casuale tenuto FISSO fra le
run": resta appaiato, e in piu' il livello e' corretto.

ATTENZIONE — I NUMERI CAMBIERANNO, ED E' IL PUNTO
-------------------------------------------------
Un campione proporzionale ha ~9% di maligni contro il ~40% del prefisso.  Le fedelta'
assolute si sposteranno: non e' una regressione, e' la correzione.  Per questo tutto quello
che sta qui scrive in una radice SEPARATA e non tocca un solo artefatto canonico.
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
    dell'arrotondamento), che e' esattamente cio' che rende non distorto il livello
    assoluto.  Deterministico dato `seed`.  Restituisce indici ORDINATI, perche'
    `_build_faith_subset` legge il dataset in ordine e poi indicizza.
    """
    y = np.asarray(labels).reshape(-1)
    n = int(n)
    if n > len(y):
        raise ValueError(f"n={n} > {len(y)} campioni disponibili nel validation set")
    rng = np.random.default_rng(seed)
    out = []
    classes = np.unique(y)
    # Quota proporzionale, poi si distribuiscono i resti dalla classe piu' grande in giu':
    # con n=256 su 9.1% di positivi il floor da' 23 positivi, non 22 o 24 a seconda del caso.
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
    y = None
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
