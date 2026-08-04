"""Evaluation-set anti-drift guard (added 2026-07-29).

Sibling of ``notebooks/modality_attendance/feature_space_guard.py``, for the other axis
along which post-hoc numbers silently stop being comparable: **which rows they were
computed on**.

Claudio, 2026-07-29: *"ottimo che adesso il validation è regolabile, ma per lo studio
primario è importante che rimanga lo stesso quando testato su differenti dataset o quando
vengono fatti studi post hoc. Gli esperimenti su validation diverso/più grande del canonico
sono esplorativi e non dovrebbero rimpiazzare il canonico."*

So the knobs stay -- ``EVAL_SET`` / ``EVAL_FOLDS`` / ``FAITH_N_SAMPLES`` are real and
useful -- but anything writing into the primary study's artifacts has to declare that it is
on the canonical set, and fail loud if it is not.  Exploratory variants keep running; they
just have to write somewhere else.

Usage, right after the post-hoc setup cells::

    from isic24.eval_guard import assert_canonical_eval
    assert_canonical_eval(globals())      # or assert_canonical_eval(NB) from a script
"""

from __future__ import annotations

import json
import os

CANONICAL_EVAL_SET = "canonical"
CANONICAL_FAITH_N = 128


def _adopted_subset():
    """Il campione XAI adottato, se c'e'.  ``(n, impronta)`` oppure ``(None, None)``.

    Sta in `xai_faithfulness`, non nel namespace: `xs.adopt()` lo installa come **default
    globale**, quindi da quel momento le misure NON sono piu' sulle prime `FAITH_N_SAMPLES`
    righe, qualunque cosa dica quella variabile.
    """
    try:
        from . import xai_faithfulness as xf
        idx = getattr(xf, "FAITH_SUBSET_IDX", None)
        return (len(idx) if idx is not None else None,
                getattr(xf, "ACTIVE_SUBSET_FINGERPRINT", None))
    except Exception:
        return None, None


def eval_set_descriptor(ns):
    """A short, loggable description of which rows the numbers will be computed on."""
    get = ns.get
    n_sub, fp = _adopted_subset()
    return {
        "eval_set": get("EVAL_SET"),
        "eval_folds": list(get("EVAL_FOLDS") or []) if get("EVAL_SET") == "custom" else None,
        "faith_n_samples": get("FAITH_N_SAMPLES"),
        "n_validation": len(get("validation_df")) if get("validation_df") is not None else None,
        "adopted_subset_n": n_sub,
        "adopted_subset_fingerprint": fp,
    }


def assert_canonical_eval(ns, faith_n=CANONICAL_FAITH_N, require_snapshot=True,
                          allow_subset=False):
    """Fail loud unless `ns` is the canonical evaluation set of the primary study.

    ``ns`` is the namespace holding the post-hoc notebook's globals: pass ``globals()``
    from a notebook cell, or the ``NB`` dict the harness scripts exec the cells into.

    Checks, in order:

    1. ``EVAL_SET == "canonical"`` -- a custom fold set mixes in TRAINING data for every
       run, so it is a metric-stability stress test, never held-out evaluation.
    2. ``FAITH_N_SAMPLES == faith_n`` -- the published faithfulness numbers integrate over
       the first 128 validation rows.  A different n is a different measurement.
    3. the validation set matches ``validation_ids.SEED49.json``.  This one follows the
       policy already written into the post-hoc notebook's cell 16: **warn**, because the
       shipped snapshot is a post-hoc regeneration that was never checked against the runs'
       real training split -- and **raise** once a real training run has rewritten it
       (``source == "training_run"``), at which point a mismatch means the post-hoc numbers
       do not correspond to the checkpoints at all.

    Returns the descriptor dict so callers can record it next to their output.
    """
    desc = eval_set_descriptor(ns)

    if desc["eval_set"] != CANONICAL_EVAL_SET:
        raise RuntimeError(
            f"[EVAL GUARD] EVAL_SET={desc['eval_set']!r} (folds {desc['eval_folds']}), non "
            f"{CANONICAL_EVAL_SET!r}. Le folds 1-4 sono dati di TRAINING per ogni run: e' uno "
            f"stress test di stabilita' della metrica, non una valutazione held-out, e i suoi "
            f"numeri sono ESPLORATIVI. Non devono finire negli artefatti dello studio primario: "
            f"rimetti EVAL_SET='canonical', oppure scrivi in una directory separata e dichiara "
            f"il set nel nome del file.")

    if desc["faith_n_samples"] != faith_n:
        raise RuntimeError(
            f"[EVAL GUARD] FAITH_N_SAMPLES={desc['faith_n_samples']} != {faith_n} canonico. "
            f"La fedelta' pubblicata integra sulle prime {faith_n} righe di validation; un n "
            f"diverso e' una misura diversa e non e' confrontabile con i numeri gia' "
            f"documentati.")

    # Il punto cieco chiuso il 2026-08-04: fino a qui il guard controllava una VARIABILE
    # (`FAITH_N_SAMPLES`) mentre cio' che decide le righe misurate e' `xf.FAITH_SUBSET_IDX`.
    # Dopo `xs.adopt()` stampava «OK, faith_n_samples: 128» con un campione da 144 attivo:
    # un guard che dichiara canonico un regime che non lo e' piu'.
    if desc["adopted_subset_n"] is not None and not allow_subset:
        raise RuntimeError(
            f"[EVAL GUARD] c'e' un campione XAI ADOTTATO ({desc['adopted_subset_n']} righe, "
            f"impronta {desc['adopted_subset_fingerprint']!r}), quindi le misure NON sono sulle "
            f"prime {faith_n} righe di validation — qualunque cosa dica FAITH_N_SAMPLES.\n"
            f"  `xs.adopt()` installa il campione come DEFAULT GLOBALE: da li' in poi "
            f"`subset_idx=None` non significa piu' «il prefisso».\n"
            f"  Se stai lavorando sul campione di proposito: allow_subset=True (e scrivi in una "
            f"directory che lo dichiara). Per tornare al canonico: xf.FAITH_SUBSET_IDX = None "
            f"e xf.ACTIVE_SUBSET_FINGERPRINT = None.")

    # Il set che conta e' quello DENTRO xf._CTX: e' da li' che `_build_faith_subset` pesca.
    # `ns["validation_df"]` puo' divergere se qualcuno lo ribinda dopo `xf.configure`.
    vdf = ns.get("validation_df")
    try:
        from . import xai_faithfulness as xf
        ctx_vdf = getattr(xf._CTX, "validation_df", None)
    except Exception:
        ctx_vdf = None
    if ctx_vdf is not None:
        desc["n_ctx_validation"] = len(ctx_vdf)
        if vdf is not None and len(ctx_vdf) != len(vdf):
            raise RuntimeError(
                f"[EVAL GUARD] il validation nel namespace ha {len(vdf)} righe ma quello "
                f"configurato in xf._CTX (da cui pesca ogni misura) ne ha {len(ctx_vdf)}. "
                f"Qualcuno ha ribindato validation_df DOPO xf.configure: rilancia le celle di "
                f"setup, altrimenti il guard controlla un set e il codice ne misura un altro.")
        vdf = ctx_vdf

    snap_path = ns.get("_val_snap") or os.path.join(
        ns.get("BASE_PATH", "."), "notebooks", "modality_attendance",
        "validation_ids.SEED49.json")
    if not os.path.exists(snap_path):
        msg = f"[EVAL GUARD] nessuno snapshot in {snap_path}: identita' del validation non verificabile."
        if require_snapshot:
            raise RuntimeError(msg + " Esegui una volta il notebook di training per scriverlo, "
                                     "oppure chiama con require_snapshot=False.")
        print(msg)
    elif vdf is not None:
        snap = json.load(open(snap_path))
        match = sorted(vdf.isic_id.tolist()) == snap["isic_ids_sorted"]
        src = snap.get("source", "?")
        desc["snapshot_source"], desc["snapshot_match"] = src, bool(match)
        if not match:
            detail = (f"[EVAL GUARD] il validation vivo ({len(vdf)} righe) NON coincide con lo "
                      f"snapshot ({snap['n']} righe, source={src}).")
            # Stessa politica della cella 16 del post-hoc: warn finche' lo snapshot e' una
            # rigenerazione post-hoc, raise quando viene da un training run vero.
            if src == "training_run":
                raise RuntimeError(detail + " Lo snapshot viene da un training run: i numeri "
                                            "post-hoc NON corrispondono ai checkpoint.")
            print(detail + " Lo snapshot e' una rigenerazione post-hoc non verificata contro lo "
                           "split di training, quindi e' un WARN e non un errore.")

    print(f"[EVAL GUARD] OK — {desc}")
    return desc
