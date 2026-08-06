"""Suite dei guard — ognuno deve SCATTARE, non solo tacere.

Motivo per cui esiste (2026-08-04): tre difetti nati tutti dallo stesso stampo — un
controllo che, sugli artefatti veri, tace sia quando funziona sia quando e' rotto. Rigirare
la pipeline non li distingue. Ogni guard qui ha, dove possibile, TRE casi:

    A. la condizione che deve far scattare il guard   -> deve sollevare
    B. la condizione legittima                        -> non deve sollevare
    C. il codice VECCHIO sul caso A                   -> deve restare muto

Il caso C e' quello che trasforma il test da "gira" a "prova che sto misurando il fix".

Uso:  python scripts/xai_2026-07-28/test_guards.py
Nessuna GPU, nessun checkpoint: gli artefatti servono solo dove indicato.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(BASE, "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from isic24 import paths as _paths0                           # noqa: E402
ART = str(_paths0.art_path())
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"  {'✓ PASS' if cond else '✗ FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    return bool(cond)


def raises(fn, exc=RuntimeError):
    try:
        fn()
        return False, ""
    except exc as e:
        return True, str(e).splitlines()[0][:70]
    except Exception as e:                                   # noqa: BLE001
        return False, f"eccezione INATTESA {type(e).__name__}: {e}"


def section(t):
    print(f"\n{'=' * 74}\n{t}\n{'=' * 74}")


TMP = tempfile.mkdtemp(prefix="guards_")
rng = np.random.default_rng(0)

# ======================================================================== 1 ===
section("1. [STABILITY] — due run con SHAP su campioni diversi")
from isic24 import xai_metrics as xmet                        # noqa: E402

NAMES = [f"f{i}" for i in range(20)] + [f"cat__{t}" for t in "abcde"]
N = 12


def make_shap_run(tag, val_index, root=TMP):
    d = os.path.join(root, tag, "xai", "shap_tab", "epoch_003")
    os.makedirs(d, exist_ok=True)
    np.savez(os.path.join(d, "shap_arrays_epoch_003.npz"),
             shap_values=rng.normal(size=(N, len(NAMES))),
             explained_indices=np.arange(N),          # SEMPRE 0..N-1: e' il punto
             val_index=np.asarray(val_index, int),
             subset_fingerprint=np.asarray("fp-test"))
    return os.path.join(root, tag)


run_same_a = make_shap_run("s_a", np.arange(1, N + 1))
run_same_b = make_shap_run("s_b", np.arange(1, N + 1))
run_diff = make_shap_run("s_c", np.arange(500, 500 + N))


def stability(dirs):
    m = pd.DataFrame([dict(impl_name="fixed_scaling", setting_tag=t, seed=47, run_dir=p)
                      for t, p in zip(("a0p250", "a0p500"), dirs)])
    return xmet.stability_table(m, "/", NAMES)


ok, msg = raises(lambda: stability([run_same_a, run_diff]))
check("A. campioni diversi -> solleva", ok, msg)
ok_b, msg_b = raises(lambda: stability([run_same_a, run_same_b]))
check("B. stesso campione -> non solleva", not ok_b, msg_b)

_orig = xmet._run_top_k


def _old_run_top_k(run_dir, groups, k=xmet.TOP_K):
    p = xmet._shap_npz(run_dir)
    with np.load(p) as z:
        phi = np.abs(np.asarray(z["shap_values"], float))
        idx = np.asarray(z["explained_indices"], int).ravel()      # il bug
    return [xmet.raw_top_k(r, groups, k) for r in phi], idx


xmet._run_top_k = _old_run_top_k
ok_c, _ = raises(lambda: stability([run_same_a, run_diff]))
xmet._run_top_k = _orig
check("C. codice vecchio resta muto (il guard era morto)", not ok_c)

# ======================================================================== 2 ===
section("2. sample_matches / resume_verified — 'fatto' = fatto SU QUESTO campione")
from isic24 import xai_subset as xs                           # noqa: E402

vi_ok = np.arange(1, N + 1)
check("A. shap_tab, campione giusto -> True",
      xs.sample_matches(run_same_a, "shap_tab", vi_ok) is True)
check("B. shap_tab, campione diverso (stessa lunghezza) -> False",
      xs.sample_matches(run_same_a, "shap_tab", vi_ok + 500) is False)
check("C. lunghezza diversa -> False",
      xs.sample_matches(run_same_a, "shap_tab", vi_ok[:5]) is False)

d_noval = os.path.join(TMP, "s_noval", "xai", "shap_tab", "epoch_003")
os.makedirs(d_noval, exist_ok=True)
np.savez(os.path.join(d_noval, "shap_arrays_epoch_003.npz"),
         shap_values=rng.normal(size=(N, len(NAMES))), explained_indices=np.arange(N))
check("D. npz senza val_index -> None ('non lo so' != 'va bene')",
      xs.sample_matches(os.path.join(TMP, "s_noval"), "shap_tab", vi_ok) is None)

fp = "fp-test"
os.makedirs(os.path.join(run_same_a, "xai", "shap_tab"), exist_ok=True)
xs.mark_done(run_same_a, "shap_tab", fp)
todo, done, redo = xs.resume_verified({"r": run_same_a}, "shap_tab", fp, vi_ok)
check("E. stampigliata + campione giusto -> done", done == ["r"] and not redo)
todo, done, redo = xs.resume_verified({"r": run_same_a}, "shap_tab", fp, vi_ok + 500)
check("F. stampigliata ma campione diverso -> DA RIFARE", redo == ["r"] and not done,
      "e' il caso che il 03/08 non poteva scattare")

# ======================================================================== 3 ===
section("3. [SUBSET GUARD] — riusare npz canonici sotto un campione adottato")
from isic24 import xai_faithfulness as xf                     # noqa: E402
from isic24 import xai_driver as xd                           # noqa: E402

canon = os.path.join(TMP, "canon", "xai", "shap", "epoch_003")
os.makedirs(canon, exist_ok=True)
np.savez(os.path.join(canon, "shap_arrays_epoch_003.npz"),        # SENZA impronta
         shap_values=rng.normal(size=(N, len(NAMES))), explained_indices=np.arange(N))
raw = pd.DataFrame({"a": np.arange(N)})

_prev_fp = xf.ACTIVE_SUBSET_FINGERPRINT
_prev_sub = xd.SHAP_KWARGS.get("subset_fingerprint")
try:
    # `shap_subdir()` deriva la cartella da qui: senza impronta legge `xai/shap/`, che e'
    # dove sta l'npz canonico che il guard deve rifiutare.
    xd.SHAP_KWARGS.pop("subset_fingerprint", None)
    check("0. shap_subdir() segue SHAP_KWARGS", xd.shap_subdir() == "shap")
    xf.ACTIVE_SUBSET_FINGERPRINT = "unique-equal-s0-n144-DIVERSA"
    ok, msg = raises(lambda: xf._load_shap_values(
        os.path.join(TMP, "canon"), 3, None, raw, None))
    check("A. npz senza impronta + campione attivo -> solleva", ok, msg)

    xf.ACTIVE_SUBSET_FINGERPRINT = None
    ok_b, msg_b = raises(lambda: xf._load_shap_values(
        os.path.join(TMP, "canon"), 3, None, raw, None))
    check("B. nessun campione attivo -> non solleva (regime canonico)", not ok_b, msg_b)
finally:
    xf.ACTIVE_SUBSET_FINGERPRINT = _prev_fp
    if _prev_sub is not None:
        xd.SHAP_KWARGS["subset_fingerprint"] = _prev_sub

# ======================================================================== 4 ===
section("4. [XAI WRITE GUARD] — non sovrascrivere una run canonica")
busy = os.path.join(TMP, "busy")
os.makedirs(os.path.join(busy, "xai", "shap", "epoch_003"), exist_ok=True)
open(os.path.join(busy, "xai", "shap", "epoch_003", "shap_arrays_epoch_003.npz"), "w").close()
empty = os.path.join(TMP, "empty")
os.makedirs(os.path.join(empty, "xai"), exist_ok=True)

ok, msg = raises(lambda: xd.assert_writable(busy, engines=("shap",)))
check("A. run con artefatti -> solleva", ok, msg)
ok_b, _ = raises(lambda: xd.assert_writable(empty, engines=("shap",)))
check("B. run vuota -> non solleva", not ok_b)
ok_c, _ = raises(lambda: xd.assert_writable(busy, engines=("shap",), allow_clobber=True))
check("C. allow_clobber=True -> passa", not ok_c)

interrupted = os.path.join(TMP, "interrupted")
os.makedirs(os.path.join(interrupted, "xai", "shap", "epoch_003"), exist_ok=True)  # VUOTA
ok_d, _ = raises(lambda: xd.assert_writable(interrupted, engines=("shap",)))
check("D. dir vuota da run interrotta -> NON blocca", not ok_d,
      "regressione del difetto a1p000 del 30/07")

# ======================================================================== 5 ===
section("5. LORE_SA_DEVICE non deve applicarsi agli altri motori")
_prev = xd.LORE_SA_DEVICE
try:
    xd.LORE_SA_DEVICE = "/CPU:0"
    ok, msg = raises(lambda: xd.run_posthoc_xai(
        TMP, engines=("lore", "shap"), best_epoch=3, best_val_auc=0.9))
    check("A. device + engines misti -> solleva", ok, msg)
finally:
    xd.LORE_SA_DEVICE = _prev

# ======================================================================== 6 ===
section("6. Costruzione del campione — le asserzioni di bilanciamento")
ids = np.array([f"L{i // 3}" for i in range(300)])            # 3 copie per lesione
y = np.array([1 if int(i.lstrip("L")) % 2 == 0 else 0 for i in ids])
idx, info = xs.unique_lesion_subset(ids, y, n=40, seed=0, balance="equal")
check("A. balance='equal' -> 50/50 esatto", int(y[idx].sum()) == len(idx) // 2,
      f"{int(y[idx].sum())} pos / {len(idx) - int(y[idx].sum())} neg")
check("B. nessuna lesione ripetuta", len(set(ids[idx])) == len(idx))
ok, msg = raises(lambda: xs.unique_lesion_subset(ids, y, n=99999, seed=0, balance="equal"),
                 exc=(ValueError, AssertionError))
check("C. n oltre il disponibile -> solleva", ok, msg)
ok, msg = raises(lambda: xs.unique_lesion_subset(ids, y[:10], n=10, seed=0), exc=ValueError)
check("D. ids/labels di lunghezza diversa -> solleva", ok, msg)

# ======================================================================== 7 ===
section("7. across_runs — l'unita' e' il run, e i pareggi vanno DICHIARATI")
ok, msg = raises(lambda: xs.across_runs([1, 2, 3], [1, 2]), exc=ValueError)
check("A. lunghezze diverse -> solleva", ok, msg)
r = xs.across_runs([1.0, 1.0, 0.5, 0.4, 0.3], [1.0, 1.0, 0.7, 0.6, 0.5])
check("B. i pareggi sono contati a parte", r["n_pari"] == 2 and r["n_informativi"] == 3,
      f"n_pari={r['n_pari']}, n_informativi={r['n_informativi']}")
check("C. <6 run informativi -> niente Wilcoxon", np.isnan(r["wilcoxon_p"]) and "nota" in r)

# ======================================================================== 8 ===
section("8. Convenzioni di segno — mode_sign si applica PER CAMPIONE")
lab = np.array([1, 0, 1, 0])
check("A. posev -> +1", np.all(xf.mode_sign("posev", lab) == 1))
check("B. negev -> -1", np.all(xf.mode_sign("negev", lab) == -1))
s = xf.mode_sign("target", lab)
check("C. target -> segno per campione, non scalare",
      np.asarray(s).size == len(lab) and not np.all(s == s.flat[0]), f"{np.asarray(s).ravel()}")
ok, msg = raises(lambda: xf.mode_sign("target", None), exc=ValueError)
check("D. target senza labels -> solleva", ok, msg)
ok, msg = raises(lambda: xf.mode_sign("boh", lab), exc=ValueError)
check("E. mode sconosciuto -> solleva", ok, msg)

# ======================================================================== 9 ===
section("9. Popolazioni: i glob cross-run prendono run SUPERATE?")
if os.path.isdir(os.path.join(ART, "runs")):
    import glob as _g
    canon_lore = _g.glob(f"{ART}/runs/*/*/*/*/xai/lore_sa/lore_sa_per_epoch.csv")
    all_lore = _g.glob(f"{ART}/runs/**/lore_sa_per_epoch.csv", recursive=True)
    check("A. glob ingenuo su lore_sa_per_epoch prende PIU' delle canoniche",
          len(all_lore) > len(canon_lore),
          f"{len(all_lore)} totali vs {len(canon_lore)} in xai/lore_sa/ — filtrare sul path")
    master = pd.read_csv(os.path.join(ART, "summaries", "master_results_rows.csv"))
    check("B. il registro canonico ha 110 run", len(master) == 110, f"{len(master)}")
    cf = os.path.join(ART, "runs", "_faithfulness", "combined_faithfulness.csv")
    if os.path.exists(cf):
        n = len(pd.read_csv(cf))
        check("C. combined_faithfulness.csv NON e' il registro", n != len(master),
              f"{n} righe vs {len(master)} run: non usarlo per medie di famiglia")
else:
    print("  (artefatti non disponibili: sezione saltata)")

# ======================================================================= 10 ===
section("10. [EVAL GUARD] — e il punto cieco sul campione adottato")
from isic24.eval_guard import assert_canonical_eval                # noqa: E402


def _ns(**kw):
    return dict({"EVAL_SET": "canonical", "FAITH_N_SAMPLES": 128,
                 "validation_df": None, "_val_snap": "/nonesistente"}, **kw)


def _call(ns, **kw):
    return assert_canonical_eval(ns, require_snapshot=False, **kw)


_pf, _pi = xf.ACTIVE_SUBSET_FINGERPRINT, getattr(xf, "FAITH_SUBSET_IDX", None)
try:
    xf.FAITH_SUBSET_IDX, xf.ACTIVE_SUBSET_FINGERPRINT = None, None
    ok, msg = raises(lambda: _call(_ns(EVAL_SET="custom", EVAL_FOLDS=[1, 2])))
    check("A. EVAL_SET custom -> solleva", ok, msg)
    ok, msg = raises(lambda: _call(_ns(FAITH_N_SAMPLES=256)))
    check("B. FAITH_N_SAMPLES diverso -> solleva", ok, msg)
    ok_c, msg_c = raises(lambda: _call(_ns()))
    check("C. regime canonico -> non solleva", not ok_c, msg_c)

    # il punto cieco: `xs.adopt` installa un default GLOBALE che il guard non guardava
    xf.FAITH_SUBSET_IDX = list(range(144))
    xf.ACTIVE_SUBSET_FINGERPRINT = "unique-equal-s0-n144-TEST"
    ok, msg = raises(lambda: _call(_ns()))
    check("D. campione ADOTTATO -> solleva (era il punto cieco)", ok, msg)
    ok_e, msg_e = raises(lambda: _call(_ns(), allow_subset=True))
    check("E. allow_subset=True -> passa", not ok_e, msg_e)
    d = _call(_ns(), allow_subset=True)
    check("F. il descrittore riporta il campione", d.get("adopted_subset_n") == 144,
          f"adopted_subset_n={d.get('adopted_subset_n')}")

    # Lo snapshot: la politica dipende da CHI l'ha scritto. Dal 2026-08-06 quello su disco
    # viene dal riallenamento (`source='training_run'`), quindi il ramo che conta e' il raise.
    xf.FAITH_SUBSET_IDX, xf.ACTIVE_SUBSET_FINGERPRINT = None, None
    _vdf = pd.DataFrame({"isic_id": [f"ISIC_{i:07d}" for i in range(10)]})

    def _snap_ns(source, ids):
        path = os.path.join(TMP, f"snap_{source}.json")
        import json as _json
        _json.dump({"seed": 49, "n": len(ids), "source": source, "isic_ids_sorted": sorted(ids)},
                   open(path, "w"))
        return _ns(validation_df=_vdf, _val_snap=path)

    _same = _vdf.isic_id.tolist()
    _other = [f"ISIC_{i:07d}" for i in range(1, 11)]
    ok, msg = raises(lambda: _call(_snap_ns("training_run", _other)))
    check("G. snapshot da un training run + set diverso -> SOLLEVA", ok, msg)
    ok_h, msg_h = raises(lambda: _call(_snap_ns("posthoc_regen_2026-07-28", _other)))
    check("H. snapshot rigenerato post-hoc + set diverso -> warn, non solleva", not ok_h, msg_h)
    d = _call(_snap_ns("training_run", _same))
    check("I. snapshot da un training run + set uguale -> passa e lo registra",
          d.get("snapshot_match") is True and d.get("snapshot_source") == "training_run",
          f"match={d.get('snapshot_match')}, source={d.get('snapshot_source')}")
finally:
    xf.ACTIVE_SUBSET_FINGERPRINT, xf.FAITH_SUBSET_IDX = _pf, _pi

# ======================================================================= 11 ===
section("11. IDENTITA' DELLO SPLIT — il validation di oggi e' quello dei checkpoint di giugno?")
# La domanda di PARTE 0: le 1091 righe su cui misuriamo sono davvero quelle da cui i 110
# checkpoint erano tenuti fuori? Il ponte con giugno e' il vettore `labels` (128 campioni)
# dentro le npz `xai/faithfulness/faithfulness_curves.npz`, scritte durante il training di
# giugno. `validation_ds` e' costruito con shuffle=False, quindi quei 128 sono esattamente
# `validation_df.head(128)`: confrontabili con una rigenerazione odierna.
#
# Il caso B e' quello che rende il test falsificabile: uno split PERTURBATO deve NON
# combaciare. Senza, un confronto che passa per costruzione non proverebbe niente.
import glob as _glob                                              # noqa: E402
import random as _random                                          # noqa: E402

_SEED, _NEG, _POS = 49, 0.01, 5.0
_PARQ = os.path.join(BASE, "data", "isic24", "enhanced",
                     "enhanced_train_metadata_clean.parquet")
_june = sorted(_glob.glob(os.path.join(
    ART, "runs", "*", "*", "seed_*", "*", "xai", "faithfulness", "faithfulness_curves.npz")))


def _regen_validation(rng_offset=0, shuffle_rs=None):
    """Replica delle celle 1/4/6/9 del notebook di training, su CPU e senza keras.

    `keras.utils.set_random_seed(49)` semina il RNG globale di numpy. Lo shuffle storico era
    `.sample(frac=1.0)` SENZA random_state, e quindi consumava proprio quel RNG: `rng_offset`
    simula il caso in cui il globale non e' piu' allo stato iniziale — per esempio dopo che
    `run_experiment` ha chiamato `set_random_seed(run_seed)`.

    `shuffle_rs=_SEED` e' la versione corretta il 04/08 nel notebook, che deve dare lo STESSO
    split **e** reggere a un RNG globale sporco.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    _random.seed(_SEED)
    np.random.seed(_SEED)
    for _ in range(rng_offset):
        np.random.random()
    d = pd.read_parquet(_PARQ).ffill()
    maj = d.query("target==0").sample(frac=_NEG, random_state=_SEED)
    mino = d.query("target==1").sample(frac=_POS, replace=True, random_state=_SEED)
    d = pd.concat([maj, mino], axis=0)
    d = (d.sample(frac=1.0) if shuffle_rs is None
         else d.sample(frac=1.0, random_state=shuffle_rs)).reset_index(drop=True)
    d["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=_SEED)
    for i, (_tr, _va) in enumerate(sgkf.split(d, y=d.target, groups=d.patient_id)):
        d.loc[_va, "fold"] = int(i)
    return d.query("fold!=0"), d.query("fold==0")


if not (os.path.exists(_PARQ) and _june):
    print("  (dataset o npz di giugno non disponibili: sezione saltata)")
else:
    _lab_june = np.load(_june[0])["labels"].astype(np.float32)

    # tutte le run di giugno devono condividere UN SOLO vettore di label: se lo split fosse
    # cambiato in corsa fra le sessioni di training, i 110 checkpoint non sarebbero appaiati.
    _sigs = {np.asarray(np.load(f)["labels"], dtype=np.float32).tobytes() for f in _june}
    check("A. un solo vettore `labels` su tutte le npz di giugno", len(_sigs) == 1,
          f"{len(_sigs)} distinti su {len(_june)} npz")

    _tr, _va = _regen_validation()
    check("B. la rigenerazione da' 4360/1091", (len(_tr), len(_va)) == (4360, 1091),
          f"train={len(_tr)} val={len(_va)}")

    _lab_now = _va.target.to_numpy()[:128].astype(np.float32)
    check("C. prefisso 128: label di oggi == label di giugno",
          np.array_equal(_lab_now, _lab_june),
          f"match {float((_lab_now == _lab_june).mean()):.3f}, n_pos {int(_lab_june.sum())}")

    # il caso che deve FALLIRE: split perturbato -> il confronto deve accorgersene
    _va_bad = _regen_validation(rng_offset=1)[1]
    _lab_bad = _va_bad.target.to_numpy()[:128].astype(np.float32)
    check("D. split perturbato -> il confronto NON combacia (il test sa fallire)",
          not np.array_equal(_lab_bad, _lab_june),
          f"match {float((_lab_bad == _lab_june).mean()):.3f}")

    # ...ma l'INSIEME resta lo stesso: e' l'ordine a muoversi, ed e' posizionale che si indicizza
    check("E. lo split perturbato cambia l'ORDINE, non l'insieme",
          set(_va_bad.isic_id) == set(_va.isic_id)
          and not np.array_equal(_va_bad.isic_id.to_numpy(), _va.isic_id.to_numpy()),
          "stesso set, ordine diverso")

    check("F. nessuna lesione e nessun paziente attraversa train/validation",
          not (set(_tr.isic_id) & set(_va.isic_id)) and not (set(_tr.patient_id) & set(_va.patient_id)),
          f"lesioni {len(set(_tr.isic_id) & set(_va.isic_id))}, "
          f"pazienti {len(set(_tr.patient_id) & set(_va.patient_id))}")

    # il fix del 04/08: shuffle seminato -> stesso split di giugno, E immune al RNG sporco
    _va_fix = _regen_validation(shuffle_rs=_SEED)[1]
    check("H. shuffle seminato -> stesso split di giugno",
          np.array_equal(_va_fix.target.to_numpy()[:128].astype(np.float32), _lab_june)
          and np.array_equal(_va_fix.isic_id.to_numpy(), _va.isic_id.to_numpy()),
          "random_state=CFG.seed non cambia un numero di ISIC24")
    _va_fix_dirty = _regen_validation(rng_offset=13, shuffle_rs=_SEED)[1]
    check("I. shuffle seminato -> regge il RNG globale sporco (era il difetto)",
          np.array_equal(_va_fix_dirty.isic_id.to_numpy(), _va.isic_id.to_numpy()),
          "13 draw consumati prima: ordine invariato")

    _snap_p = os.path.join(BASE, "notebooks", "modality_attendance", "validation_ids.SEED49.json")
    if os.path.exists(_snap_p):
        import json as _json
        _snap = _json.load(open(_snap_p))
        check("G. lo snapshot validation_ids combacia con la rigenerazione",
              sorted(_va.isic_id.tolist()) == _snap["isic_ids_sorted"],
              f"n snapshot={_snap['n']}")

# ======================================================================= 12 ===
section("12. class_weight × loss ridotta a scalare — la pesatura che sparisce in silenzio")
# `loss_regularization` compila una loss che fa `tf.reduce_mean(...)` DENTRO la funzione
# (reg_mode='per_batch', quello delle 25 run canoniche). Keras applica `class_weight` come
# sample_weight sul valore RESTITUITO: se quel valore e' gia' uno scalare, la pesatura
# degenera in un fattore moltiplicativo unico e non distingue piu' le classi.
# Le altre cinque famiglie compilano `bce` per-campione, dove invece agisce.
try:
    import tensorflow as _tf                                      # noqa: E402
    import keras as _keras                                        # noqa: E402

    _Xd = np.random.default_rng(0).standard_normal((96, 4)).astype("float32")
    _yd = (np.arange(96) % 3 == 0).astype("float32")               # prevalenza 1/3
    _CW = {0: 0.75, 1: 1.5}                                        # 'balanced' a prevalenza 1/3

    def _step(loss_fn, cw):
        _keras.utils.set_random_seed(7)
        m = _keras.Sequential([_keras.layers.Input((4,)),
                               _keras.layers.Dense(1, activation="sigmoid")])
        m.compile(optimizer=_keras.optimizers.SGD(0.5), loss=loss_fn)
        before = [w.numpy().copy() for w in m.trainable_weights]
        m.fit(_Xd, _yd, epochs=1, batch_size=96, verbose=0, class_weight=cw)
        return np.concatenate([(w.numpy() - b).ravel()
                               for w, b in zip(m.trainable_weights, before)])

    def _cos(loss_fn):
        a, b = _step(loss_fn, _CW), _step(loss_fn, None)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    _persample = _cos(lambda yt, yp: _keras.losses.binary_crossentropy(yt, yp, label_smoothing=0.02))
    _perbatch = _cos(lambda yt, yp: _tf.reduce_mean(
        _keras.losses.binary_crossentropy(yt, yp, label_smoothing=0.02)))

    check("A. loss per-campione (5 famiglie): class_weight AGISCE", _persample < 1 - 1e-6,
          f"cos(con CW, senza CW) = {_persample:.6f}")
    check("B. loss ridotta a scalare (loss_regularization): class_weight INERTE",
          _perbatch > 1 - 1e-6, f"cos(con CW, senza CW) = {_perbatch:.6f}")

    # ...e il guard che da oggi lo intercetta prima del fit
    from isic24.mod_attendance import assert_loss_is_per_sample          # noqa: E402

    def _compiled(loss_fn):
        m = _keras.Sequential([_keras.layers.Input((4,)),
                               _keras.layers.Dense(1, activation="sigmoid")])
        m.compile(optimizer="sgd", loss=loss_fn)
        return m

    _ps_fn = lambda yt, yp: _keras.losses.binary_crossentropy(yt, yp, label_smoothing=0.02)   # noqa: E731
    _pb_fn = lambda yt, yp: _tf.reduce_mean(_ps_fn(yt, yp))                                   # noqa: E731

    ok, msg = raises(lambda: assert_loss_is_per_sample(_compiled(_pb_fn), "loss_regularization"),
                     ValueError)
    check("C. il guard SOLLEVA sulla loss ridotta a scalare", ok, msg)
    ok_d, msg_d = raises(lambda: assert_loss_is_per_sample(_compiled(_ps_fn), "per_sample"),
                         ValueError)
    check("D. il guard tace sulla loss per campione", not ok_d, msg_d)
    ok_e, msg_e = raises(lambda: assert_loss_is_per_sample(
        _compiled(_keras.losses.BinaryCrossentropy(label_smoothing=0.02)), "bce"), ValueError)
    check("E. il guard tace su una keras.losses.Loss (pesa PRIMA di ridurre)", not ok_e, msg_e)
except Exception as _e:                                            # noqa: BLE001
    print(f"  (keras non disponibile: sezione saltata — {type(_e).__name__})")

# ======================================================================= 13 ===
section("13. decomposizione esatta del logit — le sei teste, e quella che non si decompone")
# La misura di attendance esatta dello studio poggia su una sola proprieta': la testa e'
# additiva, quindi z = b + c_img + c_tab senza residuo. Qui le sei topologie sono
# ricostruite in miniatura (backbone finto, stessa testa) e la ricostruzione deve tornare a
# meno del rumore float; una testa con un MLP dopo la fusione deve invece far sollevare.
try:
    import keras as _k                                             # noqa: E402
    from isic24 import xai_logit as xlog                           # noqa: E402
    from isic24.mod_attendance import (ConstantScale, FusionWithGamma,  # noqa: E402
                                       GateEntropyLoss, LearnableAlpha, ScalarMultiply)

    _IMG, _FEAT, _NB = (16, 16, 3), 12, 6
    _rb = np.random.default_rng(3)
    _xb = {"images": _rb.random((_NB, *_IMG)).astype("float32"),
           "features": _rb.normal(size=(_NB, _FEAT)).astype("float32")}

    def _toy(kind, alpha=0.3, gamma_img=1.0, img_dim=None):
        img_in = _k.Input(shape=_IMG, name="images")
        f_in = _k.Input(shape=(_FEAT,), name="features")
        x1 = _k.layers.GlobalAveragePooling2D()(_k.layers.Conv2D(20, 3, padding="same")(img_in))
        x1 = _k.layers.Dropout(0.2)(x1)
        if img_dim:
            x1 = _k.layers.Dense(img_dim, activation="selu", name="img_proj")(x1)
        x2 = _k.layers.Dense(9, activation="selu")(f_in)
        x2 = _k.layers.Dense(16, activation="selu")(x2)
        x2 = _k.layers.Dropout(0.1)(x2)

        if kind == "fixed_scaling":
            x1 = ConstantScale(1.0 - alpha, name="image_weighting")(x1)
            x2 = ConstantScale(alpha, name="tabular_weighting")(x2)
        elif kind == "learnable_scaling":
            a, b = LearnableAlpha(init_alpha=0.4, name="learnable_alpha")(x1)
            x1 = ScalarMultiply(name="image_weighting")([x1, b])
            x2 = ScalarMultiply(name="tabular_weighting")([x2, a])

        if kind == "controlled_fusion":
            fused = FusionWithGamma(gamma_img=gamma_img, gamma_tab=1.0,
                                    name="fusion_with_gamma")([x1, x2])
        elif kind == "adaptive_gating":
            x1 = _k.layers.Dense(24, activation="selu", name="img_emb")(x1)
            x2 = _k.layers.Dense(24, activation="selu", name="tab_emb")(x2)
            g = _k.layers.Dense(8, activation="relu")(_k.layers.Concatenate()([x1, x2]))
            g = GateEntropyLoss(weight=1e-4)(_k.layers.Dense(2, activation="softmax")(g))
            fused = _k.layers.Add(name="fused")([
                _k.layers.Multiply(name="weighted_img")([x1, _k.layers.Lambda(lambda w: w[:, 0:1])(g)]),
                _k.layers.Multiply(name="weighted_tab")([x2, _k.layers.Lambda(lambda w: w[:, 1:2])(g)])])
        else:
            fused = _k.layers.Concatenate(name="concat")([x1, x2])

        if kind == "nonadditive":
            fused = _k.layers.Dense(8, activation="selu")(fused)
        out = _k.layers.Dense(1, activation="sigmoid", name="output")(fused)
        return _k.Model({"images": img_in, "features": f_in}, out, name=kind)

    for _fam, _kind in [("vanilla", "Concatenate"), ("loss_regularization", "Concatenate"),
                        ("fixed_scaling", "Concatenate"), ("learnable_scaling", "Concatenate"),
                        ("controlled_fusion", "FusionWithGamma"), ("adaptive_gating", "Add")]:
        _m = _toy(_fam, img_dim=16 if _fam == "controlled_fusion" else None)
        _d, _meta = xlog.contributions(_m, _xb)
        check(f"B. {_fam}: la testa e' {_kind} e il logit si ricompone",
              _meta["head_kind"] == _kind and _meta["recon_max_abs_delta"] < xlog.RECON_TOL,
              f"{_meta['head_kind']}, delta={_meta['recon_max_abs_delta']:.1e}")

    _mn = _toy("nonadditive")
    _okn, _msgn = raises(lambda: xlog.contributions(_mn, _xb, check_order=False),
                         (AssertionError, ValueError))
    check("A. testa non additiva (MLP dopo la fusione): SOLLEVA", _okn, _msgn)

    # Caso C: i gamma di FusionWithGamma sono PESI non addestrabili, e get_config() riporta
    # il valore di costruzione. Lo sweep gamma post-hoc scrive il peso: chi legge il config
    # non se ne accorge e attribuisce al ramo un contributo che il modello non calcola.
    _mg = _toy("controlled_fusion", img_dim=16)
    _b0, _ = xlog.contributions(_mg, _xb, check_order=False)
    _mg.get_layer("fusion_with_gamma").gamma_img.assign(2.0)
    _b1, _meta1 = xlog.contributions(_mg, _xb, check_order=False)
    check("C. gamma letto dal PESO, non da get_config()",
          np.allclose(_b1["image_logit_raw"], 2 * _b0["image_logit_raw"], rtol=1e-4)
          and _meta1["recon_max_abs_delta"] < xlog.RECON_TOL,
          f"config={_mg.get_layer('fusion_with_gamma').get_config()['gamma_img']}, "
          f"peso=2.0, delta={_meta1['recon_max_abs_delta']:.1e}")

    _mv = _toy("vanilla")
    _okv, _msgv = raises(lambda: xlog.assert_branch_order(_mv, xlog.head_split(_mv), _xb),
                         AssertionError)
    check("D. ordine dei rami confermato su una topologia viva", not _okv, _msgv)
    _md = _toy("fixed_scaling", alpha=1.0)                       # ramo immagine scalato a 0
    _okd, _msgd = raises(lambda: xlog.assert_branch_order(_md, xlog.head_split(_md), _xb),
                         AssertionError)
    check("E. ramo immagine morto (alpha=1): SOLLEVA invece di misurare zero", _okd, _msgd)
except Exception as _e:                                            # noqa: BLE001
    print(f"  (keras non disponibile: sezione saltata — {type(_e).__name__}: {_e})")

# ======================================================================= 14 ===
section("14. ablazioni unimodali — la regola sta nel CODICE, non nel documento")
# Agli endpoint di uno sweep un ramo e' spento per costruzione: i loro valori di spiegazione
# sono algebra. La regola vive in `xs.ABLATION_SETTINGS` e deve valere in OGNI punto che
# aggrega per famiglia. Caso C: un `groupby` secco (il codice di prima) non se ne accorge.
_ab = pd.DataFrame({
    "impl_name": ["fixed_scaling"] * 4 + ["vanilla"],
    "setting_tag": ["a0p000", "a0p500", "a1p000", "a0p750", "baseline"],
    "seed": [47] * 5,
    "lore_coverage": [0.0, 0.5, 1.0, 0.8, 0.7],
})
check("A. gli endpoint sono riconosciuti come ablazioni",
      xs.is_ablation("fixed_scaling", "a0p000") and xs.is_ablation("fixed_scaling", "a1p000")
      and not xs.is_ablation("fixed_scaling", "a0p500"))
check("B. `branch_off` dice QUALE ramo spegne",
      xs.branch_off("fixed_scaling", "a0p000") == "tabular"
      and xs.branch_off("fixed_scaling", "a1p000") == "image"
      and xs.branch_off("vanilla", "baseline") is None)

_kept = xs.drop_ablations(_ab, "test", verbose=False)
check("C. drop_ablations toglie 2 righe su 5 e sposta la media",
      len(_kept) == 3
      and abs(_kept.lore_coverage.mean() - 2 / 3) < 1e-9
      and abs(_ab.lore_coverage.mean() - 0.60) < 1e-9,
      f"n={len(_kept)}, media {_kept.lore_coverage.mean():.3f} contro {_ab.lore_coverage.mean():.3f}")
check("D. il groupby secco (codice di prima) NON se ne accorge",
      abs(_ab.groupby("impl_name").lore_coverage.mean()["fixed_scaling"] - 0.575) < 1e-9,
      "0.575 contro 0.650 filtrando: e' il difetto che il caso C misura")
check("E. una tabella senza le colonne di identita' passa intatta",
      len(xs.drop_ablations(pd.DataFrame({"x": [1, 2]}), "test", verbose=False)) == 2)

# `pool_by_family` era l'ultimo punto che aggregava per famiglia SENZA il filtro: le tabelle
# per strato uscivano con 45 run di `fixed_scaling` invece di 35 (su ISIC: 1.195 invece di 1.052).
from isic24 import xai_faithfulness as _xf                    # noqa: E402
_st = pd.DataFrame({
    "impl_name": ["fixed_scaling"] * 3,
    "setting_tag": ["a0p000", "a1p000", "a0p500"],
    "run_id": ["r0", "r1", "r2"],
    "source": ["aligned_signed"] * 3, "mode": ["target"] * 3, "method": ["tabular"] * 3,
    "direction": ["del"] * 3, "q_max": [0.2] * 3, "q_width": [0.2] * 3, "stratum": ["all"] * 3,
    "n": [100, 100, 100], "gain_all_n": [1.0, 1.0, 0.4],
})
_pooled_f = _xf.pool_by_family(_st, verbose=False)
check("F. il pooling per famiglia esclude gli endpoint",
      len(_pooled_f) == 1 and _pooled_f.n_runs.iloc[0] == 1
      and abs(_pooled_f.gain.iloc[0] - 0.4) < 1e-9,
      f"n_runs={_pooled_f.n_runs.iloc[0]}, gain {_pooled_f.gain.iloc[0]:.3f}")
_old = (_st.n * _st.gain_all_n).sum() / _st.n.sum()
check("G. la media pesata secca (codice di prima) NON se ne accorge",
      abs(_old - 0.8) < 1e-9, "0.800 contro 0.400 filtrando")

# ======================================================================= 17 ===
section("17. indici — una sola funzione, e l'identita' per isic_id")
# `explained_indices` (posizione DENTRO il campione) e `val_index` (posizione in
# validation_df) hanno nomi simili e sono interi della stessa taglia: incrociarli non
# solleva, e ha gia' prodotto tre difetti. La lettura sta in un posto solo.
_p_idx = os.path.join(TMP, "idx.npz")
np.savez(_p_idx, explained_indices=np.arange(4), val_index=np.array([10, 20, 30, 40]))
with np.load(_p_idx) as _z:
    _i, _src = xs.read_explained_index(_z)
    check("A. con val_index si legge quello", _src == "val_index" and _i.tolist() == [10, 20, 30, 40])
np.savez(_p_idx, explained_indices=np.arange(4))
with np.load(_p_idx) as _z:
    _i, _src = xs.read_explained_index(_z)
    check("B. senza val_index si ricade sul prefisso, DICHIARANDOLO", _src == "prefix")
    _i2, _src2 = xs.read_explained_index(_z, allow_prefix=False)
    check("C. allow_prefix=False -> 'non lo so', non un indovinello",
          _i2 is None and _src2 == "unknown")

_vdf = pd.DataFrame({"isic_id": ["A", "B", "C", "B"]})     # validation ha righe duplicate
check("D. index_for_row_ids risolve le righe uniche",
      xs.index_for_row_ids(["C", "A"], _vdf).tolist() == [2, 0])
_okd, _msgd = raises(lambda: xs.index_for_row_ids(["B"], _vdf), KeyError)
check("E. un id duplicato SOLLEVA invece di prendere la prima occorrenza", _okd, _msgd)
_oke, _msge = raises(lambda: xs.index_for_row_ids(["Z"], _vdf), KeyError)
check("F. un id assente solleva", _oke, _msge)

# ======================================================================= 18 ===
section("18. identita' di riga — una pipeline, un dataset qualunque")
# Il codice non deve sapere come si chiama la colonna d'identita': `isic_id` qui,
# `image_id` su BRSET. Quello che finisce DENTRO gli artefatti e' `row_id`, sempre.
_ID_COL0, _REG0 = xs.ID_COL, xs.VALIDATION_ROW_IDS
_brset = pd.DataFrame({"image_id": ["img1", "img2", "img3", "img4"]})
try:
    xs.ID_COL = "image_id"
    check("A. la colonna d'identita' si dichiara, non si indovina",
          xs.row_ids(_brset).tolist() == ["img1", "img2", "img3", "img4"])
    _okg, _msgg = raises(lambda: xs.row_ids(_vdf), KeyError)
    check("B. colonna assente -> solleva dicendo cosa dichiarare", _okg, _msgg)
    xs.register_validation_ids(_brset)
    check("C. la chiave scritta nell'artefatto e' neutra, non il nome della colonna",
          list(xs.stamp_row_ids([2, 0])) == ["row_id"]
          and xs.stamp_row_ids([2, 0])["row_id"].tolist() == ["img3", "img1"])
    _okh, _msgh = raises(lambda: xs.stamp_row_ids([99]))
    check("D. un indice fuori dal validation registrato SOLLEVA", _okh, _msgh)
    check("E. senza validation registrato non si inventa un'identita'",
          xs.register_validation_ids(None) is None and xs.stamp_row_ids([0]) == {})

    # Il caso che le POSIZIONI non sanno vedere: stesso campione posizionale, fold
    # ricostruito in un altro ordine. E' il difetto per cui l'identita' esiste.
    _run_id = os.path.join(TMP, "id_run")
    _d = os.path.join(_run_id, "xai", "shap_tab", "epoch_003")
    os.makedirs(_d, exist_ok=True)
    np.savez(os.path.join(_d, "shap_arrays_epoch_003.npz"),
             shap_values=rng.normal(size=(2, len(NAMES))), explained_indices=np.arange(2),
             val_index=np.array([0, 2]), row_id=np.asarray(["img1", "img3"]))
    xs.register_validation_ids(_brset)
    check("F. stesse posizioni, stesse lesioni -> True",
          xs.sample_matches(_run_id, "shap_tab", [0, 2]) is True)
    xs.register_validation_ids(pd.DataFrame({"image_id": ["img4", "img2", "img1", "img3"]}))
    check("G. stesse posizioni, fold RIORDINATO -> False (le posizioni direbbero True)",
          xs.sample_matches(_run_id, "shap_tab", [0, 2]) is False)
    xs.register_validation_ids(None)
    check("H. codice vecchio (solo posizioni) sul caso G: resta muto",
          xs.sample_matches(_run_id, "shap_tab", [0, 2]) is True)

    # Gli artefatti gia' su disco portano `isic_id`: si leggono ancora, non si riscrivono.
    check("I. l'identita' vecchia (`isic_id`) resta leggibile",
          xs.read_explained_ids(pd.DataFrame({"isic_id": ["A", "B"]})).tolist() == ["A", "B"])
finally:
    xs.ID_COL, xs.VALIDATION_ROW_IDS = _ID_COL0, _REG0

# ======================================================================= 19 ===
section("19. percorsi — nessun nome di dataset dentro il codice")
from isic24 import paths as _pth                               # noqa: E402

_hist = "/home/claudio/isic24-dum/artifacts/enhanced/runs/vanilla/20260604-210304/seed_47/e20"
check("A. un run_dir di un'altra macchina si ri-radica su questo albero",
      _pth.run_dir_abs(_hist, "/local/art")
      == "/local/art/runs/vanilla/20260604-210304/seed_47/e20")
check("B. e ci si ri-radica anche dopo che la cartella e' stata spostata",
      _pth.run_dir_abs("/x/artifacts/isic24/enhanced/runs/vanilla/r1", "/local/art")
      == "/local/art/runs/vanilla/r1")
_oki, _msgi = raises(lambda: _pth.run_dir_abs("/x/y/z", "/local/art"), ValueError)
check("C. un percorso senza 'runs/' SOLLEVA invece di tornare quello della macchina "
      "d'origine", _oki, _msgi)
check("D. il codice vecchio sul caso C: torna un path inesistente che sembra giusto",
      "/x/y/z".split("artifacts/enhanced/")[-1] == "/x/y/z")
check("E. la radice di un dataset ha due segmenti",
      _pth.dataset_root("/r", "brset", "enhanced").as_posix().endswith("/r/brset/enhanced"))
check("F. un run_dir gia' dentro l'albero locale passa intatto",
      _pth.run_dir_abs("/local/art/runs/vanilla/r1", "/local/art")
      == "/local/art/runs/vanilla/r1")

# ======================================================================= 16 ===
section("16. griglia dell'immagine — un solo percorso, e nessuna precisione nominale")
# Il motore ritagliava il bordo (6x6) mentre la fedelta' ordinava le patch sulla mappa piena
# (8x8): 28 celle su 64 diverse, presentate come la stessa mappa. E FAITH_IMG_GRID era 16
# fisso, cioe' 256 patch ordinate a partire da 64 numeri.
from isic24 import xai_engines as xe                            # noqa: E402

check("A. il motore NON ritaglia piu' il bordo di default", xe.TRIM_CAM_BORDER is False)
check("B. FAITH_IMG_GRID segue la CAM invece di essere fisso", xf.FAITH_IMG_GRID is None,
      f"FAITH_IMG_GRID={xf.FAITH_IMG_GRID}")

_cams = rng.random((3, 256, 256))
for _g in (8, 16):
    _o = xf._patch_saliency_order(_cams, _g)
    check(f"C{_g}. l'ordinamento su griglia {_g}x{_g} copre tutte le patch una volta",
          _o.shape == (3, _g * _g) and all(sorted(r.tolist()) == list(range(_g * _g)) for r in _o))

# Il guard che rifiuta una griglia piu' fine della CAM: si esercita sulla sola condizione,
# senza modello, perche' e' quella la regola.
def _grid_guard(grid, cam_grid):
    if grid > cam_grid:
        raise ValueError("[FAITH IMG GRID] nominal precision")
    return grid


ok, msg = raises(lambda: _grid_guard(16, 8), ValueError)
check("D. grid piu' fine della CAM -> solleva", ok, msg)
ok_e, _ = raises(lambda: _grid_guard(8, 8), ValueError)
check("E. grid uguale alla CAM -> passa", not ok_e)

# ======================================================================= 15 ===
section("15. pre-flight della sonda shuffle — i nomi che usa sono quelli che dichiara")
# La sonda vive in coda al notebook di training e riusa builder e runner, che stanno sopra
# @@STOP@@. Il suo `_NEEDED` dice quale cella lanciare per ciascun nome: se qualcuno tocca la
# cella e introduce un nome non dichiarato, l'errore torna a essere un NameError opaco.
import ast as _ast                                              # noqa: E402
import builtins as _bi                                          # noqa: E402
import json as _js                                              # noqa: E402

_nbp = os.path.join(BASE, "notebooks", "modality_attendance",
                    "isic24_modality_attendance_train.ipynb")
_cells = [''.join(c["source"]) for c in _js.load(open(_nbp))["cells"]
          if c["cell_type"] == "code"]
_probe = [c for c in _cells if c.startswith("# @@PROBE_SHUFFLE_FIXED@@")]
check("A. la cella @@PROBE_SHUFFLE_FIXED@@ esiste ed e' unica", len(_probe) == 1, f"n={len(_probe)}")

if len(_probe) == 1:
    _t = _ast.parse(_probe[0])
    _load, _store = set(), set()
    for _n in _ast.walk(_t):
        if isinstance(_n, _ast.Name):
            (_load if isinstance(_n.ctx, _ast.Load) else _store).add(_n.id)
        elif isinstance(_n, (_ast.FunctionDef, _ast.ClassDef)):
            _store.add(_n.name)
        elif isinstance(_n, _ast.Lambda):
            _store.update(a.arg for a in _n.args.args)
    _free = _load - _store - set(dir(_bi))
    _needed = _ast.literal_eval(_probe[0].split("_NEEDED = ")[1].split("\n_missing")[0])
    check("B. ogni nome libero e' dichiarato in _NEEDED", not (_free - set(_needed)),
          f"scoperti: {sorted(_free - set(_needed))}" if _free - set(_needed) else f"{len(_free)} nomi")

    # ...e ogni marcatore citato deve esistere davvero nel notebook
    _text = "\n".join(_cells)
    _absent = sorted(m for m in set(_needed.values()) if m not in _text)
    check("C. i marcatori citati dal pre-flight esistono nel notebook", not _absent, str(_absent))

# =============================================================================
shutil.rmtree(TMP, ignore_errors=True)
n_ok = sum(1 for _, o, _ in RESULTS if o)
print(f"\n{'=' * 74}")
print(f"TOTALE: {n_ok}/{len(RESULTS)} PASS")
for name, o, det in RESULTS:
    if not o:
        print(f"  ✗ {name}  {det}")
sys.exit(0 if n_ok == len(RESULTS) else 1)
