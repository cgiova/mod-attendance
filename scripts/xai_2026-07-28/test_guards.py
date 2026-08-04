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

ART = os.path.join(BASE, "artifacts", "enhanced")
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

# =============================================================================
shutil.rmtree(TMP, ignore_errors=True)
n_ok = sum(1 for _, o, _ in RESULTS if o)
print(f"\n{'=' * 74}")
print(f"TOTALE: {n_ok}/{len(RESULTS)} PASS")
for name, o, det in RESULTS:
    if not o:
        print(f"  ✗ {name}  {det}")
sys.exit(0 if n_ok == len(RESULTS) else 1)
