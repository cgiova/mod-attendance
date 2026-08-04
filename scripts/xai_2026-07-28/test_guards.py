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

# =============================================================================
shutil.rmtree(TMP, ignore_errors=True)
n_ok = sum(1 for _, o, _ in RESULTS if o)
print(f"\n{'=' * 74}")
print(f"TOTALE: {n_ok}/{len(RESULTS)} PASS")
for name, o, det in RESULTS:
    if not o:
        print(f"  ✗ {name}  {det}")
sys.exit(0 if n_ok == len(RESULTS) else 1)
