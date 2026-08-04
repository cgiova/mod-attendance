"""Metriche di FORMA delle spiegazioni — un complemento, mai il focus.

Che cosa misurano, e perche' la distinzione non e' pedanteria
------------------------------------------------------------
La fedelta' (`xai_faithfulness`) misura se la spiegazione dice il vero sul modello.
L'attendance (`lore_coverage`) misura quanto il modello guarda una modalita'.  Queste qui
misurano un'altra cosa ancora: quanto la spiegazione e' **leggibile** — concentrata su poche
feature, corta, stabile.

⚠️ **LEGGIBILITA' E ATTENDANCE SONO ANTI-CORRELATE SU QUESTO DISEGNO.**  Misurato il
2026-08-01 su `fixed_scaling` seed 47, dove `a0p000` ha il ramo tabellare **spento per
costruzione**:

    setting   massa |SHAP|   compactness SHAP   rule_len   compactness LORE
    a0p000        0.0000            1.000          0.00         1.0079  (fuori range!)
    a0p500       22.9674            0.272          1.38         0.9970
    a1p000      203.8043            0.282          9.47         0.9328

Il modello che **non spiega niente** prende il punteggio massimo su entrambe.  Meno il
modello usa una modalita', piu' corta e' la sua spiegazione: e' meccanico, non un caso.

Da qui le due regole di questo modulo, che sono strutturali e non commenti:

1. **Quando il ramo e' spento, le metriche di forma sono `NaN`, non un punteggio alto.**
   Una nota a pie' di pagina in un CSV non sopravvive alla lettura successiva; un `NaN` si'.
   La soglia (`MIN_SHAP_MASS`) separa nettamente: `a0p000` ha massa **esattamente 0.0**, la
   configurazione successiva 15.5.
2. **Ogni riga porta con se' `shap_mass` / `rule_len`**, cosi' chi legge vede da cosa nasce
   il punteggio senza tornare agli artefatti.

Non si aggregano in uno scalare unico con la fedelta' (era la proposta `Q_XAI` del foglio
esterno del 01/08): sommare leggibilita' e correttezza nasconde proprio il trade-off che lo
studio vuole mostrare.  Vanno riportate **accanto**, e messe in grafico **contro**
`lore_coverage`, mai dentro.

Zero GPU: legge solo gli npz SHAP e i CSV LORE gia' su disco, e scrive CSV in
`runs/_faithfulness/` accanto agli altri riassunti.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

# Sotto questa massa |SHAP| totale il ramo tabellare e' spento e la "compattezza" e' un
# artefatto della divisione per zero. `a0p000` misura 0.0 esatto, il vicino 15.5: qualunque
# soglia in mezzo separa: 1.0 e' prudente e non arbitraria al margine.
MIN_SHAP_MASS = 1.0
# T6 sotto questo numero di campioni in comune non e' una misura, e' rumore.
MIN_OVERLAP = 20
N_PERM_AGREEMENT = 200         # permutazioni del null dell'accordo SHAP<->LORE
TOP_K = 10                      # top-k feature RAW, non encoded: vedi `raw_top_k`


# --------------------------------------------------------------------------------------
# encoded -> raw
#
# `xai_engines` ha gia' questa mappatura, ma in DUE direzioni e con una discrepanza:
# `encoded_indices_for_raw` accetta `{feat}_bin_{b}` per qualunque feature, mentre
# `aggregate_raw_importance` ha `age_approx_bin_` scritto a mano.  Su ISIC24 coincidono
# (l'unica feature discretizzata e' `age_approx`), ma il notebook costruisce i nomi con
# `f"{name}_bin_{b}"` per OGNI `Discretization` — quindi su un dataset con due feature
# discretizzate `aggregate_raw_importance` mapperebbe `foo_bin_0` sul raw `"foo_bin_0"`
# invece che su `"foo"`, in silenzio.  Qui la mappatura sta in UN posto solo ed e' generica.
# --------------------------------------------------------------------------------------
_BIN_RE = re.compile(r"^(?P<raw>.+)_bin_\d+$")


def raw_name_for(encoded_name):
    """Nome della feature RAW da cui viene questa dimensione encoded."""
    if "__" in encoded_name:                      # categorica one-hot: `feat__token`
        return encoded_name.split("__", 1)[0]
    m = _BIN_RE.match(encoded_name)               # discretizzata: `feat_bin_3`
    if m:
        return m.group("raw")
    return encoded_name                           # numerica: la dimensione E' la feature


def raw_groups(encoded_feature_names):
    """{feature raw: indici encoded che le appartengono}."""
    out = {}
    for j, name in enumerate(encoded_feature_names):
        out.setdefault(raw_name_for(name), []).append(j)
    return {k: np.asarray(v, int) for k, v in out.items()}


def raw_top_k(phi_row, groups, k=TOP_K):
    """Le k feature RAW piu' importanti per un campione.

    RAW e non encoded di proposito: con l'one-hot, le prime 10 dimensioni encoded possono
    essere 10 livelli della STESSA feature categorica — un top-10 encoded direbbe "la
    spiegazione usa 10 cose" dove ne usa una.  L'importanza di una feature raw e' la somma
    dei |valori| delle sue dimensioni, la stessa convenzione di `aggregate_raw_importance`.
    """
    imp = {r: float(np.abs(phi_row[idx]).sum()) for r, idx in groups.items()}
    order = sorted(imp, key=lambda r: -imp[r])
    return {r for r in order[:k] if imp[r] > 0}


# --------------------------------------------------------------------------------------
# T1 + T3 — forma della spiegazione, per run
# --------------------------------------------------------------------------------------
def _shap_npz(run_dir, prefer_subset=True):
    """L'npz SHAP da leggere: prima quello sul campione adottato, poi il canonico.

    `xai/shap_tab/` sono gli array calcolati sul campione a lesioni distinte, `xai/shap/`
    quelli canonici sul prefisso.  L'ordine conta per l'accordo SHAP<->LORE: e' solo con
    SHAP e LORE sullo STESSO campione che la sovrapposizione passa da 14/48 a 48/48.
    """
    for sub in (("shap_tab", "shap") if prefer_subset else ("shap",)):
        hits = sorted(glob.glob(os.path.join(run_dir, "xai", sub, "epoch_*",
                                             "shap_arrays_epoch_*.npz")))
        if hits:
            return hits[-1]
    return None


def shap_shape_from_values(shap_values, explained_indices=None):
    """T1 — compactness (entropia) e k90 sparsity, dalle ATTRIBUZIONI IN MEMORIA.

    Questa e' la funzione che chiama `ShapCallback` mentre calcola: le metriche escono
    insieme a tutto il resto, nella stessa `shap_summary.csv`, senza nessuna cella da
    lanciare dopo.  `shap_shape_metrics` qui sotto e' solo la versione che rilegge da disco,
    per gli artefatti prodotti prima che questo aggancio esistesse.

    Entrambe per campione, poi mediate.  I campioni a massa nulla sono ESCLUSI dalla media
    e contati in `frac_zero_shap`: su un campione a massa zero `p_j = 0` ovunque, l'entropia
    e' 0 e la compattezza risulterebbe 1 — il punteggio massimo per "non ho spiegato nulla".
    """
    phi = np.abs(np.asarray(shap_values, float))
    if phi.ndim == 3:                                  # (n, 1, d) o (n, d, 1)
        phi = phi[:, 0, :] if phi.shape[1] == 1 else phi[..., 0]
    idx = (np.asarray(explained_indices, int).ravel()
           if explained_indices is not None else np.arange(phi.shape[0]))
    d = phi.shape[1]
    mass = phi.sum(axis=1)
    live = mass > 0
    rec = {"n_shap": int(phi.shape[0]), "d_encoded": int(d),
           "shap_mass": float(mass.sum()), "shap_mass_per_sample": float(mass.mean()),
           "frac_zero_shap": float((~live).mean()),
           "shap_sample_hash": int(idx.sum())}      # per verificare che le run condividano il campione
    if not live.any() or rec["shap_mass"] < MIN_SHAP_MASS:
        # ramo tabellare spento: non c'e' nessuna forma da misurare
        rec.update(shap_compactness=np.nan, shap_k90=np.nan, shap_k90_sparsity=np.nan,
                   branch_off=True)
        return rec
    P = phi[live] / mass[live, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    H = -(P * logP).sum(axis=1) / np.log(d)
    srt = -np.sort(-phi[live], axis=1)
    cum = np.cumsum(srt, axis=1) / srt.sum(axis=1, keepdims=True)
    k90 = (cum < 0.90).sum(axis=1) + 1               # primo k che raggiunge il 90%
    rec.update(shap_compactness=float(np.mean(1.0 - H)),
               shap_k90=float(np.mean(k90)),
               shap_k90_sparsity=float(np.mean(1.0 - (k90 - 1) / (d - 1))),
               branch_off=False)
    return rec


def shap_shape_metrics(run_dir, encoded_feature_names=None, k=TOP_K):
    """Come sopra, ma rileggendo l'npz gia' scritto — per gli artefatti storici."""
    p = _shap_npz(run_dir)
    if p is None:
        return None
    with np.load(p) as z:
        return shap_shape_from_values(z["shap_values"], z["explained_indices"])


def _lore_persample(run_dir):
    from . import xai_subset as xs
    hits = sorted(glob.glob(os.path.join(xs.out_dir_for(run_dir, "lore_sa"), "epoch*",
                                         "lore_sa_persample_*.csv")))
    if not hits:      # artefatti storici (fino al 31/07), un'altra cartella
        hits = sorted(glob.glob(os.path.join(run_dir, "xai", "lore", "epoch*", "tabular_sa",
                                             "lore_sa_persample_*.csv")))
    return pd.read_csv(hits[-1]) if hits else None


def lore_shape_from_df(d, d_raw=None):
    """T3 — rule compactness, con la formula RIPARATA, dal DataFrame IN MEMORIA.

    La chiama `LoreSACallback` appena finito di spiegare: le colonne finiscono in
    `lore_sa_per_epoch.csv` insieme a `lore_coverage` e alle altre, senza cella aggiuntiva.

    Il foglio propone `1 - (|R|-1)/(d-1)`, che con una regola vuota (`|R| = 0`, che LORE
    scrive come "IF <nessuna premessa>") da' **1.0079**: fuori dall'intervallo [0,1] e
    massima.  Una regola vuota non e' una spiegazione compatta, e' l'assenza di spiegazione.
    Qui le regole vuote sono escluse dalla media e contate in `frac_empty_rules`; se sono
    tutte vuote il risultato e' `NaN`.
    """
    if d is None or not len(d):
        return None
    rl = d["rule_len"].astype(float)
    n_raw = int(d_raw) if d_raw else int(max(rl.max(), 2))
    rec = {"n_lore": int(len(d)), "lore_rule_len": float(rl.mean()),
           "frac_empty_rules": float((rl < 1).mean()),
           "lore_pos_rate": float(d["true_label"].mean()) if "true_label" in d else np.nan,
           # IDENTITA' del campione, non solo la sua dimensione. `n` non basta: il 01/08
           # `vanilla` si e' ritrovato tre run tutte a n=48 ma su DUE campioni diversi —
           # seed 47 dal giro del 31/07 (prefisso dei 144, pos_rate 0.729), seed 53 e 71
           # col campione bilanciato (0.500). Mediarle da' un `lore_coverage` di famiglia
           # che non appartiene a nessuna popolazione. Le run nuove hanno `val_index` e la
           # chiave e' esatta; per quelle vecchie (n, prevalenza) le separa comunque.
           "lore_sample_key": (f"h{abs(hash(tuple(d['val_index'].tolist()))) % 10**8:08d}"
                               if "val_index" in d.columns else
                               f"n{len(d)}-p{float(d['true_label'].mean()):.3f}"
                               if "true_label" in d.columns else f"n{len(d)}")}
    live = rl >= 1
    if not live.any():
        rec.update(lore_rule_compactness=np.nan, branch_off_lore=True)
        return rec
    comp = 1.0 - (rl[live] - 1) / max(n_raw - 1, 1)
    rec.update(lore_rule_compactness=float(np.clip(comp, 0.0, 1.0).mean()),
               branch_off_lore=False)
    return rec


def lore_shape_metrics(run_dir, d_raw=None):
    """Come sopra, ma rileggendo il CSV per campione — per gli artefatti storici."""
    return lore_shape_from_df(_lore_persample(run_dir), d_raw)


# --------------------------------------------------------------------------------------
# T6 — accordo SHAP <-> LORE
# --------------------------------------------------------------------------------------
# Il nome puo' cominciare con una CIFRA: su ISIC24 ci sono `3d_lesion_orientation` e
# `3d_position_distance`.  Con un `[A-Za-z_]` iniziale — l'identificatore Python — quelle
# 14 condizioni sparivano in silenzio e l'accordo con SHAP risultava piu' basso del vero.
_COND_RE = re.compile(r"^\s*([A-Za-z0-9_][A-Za-z0-9_.]*)\s*(<=|>=|!=|<|>|=)")


def rule_features(rule_text):
    """Le feature RAW citate in una regola LORE.  Vuoto per "IF <nessuna premessa>".

    E' un INSIEME: una regola puo' vincolare due volte la stessa feature (`x > a AND
    x <= b`), quindi `len(rule_features) <= rule_len` — misurato, 33/48 coincidono e le
    altre hanno intervalli bilateri.  E' la semantica giusta per una Jaccard fra insiemi di
    feature, che e' quello che chiedono sia il §2.4 sia il §3.5.
    """
    if not isinstance(rule_text, str) or "nessuna premessa" in rule_text:
        return set()
    body = rule_text.split("THEN")[0].replace("IF", "", 1)
    out = set()
    for cond in body.split(" AND "):
        m = _COND_RE.match(cond.strip())
        if m:
            out.add(m.group(1))
    return out


def shap_lore_agreement(run_dir, encoded_feature_names, k=TOP_K, lore_idx=None,
                        n_perm=N_PERM_AGREEMENT, seed=0):
    """T6 — Jaccard fra le top-k feature RAW di SHAP e le feature nella regola LORE.

    ⚠️ DUE OSTACOLI, entrambi misurati, entrambi riportati come colonne invece che nascosti.

    1. **Spazi diversi.** SHAP vive su 127 dimensioni ENCODED, le regole LORE su feature
       RAW.  L'aggregazione encoded->raw e' `raw_top_k`, non una Jaccard diretta.
    2. **I due explainer indicizzano in spazi diversi, e i due nomi si somigliano.**
       `explained_indices` degli npz SHAP e' la posizione DENTRO il campione (`0..143`);
       `val_index` e' la posizione in `validation_df` (`1..1059`), ed e' quella che scrive
       LORE.  Appaiare l'uno con l'altro non solleva: incrocia i due spazi e restituisce
       le collisioni fortuite — **15/48** sulle run del 03/08, sotto `MIN_OVERLAP`, cioe'
       un `NaN` che sembra un campione sbagliato e invece era un confronto sbagliato.
       Con `val_index` su entrambi i lati la sovrapposizione e' **48/48**.
       Gli npz canonici in `xai/shap/` sono anteriori a `val_index`: li' il campione E' il
       prefisso, quindi `explained_indices` vive gia' in spazio `validation_df` ed e' il
       fallback corretto.  Sotto `MIN_OVERLAP` il risultato resta `NaN`.
    """
    p = _shap_npz(run_dir)
    lore = _lore_persample(run_dir)
    if p is None or lore is None or "rule_text" not in lore:
        return None
    with np.load(p) as z:
        phi = np.abs(np.asarray(z["shap_values"], float))
        # `val_index` per le run nuove, `explained_indices` per i canonici sul prefisso:
        # LORE indicizza su `validation_df`, quindi SHAP deve farlo anche lui.
        key = "val_index" if "val_index" in z.files else "explained_indices"
        shap_idx = np.asarray(z[key], int).ravel()

    # ⚠️ La colonna `i` del CSV LORE e' la posizione DENTRO il campione LORE, NON in
    # `validation_df`.  Le due cose coincidono solo se il campione LORE era il prefisso —
    # e dal 01/08 non lo e' piu'.  Quindi gli indici veri o li passa il chiamante, o li
    # legge la colonna `val_index` che `LoreSACallback` scrive dalle run nuove; se non ci
    # sono, questa funzione NON indovina: restituisce NaN e dice perche'.  Assumere il
    # prefisso qui darebbe un accordo su 48 campioni sbagliati con numeri plausibili.
    if "val_index" in lore.columns:                      # run nuove: se lo porta dietro
        lore_idx = lore["val_index"].to_numpy()
    elif isinstance(lore_idx, dict):                     # {n_campioni: indici}
        lore_idx = lore_idx.get(len(lore))
    elif lore_idx is not None and len(lore_idx) != len(lore):
        # Un solo array applicato a run con campioni DIVERSI appaierebbe le righe sbagliate
        # in silenzio: le 6 run del 31/07 hanno n=48, quelle storiche n=32 sul prefisso
        # grezzo. Se le lunghezze non tornano, non si indovina.
        return {"n_overlap": np.nan, "agreement_shap_lore": np.nan,
                "agreement_note": f"lore_idx ha {len(lore_idx)} indici ma la run ne ha "
                                  f"spiegati {len(lore)}: campioni diversi, non appaiabili"}
    if lore_idx is None:
        return {"n_overlap": np.nan, "agreement_shap_lore": np.nan,
                "agreement_note": "indici di validation ignoti: passare lore_idx="
                                  "<indici posizionali> o {n: indici}"}
    lore_idx = np.asarray(lore_idx, int)[:len(lore)]

    pos_in_shap = {int(v): j for j, v in enumerate(shap_idx)}
    pairs = [(pos_in_shap[int(v)], r) for r, v in enumerate(lore_idx)
             if int(v) in pos_in_shap]
    rec = {"n_overlap": len(pairs)}
    if len(pairs) < MIN_OVERLAP:
        rec["agreement_shap_lore"] = np.nan
        rec["agreement_note"] = (f"solo {len(pairs)} campioni in comune fra SHAP e LORE "
                                 f"(minimo {MIN_OVERLAP})")
        return rec
    g = raw_groups(encoded_feature_names)
    A_sets, B_sets = [], []
    for si, li in pairs:
        A = raw_top_k(phi[si], g, k=k)
        B = rule_features(lore["rule_text"].iloc[li])
        if not A and not B:
            continue
        A_sets.append(A)
        B_sets.append(B)

    def _j(a, b):
        return len(a & b) / max(len(a | b), 1)

    rec["agreement_shap_lore"] = (float(np.mean([_j(a, b) for a, b in zip(A_sets, B_sets)]))
                                  if A_sets else np.nan)
    rec["agreement_n_used"] = len(A_sets)
    if A_sets:
        rec.update(_agreement_null(A_sets, B_sets, sorted(g), n_perm=n_perm, seed=seed))
    return rec


def _agreement_null(A_sets, B_sets, raw_features, n_perm=N_PERM_AGREEMENT, seed=0):
    """Il null dell'accordo SHAP<->LORE, su DUE scale. Senza, il numero non si legge.

    Una Jaccard fra un top-k e un insieme di regole cresce **meccanicamente** con la
    lunghezza della regola, e `rule_len` e' esattamente cio' che ordina le famiglie: il
    valore grezzo produce quindi un ordinamento che sembra un risultato e non lo e'.
    Misurato il 2026-08-04 su 110 run: rho con `mean_rule_feats` +0.937 sull'osservato e
    **+0.945 sul null**, eccesso -0.008.  Percio' queste colonne escono INSIEME
    all'accordo, non "dopo, se serve".

    * `agreement_null_*` — permutazione DENTRO la run: la riga SHAP *i* si appaia con una
      regola rimescolata.  Conserva le dimensioni degli insiemi da entrambi i lati e
      distrugge solo la corrispondenza **per campione**.
    * `agreement_rho_global` — e quello da solo non basta: conservando le marginali, un
      null-level e' compatibile con «concordano su quali feature contano globalmente, e non
      su quali contino per QUESTA lesione».  Si discrimina correlando le due frequenze per
      feature raw sull'intera run.  Il suo null rimescola l'identita' della feature.
    """
    from scipy import stats

    rng = np.random.default_rng(seed)

    def _j(a, b):
        return len(a & b) / max(len(a | b), 1)

    obs = float(np.mean([_j(a, b) for a, b in zip(A_sets, B_sets)]))
    null = np.array([np.mean([_j(A_sets[i], B_sets[j]) for i, j in
                              enumerate(rng.permutation(len(B_sets)))])
                     for _ in range(n_perm)])
    sd = float(null.std())
    out = {
        "agreement_null_mean": float(null.mean()),
        "agreement_null_sd": sd,
        "agreement_z": float((obs - null.mean()) / sd) if sd > 0 else np.nan,
        "agreement_p_perm": float((1 + int((null >= obs).sum())) / (1 + len(null))),
        "agreement_mean_rule_feats": float(np.mean([len(b) for b in B_sets])),
    }
    fa = np.array([sum(f in s for s in A_sets) for f in raw_features], float)
    fb = np.array([sum(f in s for s in B_sets) for f in raw_features], float)
    if fa.std() > 0 and fb.std() > 0:
        out["agreement_rho_global"] = float(stats.spearmanr(fa, fb)[0])
        gn = [stats.spearmanr(fa, rng.permutation(fb))[0] for _ in range(max(n_perm // 4, 10))]
        out["agreement_rho_global_null"] = float(np.median(gn))
        ta = {raw_features[i] for i in np.argsort(-fa)[:TOP_K]}
        tb = {raw_features[i] for i in np.argsort(-fb)[:TOP_K]}
        out["agreement_top_global_jaccard"] = len(ta & tb) / max(len(ta | tb), 1)
    return out


# --------------------------------------------------------------------------------------
# T2 — stabilita' fra configurazioni
# --------------------------------------------------------------------------------------
def setting_value(impl_name, setting_tag):
    """Il valore NUMERICO dell'asse di parametro di una famiglia, o None se non ne ha uno.

    Tre famiglie su sei hanno un asse (`fixed_scaling` 9 punti, `learnable_scaling` 5,
    `loss_regularization` 5).  `vanilla`, `controlled_fusion` e `adaptive_gating` hanno un
    solo setting: per loro "configurazioni vicine" NON e' definito, e la metrica non si
    calcola invece di calcolarsi su una coppia inventata.
    """
    t = str(setting_tag)
    m = re.search(r"a(\d)p(\d{3})$", t)                    # a0p500 / init_a0p500
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"lam(\d)e([+-]\d\d)$", t)               # lam1e-03 / lam0e+00
    if m:
        return float(f"{m.group(1)}e{m.group(2)}")
    return None


def _run_top_k(run_dir, groups, k=TOP_K):
    p = _shap_npz(run_dir)
    if p is None:
        return None, None
    with np.load(p) as z:
        phi = np.abs(np.asarray(z["shap_values"], float))
        # `val_index` e non `explained_indices`: il secondo vale `0..n-1` per QUALUNQUE
        # campione, quindi il guard sull'identita' in `stability_table` non potrebbe mai
        # scattare — due campioni diversi darebbero indici identici. Con `val_index` il
        # confronto e' sulle lesioni vere. Sui canonici (prefisso) i due coincidono.
        key = "val_index" if "val_index" in z.files else "explained_indices"
        idx = np.asarray(z[key], int).ravel()
    return [raw_top_k(r, groups, k) for r in phi], idx


def _jaccard_runs(a, b):
    js = [len(x & y) / len(x | y) for x, y in zip(a, b) if (x | y)]
    return float(np.mean(js)) if js else np.nan


def stability_table(master, art_path, encoded_feature_names, k=TOP_K):
    """T2 — Stability@k, con il suo NULL accanto.

    Due tipi di coppia, sugli STESSI campioni:

    * ``adjacent_param``  stesso seed, setting consecutivi sull'asse -> l'effetto del peso;
    * ``same_param_seed`` stesso setting, seed diversi -> **il rumore di seed**.

    Senza il secondo, "Stability@5 = 0.6 fra alpha=0.5 e alpha=0.625" non e' interpretabile:
    potrebbe essere tutto rumore di inizializzazione.  Il numero che conta e' la differenza
    fra i due, ed e' per questo che il null si calcola qui e non "dopo, se serve".

    NB `lam0e+00` resta nell'asse ordinato per VALORE (0 < 1e-5): non si prende nessun
    logaritmo, ma la spaziatura fra coppie e' molto diversa e la colonna `param_a`/`param_b`
    la mette in chiaro invece di lasciarla dedurre.
    """
    g = raw_groups(encoded_feature_names)
    m = master.copy()
    m["param"] = [setting_value(i, t) for i, t in zip(m.impl_name, m.setting_tag)]
    m["run_dir_abs"] = [os.path.join(art_path, str(d).split("artifacts/enhanced/")[-1])
                        for d in m.run_dir]

    cache, ref_idx = {}, None
    def topk(run_dir):
        nonlocal ref_idx
        if run_dir not in cache:
            tk, idx = _run_top_k(run_dir, g, k)
            if tk is not None:
                if ref_idx is None:
                    ref_idx = idx
                elif not np.array_equal(ref_idx, idx):
                    raise RuntimeError(
                        "[STABILITY] due run hanno SHAP su campioni diversi: la Jaccard "
                        "confronterebbe spiegazioni di lesioni diverse.")
            cache[run_dir] = tk
        return cache[run_dir]

    rows = []
    for impl, grp in m.groupby("impl_name"):
        # (a) coppie adiacenti sull'asse, a parita' di seed
        if grp["param"].notna().all() and grp.setting_tag.nunique() > 1:
            for seed, gs in grp.groupby("seed"):
                gs = gs.sort_values("param")
                for (_, ra), (_, rb) in zip(gs.iloc[:-1].iterrows(), gs.iloc[1:].iterrows()):
                    A, B = topk(ra.run_dir_abs), topk(rb.run_dir_abs)
                    if A is None or B is None:
                        continue
                    rows.append(dict(impl_name=impl, pair_kind="adjacent_param", seed=seed,
                                     tag_a=ra.setting_tag, tag_b=rb.setting_tag,
                                     param_a=ra.param, param_b=rb.param,
                                     stability_at_k=_jaccard_runs(A, B), k=k))
        # (b) NULL: stesso setting, seed diversi
        for tag, gt in grp.groupby("setting_tag"):
            gt = gt.sort_values("seed")
            for (_, ra), (_, rb) in zip(gt.iloc[:-1].iterrows(), gt.iloc[1:].iterrows()):
                A, B = topk(ra.run_dir_abs), topk(rb.run_dir_abs)
                if A is None or B is None:
                    continue
                rows.append(dict(impl_name=impl, pair_kind="same_param_seed",
                                 seed=f"{ra.seed}->{rb.seed}", tag_a=tag, tag_b=tag,
                                 param_a=ra.param, param_b=rb.param,
                                 stability_at_k=_jaccard_runs(A, B), k=k))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Raccolta + riassunto per famiglia
# --------------------------------------------------------------------------------------
def collect(master, art_path, encoded_feature_names=None, k=TOP_K, d_raw=None,
            lore_idx=None):
    """Una riga per run con tutte le metriche di forma disponibili.

    `d_raw` — numero di feature RAW, il denominatore di `lore_rule_compactness`.  Dedotto da
    `encoded_feature_names` se non passato: dedurlo dalla lunghezza massima delle regole,
    come faceva la prima stesura, rende il denominatore diverso da run a run e i valori non
    confrontabili fra famiglie.
    `lore_idx` — indici POSIZIONALI in validation_df dei campioni spiegati da LORE, per T6.
    """
    if d_raw is None and encoded_feature_names is not None:
        d_raw = len(raw_groups(encoded_feature_names))
    rows = []
    for _, r in master.iterrows():
        rd = os.path.join(art_path, str(r.run_dir).split("artifacts/enhanced/")[-1])
        rec = dict(impl_name=r.impl_name, setting_tag=r.setting_tag, seed=r.seed,
                   run_id=r.run_id, param=setting_value(r.impl_name, r.setting_tag))
        for f in (shap_shape_metrics(rd, encoded_feature_names, k),
                  lore_shape_metrics(rd, d_raw)):
            if f:
                rec.update(f)
        if encoded_feature_names is not None:
            a = shap_lore_agreement(rd, encoded_feature_names, k, lore_idx=lore_idx)
            if a:
                rec.update(a)
        rows.append(rec)
    return pd.DataFrame(rows)


def summary_by_family(art_path, shape_df=None, stab_df=None, central=None, out_dir=None,
                      verbose=True):
    """La tabella UNICA per famiglia: fedelta' + attendance + forma, affiancate.

    L'ordine delle colonne e' la tesi: prima le metriche PRIMARIE (fedelta', attendance),
    poi quelle di forma, marcate `shape__`.  Non c'e' nessuno scalare aggregato che le
    fonda: sommare leggibilita' e correttezza nasconde il trade-off invece di mostrarlo.
    """
    from . import xai_subset as xs
    out_dir = out_dir or os.path.join(art_path, "runs", xs.AGGREGATE_DIRNAME)
    parts = []

    f = os.path.join(out_dir, "faithfulness_aligned_signed_n144.csv")
    if os.path.exists(f):
        d = pd.read_csv(f)
        if central:
            d = d[[central.get(i) == t for i, t in zip(d.impl_name, d.setting_tag)]]
        parts.append(d.groupby("impl_name").agg(
            n_runs=("run_id", "nunique"),
            faith_target_q20=("target_del_q20norm", "mean"),
            faith_posev_q20=("posev_del_q20norm", "mean")))

    l = os.path.join(out_dir, "lore_sa_quality_n48.csv")
    if os.path.exists(l):
        d = pd.read_csv(l)
        parts.append(d.groupby("impl_name").agg(
            lore_coverage=("lore_coverage_5050", "mean"),
            lore_rule_len=("rule_len_5050", "mean")))

    if shape_df is not None and len(shape_df):
        d = shape_df
        if central:
            d = d[[central.get(i) == t for i, t in zip(d.impl_name, d.setting_tag)]]
        agg = {"shape__shap_compactness": ("shap_compactness", "mean"),
               "shape__shap_k90_sparsity": ("shap_k90_sparsity", "mean"),
               "shape__shap_mass": ("shap_mass", "mean")}
        for src, dst in [("lore_rule_compactness", "shape__lore_rule_compactness"),
                         ("agreement_shap_lore", "shape__agreement_shap_lore"),
                         ("n_overlap", "shape__agreement_n_overlap")]:
            if src in d:
                agg[dst] = (src, "mean")
        fam = d.groupby("impl_name").agg(**agg)
        # ⚠️ Le colonne LORE si mediano fra run solo se le run hanno spiegato lo STESSO
        # numero di campioni.  `controlled_fusion base` ha seed 47 col campione nuovo (48) e
        # seed 8 con quello storico (32): la media dava `n_overlap = 36.5`, cioe' un accordo
        # calcolato su due campioni diversi e presentato come un numero solo.  E' lo stesso
        # errore del doppio conteggio della sezione LORE, un livello piu' su.  Qui esce NaN
        # e la colonna `shape__lore_n` mostra il perche'.
        key = "lore_sample_key" if "lore_sample_key" in d else "n_lore"
        if key in d:
            mix = d.groupby("impl_name")[key].agg(
                lambda s: sorted({str(v) for v in s.dropna()}))
            n_mix = (d.groupby("impl_name")["n_lore"]
                     .agg(lambda s: sorted({int(v) for v in s.dropna()})))
            fam["shape__lore_n"] = [",".join(map(str, v)) or None for v in n_mix]
            fam["shape__lore_samples"] = [len(v) for v in mix]
            mixed = [len(v) > 1 for v in mix]
            for c in ("shape__lore_rule_compactness", "shape__agreement_shap_lore",
                      "shape__agreement_n_overlap"):
                if c in fam:
                    fam.loc[mixed, c] = np.nan
        parts.append(fam)

    if stab_df is not None and len(stab_df):
        piv = (stab_df.pivot_table(index="impl_name", columns="pair_kind",
                                   values="stability_at_k", aggfunc="mean")
               .rename(columns={"adjacent_param": "shape__stability_adjacent",
                                "same_param_seed": "shape__stability_seed_null"}))
        if {"shape__stability_adjacent", "shape__stability_seed_null"} <= set(piv.columns):
            # E' QUESTA la lettura: la stabilita' fra pesi vicini vale qualcosa solo se
            # scende sotto il rumore di seed. Positivo = i pesi spostano piu' del seed.
            piv["shape__stability_effect"] = (piv["shape__stability_seed_null"]
                                              - piv["shape__stability_adjacent"])
        parts.append(piv)

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, axis=1)
    p = os.path.join(out_dir, "xai_summary_by_family.csv")
    out.to_csv(p)
    if verbose:
        print(f"[metriche] {os.path.basename(p)} — {len(out)} famiglie, "
              f"{len(out.columns)} colonne ({sum(c.startswith('shape__') for c in out.columns)} di forma)")
    return out


# --------------------------------------------------------------------------------------
# Grad-CAM — metriche di forma della heatmap (T4)
#
# Stesse due regole del lato tabellare: si calcolano DENTRO il motore (`MultiModalGradCAM
# Callback` le chiama appena ha le heatmap) e sono un COMPLEMENTO, mai un punteggio unico.
#
# ⚠️ E la stessa trappola dell'inversione. Una heatmap concentrata non vuol dire una
# spiegazione buona: un modello che ignora l'immagine puo' risultare il piu' "focalizzato".
# Vanno lette accanto alla fedelta', non al posto suo.
# --------------------------------------------------------------------------------------
CAM_TAU = 0.5        # soglia della maschera binaria, sulla heatmap normalizzata in [0,1]
CAM_Q = 0.10         # top-q% dei pixel per l'energy concentration


def gradcam_shape_from_cams(cams, tau=CAM_TAU, q=CAM_Q):
    """Compactness, focus, energy concentration e connectedness, mediate sui campioni.

    `cams` (n, Hc, Wc), le heatmap GREZZE. I valori negativi si azzerano prima di
    normalizzare: Grad-CAM li produce e includerli nella massa renderebbe `p_j` non una
    distribuzione.
    """
    c = np.maximum(np.asarray(cams, float), 0.0)
    if c.ndim == 2:
        c = c[None]
    n, H, W = c.shape
    flat = c.reshape(n, -1)
    tot = flat.sum(axis=1)
    live = tot > 0
    rec = {"cam_frac_empty": float((~live).mean())}
    if not live.any():
        # Ramo immagine SPENTO (`fixed_scaling` a1p000: beta=0 azzera il gradiente, la
        # heatmap e' tutta zeri). Le metriche sono NaN — regola 1 di questo modulo — ma le
        # CHIAVI ci sono tutte: uscendo con quattro chiavi su sei, queste run scrivevano un
        # CSV con uno SCHEMA DIVERSO dalle altre 105 e ogni concat cross-run si rompeva su
        # `cam_area`. Un modo che non ha risultato non e' un modo che ha meno colonne.
        rec.update(cam_compactness=np.nan, cam_focus=np.nan,
                   cam_energy_top10=np.nan, cam_connectedness=np.nan,
                   cam_area=np.nan, cam_n_components=np.nan)
        return rec
    P = flat[live] / tot[live, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log(P), 0.0)
    H_norm = -(P * logP).sum(axis=1) / np.log(H * W)
    srt = -np.sort(-P, axis=1)
    k = max(1, int(np.ceil(q * H * W)))

    # maschera binaria sulla heatmap riscalata in [0,1] per campione
    mn = flat[live].min(axis=1, keepdims=True)
    mx = flat[live].max(axis=1, keepdims=True)
    B = ((flat[live] - mn) / (mx - mn + 1e-12) >= tau).reshape(-1, H, W)
    area = B.reshape(len(B), -1).mean(axis=1)

    # componenti connesse (4-vicinato) senza dipendere da scipy/cv2
    conn = []
    for b in B:
        conn.append(_n_components(b))
    conn = np.asarray(conn, float)
    frag = conn / (B.reshape(len(B), -1).sum(axis=1) + 1e-12)

    rec.update(cam_compactness=float(np.mean(1.0 - H_norm)),
               cam_focus=float(np.mean(1.0 - area)),
               cam_energy_top10=float(np.mean(srt[:, :k].sum(axis=1))),
               cam_connectedness=float(np.mean(1.0 / (1.0 + frag))),
               cam_area=float(np.mean(area)),
               cam_n_components=float(conn.mean()))
    return rec


def _n_components(mask):
    """Componenti connesse a 4-vicinato di una maschera booleana. Flood fill iterativo."""
    m = np.asarray(mask, bool).copy()
    n = 0
    H, W = m.shape
    for i in range(H):
        for j in range(W):
            if not m[i, j]:
                continue
            n += 1
            stack = [(i, j)]
            m[i, j] = False
            while stack:
                a, b = stack.pop()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < H and 0 <= y < W and m[x, y]:
                        m[x, y] = False
                        stack.append((x, y))
    return n


def cam_masks(cams, tau=CAM_TAU):
    """Maschere binarie per campione — la base della IoU fra configurazioni (Stability_CAM)."""
    c = np.maximum(np.asarray(cams, float), 0.0).reshape(len(cams), -1)
    mn, mx = c.min(axis=1, keepdims=True), c.max(axis=1, keepdims=True)
    return (c - mn) / (mx - mn + 1e-12) >= tau


# Sotto questa soglia la differenza fra due heatmap e' rumore di float32, non un effetto:
# su una griglia normalizzata in [0,1] due riduzioni con ordine diverso divergono di qualche
# ulp (misurato: 2.4e-07 su 100 run dove la heatmap e' identica per costruzione).
CAM_EPS_FLOAT32 = 1e-05


def cam_mode_contrast(cams_a, cams_b, preds_a=None, preds_b=None, tau=CAM_TAU):
    """Quanto due configurazioni (es. `context` vs `image_only`) differiscono DAVVERO.

    Restituisce tre cose che vanno lette insieme, perche' separate mentono:

    * `cam_iou_modes` — sovrapposizione delle maschere binarie. E' la metrica storica ed e'
      GROSSOLANA: su una griglia 6x6 (36 celle, 16 dopo il bordo) non distingue uno
      spostamento piccolo da nessuno spostamento.
    * `cam_dmean` / `cam_dmax` — differenza sulle heatmap continue, che la IoU non vede.
      `cam_modes_identical` dice se sta sotto il rumore di float32: quando e' vero, i due
      modi NON sono stati confrontati davvero, sono lo stesso calcolo due volte.
    * `pred_dmean` / `pred_dmax` — quanto si muove la predizione. E' l'unico canale attivo
      quando la testa e' additiva nei due rami: senza, la riga sembra dire che azzerare il
      tabellare non fa niente.

    `pred_dcam_rho` (Spearman fra |Δcam| e |Δpred| per campione) METTE ALLA PROVA l'ipotesi
    che le due differenze siano lo stesso fenomeno — non la conferma. Su `adaptive_gating`,
    l'unica famiglia dove le heatmap cambiano davvero, la risposta e' **debolmente si'**:
    +0.17 in media, 4 seed su 5 positivi, uno a -0.10. Un valore basso qui NON significa che
    l'effetto non c'e', significa che «dove guarda» e «quanto e' sicuro» si muovono su assi
    in gran parte indipendenti. Se le heatmap sono identiche e' NaN per definizione.
    """
    a = np.asarray(cams_a, float).reshape(len(cams_a), -1)
    b = np.asarray(cams_b, float).reshape(len(cams_b), -1)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    ma, mb = cam_masks(a, tau), cam_masks(b, tau)
    inter, union = (ma & mb).sum(axis=1), (ma | mb).sum(axis=1)
    live = union > 0
    rec = {
        "cam_iou_modes": float(np.mean(inter[live] / union[live])) if live.any() else np.nan,
        "cam_iou_n_valid": int(live.sum()),   # 0 = ramo immagine spento, la IoU e' NaN a ragione
    }

    d = np.abs(a - b)
    d_per_sample = d.mean(axis=1)
    rec["cam_dmean"] = float(d_per_sample.mean())
    rec["cam_dmax"] = float(d.max()) if d.size else np.nan
    rec["cam_modes_identical"] = bool(rec["cam_dmax"] <= CAM_EPS_FLOAT32)

    if preds_a is None or preds_b is None:
        return rec
    pa = np.asarray(preds_a, float).reshape(len(preds_a), -1)[:n]
    pb = np.asarray(preds_b, float).reshape(len(preds_b), -1)[:n]
    dp = np.abs(pa - pb).mean(axis=1)
    rec["pred_dmean"] = float(dp.mean())
    rec["pred_dmax"] = float(dp.max()) if dp.size else np.nan
    rho = np.nan
    if not rec["cam_modes_identical"] and d_per_sample.std() > 0 and dp.std() > 0:
        from scipy.stats import spearmanr
        rho = float(spearmanr(d_per_sample, dp).correlation)
    rec["pred_dcam_rho"] = rho
    return rec
