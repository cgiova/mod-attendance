"""LORE on `lore_sa` -- the rewrite that actually has a batched genetic generator.

WHY A SECOND LORE ENGINE.  The stack we were on (`xailib` -> `lore_explainer`/LORE_ext) is
already at its newest available code: XAI-Lib is pinned at upstream HEAD (8ea5a31) and
LORE_ext is 1.0.3, the latest release.  Its slowness is architectural, not staleness:
`lore_explainer/neighgen.py` evaluates GA fitness ONE INDIVIDUAL AT A TIME
(`bb_predict(x1.reshape(1, -1))`), so a single explained instance costs on the order of
44k single-row black-box calls.

`lore_sa` 1.1.3 keeps that implementation as `LegacyGeneticGenerator` and adds a
`GeneticGenerator` whose `population_fitness_equal`/`_notequal` decode and predict the
WHOLE population in one call.  That is the "the code is faster now" everyone was
referring to -- a different package, not a newer version of ours.

Two further inefficiencies were OURS and are fixed here by construction:
  * `_BB.predict` returned a 1-D array, so `bb_predict(x)[0]` was a 0-d scalar and the
    `neighgen.py:260-263` shape fallback fired on EVERY evaluation, doubling 2 predict
    calls into 4;
  * `lore_config['size']` never reached the generator -- `xailib`'s wrapper hardcodes
    `explain_instance(x, samples=1000)`.  Here the budget is a real argument.

The legacy `LoreEngine` stays in `xai_engines` and remains the reference: it is what
produced every published `lore_fidelity`.  This engine is for measurement, and its
numbers are NOT interchangeable with the published ones.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd


def _rule_len(rule):
    """Number of premises in a LORE rule.

    `lore_sa`'s `explain()` returns the rule as a **dict** (`Rule.to_dict()`, key
    'premises'), not as the `Rule` object -- reading it with `getattr(rule, 'premises')`
    silently yields 0 for every sample and makes the rules look empty. Handle both.
    """
    if rule is None:
        return None
    if isinstance(rule, dict):
        return len(rule.get("premises") or rule.get("premise") or [])
    return len(getattr(rule, "premises", []) or [])


def _rule_text(rule, max_premises=12):
    """Render a `lore_sa` rule the way the legacy engine writes them, consequent included.

    `LoreEngineSA` used to keep only `len(premises)` and drop everything else, so the rules
    could not be read at all -- which is how a `rule_len` of 1 went unexplained.  The
    consequent is `dt.predict(z)` and therefore always equals the black box's own class for
    the sample; it is printed for completeness, not as information.
    """
    if not isinstance(rule, dict):
        return None
    prem = rule.get("premises") or rule.get("premise") or []
    parts = [f"{p.get('attr')} {p.get('op')} {p.get('val')}" for p in prem[:max_premises]]
    if len(prem) > max_premises:
        parts.append(f"... (+{len(prem) - max_premises})")
    cons = rule.get("consequence") or {}
    head = " AND ".join(parts) if parts else "<nessuna premessa>"
    return f"IF {head} THEN {cons.get('attr', 'target')} {cons.get('op', '=')} {cons.get('val')}"


def neighbourhood_stats(X, y=None):
    """Is this neighbourhood usable at all?

    The number that was missing from BOTH engines.  A neighbourhood can contain both
    classes and still be worthless if it is a handful of points repeated hundreds of times:
    a tree then separates the duplicate block with a single cut, which is exactly how
    `lore_sa`'s genetic generator produces `rule_len = 1` and an in-sample fidelity of 1.0.
    Measured on the default generator: 331 rows, 88 distinct, 122 of them identical to the
    explained instance.

    `unique_frac` is the one to look at; `pos_rate` alone is not enough.
    """
    if X is None or len(X) == 0:
        return {}
    arr = np.asarray(X, dtype=object)
    keys = [tuple(r) for r in arr]
    counts = pd.Series(keys, dtype=object).value_counts()
    out = {
        "neigh_rows": int(len(arr)),
        "neigh_unique": int(len(counts)),
        "neigh_unique_frac": float(len(counts) / len(arr)),
        "neigh_max_multiplicity": int(counts.iloc[0]),
        "neigh_top_frac": float(counts.iloc[0] / len(arr)),
    }
    if y is not None and len(y):
        y = np.asarray(y).ravel()
        out["neigh_pos_rate"] = float(np.mean(y))
        out["neigh_n_classes"] = int(len(np.unique(y)))
    return out


def degeneracy_warning(stats, min_unique_frac=0.5):
    """A sentence naming what is wrong, or None. Returned so callers can persist it."""
    if not stats:
        return None
    if stats.get("neigh_n_classes", 2) < 2:
        return ("vicinato MONO-CLASSE: il surrogato non ha niente da separare, quindi "
                "`fidelity` e `rule_len` non misurano la qualita' della spiegazione.")
    uf = stats.get("neigh_unique_frac")
    if uf is not None and uf < min_unique_frac:
        return (f"vicinato SATURO DI DUPLICATI: solo {uf:.0%} di righe distinte, la piu' "
                f"frequente vale il {stats['neigh_top_frac']:.0%} del totale. Un albero le "
                f"separa con un taglio solo, quindi `rule_len` risulta corto e la fedelta' "
                f"in-sample risulta ~1.0 per costruzione.")
    return None


def _split_image_head(model):
    """Split at GlobalAveragePooling2D so the CNN backbone runs ONCE per explained sample.

    Everything downstream then re-scores only the cheap head, which is what makes a
    thousand-neighbour genetic neighbourhood affordable at all.
    """
    import keras
    import tensorflow as tf

    gap = [l for l in model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D)]
    if not gap:
        raise ValueError(
            "this model has no GlobalAveragePooling2D: the image/head split LORE relies "
            "on is undefined for it.")
    img_emb, feat_in = gap[0].output, model.input["features"]
    img_enc = keras.Model(model.input["images"], img_emb)
    head = keras.Model([img_emb, feat_in], model.output)
    head_fn = tf.function(lambda e, f: head([e, f], training=False), reduce_retracing=True)
    return img_enc, head_fn


class KerasTabularBBox:
    """`lore_sa.AbstractBBox`-shaped wrapper: batched, one fixed image embedding.

    `lore_sa`'s generator hands `predict` the DECODED neighbourhood -- the whole
    population at once -- which is exactly the shape our vectorised predictor always
    wanted and never got from the old library.
    """

    def __init__(self, head_fn, image_embedding, feat_cols, feature_space,
                 to_feature_dict, cat_cols, threshold=0.5):
        import tensorflow as tf
        self._head_fn = head_fn
        self._emb = tf.constant(np.asarray(image_embedding)[np.newaxis], dtype=tf.float32)
        self._feat_cols = list(feat_cols)
        self._fs = feature_space
        self._to_fd = to_feature_dict
        self._cat = set(cat_cols)
        self.thr = threshold
        self.n_calls = 0
        self.n_rows = 0
        # Last multi-row batch handed to `predict`.  `Lore.explain` scores the finished
        # neighbourhood at lore.py:151 and afterwards only ever asks for the single
        # original row (:195), so "the last call with more than one row" IS the
        # neighbourhood the surrogate gets fitted on -- the thing we need to inspect.
        self.neigh_X = None
        self.neigh_y = None

    def _proba(self, X):
        import tensorflow as tf
        if isinstance(X, pd.DataFrame):
            df = X[self._feat_cols].copy()
        else:
            arr = np.atleast_2d(np.asarray(X, dtype=object))
            df = pd.DataFrame(arr, columns=self._feat_cols)
        for c in self._feat_cols:
            df[c] = df[c].astype(str) if c in self._cat else pd.to_numeric(
                df[c], errors="coerce").astype("float32")
        # Numerics that failed to parse must not reach the FeatureSpace as NaN: `nan >= thr`
        # is False, so the row would silently score as class 0.  The categorical columns are
        # strings by now and are unaffected, hence filling the frame is always safe.
        # (This condition used to be `if not self._cat`, i.e. inverted: the fill was skipped
        # exactly when categoricals existed, which is our case.)
        df = df.fillna(0)

        enc = self._fs(self._to_fd(df))
        n = int(enc.shape[0])
        self.n_calls += 1
        self.n_rows += n
        return self._head_fn(tf.repeat(self._emb, n, axis=0), enc).numpy()[:, 0]

    def predict(self, X):
        # 2-D on purpose: the LORE_ext shape fallback that doubled the predict count
        # triggered precisely because the old wrapper returned 1-D.
        y = (self._proba(X) >= self.thr).astype(int)
        if y.size > 1:
            self.neigh_X = np.atleast_2d(np.asarray(
                X[self._feat_cols] if isinstance(X, pd.DataFrame) else X, dtype=object))
            self.neigh_y = y
        return y

    def predict_proba(self, X):
        p = self._proba(X)
        return np.column_stack([1 - p, p])

    def model(self):
        return self._head_fn


def _diversify_init(generator, init_mutpb):
    """Seed the GA with mutated copies of `x` instead of `x` itself.

    `GeneticGenerator.record_init` (`lore_sa/neighgen/genetic.py:176-184`) returns the
    instance unchanged -- its own docstring says "we repeat the input instance x for all the
    initial population".  Every individual therefore starts identical; crossover between
    identical parents is a no-op, `mutpb=0.2` touches a fifth of them, tournament selection
    removes what little diversity appears, and the hall of fame puts the duplicates back.
    Measured outcome: 331 rows of which 88 distinct, 122 identical to `x`.

    This replaces the initialiser on the instance (the convenience explainers construct the
    generator themselves, so there is no constructor to pass through).  It changes ONLY the
    starting population; selection, crossover and the fitness functions are untouched.

    THE COPY IS LOAD-BEARING.  `generate_synthetic_instance` does ``instance = from_z``
    without copying (`neighborhood_generator.py:113-115`) and returns that same object, so
    it mutates its argument IN PLACE.  Handing it `x` directly would re-mutate the one
    array on every call and return the same object every time -- every individual would
    still be identical, just progressively more scrambled.  Verified: five calls returned
    `is`-identical objects.
    """
    import types

    def record_init(self, x):
        return self.generate_synthetic_instance(from_z=np.array(x, copy=True),
                                                mutpb=init_mutpb)

    generator.record_init = types.MethodType(record_init, generator)
    return generator


def _audited_surrogate_cls():
    """`DecisionTreeSurrogate` that also reports a fidelity which is not in-sample.

    `lore_sa` sets ``fidelity = dt.score(Z, Yb)`` (decision_tree.py:85) on a tree built
    with no `max_depth`, no `min_samples_leaf` and no pruning -- `grid_search_tree` and
    `prune_tree` default to False and the convenience explainers never set them, so those
    branches are dead code.  That number is the training accuracy of an unconstrained tree
    on its own training set: 1.0 unless duplicate rows carry conflicting labels.  It says
    nothing about the explanation.

    The legacy engine already knew this and repaired it by redrawing a neighbourhood
    (`xai_engines.py:1258-1273`).  Redrawing costs a second full GA run per sample, so the
    cheaper question is asked here instead: cross-validated on held-out folds of the SAME
    neighbourhood, does a tree of this family generalise?  Zero extra black-box calls.

    TWO WAYS IT IS WEAKER THAN A FRESH DRAW, both of which matter:

    * it cannot see an *unrepresentative* neighbourhood, only an *unlearnable* one -- a
      neighbourhood concentrated in the wrong place scores well here;
    * on a duplicate-saturated neighbourhood it is **optimistically biased**, because the
      same repeated point lands on both sides of the split, so the held-out fold is partly
      memorised.  That is exactly the regime this engine produces by default, i.e. the
      number is least trustworthy precisely where the diagnosis is worst.

    So read it next to `neigh_unique_frac`, never alone, and never call it `fidelity`.
    """
    from lore_sa.surrogate import DecisionTreeSurrogate

    class _AuditedSurrogate(DecisionTreeSurrogate):
        def train(self, Z, Yb, weights=None):
            dt = super().train(Z, Yb, weights)
            self.fid_insample = self.fidelity
            self.fid_cv_acc = self.fid_cv_bacc = None
            try:
                from sklearn.base import clone
                from sklearn.model_selection import StratifiedKFold, cross_val_score
                Yb_ = np.asarray(Yb).ravel()
                _, counts = np.unique(Yb_, return_counts=True)
                if len(counts) >= 2 and counts.min() >= 3:
                    k = int(min(5, counts.min()))
                    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
                    self.fid_cv_acc = float(cross_val_score(
                        clone(self.dt), Z, Yb_, cv=cv, scoring="accuracy").mean())
                    self.fid_cv_bacc = float(cross_val_score(
                        clone(self.dt), Z, Yb_, cv=cv, scoring="balanced_accuracy").mean())
            except Exception:
                pass          # diagnostics must never break an explanation
            return dt

    return _AuditedSurrogate


class LoreEngineSA:
    """Explain the tabular branch with `lore_sa`, image held fixed per sample."""

    def __init__(self, feature_space, feat_cols, to_feature_dict, training_df,
                 neigh_type="genetic", num_instances=1000, threshold=0.5, random_state=0,
                 mutpb=0.5, init_mutpb=0.3, gen_kwargs=None, audit=True):
        """
        ``mutpb`` / ``init_mutpb`` DEFAULT TO OUR TUNED VALUES, NOT `lore_sa`'s.  Upstream
        uses ``mutpb=0.2`` and an initial population that is the explained instance
        repeated, which on ISIC-24 produces a **mono-class neighbourhood on every sample**:
        26% distinct rows, one row worth 74% of the total, `rule_len = 0`.  Measured
        (vanilla seed_47, 2 samples, `num_instances=1000`):

            mutpb  init   unique_frac   pos-rate del vicinato   rule_len   fid_cv_bacc
            .2     --          0.260                    0.000        0.0          n/d
            .35    --          0.483                    0.389        1.5        0.926
            .5     --          0.694                    0.324        2.0        0.898
            .5     .3          0.742                    0.307        3.0        0.888
            .8     --          0.982                    0.079        3.0        0.750
            random             0.982                    0.017        1.0        0.823

        The binding constraint is the THIRD column, not the second: `random` and
        ``mutpb=.8`` reach 98% distinct rows but almost no counter-class instances, so the
        surrogate has nothing to separate.  Diversity in feature space is not diversity in
        the label.  ``mutpb=.5`` + ``init_mutpb=.3`` is the best joint point measured --
        2.9x the diversity of upstream, the longest rules, and 19% extra cost.

        Pass ``mutpb=None, init_mutpb=None`` for `lore_sa`'s own behaviour (what
        `bench_lore.py`'s timing numbers were measured with).

        ``gen_kwargs``: further attributes pushed onto the generator after construction, e.g.
        ``{"mutpb": 0.5, "ngen": 50}``.  The convenience explainers build the generator
        with fixed defaults (`GeneticGenerator(bbox, dataset, encoder, 0.1)`), so this is
        the only way to reach `mutpb` / `ngen` / `cxpb` / `alpha1` / `alpha2` / `ocr`.
        They matter: with the default `mutpb=0.2` and an initial population that is the
        explained instance repeated (`record_init` returns `x` unchanged), the
        neighbourhood stays saturated with duplicates.

        ``init_mutpb``: if set, seed the GA with copies of the instance mutated at this
        rate instead of identical copies -- see :func:`_diversify_init`.  ``None`` keeps
        `lore_sa`'s own behaviour, so nothing already measured changes silently.

        ``audit``: swap in the surrogate that also reports a non-in-sample fidelity.  On
        by default because `fidelity` alone is 1.0 by construction and reads as a result.
        """
        self.feature_space = feature_space
        self.feat_cols = list(feat_cols)
        self.to_feature_dict = to_feature_dict
        self.cat_cols = [c for c in self.feat_cols
                         if not pd.api.types.is_numeric_dtype(training_df[c])]
        self.neigh_type = neigh_type
        self.num_instances = num_instances       # a REAL knob here, unlike lore_config['size']
        self.threshold = threshold
        self.random_state = random_state
        self.gen_kwargs = dict(gen_kwargs or {})
        # `mutpb` e `init_mutpb` sono leve del solo generatore genetico: `RandomGenerator`
        # non ha ne' l'attributo ne' `record_init`, e passarglieli farebbe alzare
        # l'AttributeError del guard in `_make_explainer` per un default che non ha chiesto.
        if neigh_type == "genetic":
            if mutpb is not None:
                self.gen_kwargs.setdefault("mutpb", mutpb)   # gen_kwargs esplicito ha la precedenza
            self.init_mutpb = init_mutpb
        else:
            self.init_mutpb = None
        self.audit = audit

    def _make_dataset(self, raw_df, labels):
        from lore_sa.dataset import TabularDataset
        df = raw_df[self.feat_cols].copy()
        for c in self.cat_cols:
            df[c] = df[c].astype(str)
        df["target"] = np.asarray(labels).astype(int)
        return TabularDataset(data=df, class_name="target",
                              categorial_columns=self.cat_cols + ["target"])

    def _make_explainer(self, bbox, dataset):
        from lore_sa.lore import TabularGeneticGeneratorLore, TabularRandomGeneratorLore
        cls = {"genetic": TabularGeneticGeneratorLore,
               "random": TabularRandomGeneratorLore}[self.neigh_type]
        expl = cls(bbox, dataset)
        if self.audit:
            expl.surrogate = _audited_surrogate_cls()()
        for k, v in self.gen_kwargs.items():
            if not hasattr(expl.generator, k):
                raise AttributeError(
                    f"gen_kwargs: il generatore {type(expl.generator).__name__} non ha "
                    f"l'attributo {k!r} -- un refuso qui verrebbe ignorato in silenzio.")
            setattr(expl.generator, k, v)
        if self.init_mutpb is not None and hasattr(expl.generator, "record_init"):
            _diversify_init(expl.generator, self.init_mutpb)
        return expl

    def explain(self, model, raw_df, imgs, labels, max_explain_samples=2, verbose=True):
        """Explain the first `max_explain_samples` rows; returns (stats, per-sample rows)."""
        img_enc, head_fn = _split_image_head(model)
        embs = img_enc.predict(imgs, batch_size=32, verbose=0)
        dataset = self._make_dataset(raw_df, labels)

        rows = []
        n = min(max_explain_samples, len(raw_df))
        for i in range(n):
            bbox = KerasTabularBBox(
                head_fn, embs[i], self.feat_cols, self.feature_space,
                self.to_feature_dict, self.cat_cols, self.threshold)
            expl = self._make_explainer(bbox, dataset)

            x = raw_df.iloc[i][self.feat_cols].copy()
            for c in self.cat_cols:
                x[c] = str(x[c])

            t0 = time.time()
            try:
                out = expl.explain(x.values, num_instances=self.num_instances)
                err = None
            except Exception as e:
                out, err = {}, f"{type(e).__name__}: {e}"
            dt = time.time() - t0

            nstats = neighbourhood_stats(bbox.neigh_X, bbox.neigh_y)
            rule = (out or {}).get("rule")
            sur = getattr(expl, "surrogate", None)
            rows.append({
                "i": i, "seconds": dt, "min_per_sample": dt / 60.0,
                # NOT a quality measure: lore_sa sets it to dt.score(Z, Yb) on the very
                # matrix the tree was fitted on. Named so it cannot be mistaken for one.
                "fid_insample": (out or {}).get("fidelity"),
                "fid_cv_acc": getattr(sur, "fid_cv_acc", None),
                "fid_cv_bacc": getattr(sur, "fid_cv_bacc", None),
                "rule_len": _rule_len(rule),
                "rule_text": _rule_text(rule),
                "rule_class": ((rule or {}).get("consequence") or {}).get("val")
                              if isinstance(rule, dict) else None,
                "n_counterfactuals": len((out or {}).get("counterfactuals") or []),
                "true_label": int(np.asarray(labels).ravel()[i]),
                "n_bbox_calls": bbox.n_calls, "n_bbox_rows": bbox.n_rows,
                "rows_per_call": (bbox.n_rows / bbox.n_calls) if bbox.n_calls else None,
                **nstats,
                "warning": degeneracy_warning(nstats),
                "error": err,
            })
            if verbose:
                r = rows[-1]
                cv = f"{r['fid_cv_bacc']:.3f}" if r["fid_cv_bacc"] is not None else "n/d"
                print(f"  [{i+1}/{n}] {dt/60:.2f} min | rule_len={r['rule_len']} "
                      f"| fid in-sample={r['fid_insample']} cv_bacc={cv} "
                      f"| vicinato {r.get('neigh_unique', '?')}/{r.get('neigh_rows', '?')} distinte"
                      f"{' | ' + err if err else ''}", flush=True)
                if r["warning"]:
                    print(f"        ⚠ {r['warning']}", flush=True)

        df = pd.DataFrame(rows)
        mean = lambda c: float(df[c].dropna().mean()) if c in df and df[c].notna().any() else None
        stats = {
            "engine": "lore_sa", "neigh_type": self.neigh_type,
            "num_instances": self.num_instances, "gen_kwargs": dict(self.gen_kwargs),
            "n_explained": int(len(df)),
            "min_per_sample": mean("min_per_sample"),
            # cost
            "bbox_calls_mean": mean("n_bbox_calls"), "rows_per_call_mean": mean("rows_per_call"),
            # quality of the neighbourhood -- read these BEFORE the fidelity columns
            "neigh_unique_frac_mean": mean("neigh_unique_frac"),
            "neigh_top_frac_mean": mean("neigh_top_frac"),
            "neigh_pos_rate_mean": mean("neigh_pos_rate"),
            "n_degenerate": int(df.warning.notna().sum()) if "warning" in df else 0,
            # quality of the explanation
            "rule_len_mean": mean("rule_len"),
            "fid_insample_mean": mean("fid_insample"),      # ~1.0 by construction
            "fid_cv_acc_mean": mean("fid_cv_acc"),
            "fid_cv_bacc_mean": mean("fid_cv_bacc"),        # the one that can discriminate
            "n_errors": int(df.error.notna().sum()),
        }
        if verbose and stats["n_degenerate"]:
            print(f"\n  ⚠ {stats['n_degenerate']}/{len(df)} vicinati degeneri: le colonne di "
                  f"fedelta' su quei campioni non sono interpretabili.", flush=True)
        return stats, df


# --------------------------------------------------------------------------------------
# Callback -- l'aggancio alla pipeline
#
# Fino al 2026-07-29 questo motore era raggiungibile solo dai benchmark: `make_xai_callbacks`
# costruiva sempre il `LoreCallback` legacy, quindi la configurazione scelta misurando non
# arrivava a nessun artefatto.  Questo callback ha la stessa forma di quello legacy (stesso
# `base_dir`, stesso `epoch{NNN}/` per epoca, stesso `lore_fidelity_per_epoch.csv` con dedupe
# via `append_epoch_row`) cosi' i due percorsi si possono confrontare senza riscrivere nulla
# a valle.
#
# NON sostituisce il legacy di default: ogni `lore_fidelity` pubblicato viene da li', e i due
# motori non spiegano la stessa funzione (embedding immagine condiviso vs per-campione).
# La scelta si fa con `xai_driver.LORE_ENGINE`.
# --------------------------------------------------------------------------------------

def _lore_sa_callback_cls():
    import keras

    class LoreSACallback(keras.callbacks.Callback):
        """`LoreEngineSA` guidato come il `LoreCallback` legacy, sugli stessi percorsi."""

        def __init__(self, data_source, raw_feature_df, feat_cols, feature_space,
                     to_feature_dict, training_df, base_dir, max_explain_samples=32,
                     every_n_epochs=1, max_samples=256, max_batches=2, rng_seed=0,
                     subset_idx=None, **engine_kw):
            super().__init__()
            from pathlib import Path
            self.data_source = data_source
            self.raw_feature_df = raw_feature_df
            self.feat_cols = list(feat_cols)
            self.max_explain_samples = int(max_explain_samples)
            self.every_n_epochs = max(1, int(every_n_epochs))
            self.max_samples = int(max_samples)
            self.max_batches = max_batches
            self.rng_seed = int(rng_seed)
            # subset_idx: indici POSIZIONALI in validation_df. None = comportamento storico
            # (le prime N righe), che NON e' un campione casuale -- vedi isic24.xai_subset.
            self.subset_idx = None if subset_idx is None else np.asarray(subset_idx, int)
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._images = self._raw_df = self._labels = None
            self.last_quality = {}
            self._engine = LoreEngineSA(
                feature_space=feature_space, feat_cols=feat_cols,
                to_feature_dict=to_feature_dict, training_df=training_df, **engine_kw)

        def _freeze_batch(self, ds):
            # Con subset_idx si legge fino alla posizione piu' alta richiesta e poi si
            # indicizza: max_batches/max_samples sono tarati sul prefisso e taglierebbero
            # via i campioni alti del campione stratificato.
            want = None if self.subset_idx is None else int(self.subset_idx.max()) + 1
            imgs, labels, n = [], [], 0
            for b_idx, (x, y) in enumerate(ds):
                if want is None:
                    if self.max_batches and b_idx >= self.max_batches:
                        break
                    take = self.max_samples - n
                else:
                    take = want - n
                if take <= 0:
                    break
                imgs.append(x["images"].numpy()[:take])
                labels.append(np.asarray(y)[:take].reshape(-1))
                n += imgs[-1].shape[0]
            if n == 0:
                raise RuntimeError("nessun campione raccolto per LORE.")
            self._images = np.concatenate(imgs)
            self._labels = np.concatenate(labels)
            self._raw_df = self.raw_feature_df.iloc[:n][self.feat_cols].reset_index(drop=True)
            if self.subset_idx is not None:
                if n < want:
                    raise RuntimeError(
                        f"validation_ds ha dato {n} campioni ma il subset ne chiede fino a "
                        f"{want}: il campione non e' applicabile a questo dataset.")
                self._images = self._images[self.subset_idx]
                self._labels = self._labels[self.subset_idx]
                self._raw_df = (self._raw_df.iloc[self.subset_idx]
                                .reset_index(drop=True))

        def on_train_begin(self, logs=None):
            self._freeze_batch(self.data_source)

        def on_epoch_end(self, epoch, logs=None):
            if epoch % self.every_n_epochs != 0 or self._images is None:
                return
            from .xai_engines import append_epoch_row
            np.random.seed(self.rng_seed)
            epoch_name = f"epoch{epoch:03d}"
            out_dir = self.base_dir / epoch_name / "tabular_sa"

            # La directory si crea DOPO il calcolo, non prima: `explain` dura ~20 minuti e
            # un'interruzione lasciava una `tabular_sa/` vuota che il write guard poi
            # scambiava per un artefatto, bloccando la ri-esecuzione. Il guard ora ignora le
            # dir vuote (`xai_driver._is_artifact`); questo evita di crearle del tutto.
            stats, df = self._engine.explain(
                self.model, self._raw_df, self._images, self._labels,
                max_explain_samples=self.max_explain_samples, verbose=False)

            out_dir.mkdir(parents=True, exist_ok=True)

            # Le regole per esteso: il legacy le scrive, e senza di queste `rule_len` resta
            # un numero che non si puo' controllare.
            with open(out_dir / f"lore_sa_rules_{epoch_name}.txt", "w") as f:
                for _, r in df.iterrows():
                    f.write(f"[inst {int(r['i']):02d}] label={r.get('true_label')} "
                            f"classe_regola={r.get('rule_class')}\n  {r.get('rule_text')}\n\n")
            df.to_csv(out_dir / f"lore_sa_persample_{epoch_name}.csv", index=False)

            self.last_quality = {
                "lore_sa_fid_cv_bacc": stats.get("fid_cv_bacc_mean"),
                "lore_sa_fid_insample": stats.get("fid_insample_mean"),
                "lore_sa_rule_len": stats.get("rule_len_mean"),
                "lore_sa_neigh_unique_frac": stats.get("neigh_unique_frac_mean"),
                "lore_sa_neigh_pos_rate": stats.get("neigh_pos_rate_mean"),
                "lore_sa_n_degenerate": stats.get("n_degenerate"),
                # copertura: su quanti campioni la spiegazione e' interpretabile. Va
                # RIPORTATA, non assunta -- misurato che con l'immagine congelata una parte
                # dei campioni ha un vicinato mono-classe qualunque sia il generatore.
                "lore_sa_coverage": (1 - stats.get("n_degenerate", 0) / max(1, len(df))),
            }
            append_epoch_row(self.base_dir / "lore_sa_per_epoch.csv",
                             {"epoch": int(epoch), **self.last_quality})

    return LoreSACallback
