"""ARCHIVED — Stage 0-3 verification runs, extracted verbatim from the post-hoc notebook.

These are the checks whose CONCLUSIONS WERE RETRACTED (see
`artifacts/enhanced/reports/FINDINGS_insertion_deletion_2026-07-21.md`).  They are kept
because two of them are still cited as primary artifact sources in the reports:

  * Stage 2  -> `runs/_robustness/stage2_aligned_attr/faithfulness_aligned_attr.csv`
                (the aligned Permutation attributor; REPORT_S sec.3.3 column)
  * Stage 3  -> `runs/_robustness/stage3_image/`  (image subset + blur-strength checks)

They were moved out of the notebook on 2026-07-28: at 1599 lines they were 46% of it,
while contributing nothing to the live analysis.

HOW TO RUN
----------
This is a straight extraction, not a standalone program: the blocks read the post-hoc
notebook's session state (`validation_df`, `feature_space`, `FEAT_COLS`, `RUNS_PATH`,
`_build_faith_subset`, `explanation_faithfulness`, ...).  Execute the post-hoc notebook's
setup section first, then run the block you want -- e.g. paste it into a cell, or
`exec(open(...).read())` after `xf.configure(...)`.

Note the duplicate that was dropped in the move: the notebook carried TWO near-identical
Stage-2b implementations (Italian- and English-commented) appending to the SAME csv, with
the explanatory markdown between them.  Only one is kept here.
"""

# ======================================================================================
# [notebook cell 37] dependency install (LORE via XAI-Lib)
# ======================================================================================

# !pip install git+https://github.com/kdd-lab/XAI-Lib.git


# ======================================================================================
# [notebook cell 38] engine + plot definitions (now largely superseded by isic24.xai_engines)
# ======================================================================================

import seaborn as sns

def plot_topk_grouped_raw_global_pos_neg(
    global_imp, pos_imp, neg_imp, k=10, epoch_num=None, figsize=(10, 5), out_path=None
):
    # Ranking
    items = sorted(global_imp.items(), key=lambda kv: kv[1], reverse=True)[:k]
    names = [k for k, _ in items]

    df = pd.DataFrame({
        "feature": names,
        "Global":   [global_imp.get(n, 0.0) for n in names],
        "Pos preds":[pos_imp.get(n,   0.0) for n in names],
        "Neg preds":[neg_imp.get(n,   0.0) for n in names],
    })

    df_melt = df.melt(id_vars="feature", var_name="Class", value_name="Importance")

    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_melt,
        y="feature", x="Importance", hue="Class",
        palette=(
            sns.color_palette("bright")[4],  # Global
            sns.color_palette("bright")[0],  # Pos
            sns.color_palette("bright")[3],  # Neg
        ),
        edgecolor="black",
    )

    for container in ax.containers:
        ax.bar_label(container, fmt="{:.2f}", padding=4, fontsize=7)

    plt.ylabel("Mean |SHAP|")
    plt.xlabel("")
    title = f"Feature Importance (top {k}) – Global vs Pos vs Neg"
    if epoch_num is not None:
        title += f" – {epoch_num}"
    plt.title(title)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

def encoded_indices_for_raw(encoded_feature_names, feat_name):
    idxs = []
    for j, name in enumerate(encoded_feature_names):
        if "__" in name:
            if name.split("__", 1)[0] == feat_name:
                idxs.append(j)
        elif name.startswith(f"{feat_name}_bin_"):
            idxs.append(j)
        elif name == feat_name:
            idxs.append(j)
    return np.array(idxs, dtype=int)


def select_top_numeric_features(global_importance_raw, raw_batch_df, k=4):
    """Return top-k raw features among those that are numeric-only."""
    numeric_feats = [
        f for f in global_importance_raw.keys()
        if f in raw_batch_df.columns and raw_batch_df[f].dtype.kind in "if"  # int or float
    ]

    ordered = sorted(
        numeric_feats,
        key=lambda name: global_importance_raw[name],
        reverse=True,
    )

    return ordered[:k]


def plot_shap_scatter_top4_numeric(
    shap_vals,
    preds,
    encoded_feature_names,
    global_importance_raw,
    raw_batch_df,
    explained_indices,
    epoch_num=None,
    positive_threshold=0.5,
    out_path=None,
):
    top_feats = select_top_numeric_features(global_importance_raw, raw_batch_df, k=4)

    pred_scores = preds[:, 0]
    pos_mask = pred_scores >= positive_threshold
    neg_mask = ~pos_mask

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    title = "SHAP values vs. top numeric features (pos vs. neg predictions)"
    if epoch_num is not None:
        title += f" – {epoch_num}"
    fig.suptitle(title, fontsize=14)

    colors = sns.color_palette("bright", 5)
    pos_color, neg_color = colors[0], colors[3]

    for i, (ax, feat_name) in enumerate(zip(axes, top_feats)):
        enc_idx = encoded_indices_for_raw(encoded_feature_names, feat_name)
        if enc_idx.size == 0:
            ax.set_title(f"{feat_name} (no encoded dims)")
            ax.axis("off")
            continue

        shap_feat = shap_vals[:, enc_idx].sum(axis=1)

        raw_vals_batch = raw_batch_df[feat_name].values
        raw_vals = raw_vals_batch[explained_indices]

        x_pos = raw_vals[pos_mask]
        y_pos = shap_feat[pos_mask]
        x_neg = raw_vals[neg_mask]
        y_neg = shap_feat[neg_mask]

        ax.scatter(
            x_pos, y_pos,
            s=18, marker="o", alpha=0.7,
            label="Positive", color=pos_color, edgecolors="none",
        )
        ax.scatter(
            x_neg, y_neg,
            s=18, marker="o", alpha=0.8,
            label="Negative", color=neg_color, edgecolors="none",
        )

        ax.axhline(0.0, color="grey", linewidth=1, linestyle="--", alpha=0.7)
        ax.set_title(feat_name)

        if i in (0, 2):
            ax.set_ylabel("SHAP value")

    for i in range(len(top_feats), 4):
        axes[i].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path is not None:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_lore_feature_frequency(
    freq_all: dict, freq_pos: dict, freq_neg: dict,
    k: int = 10, epoch_num=None, mode: str = '', figsize=(10, 5), out_path=None,
):
    """Barplot of rule-feature frequencies: Global / Pos / Neg predictions."""
    items = sorted(freq_all.items(), key=lambda kv: kv[1], reverse=True)[:k]
    if not items:
        return
    names = [n for n, _ in items]
    df = pd.DataFrame({
        'feature': names,
        'Global': [freq_all.get(n, 0) for n in names],
        'Pos preds': [freq_pos.get(n, 0) for n in names],
        'Neg preds': [freq_neg.get(n, 0) for n in names],
    })
    df_melt = df.melt(id_vars='feature', var_name='Class', value_name='Rule frequency')
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=df_melt, y='feature', x='Rule frequency', hue='Class',
        palette=(sns.color_palette('bright')[4], sns.color_palette('bright')[0], sns.color_palette('bright')[3]),
        edgecolor='black',
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%d', padding=4, fontsize=7)
    title = f'LORE Feature Frequency in Rules (top {k})'
    if mode:
        title += f' [{mode}]'
    if epoch_num is not None:
        title += f' – {epoch_num}'
    plt.title(title)
    plt.xlabel('Rule frequency (# instances)')
    plt.ylabel('')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_lore_rules_text(rules: list, epoch_num=None, mode: str = '', out_path=None) -> str:
    """Write per-instance LORE rules to a .txt file."""
    header = 'LORE Rules'
    if mode:
        header += f' [{mode}]'
    if epoch_num is not None:
        header += f' – {epoch_num}'
    lines = [header, '=' * len(header), '']

    def _fmt(subset, label):
        lines.append(f'=== {label} ({len(subset)} instances) ===')
        for i, r in subset:
            lines.append(f'[inst {i:02d}] pred={r["pred"]:.3f} label={r["label"]}')
            lines.append(f'  RULE: {r["rule_str"]}')
            for cr in r.get('crules_str', []):
                lines.append(f'  COUNTERFACTUAL: {cr}')
            lines.append('')

    pos = [(i, r) for i, r in enumerate(rules) if r['pred'] >= 0.5]
    neg = [(i, r) for i, r in enumerate(rules) if r['pred'] < 0.5]
    _fmt(pos, 'POSITIVE PREDICTIONS')
    _fmt(neg, 'NEGATIVE PREDICTIONS')

    text = "\n".join(lines)
    if out_path:
        with open(out_path, 'w') as f:
            f.write(text)
    return text


from xailib.explainers.lore_explainer import LoreTabularExplainer
from collections import defaultdict
from typing import Dict, List, Optional


class LoreEngine:
    """
    LORE on raw (pre-FeatureSpace) tabular features.
    Categoricals are label-encoded before LORE, decoded back to strings
    in the predictor before FeatureSpace.
    """

    def __init__(
        self,
        feature_space,
        feat_cols: List[str],
        max_explain_samples: int = 32,
        positive_threshold: float = 0.5,
        lore_config: Optional[Dict] = None,
        lore_random_state: int = 0,
        oos_samples: int = 1000,
    ):
        self.feature_space = feature_space
        self.feat_cols = feat_cols
        self.max_explain_samples = int(max_explain_samples)
        self.positive_threshold = float(positive_threshold)
        # NOTE: 'random_state' inside lore_config is SILENTLY IGNORED by the xailib wrapper --
        # LoreTabularExplainer.fit() passes random_state=self.random_state (a class attribute,
        # default 0) and never reads the config dict. It is set explicitly in _build_explainer.
        self.lore_random_state = int(lore_random_state)
        self.oos_samples = int(oos_samples)
        self.lore_config = lore_config or {
            'neigh_type': 'random',
            'size': 300,
            'verbose': False,
        }
        self._image_encoder = None
        self._head_model = None
        self._predict_fn = None
        # Label encoding maps — built per explain() call from actual data
        self._label_maps = {}   # col → {string: int}
        self._inv_maps = {}     # col → {int: string}
        self._cat_cols = []

    # ── Model splitting ──────────────────────────────────────────────────
    def _build_sub_models(self, model):
        if self._image_encoder is not None:
            return
        # Cut the graph right after the (expensive) image backbone, at the
        # GlobalAveragePooling2D output. This pooled image feature is the single
        # trunk through which *all* image-side computation flows, so the head can
        # rebuild any downstream fusion scheme (plain concat, gamma fusion,
        # learnable scaling, adaptive gating) from [pooled_image, features].
        # The backbone stays in image_encoder (run once, cached); the head holds
        # only the cheap dense/fusion layers, so predicting over the LORE
        # neighbourhood is fast and works for every architecture.
        gap_layers = [l for l in model.layers
                      if isinstance(l, keras.layers.GlobalAveragePooling2D)]
        if not gap_layers:
            raise RuntimeError(
                "LORE: no GlobalAveragePooling2D found in "
                f"'{model.name}' to split the image branch."
            )
        img_emb = gap_layers[0].output            # (None, C) pooled image feature
        feat_in = model.input["features"]
        self._image_encoder = keras.Model(
            inputs=model.input["images"], outputs=img_emb, name="image_encoder",
        )
        # Reuse the intermediate pooled tensor directly as a head input (Keras
        # functional API): architecture-agnostic, no hardcoded layer names.
        self._head_model = keras.Model(
            inputs=[img_emb, feat_in], outputs=model.output, name="head_model",
        )
        self._predict_fn = tf.function(
            lambda embs, enc: self._head_model([embs, enc], training=False),
            reduce_retracing=True,
        )

    # ── Label encoding ───────────────────────────────────────────────────
    def _label_encode_df(self, df):
        """Label-encode categorical columns to integers. Returns encoded copy."""
        # self._cat_cols = [c for c in df.columns if df[c].dtype == 'object']
        self._cat_cols = [c for c in df.columns if c != 'target'
                          and not pd.api.types.is_numeric_dtype(df[c])]
        self._label_maps, self._inv_maps = {}, {}
        out = df.copy()
        for col in self._cat_cols:
            uniq = sorted(df[col].dropna().unique())
            lm = {v: i for i, v in enumerate(uniq)}
            self._label_maps[col] = lm
            self._inv_maps[col] = {i: v for v, i in lm.items()}
            out[col] = df[col].map(lm).astype('float64')
        return out

    # ── Predictor ────────────────────────────────────────────────────────
    def _make_predictor(self, image_embedding, feat_cols_no_target):
        head = self._head_model
        fs = self.feature_space
        emb_const = tf.constant(image_embedding[np.newaxis], dtype=tf.float32)
        cat_cols = self._cat_cols
        inv_maps = self._inv_maps
        predict_fn = self._predict_fn

        def predict(X) -> np.ndarray:
            if isinstance(X, pd.DataFrame):
                df = X.copy()
                cols = [c for c in df.columns if c != 'target']
            else:
                arr = np.asarray(X, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                df = pd.DataFrame(arr, columns=feat_cols_no_target)
                cols = feat_cols_no_target

            # Decode label-encoded categoricals back to strings
            feat_dict = {}
            for col in cols:
                if col in cat_cols:
                    ints = df[col].values.round().astype(int).clip(0, len(inv_maps[col]) - 1)
                    strings = [inv_maps[col].get(i, inv_maps[col][0]) for i in ints]
                    feat_dict[col] = tf.constant(strings)
                else:
                    feat_dict[col] = tf.constant(df[col].values.astype(np.float32))

            encoded = fs(feat_dict)
            n = encoded.shape[0]
            embs = tf.repeat(emb_const, n, axis=0)
            return predict_fn(embs, encoded).numpy()[:, 0]

        return predict

    # ── LORE wrappers ────────────────────────────────────────────────────
    def _build_explainer(self, predictor, background_df):
        class _BB:
            def __init__(self, fn, thr):
                self.fn, self.thr = fn, thr
            def predict(self, X):
                return (self.fn(X) >= self.thr).astype(int)
            def predict_proba(self, X):
                p = self.fn(X)
                return np.column_stack([1 - p, p])

        bb = _BB(predictor, self.positive_threshold)
        df_bg = background_df.copy().reset_index(drop=True)
        df_bg['target'] = bb.predict(df_bg).astype(np.int64)
        present = set(df_bg['target'].dropna().astype(int).unique().tolist())
        missing = [cls for cls in (0, 1) if cls not in present]
        if missing and len(df_bg) > 0:
            base_row = df_bg.iloc[0].copy()
            filler_rows = []
            for cls in missing:
                row = base_row.copy()
                row['target'] = cls
                filler_rows.append(row)
            df_bg = pd.concat([df_bg, pd.DataFrame(filler_rows)], ignore_index=True)
        exp = LoreTabularExplainer(bb=bb)
        exp.random_state = self.lore_random_state   # FIX: the config key is ignored by the wrapper
        exp.fit(df_bg, 'target', self.lore_config)
        self._predictor = predictor                 # kept for the out-of-sample fidelity below
        return exp

    # def _explain_one(self, lore_exp, x_series):
    #     try:
    #         exp = lore_exp.explain(x_series.values)
    #         rule = exp.getRules()
    #         crules = exp.getCounterfactualRules()
    #         premise = rule.get('premise', [])
    #
    #         def fmt(p):
    #             att, op, thr = p['att'], p['op'], float(p['thr'])
    #             if att in self._cat_cols:
    #                 idx = int(round(thr))
    #                 idx = max(0, min(idx, len(self._inv_maps[att]) - 1))
    #                 label = self._inv_maps[att][idx]
    #                 readable_op = '=' if op in ('>', '>=') else '≠'
    #                 return f"{att} {readable_op} '{label}'"
    #             return f"{att} {op} {thr:.4g}"
    #
    #         rule_str = ' AND '.join(fmt(p) for p in premise)
    #         rule_features = [p['att'] for p in premise]
    #         crules_str = [
    #             ' AND '.join(fmt(p) for p in cr.get('premise', []))
    #             for cr in (crules or [])
    #         ]
    #         return {'rule_str': rule_str, 'rule_features': rule_features, 'crules_str': crules_str}
    #     except Exception as e:
    #         return {'rule_str': f'[error: {e}]', 'rule_features': [], 'crules_str': []}

    def _explain_one(self, lore_exp, x_series):
        try:
            exp = lore_exp.explain(x_series.values)
            rule = exp.getRules()
            crules = exp.getCounterfactualRules()
            premise = rule.get('premise', [])

            def fmt(p):
                att, op, thr = p['att'], p['op'], float(p['thr'])
                if att in self._cat_cols:
                    idx = int(round(thr))
                    idx = max(0, min(idx, len(self._inv_maps[att]) - 1))
                    label = self._inv_maps[att][idx]
                    readable_op = '=' if op in ('>', '>=') else '!='
                    return f"{att} {readable_op} '{label}'"
                return f"{att} {op} {thr:.4g}"

            rule_str = ' AND '.join(fmt(p) for p in premise)
            rule_features = [p['att'] for p in premise]
            crules_str = [
                ' AND '.join(fmt(p) for p in cr.get('premise', []))
                for cr in (crules or [])
            ]
            # ---- native fidelity: dt.score(Z, Yb) -- the surrogate scored on the SAME neighbourhood
            # it was fitted on, i.e. IN-SAMPLE. A decision tree evaluated on its own training data
            # saturates by construction: this is a second, independent reason for the ~0.97-0.98
            # ceiling (the first being n=6 explained samples).
            fid = exp.expDict.get('fidelity')

            # ---- repaired fidelity: redraw a FRESH neighbourhood with the same generator and
            # score the surrogate out-of-sample, on continuous outputs and balanced classes.
            oos = {'lore_fidelity_oos_acc': float('nan'), 'lore_fidelity_oos_bacc': float('nan'),
                   'lore_fidelity_oos_r2': float('nan'), 'neigh_pos_rate': float('nan')}
            try:
                from sklearn.metrics import balanced_accuracy_score, r2_score
                dt = exp.exp.dt                                             # the surrogate tree
                Z = lore_exp.lore_explainer.neighgen_fn(x_series.values, self.oos_samples)
                p_model = self._predictor(Z)                                # continuous model prob
                y_model = (p_model >= self.positive_threshold).astype(int)
                y_surr = dt.predict(Z)
                oos['lore_fidelity_oos_acc'] = float((y_surr == y_model).mean())
                oos['lore_fidelity_oos_r2'] = float(r2_score(p_model, dt.predict_proba(Z)[:, 1]))
                oos['neigh_pos_rate'] = float(y_model.mean())               # did Z cross the boundary?
                if len(set(y_model.tolist())) > 1:
                    oos['lore_fidelity_oos_bacc'] = float(balanced_accuracy_score(y_model, y_surr))
            except Exception as e:
                oos['oos_error'] = f'{type(e).__name__}: {e}'

            return {'rule_str': rule_str, 'rule_features': rule_features, 'crules_str': crules_str,
                    'fidelity': float(fid) if fid is not None else float('nan'),
                    'rule_len': len(premise), **oos}
        except Exception as e:
            return {'rule_str': f'[LORE skipped: {type(e).__name__}: {e}]', 'rule_features': [], 'crules_str': [],
                    'fidelity': float('nan'), 'rule_len': 0}

    # ── Public API ───────────────────────────────────────────────────────
    def explain(self, raw_df, images, labels, model, mode='context'):
        self._build_sub_models(model)

        # Label-encode categoricals → all-numeric for LORE
        encoded_df = self._label_encode_df(raw_df)
        feat_cols_no_target = [c for c in encoded_df.columns if c != 'target']

        n = len(encoded_df)
        ex_idx = np.arange(n)
        if n > self.max_explain_samples:
            ex_idx = np.sort(np.random.choice(n, self.max_explain_samples, replace=False))

        ex_df = encoded_df.iloc[ex_idx].reset_index(drop=True)
        ex_labels = labels[ex_idx].ravel()

        if mode == 'context':
            img = images.mean(axis=0, keepdims=True).astype(np.float32)
        elif mode == 'tabular_only':
            img = np.zeros((1, *images.shape[1:]), dtype=np.float32)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        image_embedding = self._image_encoder(img, training=False).numpy()[0]
        predictor = self._make_predictor(image_embedding, feat_cols_no_target)
        preds = predictor(ex_df)
        lore_exp = self._build_explainer(predictor, encoded_df)

        freq_all, freq_pos, freq_neg = defaultdict(int), defaultdict(int), defaultdict(int)
        rules = []

        for i in range(len(ex_df)):
            result = self._explain_one(lore_exp, ex_df.iloc[i])
            print(f"[LORE {i+1}/{len(ex_df)}] rule_features={result['rule_features']}")
            result['pred'] = float(preds[i])
            result['label'] = int(ex_labels[i])
            rules.append(result)
            for feat in result['rule_features']:
                raw_feat = feat.split("=")[0] if "=" in feat else feat
                freq_all[raw_feat] += 1
                (freq_pos if preds[i] >= self.positive_threshold else freq_neg)[raw_feat] += 1

        def _agg(key):
            v = [r[key] for r in rules
                 if isinstance(r.get(key), float) and not np.isnan(r[key])]
            return float(np.mean(v)) if v else float('nan')

        def _sem(key):
            v = [r[key] for r in rules
                 if isinstance(r.get(key), float) and not np.isnan(r[key])]
            return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float('nan')

        _lens = [r['rule_len'] for r in rules if r.get('rule_len')]
        return {
            'feature_freq_all': dict(freq_all),
            'feature_freq_pos': dict(freq_pos),
            'feature_freq_neg': dict(freq_neg),
            'explained_indices': ex_idx,
            'n_explained': len(rules),
            'fidelity': _agg('fidelity'),                       # native, IN-SAMPLE (saturates)
            'fidelity_sem': _sem('fidelity'),                   # <- with n=6 this did not exist
            'fidelity_oos_acc': _agg('lore_fidelity_oos_acc'),  # out-of-sample accuracy
            'fidelity_oos_bacc': _agg('lore_fidelity_oos_bacc'),# balanced (kills class inflation)
            'fidelity_oos_r2': _agg('lore_fidelity_oos_r2'),    # continuous (kills 0.5-threshold)
            'fidelity_oos_r2_sem': _sem('lore_fidelity_oos_r2'),
            'neigh_pos_rate': _agg('neigh_pos_rate'),
            'rule_len': float(np.mean(_lens)) if _lens else float('nan'),
        }, rules, preds


    from pathlib import Path


class LoreCallback(keras.callbacks.Callback):
    """
    Runs LoreEngine every N epochs on a frozen validation subset.
    Operates on RAW features (pre-FeatureSpace) — XAI-Lib handles
    categorical encoding internally.
    """

    def __init__(
        self,
        data_source: tf.data.Dataset,
        raw_feature_df: pd.DataFrame,
        feat_cols: List[str],
        feature_space,
        base_dir: str,
        max_explain_samples: int = 12,
        positive_threshold: float = 0.5,
        top_k: int = 10,
        every_n_epochs: int = 2,
        max_samples: int = 256,
        max_batches: Optional[int] = None,
        lore_config: Optional[Dict] = None,
        rng_seed: int = 0,
        modes: tuple = ('context',),
    ):
        super().__init__()
        self.data_source = data_source
        self.raw_feature_df = raw_feature_df  # pre-FeatureSpace validation DataFrame
        self.feat_cols = feat_cols
        self.top_k = int(top_k)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.max_samples = int(max_samples)
        self.max_batches = max_batches
        self.rng_seed = int(rng_seed)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self._images = None
        self._raw_df = None
        self._labels = None
        self.last_quality = {}   # aggregate LORE quality (fidelity); merged into run diagnostics

        self.modes = tuple(modes)
        self._engine = LoreEngine(
            feature_space=feature_space,
            feat_cols=feat_cols,
            max_explain_samples=int(max_explain_samples),
            positive_threshold=float(positive_threshold),
            lore_config=lore_config,
        )

    def _freeze_batch(self, ds: tf.data.Dataset):
        """Collect images and labels from dataset, raw features from DataFrame."""
        imgs, labels = [], []
        n = 0
        for b_idx, (x, y) in enumerate(ds):
            if self.max_batches and b_idx >= self.max_batches:
                break
            remaining = self.max_samples - n
            if remaining <= 0:
                break
            imgs.append(x['images'].numpy()[:remaining])
            labels.append(y.numpy()[:remaining])
            n += imgs[-1].shape[0]

        if n == 0:
            raise RuntimeError('No samples collected for LORE.')
        self._images = np.concatenate(imgs)
        self._labels = np.concatenate(labels)
        # Take corresponding rows from raw DataFrame (same order, no shuffle in val)
        self._raw_df = self.raw_feature_df.iloc[:n][self.feat_cols].reset_index(drop=True)

    def on_train_begin(self, logs=None):
        self._freeze_batch(self.data_source)

    def on_epoch_end(self, epoch: int, logs=None):
        if epoch % self.every_n_epochs != 0 or self._images is None:
            return

        np.random.seed(self.rng_seed)
        epoch_name = f'epoch{epoch:03d}'
        epoch_dir = self.base_dir / epoch_name
        epoch_dir.mkdir(parents=True, exist_ok=True)

        row = {'epoch': int(epoch)}
        # 'tabular_only' doubled the cost but never reached the report -- the published
        # lore_fidelity is the 'context' one. Dropping it pays for max_explain_samples 6 -> 32.
        for mode in self.modes:
            mode_dir = epoch_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)

            stats, rules, preds = self._engine.explain(
                raw_df=self._raw_df,
                images=self._images,
                labels=self._labels,
                model=self.model,
                mode=mode,
            )

            self.last_quality[f'lore_fidelity_{mode}'] = stats.get('fidelity')
            self.last_quality[f'lore_rule_len_{mode}'] = stats.get('rule_len')
            row[f'lore_fidelity_{mode}'] = stats.get('fidelity')

            plot_lore_feature_frequency(
                freq_all=stats['feature_freq_all'],
                freq_pos=stats['feature_freq_pos'],
                freq_neg=stats['feature_freq_neg'],
                k=self.top_k,
                epoch_num=epoch_name,
                mode=mode,
                out_path=str(mode_dir / f'lore_feature_freq_top{self.top_k:02d}_{mode}_{epoch_name}.png'),
            )

            save_lore_rules_text(
                rules=rules,
                epoch_num=epoch_name,
                mode=mode,
                out_path=str(mode_dir / f'lore_rules_{mode}_{epoch_name}.txt'),
            )

        # headline quality (context mode) + per-epoch trajectory CSV (one row/epoch when run each epoch)
        self.last_quality['lore_fidelity'] = self.last_quality.get('lore_fidelity_context')
        self.last_quality['lore_rule_len'] = self.last_quality.get('lore_rule_len_context')
        _csv = self.base_dir / 'lore_fidelity_per_epoch.csv'
        pd.DataFrame([row]).to_csv(_csv, mode='a', header=not _csv.exists(), index=False)


# ======================================================================================
# [notebook cell 39] LORE post-hoc fidelity loop -> runs/_faithfulness/lore_fidelity_posthoc.csv
# ======================================================================================


import numpy as np, pandas as pd, os, json, gc

# relmap is defined in the core utilities cell above.

# congela un batch di validation (immagini+label) + raw_df allineato (val non è shufflata)
N_BG = 128; imgs, labs, n = [], [], 0
for x, y in validation_ds:
  take = min(N_BG-n, x["images"].shape[0])
  imgs.append(x["images"].numpy()[:take]); labs.append(y.numpy()[:take]); n += take
  if n >= N_BG: break
images = np.concatenate(imgs); labels = np.concatenate(labs)
raw_df = validation_df.iloc[:n][FEAT_COLS].reset_index(drop=True)

engine = LoreEngine(feature_space=feature_space, feat_cols=FEAT_COLS,
                  max_explain_samples=12, positive_threshold=0.5,
                  lore_config={"neigh_type":"random","size":300,"random_state":42,"verbose":False})

TARGETS = {"vanilla":"20260604-210304", "controlled_fusion":"20260616-102253"}
rows = []
for impl, ts in TARGETS.items():
  comb = pd.read_csv(os.path.join(RUNS_PATH, impl, ts, "combined_summary.csv"))
  for _, r in comb.iterrows():
      rd = relmap(r["run_dir"]); seed = int(r["seed"])
      ck = os.path.join(rd, "checkpoints", f"best_{r['model_name']}.keras")
      model = keras.models.load_model(ck, compile=False, safe_mode=False)
      assert_fs_matches_model(model, feature_space, r["run_id"])
      np.random.seed(42); sc, _, _ = engine.explain(raw_df=raw_df, images=images, labels=labels, model=model,
  mode="context")
      np.random.seed(42); st, _, _ = engine.explain(raw_df=raw_df, images=images, labels=labels, model=model,
  mode="tabular_only")
      rec = {"impl_name":impl, "seed":seed, "run_dir":rd,
             "lore_fidelity_context":sc.get("fidelity"), "lore_rule_len_context":sc.get("rule_len"),
             "lore_fidelity_tabular_only":st.get("fidelity"), "lore_rule_len_tabular_only":st.get("rule_len"),
             "lore_fidelity":sc.get("fidelity"), "lore_rule_len":sc.get("rule_len")}
      rows.append(rec)
      dj = os.path.join(rd, "diagnostics.json")
      d = json.load(open(dj)) if os.path.exists(dj) else {}
      d.update({k:rec[k] for k in list(rec)[3:]})
      json.dump(d, open(dj,"w"), indent=2)
      print(f"{impl} seed {seed}: lore_fidelity={rec['lore_fidelity']:.4f}")
      keras.backend.clear_session(); gc.collect()

out = os.path.join(RUNS_PATH, "_faithfulness", "lore_fidelity_posthoc.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False)
print("scritto:", out, "-", len(rows), "righe")


# ======================================================================================
# [notebook cell 40] STAGE 0 - zero-GPU stats over master_results_rows.csv
# ======================================================================================

# ============================ STAGE 0 — zero GPU ============================
import numpy as np, pandas as pd, os
from scipy import stats

_rng = np.random.default_rng(0)
_d = pd.read_csv(os.path.join(SUMMARIES_PATH, "master_results_rows.csv"))
assert len(_d) == 110, f"expected the 110 canonical runs, got {len(_d)}"

REWEIGHT = ["fixed_scaling", "learnable_scaling", "loss_regularization"]   # the attendance KNOBS
FUSION   = ["controlled_fusion", "adaptive_gating"]                        # the fusion ARCHITECTURES
_d["mech"] = np.where(_d.impl_name.isin(FUSION), "fusion",
              np.where(_d.impl_name.isin(REWEIGHT), "reweight", "baseline"))
_d["img_reliance"] = 1.0 - _d["modality_balance"]

TAB, IMG, JNT = ("faithfulness_tabular_del_gain", "faithfulness_image_del_gain", "faithfulness_gain")
_out = []

# ---- 1. H1: does boosting a modality make ITS OWN explanation more faithful? ----
print("=" * 78); print("H1 TEST"); print("=" * 78)
for label, sub in [("all runs", _d),
                   ("reweighting only (the explicit knobs)", _d[_d.mech == "reweight"]),
                   ("reweighting, excl. degenerate alpha 0/1", _d[(_d.mech == "reweight") & (~_d.alpha.isin([0.0, 1.0]))])]:
    print(f"--- {label} (n={len(sub)})")
    for yname, y in [("image_gain", sub[IMG]), ("tabular_gain", sub[TAB])]:
        r, pr = stats.pearsonr(sub.img_reliance, y)
        print(f"    img_reliance <-> {yname:<13s} r={r:+.3f} (p={pr:.2g})")
        _out.append(dict(analysis="H1", subset=label, n=len(sub), y=yname, r=r, p=pr))
print("\nIf H1 held, img_reliance <-> image_gain would be POSITIVE and strong.\n")

# ---- 2. mechanism contrast: fusion architecture vs attendance reweighting ----
print("=" * 78); print("MECHANISM CONTRAST — fusion vs reweighting"); print("=" * 78)
_fus, _rest = _d[_d.mech == "fusion"], _d[_d.mech != "fusion"]
for comp, col in [("image", IMG), ("joint", JNT), ("tabular", TAB)]:
    a, b = _fus[col].values, _rest[col].values
    bs = np.array([_rng.choice(a, len(a)).mean() - _rng.choice(b, len(b)).mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    mwu = stats.mannwhitneyu(a, b).pvalue
    print(f"  {comp:<8s} fusion={a.mean():+.3f}  rest={b.mean():+.3f}  diff={a.mean()-b.mean():+.3f}"
          f"  CI[{lo:+.3f},{hi:+.3f}]  MWU p={mwu:.2g}")
    _out.append(dict(analysis="mech_contrast", component=comp, diff=a.mean()-b.mean(),
                     ci_lo=lo, ci_hi=hi, mwu_p=mwu))

# ---- 3. paired-by-seed (free power: all families share seeds 47/53/71/8/90) ----
print("\n" + "=" * 78); print("PAIRED BY SEED"); print("=" * 78)
_van = _d[_d.impl_name == "vanilla"].set_index("seed")
for fam in FUSION:
    f = _d[_d.impl_name == fam].set_index("seed")
    seeds = sorted(set(f.index) & set(_van.index))
    for comp, col in [("image", IMG), ("joint", JNT), ("tabular", TAB)]:
        diffs = f.loc[seeds, col].values - _van.loc[seeds, col].values
        tp = stats.ttest_rel(f.loc[seeds, col], _van.loc[seeds, col]).pvalue
        up = stats.ttest_ind(f.loc[seeds, col], _van.loc[seeds, col], equal_var=False).pvalue
        print(f"  {fam:<18s} {comp:<8s} diff={diffs.mean():+.3f}  paired p={tp:.3g}  (unpaired {up:.3g})")
        _out.append(dict(analysis="paired", family=fam, component=comp,
                         mean_diff=diffs.mean(), paired_p=tp, unpaired_p=up))
print("  NOTE: at n=5 Wilcoxon signed-rank CANNOT go below p=0.0625, whatever the effect size.")

# ---- 4. variance decomposition + hierarchical bootstrap (per-sample curves from disk) ----
def _curves(run_dir):
    return os.path.join(ART_PATH, run_dir.split("artifacts/enhanced/")[-1],
                        "xai", "faithfulness", "faithfulness_curves.npz")

_ps, _Q = {}, None
_rows = []
for _, r in _d.iterrows():
    z = np.load(_curves(r.run_dir)); _Q = z["q_grid"]
    rec = dict(run_id=r.run_id, impl_name=r.impl_name, setting_tag=r.setting_tag, mech=r.mech)
    for comp in ("tabular", "image", "joint"):
        g = (np.trapezoid(z[f"del_exp_{comp}"], _Q, axis=1)
             - np.trapezoid(z[f"del_rand_{comp}"], _Q, axis=1))   # per-sample gain
        _ps[(r.run_id, comp)] = g
        rec[f"{comp}_mean"] = g.mean()
        rec[f"{comp}_sem"] = g.std(ddof=1) / np.sqrt(len(g))
    _rows.append(rec)
_psdf = pd.DataFrame(_rows)

# reproduction anchor: per-sample mean MUST equal the published gain
_m = _d.merge(_psdf, on="run_id")
assert len(_m) == 110
for comp, col in [("tabular", TAB), ("image", IMG), ("joint", JNT)]:
    err = (_m[f"{comp}_mean"] - _m[col]).abs().max()
    assert err < 1e-9, f"{comp}: per-sample curves do not reproduce the published gain ({err:.2e})"
print("\nreproduction anchor OK — per-sample curves reproduce the published gains exactly")

print("\n" + "=" * 78); print("VARIANCE: seeds vs samples"); print("=" * 78)
print(f"  {'family':<20s} {'comp':<8s} {'sd(seed)':>10s} {'sem(128)':>10s}  dominant")
for fam, gf in _psdf.groupby("impl_name"):
    for comp in ("tabular", "image", "joint"):
        btw = gf.groupby("setting_tag")[f"{comp}_mean"].std(ddof=1).mean()  # across seeds, within setting
        wit = gf[f"{comp}_sem"].mean()
        print(f"  {fam:<20s} {comp:<8s} {btw:10.4f} {wit:10.4f}  "
              f"{'SEED' if btw > wit else 'SAMPLES'} ({btw/wit:.1f}x)")
        _out.append(dict(analysis="variance", family=fam, component=comp,
                         sd_seed=btw, sem_samples=wit, ratio=btw/wit))
print("  -> if SEED dominates, the 128-sample batch is NOT the binding constraint; the 5 seeds are.")

print("\n" + "=" * 78); print("HIERARCHICAL BOOTSTRAP (resample seeds, then samples)"); print("=" * 78)
for comp in ("image", "joint", "tabular"):
    fk = [r.run_id for _, r in _d.iterrows() if r.mech == "fusion"]
    rk = [r.run_id for _, r in _d.iterrows() if r.mech != "fusion"]
    def _draw(keys):
        ks = [keys[i] for i in _rng.integers(0, len(keys), len(keys))]
        return np.mean([_ps[(k, comp)][_rng.integers(0, 128, 128)].mean() for k in ks])
    bs = np.array([_draw(fk) - _draw(rk) for _ in range(2000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pt = np.mean([_ps[(k, comp)].mean() for k in fk]) - np.mean([_ps[(k, comp)].mean() for k in rk])
    print(f"  {comp:<8s} diff={pt:+.3f}  CI[{lo:+.3f},{hi:+.3f}]  "
          f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'includes 0'}")
    _out.append(dict(analysis="hier_boot", component=comp, diff=pt, ci_lo=lo, ci_hi=hi))

# ── ALL Stage-0 output lives under runs/_robustness/stage0/ ──────────────────────────
_S0_DIR = os.path.join(RUNS_PATH, "_robustness", "stage0")
os.makedirs(_S0_DIR, exist_ok=True)
_p = os.path.join(_S0_DIR, "stage0_stats.csv")
pd.DataFrame(_out).to_csv(_p, index=False)
print(f"\nsaved -> {_p}")


# ======================================================================================
# [notebook cell 41] genetic-neighbourhood LORE smoke test
# ======================================================================================

# %load artifacts/_tmp_stage1_genetic_test.py
# ============================================================================
# QUICK TEST — does a GENETIC neighbourhood rescue the degenerate multimodal
# LORE fidelity, and how much does it cost per sample?
#
# Run in the SAME kernel that just ran Stage-1 Phase A (needs: LoreEngine,
# feature_space, FEAT_COLS, _build_faith_subset, ART_PATH, SUMMARIES_PATH,
# and the '[XAI] registered custom layers:' cell already executed).
# ============================================================================
import glob, gc, time, numpy as np, pandas as pd, os, keras

# ---- cost knobs: start SMALL. If seed_53 already answers it, Ctrl-C. --------
NEIGH_TYPE = "genetic"   # 'genetic' = fitness on hard labels (targets opposite class directly).
                         # try 'geneticp' (proba-based, smoother) if 'genetic' underperforms.
TEST_N     = 4           # explained samples / run (small = fast)
TEST_OOS   = 300         # fresh-neighbourhood size for oos fidelity (was 500 under random)
TEST_NGEN  = 10          # GA generations. Lower (e.g. 5) ~halves cost; 10 = library default.
TEST_RUNS  = [
    "20260604-210304/seed_53__baseline__e20",   # vanilla, DEGENERATE under random -> the decisive test
    "20260604-210304/seed_47__baseline__e20",   # vanilla, already healthy under random -> sanity control
]

# random baselines for reference (from lore_fidelity_v2.csv)
_BASE = {
    "20260604-210304/seed_53__baseline__e20": dict(pos=0.020, bacc=0.540, ins=0.9926),
    "20260604-210304/seed_47__baseline__e20": dict(pos=0.571, bacc=0.642, ins=0.9827),
}

_rows = pd.read_csv(os.path.join(SUMMARIES_PATH, "master_results_rows.csv"))
_rows = _rows[_rows.run_id.isin(TEST_RUNS)].reset_index(drop=True)

print(f"GENETIC LORE TEST — neigh_type={NEIGH_TYPE}  n={TEST_N}  oos={TEST_OOS}  ngen={TEST_NGEN}\n")
for _, r in _rows.iterrows():
    rd = os.path.join(ART_PATH, r.run_dir.split("artifacts/enhanced/")[-1])
    ck = glob.glob(os.path.join(rd, "checkpoints", "best_*.keras"))
    if not ck:
        print(f"[SKIP] {r.run_id} (no checkpoint)"); continue
    try:
        model = keras.models.load_model(ck[0], compile=False, safe_mode=False)
        eng = LoreEngine(feature_space=feature_space, feat_cols=FEAT_COLS,
                         max_explain_samples=TEST_N, positive_threshold=0.5,
                         lore_random_state=0, oos_samples=TEST_OOS,
                         lore_config={"neigh_type": NEIGH_TYPE, "size": 300,
                                      "ngen": TEST_NGEN, "verbose": False})
        raw_df, imgs, labs = _build_faith_subset(128)
        t0 = time.time()
        stats_, rules, preds = eng.explain(raw_df=raw_df, images=imgs, labels=labs,
                                           model=model, mode="context")
        dt = (time.time() - t0) / 60.0
        n  = max(int(stats_["n_explained"]), 1)
        b  = _BASE.get(r.run_id, {})
        print(f"{r.run_id}")
        print(f"   RANDOM  (ref): pos_rate={b.get('pos', float('nan')):.3f}  "
              f"oos_bacc={b.get('bacc', float('nan')):.3f}  insample={b.get('ins', float('nan')):.4f}")
        print(f"   GENETIC (now): pos_rate={stats_['neigh_pos_rate']:.3f}  "
              f"oos_bacc={stats_['fidelity_oos_bacc']:.3f}  insample={stats_['fidelity']:.4f}"
              f"   [{dt:.1f} min / {n} samples -> {dt/n:.1f} min/sample]", flush=True)
    except Exception as e:
        print(f"[FAIL] {r.run_id}: {type(e).__name__}: {e}", flush=True)
    finally:
        keras.backend.clear_session(); gc.collect()

print("\n" + "=" * 74)
print("HOW TO READ IT")
print("=" * 74)
print(" seed_53 (random pos_rate 0.020, bacc 0.540) is the decisive one:")
print("  - if GENETIC lifts pos_rate to ~0.3-0.5 AND bacc rises clearly (>~0.6)")
print("    => the boundary-crossing neighbourhood FIXES the confound => a scoped genetic")
print("       rerun buys a REAL cross-strategy LORE ranking (a genuinely new result). Worth it.")
print("  - if pos_rate stays ~0.02 => the model is genuinely tabular-invariant on these samples;")
print("    genetic cannot manufacture positives => LORE conclusion stands, and we upgrade")
print("    'plausibly structural' -> 'structural' (rules out the base-rate competing story).")
print(" seed_47 is the control: already healthy under random; genetic must NOT break it.")
print(" min/sample above x (16 samples x N runs) = price of the full rerun. Decide from that.")


# ======================================================================================
# [notebook cell 42] STAGE 1 - LORE repaired (oos bacc, neigh_pos_rate)
# ======================================================================================

# ======================= STAGE 1 — LORE repaired =======================
import glob, gc, time, numpy as np, pandas as pd, os, keras

# --- GUARD: the custom-layer registration cell MUST have run in this kernel ---------
# Without it, every fixed_scaling / learnable_scaling / controlled_fusion / adaptive_gating
# checkpoint fails to deserialize and the loop silently [FAIL]s 70+ runs while looking busy.
_need = ["modality_attendance>ConstantScale", "modality_attendance>LearnableAlpha",
         "modality_attendance>ScalarMultiply", "modality_attendance>FusionWithGamma",
         "modality_attendance>GateEntropyLoss"]
_missing = [n for n in _need if n not in keras.saving.get_custom_objects()]
if _missing:
    raise RuntimeError(
        "Custom layers NOT registered: " + ", ".join(m.split(">")[1] for m in _missing) +
        "\n-> run the cell that prints '[XAI] registered custom layers:' first.")

LORE2_N           = 16      # explained samples/run. Published n=6 had no SEM; measured cost ~1.9 min/sample.
LORE2_OOS_SAMPLES = 500     # fresh neighbourhood for out-of-sample fidelity
LORE2_OVERWRITE   = False   # False = resume: skip run_ids already in the CSV

# ── STAGED DESIGN: the full 110-run sweep is ~110 h at n=32. Budget is 12 h. ──────────
# Phase A is a GATE, not a survey. A tabular explainer MUST be vacuous on a model that ignores
# the table (a0p000, image-only) and informative on one that is nothing but the table (a1p000).
# If LORE fidelity cannot separate those two, the metric is dead and NO parameter change saves it
# -- and we stop, having spent 8 h instead of 110. Only if the gate opens do we pay for Phase B.
LORE2_PHASE = "A"

_PHASES = {
    # 15 runs x ~30 min = ~8 h   <- fits the budget, and is decisive
    "A": {"vanilla":             ["baseline"],       # reference
          "fixed_scaling":       ["a0p000",          # image-only : LORE should be VACUOUS here
                                  "a1p000"]},        # tabular-only: LORE should be INFORMATIVE here
    # 25 runs x ~30 min = ~12 h  <- only if A opens the gate. Resume makes this additive.
    "B": {"fixed_scaling":       ["a0p500"],
          "learnable_scaling":   ["init_a0p500"],
          "loss_regularization": ["lam1e-03"],
          "controlled_fusion":   ["base"],
          "adaptive_gating":     ["base"]},
}
LORE2_SETTINGS = _PHASES[LORE2_PHASE]

# ── ALL Stage-1 output lives under runs/_robustness/stage1_lore/ ──────────────────────
_S1_DIR = os.path.join(RUNS_PATH, "_robustness", "stage1_lore")
os.makedirs(_S1_DIR, exist_ok=True)
_OUT = os.path.join(_S1_DIR, "lore_fidelity_v2.csv")
if LORE2_OVERWRITE and os.path.exists(_OUT):
    os.remove(_OUT)
_done = set(pd.read_csv(_OUT).run_id) if os.path.exists(_OUT) else set()

_rows = pd.read_csv(os.path.join(SUMMARIES_PATH, "master_results_rows.csv"))
_rows = _rows[[t in LORE2_SETTINGS.get(f, []) for f, t in zip(_rows.impl_name, _rows.setting_tag)]]
_rows = _rows[~_rows.run_id.isin(_done)].reset_index(drop=True)
print(f"PHASE {LORE2_PHASE}: {len(_rows)} runs to do ({len(_done)} already in "
      f"{os.path.basename(_OUT)}), ~{LORE2_N * 2} min/run -> ~{len(_rows) * LORE2_N * 2 / 60:.1f} h\n")

_t0 = time.time()
for i, r in _rows.iterrows():
    rd = os.path.join(ART_PATH, r.run_dir.split("artifacts/enhanced/")[-1])
    ck = glob.glob(os.path.join(rd, "checkpoints", "best_*.keras"))
    if not ck:
        print(f"[SKIP] {r.run_id}"); continue
    try:
        model = keras.models.load_model(ck[0], compile=False, safe_mode=False)
        eng = LoreEngine(feature_space=feature_space, feat_cols=FEAT_COLS,
                         max_explain_samples=LORE2_N, positive_threshold=0.5,
                         lore_random_state=0, oos_samples=LORE2_OOS_SAMPLES)
        raw_df, imgs, labs = _build_faith_subset(128)
        stats_, rules, preds = eng.explain(raw_df=raw_df, images=imgs, labels=labs,
                                           model=model, mode="context")
        row = {
            "impl_name": r.impl_name, "run_id": r.run_id, "seed": r.seed,
            "setting_tag": r.setting_tag, "n_explained": stats_["n_explained"],
            "fidelity_insample": stats_["fidelity"],          # the old, saturating metric
            "fidelity_sem": stats_["fidelity_sem"],           # <- did not exist at n=6
            "fidelity_oos_acc": stats_["fidelity_oos_acc"],
            "fidelity_oos_bacc": stats_["fidelity_oos_bacc"], # <- PRIMARY (kills class inflation)
            "fidelity_oos_r2": stats_["fidelity_oos_r2"],     # <- fragile, see note below
            "fidelity_oos_r2_sem": stats_["fidelity_oos_r2_sem"],
            "neigh_pos_rate": stats_["neigh_pos_rate"],       # <- DIAGNOSTIC, read this first
            "rule_len": stats_["rule_len"],
            "modality_balance": r.modality_balance,
        }
        # incremental append: a kill now costs ONE run, not the whole loop
        pd.DataFrame([row]).to_csv(_OUT, mode="a", header=not os.path.exists(_OUT), index=False)
        el = time.time() - _t0
        print(f"[{i+1}/{len(_rows)}] {r.run_id}  insample={row['fidelity_insample']:.4f} "
              f"oos_bacc={row['fidelity_oos_bacc']:.4f} oos_r2={row['fidelity_oos_r2']:.2f} "
              f"pos_rate={row['neigh_pos_rate']:.3f}  "
              f"({el/60:.0f} min, ETA {el/(i+1)*(len(_rows)-i-1)/60:.0f} min)", flush=True)
    except Exception as e:
        print(f"[FAIL] {r.run_id}: {type(e).__name__}: {e}", flush=True)
    finally:
        keras.backend.clear_session(); gc.collect()

# ============================== READ-OUT ==============================
_l = pd.read_csv(_OUT)

print("\n" + "=" * 78)
print("STEP 1 — IS THE NEIGHBOURHOOD DEGENERATE?  (read this BEFORE the fidelity numbers)")
print("=" * 78)
print("neigh_pos_rate = fraction of the fresh neighbourhood the MODEL calls positive.")
print("If it sits at ~0 or ~1, the neighbourhood never crosses the decision boundary: then a")
print("low oos_bacc (~0.5) and a very negative oos_r2 are re-confirming THAT degeneracy, not")
print("showing 'the surrogate does not generalize'. The fix would be the neighbourhood GENERATOR")
print("(genetic / class-balanced), not the fidelity metric.")
print()
print(_l.groupby("impl_name")["neigh_pos_rate"].agg(["mean", "min", "max"]).round(3).to_string())

print("\n" + "=" * 78)
print("STEP 2 — DOES LORE FIDELITY DISCRIMINATE BETWEEN STRATEGIES NOW?")
print("=" * 78)
print(_l.groupby("impl_name")[["fidelity_insample", "fidelity_sem", "fidelity_oos_acc",
                               "fidelity_oos_bacc", "fidelity_oos_r2", "neigh_pos_rate"]]
      .mean().round(4).to_string())
print("\nSpread across families (a dead metric has ~0 spread):")
for c in ("fidelity_insample", "fidelity_oos_acc", "fidelity_oos_bacc"):
    fam = _l.groupby("impl_name")[c].mean()
    print(f"  {c:<20s} range={fam.max()-fam.min():.4f}   sd={fam.std():.4f}")
print("  (oos_r2 deliberately omitted here: its denominator is the variance of the model prob")
print("   INSIDE the neighbourhood, so it explodes negative whenever that variance is small.")
print("   Report balanced accuracy; treat r2 as directional only.)")

print("\n" + "=" * 78)
print("STEP 3 — THE ACID TEST: two deliberately opposite models")
print("=" * 78)
_ext = _l[_l.setting_tag.isin(["a0p000", "a1p000"])]
if len(_ext):
    print(_ext.groupby("setting_tag")[["fidelity_insample", "fidelity_oos_bacc",
                                       "neigh_pos_rate"]].agg(["mean", "std"]).round(4).to_string())
    print("  a0p000 = image-only (ignores the table); a1p000 = tabular-only.")
    print("  A tabular explainer MUST be vacuous on a0p000 and informative on a1p000.")
    print("  If it cannot tell THESE apart, LORE fidelity is dead and no parameter fixes it.")
else:
    print("  [missing — the fixed_scaling runs did not load. Check the registration guard.]")


# ======================================================================================
# [notebook cell 44] STAGE 2 - aligned Permutation attributor (STILL CITED: REPORT_S sec.3.3)
# ======================================================================================

# ================= STAGE 2 — aligned attributor (the decisive test) =================
import shap, glob, gc, time, numpy as np, pandas as pd, os, keras, tensorflow as tf

# --- GUARD: the custom-layer registration cell MUST have run in this kernel ---------
# Without it, every fixed_scaling / learnable_scaling / controlled_fusion / adaptive_gating
# checkpoint fails to deserialize and the loop silently [FAIL]s 70+ runs while looking busy.
_need = ["modality_attendance>ConstantScale", "modality_attendance>LearnableAlpha",
         "modality_attendance>ScalarMultiply", "modality_attendance>FusionWithGamma",
         "modality_attendance>GateEntropyLoss"]
_missing = [n for n in _need if n not in keras.saving.get_custom_objects()]
if _missing:
    raise RuntimeError(
        "Custom layers NOT registered: " + ", ".join(m.split(">")[1] for m in _missing) +
        "\n-> run the cell that prints '[XAI] registered custom layers:' first.")


S2_BG_SIZE   = 100     # background rows drawn from TRAINING (disjoint from what we explain)
S2_MAX_EVALS = 2 * len(FEAT_COLS) + 1     # Permutation: exact antithetic pass over 39 features
S2_LIMIT     = None    # set to an int to smoke-test first

_bg_raw = training_df[FEAT_COLS].sample(S2_BG_SIZE, random_state=0).reset_index(drop=True)


# ── label encoding: SHAP needs an all-numeric matrix ──────────────────────────────────
# shap.explainers.Permutation does arithmetic on the background matrix, so an object array holding
# the 6 string columns blows up with "unsupported operand type(s) for -: 'str' and 'str'".
# Encode the categoricals as integer codes for SHAP, decode them back to strings inside f() right
# before the FeatureSpace -- the same trick LoreEngine already uses.
# SOUND for Permutation specifically: the explainer only ever SUBSTITUTES whole values taken from
# the background rows, it never interpolates between them, so the integer code is an opaque label
# and no arithmetic meaning is imputed to it.
_S2_CAT = [c for c in FEAT_COLS if not pd.api.types.is_numeric_dtype(training_df[c])]
_S2_LMAP, _S2_IMAP = {}, {}
for _c in _S2_CAT:
    # vocabulary as STRINGS, matching what _to_feature_dict feeds the FeatureSpace (NaN included:
    # it becomes the literal level "nan" there too)
    _lv = sorted(pd.concat([training_df[_c], _bg_raw[_c]]).astype(str).unique())
    _S2_LMAP[_c] = {v: i for i, v in enumerate(_lv)}
    _S2_IMAP[_c] = {i: v for v, i in _S2_LMAP[_c].items()}
print(f"[S2] label-encoded {len(_S2_CAT)} categorical cols: "
      f"{ {c: len(_S2_LMAP[c]) for c in _S2_CAT} }")


def _s2_encode(df):
    """raw DataFrame -> float matrix SHAP can do arithmetic on."""
    out = df[FEAT_COLS].copy()
    for c in FEAT_COLS:
        if c in _S2_CAT:
            out[c] = out[c].astype(str).map(_S2_LMAP[c]).fillna(0)
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.to_numpy(dtype=np.float64)


def _s2_decode(X):
    """float matrix -> raw DataFrame with the dtypes the FeatureSpace expects."""
    df = pd.DataFrame(np.atleast_2d(np.asarray(X, dtype=np.float64)), columns=FEAT_COLS)
    for c in FEAT_COLS:
        if c in _S2_CAT:
            idx = df[c].round().fillna(0).astype(int).clip(0, len(_S2_IMAP[c]) - 1)
            df[c] = [_S2_IMAP[c][i] for i in idx]
        else:
            df[c] = df[c].astype("float32")
    return df


def aligned_tab_importance(model, raw_df, imgs):
    """Per-sample |attribution| over the 39 RAW features, from a Permutation explainer that
    scores the model head with THIS sample's image embedding held fixed -- i.e. exactly the
    function the tabular deletion perturbs."""
    gap = [l for l in model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D)]
    img_emb, feat_in = gap[0].output, model.input["features"]
    img_enc = keras.Model(model.input["images"], img_emb)
    head = keras.Model([img_emb, feat_in], model.output)
    head_fn = tf.function(lambda e, f: head([e, f], training=False), reduce_retracing=True)

    embs = img_enc.predict(imgs, batch_size=32, verbose=0)      # backbone runs ONCE per sample

    def make_f(emb_i):
        e = tf.constant(emb_i[np.newaxis], dtype=tf.float32)
        def f(X):
            enc = feature_space(_to_feature_dict(_s2_decode(X)))
            n = int(enc.shape[0])
            return head_fn(tf.repeat(e, n, axis=0), enc).numpy()[:, 0]
        return f

    X  = _s2_encode(raw_df)
    BG = _s2_encode(_bg_raw)                                     # background = TRAINING rows
    imp = np.zeros((len(raw_df), len(FEAT_COLS)))
    for i in range(len(raw_df)):
        ex = shap.explainers.Permutation(make_f(embs[i]), BG)
        sv = ex(X[i:i + 1], max_evals=S2_MAX_EVALS, silent=True)
        imp[i] = np.abs(np.asarray(sv.values).reshape(-1))
    return imp


# ── ALL Stage-2 output lives under runs/_robustness/stage2_aligned_attr/ ──────────────
# Nothing is written into runs/<run_id>/, so these checks stay physically separate from the
# published artifacts and can be `rm -rf`'d without touching a single canonical file.
_S2_DIR = os.path.join(RUNS_PATH, "_robustness", "stage2_aligned_attr")
os.makedirs(_S2_DIR, exist_ok=True)
_OUT2 = os.path.join(_S2_DIR, "faithfulness_aligned_attr.csv")
_done2 = set(pd.read_csv(_OUT2).run_id) if os.path.exists(_OUT2) else set()

_rows = pd.read_csv(os.path.join(SUMMARIES_PATH, "master_results_rows.csv"))
_rows = _rows[~_rows.run_id.isin(_done2)]
if S2_LIMIT:
    _rows = _rows.head(S2_LIMIT)
print(f"{len(_rows)} runs to do ({len(_done2)} already done)")

_t0 = time.time()
for i, r in _rows.reset_index(drop=True).iterrows():
    rd = os.path.join(ART_PATH, r.run_dir.split("artifacts/enhanced/")[-1])
    ck = glob.glob(os.path.join(rd, "checkpoints", "best_*.keras"))
    try:
        model = keras.models.load_model(ck[0], compile=False, safe_mode=False)
        raw_df, imgs, labs = _build_faith_subset(FAITH_N_SAMPLES)
        imp = aligned_tab_importance(model, raw_df, imgs)
        out = explanation_faithfulness(
            model, rd, int(r.best_epoch),
            methods=("tabular",),          # only the tabular ranking changed
            tab_imp_override=imp,
            out_name="faithfulness_aligned_attr",
            out_dir=os.path.join(_S2_DIR, str(r.run_id)),   # NOT runs/<run_id>/xai/
            save=True, verbose=False,
        )
        row = {"impl_name": r.impl_name, "run_id": r.run_id, "seed": r.seed,
               "setting_tag": r.setting_tag,
               "tab_gain_aligned": out["faithfulness_tabular_del_gain"],
               "tab_gain_published": r.faithfulness_tabular_del_gain}
        pd.DataFrame([row]).to_csv(_OUT2, mode="a", header=not os.path.exists(_OUT2), index=False)
        el = time.time() - _t0
        print(f"[{i+1}/{len(_rows)}] {r.run_id}  aligned={out['faithfulness_tabular_del_gain']:+.4f}  "
              f"published={r.faithfulness_tabular_del_gain:+.4f}   "
              f"({el/60:.0f} min, ETA {el/(i+1)*(len(_rows)-i-1)/60:.0f} min)", flush=True)
    except Exception as e:
        print(f"[FAIL] {r.run_id}: {type(e).__name__}: {e}")
    finally:
        keras.backend.clear_session(); gc.collect()

_a = pd.read_csv(_OUT2)

print("\n" + "=" * 78)
print("DOES THE TABULAR ANTI-FAITHFULNESS SURVIVE A COMPLETELY DIFFERENT ATTRIBUTOR?")
print("=" * 78)
print(_a.groupby("impl_name")[["tab_gain_published", "tab_gain_aligned"]].mean().round(4).to_string())
_n = (_a.tab_gain_aligned < 0).sum()
print(f"\nstill NEGATIVE (anti-faithful) in {_n}/{len(_a)} runs")
print(f"mean published = {_a.tab_gain_published.mean():+.4f}")
print(f"mean aligned   = {_a.tab_gain_aligned.mean():+.4f}")
print("\nSURVIVES  -> not an explainer artifact. Claim strengthened, BUT redundancy remains untested.")
print("FLIPS     -> the study's central claim is an artifact. Ablate one factor at a time.")


# ======================================================================================
# [notebook cell 47] STAGE 2b - control: original SHAP, current FeatureSpace
# ======================================================================================

# ============ STAGE 2b — control: SHAP originale, FeatureSpace attuale ============
import glob, gc, time, numpy as np, pandas as pd, os, keras

# --- GUARD: the custom-layer registration cell MUST have run in this kernel ---------
_need = ["modality_attendance>ConstantScale", "modality_attendance>LearnableAlpha",
         "modality_attendance>ScalarMultiply", "modality_attendance>FusionWithGamma",
         "modality_attendance>GateEntropyLoss"]
_missing = [n for n in _need if n not in keras.saving.get_custom_objects()]
if _missing:
    raise RuntimeError(
        "Custom layers NOT registered: " + ", ".join(m.split(">")[1] for m in _missing) +
        "\n-> run the cell that prints '[XAI] registered custom layers:' first.")

S2B_LIMIT = None

# ── ALL Stage-2b output lives under runs/_robustness/stage2b_control/ ─────────────────
_S2B_DIR = os.path.join(RUNS_PATH, "_robustness", "stage2b_control")
os.makedirs(_S2B_DIR, exist_ok=True)
_OUT2B = os.path.join(_S2B_DIR, "faithfulness_control_fs.csv")
_done2b = set(pd.read_csv(_OUT2B).run_id) if os.path.exists(_OUT2B) else set()

_rows = pd.read_csv(os.path.join(SUMMARIES_PATH, "master_results_rows.csv"))
_rows = _rows[~_rows.run_id.isin(_done2b)]
if S2B_LIMIT:
    _rows = _rows.head(S2B_LIMIT)
print(f"{len(_rows)} runs to do ({len(_done2b)} already done), ~36 s/run\n")

_t0 = time.time()
for i, r in _rows.reset_index(drop=True).iterrows():
    rd = os.path.join(ART_PATH, r.run_dir.split("artifacts/enhanced/")[-1])
    ck = glob.glob(os.path.join(rd, "checkpoints", "best_*.keras"))
    try:
        model = keras.models.load_model(ck[0], compile=False, safe_mode=False)
        # NO tab_imp_override -> explanation_faithfulness loads the |SHAP| arrays saved by
        # ShapCallback at train time. Same ranking as the published run; only the FeatureSpace
        # (and thus the model's predictions on perturbed rows) differs.
        out = explanation_faithfulness(
            model, rd, int(r.best_epoch),
            methods=("tabular",),
            out_name="faithfulness_control_fs",
            out_dir=os.path.join(_S2B_DIR, str(r.run_id)),   # NOT runs/<run_id>/xai/
            save=True, verbose=False,
        )
        row = {"impl_name": r.impl_name, "run_id": r.run_id, "seed": r.seed,
               "setting_tag": r.setting_tag,
               "tab_gain_control": out["faithfulness_tabular_del_gain"],
               "tab_gain_published": r.faithfulness_tabular_del_gain,
               "modality_balance": r.modality_balance}
        pd.DataFrame([row]).to_csv(_OUT2B, mode="a", header=not os.path.exists(_OUT2B), index=False)
        el = time.time() - _t0
        print(f"[{i+1}/{len(_rows)}] {r.run_id}  control={row['tab_gain_control']:+.4f}  "
              f"published={row['tab_gain_published']:+.4f}   "
              f"({el/60:.0f} min, ETA {el/(i+1)*(len(_rows)-i-1)/60:.0f} min)", flush=True)
    except Exception as e:
        print(f"[FAIL] {r.run_id}: {type(e).__name__}: {e}", flush=True)
    finally:
        keras.backend.clear_session(); gc.collect()

# ============================== THE 2x2 READ-OUT ==============================
from scipy import stats

_c = pd.read_csv(_OUT2B)
_apath = os.path.join(RUNS_PATH, "_robustness", "stage2_aligned_attr",
                      "faithfulness_aligned_attr.csv")
_a = pd.read_csv(_apath) if os.path.exists(_apath) else None

print("\n" + "=" * 78)
print("A. FEATURESPACE DRIFT   (published vs control -- same SHAP ranking, different FeatureSpace)")
print("=" * 78)
_r, _p = stats.pearsonr(_c.tab_gain_published, _c.tab_gain_control)
_rho, _ps = stats.spearmanr(_c.tab_gain_published, _c.tab_gain_control)
_d = (_c.tab_gain_control - _c.tab_gain_published)
print(f"  n={len(_c)}   pearson r={_r:+.3f} (p={_p:.2g})   spearman rho={_rho:+.3f} (p={_ps:.2g})")
print(f"  mean published={_c.tab_gain_published.mean():+.4f}   mean control={_c.tab_gain_control.mean():+.4f}")
print(f"  mean |difference| = {_d.abs().mean():.4f}   max |difference| = {_d.abs().max():.4f}")
print("  -> r near 1 means the FeatureSpace drift is NOT what moves the numbers.")

if _a is not None:
    _m = _c.merge(_a[["run_id", "tab_gain_aligned"]], on="run_id")
    print("\n" + "=" * 78)
    print("B. ATTRIBUTOR EFFECT   (control vs aligned -- SAME FeatureSpace, different attributor)")
    print("=" * 78)
    _r2, _p2 = stats.pearsonr(_m.tab_gain_control, _m.tab_gain_aligned)
    _rho2, _ps2 = stats.spearmanr(_m.tab_gain_control, _m.tab_gain_aligned)
    print(f"  n={len(_m)}   pearson r={_r2:+.3f} (p={_p2:.2g})   spearman rho={_rho2:+.3f} (p={_ps2:.2g})")
    print(f"  mean control={_m.tab_gain_control.mean():+.4f}   mean aligned={_m.tab_gain_aligned.mean():+.4f}")
    print(f"  anti-faithful (<0):  control {(_m.tab_gain_control < 0).sum()}/{len(_m)}   "
          f"aligned {(_m.tab_gain_aligned < 0).sum()}/{len(_m)}")

    print("\n" + "=" * 78)
    print("C. THE CLAIM THAT DEPENDS ON THE RANKING")
    print("=" * 78)
    print("Published report: modality_balance <-> tabular gain, rho = -0.63 (p~1e-13), i.e. 'the more")
    print("a model leans on the tabular branch, the more anti-faithful its explanation'. That is a")
    print("claim about the RANKING, so it only survives if it survives in all three columns:")
    for _lab, _col in [("published", "tab_gain_published"), ("control", "tab_gain_control"),
                       ("aligned", "tab_gain_aligned")]:
        _rr, _pp = stats.spearmanr(_m.modality_balance, _m[_col])
        _flag = "HOLDS" if (_pp < 0.05 and _rr < 0) else "DOES NOT HOLD"
        print(f"  {_lab:<10s} rho={_rr:+.3f} (p={_pp:.2g})   {_flag}")

    print("\n" + "=" * 78)
    print("D. PER-FAMILY MEANS (all three columns)")
    print("=" * 78)
    print(_m.groupby("impl_name")[["tab_gain_published", "tab_gain_control",
                                   "tab_gain_aligned"]].mean().round(4).to_string())
    print("\nVERDICT TEMPLATE:")
    print("  sign anti-faithful in all three  -> robust to BOTH confounders. State it.")
    print("  A: r~1 but B: r low             -> the ATTRIBUTOR moves the ranking, not the FeatureSpace.")
    print("     Then any rho-based trade-off claim must be re-stated on the aligned column, or dropped.")
else:
    print("\n[Stage 2 CSV not found -- run the aligned-attributor cell first for the B/C/D blocks]")


# ======================================================================================
# [notebook cell 50] STAGE 3 - image robustness: disjoint subsets + sigma sweep (STILL CITED)
# ======================================================================================

# ===================== STAGE 3 — image robustness =====================
import glob, gc, time, numpy as np, pandas as pd, os, keras

# --- GUARD: the custom-layer registration cell MUST have run in this kernel ---------
# Without it, every fixed_scaling / learnable_scaling / controlled_fusion / adaptive_gating
# checkpoint fails to deserialize and the loop silently [FAIL]s 70+ runs while looking busy.
_need = ["modality_attendance>ConstantScale", "modality_attendance>LearnableAlpha",
         "modality_attendance>ScalarMultiply", "modality_attendance>FusionWithGamma",
         "modality_attendance>GateEntropyLoss"]
_missing = [n for n in _need if n not in keras.saving.get_custom_objects()]
if _missing:
    raise RuntimeError(
        "Custom layers NOT registered: " + ", ".join(m.split(">")[1] for m in _missing) +
        "\n-> run the cell that prints '[XAI] registered custom layers:' first.")


S3_K          = 3         # disjoint subsets. 5 would be ~17 h on 110 runs; 3 is ~10 h (budget: 12 h).
S3_DO_SUBSETS = False      # subset 0 == the canonical published batch (reproduction anchor)
S3_DO_SIGMA   = True     # sigma sweep on subset 0 only (~10 h) — separate pass, separate night
S3_SIGMAS     = (4, 8, 16)
S3_LIMIT      = None

_subsets = faith_subsets(k=S3_K, n=FAITH_N_SAMPLES, seed=0)   # subset 0 == the canonical first 128
print("subsets:", [(len(s), int(s[0]), int(s[-1])) for s in _subsets])

# ── ALL Stage-3 output lives under runs/_robustness/stage3_image/ (see Stage 2 note) ──
_S3_DIR = os.path.join(RUNS_PATH, "_robustness", "stage3_image")
os.makedirs(_S3_DIR, exist_ok=True)
_OUT3 = os.path.join(_S3_DIR, "faithfulness_image_sigma.csv")
_done3 = set(pd.read_csv(_OUT3).run_id) if os.path.exists(_OUT3) else set()

_rows = pd.read_csv(os.path.join(SUMMARIES_PATH, "master_results_rows.csv"))
_rows = _rows[~_rows.run_id.isin(_done3)]
if S3_LIMIT:
    _rows = _rows.head(S3_LIMIT)
print(f"{len(_rows)} runs to do ({len(_done3)} already done)")

_t0 = time.time()
for i, r in _rows.reset_index(drop=True).iterrows():
    rd = os.path.join(ART_PATH, r.run_dir.split("artifacts/enhanced/")[-1])
    ck = glob.glob(os.path.join(rd, "checkpoints", "best_*.keras"))
    try:
        model = keras.models.load_model(ck[0], compile=False, safe_mode=False)

        jobs = []
        if S3_DO_SUBSETS:
            jobs += [dict(kind="subset", subset=k, sigma=8, idx=s) for k, s in enumerate(_subsets)]
        if S3_DO_SIGMA:
            jobs += [dict(kind="sigma", subset=0, sigma=sg, idx=_subsets[0])
                     for sg in S3_SIGMAS if not (S3_DO_SUBSETS and sg == 8)]

        for j in jobs:
            # NOTE: the |SHAP| arrays on disk only cover the canonical first 128, so for the other
            # subsets the tabular ranking is recomputed inside explanation_faithfulness (fallback).
            o = explanation_faithfulness(
                model, rd, int(r.best_epoch),
                methods=("image",) if j["kind"] == "sigma" else ("tabular", "image", "joint"),
                subset_idx=j["idx"], blur_sigma=j["sigma"],
                out_name=f"faith_s{j['subset']}_sig{j['sigma']}",
                out_dir=os.path.join(_S3_DIR, str(r.run_id)),   # NOT runs/<run_id>/xai/
                save=True, verbose=False,
            )
            row = {"impl_name": r.impl_name, "run_id": r.run_id, "seed": r.seed,
                   "setting_tag": r.setting_tag, "subset": j["subset"], "sigma": j["sigma"]}
            row.update({k: v for k, v in o.items() if k.startswith("faithfulness_")})
            pd.DataFrame([row]).to_csv(_OUT3, mode="a", header=not os.path.exists(_OUT3), index=False)

        el = time.time() - _t0
        print(f"[{i+1}/{len(_rows)}] {r.run_id}  ({el/60:.0f} min, "
              f"ETA {el/(i+1)*(len(_rows)-i-1)/60:.0f} min)", flush=True)
    except Exception as e:
        print(f"[FAIL] {r.run_id}: {type(e).__name__}: {e}")
    finally:
        keras.backend.clear_session(); gc.collect()

_s = pd.read_csv(_OUT3)

# ---- reproduction anchor: subset 0 at sigma=8 must match the published numbers ----
_anchor = _s[(_s.subset == 0) & (_s.sigma == 8)].merge(
    pd.read_csv(os.path.join(SUMMARIES_PATH, "master_results_rows.csv"))[["run_id", "faithfulness_image_del_gain"]], on="run_id", suffixes=("", "_pub"))
if len(_anchor):
    err = (_anchor.faithfulness_image_del_gain - _anchor.faithfulness_image_del_gain_pub).abs().max()
    print(f"\nANCHOR subset0/sigma8 vs published: max diff = {err:.4f}")
    print("  (a small non-zero diff is EXPECTED: the FeatureSpace was re-adapted non-deterministically")
    print("   when the published numbers were produced — see the report's reproducibility section)")

# ---- is the family ranking stable across subsets? ----
if S3_DO_SUBSETS:
    print("\n" + "=" * 78); print("IS THE FAMILY RANKING STABLE ACROSS SUBSETS?"); print("=" * 78)
    piv = _s[_s.sigma == 8].pivot_table(index="impl_name", columns="subset",
                                        values="faithfulness_image_del_gain", aggfunc="mean")
    print(piv.round(3).to_string())
    print("\nrank per subset:"); print(piv.rank(ascending=False).astype(int).to_string())
    print("\nIf the ranking reshuffles across subsets, the fusion claim rests on one lucky batch.")

    print("\ndeletion <-> insertion correlation, per subset (published: r = +0.008 on the image side):")
    for k in sorted(_s.subset.unique()):
        d = _s[(_s.subset == k) & (_s.sigma == 8)]
        if {"faithfulness_image_del_gain", "faithfulness_image_ins_gain"} <= set(d.columns):
            print(f"  subset {k}: r = {d.faithfulness_image_del_gain.corr(d.faithfulness_image_ins_gain):+.3f}")

if S3_DO_SIGMA:
    print("\n" + "=" * 78); print("IS THE DIVERGENCE AN OOD ARTIFACT OF THE BLUR BASELINE?"); print("=" * 78)
    print(_s[_s.subset == 0].pivot_table(index="impl_name", columns="sigma",
                                         values="faithfulness_image_del_gain").round(3).to_string())


