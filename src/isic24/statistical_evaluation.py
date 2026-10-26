"""Frozen test evaluation with patient-paired inference, conditional on fitted models."""

import gc
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from . import paths

METRICS = ("auc", "brier", "ece")
CONTROLS = {"image_only_control": "a0p000", "table_only_control": "a1p000"}


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def select_evaluations(registry, gamma, reference, seeds):
    """Select common settings across seeds using validation alone; keep controls separate."""
    if registry.duplicated(["impl_name", "setting_tag", "seed"]).any():
        raise ValueError("Duplicate setting/seed in the training registry")
    for _, group in registry.groupby(["impl_name", "setting_tag"]):
        if set(group.seed) != set(seeds) or not np.isfinite(group.best_val_auc).all():
            raise ValueError("Every validation candidate requires the complete seed grid")
    cf = registry[registry.impl_name.eq("controlled_fusion")]
    expected_delta = np.linspace(-0.5, 0.5, 9)
    if (set(gamma.run_id) != set(cf.run_id)
            or gamma.duplicated(["run_id", "delta"]).any()
            or not np.isfinite(gamma[["val_auc", "delta", "gamma_img", "gamma_tab"]]).all().all()):
        raise ValueError("Invalid validation gamma sweep")
    for _, group in gamma.groupby("run_id"):
        if len(group) != 9 or not np.allclose(sorted(group.delta), expected_delta):
            raise ValueError("Complete nine-point validation gamma grid required")
    if not (np.allclose(gamma.gamma_img, 1 - gamma.delta)
            and np.allclose(gamma.gamma_tab, 1 + gamma.delta)):
        raise ValueError("Validation gamma coefficients disagree with delta")
    neutral = gamma[gamma.delta.eq(0)].set_index("run_id").val_auc
    if not np.allclose(neutral.loc[cf.run_id], cf.base_auc, atol=5e-4, rtol=0):
        raise ValueError("Neutral gamma AUC disagrees with the selected checkpoints")
    rows = []
    for panel in ("reference", "selected"):
        for family, tag in reference.items():
            candidates = registry[registry.impl_name.eq(family)]
            delta = 0.0
            if panel == "selected":
                candidates = candidates[~candidates.setting_tag.isin(CONTROLS.values())]
                scores = candidates.groupby("setting_tag").best_val_auc.mean()
                tag = scores.sort_index().idxmax()
                if family == "controlled_fusion":
                    scores = gamma.groupby("delta").val_auc.mean().sort_index()
                    delta = float(scores.idxmax())
            chosen = candidates[candidates.setting_tag.eq(tag)]
            if len(chosen) != len(seeds):
                raise ValueError(f"Missing {panel} setting: {family}/{tag}")
            for row in chosen.to_dict("records"):
                score = row["best_val_auc"]
                if family == "controlled_fusion":
                    score = float(gamma.loc[gamma.run_id.eq(row["run_id"])
                                            & gamma.delta.eq(delta), "val_auc"].iloc[0])
                rows.append(dict(row, panel=panel, strategy=family, delta=delta,
                                 selection_val_auc=score))
        for name, tag in CONTROLS.items():
            controls = registry[registry.impl_name.eq("fixed_scaling")
                                & registry.setting_tag.eq(tag)]
            if len(controls) != len(seeds):
                raise ValueError(f"Missing unimodal endpoint control: {tag}")
            rows.extend(dict(row, panel=panel, strategy=name, delta=0.0,
                             selection_val_auc=row["best_val_auc"])
                        for row in controls.to_dict("records"))
    plan = pd.DataFrame(rows).sort_values(["panel", "strategy", "seed"])
    plan["evaluation_id"] = plan.apply(
        lambda r: f"{r.impl_name}:{r.setting_tag}:d{r.delta:+.3f}:s{int(r.seed)}", axis=1)
    return plan[["panel", "strategy", "impl_name", "setting_tag", "seed", "run_id",
                 "model_name", "checkpoint", "config_file", "delta", "selection_val_auc",
                 "evaluation_id"]].reset_index(drop=True)


def freeze_protocol(canonical, art_root, output_dir, seeds, expected_settings, reference,
                    *, n_boot=2000, n_permutations=2000, random_seed=2026):
    """Persist validation-only choices before reading test outcomes; reject protocol drift."""
    art_root, output_dir = Path(art_root).resolve(), Path(output_dir).resolve()
    dataset = f"{paths.DATASET}/{paths.VARIANT}"
    if (len(set(seeds)) != len(seeds) or len(seeds) < 2
            or n_boot < 100 or n_permutations < 100):
        raise ValueError("At least two distinct seeds and 100 resamples are required")
    records, sources = [], {}
    for family, study in canonical.items():
        study = Path(study).resolve()
        configs = sorted(study.glob("*/*/config.json"))
        for path in configs:
            cfg = json.loads(path.read_text())
            if cfg["impl_name"] != family or cfg["dataset"] != dataset:
                raise ValueError(f"Wrong run family or dataset: {path}")
            checkpoint = path.parent / "checkpoints" / f"best_{cfg['model_name']}.keras"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            records.append(dict(cfg, checkpoint=str(checkpoint), config_file=str(path)))
            sources[str(path)] = file_hash(path)
        family_rows = pd.DataFrame([r for r in records if r["impl_name"] == family])
        if (len(configs) != len(seeds) * expected_settings[family]
                or family_rows.setting_tag.nunique() != expected_settings[family]):
            raise ValueError(f"Incomplete {family} training grid; finish training first")
    registry = pd.DataFrame(records)
    if not registry.run_id.is_unique:
        raise ValueError("Duplicate run identities")
    for key in ("feature_space_file", "feature_space_fingerprint", "validation_snapshot",
                "n_train", "n_val", "batch_size"):
        if registry[key].isna().any() or registry[key].nunique() != 1:
            raise ValueError(f"Inconsistent frozen input: {key}")
    weights = registry[registry.impl_name.eq("adaptive_gating")].groupby("setting_tag").entropy_weight
    if (not weights.nunique().eq(1).all()
            or set(weights.first()) != {0, 1e-4, 1e-3, 1e-2, 1e-1}
            or weights.first().loc["base"] != 1e-4):
        raise ValueError("Incomplete or inconsistent gating entropy grid")
    gamma_file = Path(canonical["controlled_fusion"]) / "controlled_fusion_posthoc_gamma_summary.csv"
    gamma = pd.read_csv(gamma_file, skipinitialspace=True)
    gamma.columns = gamma.columns.str.strip()
    plan = select_evaluations(registry, gamma, reference, seeds)
    first = records[0]
    for path in (gamma_file, art_root / first["feature_space_file"],
                 art_root / first["validation_snapshot"], art_root / "training_frame.parquet",
                 art_root / "evaluation_frame.parquet", Path(__file__)):
        sources[str(path)] = file_hash(path)
    for checkpoint in plan.checkpoint.unique():
        sources[checkpoint] = file_hash(checkpoint)
    payload = dict(
        version=2, dataset=dataset, art_root=str(art_root), seeds=list(seeds),
        evaluations=plan.to_dict("records"), sources=sources,
        feature_space_file=first["feature_space_file"],
        feature_space_fingerprint=first["feature_space_fingerprint"],
        validation_snapshot=first["validation_snapshot"], batch_size=int(first["batch_size"]),
        n_train=int(first["n_train"]), n_val=int(first["n_val"]),
        n_boot=int(n_boot), n_permutations=int(n_permutations), random_seed=int(random_seed),
        auc="exact rank AUC; training selection uses the saved Keras validation AUC",
        ece="15 fixed equal-width bins; positive-class probability versus event frequency",
        estimand="image-level metric averaged over fitted seeds; no probability ensemble",
        interval="pointwise 95% patient-cluster percentile bootstrap, conditional on fitted seeds",
        patient_test="two-sided patient-block prediction swap across all paired seeds; exchangeability null",
        seed_test="two-sided exact sign test; conditional on the fixed test cohort",
        multiplicity="Holm across every contrast and metric, separately by panel and test unit",
        selection="one common setting per family maximizes mean validation AUC; lexicographic/lowest-delta tie break",
        controls="trained fixed-scaling endpoints; not independent unimodal architectures",
    )
    payload["protocol_hash"] = _digest(payload)
    target = output_dir / "protocol.json"
    if target.exists():
        if json.loads(target.read_text()) != payload:
            raise RuntimeError("Frozen protocol differs; do not revise choices after test inspection")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        with target.open("x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
    return payload


def prepare_test_frame(test, source, training, validation, *, dataset="cmmd/processed",
                       features=None):
    """Check patient separation and reproduce imputation from natural training rows only."""
    row_id = "isic_id" if dataset == "isic24/enhanced" else "image_id"
    if (test[[row_id, "patient_id", "target"]].isna().any().any()
            or not test[row_id].astype(str).is_unique):
        raise ValueError("Test identities must be present and unique")
    if set(test.target.unique()) != {0, 1}:
        raise ValueError("Both binary test classes are required")
    for name, frame in (("source", source), ("training", training), ("validation", validation)):
        for key in (row_id, "patient_id"):
            if frame[key].isna().any() or set(test[key].astype(str)) & set(frame[key].astype(str)):
                raise ValueError(f"Test/{name} {key} overlap or missing identities")
    if set(training.patient_id.astype(str)) & set(validation.patient_id.astype(str)):
        raise ValueError("Frozen training/validation patients overlap")
    natural = source[~source[row_id].astype(str).isin(validation[row_id].astype(str))].copy()
    if (not source[row_id].astype(str).is_unique
            or not set(validation[row_id].astype(str)) <= set(source[row_id].astype(str))
            or set(natural.patient_id.astype(str)) & set(validation.patient_id.astype(str))
            or not set(training[row_id].astype(str)) <= set(natural[row_id].astype(str))):
        raise ValueError("Frozen development frames disagree with the source split")
    result = test.copy()
    if features is None:
        features = {"age": "float32", "view": "string", "laterality": "string"}
    numeric = [col for col, dtype in features.items() if dtype != "string"]
    categorical = [col for col in features if col not in numeric]
    if dataset == "brset/plain":
        for frame in (natural, result):
            frame["diabetes_time_y"] = pd.to_numeric(frame["diabetes_time_y"], errors="coerce")
    medians = natural[numeric].median()
    if not np.isfinite(medians).all():
        raise ValueError("Natural training feature medians are undefined or non-finite")
    result[numeric] = result[numeric].fillna(medians)
    if dataset in {"cmmd/processed", "symile/processed"}:
        result[numeric] = result[numeric].astype("float32")
    for col in categorical:
        result[col] = result[col].fillna("unknown").astype(str)
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numeric test features")
    result["image_id"] = result[row_id].astype(str)
    return result, medians.to_dict()


def _test_dataset(frame, fs, image_file, image_shape, batch_size):
    import h5py
    import tensorflow as tf

    if image_shape[-1] != 3:
        raise ValueError("Expected RGB model input")
    raw = {col: frame[col].to_numpy() for col in fs.features}
    encoded = np.asarray(fs(raw))
    with h5py.File(image_file, "r") as store:
        images = [store[str(row_id)][()] for row_id in frame.image_id]

    def decode(image, features):
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_shape[:2], method="area")
        image = tf.ensure_shape(tf.cast(image, tf.float32) / 255.0, image_shape)
        return {"images": image, "features": features}

    ds = tf.data.Dataset.from_tensor_slices((images, encoded))
    return (ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
            .cache().batch(batch_size).prefetch(tf.data.AUTOTUNE))


def predict_test(protocol, output_dir):
    """Infer selected frozen checkpoints; reuse predictions only for identical inputs."""
    import keras
    import keras_cv  # Registers saved backbone serializers.
    from . import mod_attendance  # Registers saved fusion layers.
    from .xai_driver import assert_fs_matches_model, feature_space_fingerprint

    root, output_dir = Path(protocol["art_root"]), Path(output_dir)
    frozen = json.loads((output_dir / "protocol.json").read_text())
    if frozen != protocol:
        raise ValueError("Run the protocol cell before test inference")
    for path, expected in protocol["sources"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")
    dataset, variant = protocol["dataset"].split("/")
    data = paths.data_path(dataset=dataset, variant=variant)
    if dataset == "isic24":
        test_file = data / "enhanced_test_metadata_clean.parquet"
        source_file = data / "enhanced_train_metadata_clean.parquet"
    else:
        test_file, source_file = data / "test-metadata.parquet", data / "train-metadata.parquet"
    image_file = paths.data_dir(dataset=dataset) / "test-image.hdf5"
    fs = keras.models.load_model(root / protocol["feature_space_file"])
    if feature_space_fingerprint(fs)[1] != protocol["feature_space_fingerprint"]:
        raise ValueError("FeatureSpace fingerprint mismatch")
    validation = pd.read_parquet(root / "evaluation_frame.parquet")
    training = pd.read_parquet(root / "training_frame.parquet")
    snapshot = json.loads((root / protocol["validation_snapshot"]).read_text())
    if (validation[snapshot["row_id_col"]].astype(str).tolist() != list(map(str, snapshot["row_ids"]))
            or len(validation) != protocol["n_val"] or len(training) != protocol["n_train"]):
        raise ValueError("Frozen validation snapshot mismatch")
    frame, medians = prepare_test_frame(
        pd.read_parquet(test_file), pd.read_parquet(source_file),
        training, validation, dataset=protocol["dataset"],
        features={col: spec.dtype for col, spec in fs.features.items()})
    population = dict(protocol_hash=protocol["protocol_hash"], training_feature_medians=medians,
                      files={str(p): file_hash(p) for p in (test_file, source_file, image_file)},
                      image_ids=frame.image_id.astype(str).tolist(),
                      patient_ids=frame.patient_id.astype(str).tolist(), labels=frame.target.tolist())
    population["population_hash"] = _digest(population)
    identity_file = output_dir / "test_population.json"
    if identity_file.exists() and json.loads(identity_file.read_text()) != population:
        raise ValueError("Test population changed after protocol execution")
    if not identity_file.exists():
        with identity_file.open("x") as stream:
            json.dump(population, stream, indent=2)
    plan = pd.DataFrame(protocol["evaluations"]).drop_duplicates("evaluation_id")
    predictions, input_dataset, expected_shape = {}, None, None
    cache = output_dir / "predictions"
    cache.mkdir(exist_ok=True)
    for checkpoint, group in plan.groupby("checkpoint", sort=False):
        model = None
        try:
            for row in group.itertuples(index=False):
                target = cache / f"{_digest(row.evaluation_id)}.npz"
                if target.exists():
                    with np.load(target, allow_pickle=False) as saved:
                        if (str(saved["population_hash"]) != population["population_hash"]
                                or str(saved["evaluation_id"]) != row.evaluation_id
                                or not np.array_equal(saved["image_ids"], population["image_ids"])):
                            raise ValueError(f"Prediction identity mismatch: {target}")
                        probability = saved["probability"]
                else:
                    if model is None:
                        model = keras.models.load_model(checkpoint, compile=False, safe_mode=False)
                        assert_fs_matches_model(model, fs, row.run_id)
                        shape = next(tuple(t.shape[1:]) for t in model.inputs if len(t.shape) == 4)
                        if input_dataset is None:
                            expected_shape = shape
                            input_dataset = _test_dataset(frame, fs, image_file, shape, protocol["batch_size"])
                        if shape != expected_shape:
                            raise ValueError("Checkpoints have different image shapes")
                    if row.impl_name == "controlled_fusion":
                        fusion = model.get_layer("fusion_with_gamma")
                        fusion.gamma_img.assign(1 - row.delta)
                        fusion.gamma_tab.assign(1 + row.delta)
                    probability = np.concatenate([
                        np.asarray(model(x, training=False)).reshape(-1) for x in input_dataset])
                    _validate_probability(probability, len(frame))
                    np.savez_compressed(target, probability=probability,
                                        image_ids=np.asarray(population["image_ids"]),
                                        population_hash=population["population_hash"],
                                        evaluation_id=row.evaluation_id)
                _validate_probability(probability, len(frame))
                predictions[row.evaluation_id] = probability
                print(f"[TEST] {len(predictions)}/{len(plan)} {row.evaluation_id}")
        finally:
            del model
            keras.backend.clear_session()
            gc.collect()
    return frame, predictions


def _validate_probability(p, n):
    if (np.shape(p) != (n,) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)):
        raise ValueError("Expected one finite binary probability per test row")


def metric_values(y, p, weights=None):
    """Exact weighted AUC, Brier and fixed-bin positive-class ECE, for each seed."""
    y, p = np.asarray(y), np.atleast_2d(np.asarray(p, dtype=float))
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if (y.ndim != 1 or p.shape[1] != len(y) or w.shape != y.shape
            or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1)) or not np.isfinite(w).all()
            or np.any(w < 0) or w[y == 0].sum() == 0 or w[y == 1].sum() == 0):
        raise ValueError("Invalid probabilities, labels or weights; both classes need positive weight")
    result = []
    for probabilities in p:
        order = np.argsort(probabilities, kind="stable")
        starts = np.r_[0, np.flatnonzero(np.diff(probabilities[order])) + 1]
        positive = np.add.reduceat((w * y)[order], starts)
        negative = np.add.reduceat((w * (1 - y))[order], starts)
        auc = np.sum(positive * (np.cumsum(negative) - negative / 2)) / (
            positive.sum() * negative.sum())
        bins = np.minimum((probabilities * 15).astype(int), 14)
        residual = np.bincount(bins, weights=w * (probabilities - y), minlength=15)
        result.append([auc,
                       np.average((probabilities - y) ** 2, weights=w),
                       np.abs(residual).sum() / w.sum()])
    return np.asarray(result)


def _holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def patient_bootstrap(y, probabilities, patient_ids, n_boot, rng):
    """Use identical patient multiplicities for every strategy and every seed."""
    _, inverse = np.unique(np.asarray(patient_ids).astype(str), return_inverse=True)
    n_patients = int(inverse.max()) + 1
    draws = {key: [] for key in probabilities}
    rejected = 0
    while len(next(iter(draws.values()))) < n_boot:
        counts = np.bincount(rng.integers(0, n_patients, size=n_patients), minlength=n_patients)
        weights = counts[inverse]
        if weights[np.asarray(y) == 0].sum() == 0 or weights[np.asarray(y) == 1].sum() == 0:
            rejected += 1
            if rejected > 100 * n_boot:
                raise ValueError("Too few class-supporting patient clusters for bootstrap")
            continue
        for key, p in probabilities.items():
            draws[key].append(metric_values(y, p, weights).mean(axis=0))
    return {key: np.asarray(value) for key, value in draws.items()}, rejected


def patient_swap_pvalues(y, a, b, patient_ids, n_permutations, rng):
    """Swap complete patient prediction blocks, jointly across seeds; use a +1 MC correction."""
    _, inverse = np.unique(np.asarray(patient_ids).astype(str), return_inverse=True)
    observed = metric_values(y, b).mean(axis=0) - metric_values(y, a).mean(axis=0)
    exceed = np.zeros(len(METRICS), dtype=int)
    for _ in range(n_permutations):
        swap = rng.integers(0, 2, size=int(inverse.max()) + 1)[inverse].astype(bool)
        aa, bb = np.where(swap, b, a), np.where(swap, a, b)
        difference = metric_values(y, bb).mean(axis=0) - metric_values(y, aa).mean(axis=0)
        exceed += np.abs(difference) >= np.maximum(0, np.abs(observed) - 1e-12)
    return (exceed + 1) / (n_permutations + 1)


def analyze_test(protocol, frame, predictions):
    """Summarize both panels and prespecified contrasts; no test-driven model selection."""
    plan = pd.DataFrame(protocol["evaluations"])
    y = frame.target.to_numpy()
    probabilities, point, per_seed = {}, {}, []
    for (panel, strategy), group in plan.groupby(["panel", "strategy"], sort=True):
        group = group.sort_values("seed")
        if group.seed.tolist() != sorted(protocol["seeds"]):
            raise ValueError("Test comparisons require identical paired seed grids")
        p = np.stack([predictions[key] for key in group.evaluation_id])
        if p.shape != (len(protocol["seeds"]), len(frame)):
            raise ValueError("Prediction shape disagrees with the frozen test population")
        key = (panel, strategy)
        probabilities[key], point[key] = p, metric_values(y, p)
        for row, metrics in zip(group.itertuples(index=False), point[key]):
            per_seed.append(dict(panel=panel, strategy=strategy, seed=int(row.seed),
                                 setting_tag=row.setting_tag, delta=row.delta, run_id=row.run_id,
                                 **dict(zip(METRICS, metrics))))
    rng = np.random.default_rng(protocol["random_seed"])
    draws, rejected = patient_bootstrap(y, probabilities, frame.patient_id,
                                        protocol["n_boot"], rng)
    print("[STATISTICS] Patient bootstrap complete; computing paired contrasts")
    summaries, contrasts = [], []
    for (panel, strategy), values in point.items():
        for i, metric in enumerate(METRICS):
            lo, hi = np.quantile(draws[panel, strategy][:, i], [0.025, 0.975])
            summaries.append(dict(panel=panel, strategy=strategy, metric=metric,
                                  mean=values[:, i].mean(), seed_sd=values[:, i].std(ddof=1),
                                  seed_variance=values[:, i].var(ddof=1), ci_lo=lo, ci_hi=hi,
                                  n_seeds=len(values), n_images=len(y),
                                  n_patients=frame.patient_id.nunique()))
    for panel in sorted(plan.panel.unique()):
        strategies = sorted(plan.loc[plan.panel.eq(panel), "strategy"].unique())
        multimodal = [s for s in strategies if s not in CONTROLS]
        pairs = [("vanilla", s) for s in multimodal if s != "vanilla"]
        pairs += [(control, s) for control, s in itertools.product(CONTROLS, multimodal)]
        for baseline, strategy in pairs:
            a, b = (panel, baseline), (panel, strategy)
            patient_p = patient_swap_pvalues(y, probabilities[a], probabilities[b],
                                             frame.patient_id, protocol["n_permutations"], rng)
            for i, metric in enumerate(METRICS):
                d = point[b][:, i] - point[a][:, i]
                lo, hi = np.quantile(draws[b][:, i] - draws[a][:, i], [0.025, 0.975])
                informative = d[np.abs(d) > 1e-12]
                seed_p = (binomtest(int((informative > 0).sum()), len(informative)).pvalue
                          if len(informative) else 1.0)
                contrasts.append(dict(panel=panel, baseline=baseline, strategy=strategy,
                                      metric=metric, difference=d.mean(), ci_lo=lo, ci_hi=hi,
                                      favorable_seeds=int(((d > 1e-12) if metric == "auc" else (d < -1e-12)).sum()),
                                      n_informative_seeds=len(informative),
                                      patient_swap_p=patient_p[i], seed_sign_p=seed_p))
        print(f"[STATISTICS] {panel}: {len(pairs)} paired comparisons complete")
    contrasts = pd.DataFrame(contrasts)
    for _, group in contrasts.groupby("panel"):
        for test in ("patient_swap", "seed_sign"):
            contrasts.loc[group.index, f"{test}_p_holm"] = _holm(group[f"{test}_p"])
    return dict(per_seed=pd.DataFrame(per_seed), summary=pd.DataFrame(summaries),
                contrasts=contrasts, bootstrap_rejected_single_class=rejected)


def save_results(results, protocol, output_dir):
    """Write separate final-test tables and a forest plot; never alter training/XAI artifacts."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    for name in ("per_seed", "summary", "contrasts"):
        results[name].to_csv(output_dir / f"test_{name}.csv", index=False)
    metadata = dict(protocol_hash=protocol["protocol_hash"],
                    bootstrap_rejected_single_class=results["bootstrap_rejected_single_class"],
                    intervals="pointwise; patient sampling only; fitted seeds held fixed",
                    differences="strategy minus baseline; higher AUC/lower Brier/ECE is favorable",
                    pvalues="patient exchangeability and seed sign nulls differ; Holm within each panel/test unit",
                    scope="internal held-out predictive evaluation; no external validation or test XAI")
    (output_dir / "statistics.json").write_text(json.dumps(metadata, indent=2))
    figure, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    for axis, metric in zip(axes, METRICS):
        table = results["summary"].query("panel == 'selected' and metric == @metric").sort_values("strategy")
        axis.hlines(table.strategy, table.ci_lo, table.ci_hi)
        axis.plot(table["mean"], table.strategy, "o")
        axis.set_xlabel(f"{metric.upper()} (95% patient-bootstrap CI)")
        axis.grid(axis="x", alpha=0.3)
    figure.savefig(output_dir / "test_selected_confidence_intervals.png", dpi=160)
    plt.close(figure)
