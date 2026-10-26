"""Synthetic final-test checks; every artifact write and experimental read is mocked."""

import copy
import io
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from isic24 import statistical_evaluation as se
from isic24.xai_subset import CENTRAL_SETTINGS


SEEDS = [8, 47, 53, 71, 90, 101]
SETTINGS = {
    "vanilla": ["baseline"],
    "fixed_scaling": [f"a{v:.3f}".replace(".", "p") for v in np.linspace(0, 1, 9)],
    "learnable_scaling": [f"init_a{v:.3f}".replace(".", "p") for v in (.1, .3, .5, .7, .9)],
    "loss_regularization": ["lam0e+00", "lam1e-04", "lam1e-03", "lam1e-02", "lam1e-01"],
    "controlled_fusion": ["base"],
    "adaptive_gating": ["ent0e+00", "base", "ent1e-03", "ent1e-02", "ent1e-01"],
}


def synthetic_registry():
    records = []
    for family, tags in SETTINGS.items():
        for i, tag in enumerate(tags):
            for j, seed in enumerate(SEEDS):
                directory = Path("/virtual/art/runs") / family / "study" / f"seed_{seed}" / tag
                auc = .65 + i * .01 + j * .001
                if tag in se.CONTROLS.values():
                    auc = .99
                records.append(dict(
                    impl_name=family, setting_tag=tag, seed=seed, model_name=f"{family}_{tag}",
                    run_id=f"{family}/study/seed_{seed}/{tag}", best_val_auc=auc, base_auc=auc,
                    checkpoint=str(directory / "checkpoints" / f"best_{family}_{tag}.keras"),
                    config_file=str(directory / "config.json"), dataset="cmmd/processed",
                    feature_space_file="feature_space/test.keras", feature_space_fingerprint="fs",
                    validation_snapshot="validation_snapshot.json", n_train=30, n_val=12, batch_size=4,
                    entropy_weight=[0, 1e-4, 1e-3, 1e-2, 1e-1][i] if family == "adaptive_gating" else np.nan,
                ))
    registry = pd.DataFrame(records)
    gamma = []
    for row in registry[registry.impl_name.eq("controlled_fusion")].itertuples():
        for delta in np.linspace(-.5, .5, 9):
            gamma.append(dict(run_id=row.run_id, seed=row.seed, delta=delta,
                              gamma_img=1-delta, gamma_tab=1+delta,
                              val_auc=row.base_auc + (.0625 - (delta-.25)**2) * .02))
    return registry, pd.DataFrame(gamma)


def rejects(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError(f"Expected {error.__name__}")


def test_metrics():
    rng = np.random.default_rng(7)
    y = np.tile([0, 1], 20)
    p = np.round(rng.random((6, len(y))), 1)
    p[:, :2] = [0, 1]
    weights = rng.integers(0, 5, len(y))
    actual = se.metric_values(y, p, weights)
    for i, probability in enumerate(p):
        auc = roc_auc_score(y, probability, sample_weight=weights)
        brier = np.average((probability-y)**2, weights=weights)
        ece = 0.0
        edges = np.linspace(0, 1, 16)
        for k in range(15):
            mask = (probability >= edges[k]) & ((probability <= edges[k+1]) if k == 14 else (probability < edges[k+1]))
            if weights[mask].sum():
                ece += abs(np.sum(weights[mask] * (probability[mask]-y[mask]))) / weights.sum()
        np.testing.assert_allclose(actual[i], [auc, brier, ece], atol=1e-14)
    np.testing.assert_allclose(se.metric_values([0, 1], [0, 1]), [[1, 0, 0]])
    np.testing.assert_allclose(se.metric_values([0, 1], [.5, .5]), [[.5, .25, 0]])
    for invalid in ([np.nan, .5], [-.1, 1], [0, 1.1]):
        rejects(lambda: se.metric_values([0, 1], invalid))
    rejects(lambda: se.metric_values([0, 1], [.2, .8], [1, 0]))


def test_selection():
    registry, gamma = synthetic_registry()
    selected = se.select_evaluations(registry, gamma, CENTRAL_SETTINGS, SEEDS)
    assert len(selected) == 2 * 8 * 6
    best = selected.query("panel == 'selected'")
    assert best.query("strategy == 'fixed_scaling'").setting_tag.eq("a0p875").all()
    assert best.query("strategy == 'adaptive_gating'").setting_tag.eq("ent1e-01").all()
    assert best.query("strategy == 'controlled_fusion'").delta.eq(.25).all()
    assert selected.query("panel == 'reference' and strategy == 'controlled_fusion'").delta.eq(0).all()
    assert not best[best.strategy.isin(CENTRAL_SETTINGS)].setting_tag.isin(se.CONTROLS.values()).any()
    assert selected.groupby(["panel", "strategy"]).seed.apply(lambda x: set(x) == set(SEEDS)).all()
    json.dumps(selected.to_dict("records"), allow_nan=False)
    rejects(lambda: se.select_evaluations(registry.iloc[1:], gamma, CENTRAL_SETTINGS, SEEDS))
    rejects(lambda: se.select_evaluations(pd.concat([registry, registry.iloc[:1]]), gamma, CENTRAL_SETTINGS, SEEDS))
    rejects(lambda: se.select_evaluations(registry, gamma.iloc[1:], CENTRAL_SETTINGS, SEEDS))
    broken = gamma.copy()
    broken.loc[0, "gamma_tab"] = 9
    rejects(lambda: se.select_evaluations(registry, broken, CENTRAL_SETTINGS, SEEDS))
    broken = gamma.copy()
    broken.loc[broken.delta.eq(0), "val_auc"] += .1
    rejects(lambda: se.select_evaluations(registry, broken, CENTRAL_SETTINGS, SEEDS))


def test_freeze():
    registry, gamma = synthetic_registry()
    files = {}
    for row in registry.to_dict("records"):
        cfg = {key: value for key, value in row.items()
               if key not in ("checkpoint", "config_file", "entropy_weight")}
        if row["impl_name"] == "adaptive_gating":
            cfg["entropy_weight"] = row["entropy_weight"]
        files[row["config_file"]] = json.dumps(cfg)
    canonical = {family: Path("/virtual/art/runs") / family / "study" for family in SETTINGS}
    gamma_path = str(canonical["controlled_fusion"] / "controlled_fusion_posthoc_gamma_summary.csv")
    files[gamma_path] = gamma.to_csv(index=False)

    class Capture(io.StringIO):
        def __init__(self, path):
            super().__init__()
            self.path = path

        def close(self):
            files[self.path] = self.getvalue()
            super().close()

    def glob(path, pattern):
        return [Path(key) for key in sorted(files)
                if key.startswith(str(path) + "/") and key.endswith("/config.json")]

    with patch.object(Path, "glob", glob), \
            patch.object(se.paths, "DATASET", "cmmd"), \
            patch.object(se.paths, "VARIANT", "processed"), \
            patch.object(Path, "is_file", return_value=True), \
            patch.object(Path, "exists", lambda p: str(p) in files), \
            patch.object(Path, "mkdir"), \
            patch.object(Path, "read_text", lambda p: files[str(p)]), \
            patch.object(Path, "open", lambda p, mode: Capture(str(p))), \
            patch.object(se, "file_hash", side_effect=lambda p: se._digest(files.get(str(p), str(p)))), \
            patch.object(pd, "read_csv", side_effect=lambda p, **kwargs: pd_read_csv(io.StringIO(files[str(p)]), **kwargs)):
        arguments = (canonical, "/virtual/art", "/virtual/out", SEEDS,
                     {key: len(value) for key, value in SETTINGS.items()}, CENTRAL_SETTINGS)
        frozen = se.freeze_protocol(*arguments, n_boot=100, n_permutations=100)
        assert len(frozen["evaluations"]) == 96
        assert "test-metadata" not in " ".join(frozen["sources"])
        assert se.freeze_protocol(*arguments, n_boot=100, n_permutations=100) == frozen
        rejects(lambda: se.freeze_protocol(*arguments, n_boot=101, n_permutations=100), RuntimeError)
        config_file = next(key for key in files if key.endswith("config.json"))
        cfg = json.loads(files[config_file])
        cfg["best_val_auc"] += .001
        files[config_file] = json.dumps(cfg)
        rejects(lambda: se.freeze_protocol(*arguments, n_boot=100, n_permutations=100), RuntimeError)
        for dataset in ("isic24/enhanced", "brset/plain", "symile/processed"):
            files.pop("/virtual/out/protocol.json")
            for path in [key for key in files if key.endswith("/config.json")]:
                cfg = json.loads(files[path])
                cfg["dataset"] = dataset
                files[path] = json.dumps(cfg)
            name, variant = dataset.split("/")
            with patch.object(se.paths, "DATASET", name), patch.object(se.paths, "VARIANT", variant):
                frozen = se.freeze_protocol(*arguments, n_boot=100, n_permutations=100)
                assert frozen["dataset"] == dataset and len(frozen["evaluations"]) == 96
                cfg["dataset"] = "cmmd/processed"
                files[path] = json.dumps(cfg)
                rejects(lambda: se.freeze_protocol(*arguments, n_boot=100, n_permutations=100))


pd_read_csv = pd.read_csv


def test_population():
    source = pd.DataFrame(dict(image_id=["a", "b", "c", "v"], patient_id=["p1", "p2", "p3", "pv"],
                               age=[10, 20, 30, 999], target=[0, 1, 0, 1]))
    training = source.iloc[[0, 1, 2, 2, 2]].copy()
    validation = source.iloc[[3]].copy()
    test = pd.DataFrame(dict(image_id=["t1", "t2"], patient_id=["pt1", "pt2"], target=[0, 1],
                             age=[np.nan, 50], view=[None, "MLO"], laterality=["L", None]))
    result, medians = se.prepare_test_frame(test, source, training, validation)
    assert medians == {"age": 20} and result.age.tolist() == [20, 50]
    assert result.view.tolist() == ["unknown", "MLO"]
    assert test.age.isna().any()
    for key, value in (("patient_id", "pv"), ("image_id", "a")):
        bad = test.copy()
        bad.loc[0, key] = value
        rejects(lambda: se.prepare_test_frame(bad, source, training, validation))
    rejects(lambda: se.prepare_test_frame(pd.concat([test, test.iloc[:1]]), source, training, validation))


def test_cross_dataset_population():
    for dataset in ("isic24/enhanced", "brset/plain", "symile/processed", "cmmd/processed"):
        row_id = "isic_id" if dataset.startswith("isic24/") else "image_id"
        numeric = "age_approx" if dataset.startswith(("isic24/", "brset/")) else "age"
        source = pd.DataFrame({row_id: ["a", "b", "c", "v"],
                               "patient_id": ["pa", "pb", "pc", "pv"], "target": [0, 1, 0, 1],
                               numeric: [10., 20., 30., 999.], "category": ["x"] * 4})
        test = pd.DataFrame({row_id: ["t1", "t2"], "patient_id": ["pt1", "pt2"],
                             "target": [0, 1], numeric: [np.nan, 50.], "category": [None, "y"]})
        features = {numeric: "float32", "category": "string"}
        if dataset == "brset/plain":
            source["diabetes_time_y"] = ["1", "3", "invalid", "999"]
            test["diabetes_time_y"] = ["invalid", "5"]
            features["diabetes_time_y"] = "float32"
        training, validation = source.iloc[[0, 1, 2, 2, 2]], source.iloc[[3]]
        result, medians = se.prepare_test_frame(test, source, training, validation,
                                                dataset=dataset, features=features)
        assert medians[numeric] == 20 and result[numeric].tolist() == [20, 50]
        assert result.category.tolist() == ["unknown", "y"]
        assert result.image_id.tolist() == ["t1", "t2"]
        assert test[numeric].isna().any()
        if dataset == "brset/plain":
            assert medians["diabetes_time_y"] == 2
            assert result.diabetes_time_y.tolist() == [2, 5]
        bad = test.copy()
        bad.loc[0, "patient_id"] = "pv"
        rejects(lambda: se.prepare_test_frame(bad, source, training, validation,
                                               dataset=dataset, features=features))


def test_cluster_bootstrap():
    y = np.array([0, 1, 0, 1, 0, 1])
    patients = np.repeat(["a", "b", "c"], 2)
    p = np.array([[.1, .9, .4, .6, .3, .7], [.2, .8, .45, .55, .1, .9]])
    seen = []
    actual = se.metric_values

    def capture(labels, probability, weights):
        assert np.array_equal(weights[::2], weights[1::2])
        seen.append(weights.copy())
        return actual(labels, probability, weights)

    with patch.object(se, "metric_values", side_effect=capture):
        draws, rejected = se.patient_bootstrap(y, {"a": p, "b": p.copy()}, patients, 20,
                                              np.random.default_rng(5))
    assert len(seen) == 40 and rejected == 0
    for i in range(0, len(seen), 2):
        assert np.array_equal(seen[i], seen[i+1])
    np.testing.assert_array_equal(draws["a"], draws["b"])
    np.testing.assert_array_equal(draws["a"] - draws["b"], 0)
    np.testing.assert_array_equal(se.patient_swap_pvalues(y, p, p, patients, 20,
                                                         np.random.default_rng(0)), np.ones(3))
    np.testing.assert_allclose(se._holm([.01, .04, .03]), [.03, .06, .06])


def test_patient_swaps():
    y = np.array([0, 1, 0, 1, 0, 1])
    patients = np.repeat(["a", "b", "c"], 2)
    a = np.array([[.2, .7, .3, .8, .4, .9], [.1, .6, .2, .7, .3, .8]])
    b = 1 - a
    actual, seen = se.metric_values, []

    def capture(labels, probability, weights=None):
        for i in (0, 2, 4):
            block = probability[:, i:i+2]
            assert np.array_equal(block, a[:, i:i+2]) or np.array_equal(block, b[:, i:i+2])
        seen.append(probability.copy())
        return actual(labels, probability, weights)

    with patch.object(se, "metric_values", side_effect=capture):
        p = se.patient_swap_pvalues(y, a, b, patients, 30, np.random.default_rng(0))
    assert len(seen) == 62 and np.all((p > 0) & (p <= 1))


def test_analysis():
    registry, gamma = synthetic_registry()
    plan = se.select_evaluations(registry, gamma, CENTRAL_SETTINGS, SEEDS)
    frame = pd.DataFrame(dict(image_id=[f"i{i}" for i in range(8)],
                              patient_id=np.repeat(["a", "b", "c", "d"], 2), target=[0, 1]*4))
    base = np.array([.2, .8, .4, .6, .6, .4, .3, .7])
    predictions = {r.evaluation_id: np.clip(base + (r.seed % 3 - 1) * .01, 0, 1)
                   for r in plan.itertuples()}
    protocol = dict(evaluations=plan.to_dict("records"), seeds=SEEDS,
                    n_boot=15, n_permutations=15, random_seed=1)
    result = se.analyze_test(protocol, frame, predictions)
    assert len(result["per_seed"]) == 96 and len(result["summary"]) == 48
    assert len(result["contrasts"]) == 2 * (5 + 2*6) * 3
    np.testing.assert_array_equal(result["contrasts"].difference, 0)
    np.testing.assert_array_equal(result["contrasts"].patient_swap_p_holm, 1)
    np.testing.assert_array_equal(result["contrasts"].seed_sign_p_holm, 1)
    assert result["summary"].seed_sd.gt(0).any()
    other = se.analyze_test(protocol, frame, predictions)
    pd.testing.assert_frame_equal(result["summary"], other["summary"])
    pd.testing.assert_frame_equal(result["contrasts"], other["contrasts"])


def test_prediction_resume(dataset="cmmd/processed"):
    import keras
    from isic24 import xai_driver

    registry, gamma = synthetic_registry()
    plan = se.select_evaluations(registry, gamma, CENTRAL_SETTINGS, SEEDS)
    plan = plan[plan.seed.eq(8) & plan.strategy.isin(["vanilla", "controlled_fusion"])]
    protocol = dict(dataset=dataset, art_root="/virtual/art", evaluations=plan.to_dict("records"),
                    sources={"/virtual/source": "hash"}, protocol_hash="protocol",
                    feature_space_file="feature_space/test.keras", feature_space_fingerprint="fs",
                    validation_snapshot="validation_snapshot.json", batch_size=2, n_train=2, n_val=1)
    source = pd.DataFrame(dict(image_id=["a", "b", "v"], patient_id=["pa", "pb", "pv"],
                               age=[10, 30, 999], target=[0, 1, 0]))
    test = pd.DataFrame(dict(image_id=["t1", "t2"], patient_id=["pt1", "pt2"], target=[0, 1],
                             age=[None, 50], view=["CC", "MLO"], laterality=["L", "R"]))
    feature_types = {"age": "float32", "view": "string", "laterality": "string"}
    row_id = "isic_id" if dataset == "isic24/enhanced" else "image_id"
    source, test = (frame.rename(columns={"image_id": row_id}) for frame in (source, test))
    if dataset == "brset/plain":
        source["diabetes_time_y"] = ["1", "3", "999"]
        test["diabetes_time_y"] = [None, "5"]
        feature_types["diabetes_time_y"] = "float32"
    test_name = "enhanced_test_metadata_clean.parquet" if dataset.startswith("isic24/") else "test-metadata.parquet"
    source_name = "enhanced_train_metadata_clean.parquet" if dataset.startswith("isic24/") else "train-metadata.parquet"
    frames = {test_name: test, source_name: source,
              "training_frame.parquet": source.iloc[:2], "evaluation_frame.parquet": source.iloc[2:]}
    files = {"/virtual/out/protocol.json": json.dumps(protocol),
             "/virtual/art/validation_snapshot.json": json.dumps(dict(row_id_col=row_id, row_ids=["v"]))}
    cache, loaded = {}, []

    class Capture(io.StringIO):
        def close(self):
            files["/virtual/out/test_population.json"] = self.getvalue()
            super().close()

    class Saved(dict):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Scalar:
        def __init__(self):
            self.value = 1.0

        def assign(self, value):
            self.value = value

    class Model:
        inputs = [SimpleNamespace(shape=(None, 2, 2, 3)), SimpleNamespace(shape=(None, 1))]

        def __init__(self):
            self.fusion = SimpleNamespace(gamma_img=Scalar(), gamma_tab=Scalar())

        def get_layer(self, name):
            assert name == "fusion_with_gamma"
            return self.fusion

        def __call__(self, x, training):
            assert training is False
            return np.array([[.2], [.7]]) + (self.fusion.gamma_tab.value-1) * .1

    def load(path, **kwargs):
        if str(path).endswith("test.keras"):
            return SimpleNamespace(features={
                name: SimpleNamespace(dtype=dtype) for name, dtype in feature_types.items()})
        loaded.append(str(path))
        return Model()

    def save(path, **values):
        cache[str(path)] = Saved(values)

    with ExitStack() as stack:
        location_calls = {}
        for target, value in (("data_path", Path("/virtual/data/processed")),
                              ("data_dir", Path("/virtual/data"))):
            location_calls[target] = stack.enter_context(patch.object(se.paths, target, return_value=value))
        stack.enter_context(patch.object(Path, "mkdir"))
        stack.enter_context(patch.object(Path, "exists", lambda p: str(p) in files or str(p) in cache))
        stack.enter_context(patch.object(Path, "read_text", lambda p: files[str(p)]))
        stack.enter_context(patch.object(Path, "open", lambda p, mode: Capture()))
        stack.enter_context(patch.object(se, "file_hash", return_value="hash"))
        stack.enter_context(patch.object(pd, "read_parquet", side_effect=lambda p: frames[Path(p).name].copy()))
        stack.enter_context(patch.object(keras.models, "load_model", side_effect=load))
        stack.enter_context(patch.object(keras.backend, "clear_session"))
        stack.enter_context(patch.object(xai_driver, "feature_space_fingerprint", return_value=(1, "fs")))
        stack.enter_context(patch.object(xai_driver, "assert_fs_matches_model"))
        stack.enter_context(patch.object(se, "_test_dataset", return_value=[{"images": np.zeros((2, 2, 2, 3))}]))
        stack.enter_context(patch.object(np, "savez_compressed", side_effect=save))
        stack.enter_context(patch.object(np, "load", side_effect=lambda p, **kwargs: cache[str(p)]))
        output, predictions = se.predict_test(protocol, "/virtual/out")
        name, variant = dataset.split("/")
        location_calls["data_path"].assert_called_once_with(dataset=name, variant=variant)
        location_calls["data_dir"].assert_called_once_with(dataset=name)
        assert len(loaded) == 2 and len(predictions) == 3
        assert output.age.tolist() == [20, 50]
        cf = plan[plan.strategy.eq("controlled_fusion")]
        neutral = cf[cf.delta.eq(0)].evaluation_id.iloc[0]
        shifted = cf[cf.delta.eq(.25)].evaluation_id.iloc[0]
        np.testing.assert_allclose(predictions[shifted]-predictions[neutral], .025)
        loaded.clear()
        _, resumed = se.predict_test(protocol, "/virtual/out")
        assert not loaded
        for key in predictions:
            np.testing.assert_array_equal(predictions[key], resumed[key])
        item = next(iter(cache.values()))
        item["population_hash"] = "wrong"
        rejects(lambda: se.predict_test(protocol, "/virtual/out"))


def test_cross_dataset_prediction_resume():
    for dataset in ("isic24/enhanced", "brset/plain", "symile/processed"):
        test_prediction_resume(dataset)


def test_decoder():
    import h5py
    import tensorflow as tf

    pixels = np.array([[[0], [100]], [[200], [255]]], dtype=np.uint8)
    payload = tf.io.encode_png(pixels).numpy()
    frame = pd.DataFrame(dict(image_id=["t"], age=[20.0], view=["CC"], laterality=["L"]))

    class FeatureSpace:
        features = {"age": None, "view": None, "laterality": None}

        def __call__(self, features):
            assert features["view"].tolist() == ["CC"]
            return np.array([[20]], dtype=np.float32)

    class Store(dict):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Image:
        def __getitem__(self, key):
            assert key == ()
            return payload

    with patch.object(h5py, "File", return_value=Store(t=Image())):
        dataset = se._test_dataset(frame, FeatureSpace(), "/virtual/images", (2, 2, 3), 4)
        batches = list(dataset)
    assert len(batches) == 1
    np.testing.assert_allclose(batches[0]["images"], np.repeat(pixels, 3, axis=2)[None]/255, atol=1e-7)
    np.testing.assert_array_equal(batches[0]["features"], [[20]])


def test_export():
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    rows = []
    for metric in se.METRICS:
        for strategy in ("vanilla", "image_only_control"):
            rows.append(dict(panel="selected", strategy=strategy, metric=metric,
                             mean=.5, ci_lo=.4, ci_hi=.6))
    results = dict(per_seed=pd.DataFrame(), summary=pd.DataFrame(rows),
                   contrasts=pd.DataFrame(), bootstrap_rejected_single_class=0)
    with patch.object(pd.DataFrame, "to_csv") as csv, \
            patch.object(Path, "write_text") as metadata, \
            patch.object(Figure, "savefig") as figure:
        se.save_results(results, {"protocol_hash": "hash"}, "/virtual/out")
    assert csv.call_count == 3 and metadata.call_count == 1 and figure.call_count == 1
    assert Path(figure.call_args.args[0]).name == "test_selected_confidence_intervals.png"
    assert json.loads(metadata.call_args.args[0])["protocol_hash"] == "hash"
    assert not plt.get_fignums()


def test_notebook():
    root = Path(__file__).resolve().parents[2] / "notebooks/modality_attendance"
    paths = list(root.rglob("*_modattendance_statest.ipynb"))
    assert {path.name for path in paths} == {
        f"{name}_modattendance_statest.ipynb" for name in ("isic24", "brset", "symile", "cmmd")}
    for path in paths:
        check_notebook(path)


def check_notebook(path):
    notebook = json.loads(path.read_text())
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(cells) == 6
    for cell in cells:
        source = "".join(cell["source"])
        compile(source, str(path), "exec")
        assert source.startswith("# @@") and cell["outputs"] == []
        assert cell["execution_count"] is None
        assert sum(line.lstrip().startswith("#") for line in source.splitlines()) == 1
    text = "\n".join("".join(c["source"]) for c in cells)
    for call in ("se.freeze_protocol(", "se.predict_test(", "se.analyze_test(", "se.save_results("):
        assert call in text
    assert "fit(" not in text and "adapt(" not in text and "run_posthoc_xai" not in text


if __name__ == "__main__":
    for test in (test_metrics, test_selection, test_freeze, test_population, test_cross_dataset_population,
                 test_cluster_bootstrap, test_patient_swaps, test_analysis,
                 test_prediction_resume, test_cross_dataset_prediction_resume,
                 test_decoder, test_export, test_notebook):
        test()
        print(f"PASS {test.__name__}")
    print("PASS final-test statistics; no experimental artifact read or written")
