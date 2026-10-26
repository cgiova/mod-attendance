"""Exercise standalone gamma completion with synthetic data and mocked artifact writes."""

from contextlib import ExitStack
import ast
import json
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

from isic24 import fusion_calibration as fc

STUDY = Path("/memory/artifacts/demo/processed/runs/controlled_fusion/study")
ART_ROOT = STUDY.parents[2]
RUNS = pd.DataFrame([
    dict(run_id=f"study/seed_{seed}__base__e20", impl_name="controlled_fusion",
         seed=seed, setting_tag="base", model_name="controlled_fusion__base",
         run_dir=str(STUDY / f"seed_{seed}" / "base__e20"))
    for seed in [8, 47]
])
CONFIGS = {
    row.run_id: dict(run_id=row.run_id, impl_name="controlled_fusion", seed=row.seed,
                     model_name=row.model_name, dataset="demo/processed", n_val=4,
                     batch_size=2, feature_space_file="feature_space/fs.keras",
                     feature_space_fingerprint="fingerprint",
                     validation_snapshot="validation_snapshot.json", base_auc=0.5)
    for row in RUNS.itertuples(index=False)
}
CSV = STUDY / "controlled_fusion_posthoc_gamma_summary.csv"
DATASET = [(np.array([i]), np.array([[y]], dtype=np.float32))
           for i, y in enumerate([0, 0, 1, 1])]


class Weight:
    def assign(self, value):
        self.value = value


class Model:
    inputs = [types.SimpleNamespace(shape=(None, 2, 2, 3))]

    def __init__(self):
        self.gamma_img, self.gamma_tab = Weight(), Weight()

    def get_layer(self, name):
        assert name == "fusion_with_gamma"
        return self

    def __call__(self, x, training=False):
        p = [0., .2, .8, 1.] if self.gamma_tab.value < 1 else [.4, .6, .4, .6]
        return np.asarray(p, dtype=np.float32)[x, None]


def run_completion(saved=None, config_change=None, *, include_attendance=False,
                   model_class=Model, dataset=DATASET):
    reads, writes, loads = {}, {}, []
    configs = {key: value.copy() for key, value in CONFIGS.items()}
    if config_change:
        config_change(configs)
    for row in RUNS.itertuples(index=False):
        reads[Path(row.run_dir) / "config.json"] = json.dumps(configs[row.run_id])
    prior_provenance = dict(existing_key="preserved")
    reads[CSV.with_suffix(".provenance.json")] = json.dumps(prior_provenance)

    def read_csv(path, **kwargs):
        if Path(path).name == "combined_summary.csv":
            return RUNS.copy()
        assert Path(path) == CSV and saved is not None
        return saved.copy()

    def load_model(path, **kwargs):
        loads.append(Path(path))
        return model_class()

    def read_text(path, **kwargs):
        return reads[path]

    def write_text(path, data, **kwargs):
        writes[path] = json.loads(data)

    def to_csv(frame, path, **kwargs):
        writes[Path(path)] = frame.copy()

    with ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, {
            "keras_cv": types.ModuleType("keras_cv"),
            "isic24.mod_attendance": types.ModuleType("isic24.mod_attendance"),
        }))
        stack.enter_context(patch.object(pd, "read_csv", side_effect=read_csv))
        stack.enter_context(patch.object(Path, "read_text", read_text))
        stack.enter_context(patch.object(Path, "write_text", write_text))
        stack.enter_context(patch.object(Path, "exists", lambda p: p in reads or (p == CSV and saved is not None)))
        stack.enter_context(patch.object(Path, "mkdir"))
        stack.enter_context(patch.object(pd.DataFrame, "to_csv", to_csv))
        stack.enter_context(patch.object(keras.models, "load_model", side_effect=load_model))
        validation = stack.enter_context(patch.object(fc, "_validation_dataset", return_value=(dataset, (2, 2, 3))))
        aggregate = fc.complete_gamma_calibration(STUDY, include_attendance=include_attendance)
    return aggregate, writes, loads, validation


def test_completion_and_resume():
    aggregate, writes, loads, validation = run_completion()
    table = writes[CSV]
    assert len(table) == 18 and len(aggregate) == 9 and len(loads) == 2
    validation.assert_called_once()
    assert table.groupby("run_id").size().eq(9).all()
    assert np.allclose(table.loc[table.delta < 0, "brier"], .02)
    assert np.allclose(table.loc[table.delta >= 0, "brier"], .26)
    assert np.allclose(table.ece, .1)
    assert writes[CSV.with_suffix(".provenance.json")]["existing_key"] == "preserved"
    for row in RUNS.itertuples(index=False):
        assert len(writes[Path(row.run_dir) / "diagnostics/posthoc_gamma_summary.csv"]) == 9

    partial = table.copy()
    partial.loc[0, "ece"] = np.nan
    partial.loc[0, "brier"] = .123
    aggregate, writes, loads, validation = run_completion(partial)
    assert len(loads) == 1
    assert writes[CSV].iloc[0].brier == .123
    assert np.isclose(writes[CSV].iloc[0].ece, .1)
    pd.testing.assert_series_equal(writes[CSV].val_auc, partial.val_auc)
    complete = writes[CSV]
    _, writes, loads, validation = run_completion(complete)
    assert not loads
    validation.assert_not_called()
    pd.testing.assert_frame_equal(writes[CSV], complete)

    legacy = table.drop(columns=["brier", "ece"])
    _, writes, loads, _ = run_completion(legacy)
    assert len(loads) == 2
    pd.testing.assert_series_equal(writes[CSV].val_auc, legacy.val_auc)

    bad = legacy.copy()
    bad.loc[0, "val_auc"] = .2
    try:
        run_completion(bad)
    except AssertionError as error:
        assert "Saved sweep AUC mismatch" in str(error)
    else:
        raise AssertionError("Saved AUC mismatch must stop the backfill")

    try:
        run_completion(config_change=lambda configs: configs[RUNS.iloc[0].run_id].update(base_auc=.9))
    except AssertionError as error:
        assert "Neutral AUC mismatch" in str(error)
    else:
        raise AssertionError("Wrong validation/checkpoint pairing must be rejected")


def test_validation_inputs_and_decoder():
    import h5py

    frame = pd.DataFrame(dict(image_id=["gray", "rgb"], age=[40., 60.], sex=["F", "M"], target=[0, 1]))
    snapshot = dict(row_id_col="image_id", row_ids=["gray", "rgb"])
    encoded = np.array([[40., 0.], [60., 1.]], dtype=np.float32)
    gray = np.full((2, 2, 1), 128, dtype=np.uint8)
    rgb = np.full((2, 2, 3), 255, dtype=np.uint8)
    store = {
        "gray": np.asarray(tf.io.encode_png(gray).numpy()),
        "rgb": np.asarray(tf.io.encode_jpeg(rgb).numpy()),
    }

    class FeatureSpace:
        features = {"age": None, "sex": None}

        def __call__(self, raw):
            assert raw["sex"].tolist() == ["F", "M"]
            return encoded

    driver = types.ModuleType("isic24.xai_driver")
    driver.feature_space_fingerprint = lambda fs: (2, "fingerprint")
    driver.assert_fs_matches_model = lambda *args: 2
    cfg = next(iter(CONFIGS.values())).copy()
    cfg["n_val"] = 2
    with patch.dict(sys.modules, {"isic24.xai_driver": driver}), \
         patch.object(Path, "read_text", return_value=json.dumps(snapshot)), \
         patch.object(pd, "read_parquet", return_value=frame), \
         patch.object(keras.models, "load_model", return_value=FeatureSpace()), \
         patch.object(h5py, "File") as image_file:
        image_file.return_value.__enter__.return_value = store
        ds, shape = fc._validation_dataset(ART_ROOT, cfg, Model())
        inputs, labels = next(iter(ds))
        assert shape == (2, 2, 3)
        assert labels.numpy().ravel().tolist() == [0, 1]
        assert np.allclose(inputs["images"][0], 128 / 255)
        assert np.allclose(inputs["images"][1], 1.)
        assert np.array_equal(inputs["features"], encoded)
        assert str(image_file.call_args.args[0]).endswith("data/demo/train-image.hdf5")

        snapshot["row_ids"].reverse()
        with patch.object(Path, "read_text", return_value=json.dumps(snapshot)):
            try:
                fc._validation_dataset(ART_ROOT, cfg, Model())
            except AssertionError as error:
                assert "Validation snapshot mismatch" in str(error)
            else:
                raise AssertionError("Frozen row order must be checked")


def test_attendance_completion():
    class AttendanceModel(Model):
        def __call__(self, x, training=False):
            p = super().__call__(x["sample_index"], training)
            if np.all(np.asarray(x["images"]) == .5):
                return np.full_like(p, .5)
            if np.all(np.asarray(x["features"]) == 0):
                return 1 - p
            return p

    dataset = [({"sample_index": x, "images": np.ones((1, 2, 2, 3)),
                 "features": np.ones((1, 2))}, y) for x, y in DATASET]
    _, original, _, _ = run_completion()
    saved = original[CSV]
    aggregate, writes, loads, _ = run_completion(
        saved, include_attendance=True, model_class=AttendanceModel, dataset=dataset,
    )
    table = writes[CSV]
    pd.testing.assert_frame_equal(table[saved.columns], saved)
    assert len(loads) == 2
    assert np.allclose(table.loc[table.delta < 0, "img_drop_auc"], .5)
    assert np.allclose(table.loc[table.delta < 0, "tab_drop_auc"], 1.)
    assert np.allclose(table.loc[table.delta < 0, "modality_balance"], 2 / 3)
    assert np.allclose(table.loc[table.delta >= 0, "modality_balance"], 0.)
    assert {"img_drop_auc_mean", "tab_drop_auc_std", "modality_balance_mean"} <= set(aggregate)
    assert writes[CSV.with_suffix(".provenance.json")]["attendance"]["image_replacement"] == .5
    for inputs, _ in dataset:
        assert np.all(inputs["images"] == 1) and np.all(inputs["features"] == 1)

    _, writes, loads, _ = run_completion(
        table, include_attendance=True, model_class=AttendanceModel, dataset=dataset,
    )
    assert not loads
    pd.testing.assert_frame_equal(writes[CSV], table)
    _, writes, loads, _ = run_completion(table)
    assert not loads and "img_drop_auc_mean" in writes[
        STUDY / "controlled_fusion_posthoc_gamma_aggregated.csv"
    ]

    notebook = json.loads((ROOT / "notebooks/modality_attendance/cmmd/cmmd_modattendance_train.ipynb").read_text())
    runner = next("".join(c["source"]) for c in notebook["cells"]
                  if "".join(c.get("source", [])).startswith("# @@RUNNER@@"))
    names = {"_ablate_batch", "_eval_auc", "modality_contribution"}
    helpers = [node for node in ast.parse(runner).body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(tf=tf, keras=keras)
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "training attendance", "exec"), namespace)
    model = AttendanceModel()
    model.gamma_tab.assign(.5)
    expected = namespace["modality_contribution"](model, dataset)
    actual = fc._metrics(model, dataset, include_attendance=True)
    for metric in ("img_drop_auc", "tab_drop_auc", "modality_balance"):
        assert np.isclose(actual[metric], expected[metric]), metric


def test_notebook_cell_uses_selected_study():
    for dataset in ("cmmd", "isic24", "brset", "symile"):
        path = ROOT / f"notebooks/modality_attendance/{dataset}/{dataset}_modattendance_train.ipynb"
        nb = json.loads(path.read_text())
        source = next("".join(c["source"]) for c in nb["cells"]
                      if "".join(c.get("source", [])).startswith("# @@CF_GAMMA_CALIBRATION@@"))
        compile(source, f"{dataset} @@CF_GAMMA_CALIBRATION@@", "exec")
        assert all(name not in source for name in ["cf_study_dir", "cf_combined", "validation_ds", "stamp_aggregate"])
        with patch.object(fc, "complete_gamma_calibration", return_value="complete") as complete:
            displayed = []
            for study_dir in [STUDY, "/memory/artifacts/brset/plain/runs/controlled_fusion/other"]:
                exec(source, {"display": displayed.append, "study_dir": study_dir})
                complete.assert_called_with(study_dir, include_attendance=True)
            assert displayed == ["complete", "complete"]
    trainer_nb = json.loads((ROOT / "notebooks/modality_attendance/cmmd/cmmd_modattendance_train.ipynb").read_text())
    runner = next("".join(c["source"]) for c in trainer_nb["cells"]
                  if "".join(c.get("source", [])).startswith("# @@RUNNER@@"))
    helper = next(node for node in ast.parse(runner).body
                  if isinstance(node, ast.FunctionDef) and node.name == "brier_ece")
    namespace = dict(tf=tf, np=np)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), "brier_ece", "exec"), namespace)
    model = Model()
    model.gamma_tab.assign(.5)
    x, y = np.arange(4), tf.constant([[0.], [0.], [1.], [1.]])
    class TensorModel:
        def __call__(self, *args, **kwargs):
            return tf.constant(model(*args, **kwargs))
    expected = namespace["brier_ece"](TensorModel(), [(x, y)])
    actual = fc._metrics(model, [(x, y)])
    assert np.isclose(actual["brier"], expected["brier"])
    assert np.isclose(actual["ece"], expected["ece"])


def test_posthoc_replacement_reuses_enriched_sweep():
    aggregate, saved, _, _ = run_completion()
    table = saved[CSV].assign(img_drop_auc=.02, tab_drop_auc=.03, modality_balance=.6)
    original = table.copy(deep=True)

    class Plot:
        def __getattr__(self, name):
            return lambda *args, **kwargs: figures.append(Path(args[0])) if name == "savefig" else None

    nb = json.loads((ROOT / "notebooks/modality_attendance/cmmd/cmmd_modattendance_posthoc.ipynb").read_text())
    integrated = next("".join(c["source"]) for c in nb["cells"]
                      if "".join(c.get("source", [])).startswith("# @@CF_GAMMA_SWEEP@@"))
    replacement = (ROOT / "docs/CF_GAMMA_SWEEP.py").read_text()
    for label, source in (("cmmd", integrated), ("replacement", replacement)):
        figures, displayed = [], []
        namespace = dict(cf_study_dir=STUDY, study_dir="unrelated-gating-study",
                         cf_combined=RUNS, os=os, pd=pd, ART_PATH=ART_ROOT,
                         xs=types.SimpleNamespace(run_dir_abs=lambda path, root: path),
                         plt=Plot(), display=displayed.append)
        with patch.object(fc, "complete_gamma_calibration", return_value=aggregate) as complete, \
             patch.object(pd, "read_csv", return_value=table) as read, \
             patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError("Notebook must not rewrite sweep CSVs")), \
             patch.object(os, "makedirs"):
            exec(compile(source, f"{label} @@CF_GAMMA_SWEEP@@", "exec"), namespace)
            complete.assert_called_once_with(STUDY, include_attendance=True)
            assert Path(read.call_args.args[0]) == CSV
        pd.testing.assert_frame_equal(namespace["posthoc_all_df"], original)
        pd.testing.assert_frame_equal(displayed[-1], aggregate)
        assert set(figures) == {
            Path(row.run_dir) / "plots" / name
            for row in RUNS.itertuples(index=False)
            for name in ("posthoc_val_auc_vs_gamma_tab.png", "posthoc_gamma_sweep_scatter.png")
        }
        assert len(figures) == 2 * len(RUNS)


if __name__ == "__main__":
    test_completion_and_resume()
    test_validation_inputs_and_decoder()
    test_notebook_cell_uses_selected_study()
    test_attendance_completion()
    test_posthoc_replacement_reuses_enriched_sweep()
    print("PASS: gamma calibration and attendance, restart, identity guards, decoder, training callers and proposed post-hoc cell; all artifact writes mocked.")
