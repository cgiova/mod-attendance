"""Check integrated attendance cells for all four datasets with artifact writes mocked."""

import ast
import glob
import io
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import types
from unittest.mock import mock_open, patch
import warnings

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from isic24 import xai_metrics as xm
from isic24.run_ids import run_id_from_parts

CELLS = re.findall(r"```python\n(.*?)```", (ROOT / "docs/CMMD_ATTENDANCE_SWEEPS.md").read_text(), re.S)
TRAINING = ROOT / "notebooks/modality_attendance/cmmd/cmmd_modattendance_train.ipynb"
POSTHOC = ROOT / "notebooks/modality_attendance/cmmd/cmmd_modattendance_posthoc.ipynb"
SEEDS = [8, 47, 53, 71, 90, 101]
STUDIES = {
    "cmmd": ("20261015-114125", "20261016-234343", "20261020-224124", "20261019-233600", "20261020-063421", "20261020-020650"),
    "brset": ("20260909-150705", "20260914-094615", "20260909-183931", "20260910-104319", "20260911-103310", "20260914-235654"),
    "isic24": ("20260814-102807", "20260812-001020", "20260812-063607", "20260812-180911", "20260812-052145", "20260812-060120"),
    "symile": ("20260927-222132", "20260928-052108", "20260928-171346", "20260928-223034", "20260930-124221", "20260930-211234"),
}


def notebook_cells(path):
    return ["".join(c.get("source", [])) for c in json.loads(path.read_text())["cells"]]


def test_training_replacement():
    assert len(CELLS) == 6
    for cell in CELLS:
        compile(cell, "CMMD replacement", "exec")
    for dataset, studies in STUDIES.items():
        path = ROOT / f"notebooks/modality_attendance/{dataset}/{dataset}_modattendance_train.ipynb"
        source = next(c for c in notebook_cells(path) if c.startswith("# @@SWEEP_GATING@@"))
        check_training_sweep(source, studies[-1])


def check_training_sweep(source, study):
    compile(source, "@@SWEEP_GATING@@ replacement", "exec")
    selected, reused, planned = [], [], []
    writes = {}
    run_ids = {run_id_from_parts(study, seed, "base", 20) for seed in SEEDS}

    def open_study(impl, tag):
        selected.append((impl, tag))
        return tag, f"/memory/{impl}/{tag}"

    def experiment(**kwargs):
        records = []
        for setting in kwargs["settings"]:
            run_id = f"{kwargs['exp_tag']}__{setting['tag']}__e{kwargs['epochs']}"
            (reused if run_id in run_ids else planned).append(run_id)
            kwargs["build_model_from_setting_fn"](setting, False)
            records.append(dict(run_id=run_id, impl_name=kwargs["impl_name"],
                setting_tag=setting["tag"], seed=kwargs["seed"],
                best_val_auc=.8, brier=.1, ece=.05, img_drop_auc=.1,
                tab_drop_auc=.2, modality_balance=2 / 3, best_epoch=1,
                train_seconds=1., xai_done=False, **kwargs["extra_meta_fn"](setting)))
        return pd.DataFrame(records), {}, "/memory/summary"

    noop = lambda *args, **kwargs: None
    ns = dict(CFG=types.SimpleNamespace(epochs=20, seeds=SEEDS), os=os, pd=pd,
              open_study=open_study, run_model_experiment=experiment,
              fam=types.SimpleNamespace(build_adaptive_gating=noop), SPEC=None,
              plot_val_auc_vs_epoch_sweep=noop, display=noop)
    with patch.object(pd.DataFrame, "to_csv", lambda df, p, **kw: writes.update({p: df.copy()})):
        exec(source, ns)
    assert selected == [("adaptive_gating", study)]
    assert len(reused) == 6 and set(reused) == run_ids
    assert len(planned) == 24 and len(set(planned)) == 24
    assert ns["ag_combined"].groupby("entropy_weight").size().eq(6).all()
    assert ns["ag_agg"]["count"].eq(6).all()
    assert {"mean", "std", "count", "brier_mean", "ece_mean", "balance_mean"} <= set(ns["ag_agg"])


def test_integrated_studies_and_producers():
    families = ("vanilla", "fixed_scaling", "learnable_scaling", "loss_regularization",
                "controlled_fusion", "adaptive_gating")
    markers = ("VANILLA", "FIXED", "LEARNABLE", "LOSSREG", "CF", "GATING")
    for dataset, studies in STUDIES.items():
        base = ROOT / f"notebooks/modality_attendance/{dataset}/{dataset}_modattendance"
        train = notebook_cells(Path(f"{base}_train.ipynb"))
        posthoc = notebook_cells(Path(f"{base}_posthoc.ipynb"))
        for path in (Path(f"{base}_train.ipynb"), Path(f"{base}_posthoc.ipynb")):
            for cell in json.loads(path.read_text())["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), str(path), "exec")
        parameters = next(s for s in posthoc if s.startswith("# @@PARAMETERS@@"))
        canonical = next(n.value for n in ast.parse(parameters).body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "CANONICAL" for t in n.targets))
        observed = {ast.literal_eval(k): ast.literal_eval(v.args[-1])
                    for k, v in zip(canonical.keys, canonical.values)}
        assert observed == dict(zip(families, studies)), dataset
        for marker, study in zip(markers, studies):
            source = next(s for s in train if s.startswith(f"# @@SWEEP_{marker}@@"))
            tags = [ast.literal_eval(n.value) for n in ast.parse(source).body
                    if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "STUDY_TAG" for t in n.targets)]
            assert tags == [study], (dataset, marker, tags)
        subset = next(s for s in posthoc if s.startswith("# @@SUBSET@@ 1/3"))
        jobs = next(ast.literal_eval(n.value) for n in ast.parse(subset).body
                    if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "JOBS_TO_RUN" for t in n.targets))
        assert set(jobs) == {"gradcam", "shap"}, dataset
        lore = next(s for s in posthoc if s.startswith("# @@SUBSET@@ 3/3"))
        assert "xs.resume_verified" in lore and "lore_config" in lore, dataset
        assert f"STRAT_LORE_N          = {48 if dataset == 'isic24' else 60}" in lore
    legacy = ROOT / "notebooks/modality_attendance/symile_modattendance_train.ipynb"
    assert notebook_cells(legacy) == notebook_cells(
        ROOT / "notebooks/modality_attendance/symile/symile_modattendance_train.ipynb")


def parameter_fixture():
    counts = dict(vanilla=1, fixed_scaling=9, learnable_scaling=5,
                  loss_regularization=5, controlled_fusion=1, adaptive_gating=5)
    canonical, configs = {}, {}
    tags = ["ent0e+00", "base", "ent1e-03", "ent1e-02", "ent1e-01"]
    weights = [0., 1e-4, 1e-3, 1e-2, 1e-1]
    for impl, count in counts.items():
        canonical[impl] = f"/memory/{impl}/study"
        for j in range(count):
            tag = tags[j] if impl == "adaptive_gating" else f"setting{j}"
            for seed in SEEDS:
                path = f"{canonical[impl]}/seed_{seed}/{tag}__e20/config.json"
                configs[path] = dict(run_id=f"{impl}/seed_{seed}__{tag}__e20", seed=seed,
                    setting_tag=tag, best_val_auc=.8, feature_space_file="fs.keras",
                    feature_space_fingerprint="shared", entropy_weight=weights[j] if impl == "adaptive_gating" else None)
    return canonical, configs


def evaluate_parameters(canonical, configs, source=CELLS[3]):
    ns = dict(CANONICAL=canonical, glob=glob, json=json, os=os, pd=pd)
    def files(pattern):
        prefix = pattern.split("*")[0]
        return [p for p in configs if p.startswith(prefix)]
    with patch.object(glob, "glob", files), \
         patch("builtins.open", side_effect=lambda path: io.StringIO(json.dumps(configs[path]))):
        exec(source, ns)
    return ns


def test_posthoc_grid_guards():
    canonical, configs = parameter_fixture()
    ns = evaluate_parameters(canonical, configs)
    assert ns["N_CANONICAL_RUNS"] == 156
    posthoc = next(c for c in notebook_cells(POSTHOC) if c.startswith("# @@PARAMETERS@@"))
    assert '"20261020-020650"' in posthoc and '"20261020-063421"' in posthoc
    ag = pd.DataFrame([c for p, c in configs.items() if "/adaptive_gating/" in p])
    ag["run_dir"] = "/memory/run"
    for metric in ("base_auc", "img_drop_auc", "tab_drop_auc", "brier", "ece"):
        ag[metric] = .1
    ns["ag_combined"] = ag
    exec(CELLS[4], ns)
    for bad in (ag.iloc[:-1], pd.concat([ag.iloc[:-1], ag.iloc[[0]]])):
        try:
            exec(CELLS[4], dict(ns, ag_combined=bad))
        except AssertionError:
            pass
        else:
            raise AssertionError("Incomplete/duplicate gate rows must be rejected")
    broken = configs.copy()
    broken.pop(next(p for p in broken if "/adaptive_gating/" in p))
    try:
        evaluate_parameters(canonical, broken)
    except AssertionError as error:
        assert "incomplete training grid" in str(error)
    else:
        raise AssertionError("Incomplete training grid must be rejected")
    for dataset in STUDIES:
        path = ROOT / f"notebooks/modality_attendance/{dataset}/{dataset}_modattendance_posthoc.ipynb"
        cells = notebook_cells(path)
        parameters = next(s for s in cells if s.startswith("# @@PARAMETERS@@"))
        grid = parameters[parameters.index("TRAINING_SEEDS ="):parameters.index("cf_study_dir =")]
        ns = evaluate_parameters(canonical, configs, grid)
        assert ns["N_CANONICAL_RUNS"] == 156
        gate = next(s for s in cells if s.startswith("# @@GATE_DIAGNOSTICS@@"))
        guard = gate[gate.index("_TRAINED ="):gate.index("all_diag =")]
        exec(guard, dict(ns, ag_combined=ag))
        for bad in (ag.iloc[:-1], pd.concat([ag.iloc[:-1], ag.iloc[[0]]])):
            try:
                exec(guard, dict(ns, ag_combined=bad))
            except AssertionError:
                pass
            else:
                raise AssertionError(f"{dataset}: incomplete/duplicate gate rows accepted")
        try:
            evaluate_parameters(canonical, broken, grid)
        except AssertionError as error:
            assert "incomplete training grid" in str(error)
        else:
            raise AssertionError(f"{dataset}: incomplete registry accepted")


def test_posthoc_legacy_feature_space():
    tree = ast.parse((ROOT / "src/isic24/xai_driver.py").read_text())
    binding = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "assert_run_feature_space")
    helpers = {}
    exec(compile(ast.Module(body=[binding], type_ignores=[]), "FeatureSpace binding", "exec"), helpers)
    canonical, configs = parameter_fixture()
    legacy = {p: {k: v for k, v in c.items() if k != "feature_space_fingerprint"}
              for p, c in configs.items()}
    mixed = {p: (c.copy() if "/adaptive_gating/" in p and c["setting_tag"] != "base"
                 else legacy[p].copy()) for p, c in configs.items()}
    assert sum("feature_space_fingerprint" in c for c in mixed.values()) == 24
    for dataset in STUDIES:
        path = ROOT / f"notebooks/modality_attendance/{dataset}/{dataset}_modattendance_posthoc.ipynb"
        cells = notebook_cells(path)
        parameters = next(s for s in cells if s.startswith("# @@PARAMETERS@@"))
        grid = parameters[parameters.index("TRAINING_SEEDS ="):parameters.index("cf_study_dir =")]
        feature_space = next(s for s in cells if s.startswith("# @@FEATURE_SPACE@@"))
        check = feature_space[feature_space.index("for _config_path, _config in"):
                              feature_space.index("_missing =")]
        assert "assert_run_feature_space" in feature_space.splitlines()[1]
        for records in (configs, legacy, mixed):
            ns = evaluate_parameters(canonical, records, grid)
            ns.update(helpers, FEATURE_SPACE_FILE="fs.keras", FEATURE_SPACE_FP="shared")
            exec(check, ns)
            if records is legacy:
                exec(check, dict(ns, FEATURE_SPACE_FP="unrecorded-legacy-state"))
            for changed in ({"FEATURE_SPACE_FILE": "other.keras", "FEATURE_SPACE_FP": "other"},
                            {"FEATURE_SPACE_FP": "other"} if records is not legacy else
                            {"FEATURE_SPACE_FILE": "other.keras"}):
                try:
                    exec(check, dict(ns, **changed))
                except RuntimeError as error:
                    assert "[FS GUARD]" in str(error)
                else:
                    raise AssertionError(f"{dataset}: incompatible live FeatureSpace accepted")
        first = next(iter(mixed))
        for changed in ({"feature_space_file": None}, {"feature_space_file": "other.keras"},
                        {"feature_space_fingerprint": "other"}):
            bad = {**mixed, first: {**mixed[first], **changed}}
            try:
                evaluate_parameters(canonical, bad, grid)
            except AssertionError as error:
                assert "FeatureSpace" in str(error)
            else:
                raise AssertionError(f"{dataset}: inconsistent FeatureSpace metadata accepted")


def test_gate_statistics_and_stability():
    tree = ast.parse((ROOT / "src/isic24/xai_logit.py").read_text())
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"_logit", "gate_summary"}]
    ns = dict(np=np)
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "gate statistics", "exec"), ns)
    base = dict(prediction=[.1, .3, .7, .9], prediction_img_off=[.2, .3, .4, .6],
                prediction_tab_off=[.1, .4, .6, .8])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        uniform = ns["gate_summary"](pd.DataFrame(dict(base, w_img=[.5] * 4, w_tab=[.5] * 4)))
        selective = ns["gate_summary"](pd.DataFrame(dict(base, w_img=[0, 1, 0, 1], w_tab=[1, 0, 1, 0])))
    assert uniform["gate_weight_mean_img"] == selective["gate_weight_mean_img"] == .5
    assert np.isclose(uniform["gate_entropy_mean"], np.log(2))
    assert uniform["gate_weight_std_img"] == uniform["gate_selective_fraction"] == 0.
    assert selective["gate_entropy_mean"] == 0. and selective["gate_weight_std_img"] == .5
    assert selective["gate_selective_fraction"] == 1.

    tags = ["ent0e+00", "base", "ent1e-03", "ent1e-02", "ent1e-01"]
    assert [xm.setting_value("adaptive_gating", t) for t in tags] == [0., 1e-4, 1e-3, 1e-2, 1e-1]
    assert xm.setting_value("controlled_fusion", "base") is None
    assert xm.setting_value("vanilla", "baseline") is None
    assert xm.setting_value("fixed_scaling", "a0p500") == .5
    assert xm.setting_value("learnable_scaling", "init_a0p500") == .5
    assert xm.setting_value("loss_regularization", "lam1e-03") == .001
    master = pd.DataFrame([dict(impl_name="adaptive_gating", setting_tag=tag, seed=seed,
        run_id=f"{tag}/{seed}", run_dir=f"/memory/{tag}/{seed}") for tag in tags for seed in SEEDS])
    with patch.object(xm, "_run_top_k", return_value=([{0, 1}], np.array([0]))):
        pairs = xm.stability_table(master, "/memory", ["age", "view"], k=2)
    assert pairs.pair_kind.eq("adjacent_param").sum() == 24
    assert pairs.pair_kind.eq("same_param_seed").sum() == 75
    adjacent = pairs[pairs.pair_kind.eq("adjacent_param")]
    assert adjacent.seed_a.eq(adjacent.seed_b).all()
    assert adjacent.param_a.lt(adjacent.param_b).all()
    assert len(adjacent[adjacent.tag_a.eq("ent0e+00") & adjacent.tag_b.eq("base")]) == 6


def test_gamma_cells_reuse_evaluator():
    calls, figures = [], []
    rows = pd.DataFrame([dict(run_id="run", gamma_img=1 - d, gamma_tab=1 + d, val_auc=.8)
                         for d in [-.5, 0., .5]])
    module = types.ModuleType("isic24.fusion_calibration")
    def complete(path, **kwargs):
        calls.append((path, kwargs))
        return rows
    module.complete_gamma_calibration = complete
    class Plot:
        def __getattr__(self, name):
            return lambda *args, **kwargs: figures.append(args[0]) if name == "savefig" else None
    ns = dict(study_dir="selected-cf", cf_study_dir="selected-cf", os=os, pd=pd,
              cf_combined=pd.DataFrame([dict(run_id="run", run_dir="/memory/run", seed=8)]),
              xs=types.SimpleNamespace(run_dir_abs=lambda path, root: path), ART_PATH="/memory",
              plt=Plot(), display=lambda value: None)
    with patch.dict(sys.modules, {"isic24.fusion_calibration": module}), \
         patch.object(pd, "read_csv", return_value=rows), patch.object(os, "makedirs"):
        exec(CELLS[2], ns)
        exec(CELLS[5], ns)
    assert calls == [("selected-cf", {"include_attendance": True})] * 2
    assert {Path(p).name for p in figures} == {"posthoc_val_auc_vs_gamma_tab.png", "posthoc_gamma_sweep_scatter.png"}


def test_posthoc_replacement_dependencies():
    path = ROOT / "scripts/xai_2026-07-28/test_symile_posthoc_setup.py"
    spec = importlib.util.spec_from_file_location("posthoc_setup_check", path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    for dataset in STUDIES:
        checker.test_posthoc_setup(
            ROOT / f"notebooks/modality_attendance/{dataset}/{dataset}_modattendance_posthoc.ipynb")


def test_cam_overlap_export_identity_and_resume():
    from isic24 import xai_subset as xs

    for dataset in STUDIES:
        source = next(c for c in notebook_cells(
            ROOT / f"notebooks/modality_attendance/{dataset}/{dataset}_modattendance_posthoc.ipynb"
        ) if c.startswith("# @@CAM_OVERLAP@@"))
        assert "archive_legacy=True" in source and "importlib.reload(xmet)" in source

    master = pd.DataFrame([
        dict(impl_name=family, setting_tag=tag, seed=seed,
             run_id=f"{family}/seed_{seed}", run_dir=f"/memory/{family}/seed_{seed}")
        for family, tag in (("vanilla", "baseline"), ("controlled_fusion", "base"))
        for seed in (8, 47)
    ])
    cams = np.tile(np.array([[1., 0.], [0., 0.]]), (3, 1, 1))
    path = "/memory/aggregate/cam_overlap.csv"
    saved, writes = {}, []

    def read_csv(filename, nrows=None, usecols=None):
        frame = saved[str(filename)]
        return (frame.iloc[:nrows] if nrows is not None else
                frame[usecols] if usecols is not None else frame).copy()

    def save_csv(frame, filename, **kwargs):
        saved[str(filename)] = frame.copy()
        writes.append(str(filename))

    with patch.object(xm, "_run_cams", return_value=(cams, "sample")), \
         patch.object(xm._paths, "run_dir_abs", side_effect=lambda p, root: p), \
         patch.object(xs.os.path, "exists", side_effect=lambda p: str(p) in saved), \
         patch.object(pd, "read_csv", side_effect=read_csv), \
         patch.object(pd.DataFrame, "to_csv", save_csv), \
         patch.object(xs, "aggregate_provenance", return_value={"dataset": "brset/plain"}), \
         patch("builtins.open", mock_open()):
        out = xm.cam_overlap_table(master, "/memory", out_dir="/memory/aggregate")
        assert out.pair_kind.value_counts().to_dict() == {"cross_family": 2, "seed_null": 2}
        meta = master.set_index("run_id")
        for row in out.itertuples():
            assert (meta.loc[row.run_id_a, "impl_name"], meta.loc[row.run_id_a, "seed"]) == (row.impl_a, row.seed_a)
            assert (meta.loc[row.run_id_b, "impl_name"], meta.loc[row.run_id_b, "seed"]) == (row.impl_b, row.seed_b)
        assert set(out.run_id_a) | set(out.run_id_b) == set(master.run_id)
        np.testing.assert_allclose(out[["iou_half", "iou_top"]], 1.)
        again = xm.cam_overlap_table(master, "/memory", out_dir="/memory/aggregate")
        pd.testing.assert_frame_equal(out, again)
        assert len(writes) == 2

        saved[path] = out.drop(columns=["run_id_a", "run_id_b"])
        try:
            xm.cam_overlap_table(master, "/memory", out_dir="/memory/aggregate")
        except RuntimeError as error:
            assert "carries no run identity" in str(error)
        else:
            raise AssertionError("Anonymous legacy CAM tables must still require archiving")
        assert len(writes) == 2

        legacy = saved[path].copy()
        sidecar = path.replace(".csv", ".provenance.json")
        saved[sidecar] = {"dataset": "brset/plain", "n_runs": 4}
        archive = "/memory/aggregate/cam_overlap.legacy-first"
        archive2 = "/memory/aggregate/cam_overlap.legacy-second"

        def rename(source, target):
            assert target not in saved, "Archiving must preserve existing files"
            saved[target] = saved.pop(source)

        with patch.object(xm.tempfile, "mkdtemp", side_effect=[archive, archive2]) as allocate, \
             patch.object(xm.os, "rename", side_effect=rename):
            recovered = xm.cam_overlap_table(
                master, "/memory", out_dir="/memory/aggregate", archive_legacy=True)
            pd.testing.assert_frame_equal(recovered, out)
            pd.testing.assert_frame_equal(saved[archive + "/cam_overlap.csv"], legacy)
            assert saved[archive + "/cam_overlap.provenance.json"] == {
                "dataset": "brset/plain", "n_runs": 4}
            assert sidecar not in saved
            allocate.assert_called_once_with(prefix="cam_overlap.legacy-", dir="/memory/aggregate")

            xm.cam_overlap_table(master, "/memory", out_dir="/memory/aggregate", archive_legacy=True)
            assert allocate.call_count == 1, "Current exports must not be archived on rerun"
            saved[path] = legacy.copy()
            xm.cam_overlap_table(master, "/memory", out_dir="/memory/aggregate", archive_legacy=True)
            pd.testing.assert_frame_equal(saved[archive2 + "/cam_overlap.csv"], legacy)
            assert archive2 + "/cam_overlap.provenance.json" not in saved
            assert allocate.call_count == 2
        assert len(writes) == 5

        saved[path] = out.copy()
        other = master.assign(run_id="other/" + master.run_id)
        with patch.object(xm.tempfile, "mkdtemp") as allocate:
            try:
                xm.cam_overlap_table(other, "/memory", out_dir="/memory/aggregate", archive_legacy=True)
            except RuntimeError as error:
                assert "another study" in str(error)
            else:
                raise AssertionError("CAM tables from a disjoint study must not be overwritten")
            allocate.assert_not_called()
        assert len(writes) == 5
        with patch.object(xm, "_run_cams", side_effect=[(cams, "sample"), (cams, "other")]):
            try:
                xm.cam_overlap_table(master, "/memory", out_dir="/memory/aggregate")
            except RuntimeError as error:
                assert "different samples" in str(error)
            else:
                raise AssertionError("CAM pairs must use the same explanation sample")
        assert len(writes) == 5


if __name__ == "__main__":
    test_training_replacement()
    test_integrated_studies_and_producers()
    test_posthoc_grid_guards()
    test_posthoc_legacy_feature_space()
    test_gate_statistics_and_stability()
    test_gamma_cells_reuse_evaluator()
    test_posthoc_replacement_dependencies()
    test_cam_overlap_export_identity_and_resume()
    print("PASS: four integrated dataset grids, canonical studies, producer dependencies, gate entropy, stability and CAM export identity/resume; writes mocked.")
