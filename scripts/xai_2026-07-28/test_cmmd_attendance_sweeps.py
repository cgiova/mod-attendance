"""Check the proposed CMMD notebook replacements without producing artifacts."""

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
from unittest.mock import patch
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


def notebook_cells(path):
    return ["".join(c.get("source", [])) for c in json.loads(path.read_text())["cells"]]


def test_training_replacement():
    assert len(CELLS) == 6
    for cell in CELLS:
        compile(cell, "CMMD replacement", "exec")
    source = next(c for c in notebook_cells(TRAINING) if c.startswith("# @@SWEEP_GATING@@"))
    source = CELLS[0] + "\n" + source[source.index("ag_all_results = []"):]
    tree = ast.parse(source)
    old_aggregate = next(n for n in tree.body if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == "ag_agg" for t in n.targets))
    lines = source.splitlines(keepends=True)
    source = "".join(lines[:old_aggregate.lineno - 1]) + CELLS[1] + "\n" + "".join(lines[old_aggregate.end_lineno:])
    compile(source, "@@SWEEP_GATING@@ replacement", "exec")
    selected, reused, planned = [], [], []
    writes = {}
    run_ids = {run_id_from_parts("20261020-020650", seed, "base", 20) for seed in SEEDS}

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
    assert selected == [("adaptive_gating", "20261020-020650")]
    assert len(reused) == 6 and set(reused) == run_ids
    assert len(planned) == 24 and len(set(planned)) == 24
    assert ns["ag_combined"].groupby("entropy_weight").size().eq(6).all()
    assert ns["ag_agg"]["count"].eq(6).all()
    assert {"mean", "std", "count", "brier_mean", "ece_mean", "balance_mean"} <= set(ns["ag_agg"])


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


def evaluate_parameters(canonical, configs):
    ns = dict(CANONICAL=canonical, glob=glob, json=json, os=os, pd=pd)
    def files(pattern):
        prefix = pattern.split("*")[0]
        return [p for p in configs if p.startswith(prefix)]
    with patch.object(glob, "glob", files), \
         patch("builtins.open", side_effect=lambda path: io.StringIO(json.dumps(configs[path]))):
        exec(CELLS[3], ns)
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
    notebook = json.loads(POSTHOC.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("# @@PARAMETERS@@"):
            start = source.index("N_CANONICAL_RUNS =")
            end = source.index("cf_study_dir =")
            source = source[:start] + CELLS[3] + "\n" + source[end:]
        elif source.startswith("# @@GATE_DIAGNOSTICS@@"):
            start, end = source.index("_TRAINED ="), source.index("all_diag =")
            source = source[:start] + CELLS[4] + "\n" + source[end:]
        elif source.startswith("# @@CF_GAMMA_SWEEP@@"):
            source = CELLS[5]
        cell["source"] = source.splitlines(keepends=True)
    path = ROOT / "scripts/xai_2026-07-28/test_symile_posthoc_setup.py"
    spec = importlib.util.spec_from_file_location("posthoc_setup_check", path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    checker.NOTEBOOK = POSTHOC
    with patch.object(Path, "read_text", return_value=json.dumps(notebook)):
        checker.test_posthoc_setup()


if __name__ == "__main__":
    test_training_replacement()
    test_posthoc_grid_guards()
    test_gate_statistics_and_stability()
    test_gamma_cells_reuse_evaluator()
    test_posthoc_replacement_dependencies()
    print("PASS: proposed CMMD cells, six reused plus 24 planned runs, registry guards, gate entropy and adjacent-setting stability; writes mocked.")
