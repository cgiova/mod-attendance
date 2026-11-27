"""Verify frozen-gamma orchestration with synthetic arrays and mocked artifact I/O."""

import ast
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/isic24/cf_xai.py"


def module():
    tree = ast.parse(SOURCE.read_text())
    tree.body = [node for node in tree.body
                 if not isinstance(node, ast.ImportFrom) or node.level == 0]
    result = ModuleType("cf_xai_test")
    result.__file__ = str(SOURCE)
    exec(compile(tree, str(SOURCE), "exec"), result.__dict__)
    result.xf = SimpleNamespace(
        FAITH_SUBSET_IDX=np.array([0, 2]), ACTIVE_SUBSET_FINGERPRINT="sample",
        FAITH_BUDGETS=(.05, .1, .2, .4, .6), FAITH_RNG_SEED=0, FAITH_N_RANDOM=3,
        FAITH_BLUR_SIGMA=8, FAITH_IMG_REPLACE="blur", FAITH_DO_INSERTION=True,
        SIGNED_MODES=("abs", "posev", "negev", "target"),
        _CTX=SimpleNamespace(feature_space_file="fs", feature_space_fingerprint="space",
                             feat_cols=["a", "b"],
                             validation_df=pd.DataFrame({"target": [0, 0, 1]})),
        _build_faith_subset=Mock(return_value=(np.zeros((2, 2)), np.zeros((2, 1)), np.array([0, 1]))),
        _encode_raw=Mock(return_value=np.zeros((2, 2))),
        _score_logit=Mock(return_value=np.zeros(2)), explanation_faithfulness=Mock())
    result.xd = SimpleNamespace(
        SHAP_KWARGS={"subset_idx": np.array([0, 2]), "subset_fingerprint": "sample", "rng_seed": 0},
        GRADCAM_KWARGS={"use_guided_grads": True}, GRADCAM_CB_KWARGS={"class_index": 0},
        GRADCAM_MODES=("context", "image_only"), GRADCAM_BATCH=16,
        LORE_SA_KWARGS={"rng_seed": 0, "max_samples": 256}, LORE_SA_DEVICE=None,
        lore_sa_config=Mock(return_value={"fingerprint": "lore-v2"}),
        assert_run_feature_space=Mock())
    result.xs = SimpleNamespace(stamp_row_ids=lambda idx: {"row_id": np.array([f"id{i}" for i in idx])})
    result.paths = SimpleNamespace(run_dir_abs=lambda d, root: str(d))
    return result


class Arrays(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def plan(seeds=(8, 47)):
    return pd.DataFrame([
        dict(source_run_id=f"study/seed_{seed}", seed=seed, delta=delta,
             gamma_img=1-delta, gamma_tab=1+delta, source_run_dir=f"/memory/source/seed_{seed}",
             setting_tag=f"g{delta}", run_id=f"seed_{seed}@{delta}",
             base_model_name="fusion", best_epoch=2, impl_name="controlled_fusion",
             val_auc=.7, brier=.2, ece=.1, img_drop_auc=.1, tab_drop_auc=.05, modality_balance=1/3)
        for seed in seeds for delta in np.arange(-.5, .501, .125)])


class Weight:
    def assign(self, value):
        self.value = value


class Model:
    def __init__(self, device):
        self.device = device
        self.gamma_img, self.gamma_tab = Weight(), Weight()

    def get_layer(self, name):
        assert name == "fusion_with_gamma"
        return self


class GammaTests(unittest.TestCase):
    def test_complete_grid_pilot_resume_and_cpu_failure(self):
        for pilot, resumed, fail_cpu in [(False, False, False), (True, False, False),
                                        (False, True, False), (True, False, True)]:
            with self.subTest(pilot=pilot, resumed=resumed, fail_cpu=fail_cpu):
                cf = module()
                loads, gpu, cpu, writes = [], [], [], []

                def load(source, device=None):
                    model = Model(device)
                    loads.append((source, model))
                    return model, {}

                def check(model, row, directory, *args):
                    self.assertEqual(model.gamma_img.value, row["gamma_img"])
                    self.assertEqual(model.gamma_tab.value, row["gamma_tab"])
                    self.assertIn("/interventions/controlled_fusion/", str(directory))
                    if model.device:
                        np.testing.assert_array_equal(cf.xd.LORE_SA_KWARGS["subset_idx"], [0, 2])
                        self.assertEqual(model.device, "/CPU:0")
                        cpu.append(row)
                        if fail_cpu:
                            raise RuntimeError("Injected CPU failure")
                    else:
                        gpu.append(row)

                cf.xd.load_run_model = Mock(side_effect=load)
                backend = SimpleNamespace(clear_session=Mock())
                with ExitStack() as stack:
                    stack.enter_context(patch.dict(sys.modules, {"keras": SimpleNamespace(backend=backend)}))
                    for name, value in {
                        "_plan": Mock(return_value=(plan((8, 47, 53, 71, 90, 101)), np.array([0, 2]))),
                        "version": Mock(return_value="test"), "_sha256": Mock(return_value="digest"),
                        "_write_json": Mock(side_effect=lambda path, data: writes.append((path, data))),
                        "_check_array_sample": Mock(), "_controls": Mock(return_value=[]),
                        "_complete": Mock(return_value=resumed), "_gpu": Mock(side_effect=check),
                        "_callbacks": Mock(side_effect=check),
                        "_stage": Mock(side_effect=lambda d, n, p, action: action()),
                        "_summaries": Mock(return_value=pd.DataFrame()),
                    }.items():
                        stack.enter_context(patch.object(cf, name, value))
                    stack.enter_context(patch.object(Path, "exists", return_value=False))
                    stack.enter_context(patch.object(Path, "glob", lambda p, pattern: [p / "best.keras"]))
                    stack.enter_context(patch.object(Path, "mkdir"))
                    stack.enter_context(patch.object(pd.DataFrame, "to_csv"))
                    stack.enter_context(patch.object(np, "load", return_value=Arrays(s_orig=np.zeros(2))))
                    if fail_cpu:
                        with self.assertRaisesRegex(RuntimeError, "Injected CPU"):
                            cf.run_cf_gamma_xai("/memory/artifacts/demo", "/memory/study", pilot=pilot)
                    else:
                        cf.run_cf_gamma_xai("/memory/artifacts/demo", "/memory/study", pilot=pilot)
                self.assertEqual(cf.xd.LORE_SA_KWARGS, {"rng_seed": 0, "max_samples": 256})
                self.assertIsNone(cf.xd.LORE_SA_DEVICE)
                expected = 3 if pilot else 54
                self.assertEqual(len(gpu), 0 if resumed else expected)
                if resumed:
                    self.assertFalse(loads)
                elif not fail_cpu:
                    self.assertEqual(len(cpu), expected)
                if pilot:
                    self.assertEqual({row["delta"] for row in gpu}, {-.5, 0, .5})
                self.assertEqual(len(writes), expected)
                self.assertTrue(all("checkpoint_sha256" in data for _, data in writes))

    def test_gradient_reads_intervention_arrays(self):
        cf, model = module(), Model(None)
        model.gamma_img.assign(.5)
        model.gamma_tab.assign(1.5)
        directory = Path("/memory/interventions/controlled_fusion/study/seed_8/high")
        row = plan().iloc[0].to_dict()
        with patch.object(cf, "_stage", side_effect=lambda d, n, p, action: action()), \
             patch.object(cf, "_callbacks") as callbacks, \
             patch.object(cf, "_permutation"), patch.object(cf, "_signed"):
            cf._gpu(model, row, directory, ())
        self.assertEqual(callbacks.call_args_list[0].args[0], model)
        self.assertEqual(callbacks.call_args_list[0].args[3], ("shap",))
        faithful = cf.xf.explanation_faithfulness.call_args
        self.assertEqual(faithful.args[:2], (model, str(directory)))
        self.assertEqual(faithful.kwargs["out_dir"], str(directory / "xai/faith_gradient"))
        self.assertNotIn(row["source_run_dir"], str(faithful))

    def test_stage_resume_protects_and_recovers(self):
        cf, directory = module(), Path("/memory/intervention")
        files, ledger = set(), {}
        output = directory / "xai/curves.npz"

        def action():
            files.add(output)

        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "exists", lambda p: p == directory / "completed.json" and bool(ledger)))
            stack.enter_context(patch.object(Path, "read_text", lambda p: json.dumps(ledger)))
            stack.enter_context(patch.object(Path, "is_file", lambda p: p in files))
            stack.enter_context(patch.object(Path, "glob", lambda p, pattern: list(files)))
            stack.enter_context(patch.object(cf, "_sha256", return_value="digest"))
            stack.enter_context(patch.object(cf, "_write_json", side_effect=lambda p, data: ledger.update(data)))
            compute = Mock(side_effect=action)
            cf._stage(directory, "faith", ("xai/*.npz",), compute)
            cf._stage(directory, "faith", ("xai/*.npz",), compute)
            self.assertEqual(compute.call_count, 1)
            files.clear()
            with self.assertRaisesRegex(RuntimeError, "missing completed"):
                cf._stage(directory, "faith", ("xai/*.npz",), compute)
            ledger.clear()
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                cf._stage(directory, "faith", ("xai/*.npz",), Mock(side_effect=RuntimeError("interrupted")))
            self.assertFalse(ledger)
            cf._stage(directory, "faith", ("xai/*.npz",), compute)
            self.assertEqual(compute.call_count, 2)

    def test_plan_rejects_wrong_grid_and_sample(self):
        cf = module()
        root = Path("/memory/artifacts/demo")
        study = root / "runs/controlled_fusion/study"
        master = pd.DataFrame([dict(impl_name="controlled_fusion", seed=s, run_id=f"study/seed_{s}",
                                   run_dir=str(study / f"seed_{s}/base"), best_epoch=2, model_name="fusion")
                               for s in [8, 47, 53, 71, 90, 101]])
        gamma = pd.DataFrame([dict(run_id=r.run_id, delta=d, gamma_img=1-d, gamma_tab=1+d,
                                   val_auc=.7, brier=.2, ece=.1, img_drop_auc=.1,
                                   tab_drop_auc=.05, modality_balance=1/3)
                              for r in master.itertuples() for d in np.arange(-.5, .501, .125)])
        lore = pd.DataFrame(dict(val_index=[0, 2], row_id=["id0", "id2"]))

        def read_csv(path, **kwargs):
            if Path(path).name == "master_results_rows.csv":
                return master.copy()
            if Path(path).name == "controlled_fusion_posthoc_gamma_summary.csv":
                return gamma.copy()
            return lore.copy()

        with patch.object(pd, "read_csv", side_effect=read_csv), \
             patch.object(Path, "read_text", return_value=json.dumps({"gamma_img": 1, "gamma_tab": 1})):
            result, indices = cf._plan(root, study)
            self.assertEqual(len(result), 54)
            np.testing.assert_array_equal(indices, [0, 2])
            gamma.loc[0, "gamma_tab"] = 42
            with self.assertRaisesRegex(ValueError, "Gamma coefficients"):
                cf._plan(root, study)
            gamma.loc[0, "gamma_tab"] = 1 + gamma.loc[0, "delta"]
            lore.loc[0, "row_id"] = "wrong"
            with self.assertRaisesRegex(ValueError, "nested LORE sample"):
                cf._plan(root, study)

    def test_clipping_observer_restores_scoring(self):
        cf = module()
        original = Mock(side_effect=lambda p, eps=1e-6: np.asarray(p))
        cf.xf._logit_np = original
        cf.xf.aligned_tab_attributions = Mock(side_effect=lambda *a, **k: cf.xf._logit_np([np.nan]))
        with patch.object(Path, "exists", return_value=False), self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            cf._permutation(None, {"best_epoch": 2}, Path("/memory/gamma"),
                            (np.zeros((2, 2)), np.zeros((2, 1)), np.array([0, 1])))
        self.assertIs(cf.xf._logit_np, original)

    def test_summaries_keep_gamma_and_seed_units(self):
        cf = module()
        table = plan()
        table["output_dir"] = [f"/memory/interventions/{i}" for i in range(len(table))]
        lore = pd.DataFrame(dict(neigh_n_classes=[2, 1], rule_len=[3., np.nan],
                                 fid_cv_bacc=[.8, np.nan], neigh_unique_frac=[.4, .6], error=[None, None]))

        def strata(root, files, verbose):
            self.assertEqual(len(files), len(table) * 6)
            return pd.DataFrame([
                dict(run_id=r.run_id, source="canonical", mode="abs", method="tabular",
                     direction="del", q_max=.2, stratum="all", gain_norm=float(r.seed))
                for r in table.itertuples()])

        cf.xf.stratified_table = Mock(side_effect=strata)
        with patch.object(cf, "_check_array_sample"), patch.object(pd, "read_csv", return_value=lore), \
             patch.object(pd.DataFrame, "to_csv") as write:
            summary = cf._summaries(table, Path("/memory/summary"))
        self.assertEqual(len(summary), 9)
        self.assertTrue(summary.n_seeds.eq(2).all())
        np.testing.assert_allclose(summary.canonical_abs_tabular_del_q20_mean, (8+47)/2)
        np.testing.assert_allclose(summary.canonical_abs_tabular_del_q20_std, np.std([8, 47], ddof=1))
        np.testing.assert_allclose(summary.lore_rule_length_mean, 3)
        self.assertEqual(write.call_count, 3)

    def test_structural_controls_keep_clipping_visible(self):
        cf = module()
        rows = plan((8,)).loc[lambda d: d.delta.isin([0, .5])].copy()
        rows["output_dir"] = [f"/memory/gamma/{d}" for d in rows.delta]
        phi = np.array([[1., -2.], [.5, .2]])

        def load(path):
            path = str(path)
            high = "/0.5/" in path
            if "aligned_attributions" in path:
                return Arrays(phi=phi * (1.5 if high else 1))
            cams = np.array([[[0., 1.]], [[0., 1.]]])
            if high:
                cams[1] = [[1., 0.]]
            return Arrays(cams=cams, preds=np.array([.5, 1. if high else .9]))

        def read(path):
            return json.dumps(dict(clipped=int("/0.5/" in str(path))))

        with patch.object(np, "load", side_effect=load), patch.object(Path, "read_text", read):
            controls = cf._controls(rows.to_dict("records"))
        self.assertEqual(controls[1]["permutation_scale_max_abs_error"], 0)
        self.assertFalse(controls[1]["permutation_scaling_unclipped"])
        self.assertEqual(controls[1]["cam_context_unclipped_n"], 1)
        self.assertEqual(controls[1]["cam_context_max_abs_delta_unclipped"], 0)

    def test_array_identity_includes_row_order(self):
        cf = module()
        saved = Arrays(val_index=np.array([0, 2]), row_id=np.array(["id0", "id2"]),
                       labels=np.array([0, 1]))
        with patch.object(np, "load", return_value=saved):
            cf._check_array_sample(Path("/memory/curves.npz"), np.array([0, 2]))
            saved["val_index"] = np.array([2, 0])
            with self.assertRaisesRegex(RuntimeError, "Sample identity mismatch"):
                cf._check_array_sample(Path("/memory/curves.npz"), np.array([0, 2]))

    def test_notebook_calls_are_identical(self):
        sources = []
        for name in ["isic24", "brset", "symile", "cmmd"]:
            path = ROOT / "notebooks/modality_attendance" / name / f"{name}_modattendance_posthoc.ipynb"
            cells = json.loads(path.read_text())["cells"]
            index = next(i for i, c in enumerate(cells)
                         if "@@CF_XAI_GAMMA@@" in "".join(c.get("source", [])))
            cell = cells[index]
            self.assertIn("@@STOP@@", "".join(cells[index-1]["source"]))
            source = "".join(cell["source"])
            ast.parse(source)
            self.assertEqual(sum(line.lstrip().startswith("#") for line in source.splitlines()), 1)
            self.assertIn('CANONICAL["controlled_fusion"]', source)
            sources.append(source)
        self.assertEqual(len(set(sources)), 1)


if __name__ == "__main__":
    unittest.main()
