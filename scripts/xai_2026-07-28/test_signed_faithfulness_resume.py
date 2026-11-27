"""Check signed-mode recovery with synthetic curves and all artifact I/O mocked."""

import ast
import copy
from contextlib import contextmanager
import fnmatch
import glob
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/isic24/xai_faithfulness.py"
FUNCTIONS = {"_trapz", "persample_gains", "gain_window", "mode_sign",
             "attribution_order_key", "_target_gain_persample", "_load_signed_mode",
             "aligned_signed_faithfulness"}
CONSTANTS = {"SIGNED_MODES", "SIGN_FLIPPED_MODES", "PERSAMPLE_SIGNED_MODES",
             "FAITH_BUDGETS", "FAITH_DO_INSERTION", "FAITH_SUBSET_IDX"}
TREE = ast.parse(SOURCE.read_text())
BODY = [node for node in TREE.body if (
    isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS
    or isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id in CONSTANTS for target in node.targets))]


class Curves(dict):
    @property
    def files(self):
        return list(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@contextmanager
def fixture(completed=("abs", "posev", "negev")):
    ns = dict(np=np, json=json, os=os, __package__="isic24")
    exec(compile(ast.Module(body=BODY, type_ignores=[]), str(SOURCE), "exec"), ns)
    ns["ACTIVE_SUBSET_FINGERPRINT"] = "sample-fingerprint"
    indices = np.array([4, 9, 12])
    labels = np.array([0, 1, 0])
    ids = np.array(["lesion-a", "lesion-b", "lesion-c"])
    q = np.array((0.,) + ns["FAITH_BUDGETS"])
    prefix = "/memory/shap_tab/faithfulness_aligned_signed_"
    files = {}

    def outputs(mode):
        amplitude = np.array([1., 2., 3.])[:, None]
        curves = Curves(val_index=indices.copy(), labels=labels.copy(), row_id=ids.copy(),
                        q_grid=q.copy())
        result = dict(subset_fingerprint=ns["ACTIVE_SUBSET_FINGERPRINT"],
                      n_faith_samples=len(labels), tab_ranking="override", q_grid=q.tolist())
        for direction in ("del", "ins"):
            curves[f"{direction}_exp_tabular"] = amplitude * q
            curves[f"{direction}_rand_tabular"] = np.zeros((len(labels), len(q)))
            result[f"faithfulness_tabular_{direction}_gain"] = float(np.mean(
                ns["persample_gains"](curves, "tabular", direction)))
        return result, curves

    for mode in completed:
        result, curves = outputs(mode)
        files[f"{prefix}{mode}.json"] = result
        files[f"{prefix}{mode}_curves.npz"] = curves

    def compute(*args, out_name, **kwargs):
        mode = out_name.removeprefix("faithfulness_aligned_signed_")
        assert f"{prefix}{mode}.json" not in files
        assert f"{prefix}{mode}_curves.npz" not in files
        result, curves = outputs(mode)
        files[f"{prefix}{mode}.json"] = result
        files[f"{prefix}{mode}_curves.npz"] = curves
        return result

    ns["explanation_faithfulness"] = Mock(side_effect=compute)
    ns["aligned_tab_attributions"] = Mock(side_effect=AssertionError("Unexpected attribution pass"))
    ns["_write_aligned_signed_summary"] = Mock()
    package = ModuleType("isic24")
    subset = ModuleType("isic24.xai_subset")
    subset.out_dir_for = lambda *args: "/memory/shap_tab"
    subset.stamp_row_ids = lambda idx: {"row_id": ids}
    driver = ModuleType("isic24.xai_driver")
    driver.assert_writable = Mock(side_effect=RuntimeError("Write guard remains active"))
    args = dict(model=None, run_dir="/memory", best_epoch=1, raw_df=np.zeros((3, 2)),
                imgs=np.zeros((3, 1)), labels=labels, subset_idx=indices,
                phi=np.array([[1., -2.], [2., -1.], [-1., 2.]]), verbose=False)
    with patch.dict(sys.modules, {"isic24": package, "isic24.xai_subset": subset,
                                 "isic24.xai_driver": driver}), \
         patch("builtins.open", side_effect=lambda path: io.StringIO(json.dumps(files[path]))), \
         patch.object(os.path, "exists", side_effect=lambda path: path in files), \
         patch.object(np, "load", side_effect=lambda path: files[path]):
        yield ns, files, prefix, args, driver


def test_partial_resume():
    with fixture() as (ns, files, prefix, args, driver):
        original = copy.deepcopy(files)
        result, phi = ns["aligned_signed_faithfulness"](**args, resume=True)
        calls = ns["explanation_faithfulness"].call_args_list
        assert [c.kwargs["out_name"] for c in calls] == ["faithfulness_aligned_signed_target"]
        assert phi is args["phi"]
        driver.assert_writable.assert_not_called()
        for path, before in original.items():
            if path.endswith(".json"):
                assert files[path] == before
            else:
                for key in before:
                    np.testing.assert_array_equal(files[path][key], before[key])
        for mode in ns["SIGNED_MODES"]:
            suffix = "_labelsigned" if mode == "target" else "_signflipped" if mode == "negev" else ""
            curves = files[f"{prefix}{mode}_curves.npz"]
            for direction in ("del", "ins"):
                expected = np.mean(ns["persample_gains"](curves, "tabular", direction)
                                   * ns["mode_sign"](mode, args["labels"]))
                np.testing.assert_allclose(result[f"aligned_{mode}_{direction}_gain{suffix}"], expected)
                assert f"aligned_{mode}_{direction}_gainnorm{suffix}_q0p20" in result
            assert f"aligned_{mode}_n_eff_mean" in result
        ns["_write_aligned_signed_summary"].assert_called_once_with(
            "/memory/shap_tab", "faithfulness_aligned_signed", ns["SIGNED_MODES"], args["labels"])


def test_fresh_and_complete_resume():
    for completed in ((), ("abs", "posev", "negev", "target")):
        with fixture(completed) as (ns, files, prefix, args, driver):
            ns["aligned_signed_faithfulness"](**args, resume=True)
            assert ns["explanation_faithfulness"].call_count == 4 - len(completed)


def test_invalid_outputs_do_not_write():
    for defect in ("missing_json", "missing_npz", "indices", "row_ids", "labels",
                   "grid", "shape", "nonfinite", "fingerprint"):
        with fixture() as (ns, files, prefix, args, driver):
            path = f"{prefix}abs"
            curves, result = files[f"{path}_curves.npz"], files[f"{path}.json"]
            if defect == "missing_json":
                del files[f"{path}.json"]
            elif defect == "missing_npz":
                del files[f"{path}_curves.npz"]
            elif defect == "indices":
                curves["val_index"] = curves["val_index"][::-1]
            elif defect == "row_ids":
                curves["row_id"] = curves["row_id"][::-1]
            elif defect == "labels":
                curves["labels"] = 1 - curves["labels"]
            elif defect == "grid":
                curves["q_grid"] = curves["q_grid"] * .5
            elif defect == "shape":
                curves["ins_rand_tabular"] = np.zeros((1, 1))
            elif defect == "nonfinite":
                curves["del_exp_tabular"][0, 0] = np.nan
            elif defect == "fingerprint":
                result["subset_fingerprint"] = "other-sample"
            try:
                ns["aligned_signed_faithfulness"](
                    **{**args, "phi": None}, modes=("target", "abs"), resume=True)
            except RuntimeError as error:
                assert "[SIGNED RESUME]" in str(error), defect
            else:
                raise AssertionError(f"Unsafe resume accepted: {defect}")
            ns["aligned_tab_attributions"].assert_not_called()
            ns["explanation_faithfulness"].assert_not_called()
            ns["_write_aligned_signed_summary"].assert_not_called()


def test_guard_and_notebook_integration():
    with fixture() as (ns, files, prefix, args, driver):
        try:
            ns["aligned_signed_faithfulness"](**args)
        except RuntimeError as error:
            assert "Write guard remains active" in str(error)
        else:
            raise AssertionError("Default call bypassed the write guard")
        driver.assert_writable.assert_called_once()
    for dataset in ("isic24", "brset", "cmmd", "symile"):
        p = ROOT / f"notebooks/modality_attendance/{dataset}/{dataset}_modattendance_posthoc.ipynb"
        notebook = json.loads(p.read_text())
        source = next("".join(c.get("source", [])) for c in notebook["cells"]
                      if "".join(c.get("source", [])).startswith("# @@SUBSET@@ 2/3"))
        compile(source, str(p), "exec")
        call = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "aligned_signed_faithfulness")
        assert any(kw.arg == "resume" and ast.literal_eval(kw.value) for kw in call.keywords), dataset


def test_curve_only_output_is_guarded():
    path = ROOT / "src/isic24/xai_driver.py"
    tree = ast.parse(path.read_text())
    names = {"_patterns_for", "_is_artifact", "existing_xai_artifacts", "assert_writable"}
    body = [node for node in tree.body if (
        isinstance(node, ast.FunctionDef) and node.name in names
        or isinstance(node, ast.Assign) and any(isinstance(t, ast.Name)
            and t.id in {"_CLOBBERABLE", "CANONICAL_WRITE_GUARD"} for t in node.targets))]
    ns = dict(os=os, glob=glob, ALL_ENGINES=("shap_tab",))
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    orphan = "/memory/xai/shap_tab/faithfulness_aligned_signed_target_curves.npz"
    with patch.object(glob, "glob", side_effect=lambda pattern: (
            [orphan] if fnmatch.fnmatch(orphan, pattern) else [])), \
         patch.object(os.path, "isdir", return_value=False):
        try:
            ns["assert_writable"]("/memory", engines=("shap_tab",))
        except RuntimeError as error:
            assert "[XAI WRITE GUARD]" in str(error) and "target_curves.npz" in str(error)
        else:
            raise AssertionError("An orphan curves file bypassed the write guard")


if __name__ == "__main__":
    test_partial_resume()
    test_fresh_and_complete_resume()
    test_invalid_outputs_do_not_write()
    test_guard_and_notebook_integration()
    test_curve_only_output_is_guarded()
    print("PASS: signed-mode resume, sample/curve guards, sign conventions and four notebook callers; artifact I/O mocked.")
