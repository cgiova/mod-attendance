"""Regression checks for BRSET analysis; all synthetic inputs and outputs stay in memory."""
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("MPLBACKEND", "Agg")
from isic24 import xai_metrics as xm, xai_subset as xs

NOTEBOOK = ROOT / "notebooks/modality_attendance/brset/brset_modattendance_posthoc.ipynb"
CELLS = ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text())["cells"]
         if c["cell_type"] == "code"]
SEEDS = (8, 47, 53, 71, 90, 101)


def master_grid():
    rows = []
    for family, tags, study in (
        ("fixed_scaling", ("a0p000", "a0p250", "a0p500", "a1p000"), "20260924-000000"),
        ("vanilla", ("baseline",), "20260924-000001"),
    ):
        for tag in tags:
            for seed in SEEDS:
                rows.append(dict(impl_name=family, setting_tag=tag, seed=seed,
                    run_id=f"{study}/seed_{seed}__{tag}__e20",
                    run_dir=f"/memory/runs/{family}/{study}/seed_{seed}/{tag}__e20"))
    return pd.DataFrame(rows)


def test_top_k_and_agreement():
    names = list("abcdef")
    groups = xm.raw_groups(names)
    a = xm.raw_top_k(np.array([6, 5, 4, 3, 2, 1]), groups, k=2)
    b = xm.raw_top_k(np.array([1, 2, 6, 5, 4, 3]), groups, k=2)
    assert a == {"a", "b"} and b == {"c", "d"}
    assert xm._jaccard_runs([a], [b]) == 0
    isic_groups = xm.raw_groups([f"f{i}" for i in range(39)])
    assert len(xm.raw_top_k(np.arange(1, 40), isic_groups)) == 10

    null = xm._agreement_null([{"a", "b"}] * 20 + [{"a", "c"}] * 10,
                              [{"c", "d"}] * 20 + [{"a", "d"}] * 10,
                              names, n_perm=20, k=2)
    assert null["agreement_top_global_jaccard"] == 0

    buf = io.BytesIO()
    phi = np.tile([6., 5., 4., 3., 2., 1.], (60, 1))
    phi[30:] = [1., 2., 6., 5., 4., 3.]
    np.savez(buf, shap_values=phi, val_index=np.arange(60))
    payload = buf.getvalue()
    lore = pd.DataFrame(dict(val_index=np.arange(60),
        rule_text=["IF a > 0 THEN target = 1"] * 30 + ["IF c > 0 THEN target = 1"] * 30))
    real_load = np.load
    with patch.object(xm, "_shap_npz", return_value="memory"), \
         patch.object(xm, "_lore_persample", return_value=lore), \
         patch.object(np, "load", side_effect=lambda _: real_load(io.BytesIO(payload))), \
         patch.object(xm, "_agreement_null", wraps=xm._agreement_null) as null_call:
        result = xm.shap_lore_agreement("memory", names, k=2, n_perm=20)
    assert null_call.call_args.kwargs["k"] == 2
    assert result["agreement_shap_lore"] > result["agreement_null_mean"]


def test_report_inputs_and_top_k():
    master = master_grid()
    by_dir = master.set_index("run_dir")
    central = {"fixed_scaling": "a0p500", "vanilla": "baseline"}

    def gains(rd):
        tag = by_dir.loc[rd, "setting_tag"]
        value = 100. if xs.is_ablation(by_dir.loc[rd, "impl_name"], tag) else (
            .2 if tag == "a0p250" else .4)
        return dict(n=60, target_del_q20norm=value, posev_del_q20norm=value / 2)

    def lore_quality(rd):
        return dict(n=60, lore_coverage_5050=gains(rd)["target_del_q20norm"],
                    rule_len_5050=2.)

    for n_raw, requested_k, expected_k in ((6, 2, 2), (39, None, 10)):
        names = [f"f{i}" for i in range(n_raw)]
        seen_k = []

        def top_sets(rd, groups, k=xm.TOP_K):
            seen_k.append(k)
            row = by_dir.loc[rd]
            phi = np.roll(np.arange(1, n_raw + 1), int(row.seed) % n_raw)
            return [xm.raw_top_k(phi, groups, k)] * 60, np.arange(60)

        with patch.object(xs, "_run_gains", side_effect=gains) as read_faith, \
             patch.object(xs, "_lore_quality", side_effect=lore_quality) as read_lore, \
             patch.object(xm, "shap_shape_metrics", return_value=dict(
                 shap_compactness=.5, shap_k90_sparsity=.5, shap_mass=3.)), \
             patch.object(xm, "lore_shape_metrics", return_value={}), \
             patch.object(xm, "shap_lore_agreement", return_value={}) as agreement, \
             patch.object(xm, "_run_top_k", side_effect=top_sets), \
             patch.object(xs.os, "makedirs"), \
             patch.object(xs, "stamp_aggregate") as stamps, \
             patch.object(pd.DataFrame, "to_csv") as writes:
            out = xs.report(master, "/memory", kinds=(), central=central,
                            encoded_feature_names=names, shape_top_k=requested_k, verbose=False)
            assert read_faith.call_count == read_lore.call_count == len(master)
            assert all(c.args[2] == expected_k for c in agreement.call_args_list)
            assert set(seen_k) == {expected_k}
            assert set(out["shape"].shape_top_k) == {expected_k}
            assert set(out["stability"].k) == {expected_k}
            assert out["stability"].stability_at_k.min() < 1
            assert all(c.kwargs["shape_top_k"] == expected_k for c in stamps.call_args_list)
            assert all(c.kwargs["encoded_feature_names"] == names for c in stamps.call_args_list)
            family = out["by_family"]
            assert set(family.index) == set(central)
            assert (family.n_runs == len(SEEDS)).all()
            np.testing.assert_allclose(family.faith_target_q20, .4)
            np.testing.assert_allclose(family.lore_coverage, .4)
            np.testing.assert_allclose(family.lore_rule_len, 2.)
            assert writes.call_count == 3
            pooled = xm.summary_by_family("/memory", faith_df=out["faith"],
                                          lore_df=out["lore"], verbose=False)
            assert pooled.loc["fixed_scaling", "n_runs"] == 2 * len(SEEDS)
            assert np.isclose(pooled.loc["fixed_scaling", "faith_target_q20"], .3)

    for invalid in (None, 0, -1, 6, 10, 2.5):
        with patch.object(xs.os, "makedirs") as mkdir, patch.object(pd.DataFrame, "to_csv") as write:
            try:
                xs.report(master, "/memory", encoded_feature_names=list("abcdef"),
                          shape_top_k=invalid, verbose=False)
            except ValueError as error:
                assert "shape_top_k" in str(error)
            else:
                raise AssertionError(f"Invalid top-k accepted: {invalid}")
            mkdir.assert_not_called()
            write.assert_not_called()


def test_notebook_wiring_and_lore():
    for source in CELLS:
        ast.parse(source)
    config = next(s for s in CELLS if s.startswith("# @@CFG@@"))
    cfg_class = next(n for n in ast.parse(config).body if isinstance(n, ast.ClassDef))
    namespace = {}
    exec(compile(ast.Module(body=[cfg_class], type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    assert namespace["CFG"].shape_top_k == 2
    calls = [n for s in CELLS for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == "xs" and n.func.attr == "report"]
    assert len(calls) == 3
    assert all(any(k.arg == "shape_top_k" and ast.unparse(k.value) == "CFG.shape_top_k"
                   for k in c.keywords) for c in calls)

    master = master_grid()
    files = {}
    for row in master.itertuples():
        covered = np.r_[np.arange(30) < 10, np.arange(30) < 20]
        if xs.is_ablation(row.impl_name, row.setting_tag):
            covered[:] = row.setting_tag == "a1p000"
        path = f"{row.run_dir}/xai/lore_sa/epoch001/lore_sa_persample_epoch001.csv"
        files[path] = pd.DataFrame(dict(i=np.arange(60), true_label=np.repeat([0, 1], 30),
            neigh_n_classes=np.where(covered, 2, 1), rule_len=np.where(covered, 2, 999),
            fid_cv_bacc=np.where(covered, .8, 999.), neigh_pos_rate=.5, neigh_unique_frac=.7))
    cell = next(s for s in CELLS if s.startswith("# @@LORE_SECTION@@"))
    namespace = dict(xs=xs, np=np, pd=pd, os=os, glob=__import__("glob"),
                     RUNS_PATH="/memory/runs", SUMMARIES_PATH="/memory/summaries", display=lambda _: None)

    def read_csv(path):
        return (master if str(path).endswith("master_results_rows.csv") else files[path]).copy()

    with patch("glob.glob", return_value=list(files)), patch.object(pd, "read_csv", side_effect=read_csv), \
         patch.object(pd.DataFrame, "to_csv"), redirect_stdout(io.StringIO()) as stdout:
        exec(compile(cell, str(NOTEBOOK), "exec"), namespace)
    contrasts = namespace["_lore_contrasts"]
    assert len(contrasts) == 3
    assert (contrasts.n_seeds == len(SEEDS)).all()
    assert (contrasts.n_informative == len(SEEDS)).all()
    assert not any(xs.is_ablation(r.impl_name, r.setting_tag) for r in contrasts.itertuples())
    np.testing.assert_allclose(contrasts.diff_mean, 1 / 3)
    np.testing.assert_allclose(contrasts.sign_p, .03125)
    live = namespace["fam"].loc[~namespace["_controls"]]
    np.testing.assert_allclose(live.rule_len_covered, 2.)
    np.testing.assert_allclose(live.fid_cv_bacc_covered, .8)
    assert "No diabetic retinopathy" in stdout.getvalue()
    assert "benign" not in stdout.getvalue() and "malignant" not in stdout.getvalue()


def test_encoded_column_order_and_gradient_resume():
    import keras
    from types import SimpleNamespace
    from isic24 import xai_faithfulness as xf

    space = keras.utils.FeatureSpace(features={
        "z": keras.utils.FeatureSpace.float(), "a": keras.utils.FeatureSpace.float(),
    })
    encoded = space({"z": np.array([9.], dtype="float32"),
                     "a": np.array([2.], dtype="float32")})
    for name in ("brset_modattendance_posthoc.ipynb", "brset_modattendance_train.ipynb",
                 "isic24_modality_attendance_posthoc canon.ipynb",
                 "isic24_modality_attendance_train_canon.ipynb"):
        path = NOTEBOOK.parent / name
        sources = ["".join(c["source"]) for c in json.loads(path.read_text())["cells"]]
        source = next(s for s in sources if s.startswith("# @@ENCODED_NAMES@@"))
        namespace = dict(feature_space=space, feature_order=list(space.features))
        with redirect_stdout(io.StringIO()):
            exec(compile(source, str(path), "exec"), namespace)
        assert namespace["encoded_feature_names"] == ["a", "z"]
        np.testing.assert_allclose(np.asarray(encoded), [[2., 9.]])
        if "posthoc" in name:
            faith = next(s for s in sources if s.startswith("# @@FAITH_LOOP@@"))
            todo = next(n for n in ast.walk(ast.parse(faith)) if isinstance(n, ast.ListComp)
                        and isinstance(n.elt, ast.Name) and n.elt.id == "k")
            code = compile(ast.Expression(todo), str(path), "eval")
            for saved, expected in (({}, ["gradient"]),
                                    ({"encoded_feature_names": ["z", "a"]}, ["gradient"]),
                                    ({"encoded_feature_names": ["a", "z"]}, [])):
                for tag in ("a0p000", "a0p500"):
                    with patch("builtins.open", return_value=io.StringIO(json.dumps(saved))), \
                         patch.object(os.path, "exists", return_value=True):
                        planned = eval(code, dict(os=os, json=json, xs=xs, FAITH_OVERWRITE=False,
                            _impl="fixed_scaling", _row=SimpleNamespace(setting_tag=tag),
                            _js={"aligned": "aligned.json", "gradient": "gradient.json"},
                            encoded_feature_names=["a", "z"]))
                    assert planned == ([] if tag == "a0p000" else expected)

    raw = pd.DataFrame(dict(z=[1., 2., 3., 4.], a=[4., 3., 2., 1.]))
    phi = np.tile([2., 1.], (len(raw), 1))
    with patch.object(xf, "_CTX", SimpleNamespace(feat_cols=["z", "a"],
                                                 encoded_feature_names=["a", "z"])), \
         patch.object(xf, "_FAITH_TAB_BASELINE", {"z": 0., "a": 0.}), \
         patch.object(xf, "_build_faith_subset", return_value=(raw, np.zeros((4, 1)), np.array([0, 0, 1, 1]))), \
         patch.object(xf, "_load_shap_values", return_value=(phi, np.arange(4))), \
         patch.object(xf, "_encode_raw", side_effect=lambda frame: frame[["a", "z"]].to_numpy()), \
         patch.object(xf, "_score_logit", side_effect=lambda model, enc, img: enc @ np.array([2., 1.])):
        result = xf.explanation_faithfulness(None, "/memory", 1, methods=("tabular",),
                    budgets=(.5, 1.), subset_idx=np.arange(4), save=False, verbose=False)
    assert result["encoded_feature_names"] == ["a", "z"]


if __name__ == "__main__":
    for test in (test_top_k_and_agreement, test_report_inputs_and_top_k, test_notebook_wiring_and_lore,
                 test_encoded_column_order_and_gradient_resume):
        test()
        print(f"PASS: {test.__name__}")
