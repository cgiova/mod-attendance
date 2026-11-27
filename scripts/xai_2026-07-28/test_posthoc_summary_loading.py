"""Check notebook summary loading with padded CSVs; write no artifacts."""
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def test_posthoc_summary_loading():
    root = Path(__file__).resolve().parents[2]
    for dataset in ("brset", "cmmd", "isic24", "symile"):
        check_summary_loading(root / f"notebooks/modality_attendance/{dataset}/{dataset}_modattendance_posthoc.ipynb")


def check_summary_loading(notebook):
    cells = ["".join(cell["source"]) for cell in json.loads(notebook.read_text())["cells"]]
    parameters = next(source for source in cells if source.startswith("# @@PARAMETERS@@"))
    loading = parameters[parameters.index("cf_combined ="):parameters.index('print(f"canonical')]
    gate = next(source for source in cells if source.startswith("# @@GATE_DIAGNOSTICS@@"))
    guard = gate[gate.index("_TRAINED ="):gate.index("all_diag =")]

    seeds = [8, 47, 53, 71, 90, 101]
    settings = [("ent0e+00", 0.0), ("base", 1e-4), ("ent1e-03", 1e-3),
                ("ent1e-02", 1e-2), ("ent1e-01", 1e-1)]
    clean = pd.DataFrame([
        dict(run_id=f"study/seed_{seed}__{tag}__e20",
             run_dir=f"/data with spaces/runs/study/seed_{seed}/{tag}__e20",
             setting_tag=tag, seed=seed, entropy_weight=weight, base_auc=.9,
             img_drop_auc=.1, tab_drop_auc=.05, brier=.1, ece=.02,
             model_name=f"adaptive_gating__{tag}")
        for tag, weight in settings for seed in seeds
    ])
    clean_csv = clean.to_csv(index=False)
    padded_csv = "\n".join(
        ",".join(f"  {value}  " for value in line.split(","))
        for line in clean_csv.splitlines()
    )
    read_csv = pd.read_csv
    for cf_csv, ag_csv in ((clean_csv, padded_csv), (padded_csv, clean_csv)):
        summaries = {
            os.path.join("cf", "combined_summary.csv"): cf_csv,
            os.path.join("ag", "combined_summary.csv"): ag_csv,
        }
        namespace = dict(pd=pd, os=os, cf_study_dir="cf", ag_study_dir="ag",
                         TRAINING_SEEDS=seeds, EXPECTED_SETTINGS={"adaptive_gating": 5})
        with patch.object(pd, "read_csv", side_effect=lambda path: read_csv(io.StringIO(summaries[path]))):
            exec(compile(loading, str(notebook), "exec"), namespace)
        for name in ("cf_combined", "ag_combined"):
            pd.testing.assert_frame_equal(namespace[name], clean)
        exec(compile(guard, str(notebook), "exec"), namespace)


if __name__ == "__main__":
    test_posthoc_summary_loading()
    print("PASS: summary loading strips padding, preserves values, and passes the gate guard.")
