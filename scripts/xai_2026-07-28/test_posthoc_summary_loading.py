"""Check notebook summary loading with padded CSVs; write no artifacts."""
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd


def test_posthoc_summary_loading():
    notebook = Path(__file__).resolve().parents[2] / "notebooks/modality_attendance/brset/brset_modattendance_posthoc.ipynb"
    cells = ["".join(cell["source"]) for cell in json.loads(notebook.read_text())["cells"]]
    parameters = next(source for source in cells if source.startswith("# @@PARAMETERS@@"))
    loading = parameters[parameters.index("cf_combined ="):parameters.index('print(f"canonical')]
    gate = next(source for source in cells if source.startswith("# @@GATE_DIAGNOSTICS@@"))
    guard = gate[gate.index("_TRAINED ="):gate.index("all_diag =")]

    clean = pd.DataFrame({
        "run_id": [f"study/seed_{seed}__base__e20" for seed in range(6)],
        "run_dir": [f"/data with spaces/runs/study/seed_{seed}/base__e20" for seed in range(6)],
        "setting_tag": "base", "seed": range(6), "entropy_weight": 0.0001,
        "base_auc": 0.9, "img_drop_auc": 0.1, "tab_drop_auc": 0.05,
        "brier": 0.1, "ece": 0.02, "model_name": "adaptive_gating__base",
    })
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
        namespace = dict(pd=pd, os=os, cf_study_dir="cf", ag_study_dir="ag")
        with patch.object(pd, "read_csv", side_effect=lambda path: read_csv(io.StringIO(summaries[path]))):
            exec(compile(loading, str(notebook), "exec"), namespace)
        for name in ("cf_combined", "ag_combined"):
            pd.testing.assert_frame_equal(namespace[name], clean)
        exec(compile(guard, str(notebook), "exec"), namespace)


if __name__ == "__main__":
    test_posthoc_summary_loading()
    print("PASS: summary loading strips padding, preserves values, and passes the gate guard.")
