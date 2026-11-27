"""Check the notebook attendance figure without writing images or summary files."""

import json
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd


def test_attendance_figure():
    notebook = Path(__file__).resolve().parents[2] / "notebooks/modality_attendance/isic24/isic24_modattendance_posthoc.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(cell["source"]) for cell in cells
                  if "".join(cell.get("source", [])).startswith("# @@ATTENDANCE_FIGURE@@"))
    rows = pd.DataFrame([
        dict(impl_name="fixed_scaling", alpha=alpha, seed=seed,
             best_val_auc=auc, modality_balance=balance)
        for alpha, aucs, balances in (
            (0.0, (.82, .86), (0., 0.)),
            (.25, (.90, .94), (.15, .35)),
            (.75, (.92, .98), (.60, .80)),
            (1.0, (.96, .98), (1., 1.)),
        )
        for seed, auc, balance in zip((8, 47), aucs, balances)
    ] + [
        dict(impl_name="vanilla", alpha=np.nan, seed=8, best_val_auc=.89, modality_balance=.2),
        dict(impl_name="vanilla", alpha=np.nan, seed=47, best_val_auc=.92, modality_balance=.6),
        dict(impl_name="learnable_scaling", alpha=.25, seed=8, best_val_auc=.1, modality_balance=.9),
    ])
    root = Path("/memory/artifacts/demo/processed")
    namespace = dict(ART_PATH=str(root), SUMMARIES_PATH=str(root / "summaries"))
    with patch.object(pd, "read_csv", return_value=rows) as read, \
            patch.object(Path, "mkdir") as mkdir, \
            patch.object(Figure, "savefig", autospec=True) as save, \
            patch.object(plt, "show") as show, patch("builtins.print"):
        try:
            exec(compile(source, str(notebook), "exec"), namespace)
            fig = namespace["_attendance_fig"]
            auc_ax, balance_ax = fig.axes
            assert auc_ax.get_shared_x_axes().joined(auc_ax, balance_ax)
            for ax, means in ((auc_ax, (.92, .95)), (balance_ax, (.25, .70))):
                curve, controls = ax.containers
                np.testing.assert_allclose(curve.lines[0].get_xdata(orig=False), [.25, .75])
                np.testing.assert_allclose(curve.lines[0].get_ydata(orig=False), means)
                np.testing.assert_allclose(controls.lines[0].get_xdata(orig=False), [0., 1.])
                assert controls.lines[0].get_linestyle() == "None"
                assert curve.lines[0].get_linestyle() == "-"
            np.testing.assert_allclose(auc_ax.containers[1].lines[0].get_ydata(orig=False), [.84, .97])
            np.testing.assert_allclose(balance_ax.containers[1].lines[0].get_ydata(orig=False), [0., 1.])
            np.testing.assert_allclose(auc_ax.lines[-1].get_ydata(orig=False), [.905, .905])
            np.testing.assert_allclose(balance_ax.lines[-2].get_ydata(orig=False), [.4, .4])
            np.testing.assert_allclose(balance_ax.lines[-1].get_ydata(orig=False), [.5, .5])
            bars = balance_ax.containers[0].lines[2][0].get_segments()
            np.testing.assert_allclose(bars[0][:, 1], .25 + np.array([-1., 1.]) * np.std([.15, .35], ddof=1))
            read.assert_called_once_with(root / "summaries" / "master_results_rows.csv")
            mkdir.assert_called_once_with(parents=True, exist_ok=True)
            save.assert_called_once_with(fig, root / "figures" / "report" / "fig_fixed_scaling_auc_balance.png",
                                         dpi=300, bbox_inches="tight")
            show.assert_called_once_with()
        finally:
            plt.close(namespace.get("_attendance_fig"))


if __name__ == "__main__":
    test_attendance_figure()
    print("PASS: aligned AUC/balance panels preserve seed SD, Vanilla references and separate unimodal endpoints.")
