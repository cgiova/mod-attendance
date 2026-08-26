"""The one report figure a table cannot carry: sec. 3.6, the arms of the image metrics.

Six budget points x two arms x two directions is a curve, not a table, and the whole
argument of that section is the *shape* of the random arm. Everything else in the report
stays a table.

Pure arithmetic over the saved ``faithfulness_curves.npz``: no model, no GPU. Unimodal
endpoints are dropped (`xs.drop_ablations`) and the unit is the run.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from . import paths as _paths
from . import xai_faithfulness as xf
from . import xai_subset as xs
from .xai_engines import _finalize_plot

FIG_DIRNAME = os.path.join("figures", "report")

C_EXP, C_RAND, C_INS = "#2b7bba", "#9a9a9a", "#d95f02"


def fig_dir(art_path=None):
    d = os.path.join(str(art_path or _paths.art_path()), FIG_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d


def read_arms(art_path=None, method="image", files=None, q_max=xs.Q_CLEAN, central=None,
              source="canonical"):
    """Mean curve and arm areas of every canonical run, endpoints excluded.

    The area of a mean curve equals the mean of the per-sample areas (the integral is
    linear), so ``exp - rand`` here is the same quantity `persample_gains` returns --
    `verify_reintegration` is what pins that identity down. Areas are divided by the
    budget window, the `gain_norm` scale the report quotes.

    ``central`` restricts to one setting per family, the population sec. 3.6 reports.
    ``source`` picks the ranking: ``canonical`` is the logit-aligned Permutation,
    ``gradient`` the SHAP GradientExplainer. One figure each; they are never overlaid.

    Returns ``(q_grid, curves, per_run)``: `curves` maps ``"<direction>_<arm>"`` to a
    ``(n_runs, n_q)`` array, `per_run` carries the arm rates over ``q <= q_max``.
    """
    art_path = str(art_path or _paths.art_path())
    if files is None:
        files = xf.discover_curve_files(art_path)
    files = xs.drop_ablations(files[files.source == source], "report figures")
    if central:
        files = files[[central.get(i) == t
                       for i, t in zip(files.impl_name, files.setting_tag)]]

    q = None
    curves = {f"{d}_{a}": [] for d in ("del", "ins") for a in ("exp", "rand")}
    rows = []
    for f in files.itertuples():
        with np.load(f.npz_path) as z:
            q = np.asarray(z["q_grid"], dtype=float)
            keep = q <= q_max + 1e-12
            width = q[keep][-1] - q[keep][0]
            rec = {"impl_name": f.impl_name, "run_id": f.run_id, "seed": f.seed,
                   "setting_tag": f.setting_tag}
            for direction in ("del", "ins"):
                for arm in ("exp", "rand"):
                    c = np.asarray(z[f"{direction}_{arm}_{method}"], dtype=float).mean(0)
                    curves[f"{direction}_{arm}"].append(c)
                    rec[f"{direction}_{arm}"] = xf._trapz(c[keep], q[keep]) / width
                rec[f"{direction}_gain"] = rec[f"{direction}_exp"] - rec[f"{direction}_rand"]
            rows.append(rec)
    return q, {k: np.vstack(v) for k, v in curves.items()}, pd.DataFrame(rows)


def arms(art_path=None, method="image", out_dir=None, files=None, central=xs.CENTRAL_SETTINGS,
         fname=None, source="canonical"):
    """Sec. 3.6 — deletion is a faithfulness metric, insertion measures its own baseline.

    Left and centre: the two arms, mean over runs with a +-1 sd band, the shaded area
    between them being the gain. Right: per-run gain against the random arm, where
    insertion's dependence on the arm it should be controlling for is visible.
    """
    from scipy.stats import spearmanr

    q, curves, per_run = read_arms(art_path, method=method, files=files, central=central,
                                   source=source)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))

    clean = q <= xs.Q_CLEAN + 1e-12
    for ax, direction, title in ((axes[0], "del", "deletion"), (axes[1], "ins", "insertion")):
        exp_, rand_ = (curves[f"{direction}_exp"].mean(0), curves[f"{direction}_rand"].mean(0))
        for m, arm, color, label in ((exp_, "exp", C_EXP, "explanation"),
                                     (rand_, "rand", C_RAND, "random")):
            sd = curves[f"{direction}_{arm}"].std(0)
            ax.plot(q, m, marker="o", ms=4, color=color, label=label, zorder=3)
            ax.fill_between(q, m - sd, m + sd, color=color, alpha=.12, lw=0)
        # Only the clean window is shaded: the gain the report quotes is the area there.
        ax.fill_between(q[clean], exp_[clean], rand_[clean], color=C_EXP, alpha=.35, lw=0,
                        label=f"gain ($q\\leq${xs.Q_CLEAN})")
        ax.axvline(xs.Q_CLEAN, color="k", ls=":", lw=1)
        ax.axhline(0, color="k", lw=.6)
        ax.set_xlabel("budget $q$"); ax.set_title(title); ax.grid(alpha=.25)
        ax.legend(fontsize=8, loc="lower left")
    axes[0].set_ylabel("mean logit drop $C(q)$")

    ax = axes[2]
    for direction, color, name in (("del", C_EXP, "deletion"), ("ins", C_INS, "insertion")):
        x, y = per_run[f"{direction}_rand"], per_run[f"{direction}_gain"]
        rho_r = spearmanr(x, y).statistic
        rho_e = spearmanr(per_run[f"{direction}_exp"], y).statistic
        ax.scatter(x, y, s=18, color=color, alpha=.7,
                   label=f"{name}: $\\rho$ random {rho_r:+.2f}, explanation {rho_e:+.2f}")
    ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel(f"random arm ($q\\leq${xs.Q_CLEAN}, per unit of budget)")
    ax.set_ylabel("gain")
    ax.set_title("gain vs the arm it controls for")
    ax.grid(alpha=.25); ax.legend(fontsize=7.5, loc="upper right", framealpha=.9)

    pop = "central settings" if central else "non-endpoint runs"
    fig.suptitle(f"Faithfulness by arm — {method}, {source} ranking, "
                 f"{len(per_run)} runs ({pop})", y=1.0)
    # The name carries ranking AND branch: five figures come out of this call, and none of
    # them may land on the name another one holds.
    _finalize_plot(out_dir or fig_dir(art_path),
                   fname or f"fig_faithfulness_arms_{source}_{method}.png")
    return per_run
