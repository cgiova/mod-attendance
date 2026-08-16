"""The identity of a run: one place, both directions.

The framework writes the same run in TWO encodings:

    run_dir = <runs>/<impl_name>/<exp_tag>/<setting_folder>
    run_id  = <exp_tag>__<setting_folder>

where::

    exp_tag        = "<study_tag>"  or  "<study_tag>/seed_<seed>"
    setting_folder = "<setting_tag>__e<epochs>"   (degenerate forms below)

``run_dir`` separates with ``/`` and carries ``impl_name``; ``run_id`` fuses the last two
levels with ``__`` and has no ``impl_name``.  When ``exp_tag`` contains the seed, ``run_id``
ends up with **one** ``/`` and **two** ``__``, the second ``__`` belonging to the setting
name: that is why it cannot be re-parsed by eye, and it is the bug that on 2026-07-30 sent
every run into the ``"?"`` group, collapsing the six families into one row of means.

**This module is the definition**, and it mirrors `run_experiment` of the training notebook
(``_make_exp_tag`` / ``_make_setting_folder``), degenerate forms included:

===============  ==========  ==========================
``setting_tag``  ``epochs``  ``setting_folder``
===============  ==========  ==========================
``"a0p000"``     ``20``      ``"a0p000__e20"``
``None``         ``20``      ``"run__e20"``
``"a0p000"``     ``None``    ``"a0p000"``
``None``         ``None``    ``"run"``
===============  ==========  ==========================

and the fact that an auto-generated ``exp_tag`` is the timestamp alone, **without** a
``seed_`` level: a run launched outside a multi-seed loop lives three levels under ``runs/``,
not four.  `parse_run_dir` recognises both layouts.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime

SEED_DIR_RE = re.compile(r"^seed_(?P<seed>.+)$")
EPOCHS_SUFFIX_RE = re.compile(r"^(?P<setting_tag>.*)__e(?P<epochs>\d+)$")
NO_SETTING = "run"          # what `_make_setting_folder` writes when setting_tag is None


# --------------------------------------------------------------------------------------
# Construction -- the formulas, in one place
# --------------------------------------------------------------------------------------

def make_exp_tag(study_tag=None, seed=None) -> str:
    """``study_tag`` (default: the timestamp now), plus ``/seed_<seed>`` when there is a seed."""
    tag = datetime.now().strftime("%Y%m%d-%H%M%S") if study_tag is None else str(study_tag)
    return f"{tag}/seed_{seed}" if seed is not None else tag


def make_setting_folder(setting_tag, epochs) -> str:
    """The setting folder name. Degenerate forms included (see the table at the top)."""
    parts = [str(setting_tag)] if setting_tag else [NO_SETTING]
    if epochs is not None:
        parts.append(f"e{int(epochs)}")
    return "__".join(parts)


def make_run_id(exp_tag, setting_folder) -> str:
    """The canonical ``run_id`` from the two pieces training already holds."""
    return f"{exp_tag}__{setting_folder}"


def make_run_dir(runs_path, impl_name, exp_tag, setting_folder) -> str:
    """The canonical ``run_dir``. ``exp_tag`` may contain ``/``: os.path.join handles it."""
    return os.path.join(str(runs_path), str(impl_name), str(exp_tag), str(setting_folder))


def run_id_from_parts(study_tag, seed, setting_tag, epochs) -> str:
    """Shortcut: from the five fields to the ``run_id``, without going through the two pieces."""
    return make_run_id(make_exp_tag(study_tag, seed), make_setting_folder(setting_tag, epochs))


# --------------------------------------------------------------------------------------
# Reading -- the inverse that was missing
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RunKey:
    """The fields of a run. ``seed`` and ``epochs`` are ``None`` where the layout lacks them."""
    impl_name: str
    study_tag: str
    seed: str | None
    setting_tag: str | None
    epochs: int | None

    @property
    def exp_tag(self) -> str:
        return make_exp_tag(self.study_tag, self.seed)

    @property
    def setting_folder(self) -> str:
        return make_setting_folder(self.setting_tag, self.epochs)

    @property
    def run_id(self) -> str:
        return make_run_id(self.exp_tag, self.setting_folder)


def parse_run_dir(run_dir, runs_path=None) -> RunKey:
    """From the path to the fields, recognising both the 4-level and the 3-level layout.

    With ``runs_path`` the cut is exact.  Without it, the tail is the anchor and the
    second-to-last level decides: matching ``seed_<...>`` means the seed level is there.
    The heuristic is needed because the registry CSVs carry the absolute paths of the machine
    that trained, which do not exist here.
    """
    parts = [p for p in os.path.normpath(str(run_dir)).split(os.sep) if p]

    if runs_path is not None:
        root = [p for p in os.path.normpath(str(runs_path)).split(os.sep) if p]
        if parts[:len(root)] != root:
            raise ValueError(f"{run_dir!r} does not sit under {runs_path!r}")
        rest = parts[len(root):]
        if len(rest) < 3:
            raise ValueError(f"expected at least impl/exp_tag/setting under {runs_path!r}: {rest}")
        impl_name, setting_folder = rest[0], rest[-1]
        exp_parts = rest[1:-1]
    else:
        if len(parts) < 3:
            raise ValueError(f"run_dir too short: {run_dir!r}")
        has_seed = len(parts) >= 4 and SEED_DIR_RE.match(parts[-2]) is not None
        n = 4 if has_seed else 3
        impl_name, setting_folder = parts[-n], parts[-1]
        exp_parts = parts[-n + 1:-1]

    seed = None
    if exp_parts and (m := SEED_DIR_RE.match(exp_parts[-1])):
        seed, exp_parts = m.group("seed"), exp_parts[:-1]
    if not exp_parts:
        raise ValueError(f"no study_tag in {run_dir!r} (exp_tag reduced to the seed alone)")
    study_tag = "/".join(exp_parts)

    epochs = None
    setting_tag = setting_folder
    if m := EPOCHS_SUFFIX_RE.match(setting_folder):
        setting_tag, epochs = m.group("setting_tag"), int(m.group("epochs"))
    if setting_tag == NO_SETTING:
        setting_tag = None

    return RunKey(impl_name=impl_name, study_tag=study_tag, seed=seed,
                  setting_tag=setting_tag, epochs=epochs)


def run_id_from_run_dir(run_dir, runs_path=None) -> str:
    """The ``run_id`` of a run, deduced from its path."""
    return parse_run_dir(run_dir, runs_path).run_id


def run_id_from_artifact(path, marker="xai", runs_path=None) -> str:
    """``run_id`` from a file INSIDE the run (e.g. ``<run_dir>/xai/lore/.../x.csv``)."""
    parts = os.path.normpath(str(path)).split(os.sep)
    if marker not in parts:
        raise ValueError(f"{marker!r} does not appear in {path!r}: cannot isolate the run dir")
    cut = len(parts) - 1 - parts[::-1].index(marker)
    return run_id_from_run_dir(os.sep.join(parts[:cut]), runs_path)


def round_trip_check(master_rows_csv):
    """Check that `run_id_from_run_dir` reproduces the canonical registry row by row.

    The list of disagreements, empty when everything matches. It is the test the hand-written
    reconstruction of the post-hoc notebook would have failed on all 110 rows.
    """
    import pandas as pd
    bad = []
    for _, r in pd.read_csv(master_rows_csv).iterrows():
        try:
            got = run_id_from_run_dir(r.run_dir)
        except ValueError as e:
            bad.append((str(r.run_dir), str(r.run_id), f"ERROR: {e}"))
            continue
        if got != str(r.run_id):
            bad.append((str(r.run_dir), str(r.run_id), got))
    return bad
