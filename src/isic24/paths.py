"""Where data and artifacts live, for ANY dataset of the pipeline.

A dataset root has two segments — ``<dataset>/<variant>`` — and the shape is the same for
data and for artifacts:

    data/isic24/enhanced/...        artifacts/isic24/enhanced/...
    data/brset/<variant>/...        artifacts/brset/<variant>/...

No module composes these paths by hand: whoever needs a root asks for it here. This is the
only place a dataset name appears, and it appears as a DEFAULT, not as a hard-wired
constant.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# The active dataset. Changed from the environment or by passing `dataset=`/`variant=` to the
# functions: downstream code must never know it.
DATASET = os.environ.get("DUM_DATASET", "isic24")
VARIANT = os.environ.get("DUM_VARIANT", "enhanced")

# The segment the run tree starts at inside a dataset root. It is OUR structure, the same on
# every dataset, and it is what makes a `run_dir` re-rootable without knowing what the dataset
# folder is called (cf. `run_dir_abs`).
RUNS_DIRNAME = "runs"


def _env_root(*names):
    """The first non-empty path among these environment variables, or ``None``."""
    for n in names:
        p = os.environ.get(n)
        if p:
            return Path(p).expanduser().resolve()
    return None


def get_base() -> Path:
    """Repository root, which data and artifacts derive from unless moved elsewhere."""
    return _env_root("DUM_BASE", "ISIC24_BASE") or REPO_ROOT


def get_data_root() -> Path:
    """Root of ALL data (the datasets live inside it)."""
    return _env_root("DUM_DATA_DIR", "ISIC24_DATA_DIR") or (get_base() / "data").resolve()


def get_artifacts_root() -> Path:
    """Root of ALL artifacts (the datasets live inside it)."""
    return (_env_root("DUM_ARTIFACT_DIR", "ISIC24_ARTIFACT_DIR")
            or (get_base() / "artifacts").resolve())


def dataset_root(root: Path, dataset=None, variant=None) -> Path:
    """``<root>/<dataset>/<variant>``."""
    dataset = DATASET if dataset is None else dataset
    variant = VARIANT if variant is None else variant
    return (Path(root) / dataset / variant).resolve()


def data_dir(dataset=None) -> Path:
    """DATASET level of the data: the images (hdf5) live here, not in the variant."""
    return (get_data_root() / (DATASET if dataset is None else dataset)).resolve()


def data_path(dataset=None, variant=None) -> Path:
    """VARIANT level of the data: the metadata of the active variant."""
    return dataset_root(get_data_root(), dataset, variant)


def art_path(dataset=None, variant=None) -> Path:
    return dataset_root(get_artifacts_root(), dataset, variant)


def run_dir_abs(run_dir, art_root) -> str:
    """Re-root a `run_dir` read from a CSV onto the LOCAL artifact tree.

    The summary CSVs carry `run_dir` as an ABSOLUTE path of the machine that did the training
    — `/home/claudio/isic24-dum/artifacts/enhanced/runs/...` — which exists nowhere else. Only
    the part from `runs/` onwards matters: it is the one piece that belongs to the pipeline
    rather than to the machine or to the dataset name.

    A path without `runs/` RAISES. It used to return the originating machine's whole path,
    that is a `File not found` on a path that looks right.
    """
    if str(run_dir).startswith(str(art_root)):   # already local: nothing to re-root
        return str(run_dir)
    parts = Path(str(run_dir)).parts
    if RUNS_DIRNAME not in parts:
        raise ValueError(
            f"run_dir carries no {RUNS_DIRNAME!r} segment, not re-rootable: {run_dir!r}")
    tail = parts[parts.index(RUNS_DIRNAME):]
    return os.path.join(str(art_root), *tail)
