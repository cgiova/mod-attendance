"""Dove stanno dati e artefatti, per QUALUNQUE dataset della pipeline.

La radice di un dataset ha due segmenti — ``<dataset>/<variante>`` — ed e' la stessa forma
per dati e artefatti:

    data/isic24/enhanced/...        artifacts/isic24/enhanced/...
    data/brset/<variante>/...       artifacts/brset/<variante>/...

Nessun modulo compone questi percorsi a mano: chi ha bisogno di una radice la chiede qui.
E' l'unico posto in cui compare il nome di un dataset, e ci compare come DEFAULT, non come
costante cablata.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Il dataset attivo. Si cambia da ambiente o passando `dataset=`/`variant=` alle funzioni:
# il codice a valle non deve mai saperlo.
DATASET = os.environ.get("DUM_DATASET", "isic24")
VARIANT = os.environ.get("DUM_VARIANT", "enhanced")

# Il segmento da cui comincia l'albero delle run dentro una radice di dataset. E' struttura
# NOSTRA, uguale su ogni dataset, ed e' cio' che permette di ri-radicare un `run_dir` senza
# sapere come si chiama la cartella del dataset (cfr. `run_dir_abs`).
RUNS_DIRNAME = "runs"


def _env_root(*names):
    """Il primo path non vuoto fra queste variabili d'ambiente, o ``None``."""
    for n in names:
        p = os.environ.get(n)
        if p:
            return Path(p).expanduser().resolve()
    return None


def get_base() -> Path:
    """Radice del repo, da cui derivano dati e artefatti se non sono spostati altrove."""
    return _env_root("DUM_BASE", "ISIC24_BASE") or REPO_ROOT


def get_data_root() -> Path:
    """Radice di TUTTI i dati (dentro ci sono i dataset)."""
    return _env_root("DUM_DATA_DIR", "ISIC24_DATA_DIR") or (get_base() / "data").resolve()


def get_artifacts_root() -> Path:
    """Radice di TUTTI gli artefatti (dentro ci sono i dataset)."""
    return (_env_root("DUM_ARTIFACT_DIR", "ISIC24_ARTIFACT_DIR")
            or (get_base() / "artifacts").resolve())


def dataset_root(root: Path, dataset=None, variant=None) -> Path:
    """``<root>/<dataset>/<variante>``."""
    dataset = DATASET if dataset is None else dataset
    variant = VARIANT if variant is None else variant
    return (Path(root) / dataset / variant).resolve()


def data_dir(dataset=None) -> Path:
    """Livello DATASET dei dati: le immagini (hdf5) stanno qui, non nella variante."""
    return (get_data_root() / (DATASET if dataset is None else dataset)).resolve()


def data_path(dataset=None, variant=None) -> Path:
    """Livello VARIANTE dei dati: i metadati della variante attiva."""
    return dataset_root(get_data_root(), dataset, variant)


def art_path(dataset=None, variant=None) -> Path:
    return dataset_root(get_artifacts_root(), dataset, variant)


def run_dir_abs(run_dir, art_root) -> str:
    """Ri-radica un `run_dir` letto da un CSV sull'albero degli artefatti LOCALE.

    I CSV di riepilogo portano `run_dir` come percorso ASSOLUTO della macchina che ha fatto
    il training — `/home/claudio/isic24-dum/artifacts/enhanced/runs/...` — che altrove non
    esiste. Del percorso conta solo la parte da `runs/` in poi: e' l'unico pezzo che
    appartiene alla pipeline invece che alla macchina o al nome del dataset.

    Un percorso senza `runs/` SOLLEVA. Prima si tornava il path intero della macchina di
    origine, cioe' un `File not found` su un percorso che sembra giusto.
    """
    if str(run_dir).startswith(str(art_root)):   # gia' locale: niente da ri-radicare
        return str(run_dir)
    parts = Path(str(run_dir)).parts
    if RUNS_DIRNAME not in parts:
        raise ValueError(
            f"run_dir senza segmento {RUNS_DIRNAME!r}, non ri-radicabile: {run_dir!r}")
    tail = parts[parts.index(RUNS_DIRNAME):]
    return os.path.join(str(art_root), *tail)
