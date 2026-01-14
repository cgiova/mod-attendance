from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[2]

def get_data_root() -> Path:
    """
    Dataset root. If ISIC24_DATA_DIR is set, use it.
    Otherwise fallback to <repo>/data.
    """
    p = os.environ.get("ISIC24_DATA_DIR")
    if p:
        return Path(p).expanduser().resolve()
    return (REPO_ROOT / "data").resolve()