import os
from pathlib import Path
from typing import Tuple, Literal


def get_project_root() -> Path:
    marker_files = ('requirements.txt', 'README.md', '.git')
    current = Path(__file__).resolve().parent

    for parent in [current] + list(current.parents):
        if any((parent / marker).exists() for marker in marker_files):
            return parent

    return current.parent.parent


def get_data_paths() -> Tuple[Path, Path, Path]:
    root = get_project_root()
    return root / 'data', root / 'figures', root / 'tables'


PROJECT_ROOT = get_project_root()
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = DATA_DIR / 'result'
FIGURES_DIR = PROJECT_ROOT / 'figures'
TABLES_DIR = PROJECT_ROOT / 'tables'

# Storage mode configuration
# - LOCAL: Read/write CSV files only
# - DB: Read/write database only
# - BOTH: Write to both DB and LOCAL, but read from LOCAL
StorageMode = Literal["LOCAL", "DB", "BOTH"]
_storage_mode_env = os.getenv("STORAGE_MODE", "LOCAL").upper()
STORAGE_MODE: StorageMode = _storage_mode_env if _storage_mode_env in ("LOCAL", "DB", "BOTH") else "LOCAL"
