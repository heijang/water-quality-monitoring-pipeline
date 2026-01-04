import logging
import os
import sys
import pandas as pd
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
report_logger = logging.getLogger(f"{__name__}.report")

if not report_logger.handlers:
    report_handler = logging.StreamHandler(sys.stdout)
    report_handler.setFormatter(logging.Formatter("%(message)s"))
    report_logger.addHandler(report_handler)
    report_logger.setLevel(logging.INFO)
    report_logger.propagate = False


def load_raw_data_from_csv(data_path=None):
    if data_path is None:
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / 'data' / 'sample' / 'water_potability.csv'

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Empty dataframe")

    logger.info(f"Loaded {df.shape[0]} rows from {data_path}")
    return df


def load_raw_data_from_db(table_name: str = "raw_water_quality"):
    # Add project root to path for imports
    project_root = Path(__file__).parent.parent.parent
    if str(project_root / 'src') not in sys.path:
        sys.path.insert(0, str(project_root / 'src'))

    from utils.db_utils import read_table

    df = read_table(table_name)
    if df.empty:
        raise ValueError(f"Table {table_name} is empty")

    df = df.drop(columns=['id', 'loaded_at'], errors='ignore')
    logger.info(f"Loaded {df.shape[0]} rows from DB table {table_name}")
    return df


def load_raw_data(data_path: Optional[str] = None, use_db: bool = False):
    return load_raw_data_from_db() if use_db else load_raw_data_from_csv(data_path)


def save_csv_to_db(csv_path: Optional[str] = None,
                   table_name: str = "raw_water_quality",
                   if_exists: str = "replace"):
    # Add project root to path for imports
    project_root = Path(__file__).parent.parent.parent
    if str(project_root / 'src') not in sys.path:
        sys.path.insert(0, str(project_root / 'src'))

    from utils.db_utils import load_csv_to_table

    if csv_path is None:
        csv_path = project_root / 'data' / 'sample' / 'water_potability.csv'

    rows = load_csv_to_table(str(csv_path), table_name, if_exists=if_exists)
    logger.info(f"Loaded {rows} rows to DB: {table_name}")
    return rows


def get_data_info(df):
    return {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'target_distribution': df['Potability'].value_counts().to_dict() if 'Potability' in df.columns else None
    }


def print_data_summary(df):
    info = get_data_info(df)
    report_logger.info(f"\nDataset: {info['n_rows']} rows x {info['n_cols']} columns")

    missing = {k: v for k, v in info['missing_values'].items() if v > 0}
    if missing:
        report_logger.info("Missing values:")
        for col, cnt in missing.items():
            pct = (cnt / info['n_rows']) * 100
            report_logger.info(f"  {col}: {cnt} ({pct:.1f}%)")

    if info['target_distribution']:
        report_logger.info("Target distribution:")
        for label, count in info['target_distribution'].items():
            name = "Potable" if label == 1 else "Not Potable"
            report_logger.info(f"  {name}: {count}")

    report_logger.info(f"\n{df.describe()}\n")


def save_data(df, output_path, table_name: str = "raw_water_quality", storage_mode: Optional[str] = None):
    """Save data based on STORAGE_MODE"""
    import sys
    from pathlib import Path

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent.parent
    if str(project_root / 'src') not in sys.path:
        sys.path.insert(0, str(project_root / 'src'))

    from utils import STORAGE_MODE as DEFAULT_MODE, save_data_with_mode

    mode = storage_mode or DEFAULT_MODE
    save_data_with_mode(df, output_path, table_name, mode)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Add project root to path for imports
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root / 'src') not in sys.path:
        sys.path.insert(0, str(project_root / 'src'))

    input_path = sys.argv[1] if len(sys.argv) > 1 else None

    df = load_raw_data(data_path=input_path)
    print_data_summary(df)

    output_path = project_root / 'data' / 'result' / 'raw_data.csv'
    save_data(df, str(output_path))