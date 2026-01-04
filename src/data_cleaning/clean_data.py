import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame, strategy: str = 'median', n_neighbors: int = 5) -> pd.DataFrame:
    df_clean = df.copy()
    feature_cols = [col for col in df.columns if col != 'Potability']
    missing_cols = [col for col in feature_cols if df[col].isnull().any()]

    if not missing_cols:
        return df_clean

    logger.info(f"Handling missing values in {len(missing_cols)} columns using {strategy}")

    if strategy == 'median':
        for col in missing_cols:
            #df_clean[col].fillna(df_clean[col].median(), inplace=True)
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    elif strategy == 'mean':
        for col in missing_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    elif strategy == 'knn':
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df_clean[feature_cols] = imputer.fit_transform(df_clean[feature_cols])
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
        logger.info(f"Dropped {len(df) - len(df_clean)} rows")

    return df_clean


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> dict:
    outliers = {}
    feature_cols = [col for col in df.columns if col != 'Potability']

    for col in feature_cols:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask = (df[col] < lower) | (df[col] > upper)
        elif method == 'zscore':
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            mask = z > threshold

        outliers[col] = df[mask].index.tolist()

    total = sum(len(v) for v in outliers.values())
    logger.info(f"Detected {total} outlier observations using {method}")
    return outliers


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df_clean = df.drop_duplicates()
    n_dupes = n_before - len(df_clean)

    if n_dupes > 0:
        logger.info(f"Removed {n_dupes} duplicate rows")

    return df_clean


def clean_dataset(df: pd.DataFrame,
                  missing_strategy: str = 'median',
                  n_neighbors: int = 5,
                  remove_dup: bool = True,
                  detect_outlier: bool = True) -> Tuple[pd.DataFrame, dict]:

    logger.info(f"Cleaning data: shape {df.shape}")

    metadata = {'initial_shape': df.shape, 'missing_strategy': missing_strategy}

    if remove_dup:
        df = remove_duplicates(df)

    df = handle_missing_values(df, strategy=missing_strategy, n_neighbors=n_neighbors)

    if detect_outlier:
        outliers = detect_outliers(df, method='iqr', threshold=1.5)
        metadata['outliers'] = outliers

    metadata['final_shape'] = df.shape
    logger.info(f"Cleaning complete: {df.shape}")

    return df, metadata


def save_cleaned_data(df: pd.DataFrame, output_path: str, table_name: str = "cleaned_water_quality", storage_mode: str = None):
    """Save cleaned data based on STORAGE_MODE"""
    import sys

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent.parent
    if str(project_root / 'src') not in sys.path:
        sys.path.insert(0, str(project_root / 'src'))

    from utils import STORAGE_MODE as DEFAULT_MODE, save_data_with_mode

    mode = storage_mode or DEFAULT_MODE
    save_data_with_mode(df, output_path, table_name, mode)


if __name__ == "__main__":
    import sys

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent.parent
    if str(project_root / 'src') not in sys.path:
        sys.path.insert(0, str(project_root / 'src'))

    from utils import RESULTS_DIR, STORAGE_MODE, load_data_with_mode

    raw_data_path = RESULTS_DIR / 'raw_data.csv'

    # Load raw data based on STORAGE_MODE
    if STORAGE_MODE == "LOCAL" and not raw_data_path.exists():
        from data_ingestion.load_data import load_raw_data, save_data
        df = load_raw_data()
        save_data(df, str(raw_data_path))
    else:
        df = load_data_with_mode(str(raw_data_path), "raw_water_quality", STORAGE_MODE)

    df_clean, metadata = clean_dataset(df, missing_strategy='median')

    output_path = RESULTS_DIR / 'cleaned_data.csv'
    save_cleaned_data(df_clean, str(output_path))
