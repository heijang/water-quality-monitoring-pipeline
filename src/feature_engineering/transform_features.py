import logging
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import pickle

logger = logging.getLogger(__name__)


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()

    df_new['Hardness_to_pH'] = df_new['Hardness'] / (df_new['ph'] + 1e-5)
    df_new['Solids_to_Conductivity'] = df_new['Solids'] / (df_new['Conductivity'] + 1e-5)
    df_new['Chloramines_to_pH'] = df_new['Chloramines'] / (df_new['ph'] + 1e-5)
    df_new['Organic_to_Trihalomethanes'] = df_new['Organic_carbon'] / (df_new['Trihalomethanes'] + 1e-5)
    df_new['Sulfate_to_Conductivity'] = df_new['Sulfate'] / (df_new['Conductivity'] + 1e-5)

    df_new['Chemical_Index'] = (df_new['Chloramines'] + df_new['Sulfate']) / 2
    df_new['Physical_Index'] = (df_new['Hardness'] + df_new['Turbidity']) / 2
    df_new['Organic_Index'] = (df_new['Organic_carbon'] + df_new['Trihalomethanes']) / 2

    df_new['ph_squared'] = df_new['ph'] ** 2
    df_new['Turbidity_squared'] = df_new['Turbidity'] ** 2
    df_new['Total_Dissolved'] = df_new['Solids'] + df_new['Sulfate']

    n_new = len(df_new.columns) - len(df.columns)
    logger.info(f"Created {n_new} derived features")

    return df_new


def create_interaction_features(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    df_new = df.copy()
    key_features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Turbidity']

    n_inter = 0
    for i, feat1 in enumerate(key_features):
        for feat2 in key_features[i+1:]:
            df_new[f'{feat1}_x_{feat2}'] = df_new[feat1] * df_new[feat2]
            n_inter += 1

    logger.info(f"Created {n_inter} interaction features")
    return df_new


def scale_features(df: pd.DataFrame,
                   scaler_type: str = 'standard',
                   fit_scaler: bool = True,
                   scaler_path: str = None) -> Tuple[pd.DataFrame, object]:

    df_scaled = df.copy()
    feature_cols = [col for col in df.columns if col != 'Potability']

    if fit_scaler:
        scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
        logger.info(f"Scaled {len(feature_cols)} features using {scaler_type}")
    else:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])

    return df_scaled, scaler


def split_data(df: pd.DataFrame,
               test_size: float = 0.2,
               random_state: int = 42,
               stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    feature_cols = [col for col in df.columns if col != 'Potability']
    X = df[feature_cols]
    y = df['Potability']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )

    logger.info(f"Split data: {len(X_train)} train, {len(X_test)} test")
    return X_train, X_test, y_train, y_test


def split_data_with_validation(df: pd.DataFrame,
                                test_size: float = 0.2,
                                val_size: float = 0.2,
                                random_state: int = 42,
                                stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                                pd.Series, pd.Series, pd.Series]:

    feature_cols = [col for col in df.columns if col != 'Potability']
    X = df[feature_cols]
    y = df['Potability']

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if stratify else None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state,
        stratify=y_temp if stratify else None
    )

    logger.info(f"Split: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    return X_train, X_val, X_test, y_train, y_val, y_test


def feature_engineering_pipeline(df: pd.DataFrame,
                                  create_derived: bool = True,
                                  create_interactions: bool = False,
                                  scale: bool = True,
                                  scaler_type: str = 'standard') -> Tuple[pd.DataFrame, dict]:

    n_init = len(df.columns) - 1
    metadata = {'initial_features': df.columns.tolist(), 'steps': []}

    df_transformed = df.copy()

    if create_derived:
        df_transformed = create_derived_features(df_transformed)
        metadata['steps'].append('derived')

    if create_interactions:
        df_transformed = create_interaction_features(df_transformed)
        metadata['steps'].append('interactions')

    scaler = None
    if scale:
        df_transformed, scaler = scale_features(df_transformed, scaler_type=scaler_type)
        metadata['steps'].append('scaling')
        metadata['scaler'] = scaler

    metadata['final_features'] = df_transformed.columns.tolist()
    n_final = len(df_transformed.columns) - 1
    logger.info(f"Feature engineering: {n_init} -> {n_final} features")

    return df_transformed, metadata


def save_engineered_data(X_train: pd.DataFrame, X_test: pd.DataFrame,
                         y_train: pd.Series, y_test: pd.Series,
                         output_dir: str):
    """Save engineered data (LOCAL only - intermediate data)"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Training/test splits are intermediate data - save locally only
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False, header=True)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False, header=True)

    logger.info(f"Saved engineered data to {output_dir} (LOCAL only)")


def save_scaler(scaler: object, output_path: str):
    """Save scaler to pickle file (always local)"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved: {output_path}")


if __name__ == "__main__":
    import sys

    # Add project root to path for imports
    project_root = Path(__file__).parent.parent.parent
    if str(project_root / 'src') not in sys.path:
        sys.path.insert(0, str(project_root / 'src'))

    from utils import RESULTS_DIR, STORAGE_MODE, load_data_with_mode

    cleaned_data_path = RESULTS_DIR / 'cleaned_data.csv'

    # Load cleaned data based on STORAGE_MODE
    if STORAGE_MODE == "LOCAL" and not cleaned_data_path.exists():
        from data_cleaning.clean_data import clean_dataset, save_cleaned_data
        from data_ingestion.load_data import load_raw_data
        df = load_raw_data()
        df_clean, _ = clean_dataset(df)
        save_cleaned_data(df_clean, str(cleaned_data_path))
    else:
        df = load_data_with_mode(str(cleaned_data_path), "cleaned_water_quality", STORAGE_MODE)

    df_transformed, metadata = feature_engineering_pipeline(
        df, create_derived=True, create_interactions=False,
        scale=True, scaler_type='standard'
    )

    X_train, X_test, y_train, y_test = split_data(df_transformed, test_size=0.2, random_state=42)
    save_engineered_data(X_train, X_test, y_train, y_test, str(RESULTS_DIR))

    if 'scaler' in metadata:
        save_scaler(metadata['scaler'], str(RESULTS_DIR / 'scaler.pkl'))
