
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
import pickle
import json
from typing import Tuple, Optional

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from utils import RESULTS_DIR

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Database support (optional)
USE_DB_TRACKING = True  # Set to False to disable DB tracking
try:
    from utils.db_utils import save_dataframe_to_table, get_engine, execute_query
    from sqlalchemy import text
except ImportError:
    USE_DB_TRACKING = False
    print("Warning: Database tracking disabled (db_utils not available)")

# ML imports
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Import project modules
from data_ingestion.load_data import load_raw_data
from data_cleaning.clean_data import clean_dataset, remove_duplicates
from feature_engineering.transform_features import (
    create_derived_features,
    scale_features,
    split_data
)


# ============================================================================
# Experiment Tracking Helper Functions
# ============================================================================

def create_experiment(experiment_name: str, research_question: str,
                     description: str, parameters: dict) -> Optional[int]:
    """
    Create experiment record in database

    Args:
        experiment_name: Name of the experiment
        research_question: Research question (e.g., 'RQ1', 'RQ2')
        description: Experiment description
        parameters: Experiment parameters as dict

    Returns:
        Experiment ID if DB tracking enabled, None otherwise
    """
    if not USE_DB_TRACKING:
        return None

    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO experiments (experiment_name, research_question, description, parameters, status)
                    VALUES (:name, :rq, :desc, :params, 'running')
                    RETURNING id
                """),
                {
                    "name": experiment_name,
                    "rq": research_question,
                    "desc": description,
                    "params": json.dumps(parameters)
                }
            )
            conn.commit()
            exp_id = result.fetchone()[0]
            print(f"Created experiment {exp_id}: {experiment_name}")
            return exp_id
    except Exception as e:
        print(f"Warning: Failed to create experiment record: {e}")
        return None


def complete_experiment(experiment_id: Optional[int]):
    
    if not USE_DB_TRACKING or experiment_id is None:
        return

    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(
                text("""
                    UPDATE experiments
                    SET status = 'completed', completed_at = CURRENT_TIMESTAMP
                    WHERE id = :exp_id
                """),
                {"exp_id": experiment_id}
            )
            conn.commit()
            print(f"Completed experiment {experiment_id}")
    except Exception as e:
        print(f"Warning: Failed to complete experiment: {e}")


def save_metrics_to_db(experiment_id: Optional[int], metrics_df: pd.DataFrame):
    """
    Save model metrics to database

    Args:
        experiment_id: Experiment ID
        metrics_df: DataFrame with metrics
    """
    if not USE_DB_TRACKING or experiment_id is None:
        return

    try:
        # Add experiment_id column
        metrics_df_copy = metrics_df.copy()
        metrics_df_copy['experiment_id'] = experiment_id

        # Rename columns to match database schema
        column_mapping = {
            'strategy': 'strategy',
            'accuracy': 'accuracy',
            'precision': 'precision_score',
            'recall': 'recall',
            'f1_score': 'f1_score',
            'roc_auc': 'roc_auc'
        }

        # Keep only columns that exist in both mapping and dataframe
        cols_to_keep = ['experiment_id'] + [col for col in column_mapping.keys() if col in metrics_df_copy.columns]
        metrics_df_copy = metrics_df_copy[cols_to_keep]

        # Rename columns
        for old, new in column_mapping.items():
            if old in metrics_df_copy.columns and old != new:
                metrics_df_copy = metrics_df_copy.rename(columns={old: new})

        save_dataframe_to_table(metrics_df_copy, 'model_metrics', if_exists='append')
        print(f"Saved {len(metrics_df_copy)} metric rows to database")

    except Exception as e:
        print(f"Warning: Failed to save metrics to database: {e}")


# ============================================================================
# Research Question Experiments
# ============================================================================

def run_rq1_imputation_comparison() -> pd.DataFrame:
    """
    RQ1: How do different imputation strategies affect XGBoost model performance?
    Tests median, mean, and KNN imputation on pH and Sulfate

    Returns:
        DataFrame with results for each strategy
    """
    print("RQ1: IMPUTATION STRATEGY COMPARISON")

    exp_id = create_experiment(
        experiment_name="Imputation Strategy Comparison",
        research_question="RQ1",
        description="Compare median, mean, and KNN imputation strategies for missing values",
        parameters={"strategies": ['median', 'mean', 'knn'], "model": "XGBoost"}
    )

    df_raw = load_raw_data()

    df_raw = remove_duplicates(df_raw)

    strategies = ['median', 'mean', 'knn']
    results = []

    for strategy in strategies:
        print(f"\n--- Testing {strategy.upper()} imputation ---")

        # Clean data with this strategy
        df_clean, _ = clean_dataset(
            df_raw.copy(),
            missing_strategy=strategy,
            n_neighbors=5,
            remove_dup=False,
            detect_outlier=False
        )

        # Scale features (no derived features for fair comparison)
        df_scaled, scaler = scale_features(df_clean, scaler_type='standard')

        # Split data
        X_train, X_test, y_train, y_test = split_data(
            df_scaled,
            test_size=0.2,
            random_state=42,
            stratify=True
        )

        model = XGBClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            eval_metric='logloss',
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        cv_scores = cross_validate(
            model, X_train, y_train,
            cv=5,
            scoring=['accuracy', 'recall'],
            return_train_score=True
        )

        stability_score = 1 - np.std(cv_scores['test_accuracy'])

        results.append({
            'strategy': strategy,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_recall_mean': np.mean(cv_scores['test_recall']),
            'cv_recall_std': np.std(cv_scores['test_recall']),
            'stability_score': stability_score
        })

        print(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, Stability: {stability_score:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / 'rq1_imputation_results.csv', index=False)
    print(f"\nRQ1 results saved to: {RESULTS_DIR / 'rq1_imputation_results.csv'}")

    save_metrics_to_db(exp_id, results_df)
    complete_experiment(exp_id)

    return results_df


def run_rq2_feature_engineering_comparison() -> pd.DataFrame:
    """
    RQ2: Does derived feature engineering improve unsafe water identification?
    Compares raw features vs derived features

    Returns:
        DataFrame with comparison results
    """
    print("RQ2: FEATURE ENGINEERING COMPARISON")

    df_raw = load_raw_data()
    df_clean, _ = clean_dataset(df_raw, missing_strategy='median')

    results = []

    print("\n--- Testing with RAW features ---")
    df_scaled_raw, _ = scale_features(df_clean.copy(), scaler_type='standard')
    X_train_raw, X_test_raw, y_train, y_test = split_data(
        df_scaled_raw, test_size=0.2, random_state=42, stratify=True
    )

    model_raw = XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss')
    model_raw.fit(X_train_raw, y_train)
    y_pred_raw = model_raw.predict(X_test_raw)

    # Focus on unsafe water (class 0) identification
    recall_unsafe_raw = recall_score(y_test, y_pred_raw, pos_label=0)
    precision_unsafe_raw = precision_score(y_test, y_pred_raw, pos_label=0, zero_division=0)
    f1_unsafe_raw = f1_score(y_test, y_pred_raw, pos_label=0)

    results.append({
        'approach': 'raw_features',
        'n_features': X_train_raw.shape[1],
        'accuracy': accuracy_score(y_test, y_pred_raw),
        'recall_unsafe': recall_unsafe_raw,
        'precision_unsafe': precision_unsafe_raw,
        'f1_unsafe': f1_unsafe_raw
    })

    print(f"Raw features - Unsafe water recall: {recall_unsafe_raw:.4f}")

    print("\n--- Testing with DERIVED features ---")
    df_derived = create_derived_features(df_clean.copy())
    df_scaled_derived, _ = scale_features(df_derived, scaler_type='standard')
    X_train_derived, X_test_derived, y_train, y_test = split_data(
        df_scaled_derived, test_size=0.2, random_state=42, stratify=True
    )

    model_derived = XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss')
    model_derived.fit(X_train_derived, y_train)
    y_pred_derived = model_derived.predict(X_test_derived)

    recall_unsafe_derived = recall_score(y_test, y_pred_derived, pos_label=0)
    precision_unsafe_derived = precision_score(y_test, y_pred_derived, pos_label=0, zero_division=0)
    f1_unsafe_derived = f1_score(y_test, y_pred_derived, pos_label=0)

    results.append({
        'approach': 'derived_features',
        'n_features': X_train_derived.shape[1],
        'accuracy': accuracy_score(y_test, y_pred_derived),
        'recall_unsafe': recall_unsafe_derived,
        'precision_unsafe': precision_unsafe_derived,
        'f1_unsafe': f1_unsafe_derived
    })

    print(f"Derived features - Unsafe water recall: {recall_unsafe_derived:.4f}")

    improvement = ((recall_unsafe_derived - recall_unsafe_raw) / recall_unsafe_raw) * 100
    print(f"\nImprovement in unsafe water recall: {improvement:.2f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / 'rq2_feature_engineering_results.csv', index=False)
    print(f"\nRQ2 results saved to: {RESULTS_DIR / 'rq2_feature_engineering_results.csv'}")

    return results_df


def run_rq3_dual_layer_safety() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    RQ3: Correlation between Isolation Forest anomaly scores and supervised model misclassifications

    Returns:
        Tuple of (correlation results, detailed anomaly analysis)
    """
    print("RQ3: DUAL-LAYER SAFETY ANALYSIS")

    df_raw = load_raw_data()
    df_clean, _ = clean_dataset(df_raw, missing_strategy='median')
    df_derived = create_derived_features(df_clean)
    df_scaled, _ = scale_features(df_derived, scaler_type='standard')

    X_train, X_test, y_train, y_test = split_data(
        df_scaled, test_size=0.2, random_state=42, stratify=True
    )

    print("\n--- Training supervised model ---")
    supervised_model = XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss')
    supervised_model.fit(X_train, y_train)
    y_pred = supervised_model.predict(X_test)

    # Identify misclassifications
    misclassified = (y_pred != y_test).astype(int)

    print("\n--- Training Isolation Forest ---")
    contamination = (y_train == 0).sum() / len(y_train)
    # Contamination must be between 0 and 0.5 for IsolationForest
    contamination = min(contamination, 0.5)
    iso_forest = IsolationForest(
        random_state=42,
        contamination=contamination,
        n_estimators=100
    )
    iso_forest.fit(X_train)

    anomaly_scores = -iso_forest.score_samples(X_test)  # Negate so higher = more anomalous

    from scipy.stats import pearsonr, spearmanr

    pearson_corr, pearson_p = pearsonr(anomaly_scores, misclassified)
    spearman_corr, spearman_p = spearmanr(anomaly_scores, misclassified)

    print(f"\nPearson correlation: {pearson_corr:.4f} (p={pearson_p:.4f})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")

    # Analyze misclassifications by anomaly score quartile
    anomaly_quartiles = pd.qcut(anomaly_scores, q=4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High'])

    quartile_analysis = []
    for q in ['Q1_Low', 'Q2', 'Q3', 'Q4_High']:
        mask = anomaly_quartiles == q
        misclass_rate = misclassified[mask].mean()
        quartile_analysis.append({
            'quartile': q,
            'misclassification_rate': misclass_rate,
            'n_samples': mask.sum()
        })

    detailed_results = pd.DataFrame({
        'anomaly_score': anomaly_scores,
        'misclassified': misclassified,
        'true_label': y_test.values,
        'predicted_label': y_pred,
        'quartile': anomaly_quartiles
    })
    detailed_results.to_csv(RESULTS_DIR / 'rq3_detailed_anomaly_analysis.csv', index=False)

    correlation_results = pd.DataFrame([{
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p
    }])
    correlation_results.to_csv(RESULTS_DIR / 'rq3_correlation_results.csv', index=False)

    quartile_df = pd.DataFrame(quartile_analysis)
    quartile_df.to_csv(RESULTS_DIR / 'rq3_quartile_analysis.csv', index=False)

    print(f"\nRQ3 results saved to data/ directory")

    return correlation_results, detailed_results


def run_rq4_pipeline_robustness() -> pd.DataFrame:
    """
    RQ4: Pipeline robustness under data noise and streaming simulation

    Returns:
        DataFrame with robustness metrics
    """
    print("RQ4: PIPELINE ROBUSTNESS TEST")

    df_raw = load_raw_data()
    df_clean, _ = clean_dataset(df_raw, missing_strategy='median')

    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    results = []

    for noise_level in noise_levels:
        print(f"\n--- Testing with {noise_level*100}% noise ---")

        # Add Gaussian noise to features
        df_noisy = df_clean.copy()
        feature_cols = [col for col in df_noisy.columns if col != 'Potability']

        for col in feature_cols:
            noise = np.random.normal(0, noise_level * df_noisy[col].std(), size=len(df_noisy))
            df_noisy[col] = df_noisy[col] + noise

        # Measure processing time
        start_time = time.time()

        # Feature engineering
        df_derived = create_derived_features(df_noisy)
        df_scaled, _ = scale_features(df_derived, scaler_type='standard')

        # Split and train
        X_train, X_test, y_train, y_test = split_data(
            df_scaled, test_size=0.2, random_state=42, stratify=True
        )

        model = XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        processing_time = time.time() - start_time

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            'noise_level': noise_level,
            'processing_time_sec': processing_time,
            'accuracy': accuracy,
            'recall': recall,
            'f1_score': f1,
            'accuracy_degradation': 0.0 if noise_level == 0 else (results[0]['accuracy'] - accuracy)
        })

        print(f"Processing time: {processing_time:.2f}s, Accuracy: {accuracy:.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / 'rq4_robustness_results.csv', index=False)
    print(f"\nRQ4 results saved to: {RESULTS_DIR / 'rq4_robustness_results.csv'}")

    return results_df


def run_rq5_shap_analysis() -> Tuple[pd.DataFrame, object]:
    """
    RQ5: Feature importance analysis and WHO standard alignment

    Returns:
        Tuple of (feature importance DataFrame, trained model)
    """
    print("RQ5: FEATURE IMPORTANCE ANALYSIS AND WHO ALIGNMENT")

    df_raw = load_raw_data()
    df_clean, _ = clean_dataset(df_raw, missing_strategy='median')
    df_derived = create_derived_features(df_clean)
    df_scaled, _ = scale_features(df_derived, scaler_type='standard')

    X_train, X_test, y_train, y_test = split_data(
        df_scaled, test_size=0.2, random_state=42, stratify=True
    )

    print("\n--- Training model for feature importance analysis ---")
    model = XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss')
    model.fit(X_train, y_train)

    print("\n--- Calculating feature importance ---")

    importance_values = model.feature_importances_

    # Convert to DataFrame with actual feature names
    importance_df = pd.DataFrame({
        'feature': X_test.columns,
        'importance_gain': importance_values
    }).sort_values('importance_gain', ascending=False).reset_index(drop=True)

    importance_df.to_csv(RESULTS_DIR / 'rq5_shap_importance.csv', index=False)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    prediction_df = pd.DataFrame({
        'prediction_probability': y_pred_proba
    })
    prediction_df.to_csv(RESULTS_DIR / 'rq5_predictions.csv', index=False)

    X_test.to_csv(RESULTS_DIR / 'rq5_test_features.csv', index=False)

    with open(RESULTS_DIR / 'rq5_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # WHO drinking water quality standards
    # Reference: WHO Guidelines for drinking-water quality, 4th edition (2017)
    # https://www.who.int/publications/i/item/9789241549950
    # Individual parameter sources:
    # - pH: https://cdn.who.int/media/docs/default-source/wash-documents/wash-chemicals/ph.pdf
    # - Turbidity, Hardness: WHO GDWQ 4th ed., Chemical fact sheets
    # - Chloramines: WHO guideline value is 3 mg/L for monochloramine
    #   (Note: 4.0 mg/L is US EPA MRDL, not WHO standard)
    # - Trihalomethanes: WHO sets individual THM limits (not total):
    #   Chloroform 300 μg/L, Bromoform/Dibromochloromethane 100 μg/L,
    #   Bromodichloromethane 60 μg/L
    #   (Note: 100 μg/L total THMs is EU Drinking Water Directive limit)
    #   https://www.who.int/docs/default-source/wash-documents/wash-chemicals/trihalomethanes.pdf
    who_standards = {
        'ph': {'min': 6.5, 'max': 8.5, 'importance': 'high'},  # WHO recommended range
        'Hardness': {'max': 500, 'importance': 'medium'},  # mg/L as CaCO3 (aesthetic guideline)
        'Sulfate': {'max': 500, 'importance': 'medium'},  # mg/L (no health-based guideline; aesthetic)
        'Chloramines': {'max': 3.0, 'importance': 'high'},  # mg/L (WHO guideline for monochloramine)
        'Turbidity': {'max': 5.0, 'importance': 'high'},  # NTU (WHO recommended value)
        'Trihalomethanes': {'max': 300, 'importance': 'high'},  # μg/L (using WHO chloroform guideline as proxy)
    }

    who_alignment = []
    for feature, standard in who_standards.items():
        if feature in importance_df['feature'].values:
            feature_rank = importance_df[importance_df['feature'] == feature].index[0] + 1
            importance_value = importance_df[importance_df['feature'] == feature]['importance_gain'].values[0]

            who_alignment.append({
                'feature': feature,
                'who_importance': standard['importance'],
                'feature_rank': feature_rank,
                'importance_gain': importance_value,
                'alignment': 'high' if feature_rank <= 10 else 'medium' if feature_rank <= 15 else 'low'
            })

    who_df = pd.DataFrame(who_alignment)
    who_df.to_csv(RESULTS_DIR / 'rq5_who_alignment.csv', index=False)

    print(f"\nRQ5 results saved to data/ directory")
    print(f"Top 5 features by importance gain:")
    print(importance_df.head())

    return importance_df, model


def run_all_experiments():
    
    print("RUNNING ALL RESEARCH QUESTION EXPERIMENTS")

    # RQ1: Imputation Strategy
    rq1_results = run_rq1_imputation_comparison()

    # RQ2: Feature Engineering
    rq2_results = run_rq2_feature_engineering_comparison()

    # RQ3: Dual-Layer Safety
    rq3_corr, rq3_detailed = run_rq3_dual_layer_safety()

    # RQ4: Pipeline Robustness
    rq4_results = run_rq4_pipeline_robustness()

    # RQ5: SHAP Analysis
    rq5_shap, rq5_model = run_rq5_shap_analysis()

    print("ALL EXPERIMENTS COMPLETED")
    print(f"Results saved to: {RESULTS_DIR}")
    print("\nGenerated files:")
    print("  - rq1_imputation_results.csv")
    print("  - rq2_feature_engineering_results.csv")
    print("  - rq3_correlation_results.csv")
    print("  - rq3_detailed_anomaly_analysis.csv")
    print("  - rq3_quartile_analysis.csv")
    print("  - rq4_robustness_results.csv")
    print("  - rq5_shap_importance.csv")
    print("  - rq5_shap_values.csv")
    print("  - rq5_test_features.csv")
    print("  - rq5_who_alignment.csv")
    print("  - rq5_model.pkl")


if __name__ == "__main__":
    run_all_experiments()
