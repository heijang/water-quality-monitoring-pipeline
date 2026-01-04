"""Water Quality Monitoring Pipeline DAG"""

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta
from pathlib import Path
import logging
import sys

logger = logging.getLogger(__name__)

# Add project src to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

# Import project utilities
from utils import PROJECT_ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR, TABLES_DIR, STORAGE_MODE

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Default Arguments
# =============================================================================

default_args = {
    'owner': 'water-quality-team',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# =============================================================================
# Task Functions
# =============================================================================

def init_database(**context):
    """Initialize database and schema"""
    from utils.init_database import initialize_water_quality_db

    logger.info("="*80)
    logger.info("TASK 0: DATABASE INITIALIZATION")
    logger.info("="*80)

    success = initialize_water_quality_db()

    if not success:
        raise Exception("Database initialization failed")

    logger.info("Database initialization completed successfully")
    logger.info("="*80)

    return "database_ready"


def extract_data(**context):
    """Extract raw water quality data from database or CSV"""
    from data_ingestion.load_data import load_raw_data, save_data, print_data_summary

    logger.info("="*80)
    logger.info("TASK 1: DATA EXTRACTION")
    logger.info(f"Storage Mode: {STORAGE_MODE}")
    logger.info("="*80)

    use_db = (STORAGE_MODE == "DB")
    df = load_raw_data(use_db=use_db)
    print_data_summary(df)

    output_path = RESULTS_DIR / 'raw_data.csv'
    save_data(df, str(output_path), table_name="raw_water_quality", storage_mode=STORAGE_MODE)

    logger.info(f"Data extraction completed. Records: {len(df)}")
    logger.info("="*80)

    context['ti'].xcom_push(key='n_records', value=len(df))
    context['ti'].xcom_push(key='n_features', value=len(df.columns) - 1)


def clean_data(**context):
    """Clean data: handle missing values, remove duplicates, detect outliers"""
    from data_cleaning.clean_data import clean_dataset, save_cleaned_data
    from utils import load_data_with_mode

    logger.info("="*80)
    logger.info("TASK 2: DATA CLEANING")
    logger.info(f"Storage Mode: {STORAGE_MODE}")
    logger.info("="*80)

    raw_data_path = RESULTS_DIR / 'raw_data.csv'
    df = load_data_with_mode(str(raw_data_path), "raw_water_quality", STORAGE_MODE)

    df_clean, metadata = clean_dataset(
        df,
        missing_strategy='median',
        remove_dup=True,
        detect_outlier=True
    )

    output_path = RESULTS_DIR / 'cleaned_data.csv'
    save_cleaned_data(df_clean, str(output_path), table_name="cleaned_water_quality", storage_mode=STORAGE_MODE)

    logger.info(f"Data cleaning completed. Clean records: {len(df_clean)}")
    logger.info("="*80)

    context['ti'].xcom_push(key='cleaned_records', value=len(df_clean))


def transform_features(**context):
    """Engineer features: create derived features, scale data, and split into train/test"""
    from feature_engineering.transform_features import (
        feature_engineering_pipeline,
        split_data,
        save_engineered_data,
        save_scaler
    )
    from utils import load_data_with_mode

    logger.info("="*80)
    logger.info("TASK 3: FEATURE ENGINEERING")
    logger.info(f"Storage Mode: {STORAGE_MODE}")
    logger.info("="*80)

    cleaned_data_path = RESULTS_DIR / 'cleaned_data.csv'
    df = load_data_with_mode(str(cleaned_data_path), "cleaned_water_quality", STORAGE_MODE)

    df_transformed, metadata = feature_engineering_pipeline(
        df,
        create_derived=True,
        create_interactions=False,
        scale=True,
        scaler_type='standard'
    )

    X_train, X_test, y_train, y_test = split_data(
        df_transformed,
        test_size=0.2,
        random_state=42,
        stratify=True
    )

    save_engineered_data(X_train, X_test, y_train, y_test, str(RESULTS_DIR))

    if 'scaler' in metadata:
        scaler_path = RESULTS_DIR / 'scaler.pkl'
        save_scaler(metadata['scaler'], str(scaler_path))

    logger.info(f"Feature engineering completed. Features: {len(X_train.columns)}")
    logger.info("="*80)

    context['ti'].xcom_push(key='n_features_engineered', value=len(X_train.columns))
    context['ti'].xcom_push(key='train_size', value=len(X_train))
    context['ti'].xcom_push(key='test_size', value=len(X_test))


def train_models(**context):
    """Train baseline XGBoost model"""
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score, recall_score, f1_score
    import pandas as pd
    import pickle

    logger.info("="*80)
    logger.info("TASK 4: BASELINE MODEL TRAINING")
    logger.info("="*80)

    X_train = pd.read_csv(RESULTS_DIR / 'X_train.csv')
    X_test = pd.read_csv(RESULTS_DIR / 'X_test.csv')
    y_train = pd.read_csv(RESULTS_DIR / 'y_train.csv').squeeze()
    y_test = pd.read_csv(RESULTS_DIR / 'y_test.csv').squeeze()

    model = XGBClassifier(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    model_path = RESULTS_DIR / 'baseline_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    logger.info(f"Baseline model training completed.")
    logger.info(f"Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    logger.info("="*80)

    context['ti'].xcom_push(key='baseline_accuracy', value=accuracy)
    context['ti'].xcom_push(key='baseline_recall', value=recall)
    context['ti'].xcom_push(key='baseline_f1', value=f1)


def run_experiments(**context):
    """Run all experiments for RQ1-RQ5"""
    from experiments.run_experiments import run_all_experiments

    logger.info("="*80)
    logger.info("TASK 5: RUNNING ALL RESEARCH EXPERIMENTS")
    logger.info("="*80)

    run_all_experiments()

    logger.info("All experiments completed.")
    logger.info("="*80)

    context['ti'].xcom_push(key='experiments_completed', value=True)


def generate_outputs(**context):
    """Generate all figures (PDF) and tables (XLSX) for research questions"""
    from evaluation.generate_outputs import generate_all_outputs

    logger.info("="*80)
    logger.info("TASK 6: GENERATE OUTPUTS")
    logger.info("="*80)

    generate_all_outputs()

    logger.info("Output generation completed.")
    logger.info("="*80)


def pipeline_summary(**context):
    """Print pipeline execution summary"""
    ti = context['ti']

    logger.info("="*80)
    logger.info("WATER QUALITY PIPELINE EXECUTION SUMMARY")
    logger.info("="*80)

    n_records = ti.xcom_pull(task_ids='extract_data', key='n_records')
    n_features = ti.xcom_pull(task_ids='extract_data', key='n_features')
    cleaned_records = ti.xcom_pull(task_ids='clean_data', key='cleaned_records')
    n_features_eng = ti.xcom_pull(task_ids='transform_features', key='n_features_engineered')
    train_size = ti.xcom_pull(task_ids='transform_features', key='train_size')
    test_size = ti.xcom_pull(task_ids='transform_features', key='test_size')
    baseline_accuracy = ti.xcom_pull(task_ids='train_models', key='baseline_accuracy')
    baseline_recall = ti.xcom_pull(task_ids='train_models', key='baseline_recall')
    baseline_f1 = ti.xcom_pull(task_ids='train_models', key='baseline_f1')
    experiments_completed = ti.xcom_pull(task_ids='run_experiments', key='experiments_completed')

    logger.info("1. DATA EXTRACTION")
    logger.info(f"   - Total records: {n_records}")
    logger.info(f"   - Original features: {n_features}")

    logger.info("2. DATA CLEANING")
    logger.info(f"   - Clean records: {cleaned_records}")
    logger.info(f"   - Records removed: {n_records - cleaned_records if n_records and cleaned_records else 0}")

    logger.info("3. FEATURE ENGINEERING")
    logger.info(f"   - Engineered features: {n_features_eng}")
    logger.info(f"   - Training set size: {train_size}")
    logger.info(f"   - Testing set size: {test_size}")

    logger.info("4. BASELINE MODEL TRAINING")
    logger.info(f"   - Baseline accuracy: {baseline_accuracy:.4f}" if baseline_accuracy else "   - N/A")
    logger.info(f"   - Baseline recall: {baseline_recall:.4f}" if baseline_recall else "   - N/A")
    logger.info(f"   - Baseline F1: {baseline_f1:.4f}" if baseline_f1 else "   - N/A")

    logger.info("5. RESEARCH EXPERIMENTS")
    logger.info(f"   - All RQ1-RQ5 experiments completed: {experiments_completed}")
    logger.info("   - RQ1: Imputation strategy comparison")
    logger.info("   - RQ2: Feature engineering comparison")
    logger.info("   - RQ3: Dual-layer safety analysis")
    logger.info("   - RQ4: Pipeline robustness testing")
    logger.info("   - RQ5: SHAP and WHO alignment")

    logger.info("6. OUTPUT GENERATION")
    logger.info("   - Figures: 10 PDFs generated")
    logger.info("     * RQ1: 2 figures")
    logger.info("     * RQ2: 2 figures")
    logger.info("     * RQ3: 2 figures")
    logger.info("     * RQ4: 2 figures")
    logger.info("     * RQ5: 2 figures")
    logger.info("   - Tables: 7 XLSX files generated")
    logger.info("     * RQ1: 1 table")
    logger.info("     * RQ2: 1 table")
    logger.info("     * RQ3: 2 tables")
    logger.info("     * RQ4: 1 table")
    logger.info("     * RQ5: 2 tables")

    logger.info("="*80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("="*80)


# =============================================================================
# DAG Definition
# =============================================================================

with DAG(
    'water_quality_pipeline',
    default_args=default_args,
    description='Complete data pipeline for water potability prediction',
    schedule=None,  # Manual trigger only
    catchup=False,
    tags=['data-engineering', 'machine-learning', 'water-quality'],
) as dag:

    init_db = PythonOperator(
        task_id='initialize_database',
        python_callable=init_database,
    )

    extract = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
    )

    clean = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
    )

    transform = PythonOperator(
        task_id='transform_features',
        python_callable=transform_features,
    )

    train = PythonOperator(
        task_id='train_models',
        python_callable=train_models,
    )

    experiments = PythonOperator(
        task_id='run_experiments',
        python_callable=run_experiments,
    )

    generate = PythonOperator(
        task_id='generate_outputs',
        python_callable=generate_outputs,
    )

    summary = PythonOperator(
        task_id='pipeline_summary',
        python_callable=pipeline_summary,
    )

    verify = BashOperator(
        task_id='verify_outputs',
        bash_command="""
        echo "Verifying generated files..."
        echo "Figures:"
        ls -lh {{ params.figures_dir }}/*.pdf 2>/dev/null || echo "  No PDF files found"
        echo ""
        echo "Tables:"
        ls -lh {{ params.tables_dir }}/*.xlsx 2>/dev/null || echo "  No XLSX files found"
        echo ""
        echo "Verification complete!"
        """,
        params={'figures_dir': str(FIGURES_DIR), 'tables_dir': str(TABLES_DIR)},
    )

    # Task dependencies
    init_db >> extract >> clean >> transform >> train >> experiments >> generate >> summary >> verify
