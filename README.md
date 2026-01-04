# Water Quality Monitoring Pipeline
* **Group Number:** WS25-DE15
* **Team Members:** Janghwan Lee, Nithin Indirala

## 1. Project Overview
This project implements an end-to-end data engineering pipeline to predict water potability using machine learning. The pipeline automates data ingestion, cleaning, feature engineering, and model training, resulting in the automated generation of research-ready figures and tables to ensure full reproducibility.

## 2. Dataset Links
* **Primary Dataset:** Water Potability Dataset (Kaggle)
* **Dataset Source:** [Water Quality](https://www.kaggle.com/datasets/adityakadiwal/water-potability)
* **Sample Data:** Located in data/sample/water_potability.csv

## 3. Research Questions
The project technical implementation answers the following research questions:
* **RQ1 (Imputation Strategy)**: How do different data imputation strategies for critical parameters (pH, Sulfate) influence the prediction stability and recall of XGBoost-based water potability models? 

* **RQ2 (Feature Engineering)**: To what extent does the incorporation of derived features based on physicochemical interactions improve the identification of unsafe water compared to raw parameter inputs? 

* **RQ3 (Dual-Layer Safety)**: What is the correlation between anomaly scores from Isolation Forest and misclassified samples in supervised models, and can this serve as a reliable safety net for real-time monitoring? 

* **RQ4 (Pipeline Robustness)**: How do simulated data noise and streaming ingestion affect the processing latency and performance robustness of the Airflow-managed ETL pipeline? 

* **RQ5 (Domain Alignment)**: To what degree do the model's decision-making drivers, identified via SHAP, align with established WHO drinking-water quality standards and domain knowledge?

## 4. How to Run the Code
Follow these instructions to execute the pipeline manually:

1. **Install Dependencies:**
   Run the command: `pip install -r requirements.txt`

2. **Execute Stage Scripts (Sequential):**
   - **Ingestion:** `python src/data_ingestion/load_data.py`
   - **Cleaning:** `python src/data_cleaning/clean_data.py`
   - **Engineering:** `python src/feature_engineering/transform_features.py`
   - **Modeling:** `python src/modeling/train_models.py`
   - **Experiments:** `python src/experiments/run_experiments.py`
   - **Evaluation:** `python src/evaluation/generate_outputs.py`

## 5. How to Run the Airflow DAG
The entire workflow is orchestrated using Apache Airflow to ensure logical sequencing and reproducibility.

### Using Docker Compose
1. **Build & Start:** Run `docker build -t airflow-simple:latest .` followed by `docker-compose up -d`.
2. **Access UI:** Go to `http://localhost:8080`
- (User: `admin` / Password: `./airflow/simple_auth_manager_passwords.json.generated`).
3. **Trigger:** Unpause and trigger the `water_quality_pipeline` DAG.
4. **Stop:** Run `docker-compose down`.

### Pipeline Workflow Logic
The DAG follows a strict sequential flow:
`initialize_database` → `extract_data` → `clean_data` → `transform_features` → `train_models` → `run_experiments` → `generate_outputs` → `pipeline_summary` → `verify_outputs`

## 6. Folder Structure Explanation
The repository is organized into a modular and professional structure to ensure scalability and maintainability:

```text
water-quality-monitoring-pipeline/
├── dags/                # Airflow DAG orchestration
├── src/                 # Modular source code
│   ├── data_ingestion/  # Stage 1: Data loading & validation
│   ├── data_cleaning/   # Stage 2: Imputation & outlier removal
│   ├── feature_engineering/ # Stage 3: Derived features & scaling
│   ├── modeling/        # Stage 4: XGBoost & Isolation Forest training
│   └── evaluation/      # Stage 5: Generation of PDFs and XLSX
├── data/
│   ├── sample/          # Original raw data
│   └── result/          # Pipeline artifacts: Intermediate CSVs and model (.pkl) files
├── figures/             # Auto-generated PDF figures (RQ1-RQ5)
├── tables/              # Auto-generated XLSX tables (RQ1-RQ5)
├── airflow/             # Configs, auth metadata, and logs
├── sql/                 # Database DDL files
├── docker-compose.yml   # Standalone Airflow configuration
├── dockerfile           # Custom image build script
└── requirements.txt     # Project dependencies
```

## 7. Storage Configuration
The pipeline uses a hybrid storage approach for efficiency:
- **Core datasets**: Can be stored in PostgreSQL (configurable via `STORAGE_MODE`)
- **Intermediate results**: Stored as CSV files only

### Storage Mode
```bash
export STORAGE_MODE="LOCAL"  # Use CSV files only (default)
export STORAGE_MODE="DB"     # Use PostgreSQL for core data
export STORAGE_MODE="BOTH"   # Save to both, read from LOCAL
```

### Database Tables (STORAGE_MODE=DB or BOTH)

**Core Data (STORAGE_MODE controlled):**
- **raw_water_quality** - Original dataset
- **cleaned_water_quality** - Preprocessed data

**Local-only Data (CSV files):**
- **Training/test splits:** X_train, X_test, y_train, y_test
- **RQ results:** rq1~rq5 experiment CSV files
- **Models:** *.pkl files (trained models)
- **Intermediate results:** baseline_results.csv, cv_results.csv