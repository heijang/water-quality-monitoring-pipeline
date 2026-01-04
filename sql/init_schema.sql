-- =============================================================================
-- CORE TABLES (Used when STORAGE_MODE=DB or BOTH)
-- =============================================================================

-- 1. Raw water quality data (original dataset)
CREATE TABLE IF NOT EXISTS raw_water_quality (
    id SERIAL PRIMARY KEY,
    ph FLOAT,
    hardness FLOAT,
    solids FLOAT,
    chloramines FLOAT,
    sulfate FLOAT,
    conductivity FLOAT,
    organic_carbon FLOAT,
    trihalomethanes FLOAT,
    turbidity FLOAT,
    potability INTEGER,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Cleaned/preprocessed water quality data
CREATE TABLE IF NOT EXISTS cleaned_water_quality (
    id SERIAL PRIMARY KEY,
    ph FLOAT,
    hardness FLOAT,
    solids FLOAT,
    chloramines FLOAT,
    sulfate FLOAT,
    conductivity FLOAT,
    organic_carbon FLOAT,
    trihalomethanes FLOAT,
    turbidity FLOAT,
    potability INTEGER,
    cleaning_strategy VARCHAR(50),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- EXPERIMENT TRACKING TABLES (Always used for experiment metadata)
-- =============================================================================

-- 3. Experiment metadata tracking
CREATE TABLE IF NOT EXISTS experiments (
    id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(100) NOT NULL,
    research_question VARCHAR(20),
    description TEXT,
    parameters JSONB,
    status VARCHAR(20) DEFAULT 'running',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- 4. Model performance metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiments(id) ON DELETE CASCADE,
    model_name VARCHAR(100),
    strategy VARCHAR(100),
    accuracy FLOAT,
    precision_score FLOAT,
    recall FLOAT,
    f1_score FLOAT,
    roc_auc FLOAT,
    cv_recall_mean FLOAT,
    cv_recall_std FLOAT,
    stability_score FLOAT,
    additional_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- INDEXES
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_experiments_rq ON experiments(research_question);
CREATE INDEX IF NOT EXISTS idx_model_metrics_exp ON model_metrics(experiment_id);