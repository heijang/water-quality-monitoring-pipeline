
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report)
from typing import Dict, Tuple, List
import pickle
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Skipping XGBoost model.")


def get_baseline_models() -> Dict[str, object]:
    """
    Get dictionary of baseline models with default parameters

    Returns:
        Dictionary mapping model names to model objects
    """
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB()
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            random_state=42,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            eval_metric='logloss'
        )

    return models


def train_baseline_models(X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Train and evaluate baseline models

    Args:
        X_train, y_train: Training data
        X_test, y_test: Testing data

    Returns:
        DataFrame containing model performance metrics
    """
    print("TRAINING BASELINE MODELS")

    models = get_baseline_models()
    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = y_test_pred

        metrics = {
            'Model': name,
            'Train Accuracy': accuracy_score(y_train, y_train_pred),
            'Test Accuracy': accuracy_score(y_test, y_test_pred),
            'Precision': precision_score(y_test, y_test_pred, zero_division=0),
            'Recall': recall_score(y_test, y_test_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_test_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y_test, y_test_proba)
        }

        results.append(metrics)

        print(f"  Test Accuracy: {metrics['Test Accuracy']:.4f}")
        print(f"  F1 Score: {metrics['F1 Score']:.4f}")
        print(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test Accuracy', ascending=False)

    print("BASELINE MODEL RESULTS")
    print(results_df.to_string(index=False))

    return results_df


def perform_cross_validation(X_train: pd.DataFrame, y_train: pd.Series,
                             cv: int = 5) -> pd.DataFrame:
    """
    Perform cross-validation on baseline models

    Args:
        X_train, y_train: Training data
        cv: Number of cross-validation folds

    Returns:
        DataFrame with cross-validation scores
    """
    print(f"CROSS-VALIDATION ({cv}-FOLD)")

    models = get_baseline_models()
    cv_results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")

        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

        cv_results.append({
            'Model': name,
            'Mean CV Accuracy': scores.mean(),
            'Std CV Accuracy': scores.std(),
            'Min CV Accuracy': scores.min(),
            'Max CV Accuracy': scores.max()
        })

        print(f"  Mean Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

    cv_df = pd.DataFrame(cv_results)
    cv_df = cv_df.sort_values('Mean CV Accuracy', ascending=False)

    print("CROSS-VALIDATION RESULTS")
    print(cv_df.to_string(index=False))

    return cv_df


def hyperparameter_tuning(X_train: pd.DataFrame, y_train: pd.Series,
                          model_name: str = 'Random Forest') -> Tuple[object, dict]:
    """
    Perform hyperparameter tuning for specified model

    Args:
        X_train, y_train: Training data
        model_name: Name of model to tune

    Returns:
        Tuple of (best model, best parameters)
    """
    print(f"HYPERPARAMETER TUNING - {model_name}")

    if model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    elif model_name == 'Gradient Boosting':
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }

    elif model_name == 'Logistic Regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }

    elif model_name == 'XGBoost' and XGBOOST_AVAILABLE:
        model = XGBClassifier(random_state=42, eval_metric='logloss')
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

    else:
        raise ValueError(f"Hyperparameter tuning not implemented for {model_name}")

    print(f"Searching over {len(param_grid)} parameters...")
    print(f"Parameter grid: {param_grid}")

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str = "Model") -> dict:
    """
    Comprehensive model evaluation

    Args:
        model: Trained model
        X_test, y_test: Testing data
        model_name: Name of the model

    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"EVALUATING {model_name}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    print("\nPerformance Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:4d}  |  FP: {cm[0,1]:4d}")
    print(f"  FN: {cm[1,0]:4d}  |  TP: {cm[1,1]:4d}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Potable', 'Potable']))


    return metrics


def get_feature_importance(model: object, feature_names: List[str],
                           top_n: int = 15) -> pd.DataFrame:
    """
    Get feature importance from trained model

    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False)

        print(f"TOP {top_n} FEATURE IMPORTANCES")
        print(feature_importance.head(top_n).to_string(index=False))

        return feature_importance

    else:
        print(f"Model does not have feature_importances_ attribute")
        return None


def train_isolation_forest(X_train: pd.DataFrame, y_train: pd.Series,
                           X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[object, dict]:
    """
    Train Isolation Forest for anomaly-based classification.
    Outliers are treated as non-potable water samples.
    """
    print("TRAINING ISOLATION FOREST (ANOMALY DETECTION APPROACH)")

    # Calculate proportion of class 0 (non-potable)
    raw_contamination = y_train.value_counts()[0] / len(y_train)
    
    # IsolationForest 'contamination' must be in range (0.0, 0.5]
    # Cap the value at 0.5 to prevent InvalidParameterError
    contamination = min(raw_contamination, 0.5)

    if raw_contamination > 0.5:
        print(f"  Note: Raw ratio ({raw_contamination:.4f}) capped to 0.5 due to sklearn constraints.")

    model = IsolationForest(
        random_state=42,
        contamination=contamination,
        n_estimators=100,
        max_samples='auto'
    )

    # Fit the model
    model.fit(X_train)

    # Predict: -1 for anomalies (class 0), 1 for normal (class 1)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Convert predictions to binary format (0 and 1)
    y_pred_train_binary = np.where(y_pred_train == -1, 0, 1)
    y_pred_test_binary = np.where(y_pred_test == -1, 0, 1)

    metrics = {
        'Train Accuracy': accuracy_score(y_train, y_pred_train_binary),
        'Test Accuracy': accuracy_score(y_test, y_pred_test_binary),
        'Precision': precision_score(y_test, y_pred_test_binary, zero_division=0),
        'Recall': recall_score(y_test, y_pred_test_binary, zero_division=0),
        'F1 Score': f1_score(y_test, y_pred_test_binary, zero_division=0)
    }

    print(f"\nApplied Contamination rate: {contamination:.4f}")
    print(f"Test Accuracy: {metrics['Test Accuracy']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")

    return model, metrics


def save_model(model: object, output_path: str):
    """
    Save trained model to file

    Args:
        model: Trained model
        output_path: Output file path
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"\nModel saved to: {output_path}")


if __name__ == "__main__":
    # Import utils for project paths
    import sys
    project_root = Path(__file__).parent.parent.parent
    if str(project_root / 'src') not in sys.path:
        sys.path.insert(0, str(project_root / 'src'))

    from utils import RESULTS_DIR

    print("Loading data...")

    X_train = pd.read_csv(RESULTS_DIR / 'X_train.csv')
    X_test = pd.read_csv(RESULTS_DIR / 'X_test.csv')
    y_train = pd.read_csv(RESULTS_DIR / 'y_train.csv').squeeze()
    y_test = pd.read_csv(RESULTS_DIR / 'y_test.csv').squeeze()

    print(f"Training set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")

    baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)

    # Cross-validation
    cv_results = perform_cross_validation(X_train, y_train, cv=5)

    print("\n\nTraining Isolation Forest as alternative approach...")
    isolation_model, isolation_metrics = train_isolation_forest(X_train, y_train, X_test, y_test)

    # Hyperparameter tuning for best model
    best_model_name = baseline_results.iloc[0]['Model']
    print(f"\n\nPerforming hyperparameter tuning for: {best_model_name}")

    if best_model_name == 'Random Forest':
        best_model, best_params = hyperparameter_tuning(X_train, y_train, 'Random Forest')
    elif best_model_name == 'Gradient Boosting':
        best_model, best_params = hyperparameter_tuning(X_train, y_train, 'Gradient Boosting')
    elif best_model_name == 'XGBoost' and XGBOOST_AVAILABLE:
        best_model, best_params = hyperparameter_tuning(X_train, y_train, 'XGBoost')
    else:
        # Default to Random Forest if best model doesn't have tuning implemented
        best_model, best_params = hyperparameter_tuning(X_train, y_train, 'Random Forest')

    # Evaluate best model
    metrics = evaluate_model(best_model, X_test, y_test, "Tuned Model")

    # Feature importance
    feature_importance = get_feature_importance(best_model, X_train.columns.tolist(), top_n=15)

    # Save model (always local for pickle files)
    model_path = RESULTS_DIR / 'best_model.pkl'
    save_model(best_model, str(model_path))

    # Save results locally (intermediate results)
    baseline_results.to_csv(RESULTS_DIR / 'baseline_results.csv', index=False)
    cv_results.to_csv(RESULTS_DIR / 'cv_results.csv', index=False)

    if feature_importance is not None:
        feature_importance.to_csv(RESULTS_DIR / 'feature_importance.csv', index=False)

    print("MODEL TRAINING COMPLETED")
