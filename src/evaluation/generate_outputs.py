
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pickle

# Add parent to path
project_root = Path(__file__).parent.parent.parent
if str(project_root / 'src') not in sys.path:
    sys.path.insert(0, str(project_root / 'src'))

from utils import RESULTS_DIR, FIGURES_DIR, TABLES_DIR

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# All RQ result files (rq1~rq5) are saved and read from LOCAL CSV only
def generate_rq1_outputs():
    """
    RQ1: Imputation Strategy Comparison
    - Fig1: Bar chart comparing recall and stability across strategies
    - Table1: Detailed metrics for all strategies
    """
    print("Generating RQ1 Outputs")

    # Read from LOCAL CSV (faster than DB for graph generation)
    df = pd.read_csv(RESULTS_DIR / 'rq1_imputation_results.csv')

    # Figure 1: Recall and Stability Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Recall comparison
    ax1.bar(df['strategy'], df['recall'], color=['#2ecc71', '#3498db', '#e74c3c'])
    ax1.set_xlabel('Imputation Strategy', fontsize=12)
    ax1.set_ylabel('Recall Score', fontsize=12)
    ax1.set_title('RQ1: XGBoost Recall by Imputation Strategy', fontsize=13, fontweight='bold')
    ax1.set_ylim([df['recall'].min() - 0.05, df['recall'].max() + 0.05])
    for i, v in enumerate(df['recall']):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)

    # Stability comparison
    ax2.bar(df['strategy'], df['stability_score'], color=['#2ecc71', '#3498db', '#e74c3c'])
    ax2.set_xlabel('Imputation Strategy', fontsize=12)
    ax2.set_ylabel('Stability Score', fontsize=12)
    ax2.set_title('RQ1: Prediction Stability by Imputation Strategy', fontsize=13, fontweight='bold')
    ax2.set_ylim([df['stability_score'].min() - 0.01, 1.0])
    for i, v in enumerate(df['stability_score']):
        ax2.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'RQ1_Fig1.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: RQ1_Fig1.pdf")

    # Figure 2: Cross-validation recall distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width/2, df['cv_recall_mean'], width, label='Mean CV Recall',
           yerr=df['cv_recall_std'], capsize=5, color='#3498db', alpha=0.8)
    ax.bar(x + width/2, df['recall'], width, label='Test Recall',
           color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Imputation Strategy', fontsize=12)
    ax.set_ylabel('Recall Score', fontsize=12)
    ax.set_title('RQ1: Cross-Validation vs Test Recall', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['strategy'])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'RQ1_Fig2.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: RQ1_Fig2.pdf")

    # Table 1: Complete metrics
    table1 = df[[
        'strategy', 'accuracy', 'precision', 'recall', 'f1_score',
        'roc_auc', 'cv_recall_mean', 'cv_recall_std', 'stability_score'
    ]].round(4)

    table1.to_excel(TABLES_DIR / 'RQ1_Table1.xlsx', index=False, engine='openpyxl')
    print(f"Generated: RQ1_Table1.xlsx")


def generate_rq2_outputs():
    """
    RQ2: Feature Engineering Comparison
    - Fig1: Bar chart comparing raw vs derived features for unsafe water detection
    - Table1: Detailed comparison metrics
    """
    print("Generating RQ2 Outputs")

    df = pd.read_csv(RESULTS_DIR / 'rq2_feature_engineering_results.csv')

    # Figure 1: Unsafe water detection comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['recall_unsafe', 'precision_unsafe', 'f1_unsafe']
    x = np.arange(len(metrics))
    width = 0.35

    raw_values = df[df['approach'] == 'raw_features'][metrics].values[0]
    derived_values = df[df['approach'] == 'derived_features'][metrics].values[0]

    ax.bar(x - width/2, raw_values, width, label='Raw Features', color='#95a5a6', alpha=0.8)
    ax.bar(x + width/2, derived_values, width, label='Derived Features', color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('RQ2: Unsafe Water Detection - Raw vs Derived Features', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Recall', 'Precision', 'F1-Score'])
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (r, d) in enumerate(zip(raw_values, derived_values)):
        ax.text(i - width/2, r + 0.02, f'{r:.3f}', ha='center', fontsize=9)
        ax.text(i + width/2, d + 0.02, f'{d:.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'RQ2_Fig1.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: RQ2_Fig1.pdf")

    # Figure 2: Overall accuracy and feature count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy comparison
    approaches = df['approach'].str.replace('_', ' ').str.title()
    ax1.bar(approaches, df['accuracy'], color=['#95a5a6', '#2ecc71'], alpha=0.8)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Overall Accuracy', fontsize=12, fontweight='bold')
    ax1.set_ylim([df['accuracy'].min() - 0.05, 1.0])
    for i, v in enumerate(df['accuracy']):
        ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', fontsize=10)

    # Feature count
    ax2.bar(approaches, df['n_features'], color=['#95a5a6', '#2ecc71'], alpha=0.8)
    ax2.set_ylabel('Number of Features', fontsize=12)
    ax2.set_title('Feature Dimensionality', fontsize=12, fontweight='bold')
    for i, v in enumerate(df['n_features']):
        ax2.text(i, v + 0.5, str(int(v)), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'RQ2_Fig2.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: RQ2_Fig2.pdf")

    # Table 1: Complete comparison
    table1 = df.round(4)
    table1.to_excel(TABLES_DIR / 'RQ2_Table1.xlsx', index=False, engine='openpyxl')
    print(f"Generated: RQ2_Table1.xlsx")


def generate_rq3_outputs():
    """
    RQ3: Dual-Layer Safety Analysis
    - Fig1: Scatter plot of anomaly scores vs misclassification
    - Fig2: Misclassification rate by anomaly score quartile
    - Table1: Correlation statistics
    - Table2: Quartile analysis
    """
    print("Generating RQ3 Outputs")

    correlation_df = pd.read_csv(RESULTS_DIR / 'rq3_correlation_results.csv')
    detailed_df = pd.read_csv(RESULTS_DIR / 'rq3_detailed_anomaly_analysis.csv')
    quartile_df = pd.read_csv(RESULTS_DIR / 'rq3_quartile_analysis.csv')

    # Figure 1: Scatter plot with correlation
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2ecc71' if m == 0 else '#e74c3c' for m in detailed_df['misclassified']]
    ax.scatter(detailed_df['anomaly_score'], detailed_df['misclassified'],
               c=colors, alpha=0.6, s=50)

    ax.set_xlabel('Anomaly Score (Higher = More Anomalous)', fontsize=12)
    ax.set_ylabel('Misclassified (0=Correct, 1=Wrong)', fontsize=12)
    ax.set_title(f'RQ3: Anomaly Score vs Misclassification\n' +
                 f'Pearson r={correlation_df["pearson_correlation"].values[0]:.3f}, ' +
                 f'Spearman ρ={correlation_df["spearman_correlation"].values[0]:.3f}',
                 fontsize=13, fontweight='bold')

    # Add trend line
    z = np.polyfit(detailed_df['anomaly_score'], detailed_df['misclassified'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(detailed_df['anomaly_score'].min(), detailed_df['anomaly_score'].max(), 100)
    ax.plot(x_trend, p(x_trend), "b--", alpha=0.8, linewidth=2, label='Trend line')

    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'RQ3_Fig1.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: RQ3_Fig1.pdf")

    # Figure 2: Misclassification rate by quartile
    fig, ax = plt.subplots(figsize=(10, 6))

    colors_q = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
    bars = ax.bar(quartile_df['quartile'], quartile_df['misclassification_rate'],
                   color=colors_q, alpha=0.8)

    ax.set_xlabel('Anomaly Score Quartile', fontsize=12)
    ax.set_ylabel('Misclassification Rate', fontsize=12)
    ax.set_title('RQ3: Misclassification Rate by Anomaly Score Quartile',
                 fontsize=13, fontweight='bold')
    ax.set_ylim([0, quartile_df['misclassification_rate'].max() + 0.1])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, quartile_df['misclassification_rate'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'RQ3_Fig2.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: RQ3_Fig2.pdf")

    # Table 1: Correlation results
    correlation_df.round(4).to_excel(TABLES_DIR / 'RQ3_Table1.xlsx', index=False, engine='openpyxl')
    print(f"Generated: RQ3_Table1.xlsx")

    # Table 2: Quartile analysis
    quartile_df.round(4).to_excel(TABLES_DIR / 'RQ3_Table2.xlsx', index=False, engine='openpyxl')
    print(f"Generated: RQ3_Table2.xlsx")


def generate_rq4_outputs():
    """
    RQ4: Pipeline Robustness
    - Fig1: Performance degradation under noise
    - Fig2: Processing time vs noise level
    - Table1: Complete robustness metrics
    """
    print("Generating RQ4 Outputs")

    df = pd.read_csv(RESULTS_DIR / 'rq4_robustness_results.csv')

    # Figure 1: Performance metrics under noise
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df['noise_level'] * 100, df['accuracy'], marker='o', linewidth=2,
            label='Accuracy', color='#3498db')
    ax.plot(df['noise_level'] * 100, df['recall'], marker='s', linewidth=2,
            label='Recall', color='#e74c3c')
    ax.plot(df['noise_level'] * 100, df['f1_score'], marker='^', linewidth=2,
            label='F1-Score', color='#2ecc71')

    ax.set_xlabel('Noise Level (%)', fontsize=12)
    ax.set_ylabel('Performance Score', fontsize=12)
    ax.set_title('RQ4: Model Performance Under Data Noise', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'RQ4_Fig1.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: RQ4_Fig1.pdf")

    # Figure 2: Processing time
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(df['noise_level'] * 100, df['processing_time_sec'],
                   color='#9b59b6', alpha=0.8)

    ax.set_xlabel('Noise Level (%)', fontsize=12)
    ax.set_ylabel('Processing Time (seconds)', fontsize=12)
    ax.set_title('RQ4: Pipeline Processing Latency Under Noise', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.2f}s', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'RQ4_Fig2.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: RQ4_Fig2.pdf")

    # Table 1: Complete metrics
    table1 = df.round(4)
    table1.to_excel(TABLES_DIR / 'RQ4_Table1.xlsx', index=False, engine='openpyxl')
    print(f"Generated: RQ4_Table1.xlsx")


def generate_rq5_outputs():
    """
    RQ5: Feature Importance Analysis and WHO Alignment
    - Fig1: Top features by importance gain
    - Fig2: WHO standard alignment
    - Table1: Feature importance
    - Table2: WHO alignment analysis
    """
    print("Generating RQ5 Outputs")

    feature_importance = pd.read_csv(RESULTS_DIR / 'rq5_shap_importance.csv')
    who_alignment = pd.read_csv(RESULTS_DIR / 'rq5_who_alignment.csv')

    # Figure 1: Top 15 features by importance gain
    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = feature_importance.head(15)
    colors = ['#e74c3c' if 'ph' in f or 'Chloramines' in f or 'Turbidity' in f
              else '#3498db' for f in top_features['feature']]

    ax.barh(range(len(top_features)), top_features['importance_gain'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (Gain)', fontsize=12)
    ax.set_title('RQ5: Top 15 Features by Importance', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, v in enumerate(top_features['importance_gain']):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'RQ5_Fig1.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: RQ5_Fig1.pdf")

    # Figure 2: WHO alignment analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    who_sorted = who_alignment.sort_values('feature_rank')
    colors_who = {'high': '#e74c3c', 'medium': '#f39c12', 'low': '#95a5a6'}
    bar_colors = [colors_who[imp] for imp in who_sorted['who_importance']]

    x_pos = np.arange(len(who_sorted))
    ax.bar(x_pos, who_sorted['feature_rank'], color=bar_colors, alpha=0.8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(who_sorted['feature'], rotation=45, ha='right')
    ax.set_ylabel('Feature Rank (Lower = More Important)', fontsize=12)
    ax.set_xlabel('WHO Key Parameters', fontsize=12)
    ax.set_title('RQ5: WHO Parameter Importance vs Model Feature Ranking', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='WHO: High Importance'),
        Patch(facecolor='#f39c12', label='WHO: Medium Importance'),
        Patch(facecolor='#95a5a6', label='WHO: Low Importance')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Add value labels
    for i, (idx, row) in enumerate(who_sorted.iterrows()):
        ax.text(i, row['feature_rank'], f"#{int(row['feature_rank'])}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'RQ5_Fig2.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Generated: RQ5_Fig2.pdf")

    # Table 1: Complete feature importance
    feature_importance.head(20).round(4).to_excel(TABLES_DIR / 'RQ5_Table1.xlsx', index=False, engine='openpyxl')
    print(f"Generated: RQ5_Table1.xlsx")

    # Table 2: WHO alignment
    who_alignment.round(4).to_excel(TABLES_DIR / 'RQ5_Table2.xlsx', index=False, engine='openpyxl')
    print(f"Generated: RQ5_Table2.xlsx")


def generate_all_outputs():
    
    print("GENERATING ALL OUTPUTS FOR RESEARCH QUESTIONS")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # RQ1: Imputation Strategy
        generate_rq1_outputs()

        # RQ2: Feature Engineering
        generate_rq2_outputs()

        # RQ3: Dual-Layer Safety
        generate_rq3_outputs()

        # RQ4: Pipeline Robustness
        generate_rq4_outputs()

        # RQ5: SHAP Analysis
        generate_rq5_outputs()

        print("ALL OUTPUTS GENERATED SUCCESSFULLY")
        print(f"\nFigures saved to: {FIGURES_DIR}")
        print(f"Tables saved to: {TABLES_DIR}")
        print("\nGenerated files:")
        print("\nFigures (PDF):")
        print("  - RQ1_Fig1.pdf: Recall and stability by imputation strategy")
        print("  - RQ1_Fig2.pdf: Cross-validation vs test recall")
        print("  - RQ2_Fig1.pdf: Unsafe water detection comparison")
        print("  - RQ2_Fig2.pdf: Overall accuracy and feature count")
        print("  - RQ3_Fig1.pdf: Anomaly score vs misclassification scatter")
        print("  - RQ3_Fig2.pdf: Misclassification rate by anomaly quartile")
        print("  - RQ4_Fig1.pdf: Performance under noise")
        print("  - RQ4_Fig2.pdf: Processing latency under noise")
        print("  - RQ5_Fig1.pdf: Top features by SHAP importance")
        print("  - RQ5_Fig2.pdf: WHO alignment analysis")
        print("\nTables (XLSX):")
        print("  - RQ1_Table1.xlsx: Complete imputation metrics")
        print("  - RQ2_Table1.xlsx: Feature engineering comparison")
        print("  - RQ3_Table1.xlsx: Correlation statistics")
        print("  - RQ3_Table2.xlsx: Quartile analysis")
        print("  - RQ4_Table1.xlsx: Robustness metrics")
        print("  - RQ5_Table1.xlsx: SHAP feature importance")
        print("  - RQ5_Table2.xlsx: WHO alignment")

    except Exception as e:
        print(f"\nError generating outputs: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    generate_all_outputs()
