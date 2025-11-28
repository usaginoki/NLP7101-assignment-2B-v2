"""
Generate visualization figures for the exploration report.

Creates:
1. Confusion matrices (full pipeline 11x11, Stage 1 binary 2x2)
2. Training curves (MLP loss/F1 over epochs)
3. CodeT5p-770m training loss curves
4. Hyperparameter comparison charts
5. Model comparison bar charts

Usage:
    uv run python scripts/generate_report_figures.py
"""

import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory
FIGURES_DIR = Path("docs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_confusion_matrix_full_pipeline():
    """Plot 11x11 confusion matrix for full pipeline."""
    print("Generating full pipeline confusion matrix...")

    # Load data
    with open("outputs/metrics/pipeline_codet5_validation.json") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"])
    labels = ["Human"] + [f"AI-{i}" for i in range(1, 11)]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))

    # Normalize for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot heatmap
    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={'label': 'Proportion'})

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Full Pipeline Confusion Matrix (XGBoost + CodeT5p-770m)\nValidation Set: 100,000 samples | Accuracy: 90.35%',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_full_pipeline.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'confusion_matrix_full_pipeline.png'}")


def plot_confusion_matrix_stage1():
    """Plot 2x2 confusion matrix for Stage 1 binary classification."""
    print("Generating Stage 1 confusion matrix...")

    # Load MLP metadata for confusion matrix
    with open("models/stage1/MLP/metadata.json") as f:
        mlp_data = json.load(f)

    cm_mlp = np.array(mlp_data["confusion_matrix"])

    # XGBoost confusion matrix (calculated from pipeline data)
    # From metadata: accuracy=96.6%, recall_human~96.4%, recall_ai~90.3%
    # Validation has ~88,490 human and ~11,510 AI
    cm_xgb = np.array([
        [85251, 3239],  # Human: TN, FP
        [1121, 10389]   # AI: FN, TP
    ])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # XGBoost
    labels = ["Human", "AI"]
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens',
                xticklabels=labels, yticklabels=labels, ax=axes[0],
                annot_kws={'size': 14})
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('Actual', fontsize=11)
    axes[0].set_title('XGBoost (Best Model)\nAccuracy: 96.60% | Macro F1: 0.9259', fontsize=12, fontweight='bold')

    # MLP
    sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Oranges',
                xticklabels=labels, yticklabels=labels, ax=axes[1],
                annot_kws={'size': 14})
    axes[1].set_xlabel('Predicted', fontsize=11)
    axes[1].set_ylabel('Actual', fontsize=11)
    axes[1].set_title('MLP (Macro F1 Optimized)\nAccuracy: 89.18% | Macro F1: 0.7749', fontsize=12, fontweight='bold')

    plt.suptitle('Stage 1 Binary Classification: Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix_stage1.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'confusion_matrix_stage1.png'}")


def plot_mlp_training_curves():
    """Plot MLP training loss and macro F1 over epochs."""
    print("Generating MLP training curves...")

    # Parse training log
    epochs = []
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    val_macro_f1 = []

    log_path = "outputs/train_stage1_mlp_macro_f1.log"
    pattern = r"Epoch (\d+): train_loss=([\d.]+), train_acc=([\d.]+), val_loss=([\d.]+), val_acc=([\d.]+), val_macro_f1=([\d.]+)"

    with open(log_path) as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epochs.append(int(match.group(1)))
                train_loss.append(float(match.group(2)))
                train_acc.append(float(match.group(3)))
                val_loss.append(float(match.group(4)))
                val_acc.append(float(match.group(5)))
                val_macro_f1.append(float(match.group(6)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss curves
    axes[0].plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, val_acc, 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy', fontsize=11)
    axes[1].set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Macro F1 curve
    axes[2].plot(epochs, val_macro_f1, 'g-', label='Val Macro F1', linewidth=2)
    axes[2].axhline(y=max(val_macro_f1), color='r', linestyle='--', alpha=0.7,
                    label=f'Best: {max(val_macro_f1):.4f}')
    axes[2].set_xlabel('Epoch', fontsize=11)
    axes[2].set_ylabel('Macro F1', fontsize=11)
    axes[2].set_title('Validation Macro F1 Score', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('MLP Stage 1 Training Dynamics (50 Epochs)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "mlp_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'mlp_training_curves.png'}")


def plot_codet5_training_curves():
    """Plot CodeT5p-770m training loss curves for hyperparameter search."""
    print("Generating CodeT5p-770m training curves...")

    # Parse hyperparameter search log
    log_path = "outputs/hyperparam_search_770m.log"

    # Config data (4 configurations, 12 epochs each)
    configs = {
        "Config 1\n(lr=1e-4, drop=0.3)": {"train": [], "val": []},
        "Config 2\n(lr=1e-4, drop=0.5)": {"train": [], "val": []},
        "Config 3\n(lr=2e-4, drop=0.3)": {"train": [], "val": []},
        "Config 4\n(lr=2e-4, drop=0.5)": {"train": [], "val": []},
    }

    pattern = r"Epoch (\d+)/12 - Train Loss: ([\d.]+), Val Loss: ([\d.]+)"

    config_idx = 0
    config_names = list(configs.keys())

    with open(log_path) as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                val_loss = float(match.group(3))

                config_name = config_names[config_idx]
                configs[config_name]["train"].append(train_loss)
                configs[config_name]["val"].append(val_loss)

                if epoch == 12:
                    config_idx += 1
                    if config_idx >= 4:
                        break

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = list(range(1, 13))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Training loss
    for i, (name, data) in enumerate(configs.items()):
        if data["train"]:
            axes[0].plot(epochs, data["train"], color=colors[i], label=name, linewidth=2, marker='o', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Training Loss', fontsize=11)
    axes[0].set_title('Training Loss per Epoch', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Validation loss
    for i, (name, data) in enumerate(configs.items()):
        if data["val"]:
            axes[1].plot(epochs, data["val"], color=colors[i], label=name, linewidth=2, marker='o', markersize=4)
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Validation Loss', fontsize=11)
    axes[1].set_title('Validation Loss per Epoch', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('CodeT5p-770m Stage 2 Training: Hyperparameter Search (4 Configurations)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "codet5_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'codet5_training_curves.png'}")


def plot_hyperparameter_comparison():
    """Plot hyperparameter search results comparison."""
    print("Generating hyperparameter comparison chart...")

    with open("outputs/hyperparam_search_770m_results.json") as f:
        data = json.load(f)

    results = data["results"]

    configs = [r["config_name"].replace("_", "\n") for r in results]
    accuracy = [r["validation_metrics"]["accuracy"] * 100 for r in results]
    macro_f1 = [r["validation_metrics"]["macro_f1"] * 100 for r in results]

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, accuracy, width, label='Accuracy (%)', color='#2ecc71', edgecolor='black')
    bars2 = ax.bar(x + width/2, macro_f1, width, label='Macro F1 (%)', color='#3498db', edgecolor='black')

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('CodeT5p-770m Hyperparameter Search: Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(35, 50)
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Highlight best
    best_idx = accuracy.index(max(accuracy))
    ax.axvline(x=best_idx, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax.annotate('BEST', xy=(best_idx, 49), fontsize=11, color='red', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "hyperparameter_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'hyperparameter_comparison.png'}")


def plot_stage1_model_comparison():
    """Plot Stage 1 model comparison bar chart."""
    print("Generating Stage 1 model comparison chart...")

    models = ['XGBoost', 'MLP', 'SVM Linear']
    accuracy = [96.60, 89.18, 75.13]
    macro_f1 = [92.59, 77.49, 62.11]
    training_time = [147, 18.5, 309]  # seconds

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#27ae60', '#f39c12', '#e74c3c']

    # Accuracy
    bars = axes[0].bar(models, accuracy, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_ylim(70, 100)
    for bar, val in zip(bars, accuracy):
        axes[0].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    # Macro F1
    bars = axes[1].bar(models, macro_f1, color=colors, edgecolor='black')
    axes[1].set_ylabel('Macro F1 (%)', fontsize=11)
    axes[1].set_title('Validation Macro F1', fontsize=12, fontweight='bold')
    axes[1].set_ylim(55, 100)
    for bar, val in zip(bars, macro_f1):
        axes[1].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    # Training time
    bars = axes[2].bar(models, training_time, color=colors, edgecolor='black')
    axes[2].set_ylabel('Training Time (seconds)', fontsize=11)
    axes[2].set_title('Training Time', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, training_time):
        axes[2].annotate(f'{val:.0f}s' if val > 1 else f'{val:.1f}s',
                         xy=(bar.get_x() + bar.get_width()/2, val),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Stage 1 Binary Classifier Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage1_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'stage1_model_comparison.png'}")


def plot_stage2_model_comparison():
    """Plot Stage 2 model comparison bar chart."""
    print("Generating Stage 2 model comparison chart...")

    models = ['CodeT5p-770m\nConfig3', 'CodeT5p-770m\nHybrid', 'CodeT5p-770m\nBase',
              'CodeT5-Large', 'CodeT5p-220m']
    accuracy = [47.12, 46.95, 45.60, 44.91, 45.6]
    macro_f1 = [42.79, 42.84, 40.81, 40.50, 41.0]
    training_time = [103, 103, 103, 46, 16]  # minutes

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = ['#27ae60', '#2ecc71', '#3498db', '#9b59b6', '#f39c12']

    # Accuracy
    bars = axes[0].bar(models, accuracy, color=colors, edgecolor='black')
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    axes[0].set_ylim(40, 50)
    axes[0].tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, accuracy):
        axes[0].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    # Macro F1
    bars = axes[1].bar(models, macro_f1, color=colors, edgecolor='black')
    axes[1].set_ylabel('Macro F1 (%)', fontsize=11)
    axes[1].set_title('Validation Macro F1', fontsize=12, fontweight='bold')
    axes[1].set_ylim(38, 45)
    axes[1].tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, macro_f1):
        axes[1].annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    # Training time
    bars = axes[2].bar(models, training_time, color=colors, edgecolor='black')
    axes[2].set_ylabel('Training Time (minutes)', fontsize=11)
    axes[2].set_title('Training Time', fontsize=12, fontweight='bold')
    axes[2].tick_params(axis='x', labelsize=9)
    for bar, val in zip(bars, training_time):
        axes[2].annotate(f'{val}m', xy=(bar.get_x() + bar.get_width()/2, val),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9, fontweight='bold')

    plt.suptitle('Stage 2 Multi-class Classifier Comparison (AI Family Detection)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "stage2_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'stage2_model_comparison.png'}")


def plot_class_distribution():
    """Plot class distribution bar chart."""
    print("Generating class distribution chart...")

    labels = ["Human", "AI-1", "AI-2", "AI-3", "AI-4", "AI-5", "AI-6", "AI-7", "AI-8", "AI-9", "AI-10"]
    train_counts = [442096, 4162, 8993, 3029, 2227, 1968, 5783, 8197, 8127, 4608, 10810]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full distribution
    colors = ['#2ecc71'] + ['#3498db'] * 10
    bars = axes[0].bar(labels, train_counts, color=colors, edgecolor='black')
    axes[0].set_ylabel('Sample Count', fontsize=11)
    axes[0].set_xlabel('Class', fontsize=11)
    axes[0].set_title('Training Set Class Distribution (500K samples)', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_yscale('log')

    # AI-only distribution
    ai_labels = labels[1:]
    ai_counts = train_counts[1:]
    colors_ai = plt.cm.Blues(np.linspace(0.3, 0.9, 10))
    bars = axes[1].bar(ai_labels, ai_counts, color=colors_ai, edgecolor='black')
    axes[1].set_ylabel('Sample Count', fontsize=11)
    axes[1].set_xlabel('AI Family', fontsize=11)
    axes[1].set_title('AI Family Distribution (57,904 samples)', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)

    for bar, val in zip(bars, ai_counts):
        pct = val / sum(ai_counts) * 100
        axes[1].annotate(f'{pct:.1f}%', xy=(bar.get_x() + bar.get_width()/2, val),
                         xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

    plt.suptitle('Dataset Class Distribution: Severe Imbalance (88% Human vs 12% AI)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "class_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'class_distribution.png'}")


def plot_per_class_f1():
    """Plot per-class F1 scores for Stage 2."""
    print("Generating per-class F1 chart...")

    # F1 scores from Stage 2 analysis
    ai_families = ["AI-10", "AI-8", "AI-6", "AI-2", "AI-7", "AI-9", "AI-1", "AI-3", "AI-5", "AI-4"]
    f1_scores = [0.65, 0.58, 0.50, 0.48, 0.47, 0.42, 0.35, 0.32, 0.28, 0.24]
    support = [2162, 1625, 1157, 1799, 1639, 922, 832, 606, 394, 446]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color by performance
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(f1_scores)))[::-1]

    bars = ax.barh(ai_families, f1_scores, color=colors, edgecolor='black')

    # Add support annotations
    for i, (bar, sup) in enumerate(zip(bars, support)):
        ax.annotate(f'n={sup}', xy=(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2),
                    va='center', fontsize=9, color='gray')

    ax.set_xlabel('F1 Score', fontsize=12)
    ax.set_ylabel('AI Family', fontsize=12)
    ax.set_title('Stage 2 Per-Class F1 Scores (CodeT5p-770m Config3)\nOrdered by Performance',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 0.75)
    ax.axvline(x=0.4279, color='red', linestyle='--', alpha=0.7, label='Macro F1 Average: 0.4279')
    ax.legend(loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "per_class_f1.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {FIGURES_DIR / 'per_class_f1.png'}")


def main():
    """Generate all figures."""
    print("\n" + "="*60)
    print("GENERATING REPORT FIGURES")
    print("="*60 + "\n")

    # Generate all plots
    plot_confusion_matrix_full_pipeline()
    plot_confusion_matrix_stage1()
    plot_mlp_training_curves()
    plot_codet5_training_curves()
    plot_hyperparameter_comparison()
    plot_stage1_model_comparison()
    plot_stage2_model_comparison()
    plot_class_distribution()
    plot_per_class_f1()

    print("\n" + "="*60)
    print(f"All figures saved to: {FIGURES_DIR.absolute()}")
    print("="*60 + "\n")

    # List all generated files
    print("Generated files:")
    for f in sorted(FIGURES_DIR.glob("*.png")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
