"""
Report Generator

Generate comparison reports and visualizations for model search results.
Creates matplotlib/seaborn charts and markdown reports.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List


class ReportGenerator:
    """Generate comparison reports and visualizations"""

    def __init__(self, output_dir: str = 'outputs/model_search'):
        """
        Initialize report generator

        Args:
            output_dir: Directory to save visualizations and reports
        """
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib style
        sns.set_style('whitegrid')
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 10

    def generate_macro_f1_comparison(self, comparison: Dict[str, Any]):
        """
        Bar chart comparing macro F1 scores

        Args:
            comparison: Comparison dictionary from ModelEvaluator.compare_models()
        """
        top_models = comparison['top_5_models']

        if not top_models:
            print("No models to visualize")
            return

        names = [m['model_name'] for m in top_models]
        scores = [m['val_macro_f1'] for m in top_models]

        # Create figure
        plt.figure(figsize=(10, 6))
        bars = plt.barh(names, scores, color='skyblue', edgecolor='navy', alpha=0.7)

        plt.xlabel('Macro F1 Score', fontsize=12)
        plt.ylabel('Classifier', fontsize=12)
        plt.title('Top 5 Models: Validation Macro F1 Comparison', fontsize=14, fontweight='bold')
        plt.xlim([0, 1])
        plt.grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.4f}', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filepath = self.viz_dir / 'macro_f1_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved macro F1 comparison to: {filepath}")

    def generate_confusion_matrices(self, top_models: List[Dict[str, Any]]):
        """
        Grid of confusion matrices for top models

        Args:
            top_models: List of top model results (from comparison)
        """
        if not top_models:
            print("No models to visualize")
            return

        n_models = min(4, len(top_models))  # Show top 4

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for idx, model_result in enumerate(top_models[:n_models]):
            cm = np.array(model_result['confusion_matrix'])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       ax=axes[idx], cbar=False,
                       xticklabels=['Human', 'AI'],
                       yticklabels=['Human', 'AI'],
                       annot_kws={'size': 12, 'weight': 'bold'})

            axes[idx].set_title(f"{model_result['model_name']}\n"
                               f"Macro F1: {model_result['val_macro_f1']:.4f}",
                               fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)

        # Hide unused subplots
        for idx in range(n_models, 4):
            axes[idx].axis('off')

        plt.suptitle('Confusion Matrices: Top 4 Models', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        filepath = self.viz_dir / 'confusion_matrices.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved confusion matrices to: {filepath}")

    def generate_training_time_comparison(self, experiments: List[Dict[str, Any]]):
        """
        Bar chart of training times

        Args:
            experiments: List of experiment dictionaries
        """
        successful = [e for e in experiments if e['status'] == 'SUCCESS']

        if not successful:
            print("No successful experiments to visualize")
            return

        names = [e['classifier'] for e in successful]
        times = [e['elapsed_time_seconds'] / 60 for e in successful]  # Convert to minutes

        # Sort by time
        sorted_indices = np.argsort(times)
        names = [names[i] for i in sorted_indices]
        times = [times[i] for i in sorted_indices]

        plt.figure(figsize=(10, 6))
        bars = plt.barh(names, times, color='coral', edgecolor='darkred', alpha=0.7)
        plt.xlabel('Training Time (minutes)', fontsize=12)
        plt.ylabel('Classifier', fontsize=12)
        plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)

        # Add time labels
        for bar, time_val in zip(bars, times):
            plt.text(time_val + max(times)*0.02, bar.get_y() + bar.get_height()/2,
                    f'{time_val:.1f}m', va='center', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filepath = self.viz_dir / 'training_time_comparison.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved training time comparison to: {filepath}")

    def generate_markdown_report(self, comparison: Dict[str, Any],
                                 experiments: List[Dict[str, Any]]):
        """
        Generate markdown summary report

        Args:
            comparison: Comparison dictionary from ModelEvaluator.compare_models()
            experiments: List of experiment dictionaries
        """
        report_lines = [
            "# Stage 1 Model Search Results",
            "",
            f"**Search Date:** {experiments[0]['timestamp'].split('T')[0] if experiments else 'N/A'}",
            f"**Total Models Tested:** {comparison['total_models_tested']}",
            f"**Successful Runs:** {comparison['successful_runs']}",
            f"**Failed Runs:** {comparison['failed_runs']}",
            "",
            "---",
            "",
            "## Top 5 Models",
            "",
            "| Rank | Model | Macro F1 | Accuracy | Precision | Recall | Training Time |",
            "|------|-------|----------|----------|-----------|--------|---------------|"
        ]

        for rank, model in enumerate(comparison['top_5_models'], 1):
            # Find corresponding experiment
            exp = next((e for e in experiments if e['classifier'] == model['model_name']), None)
            time_str = f"{exp['elapsed_time_seconds']/60:.1f}m" if exp else "N/A"

            report_lines.append(
                f"| {rank} | {model['model_name']} | "
                f"{model['val_macro_f1']:.4f} | "
                f"{model['accuracy']:.4f} | "
                f"{model['precision_macro']:.4f} | "
                f"{model['recall_macro']:.4f} | "
                f"{time_str} |"
            )

        if comparison['best_model']:
            best = comparison['best_model']
            report_lines.extend([
                "",
                "---",
                "",
                "## Best Model Details",
                "",
                f"**Model:** {best['model_name']}",
                f"**Validation Macro F1:** {best['val_macro_f1']:.4f}",
                f"**Accuracy:** {best['accuracy']:.4f}",
                f"**Precision (macro):** {best['precision_macro']:.4f}",
                f"**Recall (macro):** {best['recall_macro']:.4f}",
                "",
                "### Best Hyperparameters",
                "```python",
                json.dumps(best['best_params'], indent=2),
                "```",
                "",
                "### Confusion Matrix",
                "```",
                f"              Predicted",
                f"             Human    AI",
                f"Actual Human  {best['confusion_matrix'][0][0]:5d}  {best['confusion_matrix'][0][1]:5d}",
                f"       AI     {best['confusion_matrix'][1][0]:5d}  {best['confusion_matrix'][1][1]:5d}",
                "```",
                "",
                "### Per-Class Metrics",
                "",
                "| Class | Precision | Recall | F1-Score | Support |",
                "|-------|-----------|--------|----------|---------|"
            ])

            per_class = best['per_class']
            for class_name in ['Human', 'AI']:
                if class_name in per_class:
                    metrics = per_class[class_name]
                    report_lines.append(
                        f"| {class_name} | {metrics['precision']:.4f} | "
                        f"{metrics['recall']:.4f} | {metrics['f1-score']:.4f} | "
                        f"{int(metrics['support'])} |"
                    )

        report_lines.extend([
            "",
            "---",
            "",
            "## Visualizations",
            "",
            "![Macro F1 Comparison](visualizations/macro_f1_comparison.png)",
            "",
            "![Confusion Matrices](visualizations/confusion_matrices.png)",
            "",
            "![Training Time](visualizations/training_time_comparison.png)",
            ""
        ])

        # Save markdown
        filepath = self.output_dir / 'model_comparison_report.md'
        with open(filepath, 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"✓ Saved markdown report to: {filepath}")

    def generate_all_reports(self, comparison: Dict[str, Any],
                            experiments: List[Dict[str, Any]]):
        """
        Generate all reports and visualizations

        Args:
            comparison: Comparison dictionary from ModelEvaluator.compare_models()
            experiments: List of experiment dictionaries
        """
        print("\nGenerating reports and visualizations...")

        self.generate_macro_f1_comparison(comparison)
        self.generate_confusion_matrices(comparison['top_5_models'])
        self.generate_training_time_comparison(experiments)
        self.generate_markdown_report(comparison, experiments)

        print("\n✓ All reports and visualizations generated successfully!")
