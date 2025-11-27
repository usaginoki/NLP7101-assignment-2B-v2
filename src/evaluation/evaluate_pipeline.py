"""
Evaluation script for complete 2-Stage Pipeline
"""

import os
import sys
import json
import numpy as np
from sklearn.metrics import classification_report

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.features.feature_loader import FeatureLoader
from src.models.pipeline import TwoStagePipeline


def evaluate_pipeline(split='validation', data_dir='data/reports', models_dir='models', output_dir='outputs/metrics'):
    """
    Evaluate complete 2-stage classification pipeline

    Args:
        split: Dataset split to evaluate ('train' or 'validation')
        data_dir: Directory containing feature CSVs
        models_dir: Directory containing trained models
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print(f"FULL PIPELINE EVALUATION: {split.upper()} SET")
    print("="*60 + "\n")

    # Load pipeline
    print("Step 1/3: Loading complete pipeline...")
    pipeline = TwoStagePipeline.load(models_dir)

    # Load data
    print(f"\nStep 2/3: Loading {split} features...")
    loader = FeatureLoader(data_dir)
    df = loader.load_features(split)
    X, y = pipeline.processor.transform(df)

    # Evaluate
    print("\nStep 3/3: Evaluating...")
    metrics = pipeline.evaluate(X, y)

    # Get predictions for detailed report
    y_pred = pipeline.predict(X)

    # Print results
    print("\n" + "="*60)
    print(f"Full Pipeline Evaluation Results ({split})")
    print("="*60)
    print(f"\nStage 1 (Binary: Human vs AI)")
    print(f"  Macro F1:  {metrics['stage1_f1']:.4f}")
    print(f"  Accuracy:  {metrics['stage1_accuracy']:.4f}")

    print(f"\nStage 2 (Multi-Class: AI Families)")
    print(f"  Macro F1:  {metrics['stage2_f1']:.4f}")
    print(f"  Accuracy:  {metrics['stage2_accuracy']:.4f}")

    print(f"\nOverall (11-Class)")
    print(f"  Macro F1:     {metrics['overall_macro_f1']:.4f}")
    print(f"  Weighted F1:  {metrics['overall_weighted_f1']:.4f}")
    print(f"  Accuracy:     {metrics['overall_accuracy']:.4f}")

    print("\n" + "="*60)
    print("Detailed Classification Report (All 11 Classes)")
    print("="*60)
    print(classification_report(y, y_pred,
                               labels=list(range(11)),
                               target_names=['Human'] + [f'AI-{i}' for i in range(1, 11)],
                               digits=4,
                               zero_division=0))
    print("="*60 + "\n")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'split': split,
        'stage1_f1': float(metrics['stage1_f1']),
        'stage1_accuracy': float(metrics['stage1_accuracy']),
        'stage2_f1': float(metrics['stage2_f1']),
        'stage2_accuracy': float(metrics['stage2_accuracy']),
        'overall_macro_f1': float(metrics['overall_macro_f1']),
        'overall_weighted_f1': float(metrics['overall_weighted_f1']),
        'overall_accuracy': float(metrics['overall_accuracy'])
    }

    output_file = f'{output_dir}/pipeline_{split}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_file}\n")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Complete 2-Stage Pipeline')
    parser.add_argument('--split', default='validation', choices=['train', 'validation'],
                       help='Dataset split to evaluate')
    parser.add_argument('--data-dir', default='data/reports',
                       help='Directory containing feature CSVs')
    parser.add_argument('--models-dir', default='models',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', default='outputs/metrics',
                       help='Directory to save evaluation results')

    args = parser.parse_args()

    evaluate_pipeline(args.split, args.data_dir, args.models_dir, args.output_dir)
