"""
Evaluation script for Stage 2: Multi-Class Classification
"""

import os
import sys
import json
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.features.feature_loader import FeatureLoader
from src.features.feature_processor import FeatureProcessor
from src.models.stage2_classifier import create_stage2_classifier


def evaluate_stage2(split='validation', data_dir='data/reports', models_dir='models', output_dir='outputs/metrics'):
    """
    Evaluate Stage 2 multi-class classifier on AI samples

    Args:
        split: Dataset split to evaluate ('train' or 'validation')
        data_dir: Directory containing feature CSVs
        models_dir: Directory containing trained models
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print(f"STAGE 2 EVALUATION: {split.upper()} SET (AI Families Only)")
    print("="*60 + "\n")

    # Load model
    print("Step 1/3: Loading model and processor...")
    clf = create_stage2_classifier('random_forest')
    clf.load(f'{models_dir}/stage2/classifier.pkl')

    processor = FeatureProcessor()
    processor.load(f'{models_dir}/stage1/scaler.pkl')

    # Load data
    print(f"\nStep 2/3: Loading {split} features...")
    loader = FeatureLoader(data_dir)
    df = loader.load_features(split)
    X, y = processor.transform(df)

    # Filter AI samples only
    print("\nFiltering AI samples only...")
    ai_mask = y > 0
    X_ai, y_ai = X[ai_mask], y[ai_mask]

    print(f"Total samples: {len(y)}")
    print(f"AI samples: {len(y_ai)}")

    if len(y_ai) == 0:
        print("ERROR: No AI samples found in dataset")
        return None

    # Predict
    print("\nStep 3/3: Evaluating...")
    y_pred = clf.predict(X_ai)

    # Print metrics
    print("\n" + "="*60)
    print(f"Stage 2 Evaluation Results ({split} - AI Only)")
    print("="*60)
    print(classification_report(y_ai, y_pred,
                               labels=list(range(1, 11)),
                               target_names=[f'AI-{i}' for i in range(1, 11)],
                               digits=4,
                               zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_ai, y_pred, labels=list(range(1, 11)))
    print("Confusion Matrix (10x10):")
    print("(Rows: Actual, Columns: Predicted)")
    print("       AI-1  AI-2  AI-3  AI-4  AI-5  AI-6  AI-7  AI-8  AI-9  AI-10")
    for i, row in enumerate(cm, 1):
        row_str = "  ".join([f"{val:4d}" for val in row])
        print(f"AI-{i:2d}:  {row_str}")

    # Summary metrics
    macro_f1 = f1_score(y_ai, y_pred, average='macro',
                       labels=list(range(1, 11)), zero_division=0)
    weighted_f1 = f1_score(y_ai, y_pred, average='weighted',
                          labels=list(range(1, 11)), zero_division=0)
    accuracy = accuracy_score(y_ai, y_pred)

    # Per-class F1
    per_class_f1 = f1_score(y_ai, y_pred, average=None,
                           labels=list(range(1, 11)), zero_division=0)

    print(f"\nMacro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print("="*60 + "\n")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'split': split,
        'n_samples': int(len(y_ai)),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(accuracy),
        'per_class_f1': [float(x) for x in per_class_f1],
        'confusion_matrix': cm.tolist()
    }

    output_file = f'{output_dir}/stage2_{split}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_file}\n")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Stage 2 Multi-Class Classifier')
    parser.add_argument('--split', default='validation', choices=['train', 'validation'],
                       help='Dataset split to evaluate')
    parser.add_argument('--data-dir', default='data/reports',
                       help='Directory containing feature CSVs')
    parser.add_argument('--models-dir', default='models',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', default='outputs/metrics',
                       help='Directory to save evaluation results')

    args = parser.parse_args()

    evaluate_stage2(args.split, args.data_dir, args.models_dir, args.output_dir)
