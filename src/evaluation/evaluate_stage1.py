"""
Evaluation script for Stage 1: Binary Classification
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
from src.models.stage1_classifier import Stage1Classifier


def evaluate_stage1(split='validation', data_dir='data/reports', models_dir='models', output_dir='outputs/metrics'):
    """
    Evaluate Stage 1 binary classifier

    Args:
        split: Dataset split to evaluate ('train' or 'validation')
        data_dir: Directory containing feature CSVs
        models_dir: Directory containing trained models
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*60)
    print(f"STAGE 1 EVALUATION: {split.upper()} SET")
    print("="*60 + "\n")

    # Load model
    print("Step 1/3: Loading model and processor...")
    clf = Stage1Classifier()
    clf.load(f'{models_dir}/stage1/classifier.pkl')

    processor = FeatureProcessor()
    processor.load(f'{models_dir}/stage1/scaler.pkl')

    # Load data
    print(f"\nStep 2/3: Loading {split} features...")
    loader = FeatureLoader(data_dir)
    df = loader.load_features(split)
    X, y = processor.transform(df)

    # Predict
    print("\nStep 3/3: Evaluating...")
    y_binary = Stage1Classifier.prepare_labels(y)
    y_pred = clf.predict(X)

    # Print metrics
    print("\n" + "="*60)
    print(f"Stage 1 Evaluation Results ({split})")
    print("="*60)
    print(classification_report(y_binary, y_pred,
                               target_names=['Human', 'AI'],
                               digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_binary, y_pred)
    print("Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                Human    AI")
    print(f"Actual Human    {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       AI       {cm[1][0]:6d}  {cm[1][1]:6d}")

    # Summary metrics
    macro_f1 = f1_score(y_binary, y_pred, average='macro')
    weighted_f1 = f1_score(y_binary, y_pred, average='weighted')
    accuracy = accuracy_score(y_binary, y_pred)

    print(f"\nMacro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print("="*60 + "\n")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results = {
        'split': split,
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist()
    }

    output_file = f'{output_dir}/stage1_{split}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"✓ Results saved to: {output_file}\n")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Stage 1 Binary Classifier')
    parser.add_argument('--split', default='validation', choices=['train', 'validation'],
                       help='Dataset split to evaluate')
    parser.add_argument('--data-dir', default='data/reports',
                       help='Directory containing feature CSVs')
    parser.add_argument('--models-dir', default='models',
                       help='Directory containing trained models')
    parser.add_argument('--output-dir', default='outputs/metrics',
                       help='Directory to save evaluation results')

    args = parser.parse_args()

    evaluate_stage1(args.split, args.data_dir, args.models_dir, args.output_dir)
