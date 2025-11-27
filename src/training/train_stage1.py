"""
Training script for Stage 1: Binary Classification
"""

import os
import sys
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.features.feature_loader import FeatureLoader
from src.features.feature_processor import FeatureProcessor
from src.models.stage1_classifier import Stage1Classifier


def train_stage1(data_dir='data/reports', models_dir='models'):
    """
    Train Stage 1 binary classifier (Human vs AI)

    Args:
        data_dir: Directory containing feature CSVs
        models_dir: Directory to save trained models

    Returns:
        Tuple of (classifier, processor)
    """
    print("\n" + "="*60)
    print("STAGE 1 TRAINING: Binary Classification (Human vs AI)")
    print("="*60 + "\n")

    # Load features
    print("Step 1/4: Loading features...")
    loader = FeatureLoader(data_dir)
    train_df = loader.load_features('train')
    val_df = loader.load_features('validation')

    # Process features
    print("\nStep 2/4: Processing features...")
    processor = FeatureProcessor()
    X_train, y_train = processor.fit_transform(train_df)
    X_val, y_val = processor.transform(val_df)

    # Train classifier
    print("\nStep 3/4: Training binary classifier...")
    clf = Stage1Classifier()
    clf.fit(X_train, y_train)

    # Evaluate on validation set
    print("\nStep 4/4: Evaluating on validation set...")
    y_val_binary = Stage1Classifier.prepare_labels(y_val)
    y_pred = clf.predict(X_val)

    # Print metrics
    print("\n" + "="*60)
    print("Stage 1 Validation Results")
    print("="*60)
    print(classification_report(y_val_binary, y_pred,
                               target_names=['Human', 'AI'],
                               digits=4))

    # Confusion matrix
    cm = confusion_matrix(y_val_binary, y_pred)
    print("Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                Human    AI")
    print(f"Actual Human    {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"       AI       {cm[1][0]:6d}  {cm[1][1]:6d}")

    # Summary metrics
    macro_f1 = f1_score(y_val_binary, y_pred, average='macro')
    accuracy = accuracy_score(y_val_binary, y_pred)

    print(f"\nMacro F1:  {macro_f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print("="*60 + "\n")

    # Feature importance
    print("Top 10 Most Important Features:")
    importances = clf.get_feature_importances()
    top_features = processor.get_feature_importance_names(importances, top_k=10)
    for i, (name, importance) in enumerate(top_features, 1):
        print(f"  {i:2d}. {name:30s} {importance:.4f}")
    print()

    # Save models
    print("Saving models...")
    os.makedirs(f'{models_dir}/stage1', exist_ok=True)
    clf.save(f'{models_dir}/stage1/classifier.pkl')
    processor.save(f'{models_dir}/stage1/scaler.pkl')

    print("\n✓ Stage 1 training complete!")
    print(f"✓ Models saved to: {models_dir}/stage1/\n")

    return clf, processor


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Stage 1 Binary Classifier')
    parser.add_argument('--data-dir', default='data/reports',
                       help='Directory containing feature CSVs')
    parser.add_argument('--models-dir', default='models',
                       help='Directory to save trained models')

    args = parser.parse_args()

    train_stage1(args.data_dir, args.models_dir)
