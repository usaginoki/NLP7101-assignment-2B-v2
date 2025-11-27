"""
Training script for Stage 2: Multi-Class Classification (AI Families)
"""

import os
import sys
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.features.feature_loader import FeatureLoader
from src.features.feature_processor import FeatureProcessor
from src.models.stage2_classifier import create_stage2_classifier


def train_stage2(data_dir='data/reports', models_dir='models'):
    """
    Train Stage 2 multi-class classifier (AI model families)

    Args:
        data_dir: Directory containing feature CSVs
        models_dir: Directory to save trained models

    Returns:
        Trained classifier
    """
    print("\n" + "="*60)
    print("STAGE 2 TRAINING: Multi-Class Classification (AI Families)")
    print("="*60 + "\n")

    # Load features
    print("Step 1/4: Loading features...")
    loader = FeatureLoader(data_dir)
    train_df = loader.load_features('train')
    val_df = loader.load_features('validation')

    # Load existing scaler from Stage 1
    print("\nStep 2/4: Loading feature processor from Stage 1...")
    processor = FeatureProcessor()
    processor.load(f'{models_dir}/stage1/scaler.pkl')

    X_train, y_train = processor.transform(train_df)
    X_val, y_val = processor.transform(val_df)

    # Train classifier on AI samples only
    print("\nStep 3/4: Training multi-class classifier...")
    clf = create_stage2_classifier('random_forest')
    clf.fit(X_train, y_train)  # fit() internally filters for y > 0

    # Evaluate on AI validation samples only
    print("\nStep 4/4: Evaluating on validation set...")
    ai_mask = y_val > 0
    X_val_ai = X_val[ai_mask]
    y_val_ai = y_val[ai_mask]

    if len(y_val_ai) == 0:
        raise ValueError("No AI samples found in validation set")

    y_pred = clf.predict(X_val_ai)

    # Print metrics
    print("\n" + "="*60)
    print("Stage 2 Validation Results (AI Families Only)")
    print("="*60)
    print(classification_report(y_val_ai, y_pred,
                               labels=list(range(1, 11)),
                               target_names=[f'AI-{i}' for i in range(1, 11)],
                               digits=4,
                               zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_val_ai, y_pred, labels=list(range(1, 11)))
    print("Confusion Matrix (10x10):")
    print("(Rows: Actual, Columns: Predicted)")
    print("AI-1  AI-2  AI-3  AI-4  AI-5  AI-6  AI-7  AI-8  AI-9  AI-10")
    for i, row in enumerate(cm, 1):
        row_str = "  ".join([f"{val:4d}" for val in row])
        print(f"AI-{i:2d}: {row_str}")

    # Summary metrics
    macro_f1 = f1_score(y_val_ai, y_pred, average='macro',
                       labels=list(range(1, 11)), zero_division=0)
    weighted_f1 = f1_score(y_val_ai, y_pred, average='weighted',
                          labels=list(range(1, 11)), zero_division=0)
    accuracy = accuracy_score(y_val_ai, y_pred)

    print(f"\nMacro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Accuracy:    {accuracy:.4f}")
    print("="*60 + "\n")

    # Feature importance
    print("Top 10 Most Important Features:")
    importances = clf.get_feature_importances()
    top_features = processor.get_feature_importance_names(importances, top_k=10)
    for i, (name, importance) in enumerate(top_features, 1):
        print(f"  {i:2d}. {name:30s} {importance:.4f}")
    print()

    # Save model
    print("Saving model...")
    os.makedirs(f'{models_dir}/stage2', exist_ok=True)
    clf.save(f'{models_dir}/stage2/classifier.pkl')

    print("\n✓ Stage 2 training complete!")
    print(f"✓ Model saved to: {models_dir}/stage2/\n")

    return clf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Stage 2 Multi-Class Classifier')
    parser.add_argument('--data-dir', default='data/reports',
                       help='Directory containing feature CSVs')
    parser.add_argument('--models-dir', default='models',
                       help='Directory to save trained models')

    args = parser.parse_args()

    # Check if Stage 1 models exist
    stage1_scaler = f'{args.models_dir}/stage1/scaler.pkl'
    if not os.path.exists(stage1_scaler):
        print(f"ERROR: Stage 1 scaler not found at {stage1_scaler}")
        print("Please train Stage 1 first by running:")
        print("  python src/training/train_stage1.py")
        sys.exit(1)

    train_stage2(args.data_dir, args.models_dir)
