"""
Training script for Stage 2: Multi-Class Classification using CodeT5-220m

This script trains a CodeT5-based classifier for AI model family detection.
Unlike the Random Forest version, this uses raw code strings as input.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.stage2_classifier import create_stage2_classifier


def load_data_with_code(split='train'):
    """
    Load data including code strings

    Args:
        split: 'train' or 'validation'

    Returns:
        X (dummy features), y (labels), code_strings (list of code)
    """
    print(f"\nLoading {split} data with code strings...")

    # Load parquet file
    data_path = f'data/{split}.parquet'
    df = pd.read_parquet(data_path)

    print(f"  ✓ Loaded {len(df):,} total samples")

    # Extract components
    y = df['label'].values
    code_strings = df['code'].tolist()

    # Create dummy feature matrix (CodeT5 doesn't use engineered features)
    # But we need X for sklearn compatibility
    X = np.zeros((len(df), 36))  # 36 features to match stage 1

    # Print distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Label distribution:")
    for label, count in zip(unique, counts):
        label_name = "Human" if label == 0 else f"AI-{label}"
        print(f"    {label_name}: {count:,} ({100*count/len(y):.2f}%)")

    return X, y, code_strings


def train_stage2_codet5(
    batch_size=32,
    max_epochs=5,
    learning_rate=2e-4,
    early_stopping=True,
    validation_fraction=0.1,
    patience=3,
    models_dir='models',
    device='cuda'
):
    """
    Train Stage 2 CodeT5 classifier

    Args:
        batch_size: Training batch size
        max_epochs: Maximum training epochs
        learning_rate: Learning rate for AdamW
        early_stopping: Use early stopping
        validation_fraction: Fraction for validation split
        patience: Early stopping patience
        models_dir: Directory to save trained models
        device: 'cuda' or 'cpu'

    Returns:
        Trained classifier
    """
    print("\n" + "="*70)
    print("STAGE 2 TRAINING: CodeT5-220m Multi-Class Classification")
    print("="*70 + "\n")

    # Check GPU
    if device == 'cuda' and not torch.cuda.is_available():
        print("⚠ WARNING: CUDA not available, using CPU (will be slow)")
        device = 'cpu'

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")

    # Load training data with code strings
    print("="*70)
    print("STEP 1/5: Loading Training Data")
    print("="*70)
    X_train, y_train, code_train = load_data_with_code('train')

    # Load validation data with code strings
    print("\n" + "="*70)
    print("STEP 2/5: Loading Validation Data")
    print("="*70)
    X_val, y_val, code_val = load_data_with_code('validation')

    # Create classifier
    print("\n" + "="*70)
    print("STEP 3/5: Initializing CodeT5 Classifier")
    print("="*70)

    clf = create_stage2_classifier(
        'codet5',
        model_name='Salesforce/codet5p-220m',
        batch_size=batch_size,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        patience=patience,
        device=device,
        verbose=1
    )

    print(f"\nClassifier configuration:")
    print(f"  Model: Salesforce/codet5p-220m")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping: {early_stopping}")
    print(f"  Validation fraction: {validation_fraction}")
    print(f"  Patience: {patience}")
    print(f"  Device: {device}")

    # Train classifier
    print("\n" + "="*70)
    print("STEP 4/5: Training Classifier")
    print("="*70)

    start_time = datetime.now()
    print(f"\nTraining started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # fit() will internally filter for AI samples (y > 0)
    clf.fit(X_train, y_train, code_strings=code_train)

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()
    print(f"\nTraining completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total training time: {training_duration:.1f} seconds ({training_duration/60:.1f} minutes)")

    # Evaluate on validation set
    print("\n" + "="*70)
    print("STEP 5/5: Evaluating on Validation Set")
    print("="*70)

    # Filter for AI samples in validation
    ai_mask = y_val > 0
    X_val_ai = X_val[ai_mask]
    y_val_ai = y_val[ai_mask]
    code_val_ai = [code_val[i] for i in range(len(y_val)) if ai_mask[i]]

    if len(y_val_ai) == 0:
        raise ValueError("No AI samples found in validation set")

    print(f"\nValidation AI samples: {len(y_val_ai):,}")

    # Predict
    print("Running predictions...")
    y_pred = clf.predict(X_val_ai, code_strings=code_val_ai)

    # Print metrics
    print("\n" + "="*70)
    print("Stage 2 Validation Results (AI Families Only)")
    print("="*70)
    print(classification_report(
        y_val_ai, y_pred,
        labels=list(range(1, 11)),
        target_names=[f'AI-{i}' for i in range(1, 11)],
        digits=4,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_val_ai, y_pred, labels=list(range(1, 11)))
    print("\nConfusion Matrix (10x10):")
    print("(Rows: Actual, Columns: Predicted)")
    print("      " + "  ".join([f"AI-{i:2d}" for i in range(1, 11)]))
    for i, row in enumerate(cm, 1):
        row_str = "  ".join([f"{val:4d}" for val in row])
        print(f"AI-{i:2d}: {row_str}")

    # Summary metrics
    macro_f1 = f1_score(y_val_ai, y_pred, average='macro',
                       labels=list(range(1, 11)), zero_division=0)
    weighted_f1 = f1_score(y_val_ai, y_pred, average='weighted',
                          labels=list(range(1, 11)), zero_division=0)
    accuracy = accuracy_score(y_val_ai, y_pred)

    print(f"\n{'='*70}")
    print("Summary Metrics")
    print(f"{'='*70}")
    print(f"Macro F1 Score:    {macro_f1:.4f}")
    print(f"Weighted F1 Score: {weighted_f1:.4f}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"{'='*70}\n")

    # Save metrics
    metrics = {
        'model': 'Stage2CodeT5',
        'model_name': 'Salesforce/codet5p-220m',
        'training_samples': int(np.sum(y_train > 0)),
        'validation_samples': int(len(y_val_ai)),
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'learning_rate': learning_rate,
        'training_time_seconds': training_duration,
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().isoformat()
    }

    metrics_dir = Path('outputs/metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / 'stage2_codet5_validation.json'

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_path}")

    # Save model
    print("\n" + "="*70)
    print("Saving Model")
    print("="*70)

    models_path = Path(models_dir) / 'stage2_codet5'
    models_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = models_path / 'classifier.pth'

    clf.save(str(checkpoint_path))

    checkpoint_size = checkpoint_path.stat().st_size / (1024 * 1024)
    print(f"✓ Model saved to: {checkpoint_path}")
    print(f"  Checkpoint size: {checkpoint_size:.2f} MB")

    # Save training config
    config = {
        'model_type': 'Stage2CodeT5',
        'model_name': 'Salesforce/codet5p-220m',
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'learning_rate': learning_rate,
        'early_stopping': early_stopping,
        'validation_fraction': validation_fraction,
        'patience': patience,
        'device': device,
        'trained_at': datetime.now().isoformat(),
        'training_time_seconds': training_duration
    }

    config_path = models_path / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"✓ Config saved to: {config_path}")

    print("\n" + "="*70)
    print("✓ STAGE 2 CODET5 TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel location: {models_path}/")
    print(f"  - classifier.pth ({checkpoint_size:.2f} MB)")
    print(f"  - config.json")
    print(f"\nMetrics: {metrics_path}")
    print(f"  - Macro F1: {macro_f1:.4f}")
    print(f"  - Accuracy: {accuracy:.4f}")
    print("="*70 + "\n")

    return clf


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Stage 2 CodeT5 Classifier')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32)')
    parser.add_argument('--max-epochs', type=int, default=5,
                       help='Maximum training epochs (default: 5)')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Disable early stopping')
    parser.add_argument('--validation-fraction', type=float, default=0.1,
                       help='Validation split fraction (default: 0.1)')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience (default: 3)')
    parser.add_argument('--models-dir', default='models',
                       help='Directory to save trained models')
    parser.add_argument('--device', default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device for training (default: cuda)')

    args = parser.parse_args()

    train_stage2_codet5(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        early_stopping=not args.no_early_stopping,
        validation_fraction=args.validation_fraction,
        patience=args.patience,
        models_dir=args.models_dir,
        device=args.device
    )
