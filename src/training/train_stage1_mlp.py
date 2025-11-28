"""
Train Stage 1 MLP Classifier (GPU-Accelerated)

Fast GPU-based binary classifier for Human vs AI detection.
Alternative to XGBoost with faster training on large datasets.

Usage:
    uv run python src/training/train_stage1_mlp.py --batch-size 2048

    Options:
        --batch-size: Batch size (default: 2048 for GPU)
        --max-epochs: Max epochs (default: 50)
        --learning-rate: Learning rate (default: 1e-3)
        --dropout: Dropout rate (default: 0.3)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.stage1_mlp import Stage1MLPClassifier
from src.features.feature_loader import FeatureLoader
from src.features.feature_processor import FeatureProcessor
import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_recall_fscore_support, confusion_matrix
)


def train_stage1_mlp(
    batch_size=2048,
    max_epochs=50,
    learning_rate=1e-3,
    dropout=0.3,
    hidden_dims=[256, 128, 64],
    early_stopping=True,
    patience=5,
    class_weight_multiplier=1.0,
    optimize_metric='loss',
    optimize_threshold=False,
    models_dir='models',
    data_dir='data/reports',
    device='cuda'
):
    """
    Train Stage 1 MLP binary classifier

    Args:
        batch_size: Training batch size (default: 2048)
        max_epochs: Maximum epochs (default: 50)
        learning_rate: Learning rate (default: 1e-3)
        dropout: Dropout rate (default: 0.3)
        hidden_dims: Hidden layer dimensions (default: [256, 128, 64])
        early_stopping: Enable early stopping (default: True)
        patience: Early stopping patience (default: 5)
        class_weight_multiplier: Multiplier for AI class weight (default: 1.0)
        optimize_metric: Metric to optimize - 'loss' or 'macro_f1' (default: 'loss')
        optimize_threshold: Whether to optimize classification threshold (default: False)
        models_dir: Models directory (default: 'models')
        data_dir: Feature reports directory (default: 'data/reports')
        device: Device ('cuda' or 'cpu')

    Returns:
        clf: Trained classifier
        metadata: Training metadata dict
    """
    print("\n" + "="*80)
    print("TRAINING STAGE 1 MLP BINARY CLASSIFIER (GPU-ACCELERATED)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: MLP Neural Network")
    print(f"  Architecture: 36 → {' → '.join(map(str, hidden_dims))} → 2")
    print(f"  Task: Binary classification (Human vs AI)")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Dropout: {dropout}")
    print(f"  Early stopping: {early_stopping} (patience={patience})")
    print(f"  Class weight multiplier: {class_weight_multiplier}")
    print(f"  Optimize metric: {optimize_metric}")
    print(f"  Optimize threshold: {optimize_threshold}")
    print(f"  Device: {device}")
    print("="*80 + "\n")
    sys.stdout.flush()

    # ====================================================================
    # STEP 1: Load Training Data
    # ====================================================================
    print("STEP 1/5: Loading training data...")
    sys.stdout.flush()

    loader = FeatureLoader(data_dir=data_dir)
    train_df = loader.load_features(split='train')

    print(f"  ✓ Loaded features: {train_df.shape}")
    sys.stdout.flush()

    # ====================================================================
    # STEP 2: Process Features
    # ====================================================================
    print("\nSTEP 2/5: Processing features...")
    sys.stdout.flush()

    processor = FeatureProcessor()
    X_train, y_train = processor.fit_transform(train_df)

    print(f"  ✓ Processed features: {X_train.shape}")
    print(f"  ✓ Labels: {y_train.shape}")
    sys.stdout.flush()

    # Show class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n  Original class distribution:")
    for label, count in zip(unique, counts):
        name = 'Human' if label == 0 else f'AI-{label}' if label <= 10 else 'Other'
        pct = 100 * count / len(y_train)
        print(f"    {name:8s}: {count:6d} ({pct:5.1f}%)")
    sys.stdout.flush()

    # Binary conversion will happen inside the classifier
    y_binary = (y_train > 0).astype(int)
    print(f"\n  Binary class distribution:")
    print(f"    Human: {np.sum(y_binary == 0):6d} ({100*np.sum(y_binary == 0)/len(y_binary):.1f}%)")
    print(f"    AI:    {np.sum(y_binary == 1):6d} ({100*np.sum(y_binary == 1)/len(y_binary):.1f}%)")
    sys.stdout.flush()

    # ====================================================================
    # STEP 3: Initialize MLP Classifier
    # ====================================================================
    print(f"\nSTEP 3/5: Initializing MLP classifier...")
    sys.stdout.flush()

    clf = Stage1MLPClassifier(
        hidden_dims=hidden_dims,
        dropout=dropout,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        validation_fraction=0.1,
        patience=patience,
        weight_decay=1e-4,
        class_weight_multiplier=class_weight_multiplier,
        optimize_metric=optimize_metric,
        optimize_threshold=optimize_threshold,
        random_state=42,
        device=device,
        verbose=1
    )

    print(f"  ✓ Classifier initialized")
    sys.stdout.flush()

    # ====================================================================
    # STEP 4: Train Classifier
    # ====================================================================
    print(f"\nSTEP 4/5: Training MLP classifier...")
    print(f"  This should take ~5-10 minutes with GPU acceleration")
    print()
    sys.stdout.flush()

    start_time = time.time()
    clf.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"\n  ✓ Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    sys.stdout.flush()

    # ====================================================================
    # STEP 5: Evaluate on Validation Set
    # ====================================================================
    print(f"\nSTEP 5/5: Evaluating on validation set...")
    sys.stdout.flush()

    # Load validation data
    val_df = loader.load_features(split='validation')
    X_val, y_val = processor.transform(val_df)

    print(f"  Validation samples: {len(y_val)}")
    sys.stdout.flush()

    # Predict
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)

    # True binary labels
    y_val_binary = (y_val > 0).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_val_binary, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val_binary, y_pred, average=None, zero_division=0
    )
    macro_f1 = f1_score(y_val_binary, y_pred, average='macro')
    weighted_f1 = f1_score(y_val_binary, y_pred, average='weighted')

    print(f"\n  Validation Metrics:")
    print(f"    Accuracy:    {accuracy:.4f}")
    print(f"    Macro F1:    {macro_f1:.4f}")
    print(f"    Weighted F1: {weighted_f1:.4f}")
    sys.stdout.flush()

    # Per-class metrics
    print(f"\n  Per-class metrics:")
    class_names = ['Human', 'AI']
    for i, name in enumerate(class_names):
        print(f"    {name:8s}: precision={precision[i]:.4f}, recall={recall[i]:.4f}, "
              f"f1={f1[i]:.4f}, support={support[i]}")
    sys.stdout.flush()

    # Confusion matrix
    cm = confusion_matrix(y_val_binary, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Human    AI")
    print(f"    Human   {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"    AI      {cm[1,0]:6d} {cm[1,1]:6d}")
    sys.stdout.flush()

    # Detailed report
    print(f"\n  Classification Report:")
    print(classification_report(
        y_val_binary,
        y_pred,
        target_names=['Human', 'AI'],
        zero_division=0
    ))
    sys.stdout.flush()

    # ====================================================================
    # Save Model
    # ====================================================================
    print(f"\nSaving model...")
    sys.stdout.flush()

    save_dir = Path(models_dir) / 'stage1' / 'MLP'
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / 'classifier.pth'
    clf.save(str(model_path))

    # Save scaler (for Stage 2 compatibility)
    scaler_path = Path(models_dir) / 'stage1' / 'scaler.pkl'
    processor.save(str(scaler_path))
    print(f"  ✓ Scaler saved to: {scaler_path}")

    # Save metadata
    metadata = {
        'model_name': 'MLP',
        'model_type': 'Neural Network',
        'task': 'binary_classification',
        'classes': ['Human', 'AI'],
        'num_features': int(X_train.shape[1]),
        'training_samples': int(len(y_train)),
        'validation_samples': int(len(y_val)),
        'hyperparameters': {
            'hidden_dims': hidden_dims,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_epochs': max_epochs,
            'dropout': dropout,
            'early_stopping': early_stopping,
            'patience': patience,
            'class_weight_multiplier': class_weight_multiplier,
            'optimize_metric': optimize_metric,
            'optimize_threshold': optimize_threshold
        },
        'threshold': float(clf.threshold_),
        'validation_metrics': {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'precision_human': float(precision[0]),
            'recall_human': float(recall[0]),
            'f1_human': float(f1[0]),
            'precision_ai': float(precision[1]),
            'recall_ai': float(recall[1]),
            'f1_ai': float(f1[1])
        },
        'confusion_matrix': cm.tolist(),
        'training_time_seconds': float(training_time),
        'timestamp': datetime.now().isoformat(),
        'device': device
    }

    metadata_path = save_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Model saved to: {model_path}")
    print(f"  ✓ Metadata saved to: {metadata_path}")
    sys.stdout.flush()

    # ====================================================================
    # Final Summary
    # ====================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE: Stage 1 MLP Classifier")
    print("="*80)
    print(f"Model: {save_dir}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")
    print(f"Training time: {training_time/60:.1f} minutes")
    print("="*80 + "\n")
    sys.stdout.flush()

    return clf, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Stage 1 MLP binary classifier (GPU-accelerated)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (optimal for GPU)
  uv run python src/training/train_stage1_mlp.py

  # Optimize for macro F1 (better for imbalanced data)
  uv run python src/training/train_stage1_mlp.py \\
      --optimize-metric macro_f1 \\
      --optimize-threshold \\
      --class-weight-multiplier 2.0

  # Custom hyperparameters
  uv run python src/training/train_stage1_mlp.py \\
      --batch-size 4096 \\
      --max-epochs 30 \\
      --learning-rate 1e-3 \\
      --dropout 0.3

  # For CPU training (much slower)
  uv run python src/training/train_stage1_mlp.py --device cpu --batch-size 512
        """
    )

    parser.add_argument('--batch-size', type=int, default=2048,
                       help='Batch size (default: 2048 for GPU)')
    parser.add_argument('--max-epochs', type=int, default=50,
                       help='Maximum epochs (default: 50)')
    parser.add_argument('--learning-rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64],
                       help='Hidden layer dimensions (default: 256 128 64)')
    parser.add_argument('--class-weight-multiplier', type=float, default=1.0,
                       help='Multiplier for AI class weight (default: 1.0, try 2.0-3.0 for macro F1)')
    parser.add_argument('--optimize-metric', choices=['loss', 'macro_f1'], default='loss',
                       help='Metric to optimize for early stopping (default: loss)')
    parser.add_argument('--optimize-threshold', action='store_true',
                       help='Optimize classification threshold for macro F1')
    parser.add_argument('--models-dir', default='models',
                       help='Models directory (default: models)')
    parser.add_argument('--data-dir', default='data/reports',
                       help='Feature reports directory (default: data/reports)')
    parser.add_argument('--device', default='cuda',
                       help='Device: cuda or cpu (default: cuda)')

    args = parser.parse_args()

    # Train
    try:
        train_stage1_mlp(
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            hidden_dims=args.hidden_dims,
            class_weight_multiplier=args.class_weight_multiplier,
            optimize_metric=args.optimize_metric,
            optimize_threshold=args.optimize_threshold,
            models_dir=args.models_dir,
            data_dir=args.data_dir,
            device=args.device
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
