"""
Train Stage 2 HYBRID CodeT5p-770m Classifier

Combines CodeT5p-770m embeddings (1024-dim) with engineered features (36-dim)
for improved AI family classification.

Expected improvements over embeddings-only:
- Semantic understanding from CodeT5
- Statistical patterns from engineered features
- Better handling of edge cases

Usage:
    uv run python src/training/train_stage2_codet5p_770m_hybrid.py --batch-size 256

    Options:
        --batch-size: Batch size (default: 256)
        --max-epochs: Max epochs (default: 12)
        --learning-rate: Learning rate (default: 2e-4)
        --dropout: Dropout rate (default: 0.3)
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.models.stage2_codet5p_770m_hybrid import Stage2CodeT5p770mHybrid
from src.features.feature_loader import FeatureLoader
from src.features.feature_processor import FeatureProcessor
import joblib
import pandas as pd
import numpy as np


def train_stage2_codet5p_770m_hybrid(
    batch_size=256,
    max_epochs=12,
    learning_rate=2e-4,
    dropout=0.3,
    early_stopping=True,
    validation_fraction=0.1,
    patience=3,
    models_dir='models',
    data_dir='data/reports',
    device='cuda'
):
    """
    Train HYBRID Stage 2 classifier using CodeT5p-770m + features

    Args:
        batch_size: Training batch size (default: 256)
        max_epochs: Maximum training epochs (default: 12)
        learning_rate: Learning rate (default: 2e-4)
        dropout: Dropout rate (default: 0.3)
        early_stopping: Enable early stopping (default: True)
        validation_fraction: Validation split fraction (default: 0.1)
        patience: Early stopping patience (default: 3)
        models_dir: Directory for models (default: 'models')
        data_dir: Directory for feature reports (default: 'data/reports')
        device: Device ('cuda' or 'cpu')

    Returns:
        clf: Trained classifier
        metadata: Training metadata dict
    """
    print("\n" + "="*80)
    print("TRAINING HYBRID STAGE 2 CLASSIFIER: CodeT5p-770m + Features")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: Salesforce/codet5p-770m (HYBRID)")
    print(f"  Embeddings: 1024-dim (from CodeT5)")
    print(f"  Features: 36-dim (engineered)")
    print(f"  Combined: 1060-dim input to classification head")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Dropout: {dropout}")
    print(f"  Early stopping: {early_stopping} (patience={patience})")
    print(f"  Device: {device}")
    print("="*80 + "\n")
    sys.stdout.flush()

    # ====================================================================
    # STEP 1: Load Training Data
    # ====================================================================
    print("STEP 1/5: Loading training data...")
    sys.stdout.flush()

    # Load features
    loader = FeatureLoader(data_dir=data_dir)
    train_df = loader.load_features(split='train')
    print(f"  ✓ Loaded features: {train_df.shape}")
    print(f"  Features: {train_df.shape[1] - 4} (excluding metadata)")
    sys.stdout.flush()

    # Load raw code strings
    train_parquet = Path('data/train.parquet')
    if not train_parquet.exists():
        raise FileNotFoundError(f"Training parquet not found: {train_parquet}")

    code_df = pd.read_parquet(train_parquet)
    print(f"  ✓ Loaded code strings: {len(code_df)}")
    sys.stdout.flush()

    # Merge on 'code' column to align features with code strings
    print(f"  Merging features with code strings...")
    merged_df = train_df.merge(
        code_df[['code']],
        on='code',
        how='left'
    )

    # Verify merge
    n_missing = merged_df['code'].isna().sum()
    if n_missing > 0:
        print(f"  ⚠ WARNING: {n_missing} samples missing code strings")
        merged_df = merged_df.dropna(subset=['code'])

    print(f"  ✓ Merged dataset: {len(merged_df)} samples")
    sys.stdout.flush()

    # ====================================================================
    # STEP 2: Load Feature Processor (Scaler)
    # ====================================================================
    print("\nSTEP 2/5: Loading feature processor...")
    sys.stdout.flush()

    scaler_path = Path(models_dir) / 'stage1' / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Stage 1 scaler not found: {scaler_path}\n"
            "Train Stage 1 first: uv run python src/training/train_stage1.py"
        )

    processor = FeatureProcessor()
    scaler_obj = joblib.load(scaler_path)

    # Handle both old and new scaler formats
    if isinstance(scaler_obj, dict):
        processor.load(str(scaler_path))
        print(f"  ✓ Loaded scaler (new format)")
    else:
        processor.scaler = scaler_obj
        processor.is_fitted = True
        processor.feature_names = [col for col in merged_df.columns
                                   if col not in processor.metadata_cols]
        print(f"  ✓ Loaded scaler (old format): {type(scaler_obj).__name__}")

    print(f"  Feature names: {len(processor.feature_names)}")
    sys.stdout.flush()

    # Transform features
    X_train, y_train = processor.transform(merged_df)
    code_strings = merged_df['code'].tolist()

    print(f"  ✓ Transformed features: {X_train.shape}")
    print(f"  ✓ Labels: {y_train.shape}")
    print(f"  ✓ Code strings: {len(code_strings)}")
    sys.stdout.flush()

    # Show class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"\n  Class distribution:")
    for label, count in zip(unique, counts):
        name = 'Human' if label == 0 else f'AI-{label}'
        pct = 100 * count / len(y_train)
        print(f"    {name:8s}: {count:6d} ({pct:5.1f}%)")
    sys.stdout.flush()

    # ====================================================================
    # STEP 3: Initialize HYBRID Classifier
    # ====================================================================
    print(f"\nSTEP 3/5: Initializing HYBRID classifier...")
    sys.stdout.flush()

    clf = Stage2CodeT5p770mHybrid(
        model_name='Salesforce/codet5p-770m',
        freeze_encoder=True,
        max_length=512,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        patience=patience,
        dropout=dropout,
        weight_decay=0.01,
        random_state=42,
        device=device,
        verbose=1
    )

    print(f"  ✓ Classifier initialized")
    sys.stdout.flush()

    # ====================================================================
    # STEP 4: Train Classifier
    # ====================================================================
    print(f"\nSTEP 4/5: Training HYBRID classifier...")
    print(f"  This will take ~15-20 minutes for 12 epochs with batch_size={batch_size}")
    print(f"  Model will train on AI samples only (labels 1-10)")
    print()
    sys.stdout.flush()

    import time
    start_time = time.time()

    # Train (will internally filter for y > 0)
    clf.fit(X_train, y_train, code_strings=code_strings)

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
    val_parquet = pd.read_parquet('data/validation.parquet')

    val_merged = val_df.merge(val_parquet[['code']], on='code', how='left')
    val_merged = val_merged.dropna(subset=['code'])

    X_val, y_val = processor.transform(val_merged)
    code_val = val_merged['code'].tolist()

    print(f"  Validation samples: {len(y_val)}")
    sys.stdout.flush()

    # Filter for AI samples only
    ai_mask = y_val > 0
    X_val_ai = X_val[ai_mask]
    y_val_ai = y_val[ai_mask]
    code_val_ai = [code_val[i] for i in range(len(code_val)) if ai_mask[i]]

    print(f"  AI samples (y > 0): {len(y_val_ai)}")
    sys.stdout.flush()

    # Predict
    y_pred = clf.predict(X_val_ai, code_strings=code_val_ai)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    accuracy = accuracy_score(y_val_ai, y_pred)
    macro_f1 = f1_score(y_val_ai, y_pred, average='macro')
    weighted_f1 = f1_score(y_val_ai, y_pred, average='weighted')

    print(f"\n  Validation Metrics (AI samples only):")
    print(f"    Accuracy:    {accuracy:.4f}")
    print(f"    Macro F1:    {macro_f1:.4f}")
    print(f"    Weighted F1: {weighted_f1:.4f}")
    sys.stdout.flush()

    # Detailed report
    print(f"\n  Classification Report:")
    print(classification_report(
        y_val_ai,
        y_pred,
        labels=clf.classes_,
        target_names=[f'AI-{i}' for i in clf.classes_],
        zero_division=0
    ))
    sys.stdout.flush()

    # ====================================================================
    # STEP 6: Save Model
    # ====================================================================
    print(f"\nSaving model...")
    sys.stdout.flush()

    save_dir = Path(models_dir) / 'stage2_codet5p_770m_hybrid'
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / 'classifier.pth'
    clf.save(str(model_path))

    # Save metadata
    metadata = {
        'model_name': 'CodeT5p-770m-Hybrid',
        'model_path': 'Salesforce/codet5p-770m',
        'model_type': 'hybrid',
        'num_classes': int(clf.n_classes_),
        'model_params': '738M (encoder only: 335M)',
        'embedding_dim': 1024,
        'feature_dim': int(X_train.shape[1]),
        'combined_dim': 1024 + X_train.shape[1],
        'training_samples': int(len(y_train)),
        'validation_samples': int(len(y_val_ai)),
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_epochs': max_epochs,
            'early_stopping': early_stopping,
            'patience': patience,
            'dropout': dropout,
            'validation_fraction': validation_fraction
        },
        'validation_metrics': {
            'accuracy': float(accuracy),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1)
        },
        'training_time_seconds': float(training_time),
        'timestamp': datetime.now().isoformat(),
        'device': device
    }

    metadata_path = save_dir / 'config.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Model saved to: {model_path}")
    print(f"  ✓ Metadata saved to: {metadata_path}")
    sys.stdout.flush()

    # ====================================================================
    # Final Summary
    # ====================================================================
    print("\n" + "="*80)
    print("TRAINING COMPLETE: HYBRID CodeT5p-770m")
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
        description='Train HYBRID Stage 2 classifier (CodeT5p-770m + features)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (optimal for RTX 5090)
  uv run python src/training/train_stage2_codet5p_770m_hybrid.py

  # For smaller GPUs
  uv run python src/training/train_stage2_codet5p_770m_hybrid.py --batch-size 64

  # Custom hyperparameters
  uv run python src/training/train_stage2_codet5p_770m_hybrid.py \\
      --batch-size 256 \\
      --max-epochs 15 \\
      --learning-rate 1e-4 \\
      --dropout 0.3
        """
    )

    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size (default: 256)')
    parser.add_argument('--max-epochs', type=int, default=12,
                       help='Maximum epochs (default: 12)')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate (default: 0.3)')
    parser.add_argument('--models-dir', default='models',
                       help='Models directory (default: models)')
    parser.add_argument('--data-dir', default='data/reports',
                       help='Feature reports directory (default: data/reports)')
    parser.add_argument('--device', default='cuda',
                       help='Device: cuda or cpu (default: cuda)')

    args = parser.parse_args()

    # Check Stage 1 scaler exists
    scaler_path = Path(args.models_dir) / 'stage1' / 'scaler.pkl'
    if not scaler_path.exists():
        print(f"❌ ERROR: Stage 1 scaler not found: {scaler_path}")
        print("\nPlease train Stage 1 first:")
        print("  uv run python src/training/train_stage1.py")
        sys.exit(1)

    # Train
    try:
        train_stage2_codet5p_770m_hybrid(
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            models_dir=args.models_dir,
            data_dir=args.data_dir,
            device=args.device
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
