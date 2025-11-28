"""
Train Stage 2 CodeT5p-2b Multi-class Classifier

This script trains a CodeT5p-2b based classifier for AI family detection.
Uses a LARGER 2B parameter encoder instead of 220m.

Key differences from 220m version:
- Model: Salesforce/codet5p-2b (~2GB download vs ~1GB)
- Encoder: ~2 billion parameters (vs 109M)
- Batch size: Smaller (default 32 vs 256) due to memory requirements
- Saves to: models/stage2_codet5_2b/ (preserves existing 220m model)

Usage:
    # With default settings (batch_size=32)
    uv run python src/training/train_stage2_codet5_2b.py

    # With custom batch size for larger GPU
    uv run python src/training/train_stage2_codet5_2b.py --batch-size 64

    # For smaller GPU (16GB VRAM)
    uv run python src/training/train_stage2_codet5_2b.py --batch-size 16
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.features.feature_loader import FeatureLoader
from src.features.feature_processor import FeatureProcessor
from src.models.stage2_codet5_2b import Stage2CodeT5_2b


def train_stage2_codet5_2b(
    batch_size=32,
    max_epochs=5,
    learning_rate=2e-4,
    early_stopping=True,
    validation_fraction=0.1,
    patience=3,
    models_dir='models',
    data_dir='data/reports',
    device='cuda'
):
    """
    Train Stage 2 CodeT5p-2b classifier

    Args:
        batch_size: Training batch size (default: 32, smaller than 220m due to memory)
        max_epochs: Maximum training epochs
        learning_rate: Learning rate for AdamW optimizer
        early_stopping: Whether to use early stopping
        validation_fraction: Fraction of data for validation
        patience: Early stopping patience
        models_dir: Directory to save models
        data_dir: Directory containing feature CSVs
        device: 'cuda' or 'cpu'
    """
    print("\n" + "="*80)
    print("TRAINING STAGE 2: CodeT5p-2B Multi-class Classifier")
    print("="*80 + "\n")

    print(f"Configuration:")
    print(f"  Model: Salesforce/codet5p-2b (~2B parameters)")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping: {early_stopping} (patience={patience})")
    print(f"  Validation fraction: {validation_fraction}")
    print(f"  Device: {device}")
    print(f"  Save directory: {models_dir}/stage2_codet5_2b/")
    print()

    # ================================================================
    # STEP 1: Load Training Data
    # ================================================================
    print("="*80)
    print("STEP 1: Loading Training Data")
    print("="*80 + "\n")

    # Load features
    print("Loading training features...")
    loader = FeatureLoader(data_dir)
    df = loader.load_features('train')
    print(f"  ✓ Loaded {len(df):,} samples with {len(df.columns)} columns")

    # Load code strings
    print("\nLoading training code strings...")
    data_path = 'data/train.parquet'
    data_df = pd.read_parquet(data_path)
    code_strings = data_df['code'].tolist()
    print(f"  ✓ Loaded {len(code_strings):,} code samples")

    # Verify alignment
    if len(df) != len(code_strings):
        raise ValueError(f"Feature count ({len(df)}) != code count ({len(code_strings)})")

    # ================================================================
    # STEP 2: Load Feature Processor (Scaler)
    # ================================================================
    print("\n" + "="*80)
    print("STEP 2: Loading Feature Processor")
    print("="*80 + "\n")

    scaler_path = Path(models_dir) / 'stage1' / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            f"Please train Stage 1 first:\n"
            f"  uv run python src/training/train_stage1.py"
        )

    processor = FeatureProcessor()
    processor.load(str(scaler_path))
    print(f"  ✓ Loaded scaler with {len(processor.feature_names)} features")

    # Transform features
    print("\nTransforming features using fitted scaler...")
    X, y = processor.transform(df)
    print(f"  ✓ Features transformed: {X.shape}")
    print(f"  ✓ Labels shape: {y.shape}")

    # ================================================================
    # STEP 3: Filter AI Samples Only
    # ================================================================
    print("\n" + "="*80)
    print("STEP 3: Filtering AI Samples for Multi-class Training")
    print("="*80 + "\n")

    ai_mask = y > 0
    X_ai = X[ai_mask]
    y_ai = y[ai_mask]
    code_ai = [code_strings[i] for i in range(len(code_strings)) if ai_mask[i]]

    print(f"Total samples: {len(y):,}")
    print(f"AI samples: {len(y_ai):,} ({100*len(y_ai)/len(y):.1f}%)")
    print(f"Human samples (excluded): {len(y) - len(y_ai):,}")

    # Show class distribution
    print(f"\nAI Family Distribution:")
    unique, counts = np.unique(y_ai, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  AI-{label}: {count:,} ({100*count/len(y_ai):.1f}%)")

    # ================================================================
    # STEP 4: Initialize CodeT5p-2b Classifier
    # ================================================================
    print("\n" + "="*80)
    print("STEP 4: Initializing CodeT5p-2b Classifier")
    print("="*80 + "\n")

    clf = Stage2CodeT5_2b(
        num_classes=10,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        patience=patience,
        device=device,
        verbose=1
    )

    print(f"  ✓ CodeT5p-2b classifier initialized")
    print(f"  ✓ Ready for training on {len(y_ai):,} AI samples\n")

    # ================================================================
    # STEP 5: Train Classifier
    # ================================================================
    print("="*80)
    print("STEP 5: Training CodeT5p-2b Classifier")
    print("="*80 + "\n")

    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Train on AI samples only (y > 0)
    # The fit() method internally filters AI samples, but we pass filtered data
    clf.fit(
        X_ai,
        y_ai,
        code_strings=code_ai,
        validation_fraction=validation_fraction
    )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n✓ Training complete!")
    print(f"  Total time: {duration/60:.1f} minutes ({duration:.1f} seconds)")

    # ================================================================
    # STEP 6: Evaluate on Validation Set
    # ================================================================
    print("\n" + "="*80)
    print("STEP 6: Evaluating on Validation Set")
    print("="*80 + "\n")

    # Load validation data
    print("Loading validation features...")
    df_val = loader.load_features('validation')
    print(f"  ✓ Loaded {len(df_val):,} validation samples")

    print("\nLoading validation code strings...")
    val_data = pd.read_parquet('data/validation.parquet')
    val_code_strings = val_data['code'].tolist()
    print(f"  ✓ Loaded {len(val_code_strings):,} code samples")

    # Transform features
    print("\nTransforming validation features...")
    X_val, y_val = processor.transform(df_val)
    print(f"  ✓ Features transformed: {X_val.shape}")

    # Filter AI samples
    ai_mask_val = y_val > 0
    X_val_ai = X_val[ai_mask_val]
    y_val_ai = y_val[ai_mask_val]
    code_val_ai = [val_code_strings[i] for i in range(len(val_code_strings)) if ai_mask_val[i]]

    print(f"\nValidation AI samples: {len(y_val_ai):,}")

    # Predict
    print("\nGenerating predictions...")
    val_preds = clf.predict(X_val_ai, code_strings=code_val_ai)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    val_acc = accuracy_score(y_val_ai, val_preds)
    val_f1_macro = f1_score(y_val_ai, val_preds, average='macro', labels=list(range(1, 11)), zero_division=0)
    val_f1_weighted = f1_score(y_val_ai, val_preds, average='weighted', labels=list(range(1, 11)), zero_division=0)

    print("\n" + "="*80)
    print("VALIDATION RESULTS (AI Samples Only)")
    print("="*80)
    print(classification_report(
        y_val_ai,
        val_preds,
        labels=list(range(1, 11)),
        target_names=[f'AI-{i}' for i in range(1, 11)],
        digits=4,
        zero_division=0
    ))

    print(f"\nValidation Metrics:")
    print(f"  Macro F1:    {val_f1_macro:.4f}")
    print(f"  Weighted F1: {val_f1_weighted:.4f}")
    print(f"  Accuracy:    {val_acc:.4f}")
    print("="*80 + "\n")

    # ================================================================
    # STEP 7: Save Model
    # ================================================================
    print("="*80)
    print("STEP 7: Saving Model")
    print("="*80 + "\n")

    # Create save directory
    save_dir = Path(models_dir) / 'stage2_codet5_2b'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save classifier
    model_path = save_dir / 'classifier.pth'
    clf.save(str(model_path))
    print(f"✓ Model saved to: {model_path}")

    # Get file size
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"  Model size: {model_size_mb:.2f} MB")

    # Save training metadata
    metadata = {
        'model_name': 'CodeT5p-2b',
        'model_path': 'Salesforce/codet5p-2b',
        'num_classes': 10,
        'training_samples': int(len(y_ai)),
        'validation_samples': int(len(y_val_ai)),
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_epochs': max_epochs,
            'early_stopping': early_stopping,
            'patience': patience,
            'validation_fraction': validation_fraction
        },
        'validation_metrics': {
            'accuracy': float(val_acc),
            'macro_f1': float(val_f1_macro),
            'weighted_f1': float(val_f1_weighted)
        },
        'training_time_seconds': float(duration),
        'timestamp': datetime.now().isoformat(),
        'device': device
    }

    metadata_path = save_dir / 'config.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path}")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Model: CodeT5p-2b (Salesforce/codet5p-2b)")
    print(f"Training samples: {len(y_ai):,}")
    print(f"Validation samples: {len(y_val_ai):,}")
    print(f"Training time: {duration/60:.1f} minutes")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Macro F1: {val_f1_macro:.4f}")
    print(f"Model saved to: {save_dir}/")
    print("="*80 + "\n")

    print("✓ Stage 2 CodeT5p-2b training complete!")
    print(f"\nNext steps:")
    print(f"  1. Evaluate full pipeline: uv run python src/evaluation/evaluate_pipeline_codet5.py --stage2-model stage2_codet5_2b")
    print(f"  2. Compare with 220m version: Check models/stage2_codet5/ vs models/stage2_codet5_2b/")
    print()

    return clf, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Stage 2 CodeT5p-2b Multi-class Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (batch_size=32)
  uv run python src/training/train_stage2_codet5_2b.py

  # Train with larger batch size (for GPUs with 40GB+ VRAM)
  uv run python src/training/train_stage2_codet5_2b.py --batch-size 64

  # Train with smaller batch size (for 16GB VRAM)
  uv run python src/training/train_stage2_codet5_2b.py --batch-size 16

  # Custom learning rate
  uv run python src/training/train_stage2_codet5_2b.py --learning-rate 1e-4

  # More epochs
  uv run python src/training/train_stage2_codet5_2b.py --max-epochs 10
        """
    )

    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size (default: 32, smaller than 220m)')
    parser.add_argument('--max-epochs', type=int, default=5,
                       help='Maximum training epochs (default: 5)')
    parser.add_argument('--learning-rate', type=float, default=2e-4,
                       help='Learning rate (default: 2e-4)')
    parser.add_argument('--patience', type=int, default=3,
                       help='Early stopping patience (default: 3)')
    parser.add_argument('--validation-fraction', type=float, default=0.1,
                       help='Validation data fraction (default: 0.1)')
    parser.add_argument('--no-early-stopping', action='store_true',
                       help='Disable early stopping')
    parser.add_argument('--device', default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use (default: cuda)')
    parser.add_argument('--models-dir', default='models',
                       help='Directory to save models (default: models)')
    parser.add_argument('--data-dir', default='data/reports',
                       help='Directory containing feature CSVs (default: data/reports)')

    args = parser.parse_args()

    # Check if Stage 1 scaler exists
    scaler_path = Path(args.models_dir) / 'stage1' / 'scaler.pkl'
    if not scaler_path.exists():
        print(f"❌ ERROR: Stage 1 scaler not found: {scaler_path}")
        print("\nPlease train Stage 1 first:")
        print("  uv run python src/training/train_stage1.py")
        sys.exit(1)

    # Train
    try:
        train_stage2_codet5_2b(
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            learning_rate=args.learning_rate,
            early_stopping=not args.no_early_stopping,
            validation_fraction=args.validation_fraction,
            patience=args.patience,
            models_dir=args.models_dir,
            data_dir=args.data_dir,
            device=args.device
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
