"""
Train Stage 2 CodeT5-large Multi-class Classifier

This script trains a CodeT5-large (335M params) based classifier for AI family detection.
Uses Salesforce/codet5-large - a larger variant than the default 220M model.

Key differences from 220m version:
- Model: Salesforce/codet5-large (335M params vs 220M)
- Embedding: 1024-dim (vs 768-dim for 220M)
- Saves to: models/stage2_codet5_large/ (preserves existing models)
- Batch size: Adjusted for larger model (default 64)

Usage:
    # With default settings
    uv run python src/training/train_stage2_codet5_large.py

    # With custom batch size for larger GPU
    uv run python src/training/train_stage2_codet5_large.py --batch-size 128

    # For smaller GPU (16GB VRAM)
    uv run python src/training/train_stage2_codet5_large.py --batch-size 32
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
from src.models.stage2_codet5 import Stage2CodeT5


def train_stage2_codet5_large(
    batch_size=256,
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
    Train Stage 2 CodeT5-large classifier

    Args:
        batch_size: Training batch size (default: 256 for efficient GPU utilization)
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
    print("TRAINING STAGE 2: CodeT5-large Multi-class Classifier")
    print("="*80 + "\n")

    import sys
    print(f"Configuration:")
    print(f"  Model: Salesforce/codet5-large (335M parameters)")
    print(f"  Embedding dimension: 1024")
    print(f"  Batch size: {batch_size}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping: {early_stopping} (patience={patience})")
    print(f"  Validation fraction: {validation_fraction}")
    print(f"  Device: {device}")
    print(f"  Save directory: {models_dir}/stage2_codet5_large/")
    print()
    sys.stdout.flush()

    # ================================================================
    # STEP 1: Load Training Data
    # ================================================================
    print("="*80)
    print("STEP 1: Loading Training Data")
    print("="*80 + "\n")
    sys.stdout.flush()

    # Load features
    print("Loading training features...")
    sys.stdout.flush()
    loader = FeatureLoader(data_dir)
    df = loader.load_features('train')
    print(f"  ✓ Loaded {len(df):,} samples with {len(df.columns)} columns")
    sys.stdout.flush()

    # Load code strings
    print("\nLoading training code strings...")
    sys.stdout.flush()
    data_path = 'data/train.parquet'
    data_df = pd.read_parquet(data_path)
    code_strings = data_df['code'].tolist()
    print(f"  ✓ Loaded {len(code_strings):,} code samples")
    sys.stdout.flush()

    # Verify alignment
    if len(df) != len(code_strings):
        raise ValueError(f"Feature count ({len(df)}) != code count ({len(code_strings)})")

    # ================================================================
    # STEP 2: Load Feature Processor (Scaler)
    # ================================================================
    print("\n" + "="*80)
    print("STEP 2: Loading Feature Processor")
    print("="*80 + "\n")
    sys.stdout.flush()

    scaler_path = Path(models_dir) / 'stage1' / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}\n"
            f"Please train Stage 1 first:\n"
            f"  uv run python src/training/train_stage1.py"
        )

    processor = FeatureProcessor()

    # Handle both old format (raw StandardScaler) and new format (dict)
    import joblib
    scaler_obj = joblib.load(str(scaler_path))

    if isinstance(scaler_obj, dict):
        # New format: dict with scaler and metadata
        print(f"  ✓ Loaded scaler (new format): dict with {len(scaler_obj)} keys")
        processor.scaler = scaler_obj['scaler']
        processor.feature_names = scaler_obj.get('feature_names', [])
        processor.is_fitted = True
        print(f"  Features: {len(processor.feature_names)}")
    else:
        # Old format: raw StandardScaler object
        print(f"  ✓ Loaded scaler (old format): {type(scaler_obj).__name__}")
        processor.scaler = scaler_obj
        processor.is_fitted = True
        # Extract feature names from df
        processor.feature_names = [col for col in df.columns if col not in processor.metadata_cols]
        print(f"  Features: {len(processor.feature_names)}")

    # Transform features
    print("\nTransforming features using fitted scaler...")
    sys.stdout.flush()
    X, y = processor.transform(df)
    print(f"  ✓ Features transformed: {X.shape}")
    print(f"  ✓ Labels shape: {y.shape}")
    sys.stdout.flush()

    # ================================================================
    # STEP 3: Filter AI Samples Only
    # ================================================================
    print("\n" + "="*80)
    print("STEP 3: Filtering AI Samples for Multi-class Training")
    print("="*80 + "\n")
    sys.stdout.flush()

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
    sys.stdout.flush()

    # ================================================================
    # STEP 4: Initialize CodeT5-large Classifier
    # ================================================================
    print("\n" + "="*80)
    print("STEP 4: Initializing CodeT5-large Classifier")
    print("="*80 + "\n")
    sys.stdout.flush()

    clf = Stage2CodeT5(
        model_name='Salesforce/codet5-large',  # 335M model
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_epochs=max_epochs,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        patience=patience,
        device=device,
        verbose=1
    )

    print(f"  ✓ CodeT5-large classifier initialized")
    print(f"  ✓ Ready for training on {len(y_ai):,} AI samples\n")
    sys.stdout.flush()

    # ================================================================
    # STEP 5: Train Classifier
    # ================================================================
    print("="*80)
    print("STEP 5: Training CodeT5-large Classifier")
    print("="*80 + "\n")

    start_time = datetime.now()
    print(f"Training started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    sys.stdout.flush()

    # Train on AI samples only
    clf.fit(
        X_ai,
        y_ai,
        code_strings=code_ai
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
    save_dir = Path(models_dir) / 'stage2_codet5_large'
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
        'model_name': 'CodeT5-large',
        'model_path': 'Salesforce/codet5-large',
        'num_classes': 10,
        'model_params': '335M',
        'embedding_dim': 1024,
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
    print(f"Model: CodeT5-large (Salesforce/codet5-large, 335M params)")
    print(f"Training samples: {len(y_ai):,}")
    print(f"Validation samples: {len(y_val_ai):,}")
    print(f"Training time: {duration/60:.1f} minutes")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation Macro F1: {val_f1_macro:.4f}")
    print(f"Model saved to: {save_dir}/")
    print("="*80 + "\n")

    print("✓ Stage 2 CodeT5-large training complete!")
    print(f"\nNext steps:")
    print(f"  1. Evaluate full pipeline: uv run python src/evaluation/evaluate_pipeline_codet5.py --stage2-dir stage2_codet5_large")
    print(f"  2. Compare with 220m version: Check models/stage2_codet5/ vs models/stage2_codet5_large/")
    print()

    return clf, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Stage 2 CodeT5-large Multi-class Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (batch_size=64)
  uv run python src/training/train_stage2_codet5_large.py

  # Train with larger batch size (for GPUs with 40GB+ VRAM)
  uv run python src/training/train_stage2_codet5_large.py --batch-size 128

  # Train with smaller batch size (for 16GB VRAM)
  uv run python src/training/train_stage2_codet5_large.py --batch-size 32

  # Custom learning rate
  uv run python src/training/train_stage2_codet5_large.py --learning-rate 1e-4

  # More epochs
  uv run python src/training/train_stage2_codet5_large.py --max-epochs 10
        """
    )

    parser.add_argument('--batch-size', type=int, default=256,
                       help='Training batch size (default: 256)')
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
        train_stage2_codet5_large(
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
