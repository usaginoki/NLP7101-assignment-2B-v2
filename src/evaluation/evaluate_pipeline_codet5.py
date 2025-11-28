"""
Evaluate Full 2-Stage Pipeline: Stage 1 (SVM) + Stage 2 (CodeT5)

This script evaluates the complete cascaded pipeline on the validation set.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.features.feature_loader import FeatureLoader
from src.features.feature_processor import FeatureProcessor
from src.models.stage2_codet5 import Stage2CodeT5


def evaluate_full_pipeline(models_dir='models', data_dir='data/reports', split='validation'):
    """
    Evaluate full 2-stage pipeline

    Args:
        models_dir: Directory containing trained models
        data_dir: Directory containing feature CSVs
        split: Data split to evaluate on
    """
    print("\n" + "="*70)
    print("FULL PIPELINE EVALUATION: Stage 1 + Stage 2 (CodeT5)")
    print("="*70 + "\n")

    # Load metadata (try XGBoost first, fallback to default)
    xgb_metadata_path = Path(models_dir) / 'stage1' / 'XGBoost' / 'metadata.json'
    default_metadata_path = Path(models_dir) / 'stage1' / 'model_metadata.json'

    if xgb_metadata_path.exists():
        metadata_path = xgb_metadata_path
        stage1_model_dir = Path(models_dir) / 'stage1' / 'XGBoost'
    else:
        metadata_path = default_metadata_path
        stage1_model_dir = Path(models_dir) / 'stage1'

    with open(metadata_path) as f:
        metadata = json.load(f)

    print(f"Stage 1 Model: {metadata['model_name']}")
    print(f"  Macro F1: {metadata['val_macro_f1']:.4f}")
    print(f"  Accuracy: {metadata['accuracy']:.4f}\n")

    # ================================================================
    # STEP 1: Load Data
    # ================================================================
    print("="*70)
    print("STEP 1: Loading Data")
    print("="*70 + "\n")

    # Load features for Stage 1
    print(f"Loading {split} features...")
    loader = FeatureLoader(data_dir)
    df = loader.load_features(split)
    print(f"  ✓ Loaded {len(df):,} samples")

    # Load code strings for Stage 2
    print(f"\nLoading {split} code strings...")
    data_path = f'data/{split}.parquet'
    data_df = pd.read_parquet(data_path)
    code_strings = data_df['code'].tolist()
    print(f"  ✓ Loaded {len(code_strings):,} code samples")

    # Load scaler and transform features
    print("\nLoading feature processor...")
    scaler_path = f'{models_dir}/stage1/scaler.pkl'
    scaler_obj = joblib.load(scaler_path)

    # Handle both old format (raw StandardScaler) and new format (dict)
    if isinstance(scaler_obj, dict):
        processor = FeatureProcessor()
        processor.load(scaler_path)
    else:
        # Old format: raw StandardScaler object
        print(f"  ✓ Loaded scaler (old format): {type(scaler_obj).__name__}")
        processor = FeatureProcessor()
        processor.scaler = scaler_obj
        processor.is_fitted = True
        # Extract feature names from df
        processor.feature_names = [col for col in df.columns if col not in processor.metadata_cols]
        print(f"  Features: {len(processor.feature_names)}")

    X, y_true = processor.transform(df)
    print(f"  ✓ Features transformed: {X.shape}")

    # ================================================================
    # STEP 2: Load Stage 1 Classifier
    # ================================================================
    print("\n" + "="*70)
    print("STEP 2: Loading Stage 1 Classifier")
    print("="*70 + "\n")

    stage1_path = stage1_model_dir / 'classifier.pkl'
    stage1_clf = joblib.load(stage1_path)
    print(f"✓ Loaded Stage 1: {stage1_path}")
    print(f"  Model type: {type(stage1_clf).__name__}")

    # ================================================================
    # STEP 3: Load Stage 2 CodeT5 Classifier
    # ================================================================
    print("\n" + "="*70)
    print("STEP 3: Loading Stage 2 CodeT5 Classifier")
    print("="*70 + "\n")

    stage2_path = Path(models_dir) / 'stage2_codet5' / 'classifier.pth'
    if not stage2_path.exists():
        raise FileNotFoundError(f"Stage 2 CodeT5 model not found: {stage2_path}")

    stage2_clf = Stage2CodeT5(verbose=1)
    stage2_clf.load(str(stage2_path))
    print(f"✓ Loaded Stage 2: {stage2_path}")

    # ================================================================
    # STEP 4: Run Stage 1 Predictions
    # ================================================================
    print("\n" + "="*70)
    print("STEP 4: Running Stage 1 Predictions")
    print("="*70 + "\n")

    print("Predicting with Stage 1 (binary: Human vs AI)...")
    stage1_preds = stage1_clf.predict(X)

    # Convert multi-class labels to binary for Stage 1 evaluation
    y_binary = (y_true > 0).astype(int)

    stage1_f1 = f1_score(y_binary, stage1_preds, average='macro')
    stage1_acc = accuracy_score(y_binary, stage1_preds)

    n_human = np.sum(stage1_preds == 0)
    n_ai = np.sum(stage1_preds == 1)

    print(f"✓ Stage 1 predictions complete")
    print(f"  Predicted Human: {n_human:,} ({100*n_human/len(stage1_preds):.1f}%)")
    print(f"  Predicted AI: {n_ai:,} ({100*n_ai/len(stage1_preds):.1f}%)")
    print(f"  Stage 1 Macro F1: {stage1_f1:.4f}")
    print(f"  Stage 1 Accuracy: {stage1_acc:.4f}")

    # ================================================================
    # STEP 5: Run Stage 2 Predictions (AI samples only)
    # ================================================================
    print("\n" + "="*70)
    print("STEP 5: Running Stage 2 Predictions")
    print("="*70 + "\n")

    # Filter samples predicted as AI by Stage 1
    ai_mask = stage1_preds == 1
    X_ai = X[ai_mask]
    code_ai = [code_strings[i] for i in range(len(code_strings)) if ai_mask[i]]

    print(f"Samples sent to Stage 2: {len(X_ai):,}")

    if len(X_ai) == 0:
        print("⚠ WARNING: No samples predicted as AI by Stage 1!")
        final_preds = stage1_preds.copy()
    else:
        print("Predicting with Stage 2 (AI families 1-10)...")
        stage2_preds = stage2_clf.predict(X_ai, code_strings=code_ai)
        print(f"✓ Stage 2 predictions complete")

        # ================================================================
        # STEP 6: Combine Predictions
        # ================================================================
        print("\n" + "="*70)
        print("STEP 6: Combining Stage 1 + Stage 2 Predictions")
        print("="*70 + "\n")

        # Initialize final predictions with Stage 1 results
        final_preds = stage1_preds.copy()

        # Replace AI predictions (1) with Stage 2 predictions (1-10)
        final_preds[ai_mask] = stage2_preds

        print(f"✓ Combined predictions created")
        print(f"  Total samples: {len(final_preds):,}")
        print(f"  Predicted as Human (0): {np.sum(final_preds == 0):,}")
        for i in range(1, 11):
            count = np.sum(final_preds == i)
            print(f"  Predicted as AI-{i}: {count:,}")

    # ================================================================
    # STEP 7: Evaluate Full Pipeline
    # ================================================================
    print("\n" + "="*70)
    print("STEP 7: Evaluating Full Pipeline Performance")
    print("="*70 + "\n")

    # Overall metrics (11-class: 0-10)
    overall_f1_macro = f1_score(y_true, final_preds, average='macro', labels=list(range(11)), zero_division=0)
    overall_f1_weighted = f1_score(y_true, final_preds, average='weighted', labels=list(range(11)), zero_division=0)
    overall_acc = accuracy_score(y_true, final_preds)

    print("="*70)
    print("FULL PIPELINE RESULTS (11-class: Human + 10 AI Families)")
    print("="*70)
    print(classification_report(
        y_true, final_preds,
        labels=list(range(11)),
        target_names=['Human'] + [f'AI-{i}' for i in range(1, 11)],
        digits=4,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(y_true, final_preds, labels=list(range(11)))
    print("\nConfusion Matrix (11x11):")
    print("(Rows: Actual, Columns: Predicted)")
    header = "       " + " ".join([f"{'Hu' if i==0 else f'A{i:2d}':>3}" for i in range(11)])
    print(header)
    for i, row in enumerate(cm):
        label = "Human" if i == 0 else f"AI-{i:2d}"
        row_str = " ".join([f"{val:3d}" for val in row])
        print(f"{label:>6}: {row_str}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY METRICS")
    print(f"{'='*70}")
    print(f"Overall Macro F1:    {overall_f1_macro:.4f}")
    print(f"Overall Weighted F1: {overall_f1_weighted:.4f}")
    print(f"Overall Accuracy:    {overall_acc:.4f}")
    print(f"{'='*70}\n")

    # Stage-specific evaluation
    print("="*70)
    print("STAGE-SPECIFIC PERFORMANCE")
    print("="*70)
    print(f"Stage 1 (Binary Classification):")
    print(f"  Macro F1:  {stage1_f1:.4f}")
    print(f"  Accuracy:  {stage1_acc:.4f}")

    # Stage 2 evaluation on AI samples only
    ai_true_mask = y_true > 0
    if np.sum(ai_true_mask) > 0:
        y_true_ai = y_true[ai_true_mask]
        final_preds_ai = final_preds[ai_true_mask]

        stage2_f1 = f1_score(y_true_ai, final_preds_ai, average='macro', labels=list(range(1, 11)), zero_division=0)
        stage2_acc = accuracy_score(y_true_ai, final_preds_ai)

        print(f"\nStage 2 (AI Family Classification - on true AI samples):")
        print(f"  Macro F1:  {stage2_f1:.4f}")
        print(f"  Accuracy:  {stage2_acc:.4f}")
    print(f"{'='*70}\n")

    # Save metrics
    metrics = {
        'pipeline': f"Stage1_{metadata['model_name']} + Stage2_CodeT5",
        'stage1_model': metadata['model_name'],
        'stage2_model': 'CodeT5-220m',
        'split': split,
        'total_samples': int(len(y_true)),
        'stage1_metrics': {
            'macro_f1': float(stage1_f1),
            'accuracy': float(stage1_acc)
        },
        'stage2_metrics': {
            'macro_f1': float(stage2_f1) if np.sum(ai_true_mask) > 0 else 0.0,
            'accuracy': float(stage2_acc) if np.sum(ai_true_mask) > 0 else 0.0
        },
        'overall_metrics': {
            'macro_f1': float(overall_f1_macro),
            'weighted_f1': float(overall_f1_weighted),
            'accuracy': float(overall_acc)
        },
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().isoformat()
    }

    metrics_dir = Path('outputs/metrics')
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f'pipeline_codet5_{split}.json'

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Metrics saved to: {metrics_path}\n")

    return metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate Full Pipeline with CodeT5')
    parser.add_argument('--models-dir', default='models',
                       help='Directory containing trained models')
    parser.add_argument('--data-dir', default='data/reports',
                       help='Directory containing feature CSVs')
    parser.add_argument('--split', default='validation',
                       choices=['train', 'validation'],
                       help='Data split to evaluate')

    args = parser.parse_args()

    evaluate_full_pipeline(
        models_dir=args.models_dir,
        data_dir=args.data_dir,
        split=args.split
    )
