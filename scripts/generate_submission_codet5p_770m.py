"""
Generate submission.csv using XGBoost + CodeT5p-770m Pipeline

This script:
1. Loads test data (1000 samples with 'ID' and 'code' columns)
2. Extracts 36 engineered features on-the-fly
3. Runs Stage 1 (XGBoost binary classification)
4. Runs Stage 2 (CodeT5p-770m multi-class classification) on AI samples
5. Generates submission.csv with 'ID' and 'label' columns

Uses the best hyperparameter configuration:
- Model: Salesforce/codet5p-770m
- Learning Rate: 2e-4
- Dropout: 0.3
- Config: config3_lr2e4_drop03
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../data'))

from src.features.feature_processor import FeatureProcessor
from src.models.stage2_codet5p_770m import Stage2CodeT5p770m
from modules.feature_extractors import ComplexityFeatureExtractor, LexicalFeatureExtractor


def extract_basic_features(df):
    """
    Extract 9 basic statistical features

    Args:
        df: DataFrame with 'code' column

    Returns:
        DataFrame with basic features added
    """
    print("  Extracting basic statistics...")

    df['char_count'] = df['code'].str.len()
    df['line_count'] = df['code'].str.count('\n') + 1
    df['word_count'] = df['code'].str.split().str.len()
    df['avg_line_length'] = df['char_count'] / df['line_count']
    df['space_count'] = df['code'].str.count(' ')
    df['tab_count'] = df['code'].str.count('\t')

    # Ratios
    df['alphanum_ratio'] = df['code'].apply(
        lambda x: sum(c.isalnum() for c in x) / len(x) if len(x) > 0 else 0
    )
    df['special_char_ratio'] = df['code'].apply(
        lambda x: sum(not c.isalnum() and not c.isspace() for c in x) / len(x) if len(x) > 0 else 0
    )
    df['newline_ratio'] = df['code'].str.count('\n') / df['char_count']

    return df


def extract_complexity_features(df):
    """
    Extract 19 code complexity features using radon

    Args:
        df: DataFrame with 'code' column

    Returns:
        DataFrame with complexity features added
    """
    print("  Extracting complexity features...")

    extractor = ComplexityFeatureExtractor()
    complexity_features = []

    for code in tqdm(df['code'], desc="Complexity", leave=False):
        complexity_features.append(extractor.extract(code))

    complexity_df = pd.DataFrame(complexity_features)
    df = pd.concat([df.reset_index(drop=True), complexity_df], axis=1)

    return df


def extract_lexical_features(df):
    """
    Extract 8 lexical diversity features

    Args:
        df: DataFrame with 'code' column

    Returns:
        DataFrame with lexical features added
    """
    print("  Extracting lexical features...")

    extractor = LexicalFeatureExtractor()
    lexical_features = []

    for code in tqdm(df['code'], desc="Lexical", leave=False):
        lexical_features.append(extractor.extract(code))

    lexical_df = pd.DataFrame(lexical_features)
    df = pd.concat([df.reset_index(drop=True), lexical_df], axis=1)

    return df


def generate_submission_codet5p_770m(
    test_file='data/test.parquet',
    models_dir='models',
    stage2_config='config3_lr2e4_drop03',
    output_file='outputs/submission_codet5p_770m.csv'
):
    """
    Generate submission CSV using XGBoost + CodeT5p-770m pipeline

    Submission format:
    - CSV with 'ID' and 'label' columns
    - Values: 0-10 (0=Human, 1-10=AI families)
    - Row i corresponds to test sample with ID i

    Args:
        test_file: Path to test parquet file
        models_dir: Directory containing trained models
        stage2_config: Name of the Stage 2 config directory (default: best config)
        output_file: Path to save submission CSV
    """
    print("\n" + "="*70)
    print("GENERATING SUBMISSION: XGBoost + CodeT5p-770m Pipeline")
    print(f"Stage 2 Config: {stage2_config}")
    print("="*70 + "\n")

    # ================================================================
    # STEP 1: Load Test Data
    # ================================================================
    print("STEP 1/6: Loading test data...")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    test_df = pd.read_parquet(test_file)
    print(f"  ✓ Loaded {len(test_df)} test samples")
    print(f"  Columns: {list(test_df.columns)}")

    if 'ID' not in test_df.columns or 'code' not in test_df.columns:
        raise ValueError("Test data must have 'ID' and 'code' columns")

    # Store IDs and code strings separately
    test_ids = test_df['ID'].values
    code_strings = test_df['code'].tolist()

    # ================================================================
    # STEP 2: Extract Features
    # ================================================================
    print("\nSTEP 2/6: Extracting 36 engineered features...")
    print("  (This may take a few minutes...)\n")

    # Remove ID column to prevent it being treated as a feature
    test_df = test_df.drop(columns=['ID'])

    # Basic features (9)
    test_df = extract_basic_features(test_df)

    # Complexity features (19)
    test_df = extract_complexity_features(test_df)

    # Lexical features (8)
    test_df = extract_lexical_features(test_df)

    # Add dummy columns for processor compatibility
    test_df['generator'] = 'unknown'
    test_df['label'] = -1
    test_df['language'] = 'unknown'

    print(f"\n  ✓ Total columns: {len(test_df.columns)}")

    # ================================================================
    # STEP 3: Load Feature Processor
    # ================================================================
    print("\nSTEP 3/6: Loading feature processor...")
    scaler_path = Path(models_dir) / 'stage1' / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path}\nTrain Stage 1 first!")

    scaler_obj = joblib.load(scaler_path)

    # Handle both old format (raw StandardScaler) and new format (dict)
    if isinstance(scaler_obj, dict):
        processor = FeatureProcessor()
        processor.load(str(scaler_path))
    else:
        # Old format: raw StandardScaler object
        print(f"  ✓ Loaded scaler (old format): {type(scaler_obj).__name__}")
        processor = FeatureProcessor()
        processor.scaler = scaler_obj
        processor.is_fitted = True
        # Extract feature names from df
        processor.feature_names = [col for col in test_df.columns if col not in processor.metadata_cols]
        print(f"  Features: {len(processor.feature_names)}")

    X_test, _ = processor.transform(test_df)
    print(f"  ✓ Features transformed: {X_test.shape}")

    # ================================================================
    # STEP 4: Load Stage 1 Classifier (XGBoost)
    # ================================================================
    print("\nSTEP 4/6: Loading Stage 1 classifier...")

    # Try XGBoost first, fallback to default
    xgb_path = Path(models_dir) / 'stage1' / 'XGBoost' / 'classifier.pkl'
    default_path = Path(models_dir) / 'stage1' / 'classifier.pkl'

    if xgb_path.exists():
        stage1_path = xgb_path
        print(f"  Using XGBoost: {stage1_path}")
    elif default_path.exists():
        stage1_path = default_path
        print(f"  Using default: {stage1_path}")
    else:
        raise FileNotFoundError(f"Stage 1 model not found!\nTrain Stage 1 first!")

    stage1_clf = joblib.load(stage1_path)
    print(f"  ✓ Loaded Stage 1: {type(stage1_clf).__name__}")

    # ================================================================
    # STEP 5: Load Stage 2 Classifier (CodeT5p-770m)
    # ================================================================
    print("\nSTEP 5/6: Loading Stage 2 CodeT5p-770m classifier...")
    print(f"  Config: {stage2_config}")

    stage2_dir = Path(models_dir) / f'stage2_codet5p_770m_{stage2_config}'
    stage2_path = stage2_dir / 'classifier.pth'

    if not stage2_path.exists():
        raise FileNotFoundError(
            f"Stage 2 CodeT5p-770m model not found: {stage2_path}\n"
            f"Available configs: {list((Path(models_dir)).glob('stage2_codet5p_770m_*'))}"
        )

    print(f"  Loading model (this may take 3-5 minutes)...")
    stage2_clf = Stage2CodeT5p770m(verbose=1)
    stage2_clf.load(str(stage2_path))
    print(f"  ✓ Loaded Stage 2: {stage2_path}")

    # ================================================================
    # STEP 6: Generate Predictions
    # ================================================================
    print("\nSTEP 6/6: Generating predictions...")

    # Stage 1: Binary prediction (Human vs AI)
    print("\n  Running Stage 1 (Binary: Human vs AI)...")
    stage1_preds = stage1_clf.predict(X_test)

    n_human = np.sum(stage1_preds == 0)
    n_ai = np.sum(stage1_preds == 1)
    print(f"    Predicted Human: {n_human} ({100*n_human/len(stage1_preds):.1f}%)")
    print(f"    Predicted AI: {n_ai} ({100*n_ai/len(stage1_preds):.1f}%)")

    # Initialize final predictions with Stage 1 results
    final_preds = stage1_preds.copy()

    # Stage 2: Multi-class prediction for AI samples
    if n_ai > 0:
        print(f"\n  Running Stage 2 (AI families 1-10) on {n_ai} samples...")
        ai_mask = stage1_preds == 1
        X_ai = X_test[ai_mask]
        code_ai = [code_strings[i] for i in range(len(code_strings)) if ai_mask[i]]

        stage2_preds = stage2_clf.predict(X_ai, code_strings=code_ai)

        # Replace AI predictions (1) with Stage 2 predictions (1-10)
        final_preds[ai_mask] = stage2_preds
        print(f"    ✓ Stage 2 complete")
    else:
        print(f"\n  ⚠ WARNING: No samples predicted as AI by Stage 1!")

    # ================================================================
    # Create Submission DataFrame
    # ================================================================
    print("\nCreating submission file...")

    submission_df = pd.DataFrame({
        'ID': test_ids,
        'label': final_preds
    })

    # Show prediction distribution
    print(f"\nPrediction distribution:")
    unique, counts = np.unique(final_preds, return_counts=True)
    for label, count in zip(unique, counts):
        pct = 100 * count / len(final_preds)
        name = 'Human' if label == 0 else f'AI-{label}'
        print(f"  {name:8s}: {count:4d} ({pct:5.1f}%)")

    # Save submission
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    submission_df.to_csv(output_file, index=False)

    print(f"\n✓ Submission saved to: {output_file}")
    print(f"✓ Total predictions: {len(final_preds)}")
    print(f"✓ File size: {os.path.getsize(output_file)} bytes")

    # Validation checks
    print("\nValidation checks:")
    print(f"  ✓ Row count: {len(submission_df)} (expected: {len(test_ids)})")
    print(f"  ✓ Columns: {list(submission_df.columns)}")
    print(f"  ✓ Value range: {final_preds.min()}-{final_preds.max()} (expected: 0-10)")
    print(f"  ✓ Data type: {final_preds.dtype}")
    print(f"  ✓ No missing values: {submission_df.isnull().sum().sum() == 0}")

    print("\n" + "="*70)
    print("Submission generation complete!")
    print("="*70 + "\n")

    return submission_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate submission using XGBoost + CodeT5p-770m pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate submission with best config (config3)
  uv run python scripts/generate_submission_codet5p_770m.py

  # Use a different config
  uv run python scripts/generate_submission_codet5p_770m.py --config config1_lr1e4_drop03

  # Use custom test file
  uv run python scripts/generate_submission_codet5p_770m.py --test-file custom/test.parquet

  # Save to custom location
  uv run python scripts/generate_submission_codet5p_770m.py --output custom/submission.csv
        """
    )

    parser.add_argument('--test-file', default='data/test.parquet',
                       help='Path to test parquet file')
    parser.add_argument('--models-dir', default='models',
                       help='Directory containing trained models')
    parser.add_argument('--config', default='config3_lr2e4_drop03',
                       help='Stage 2 config directory name (default: best config)')
    parser.add_argument('--output', default='outputs/submission_codet5p_770m.csv',
                       help='Path to save submission CSV')

    args = parser.parse_args()

    # Verify models exist
    xgb_path = Path(args.models_dir) / 'stage1' / 'XGBoost' / 'classifier.pkl'
    default_path = Path(args.models_dir) / 'stage1' / 'classifier.pkl'
    stage2_path = Path(args.models_dir) / f'stage2_codet5p_770m_{args.config}' / 'classifier.pth'

    if not (xgb_path.exists() or default_path.exists()):
        print(f"❌ ERROR: Stage 1 model not found in {args.models_dir}/stage1/")
        print("\nPlease train Stage 1 first:")
        print("  uv run python src/training/train_stage1.py")
        sys.exit(1)

    if not stage2_path.exists():
        print(f"❌ ERROR: Stage 2 CodeT5p-770m model not found: {stage2_path}")
        print("\nAvailable Stage 2 configs:")
        for config_dir in sorted(Path(args.models_dir).glob('stage2_codet5p_770m_*')):
            print(f"  - {config_dir.name}")
        print("\nRun hyperparameter search to train models:")
        print("  uv run python src/training/hyperparam_search_770m.py")
        sys.exit(1)

    try:
        generate_submission_codet5p_770m(
            args.test_file,
            args.models_dir,
            args.config,
            args.output
        )
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
