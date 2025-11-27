"""
Generate submission.csv for competition
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src and data to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../data'))

from src.models.pipeline import TwoStagePipeline
from modules.feature_extractors import ComplexityFeatureExtractor, LexicalFeatureExtractor


def extract_basic_features(df):
    """
    Extract basic statistical features

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
    Extract code complexity features using radon

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
    Extract lexical diversity features

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


def generate_submission(test_file='data/test.parquet', models_dir='models', output_file='outputs/submission.csv'):
    """
    Generate submission CSV for test set

    Submission format:
    - CSV with 'label' column
    - Values: 0-10 (0=Human, 1-10=AI families)
    - Row i = prediction for test sample i

    Args:
        test_file: Path to test parquet file
        models_dir: Directory containing trained models
        output_file: Path to save submission CSV
    """
    print("\n" + "="*60)
    print("GENERATING COMPETITION SUBMISSION")
    print("="*60 + "\n")

    # Load pipeline
    print("Step 1/5: Loading trained pipeline...")
    pipeline = TwoStagePipeline.load(models_dir)

    # Load test data
    print(f"\nStep 2/5: Loading test data from {test_file}...")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")

    test_df = pd.read_parquet(test_file)
    print(f"  Loaded {len(test_df)} test samples")
    print(f"  Columns: {list(test_df.columns)}")

    # Extract features
    print("\nStep 3/5: Extracting features from test data...")
    print("  (This may take a few minutes...)")

    # Basic features
    test_df = extract_basic_features(test_df)

    # Complexity features
    test_df = extract_complexity_features(test_df)

    # Lexical features
    test_df = extract_lexical_features(test_df)

    # Add dummy label for processor
    test_df['label'] = -1

    print(f"\n  Extracted {len(test_df.columns)} total columns")

    # Transform features
    print("\nStep 4/5: Preprocessing features...")
    X_test, _ = pipeline.processor.transform(test_df)
    print(f"  Feature matrix shape: {X_test.shape}")

    # Generate predictions
    print("\nStep 5/5: Generating predictions...")
    predictions = pipeline.predict(X_test)

    # Show prediction distribution
    print(f"\nPrediction distribution:")
    unique, counts = np.unique(predictions, return_counts=True)
    for label, count in zip(unique, counts):
        pct = 100 * count / len(predictions)
        name = 'Human' if label == 0 else f'AI-{label}'
        print(f"  {name:8s}: {count:4d} ({pct:5.1f}%)")

    # Save submission
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    submission_df = pd.DataFrame({'label': predictions})
    submission_df.to_csv(output_file, index=False)

    print(f"\n✓ Submission saved to: {output_file}")
    print(f"✓ Total predictions: {len(predictions)}")
    print(f"✓ File size: {os.path.getsize(output_file)} bytes")

    # Validation checks
    print("\nValidation checks:")
    print(f"  ✓ Row count: {len(submission_df)} (expected: {len(test_df)})")
    print(f"  ✓ Column: 'label' present")
    print(f"  ✓ Value range: {predictions.min()}-{predictions.max()} (expected: 0-10)")
    print(f"  ✓ Data type: {predictions.dtype}")

    print("\n" + "="*60)
    print("Submission generation complete!")
    print("="*60 + "\n")

    return submission_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate competition submission file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate submission with default settings
  python scripts/generate_submission.py

  # Use custom test file
  python scripts/generate_submission.py --test-file custom/test.parquet

  # Save to custom location
  python scripts/generate_submission.py --output custom/submission.csv
        """
    )

    parser.add_argument('--test-file', default='data/test.parquet',
                       help='Path to test parquet file')
    parser.add_argument('--models-dir', default='models',
                       help='Directory containing trained models')
    parser.add_argument('--output', default='outputs/submission.csv',
                       help='Path to save submission CSV')

    args = parser.parse_args()

    # Verify models exist
    if not os.path.exists(f'{args.models_dir}/stage1/classifier.pkl'):
        print(f"ERROR: Trained models not found in {args.models_dir}/")
        print("\nPlease train the models first by running:")
        print("  python scripts/train_pipeline.py")
        sys.exit(1)

    try:
        generate_submission(args.test_file, args.models_dir, args.output)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
