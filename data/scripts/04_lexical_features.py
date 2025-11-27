"""
Lexical Feature Extraction
Extracts lexical diversity, token statistics, and language-specific patterns
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from tqdm import tqdm
from modules.data_loader import load_data
from modules.utils import get_code_column, get_label_column, create_output_dir, save_figure
from modules.feature_extractors import LexicalFeatureExtractor
from modules.visualizations import (
    plot_feature_boxplots,
    plot_violin,
    plot_correlation_heatmap
)
import matplotlib.pyplot as plt


def extract_lexical_features(df: pd.DataFrame, code_col: str) -> pd.DataFrame:
    """
    Extract lexical features for all code samples

    Args:
        df: DataFrame with code
        code_col: Name of the code column

    Returns:
        DataFrame with added lexical features
    """
    print("Extracting lexical features...")
    print("This may take a while for large datasets...")

    extractor = LexicalFeatureExtractor()

    # Extract features with progress bar
    feature_list = []
    for code in tqdm(df[code_col], desc="Processing code samples"):
        features = extractor.extract(str(code))
        feature_list.append(features)

    # Create DataFrame from features
    features_df = pd.DataFrame(feature_list)

    # Combine with original DataFrame
    df_with_features = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    print(f"\nExtracted {len(features_df.columns)} lexical features!")
    print(f"Feature columns: {list(features_df.columns)}")

    # Report extraction errors
    if 'lex_extraction_error' in features_df.columns:
        error_count = features_df['lex_extraction_error'].sum()
        error_pct = (error_count / len(df)) * 100
        print(f"\nExtraction errors: {error_count} ({error_pct:.2f}%)")

    return df_with_features


def analyze_lexical_features(df: pd.DataFrame, label_col: str, split_name: str = "train"):
    """
    Analyze and visualize lexical features

    Args:
        df: DataFrame with lexical features
        label_col: Label column name
        split_name: Dataset split name
    """
    print(f"\n{'=' * 80}")
    print(f"Lexical Analysis - {split_name.upper()}")
    print(f"{'=' * 80}")

    # Define lexical features
    lexical_features = [
        'token_count', 'unique_tokens', 'avg_token_length',
        'ttr', 'mtld', 'mattr',
        'keyword_count', 'keyword_ratio'
    ]

    # Filter to only existing columns
    lexical_features = [f for f in lexical_features if f in df.columns]

    # Print summary statistics
    print("\nLexical Metrics Summary:")
    print(df[lexical_features].describe().to_string())

    # Print statistics by class
    print(f"\n\nLexical Metrics by Model Family:")
    stats_by_class = df.groupby(label_col)[lexical_features].agg(['mean', 'std'])
    print(stats_by_class.to_string())

    # Analyze lexical diversity specifically
    print(f"\n\nLexical Diversity Analysis:")
    diversity_metrics = ['ttr', 'mtld', 'mattr']
    diversity_metrics = [m for m in diversity_metrics if m in df.columns]

    for metric in diversity_metrics:
        print(f"\n{metric.upper()}:")
        for model_family in df[label_col].unique():
            family_data = df[df[label_col] == model_family][metric]
            # Filter out zeros for valid samples
            valid_data = family_data[family_data > 0]
            if len(valid_data) > 0:
                print(f"  {model_family}: {valid_data.mean():.4f} ± {valid_data.std():.4f}")

    # Create visualizations
    print(f"\n\nGenerating lexical visualizations...")

    # Box plots for key lexical metrics
    key_metrics = ['token_count', 'unique_tokens', 'ttr', 'keyword_ratio']
    for metric in key_metrics:
        if metric in df.columns:
            print(f"  Creating boxplot for {metric}...")
            fig = plot_feature_boxplots(
                df, metric, label_col,
                f"{split_name.title()} - {metric} by Model Family"
            )
            save_figure(fig, f"{split_name}_{metric}_boxplot.png", "figures/lexical")
            plt.close(fig)

    # Violin plots for diversity metrics
    for metric in diversity_metrics:
        if metric in df.columns:
            print(f"  Creating violin plot for {metric}...")
            # Filter out zeros for better visualization
            df_filtered = df[df[metric] > 0].copy()
            if len(df_filtered) > 0:
                fig = plot_violin(
                    df_filtered, metric, label_col,
                    f"{split_name.title()} - {metric} Distribution by Model Family"
                )
                save_figure(fig, f"{split_name}_{metric}_violin.png", "figures/lexical")
                plt.close(fig)

    # Correlation heatmap for lexical features
    print(f"  Creating correlation heatmap...")
    fig = plot_correlation_heatmap(
        df, lexical_features,
        f"{split_name.title()} - Lexical Features Correlation"
    )
    save_figure(fig, f"{split_name}_lexical_correlation.png", "figures/correlations")
    plt.close(fig)

    print("\nLexical visualizations saved!")


def compare_lexical_patterns(df: pd.DataFrame, label_col: str, split_name: str = "train"):
    """
    Compare lexical patterns across model families

    Args:
        df: DataFrame with lexical features
        label_col: Label column name
        split_name: Dataset split name
    """
    print(f"\n{'=' * 80}")
    print("Lexical Pattern Comparison Across Model Families")
    print(f"{'=' * 80}")

    # Calculate vocabulary diversity per model family
    print("\nVocabulary Diversity:")
    for model_family in sorted(df[label_col].unique()):
        family_data = df[df[label_col] == model_family]
        avg_unique_tokens = family_data['unique_tokens'].mean()
        avg_total_tokens = family_data['token_count'].mean()
        avg_ttr = family_data['ttr'].mean()

        print(f"\n{model_family}:")
        print(f"  Avg unique tokens: {avg_unique_tokens:.2f}")
        print(f"  Avg total tokens: {avg_total_tokens:.2f}")
        print(f"  Avg TTR: {avg_ttr:.4f}")

    # Keyword usage patterns
    print(f"\n\nKeyword Usage Patterns:")
    for model_family in sorted(df[label_col].unique()):
        family_data = df[df[label_col] == model_family]
        avg_keyword_count = family_data['keyword_count'].mean()
        avg_keyword_ratio = family_data['keyword_ratio'].mean()

        print(f"\n{model_family}:")
        print(f"  Avg keyword count: {avg_keyword_count:.2f}")
        print(f"  Avg keyword ratio: {avg_keyword_ratio:.4f}")


def save_feature_matrix(df: pd.DataFrame, lexical_features: list, split_name: str = "train"):
    """
    Save lexical feature matrix to CSV

    Args:
        df: DataFrame with features
        lexical_features: List of lexical feature columns
        split_name: Dataset split name
    """
    output_dir = create_output_dir("reports")

    # Save full feature matrix
    output_file = output_dir / f"{split_name}_lexical_features.csv"
    df.to_csv(output_file, index=False)
    print(f"\nFeature matrix saved to: {output_file}")

    # Save feature summary
    feature_cols = [col for col in df.columns if col in lexical_features]
    summary_file = output_dir / f"{split_name}_lexical_summary.csv"
    summary = df[feature_cols].describe()
    summary.to_csv(summary_file)
    print(f"Feature summary saved to: {summary_file}")


def main():
    print("=" * 80)
    print("Lexical Feature Extraction - SemEval 2026 Task 13 Subtask B")
    print("=" * 80)
    print()

    # Load training data
    print("Loading training data...")
    df_train = load_data("train.parquet")

    # Sample for faster processing if dataset is large
    if len(df_train) > 10000:
        print(f"\nDataset is large ({len(df_train)} samples).")
        print("Processing first 10,000 samples for EDA...")
        df_train = df_train.sample(n=10000, random_state=42).reset_index(drop=True)

    # Identify code and label columns
    code_col = get_code_column(df_train)
    label_col = get_label_column(df_train)

    print(f"\nCode column: '{code_col}'")
    print(f"Label column: '{label_col}'")

    # Extract lexical features
    df_train = extract_lexical_features(df_train, code_col)

    # Get list of lexical features
    lexical_feature_cols = [
        'token_count', 'unique_tokens', 'avg_token_length',
        'ttr', 'mtld', 'mattr',
        'keyword_count', 'keyword_ratio'
    ]

    # Analyze features
    analyze_lexical_features(df_train, label_col, "train")

    # Compare patterns across model families
    compare_lexical_patterns(df_train, label_col, "train")

    # Save feature matrix
    save_feature_matrix(df_train, lexical_feature_cols, "train")

    # Optionally process validation set
    try:
        print("\n" + "=" * 80)
        print("Processing validation data...")
        df_val = load_data("validation.parquet")

        # Sample validation set too if large
        if len(df_val) > 5000:
            print(f"Sampling 5,000 from validation set...")
            df_val = df_val.sample(n=5000, random_state=42).reset_index(drop=True)

        df_val = extract_lexical_features(df_val, code_col)
        analyze_lexical_features(df_val, label_col, "validation")
        compare_lexical_patterns(df_val, label_col, "validation")
        save_feature_matrix(df_val, lexical_feature_cols, "validation")
    except FileNotFoundError:
        print("Validation data not found, skipping...")

    print("\n" + "=" * 80)
    print("Lexical Feature Extraction Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
