"""
Code Complexity Feature Extraction
Extracts Cyclomatic Complexity, Halstead Metrics, LOC, and other complexity metrics
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from tqdm import tqdm
from modules.data_loader import load_data
from modules.utils import get_code_column, get_label_column, create_output_dir, save_figure
from modules.feature_extractors import ComplexityFeatureExtractor
from modules.visualizations import (
    plot_feature_boxplots,
    plot_correlation_heatmap,
    plot_scatter
)
import matplotlib.pyplot as plt


def extract_complexity_features(df: pd.DataFrame, code_col: str) -> pd.DataFrame:
    """
    Extract complexity features for all code samples

    Args:
        df: DataFrame with code
        code_col: Name of the code column

    Returns:
        DataFrame with added complexity features
    """
    print("Extracting complexity features...")
    print("This may take a while for large datasets...")

    extractor = ComplexityFeatureExtractor()

    # Extract features with progress bar
    feature_list = []
    for code in tqdm(df[code_col], desc="Processing code samples"):
        features = extractor.extract(str(code))
        feature_list.append(features)

    # Create DataFrame from features
    features_df = pd.DataFrame(feature_list)

    # Combine with original DataFrame
    df_with_features = pd.concat([df.reset_index(drop=True), features_df], axis=1)

    print(f"\nExtracted {len(features_df.columns)} complexity features!")
    print(f"Feature columns: {list(features_df.columns)}")

    # Report extraction errors
    if 'extraction_error' in features_df.columns:
        error_count = features_df['extraction_error'].sum()
        error_pct = (error_count / len(df)) * 100
        print(f"\nExtraction errors: {error_count} ({error_pct:.2f}%)")

    return df_with_features


def analyze_complexity_features(df: pd.DataFrame, label_col: str, split_name: str = "train"):
    """
    Analyze and visualize complexity features

    Args:
        df: DataFrame with complexity features
        label_col: Label column name
        split_name: Dataset split name
    """
    print(f"\n{'=' * 80}")
    print(f"Complexity Analysis - {split_name.upper()}")
    print(f"{'=' * 80}")

    # Define complexity features
    complexity_features = [
        'loc', 'sloc', 'comments', 'blank',
        'cc_max', 'cc_mean', 'cc_total',
        'halstead_volume', 'halstead_difficulty', 'halstead_effort',
        'maintainability_index'
    ]

    # Filter to only existing columns
    complexity_features = [f for f in complexity_features if f in df.columns]

    # Print summary statistics
    print("\nComplexity Metrics Summary:")
    print(df[complexity_features].describe().to_string())

    # Print statistics by class
    print(f"\n\nComplexity Metrics by Model Family:")
    stats_by_class = df.groupby(label_col)[complexity_features].agg(['mean', 'std'])
    print(stats_by_class.to_string())

    # Create visualizations
    print(f"\n\nGenerating complexity visualizations...")

    # Box plots for key complexity metrics
    key_metrics = ['loc', 'cc_mean', 'halstead_volume', 'maintainability_index']
    for metric in key_metrics:
        if metric in df.columns:
            print(f"  Creating boxplot for {metric}...")
            fig = plot_feature_boxplots(
                df, metric, label_col,
                f"{split_name.title()} - {metric} by Model Family"
            )
            save_figure(fig, f"{split_name}_{metric}_boxplot.png", "figures/complexity")
            plt.close(fig)

    # Correlation heatmap for complexity features
    print(f"  Creating correlation heatmap...")
    fig = plot_correlation_heatmap(
        df, complexity_features,
        f"{split_name.title()} - Complexity Features Correlation"
    )
    save_figure(fig, f"{split_name}_complexity_correlation.png", "figures/correlations")
    plt.close(fig)

    # Scatter plots for interesting relationships
    if 'cc_mean' in df.columns and 'loc' in df.columns:
        print(f"  Creating scatter plot: CC vs LOC...")
        fig = plot_scatter(
            df, 'loc', 'cc_mean', label_col,
            f"{split_name.title()} - Cyclomatic Complexity vs Lines of Code"
        )
        save_figure(fig, f"{split_name}_cc_vs_loc.html", "figures/complexity")

    if 'halstead_difficulty' in df.columns and 'halstead_volume' in df.columns:
        print(f"  Creating scatter plot: Halstead Difficulty vs Volume...")
        fig = plot_scatter(
            df, 'halstead_volume', 'halstead_difficulty', label_col,
            f"{split_name.title()} - Halstead Volume vs Difficulty"
        )
        save_figure(fig, f"{split_name}_halstead_scatter.html", "figures/complexity")

    print("\nComplexity visualizations saved!")


def save_feature_matrix(df: pd.DataFrame, complexity_features: list, split_name: str = "train"):
    """
    Save feature matrix to CSV

    Args:
        df: DataFrame with features
        complexity_features: List of complexity feature columns
        split_name: Dataset split name
    """
    output_dir = create_output_dir("reports")

    # Select relevant columns
    feature_cols = [col for col in df.columns if col in complexity_features]
    output_file = output_dir / f"{split_name}_complexity_features.csv"

    # Save full feature matrix
    df.to_csv(output_file, index=False)
    print(f"\nFeature matrix saved to: {output_file}")

    # Save feature summary
    summary_file = output_dir / f"{split_name}_complexity_summary.csv"
    summary = df[feature_cols].describe()
    summary.to_csv(summary_file)
    print(f"Feature summary saved to: {summary_file}")


def main():
    print("=" * 80)
    print("Complexity Feature Extraction - SemEval 2026 Task 13 Subtask B")
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

    # Extract complexity features
    df_train = extract_complexity_features(df_train, code_col)

    # Get list of complexity features
    complexity_feature_cols = [
        'loc', 'lloc', 'sloc', 'comments', 'multi', 'blank', 'single_comments',
        'cc_max', 'cc_mean', 'cc_total', 'cc_count',
        'halstead_vocabulary', 'halstead_length', 'halstead_volume',
        'halstead_difficulty', 'halstead_effort', 'halstead_bugs', 'halstead_time',
        'maintainability_index'
    ]

    # Analyze features
    analyze_complexity_features(df_train, label_col, "train")

    # Save feature matrix
    save_feature_matrix(df_train, complexity_feature_cols, "train")

    # Optionally process validation set
    try:
        print("\n" + "=" * 80)
        print("Processing validation data...")
        df_val = load_data("validation.parquet")

        # Sample validation set too if large
        if len(df_val) > 5000:
            print(f"Sampling 5,000 from validation set...")
            df_val = df_val.sample(n=5000, random_state=42).reset_index(drop=True)

        df_val = extract_complexity_features(df_val, code_col)
        analyze_complexity_features(df_val, label_col, "validation")
        save_feature_matrix(df_val, complexity_feature_cols, "validation")
    except FileNotFoundError:
        print("Validation data not found, skipping...")

    print("\n" + "=" * 80)
    print("Complexity Feature Extraction Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
