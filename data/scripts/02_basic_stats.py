"""
Basic Statistical Analysis of Code Samples
Analyzes code length, token distribution, and basic text statistics
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from modules.data_loader import load_data
from modules.utils import get_code_column, get_label_column, create_output_dir, save_figure
from modules.visualizations import (
    plot_feature_boxplots,
    plot_distribution_histogram,
    plot_violin
)


def extract_basic_stats(df: pd.DataFrame, code_col: str) -> pd.DataFrame:
    """
    Extract basic statistics from code

    Args:
        df: DataFrame with code
        code_col: Name of the code column

    Returns:
        DataFrame with added statistics columns
    """
    print("Extracting basic statistics...")

    # Character count
    df['char_count'] = df[code_col].str.len()

    # Line count
    df['line_count'] = df[code_col].str.count('\n') + 1

    # Word/token count (simple whitespace split)
    df['word_count'] = df[code_col].str.split().str.len()

    # Average line length
    df['avg_line_length'] = df['char_count'] / df['line_count']

    # Number of spaces
    df['space_count'] = df[code_col].str.count(' ')

    # Indentation estimate (number of leading spaces/tabs)
    df['tab_count'] = df[code_col].str.count('\t')

    # Alphanumeric ratio
    df['alphanum_ratio'] = df[code_col].apply(
        lambda x: sum(c.isalnum() for c in x) / len(x) if len(x) > 0 else 0
    )

    # Special character ratio
    df['special_char_ratio'] = df[code_col].apply(
        lambda x: sum(not c.isalnum() and not c.isspace() for c in x) / len(x) if len(x) > 0 else 0
    )

    # Newline ratio
    df['newline_ratio'] = df[code_col].str.count('\n') / df['char_count']

    print("Basic statistics extracted!")

    return df


def analyze_and_visualize(df: pd.DataFrame, label_col: str, split_name: str = "train"):
    """
    Analyze and visualize basic statistics

    Args:
        df: DataFrame with statistics
        label_col: Label column name
        split_name: Dataset split name
    """
    print(f"\n{'=' * 80}")
    print(f"Statistical Analysis - {split_name.upper()}")
    print(f"{'=' * 80}")

    # Define features to analyze
    features = [
        'char_count', 'line_count', 'word_count',
        'avg_line_length', 'alphanum_ratio',
        'special_char_ratio', 'newline_ratio'
    ]

    # Print summary statistics
    print("\nSummary Statistics:")
    print(df[features].describe().to_string())

    # Print statistics by class
    print(f"\n\nStatistics by Model Family:")
    print(df.groupby(label_col)[features].mean().to_string())

    # Create visualizations
    print(f"\n\nGenerating visualizations...")

    # Box plots for key features
    for feature in ['char_count', 'line_count', 'word_count', 'avg_line_length']:
        print(f"  Creating boxplot for {feature}...")
        fig = plot_feature_boxplots(
            df, feature, label_col,
            f"{split_name.title()} - {feature} by Model Family"
        )
        save_figure(fig, f"{split_name}_{feature}_boxplot.png", "figures/distributions")
        plt.close(fig)

    # Histograms
    for feature in ['char_count', 'line_count']:
        print(f"  Creating histogram for {feature}...")
        fig = plot_distribution_histogram(
            df, feature, label_col,
            f"{split_name.title()} - {feature} Distribution"
        )
        save_figure(fig, f"{split_name}_{feature}_histogram.html", "figures/distributions")

    # Violin plots
    for feature in ['alphanum_ratio', 'special_char_ratio']:
        print(f"  Creating violin plot for {feature}...")
        fig = plot_violin(
            df, feature, label_col,
            f"{split_name.title()} - {feature} by Model Family"
        )
        save_figure(fig, f"{split_name}_{feature}_violin.png", "figures/distributions")
        plt.close(fig)

    print("\nVisualizations saved!")


def save_stats_to_csv(df: pd.DataFrame, features: list, split_name: str = "train"):
    """
    Save statistics to CSV file

    Args:
        df: DataFrame with statistics
        features: List of feature columns
        split_name: Dataset split name
    """
    output_dir = create_output_dir("reports")
    output_file = output_dir / f"{split_name}_basic_stats.csv"

    # Include all original columns plus new features
    df.to_csv(output_file, index=False)
    print(f"\nStatistics saved to: {output_file}")


def main():
    print("=" * 80)
    print("Basic Statistical Analysis - SemEval 2026 Task 13 Subtask B")
    print("=" * 80)
    print()

    # Load training data
    print("Loading training data...")
    df_train = load_data("train.parquet")

    # Identify code and label columns
    code_col = get_code_column(df_train)
    label_col = get_label_column(df_train)

    print(f"\nCode column: '{code_col}'")
    print(f"Label column: '{label_col}'")

    # Extract basic statistics
    df_train = extract_basic_stats(df_train, code_col)

    # Analyze and visualize
    analyze_and_visualize(df_train, label_col, "train")

    # Define statistics features
    stat_features = [
        'char_count', 'line_count', 'word_count', 'avg_line_length',
        'space_count', 'tab_count', 'alphanum_ratio',
        'special_char_ratio', 'newline_ratio'
    ]

    # Save statistics
    save_stats_to_csv(df_train, stat_features, "train")

    # Optionally process validation set
    try:
        print("\n" + "=" * 80)
        print("Processing validation data...")
        df_val = load_data("validation.parquet")
        df_val = extract_basic_stats(df_val, code_col)
        analyze_and_visualize(df_val, label_col, "validation")
        save_stats_to_csv(df_val, stat_features, "validation")
    except FileNotFoundError:
        print("Validation data not found, skipping...")

    print("\n" + "=" * 80)
    print("Basic Statistical Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Need matplotlib for some plots
    import matplotlib.pyplot as plt
    main()
