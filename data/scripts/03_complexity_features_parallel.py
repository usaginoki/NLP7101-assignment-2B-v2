"""
Code Complexity Feature Extraction - Parallel Version
Extracts Cyclomatic Complexity, Halstead Metrics, LOC, and other complexity metrics
Uses multiprocessing to accelerate extraction
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
from multiprocessing import Pool, cpu_count
from functools import partial


def extract_features_single(code_text):
    """Extract features for a single code sample"""
    extractor = ComplexityFeatureExtractor()
    return extractor.extract(str(code_text))


def extract_complexity_features_parallel(df: pd.DataFrame, code_col: str, n_jobs: int = None) -> pd.DataFrame:
    """
    Extract complexity features for all code samples using parallel processing

    Args:
        df: DataFrame with code
        code_col: Name of the code column
        n_jobs: Number of parallel jobs (default: all CPU cores)

    Returns:
        DataFrame with added complexity features
    """
    if n_jobs is None:
        n_jobs = cpu_count()

    print(f"Extracting complexity features using {n_jobs} CPU cores...")
    print(f"Processing {len(df)} samples...")

    # Extract features with progress bar using multiprocessing
    with Pool(processes=n_jobs) as pool:
        feature_list = list(tqdm(
            pool.imap(extract_features_single, df[code_col], chunksize=100),
            total=len(df),
            desc="Processing code samples"
        ))

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
    Analyze and visualize complexity features (sample only for visualization)
    """
    print(f"\n{'=' * 80}")
    print(f"Complexity Analysis - {split_name.upper()}")
    print(f"{'=' * 80}")

    # Sample for analysis if dataset is large
    df_sample = df
    if len(df) > 10000:
        print(f"\nSampling 10,000 samples for analysis and visualization...")
        df_sample = df.sample(n=10000, random_state=42)

    # Define complexity features
    complexity_features = [
        'loc', 'sloc', 'comments', 'blank',
        'cc_max', 'cc_mean', 'cc_total',
        'halstead_volume', 'halstead_difficulty', 'halstead_effort',
        'maintainability_index'
    ]

    # Filter to only existing columns
    complexity_features = [f for f in complexity_features if f in df.columns]

    # Print summary statistics (use full dataset)
    print("\nComplexity Metrics Summary:")
    print(df[complexity_features].describe().to_string())

    # Print statistics by class (use full dataset)
    print(f"\n\nComplexity Metrics by Model Family:")
    stats_by_class = df.groupby(label_col)[complexity_features].agg(['mean', 'std'])
    print(stats_by_class.to_string())

    # Create visualizations (use sample)
    print(f"\n\nGenerating complexity visualizations (using sample)...")

    # Box plots for key complexity metrics
    key_metrics = ['loc', 'cc_mean', 'halstead_volume', 'maintainability_index']
    for metric in key_metrics:
        if metric in df_sample.columns:
            print(f"  Creating boxplot for {metric}...")
            fig = plot_feature_boxplots(
                df_sample, metric, label_col,
                f"{split_name.title()} - {metric} by Model Family"
            )
            save_figure(fig, f"{split_name}_{metric}_boxplot.png", "figures/complexity")
            plt.close(fig)

    # Correlation heatmap for complexity features
    print(f"  Creating correlation heatmap...")
    fig = plot_correlation_heatmap(
        df_sample, complexity_features,
        f"{split_name.title()} - Complexity Features Correlation"
    )
    save_figure(fig, f"{split_name}_complexity_correlation.png", "figures/correlations")
    plt.close(fig)

    # Scatter plots for interesting relationships
    if 'cc_mean' in df_sample.columns and 'loc' in df_sample.columns:
        print(f"  Creating scatter plot: CC vs LOC...")
        fig = plot_scatter(
            df_sample, 'loc', 'cc_mean', label_col,
            f"{split_name.title()} - Cyclomatic Complexity vs Lines of Code"
        )
        save_figure(fig, f"{split_name}_cc_vs_loc.html", "figures/complexity")

    if 'halstead_difficulty' in df_sample.columns and 'halstead_volume' in df_sample.columns:
        print(f"  Creating scatter plot: Halstead Difficulty vs Volume...")
        fig = plot_scatter(
            df_sample, 'halstead_volume', 'halstead_difficulty', label_col,
            f"{split_name.title()} - Halstead Volume vs Difficulty"
        )
        save_figure(fig, f"{split_name}_halstead_scatter.html", "figures/complexity")

    print("\nComplexity visualizations saved!")


def save_feature_matrix(df: pd.DataFrame, complexity_features: list, split_name: str = "train"):
    """
    Save feature matrix to CSV
    """
    output_dir = create_output_dir("reports")

    # Save full feature matrix
    output_file = output_dir / f"{split_name}_complexity_features.csv"
    df.to_csv(output_file, index=False)
    print(f"\nFeature matrix saved to: {output_file}")

    # Save feature summary
    feature_cols = [col for col in df.columns if col in complexity_features]
    summary_file = output_dir / f"{split_name}_complexity_summary.csv"
    summary = df[feature_cols].describe()
    summary.to_csv(summary_file)
    print(f"Feature summary saved to: {summary_file}")


def main():
    print("=" * 80)
    print("Complexity Feature Extraction (Parallel) - SemEval 2026 Task 13 Subtask B")
    print("=" * 80)
    print()

    # Determine number of CPU cores to use
    n_cores = cpu_count()
    print(f"Using {n_cores} CPU cores for parallel processing")

    # Load training data
    print("\nLoading training data...")
    df_train = load_data("train.parquet")

    # Process all samples (no sampling)
    print(f"\nProcessing all {len(df_train)} samples...")

    # Identify code and label columns
    code_col = get_code_column(df_train)
    label_col = get_label_column(df_train)

    print(f"\nCode column: '{code_col}'")
    print(f"Label column: '{label_col}'")

    # Extract complexity features using parallel processing
    df_train = extract_complexity_features_parallel(df_train, code_col, n_jobs=n_cores)

    # Get list of complexity features
    complexity_feature_cols = [
        'loc', 'lloc', 'sloc', 'comments', 'multi', 'blank', 'single_comments',
        'cc_max', 'cc_mean', 'cc_total', 'cc_count',
        'halstead_vocabulary', 'halstead_length', 'halstead_volume',
        'halstead_difficulty', 'halstead_effort', 'halstead_bugs', 'halstead_time',
        'maintainability_index'
    ]

    # Analyze features (will use sampling for visualization)
    analyze_complexity_features(df_train, label_col, "train")

    # Save feature matrix
    save_feature_matrix(df_train, complexity_feature_cols, "train")

    # Optionally process validation set
    try:
        print("\n" + "=" * 80)
        print("Processing validation data...")
        df_val = load_data("validation.parquet")

        # Process all validation samples (no sampling)
        print(f"Processing all {len(df_val)} validation samples...")

        df_val = extract_complexity_features_parallel(df_val, code_col, n_jobs=n_cores)
        analyze_complexity_features(df_val, label_col, "validation")
        save_feature_matrix(df_val, complexity_feature_cols, "validation")
    except FileNotFoundError:
        print("Validation data not found, skipping...")

    print("\n" + "=" * 80)
    print("Complexity Feature Extraction Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
