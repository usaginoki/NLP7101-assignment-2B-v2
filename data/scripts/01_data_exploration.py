"""
Initial Data Exploration for SemEval 2026 Task 13 Subtask B
Loads datasets and provides overview statistics
"""

import sys
sys.path.append('.')

import pandas as pd
from modules.data_loader import load_all_datasets, get_dataset_info
from modules.visualizations import plot_class_distribution
from modules.utils import create_output_dir, save_figure


def main():
    print("=" * 80)
    print("SemEval 2026 Task 13 Subtask B - Data Exploration")
    print("=" * 80)
    print()

    # Load all datasets
    print("Loading datasets...")
    datasets = load_all_datasets()
    print()

    # Analyze each dataset
    for split_name, df in datasets.items():
        print(f"\n{'=' * 80}")
        print(f"{split_name.upper()} Dataset Analysis")
        print(f"{'=' * 80}")

        # Get basic info
        info = get_dataset_info(df)

        print(f"\nShape: {info['shape']}")
        print(f"Columns: {info['columns']}")
        print(f"Memory Usage: {info['memory_usage']:.2f} MB")

        print(f"\nData Types:")
        for col, dtype in info['dtypes'].items():
            print(f"  {col}: {dtype}")

        print(f"\nMissing Values:")
        missing_any = False
        for col, count in info['missing_values'].items():
            if count > 0:
                print(f"  {col}: {count}")
                missing_any = True
        if not missing_any:
            print("  None")

        # Display sample rows
        print(f"\nSample Data (first 3 rows):")
        print(df.head(3).to_string())
        print()

        # Check for label column and show distribution
        if 'label_column' in info:
            label_col = info['label_column']
            print(f"\nLabel Column: '{label_col}'")
            print(f"\nClass Distribution:")
            for class_name, count in sorted(info['class_distribution'].items(), key=lambda x: -x[1]):
                percentage = (count / len(df)) * 100
                print(f"  {class_name}: {count:,} ({percentage:.2f}%)")

            # Create visualization
            print(f"\nGenerating class distribution plot...")
            fig = plot_class_distribution(df, label_col, f"{split_name.title()} - Class Distribution")

            output_dir = create_output_dir("figures/distributions")
            save_figure(fig, f"{split_name}_class_distribution.html", "figures/distributions")

        # Check if there's a language column
        possible_lang_cols = ['language', 'lang', 'programming_language']
        for col in possible_lang_cols:
            if col in df.columns:
                print(f"\nLanguage Distribution ('{col}'):")
                lang_dist = df[col].value_counts()
                for lang, count in lang_dist.items():
                    percentage = (count / len(df)) * 100
                    print(f"  {lang}: {count:,} ({percentage:.2f}%)")
                break

        # Check for code column and analyze
        possible_code_cols = ['code', 'text', 'content', 'source', 'snippet']
        for col in possible_code_cols:
            if col in df.columns:
                print(f"\nCode Statistics ('{col}' column):")
                code_lengths = df[col].str.len()
                code_lines = df[col].str.count('\n') + 1

                print(f"  Character count:")
                print(f"    Min: {code_lengths.min():,}")
                print(f"    Max: {code_lengths.max():,}")
                print(f"    Mean: {code_lengths.mean():.2f}")
                print(f"    Median: {code_lengths.median():.2f}")

                print(f"  Line count:")
                print(f"    Min: {code_lines.min()}")
                print(f"    Max: {code_lines.max()}")
                print(f"    Mean: {code_lines.mean():.2f}")
                print(f"    Median: {code_lines.median():.2f}")
                break

    print("\n" + "=" * 80)
    print("Data Exploration Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
