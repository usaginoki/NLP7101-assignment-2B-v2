"""
Feature loader for merging precomputed feature CSVs
"""

import pandas as pd
from pathlib import Path
from typing import Optional


class FeatureLoader:
    """Load and merge precomputed features from CSV files"""

    def __init__(self, data_dir='data/reports'):
        self.data_dir = Path(data_dir)

    def load_features(self, split='train') -> pd.DataFrame:
        """
        Load and merge all precomputed features for a given split

        Merge strategy:
        1. Load basic_stats.csv as base (has all samples)
        2. Left merge complexity_features.csv on ['code', 'generator', 'label', 'language']
        3. Left merge lexical_features.csv similarly
        4. Fill NaN with 0 (represents sampling or extraction failures)
        5. Return unified DataFrame with 36+ features

        Args:
            split: One of 'train', 'validation', 'test'

        Returns:
            DataFrame with all features merged
        """
        print(f"\n{'='*60}")
        print(f"Loading features for {split} split")
        print(f"{'='*60}")

        # Load basic stats (base dataset with all samples)
        basic_stats_path = self.data_dir / f"{split}_basic_stats.csv"
        print(f"Loading basic stats from: {basic_stats_path}")

        if not basic_stats_path.exists():
            raise FileNotFoundError(f"Basic stats file not found: {basic_stats_path}")

        df = pd.read_csv(basic_stats_path)
        print(f"  Loaded {len(df)} samples with {len(df.columns)} columns")

        # Define merge keys
        merge_keys = ['code', 'generator', 'language']
        if 'label' in df.columns:
            merge_keys.append('label')

        # Load and merge complexity features
        complexity_path = self.data_dir / f"{split}_complexity_features.csv"
        if complexity_path.exists():
            print(f"Loading complexity features from: {complexity_path}")
            complexity_df = pd.read_csv(complexity_path)
            print(f"  Loaded {len(complexity_df)} samples")

            # Get feature columns (exclude merge keys)
            complexity_feature_cols = [col for col in complexity_df.columns
                                      if col not in merge_keys]

            # Merge
            df = df.merge(
                complexity_df,
                on=merge_keys,
                how='left',
                suffixes=('', '_complexity')
            )
            print(f"  Merged {len(complexity_feature_cols)} complexity features")
        else:
            print(f"  Warning: Complexity features not found at {complexity_path}")

        # Load and merge lexical features
        lexical_path = self.data_dir / f"{split}_lexical_features.csv"
        if lexical_path.exists():
            print(f"Loading lexical features from: {lexical_path}")
            lexical_df = pd.read_csv(lexical_path)
            print(f"  Loaded {len(lexical_df)} samples")

            # Get feature columns (exclude merge keys)
            lexical_feature_cols = [col for col in lexical_df.columns
                                   if col not in merge_keys]

            # Merge
            df = df.merge(
                lexical_df,
                on=merge_keys,
                how='left',
                suffixes=('', '_lexical')
            )
            print(f"  Merged {len(lexical_feature_cols)} lexical features")
        else:
            print(f"  Warning: Lexical features not found at {lexical_path}")

        # Fill missing values with 0
        initial_na_count = df.isna().sum().sum()
        if initial_na_count > 0:
            print(f"  Filling {initial_na_count} missing values with 0")
            df = df.fillna(0)

        print(f"\nFinal dataset shape: {df.shape}")
        print(f"Total features: {len(df.columns)}")
        print(f"{'='*60}\n")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Get list of feature columns (exclude metadata)

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        metadata_cols = ['code', 'generator', 'label', 'language']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        return feature_cols

    def load_feature_info(self, split='train') -> dict:
        """
        Load features and return metadata info

        Args:
            split: Dataset split to load

        Returns:
            Dictionary with dataset info
        """
        df = self.load_features(split)
        feature_cols = self.get_feature_columns(df)

        info = {
            'split': split,
            'n_samples': len(df),
            'n_features': len(feature_cols),
            'feature_columns': feature_cols,
            'has_labels': 'label' in df.columns
        }

        if 'label' in df.columns:
            info['class_distribution'] = df['label'].value_counts().to_dict()
            info['n_classes'] = df['label'].nunique()

        return info
