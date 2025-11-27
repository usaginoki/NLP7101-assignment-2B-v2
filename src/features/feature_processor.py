"""
Feature preprocessing and scaling
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
from pathlib import Path


class FeatureProcessor:
    """Preprocess and scale features for training"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metadata_cols = ['code', 'generator', 'label', 'language']
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit scaler on training data and transform

        Steps:
        1. Separate metadata from features
        2. Store feature names
        3. Handle inf/nan: np.nan_to_num()
        4. Fit StandardScaler on training data
        5. Transform features
        6. Return X, y

        Args:
            df: DataFrame with features and labels

        Returns:
            Tuple of (X, y) as numpy arrays
        """
        print("\n" + "="*60)
        print("Fitting feature processor on training data")
        print("="*60)

        # Separate features from metadata
        feature_cols = [col for col in df.columns if col not in self.metadata_cols]
        self.feature_names = feature_cols

        print(f"Total features: {len(feature_cols)}")

        # Extract features
        X = df[feature_cols].values

        # Handle inf/nan
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("  Handling NaN/Inf values...")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Fit and transform
        print("  Fitting StandardScaler...")
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True

        # Extract labels
        if 'label' in df.columns:
            y = df['label'].values
        else:
            # For test set without labels
            y = np.full(len(df), -1, dtype=int)

        print(f"  Scaled features shape: {X_scaled.shape}")
        print(f"  Labels shape: {y.shape}")

        if 'label' in df.columns:
            print(f"  Label distribution: {np.bincount(y)}")

        print("="*60 + "\n")

        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform using fitted scaler (for validation/test)

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (X, y) as numpy arrays
        """
        if not self.is_fitted:
            raise ValueError("FeatureProcessor must be fitted before transform. Use fit_transform() first.")

        print(f"\nTransforming features using fitted scaler...")

        # Extract features using stored feature names
        X = df[self.feature_names].values

        # Handle inf/nan
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            print("  Handling NaN/Inf values...")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Transform
        X_scaled = self.scaler.transform(X)

        # Extract labels
        if 'label' in df.columns:
            y = df['label'].values
        else:
            # For test set without labels
            y = np.full(len(df), -1, dtype=int)

        print(f"  Transformed features shape: {X_scaled.shape}")

        return X_scaled, y

    def save(self, filepath: str):
        """
        Save scaler and feature names

        Args:
            filepath: Path to save the processor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted FeatureProcessor")

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save as dictionary
        processor_data = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metadata_cols': self.metadata_cols,
            'is_fitted': self.is_fitted
        }

        joblib.dump(processor_data, filepath)
        print(f"✓ Saved FeatureProcessor to: {filepath}")

    def load(self, filepath: str):
        """
        Load scaler and feature names

        Args:
            filepath: Path to saved processor
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Processor file not found: {filepath}")

        # Load dictionary
        processor_data = joblib.load(filepath)

        self.scaler = processor_data['scaler']
        self.feature_names = processor_data['feature_names']
        self.metadata_cols = processor_data['metadata_cols']
        self.is_fitted = processor_data['is_fitted']

        print(f"✓ Loaded FeatureProcessor from: {filepath}")
        print(f"  Features: {len(self.feature_names)}")

    def get_feature_names(self) -> list:
        """Get list of feature names"""
        return self.feature_names

    def get_feature_importance_names(self, importances: np.ndarray, top_k: int = 20) -> list:
        """
        Get top-k feature names sorted by importance

        Args:
            importances: Feature importance scores
            top_k: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if self.feature_names is None:
            raise ValueError("Feature names not available")

        if len(importances) != len(self.feature_names):
            raise ValueError(f"Importances length ({len(importances)}) != features ({len(self.feature_names)})")

        # Create list of (name, importance) tuples
        feature_importance = list(zip(self.feature_names, importances))

        # Sort by importance descending
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        return feature_importance[:top_k]
