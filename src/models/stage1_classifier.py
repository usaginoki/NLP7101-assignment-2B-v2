"""
Stage 1: Binary Classification (Human vs AI-generated)
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


class Stage1Classifier:
    """Binary classifier: Human (0) vs AI-generated (1-10)"""

    def __init__(self, **kwargs):
        """
        Initialize Stage 1 binary classifier

        Args:
            **kwargs: Additional parameters for RandomForestClassifier
        """
        # Default parameters optimized for binary classification
        default_params = {
            'n_estimators': 200,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'class_weight': 'balanced',  # KEY: Handle 88% vs 12% imbalance
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1
        }

        # Override with user-provided kwargs
        default_params.update(kwargs)

        self.model = RandomForestClassifier(**default_params)
        self.is_fitted = False

    @staticmethod
    def prepare_labels(y: np.ndarray) -> np.ndarray:
        """
        Convert multi-class labels (0-10) to binary (0=Human, 1=AI)

        Args:
            y: Original labels (0-10)

        Returns:
            Binary labels (0=Human, 1=AI)
        """
        return (y > 0).astype(int)

    def fit(self, X, y):
        """
        Train binary classifier

        Args:
            X: Feature matrix
            y: Original labels (0-10) - will be converted to binary

        Returns:
            self
        """
        # Convert to binary labels
        y_binary = self.prepare_labels(y)

        # Count class distribution
        unique, counts = np.unique(y_binary, return_counts=True)
        print("\n" + "="*60)
        print("Training Stage 1: Binary Classifier (Human vs AI)")
        print("="*60)
        print(f"Training samples: {len(y_binary)}")
        print(f"Class distribution:")
        print(f"  Human (0): {counts[0]} ({100*counts[0]/len(y_binary):.2f}%)")
        if len(counts) > 1:
            print(f"  AI (1):    {counts[1]} ({100*counts[1]/len(y_binary):.2f}%)")
        print("="*60)

        # Train model
        self.model.fit(X, y_binary)
        self.is_fitted = True

        print("\n✓ Stage 1 training complete!")

        return self

    def predict(self, X):
        """
        Predict binary labels

        Args:
            X: Feature matrix

        Returns:
            Binary predictions (0=Human, 1=AI)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities

        Args:
            X: Feature matrix

        Returns:
            Probability matrix [P(Human), P(AI)]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_proba(X)

    def get_feature_importances(self):
        """
        Get feature importances from Random Forest

        Returns:
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.feature_importances_

    def save(self, filepath: str):
        """
        Save model to disk

        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(self.model, filepath)
        print(f"✓ Saved Stage 1 classifier to: {filepath}")

    def load(self, filepath: str):
        """
        Load model from disk

        Args:
            filepath: Path to saved model
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = joblib.load(filepath)
        self.is_fitted = True
        print(f"✓ Loaded Stage 1 classifier from: {filepath}")

    def __str__(self):
        return f"Stage1Classifier(n_estimators={self.model.n_estimators}, fitted={self.is_fitted})"
