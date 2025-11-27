"""
Stage 2: Multi-Class Classification (AI Model Families)
"""

import numpy as np
import joblib
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path


class Stage2ClassifierBase(ABC):
    """Abstract base class for Stage 2 classifiers (extensibility for CodeT5)"""

    @abstractmethod
    def fit(self, X, y):
        """Train the classifier"""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict AI family labels"""
        pass

    @abstractmethod
    def save(self, filepath: str):
        """Save model to disk"""
        pass

    @abstractmethod
    def load(self, filepath: str):
        """Load model from disk"""
        pass


class Stage2RandomForest(Stage2ClassifierBase):
    """
    Stage 2 classifier using Random Forest on engineered features
    Predicts AI model family (labels 1-10) for AI-generated code
    """

    def __init__(self, **kwargs):
        """
        Initialize Stage 2 Random Forest classifier

        Args:
            **kwargs: Additional parameters for RandomForestClassifier
        """
        # Default parameters optimized for multi-class classification
        default_params = {
            'n_estimators': 300,
            'max_depth': 25,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'class_weight': 'balanced',  # Handle imbalance across 10 AI families
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1
        }

        # Override with user-provided kwargs
        default_params.update(kwargs)

        self.model = RandomForestClassifier(**default_params)
        self.is_fitted = False

    def fit(self, X, y):
        """
        Train on AI samples only (labels 1-10)

        Args:
            X: Feature matrix
            y: Original labels (0-10) - will filter for AI samples only

        Returns:
            self
        """
        # Filter for AI samples only (y > 0)
        ai_mask = y > 0
        X_ai = X[ai_mask]
        y_ai = y[ai_mask]

        if len(y_ai) == 0:
            raise ValueError("No AI samples found in training data")

        # Count class distribution
        unique, counts = np.unique(y_ai, return_counts=True)
        print("\n" + "="*60)
        print("Training Stage 2: Multi-Class Classifier (AI Families)")
        print("="*60)
        print(f"Training samples (AI only): {len(y_ai)}")
        print(f"AI families: {len(unique)}")
        print(f"Class distribution:")
        for label, count in zip(unique, counts):
            print(f"  AI-{label}: {count} ({100*count/len(y_ai):.2f}%)")
        print("="*60)

        # Train model
        self.model.fit(X_ai, y_ai)
        self.is_fitted = True

        print("\n✓ Stage 2 training complete!")

        return self

    def predict(self, X):
        """
        Predict AI family (1-10) for given samples

        Args:
            X: Feature matrix

        Returns:
            AI family predictions (1-10)
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
            Probability matrix for AI families
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
        print(f"✓ Saved Stage 2 classifier to: {filepath}")

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
        print(f"✓ Loaded Stage 2 classifier from: {filepath}")

    def __str__(self):
        return f"Stage2RandomForest(n_estimators={self.model.n_estimators}, fitted={self.is_fitted})"


class Stage2CodeT5(Stage2ClassifierBase):
    """
    Future implementation: CodeT5 embeddings + neural classification head

    Architecture:
    - CodeT5-base encoder (frozen) → 768-dim embeddings
    - Classification head: Linear(768, 256) → ReLU → Dropout(0.3) → Linear(256, 10)
    - Softmax output for 10 AI families

    Requires: torch, transformers (not yet installed)
    """

    def __init__(self, freeze_encoder=True):
        raise NotImplementedError(
            "Stage2CodeT5 requires torch and transformers.\n"
            "Install with: uv add torch>=2.0.0 transformers>=4.40.0\n"
            "Then implement the CodeT5 encoder and classification head."
        )

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def save(self, filepath: str):
        pass

    def load(self, filepath: str):
        pass


def create_stage2_classifier(classifier_type='random_forest', **kwargs):
    """
    Factory function for creating Stage 2 classifiers

    Args:
        classifier_type: Type of classifier ('random_forest' or 'codet5')
        **kwargs: Additional parameters for the classifier

    Returns:
        Stage2ClassifierBase instance
    """
    if classifier_type == 'random_forest':
        return Stage2RandomForest(**kwargs)
    elif classifier_type == 'codet5':
        return Stage2CodeT5(**kwargs)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}. "
                        f"Choose from ['random_forest', 'codet5']")
