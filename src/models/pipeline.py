"""
2-Stage Classification Pipeline
Orchestrates binary → multi-class classification
"""

import numpy as np
from sklearn.metrics import f1_score, classification_report, accuracy_score
from pathlib import Path

from .stage1_classifier import Stage1Classifier
from .stage2_classifier import create_stage2_classifier
from ..features.feature_processor import FeatureProcessor


class TwoStagePipeline:
    """Orchestrate 2-stage cascaded classification pipeline"""

    def __init__(self, stage1_clf, stage2_clf, processor):
        """
        Initialize pipeline with both classifiers and feature processor

        Args:
            stage1_clf: Stage 1 binary classifier (Human vs AI)
            stage2_clf: Stage 2 multi-class classifier (AI families)
            processor: Feature processor for scaling
        """
        self.stage1 = stage1_clf
        self.stage2 = stage2_clf
        self.processor = processor

    def predict(self, X):
        """
        Predict final labels using 2-stage pipeline

        Pipeline logic:
        1. Stage 1: Predict binary (0=Human, 1=AI)
        2. If Human (0), return label 0
        3. If AI (1), pass to Stage 2 → predict family (1-10)
        4. Return combined predictions

        Args:
            X: Feature matrix

        Returns:
            Final predictions (0-10)
        """
        # Stage 1: Binary classification
        stage1_preds = self.stage1.predict(X)  # 0 or 1

        # Initialize final predictions with all Human (0)
        final_preds = np.zeros(len(X), dtype=int)

        # Find samples predicted as AI
        ai_mask = stage1_preds == 1
        n_ai = np.sum(ai_mask)

        if n_ai > 0:
            # Stage 2: Classify AI families
            X_ai = X[ai_mask]
            stage2_preds = self.stage2.predict(X_ai)  # 1-10

            # Insert Stage 2 predictions back into final array
            final_preds[ai_mask] = stage2_preds

        return final_preds

    def predict_with_probabilities(self, X):
        """
        Predict with probabilities from both stages

        Args:
            X: Feature matrix

        Returns:
            Tuple of (predictions, stage1_probs, stage2_probs_dict)
        """
        # Stage 1 probabilities
        stage1_probs = self.stage1.predict_proba(X)  # [P(Human), P(AI)]
        stage1_preds = self.stage1.predict(X)

        # Initialize
        final_preds = np.zeros(len(X), dtype=int)
        stage2_probs_dict = {}

        # Find AI samples
        ai_mask = stage1_preds == 1

        if np.any(ai_mask):
            X_ai = X[ai_mask]
            stage2_preds = self.stage2.predict(X_ai)
            stage2_probs = self.stage2.predict_proba(X_ai)

            final_preds[ai_mask] = stage2_preds
            stage2_probs_dict['predictions'] = stage2_preds
            stage2_probs_dict['probabilities'] = stage2_probs

        return final_preds, stage1_probs, stage2_probs_dict

    def evaluate(self, X, y_true):
        """
        Evaluate complete 2-stage pipeline

        Computes metrics for:
        - Stage 1 (binary classification)
        - Stage 2 (multi-class on AI samples)
        - Overall pipeline (11-class)

        Args:
            X: Feature matrix
            y_true: True labels (0-10)

        Returns:
            Dictionary with evaluation metrics
        """
        # Get predictions
        y_pred = self.predict(X)

        # Overall metrics (11-class)
        overall_f1_macro = f1_score(
            y_true, y_pred,
            average='macro',
            labels=list(range(11)),
            zero_division=0
        )
        overall_f1_weighted = f1_score(
            y_true, y_pred,
            average='weighted',
            labels=list(range(11)),
            zero_division=0
        )
        overall_accuracy = accuracy_score(y_true, y_pred)

        # Stage 1 metrics (binary)
        y_true_binary = (y_true > 0).astype(int)
        y_pred_binary = self.stage1.predict(X)
        stage1_f1 = f1_score(y_true_binary, y_pred_binary, average='macro')
        stage1_accuracy = accuracy_score(y_true_binary, y_pred_binary)

        # Stage 2 metrics (AI samples only)
        ai_mask = y_true > 0
        stage2_f1 = 0.0
        stage2_accuracy = 0.0

        if np.any(ai_mask):
            y_true_ai = y_true[ai_mask]
            y_pred_ai = y_pred[ai_mask]

            stage2_f1 = f1_score(
                y_true_ai, y_pred_ai,
                average='macro',
                labels=list(range(1, 11)),
                zero_division=0
            )
            stage2_accuracy = accuracy_score(y_true_ai, y_pred_ai)

        return {
            'overall_macro_f1': overall_f1_macro,
            'overall_weighted_f1': overall_f1_weighted,
            'overall_accuracy': overall_accuracy,
            'stage1_f1': stage1_f1,
            'stage1_accuracy': stage1_accuracy,
            'stage2_f1': stage2_f1,
            'stage2_accuracy': stage2_accuracy
        }

    @classmethod
    def load(cls, models_dir='models'):
        """
        Load complete pipeline from disk

        Args:
            models_dir: Directory containing saved models

        Returns:
            TwoStagePipeline instance
        """
        models_path = Path(models_dir)

        print("\n" + "="*60)
        print("Loading 2-Stage Pipeline")
        print("="*60)

        # Load feature processor
        processor = FeatureProcessor()
        processor.load(str(models_path / 'stage1' / 'scaler.pkl'))

        # Load Stage 1 classifier
        stage1 = Stage1Classifier()
        stage1.load(str(models_path / 'stage1' / 'classifier.pkl'))

        # Load Stage 2 classifier
        stage2 = create_stage2_classifier('random_forest')
        stage2.load(str(models_path / 'stage2' / 'classifier.pkl'))

        print("="*60 + "\n")

        return cls(stage1, stage2, processor)

    def save(self, models_dir='models'):
        """
        Save complete pipeline to disk

        Args:
            models_dir: Directory to save models
        """
        models_path = Path(models_dir)

        print("\n" + "="*60)
        print("Saving 2-Stage Pipeline")
        print("="*60)

        # Save Stage 1
        self.stage1.save(str(models_path / 'stage1' / 'classifier.pkl'))
        self.processor.save(str(models_path / 'stage1' / 'scaler.pkl'))

        # Save Stage 2
        self.stage2.save(str(models_path / 'stage2' / 'classifier.pkl'))

        print("="*60 + "\n")

    def __str__(self):
        return (f"TwoStagePipeline(\n"
                f"  Stage1: {self.stage1}\n"
                f"  Stage2: {self.stage2}\n"
                f")")
