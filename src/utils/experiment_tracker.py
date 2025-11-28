"""
Experiment Tracker

JSON-based logging system for model search experiments.
Tracks all experiments with detailed metadata for future analysis.
"""

import json
import numpy as np
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path


class ExperimentTracker:
    """Track model search experiments to JSON"""

    def __init__(self, output_dir: str = 'outputs/model_search'):
        """
        Initialize experiment tracker

        Args:
            output_dir: Directory to save experiment results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiments_file = self.output_dir / 'experiments.json'
        self.experiments = []

        # Load existing experiments if file exists
        if self.experiments_file.exists():
            self.experiments = self.load_experiments()

    def log_experiment(self, classifier_name: str, result: Dict[str, Any]):
        """
        Log a single experiment

        Args:
            classifier_name: Name of the classifier (e.g., 'LogisticRegression')
            result: Result dictionary from ModelSearchRunner.run_search()
        """
        experiment = {
            'timestamp': datetime.now().isoformat(),
            'classifier': classifier_name,
            'status': result['status'],
            'elapsed_time_seconds': result['elapsed_time']
        }

        if result['status'] == 'SUCCESS':
            experiment.update({
                'best_params': result['best_params'],
                'cv_best_score': float(result['best_score']),
                'val_macro_f1': float(result['val_macro_f1']),
                'n_combinations_tested': int(result['n_combinations_tested']),
                'sampling_used': result.get('sampling_used', False)
            })
        else:
            experiment['error'] = result.get('error', 'Unknown error')

        self.experiments.append(experiment)
        self._save_experiments()

        # Print confirmation
        if result['status'] == 'SUCCESS':
            print(f"\n✓ Logged experiment: {classifier_name}")
            print(f"  Validation Macro F1: {result['val_macro_f1']:.4f}")
        else:
            print(f"\n✗ Logged failed experiment: {classifier_name}")
            print(f"  Status: {result['status']}")

    def _save_experiments(self):
        """Save experiments to JSON file"""
        with open(self.experiments_file, 'w') as f:
            json.dump(self.experiments, f, indent=2)

        print(f"  Saved to: {self.experiments_file}")

    def load_experiments(self) -> List[Dict[str, Any]]:
        """
        Load experiments from JSON file

        Returns:
            List of experiment dictionaries
        """
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                return json.load(f)
        return []

    def save_best_model_comparison(self, comparison: Dict[str, Any]):
        """
        Save model comparison summary

        Args:
            comparison: Comparison dictionary from ModelEvaluator.compare_models()
        """
        comparison_file = self.output_dir / 'best_model_comparison.json'

        # Convert numpy arrays to lists for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if hasattr(obj, 'item'):  # numpy scalars
                return obj.item()
            # Skip non-serializable objects (like fitted models)
            return None

        # Create serializable comparison (exclude best_model object)
        serializable_comparison = {
            'total_models_tested': comparison['total_models_tested'],
            'successful_runs': comparison['successful_runs'],
            'failed_runs': comparison['failed_runs']
        }

        # Add best model info (exclude the actual model object)
        if comparison['best_model']:
            best = comparison['best_model'].copy()
            best.pop('best_model', None)  # Remove model object
            best.pop('cv_results', None)  # Remove large cv_results
            serializable_comparison['best_model'] = self._convert_dict(best)

        # Add top 5 models
        top_5 = []
        for model in comparison['top_5_models']:
            model_copy = model.copy()
            model_copy.pop('best_model', None)
            model_copy.pop('cv_results', None)
            top_5.append(self._convert_dict(model_copy))
        serializable_comparison['top_5_models'] = top_5

        # Save to JSON
        with open(comparison_file, 'w') as f:
            json.dump(serializable_comparison, f, indent=2)

        print(f"\n✓ Saved model comparison to: {comparison_file}")

    def _convert_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively convert numpy types in dictionary to Python types"""
        converted = {}
        for key, value in d.items():
            if isinstance(value, dict):
                converted[key] = self._convert_dict(value)
            elif isinstance(value, list):
                converted[key] = [self._convert_value(v) for v in value]
            else:
                converted[key] = self._convert_value(value)
        return converted

    def _convert_value(self, value: Any) -> Any:
        """Convert a single value to JSON-serializable type"""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            return int(value)
        if isinstance(value, (np.float64, np.float32, np.float16)):
            return float(value)
        if isinstance(value, dict):
            return self._convert_dict(value)
        return value

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of all experiments

        Returns:
            Dictionary with summary stats:
                - total_experiments: Total number of experiments
                - successful_experiments: Number of successful experiments
                - failed_experiments: Number of failed experiments
                - avg_runtime: Average runtime in seconds
                - best_f1_score: Best macro F1 score achieved
        """
        if not self.experiments:
            return {}

        successful = [e for e in self.experiments if e['status'] == 'SUCCESS']
        failed = [e for e in self.experiments if e['status'] != 'SUCCESS']

        summary = {
            'total_experiments': len(self.experiments),
            'successful_experiments': len(successful),
            'failed_experiments': len(failed),
            'avg_runtime_seconds': np.mean([e['elapsed_time_seconds'] for e in self.experiments]),
        }

        if successful:
            summary['best_f1_score'] = max([e['val_macro_f1'] for e in successful])
            summary['avg_f1_score'] = np.mean([e['val_macro_f1'] for e in successful])

        return summary
