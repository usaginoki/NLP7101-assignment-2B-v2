#!/usr/bin/env python3
"""
Stage 1 Model Search - Main Executable

Comprehensive hyperparameter search across 7 classifier types:
- Logistic Regression
- SVM (Linear & RBF)
- Random Forest
- XGBoost
- LightGBM
- MLP Neural Network

Usage:
    python scripts/train_stage1_model_search.py [--test-mode] [--classifier CLASSIFIER] [--gpu]

Options:
    --test-mode: Run with single classifier (Logistic Regression) for testing
    --classifier: Run search for specific classifier only (e.g., 'LogisticRegression')
    --gpu: Use GPU-optimized configurations (requires NVIDIA GPU)
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# Import PyTorch MLP (GPU-accelerated)
from src.models.pytorch_mlp import PyTorchMLPClassifier

# Import model search framework
from src.models.model_search.search_runner import ModelSearchRunner
from src.models.model_search.model_evaluator import ModelEvaluator
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.report_generator import ReportGenerator

# Import existing feature pipeline
from src.features.feature_loader import FeatureLoader
from src.features.feature_processor import FeatureProcessor
from src.models.stage1_classifier import Stage1Classifier


def load_and_prepare_data(verbose=True):
    """
    Load features and prepare data for model search

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, processor)
    """
    if verbose:
        print("="*80)
        print("LOADING AND PREPARING DATA")
        print("="*80)

    # Load features
    if verbose:
        print("\n1. Loading precomputed features...")
    loader = FeatureLoader()
    train_df = loader.load_features('train')
    val_df = loader.load_features('validation')

    if verbose:
        print(f"   Training samples: {len(train_df):,}")
        print(f"   Validation samples: {len(val_df):,}")
        meta_cols = ['code', 'generator', 'label', 'language']
        feature_cols = [c for c in train_df.columns if c not in meta_cols]
        print(f"   Features: {len(feature_cols)}")

    # Scale features using FeatureProcessor (expects DataFrames)
    if verbose:
        print("\n2. Scaling features with StandardScaler...")
    processor = FeatureProcessor()
    X_train, y_train_full = processor.fit_transform(train_df)
    X_val, y_val_full = processor.transform(val_df)

    # Convert to binary labels (0=Human, 1=AI)
    if verbose:
        print("\n3. Converting labels to binary (Human vs AI)...")
    y_train = Stage1Classifier.prepare_labels(y_train_full)
    y_val = Stage1Classifier.prepare_labels(y_val_full)

    if verbose:
        print(f"   Training - Human: {(y_train == 0).sum():,}, AI: {(y_train == 1).sum():,}")
        print(f"   Validation - Human: {(y_val == 0).sum():,}, AI: {(y_val == 1).sum():,}")

    if verbose:
        print(f"   Scaled training shape: {X_train.shape}")
        print(f"   Scaled validation shape: {X_val.shape}")

    return X_train, y_train, X_val, y_val, processor


def create_classifier_instances():
    """
    Create instances of all 7 classifiers

    Returns:
        Dictionary mapping classifier names to initialized instances
    """
    # Import here to avoid errors if packages not installed
    try:
        import xgboost as xgb
        import lightgbm as lgb
        xgb_available = True
        lgb_available = True
    except ImportError:
        xgb_available = False
        lgb_available = False
        print("\nWarning: XGBoost/LightGBM not installed. Skipping these classifiers.")

    classifiers = {
        'LogisticRegression': LogisticRegression(),
        'SVM_Linear': LinearSVC(),
        'SVM_RBF': SVC(),
        'RandomForest': RandomForestClassifier(),
        'MLP': PyTorchMLPClassifier()  # GPU-accelerated PyTorch MLP
    }

    if xgb_available:
        classifiers['XGBoost'] = xgb.XGBClassifier()
    if lgb_available:
        classifiers['LightGBM'] = lgb.LGBMClassifier()

    return classifiers


def run_model_search(X_train, y_train, X_val, y_val, processor,
                     test_mode=False, specific_classifier=None,
                     use_gpu=False, verbose=True):
    """
    Run hyperparameter search for all classifiers

    Args:
        X_train: Scaled training features (500k × 36)
        y_train: Binary training labels (0=Human, 1=AI)
        X_val: Scaled validation features (100k × 36)
        y_val: Binary validation labels
        processor: FeatureProcessor with fitted scaler (for saving with models)
        test_mode: If True, only run Logistic Regression for testing
        specific_classifier: If provided, only run this classifier
        use_gpu: If True, use GPU-optimized configurations
        verbose: Print progress messages

    Returns:
        Tuple of (all_results, experiments, comparison)
    """
    if verbose:
        print("\n" + "="*80)
        print("RUNNING MODEL SEARCH")
        print("="*80)
        if use_gpu:
            print("GPU Acceleration: ENABLED")
        print()

    # Initialize components
    search_runner = ModelSearchRunner(X_train, y_train, X_val, y_val,
                                     verbose=verbose, random_state=42)
    evaluator = ModelEvaluator(X_val, y_val)
    tracker = ExperimentTracker(output_dir='outputs/model_search')

    # Get search configurations (CPU or GPU)
    if use_gpu:
        from src.models.model_search.search_configs_gpu import get_all_search_configs
        if verbose:
            print("Loading GPU-optimized configurations...")
    else:
        from src.models.model_search.search_configs import get_all_search_configs
        if verbose:
            print("Loading CPU configurations...")

    search_configs = get_all_search_configs()

    # Get classifier instances
    classifiers = create_classifier_instances()

    # Filter classifiers if needed
    if test_mode:
        if verbose:
            print("\n*** TEST MODE: Running Logistic Regression only ***\n")
        classifiers = {'LogisticRegression': classifiers['LogisticRegression']}
    elif specific_classifier:
        if specific_classifier not in classifiers:
            raise ValueError(f"Unknown classifier: {specific_classifier}. "
                           f"Available: {list(classifiers.keys())}")
        if verbose:
            print(f"\n*** Running specific classifier: {specific_classifier} ***\n")
        classifiers = {specific_classifier: classifiers[specific_classifier]}

    # Run search for each classifier
    all_results = {}
    for idx, (name, classifier) in enumerate(classifiers.items(), 1):
        if verbose:
            print("\n" + "-"*80)
            print(f"CLASSIFIER {idx}/{len(classifiers)}: {name}")
            print("-"*80)

        if name not in search_configs:
            if verbose:
                print(f"⚠ No search config found for {name}. Skipping.")
            continue

        config = search_configs[name]

        if verbose:
            print(f"\nConfiguration:")
            print(f"  Search type: {config['search_type']}")
            print(f"  Grid size: {len(config['grid'])} parameters")
            if config.get('n_iter'):
                print(f"  Iterations: {config['n_iter']}")
            if config.get('sampling_fraction'):
                print(f"  Sampling: {config['sampling_fraction']*100:.0f}%")
            print(f"  Timeout: {config['timeout_seconds']/60:.0f} minutes")
            print()

        # Run search
        result = search_runner.run_search(classifier, config)

        # Log to tracker
        tracker.log_experiment(name, result)

        # If successful, evaluate on validation set
        if result['status'] == 'SUCCESS':
            eval_metrics = evaluator.evaluate_model(result['best_model'], name)

            # Merge metrics into result
            result.update({
                'model_name': name,
                'accuracy': eval_metrics['accuracy'],
                'precision_macro': eval_metrics['precision_macro'],
                'recall_macro': eval_metrics['recall_macro'],
                'confusion_matrix': eval_metrics['confusion_matrix'],
                'per_class': eval_metrics['per_class']
            })

            if verbose:
                print(f"\n✓ {name} completed successfully")
                print(f"  Best CV score: {result['best_score']:.4f}")
                print(f"  Validation Macro F1: {result['val_macro_f1']:.4f}")
                print(f"  Accuracy: {result['accuracy']:.4f}")
                print(f"  Time: {result['elapsed_time']/60:.1f} minutes")

            # Save classifier-specific model immediately
            save_classifier_model(name, result, processor, verbose=verbose)
        else:
            if verbose:
                print(f"\n✗ {name} failed: {result['status']}")
                print(f"  Error: {result.get('error', 'Unknown')}")
                print(f"  Time elapsed: {result['elapsed_time']/60:.1f} minutes")

        all_results[name] = result

    # Compare all successful models
    if verbose:
        print("\n" + "="*80)
        print("COMPARING MODELS")
        print("="*80)

    # Convert results to list format for comparison
    results_list = []
    for name, result in all_results.items():
        if result['status'] == 'SUCCESS':
            results_list.append(result)

    comparison = evaluator.compare_models(results_list)
    evaluator.print_comparison_summary(comparison)

    # Save comparison
    tracker.save_best_model_comparison(comparison)

    # Get experiment list for reporting
    experiments = tracker.load_experiments()

    return all_results, experiments, comparison


def save_classifier_model(classifier_name, result, processor, verbose=True):
    """
    Save a classifier's best model to its own subdirectory

    Args:
        classifier_name: Name of classifier (e.g., 'XGBoost')
        result: Search result dictionary with best_model
        processor: FeatureProcessor with fitted scaler
        verbose: Print messages
    """
    if result['status'] != 'SUCCESS':
        return

    # Create classifier-specific directory
    classifier_dir = Path('models/stage1') / classifier_name
    classifier_dir.mkdir(parents=True, exist_ok=True)

    # Save classifier model
    model_path = classifier_dir / 'classifier.pkl'
    joblib.dump(result['best_model'], model_path)

    # Save classifier metadata
    metadata = {
        'model_name': classifier_name,
        'best_params': result['best_params'],
        'val_macro_f1': float(result['val_macro_f1']),
        'accuracy': float(result['accuracy']),
        'precision_macro': float(result['precision_macro']),
        'recall_macro': float(result['recall_macro']),
        'elapsed_time_seconds': result['elapsed_time']
    }
    metadata_path = classifier_dir / 'metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    if verbose:
        print(f"  ✓ Saved {classifier_name} model to: {classifier_dir}/")


def save_global_best_model(comparison, processor, verbose=True):
    """
    Save the globally best model across all classifiers to Stage 1 directory

    Args:
        comparison: Comparison dictionary with best_model from all classifiers
        processor: FeatureProcessor with fitted scaler
        verbose: Print messages
    """
    if not comparison['best_model']:
        print("\n⚠ No successful models to save!")
        return

    if verbose:
        print("\n" + "="*80)
        print("SAVING GLOBAL BEST MODEL")
        print("="*80)

    best = comparison['best_model']
    model_name = best['model_name']
    best_model = best['best_model']

    # Save to Stage 1 root directory (for Stage 2 compatibility)
    stage1_dir = Path('models/stage1')
    stage1_dir.mkdir(parents=True, exist_ok=True)

    # Save global best model
    model_path = stage1_dir / 'classifier.pkl'
    joblib.dump(best_model, model_path)
    if verbose:
        print(f"\n✓ Saved global best model to: {model_path}")
        print(f"  Model: {model_name}")
        print(f"  Validation Macro F1: {best['val_macro_f1']:.4f}")

    # Save scaler (shared by all classifiers)
    scaler_path = stage1_dir / 'scaler.pkl'
    joblib.dump(processor.scaler, scaler_path)
    if verbose:
        print(f"✓ Saved scaler to: {scaler_path}")

    # Save global best metadata
    metadata = {
        'model_name': model_name,
        'best_params': best['best_params'],
        'val_macro_f1': float(best['val_macro_f1']),
        'accuracy': float(best['accuracy']),
        'precision_macro': float(best['precision_macro']),
        'recall_macro': float(best['recall_macro'])
    }
    metadata_path = stage1_dir / 'model_metadata.json'
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    if verbose:
        print(f"✓ Saved metadata to: {metadata_path}")


def generate_reports(comparison, experiments, verbose=True):
    """
    Generate all visualizations and reports

    Args:
        comparison: Comparison dictionary from ModelEvaluator
        experiments: List of experiment dicts from ExperimentTracker
        verbose: Print messages
    """
    if verbose:
        print("\n" + "="*80)
        print("GENERATING REPORTS")
        print("="*80)

    generator = ReportGenerator(output_dir='outputs/model_search')
    generator.generate_all_reports(comparison, experiments)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Stage 1 Model Search - Comprehensive Hyperparameter Exploration'
    )
    parser.add_argument('--test-mode', action='store_true',
                       help='Run with Logistic Regression only for testing')
    parser.add_argument('--classifier', type=str, default=None,
                       help='Run specific classifier only (e.g., LogisticRegression)')
    parser.add_argument('--gpu', action='store_true',
                       help='Use GPU-optimized configurations (requires NVIDIA GPU)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("STAGE 1 MODEL SEARCH")
    print("="*80)
    print(f"\nMode: {'TEST (Logistic Regression only)' if args.test_mode else 'FULL (7 classifiers)'}")
    if args.classifier:
        print(f"Specific classifier: {args.classifier}")
    if args.gpu:
        print("GPU Acceleration: ENABLED")
    print()

    # Phase 1: Load and prepare data
    X_train, y_train, X_val, y_val, processor = load_and_prepare_data(verbose=True)

    # Phase 2: Run model search
    all_results, experiments, comparison = run_model_search(
        X_train, y_train, X_val, y_val, processor,
        test_mode=args.test_mode,
        specific_classifier=args.classifier,
        use_gpu=args.gpu,
        verbose=True
    )

    # Phase 3: Save global best model
    save_global_best_model(comparison, processor, verbose=True)

    # Phase 4: Generate reports
    generate_reports(comparison, experiments, verbose=True)

    # Final summary
    print("\n" + "="*80)
    print("MODEL SEARCH COMPLETE")
    print("="*80)

    if comparison['best_model']:
        best = comparison['best_model']
        print(f"\n🏆 Best Model: {best['model_name']}")
        print(f"   Validation Macro F1: {best['val_macro_f1']:.4f}")
        print(f"   Accuracy: {best['accuracy']:.4f}")
        print(f"\n📁 Outputs:")
        print(f"   - Experiments: outputs/model_search/experiments.json")
        print(f"   - Comparison: outputs/model_search/best_model_comparison.json")
        print(f"   - Report: outputs/model_search/model_comparison_report.md")
        print(f"   - Visualizations: outputs/model_search/visualizations/")
        print(f"   - Best model: models/stage1/classifier.pkl")
        print(f"   - Scaler: models/stage1/scaler.pkl")
    else:
        print("\n⚠ No successful models found!")

    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
