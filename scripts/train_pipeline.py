"""
End-to-end training script for complete 2-stage pipeline
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.train_stage1 import train_stage1
from src.training.train_stage2 import train_stage2
from src.evaluation.evaluate_pipeline import evaluate_pipeline


def train_full_pipeline(data_dir='data/reports', models_dir='models'):
    """
    Train both stages sequentially and evaluate the complete pipeline

    Args:
        data_dir: Directory containing feature CSVs
        models_dir: Directory to save trained models
    """
    print("\n" + "="*80)
    print(" "*20 + "2-STAGE CLASSIFICATION PIPELINE")
    print(" "*15 + "AI-Generated Code Detection - SemEval 2026")
    print("="*80 + "\n")

    try:
        # Stage 1: Binary Classification
        print("\n" + "#"*80)
        print("# STAGE 1: Binary Classification (Human vs AI-Generated)")
        print("#"*80 + "\n")
        clf1, processor = train_stage1(data_dir, models_dir)

        # Stage 2: Multi-Class Classification
        print("\n" + "#"*80)
        print("# STAGE 2: Multi-Class Classification (10 AI Model Families)")
        print("#"*80 + "\n")
        clf2 = train_stage2(data_dir, models_dir)

        # Evaluate Complete Pipeline
        print("\n" + "#"*80)
        print("# EVALUATING COMPLETE PIPELINE")
        print("#"*80 + "\n")
        metrics = evaluate_pipeline('validation', data_dir, models_dir)

        # Final Summary
        print("\n" + "="*80)
        print(" "*25 + "TRAINING COMPLETE!")
        print("="*80)
        print("\nFinal Validation Metrics:")
        print(f"  Stage 1 (Binary) Macro F1:      {metrics['stage1_f1']:.4f}")
        print(f"  Stage 2 (AI Families) Macro F1: {metrics['stage2_f1']:.4f}")
        print(f"  Overall (11-Class) Macro F1:    {metrics['overall_macro_f1']:.4f}")
        print(f"  Overall Accuracy:                {metrics['overall_accuracy']:.4f}")

        print(f"\nModels saved to: {models_dir}/")
        print(f"  - Stage 1: {models_dir}/stage1/classifier.pkl")
        print(f"  - Stage 2: {models_dir}/stage2/classifier.pkl")
        print(f"  - Scaler:  {models_dir}/stage1/scaler.pkl")

        print("\nNext steps:")
        print("  1. Review evaluation metrics in outputs/metrics/")
        print("  2. Generate submission: python scripts/generate_submission.py")
        print("="*80 + "\n")

        return metrics

    except Exception as e:
        print(f"\n❌ ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train complete 2-stage classification pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python scripts/train_pipeline.py

  # Train with custom data directory
  python scripts/train_pipeline.py --data-dir custom/data/path

  # Train and save to custom models directory
  python scripts/train_pipeline.py --models-dir custom/models/
        """
    )

    parser.add_argument('--data-dir', default='data/reports',
                       help='Directory containing precomputed feature CSVs')
    parser.add_argument('--models-dir', default='models',
                       help='Directory to save trained models')

    args = parser.parse_args()

    # Verify data directory exists
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        print("\nExpected structure:")
        print(f"  {args.data_dir}/train_basic_stats.csv")
        print(f"  {args.data_dir}/train_complexity_features.csv")
        print(f"  {args.data_dir}/train_lexical_features.csv")
        print(f"  {args.data_dir}/validation_*.csv")
        sys.exit(1)

    train_full_pipeline(args.data_dir, args.models_dir)
