"""
Hyperparameter Search for CodeT5p-770m

Searches over 4 combinations:
1. lr=1e-4, dropout=0.3
2. lr=1e-4, dropout=0.5
3. lr=2e-4, dropout=0.3 (baseline)
4. lr=2e-4, dropout=0.5

All runs use:
- max_epochs=12 (increased from 5)
- early_stopping=True (patience=3)
- batch_size=256
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Import training function
from src.training.train_stage2_codet5p_770m import train_stage2_codet5p_770m


def run_hyperparam_search():
    """Run hyperparameter search with 4 configurations"""

    # Define search space
    configs = [
        {'name': 'config1_lr1e4_drop03', 'learning_rate': 1e-4, 'dropout': 0.3},
        {'name': 'config2_lr1e4_drop05', 'learning_rate': 1e-4, 'dropout': 0.5},
        {'name': 'config3_lr2e4_drop03', 'learning_rate': 2e-4, 'dropout': 0.3},  # baseline
        {'name': 'config4_lr2e4_drop05', 'learning_rate': 2e-4, 'dropout': 0.5},
    ]

    results = []

    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH: CodeT5p-770m")
    print("="*80)
    print(f"\nSearch space:")
    print(f"  Learning rates: [1e-4, 2e-4]")
    print(f"  Dropout rates: [0.3, 0.5]")
    print(f"  Total configurations: {len(configs)}")
    print(f"  Max epochs: 12 (with early stopping, patience=3)")
    print(f"  Batch size: 256")
    print("="*80 + "\n")
    sys.stdout.flush()

    for i, config in enumerate(configs, 1):
        print(f"\n{'='*80}")
        print(f"CONFIGURATION {i}/{len(configs)}: {config['name']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Dropout: {config['dropout']}")
        print(f"{'='*80}\n")
        sys.stdout.flush()

        # Create save directory for this config
        save_dir = f"models/stage2_codet5p_770m_{config['name']}"

        try:
            # Train model
            clf, metadata = train_stage2_codet5p_770m(
                batch_size=256,
                max_epochs=12,  # Increased from 5
                learning_rate=config['learning_rate'],
                dropout=config['dropout'],
                early_stopping=True,
                validation_fraction=0.1,
                patience=3,
                models_dir='models',  # Use base models dir (scaler is in models/stage1/)
                data_dir='data/reports',
                device='cuda'
            )

            # Copy model to config-specific directory
            import shutil
            src_model_dir = Path('models/stage2_codet5p_770m')
            dst_model_dir = Path(save_dir)
            if src_model_dir.exists():
                dst_model_dir.parent.mkdir(parents=True, exist_ok=True)
                if dst_model_dir.exists():
                    shutil.rmtree(dst_model_dir)
                shutil.copytree(src_model_dir, dst_model_dir)
                print(f"  ✓ Copied model to: {dst_model_dir}")

            # Store results
            result = {
                'config_name': config['name'],
                'learning_rate': config['learning_rate'],
                'dropout': config['dropout'],
                'validation_metrics': metadata['validation_metrics'],
                'training_time': metadata['training_time_seconds'],
                'training_samples': metadata['training_samples'],
                'save_dir': str(dst_model_dir)
            }
            results.append(result)

            print(f"\n✓ Configuration {i} complete!")
            print(f"  Validation Accuracy: {metadata['validation_metrics']['accuracy']:.4f}")
            print(f"  Validation Macro F1: {metadata['validation_metrics']['macro_f1']:.4f}")
            print(f"  Training time: {metadata['training_time_seconds']/60:.1f} minutes")
            sys.stdout.flush()

        except Exception as e:
            print(f"\n❌ Configuration {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save all results
    results_file = Path('outputs/hyperparam_search_770m_results.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            'search_space': {
                'learning_rates': [1e-4, 2e-4],
                'dropout_rates': [0.3, 0.5],
                'max_epochs': 12,
                'batch_size': 256
            },
            'results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    # Print summary
    print("\n" + "="*80)
    print("HYPERPARAMETER SEARCH SUMMARY")
    print("="*80)

    if results:
        # Sort by macro F1
        sorted_results = sorted(results, key=lambda x: x['validation_metrics']['macro_f1'], reverse=True)

        print(f"\nRanking by Macro F1:")
        for rank, result in enumerate(sorted_results, 1):
            metrics = result['validation_metrics']
            print(f"\n{rank}. {result['config_name']}")
            print(f"   LR: {result['learning_rate']}, Dropout: {result['dropout']}")
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Macro F1: {metrics['macro_f1']:.4f} ⭐" if rank == 1 else f"   Macro F1: {metrics['macro_f1']:.4f}")
            print(f"   Weighted F1: {metrics['weighted_f1']:.4f}")
            print(f"   Training time: {result['training_time']/60:.1f} min")

        best = sorted_results[0]
        print(f"\n{'='*80}")
        print(f"🏆 BEST CONFIGURATION: {best['config_name']}")
        print(f"   Learning rate: {best['learning_rate']}")
        print(f"   Dropout: {best['dropout']}")
        print(f"   Validation Macro F1: {best['validation_metrics']['macro_f1']:.4f}")
        print(f"   Validation Accuracy: {best['validation_metrics']['accuracy']:.4f}")
        print(f"{'='*80}")

    print(f"\n✓ Results saved to: {results_file}")
    print()

    return results


if __name__ == '__main__':
    # Check if Stage 1 scaler exists
    scaler_path = Path('models/stage1/scaler.pkl')
    if not scaler_path.exists():
        print(f"❌ ERROR: Stage 1 scaler not found: {scaler_path}")
        print("\nPlease train Stage 1 first:")
        print("  uv run python src/training/train_stage1.py")
        sys.exit(1)

    # Run search
    try:
        results = run_hyperparam_search()
        print("\n✓ Hyperparameter search complete!")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
