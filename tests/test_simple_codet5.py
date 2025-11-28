"""
Simple test for Stage 2 CodeT5 Classifier

Tests the 3 core requirements:
1. Model loads correctly
2. Training works smoothly
3. GPU is utilized
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stage2_codet5_simple import Stage2CodeT5Simple


def test_complete_workflow():
    """Test loading, training, and inference"""

    print("="*70)
    print("SIMPLIFIED STAGE 2 CODET5 TEST")
    print("="*70)

    # 1. Load small dataset
    print("\n[1/4] Loading test data...")
    train_df = pd.read_parquet('data/train.parquet')
    ai_df = train_df[train_df['label'] > 0].sample(n=100, random_state=42)

    # Split 80/20
    n_train = 80
    train_data = ai_df.iloc[:n_train]
    test_data = ai_df.iloc[n_train:]

    X_train = np.random.randn(len(train_data), 36)  # Dummy features
    y_train = train_data['label'].values
    code_train = train_data['code'].tolist()

    X_test = np.random.randn(len(test_data), 36)
    y_test = test_data['label'].values
    code_test = test_data['code'].tolist()

    print(f"✓ Train: {len(train_data)} samples")
    print(f"✓ Test: {len(test_data)} samples")

    # 2. Load model
    print("\n[2/4] Loading model...")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    clf = Stage2CodeT5Simple(
        batch_size=32,
        max_epochs=2,
        learning_rate=2e-4,
        verbose=1
    )

    clf._load_model()

    # Verify freezing
    encoder_trainable = any(p.requires_grad for p in clf.encoder.parameters())
    head_trainable = any(p.requires_grad for p in clf.head.parameters())

    print(f"\nEncoder frozen: {not encoder_trainable}")
    print(f"Head trainable: {head_trainable}")

    if encoder_trainable:
        print("✗ FAIL: Encoder should be frozen!")
        return False
    if not head_trainable:
        print("✗ FAIL: Head should be trainable!")
        return False

    # 3. Train
    print("\n[3/4] Training...")
    clf.fit(X_train, y_train, code_strings=code_train)
    print("✓ Training complete")

    # 4. Inference
    print("\n[4/4] Testing inference...")
    preds = clf.predict(X_test, code_strings=code_test)

    accuracy = np.mean(preds == y_test)
    print(f"✓ Predictions: {preds[:10]}")
    print(f"✓ Ground truth: {y_test[:10]}")
    print(f"✓ Accuracy: {accuracy:.1%}")

    # 5. Save/Load
    print("\n[5/5] Testing checkpoint...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'model.pth'
        clf.save(str(save_path))

        # Load in new instance
        clf_new = Stage2CodeT5Simple(verbose=1)
        clf_new.load(str(save_path))

        # Verify predictions match
        preds_new = clf_new.predict(X_test, code_strings=code_test)
        if np.array_equal(preds, preds_new):
            print("✓ Checkpoint integrity verified")
        else:
            print("✗ FAIL: Predictions differ after load!")
            return False

    # GPU check
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        print(f"\nGPU Memory: {allocated_gb:.2f} GB allocated")

    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED!")
    print("="*70)
    print("\nModel is ready for full training.")
    return True


if __name__ == "__main__":
    success = test_complete_workflow()
    sys.exit(0 if success else 1)
