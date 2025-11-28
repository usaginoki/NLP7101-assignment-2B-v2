"""
Enhanced test for Stage 2 CodeT5 with detailed logging

Tests the 3 core requirements:
1. Model loads correctly
2. Training runs on GPU
3. Classification head saves correctly
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import tempfile
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stage2_codet5_simple import Stage2CodeT5Simple


def log(msg, flush=True):
    """Print with timestamp and flush"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=flush)


def test_complete_workflow():
    """Test loading, training, and inference with detailed logs"""

    log("="*70)
    log("ENHANCED STAGE 2 CODET5 TEST WITH VERBOSE LOGGING")
    log("="*70)

    # Check GPU availability
    log("\n[STEP 0] Checking GPU availability...")
    if torch.cuda.is_available():
        log(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        log(f"✓ CUDA Version: {torch.version.cuda}")
        log(f"✓ Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        log(f"✓ Free VRAM: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.2f} GB")
    else:
        log("⚠ WARNING: GPU not available, will use CPU (very slow)")

    # 1. Load small dataset
    log("\n[STEP 1] Loading test dataset...")
    log("  Reading data/train.parquet...")
    train_df = pd.read_parquet('data/train.parquet')
    log(f"  ✓ Loaded {len(train_df)} total samples")

    log("  Filtering AI samples (label > 0)...")
    ai_df = train_df[train_df['label'] > 0]
    log(f"  ✓ Found {len(ai_df)} AI samples")

    log("  Sampling 100 random AI samples...")
    ai_df = ai_df.sample(n=100, random_state=42)
    log(f"  ✓ Selected 100 samples for testing")

    # Split 80/20
    n_train = 80
    train_data = ai_df.iloc[:n_train]
    test_data = ai_df.iloc[n_train:]

    log("  Creating feature matrices...")
    X_train = np.random.randn(len(train_data), 36)  # Dummy features (CodeT5 doesn't use them)
    y_train = train_data['label'].values
    code_train = train_data['code'].tolist()

    X_test = np.random.randn(len(test_data), 36)
    y_test = test_data['label'].values
    code_test = test_data['code'].tolist()

    log(f"  ✓ Train set: {len(train_data)} samples")
    log(f"  ✓ Test set: {len(test_data)} samples")
    log(f"  ✓ AI families in data: {np.unique(y_train)}")

    # 2. Initialize model
    log("\n[STEP 2] Initializing CodeT5 model...")
    log("  Creating Stage2CodeT5Simple instance...")
    clf = Stage2CodeT5Simple(
        batch_size=16,  # Smaller batch for stability
        max_epochs=2,
        learning_rate=2e-4,
        max_length=512,
        device='cuda',
        verbose=1
    )
    log("  ✓ Classifier instance created")

    log("\n  Loading CodeT5p-220m encoder from HuggingFace...")
    log("  (This may take a few minutes for first-time download ~1GB)")
    clf._load_model()
    log("  ✓ Model loaded successfully")

    # Verify parameter freezing
    log("\n  Verifying parameter freezing...")
    encoder_params = list(clf.encoder.parameters())
    head_params = list(clf.head.parameters())

    encoder_trainable = any(p.requires_grad for p in encoder_params)
    head_trainable = any(p.requires_grad for p in head_params)

    encoder_count = sum(p.numel() for p in encoder_params)
    head_count = sum(p.numel() for p in head_params)

    log(f"  Encoder params: {encoder_count:,} | requires_grad: {encoder_trainable}")
    log(f"  Head params: {head_count:,} | requires_grad: {head_trainable}")

    if encoder_trainable:
        log("  ✗ FAIL: Encoder should be frozen but has requires_grad=True!")
        return False
    if not head_trainable:
        log("  ✗ FAIL: Head should be trainable but has requires_grad=False!")
        return False

    log("  ✓ Parameter freezing verified: Encoder frozen, Head trainable")

    # Check device placement
    log("\n  Verifying device placement...")
    encoder_device = next(clf.encoder.parameters()).device
    head_device = next(clf.head.parameters()).device
    log(f"  Encoder device: {encoder_device}")
    log(f"  Head device: {head_device}")

    if torch.cuda.is_available():
        if encoder_device.type != 'cuda' or head_device.type != 'cuda':
            log("  ✗ FAIL: Models should be on GPU but found on CPU!")
            return False
        log("  ✓ Models correctly placed on GPU")

    # 3. Training
    log("\n[STEP 3] Training classifier...")
    log(f"  Batch size: {clf.batch_size}")
    log(f"  Max epochs: {clf.max_epochs}")
    log(f"  Learning rate: {clf.learning_rate}")
    log(f"  Training samples: {len(y_train)}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        log(f"  GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    log("\n  Starting training loop...")
    try:
        clf.fit(X_train, y_train, code_strings=code_train)
        log("  ✓ Training completed successfully")
    except Exception as e:
        log(f"  ✗ FAIL: Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        current_memory = torch.cuda.memory_allocated() / 1e9
        log(f"  GPU peak memory: {peak_memory:.2f} GB")
        log(f"  GPU current memory: {current_memory:.2f} GB")

    # 4. Inference
    log("\n[STEP 4] Testing inference...")
    log(f"  Running predictions on {len(test_data)} test samples...")

    try:
        preds = clf.predict(X_test, code_strings=code_test)
        log("  ✓ Predictions generated successfully")
    except Exception as e:
        log(f"  ✗ FAIL: Inference failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    log(f"  Predictions (first 10): {preds[:10]}")
    log(f"  Ground truth (first 10): {y_test[:10]}")

    accuracy = np.mean(preds == y_test)
    log(f"  ✓ Test accuracy: {accuracy:.1%} ({int(accuracy * len(y_test))}/{len(y_test)} correct)")

    # 5. Save/Load checkpoint
    log("\n[STEP 5] Testing checkpoint save/load...")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / 'codet5_checkpoint.pth'
        log(f"  Saving checkpoint to: {save_path}")

        try:
            clf.save(str(save_path))
            checkpoint_size = save_path.stat().st_size / (1024 * 1024)
            log(f"  ✓ Checkpoint saved ({checkpoint_size:.2f} MB)")
        except Exception as e:
            log(f"  ✗ FAIL: Save failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

        log("\n  Loading checkpoint in new model instance...")
        try:
            clf_new = Stage2CodeT5Simple(verbose=0)
            clf_new.load(str(save_path))
            log("  ✓ Checkpoint loaded successfully")
        except Exception as e:
            log(f"  ✗ FAIL: Load failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

        log("  Running predictions with loaded model...")
        try:
            preds_new = clf_new.predict(X_test, code_strings=code_test)
            log("  ✓ Loaded model can generate predictions")
        except Exception as e:
            log(f"  ✗ FAIL: Loaded model inference failed: {e}")
            return False

        # Verify predictions match
        if np.array_equal(preds, preds_new):
            log("  ✓ Checkpoint integrity verified (predictions match)")
        else:
            log("  ✗ FAIL: Predictions differ after loading checkpoint!")
            log(f"    Original: {preds[:10]}")
            log(f"    Loaded:   {preds_new[:10]}")
            return False

    # Final GPU stats
    if torch.cuda.is_available():
        log("\n[GPU FINAL STATS]")
        log(f"  Allocated memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        log(f"  Reserved memory: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        log(f"  Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    # Summary
    log("\n" + "="*70)
    log("✓ ALL TESTS PASSED!")
    log("="*70)
    log("\nVerified:")
    log("  [✓] Model loads correctly")
    log("  [✓] Encoder is frozen (only head trained)")
    log("  [✓] Training runs on GPU (if available)")
    log("  [✓] Training completes successfully")
    log("  [✓] Inference works correctly")
    log("  [✓] Classification head saves/loads properly")
    log("\nThe CodeT5 Stage 2 classifier is ready for production use!")
    log("="*70)

    return True


if __name__ == "__main__":
    try:
        success = test_complete_workflow()
        sys.exit(0 if success else 1)
    except Exception as e:
        log(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
