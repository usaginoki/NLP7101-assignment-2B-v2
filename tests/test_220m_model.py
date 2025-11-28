"""
Test loading CodeT5p-220m (smaller model) to compare
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, T5EncoderModel
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_model_loading(model_name):
    """Test loading a specific model"""
    print(f"\n{'='*70}")
    print(f"TESTING: {model_name}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # Check GPU
    print("[1] GPU Check")
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("  ⚠ No GPU")
        device = 'cpu'

    # Load tokenizer
    print(f"\n[2] Loading tokenizer for {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"  ✓ Tokenizer loaded ({time.time() - start_time:.1f}s)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False

    # Load encoder
    print(f"\n[3] Loading T5EncoderModel for {model_name}...")
    print("  This is where the error usually occurs...")
    try:
        encoder = T5EncoderModel.from_pretrained(model_name)
        print(f"  ✓ Encoder loaded ({time.time() - start_time:.1f}s)")
        print(f"  Hidden size: {encoder.config.hidden_size}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Move to GPU
    print(f"\n[4] Moving to {device}...")
    try:
        encoder = encoder.to(device)
        print(f"  ✓ On {device}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False

    # Test inference
    print("\n[5] Testing inference...")
    try:
        test_code = ["def hello(): print('world')"]
        inputs = tokenizer(test_code, return_tensors='pt', padding=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = encoder(**inputs)

        print(f"  ✓ Inference successful")
        print(f"  Output shape: {outputs.last_hidden_state.shape}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ SUCCESS for {model_name} (took {total_time:.1f}s)")
    print(f"{'='*70}\n")
    return True

if __name__ == "__main__":
    # Test both models
    models = [
        'Salesforce/codet5p-220m',  # Smaller model
        'Salesforce/codet5p-770m',  # Original model
    ]

    results = {}
    for model in models:
        print(f"\n\n{'#'*70}")
        print(f"# Testing {model}")
        print(f"{'#'*70}")
        results[model] = test_model_loading(model)

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for model, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {model}")
    print(f"{'='*70}\n")

    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)
