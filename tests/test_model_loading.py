"""
Minimal test to identify model loading issues
"""

import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, T5EncoderModel

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_direct_loading():
    """Test loading the model directly"""
    print("="*70)
    print("TESTING DIRECT MODEL LOADING")
    print("="*70)

    # Check GPU
    print("\n[1] Checking GPU...")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("⚠ No GPU, using CPU")
        device = 'cpu'

    # Load tokenizer
    print("\n[2] Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-770m')
        print(f"✓ Tokenizer loaded: {type(tokenizer)}")
    except Exception as e:
        print(f"✗ Tokenizer loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load encoder
    print("\n[3] Loading T5EncoderModel...")
    try:
        print("  Calling T5EncoderModel.from_pretrained()...")
        encoder = T5EncoderModel.from_pretrained('Salesforce/codet5p-770m')
        print(f"✓ Encoder loaded: {type(encoder)}")
    except Exception as e:
        print(f"✗ Encoder loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Move to device
    print(f"\n[4] Moving encoder to {device}...")
    try:
        encoder = encoder.to(device)
        print(f"✓ Encoder on {device}")
    except Exception as e:
        print(f"✗ Failed to move to {device}: {e}")
        return False

    # Freeze encoder
    print("\n[5] Freezing encoder...")
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    encoder_trainable = any(p.requires_grad for p in encoder.parameters())
    print(f"  Encoder trainable: {encoder_trainable}")
    if encoder_trainable:
        print("✗ Failed to freeze encoder!")
        return False
    print("✓ Encoder frozen")

    # Test encoding
    print("\n[6] Testing encoding...")
    try:
        test_code = ["def hello():\\n    print('world')"]
        inputs = tokenizer(test_code, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = encoder(**inputs)
            embeddings = outputs.last_hidden_state

        print(f"✓ Encoding successful")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Expected: (1, seq_len, hidden_dim)")
    except Exception as e:
        print(f"✗ Encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check config
    print("\n[7] Checking encoder config...")
    print(f"  Hidden size: {encoder.config.hidden_size}")
    print(f"  Num layers: {encoder.config.num_layers}")
    print(f"  Model type: {encoder.config.model_type}")

    print("\n" + "="*70)
    print("✓ ALL MODEL LOADING TESTS PASSED!")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_direct_loading()
    sys.exit(0 if success else 1)
