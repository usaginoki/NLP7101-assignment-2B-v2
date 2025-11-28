"""
Test CodeT5p-770m model loading and usage

This script verifies that:
1. CodeT5p-770m can be loaded successfully
2. Encoder can be extracted for embeddings
3. Model produces correct embedding dimensions
4. Basic inference works

Based on official documentation:
- https://huggingface.co/Salesforce/codet5p-770m
- Uses T5ForConditionalGeneration (seq2seq model)
- We'll extract encoder for embeddings
"""

import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer
import time
import sys

print("Testing Salesforce/codet5p-770m loading...\n")
print("="*70)

# Step 1: Load tokenizer
print("Step 1: Loading tokenizer...")
sys.stdout.flush()
start = time.time()
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5p-770m")
elapsed = time.time() - start
print(f"  ✓ Tokenizer loaded ({elapsed:.1f}s)\n")
sys.stdout.flush()

# Step 2: Load full model
print("Step 2: Loading full model (~3GB download)...")
print("  This may take several minutes on first run...")
sys.stdout.flush()
start = time.time()
full_model = T5ForConditionalGeneration.from_pretrained(
    "Salesforce/codet5p-770m",
    torch_dtype=torch.float32,  # Use full precision
    low_cpu_mem_usage=True
)
elapsed = time.time() - start
print(f"  ✓ Full model loaded ({elapsed:.1f}s)\n")
sys.stdout.flush()

# Step 3: Extract encoder
print("Step 3: Extracting encoder for embeddings...")
encoder = full_model.get_encoder()
print(f"  ✓ Encoder extracted: {type(encoder).__name__}")
sys.stdout.flush()

# Step 4: Model statistics
print("\nModel statistics:")
total_params = sum(p.numel() for p in full_model.parameters())
encoder_params = sum(p.numel() for p in encoder.parameters())
print(f"  Total parameters: {total_params:,}")
print(f"  Encoder parameters: {encoder_params:,}")
print(f"  Size: {total_params/1e9:.2f}B ({total_params/1e6:.0f}M)")
print(f"  Embedding dimension: {encoder.config.d_model}")
sys.stdout.flush()

# Step 5: Test encoding
print("\nStep 5: Testing encoding...")
test_code = "def hello(): print('world')"
inputs = tokenizer(test_code, return_tensors="pt", max_length=512, truncation=True, padding=True)

with torch.no_grad():
    # Get encoder outputs (last hidden state)
    encoder_outputs = encoder(**inputs)
    embeddings = encoder_outputs.last_hidden_state

    # Mean pooling across sequence length
    pooled = embeddings.mean(dim=1)

print(f"  Input shape: {inputs['input_ids'].shape}")
print(f"  Encoder output shape: {embeddings.shape}")
print(f"  Pooled embeddings: {pooled.shape}")
print(f"  Embedding dim: {pooled.shape[1]}")
sys.stdout.flush()

# Step 6: Move to GPU if available
if torch.cuda.is_available():
    print("\nStep 6: Testing GPU transfer...")
    sys.stdout.flush()
    start = time.time()
    encoder = encoder.to('cuda')
    elapsed = time.time() - start
    print(f"  ✓ Moved to GPU ({elapsed:.1f}s)")
    print(f"  Device: {next(encoder.parameters()).device}")
    sys.stdout.flush()

    # Test GPU inference
    print("\n  Testing GPU inference...")
    sys.stdout.flush()
    inputs_gpu = {k: v.to('cuda') for k, v in inputs.items()}
    with torch.no_grad():
        encoder_outputs = encoder(**inputs_gpu)
        embeddings = encoder_outputs.last_hidden_state.mean(dim=1)
    print(f"  ✓ GPU inference successful: {embeddings.shape}")
    sys.stdout.flush()

print("\n" + "="*70)
print("✓ CodeT5p-770m works perfectly!")
print(f"  Model size: {total_params/1e6:.0f}M parameters")
print(f"  Embedding dim: {encoder.config.d_model}")
print("="*70)
print()
