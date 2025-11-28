"""
GPU Batch Size Optimization for CodeT5p-770m

Determines optimal batch size for RTX 5090 (32GB VRAM) by testing
increasing batch sizes until OOM, then recommends safe batch size.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.stage2_classifier import create_stage2_classifier


def test_batch_sizes():
    """Test increasing batch sizes to find GPU limit"""

    if not torch.cuda.is_available():
        print("⚠ CUDA not available - skipping optimization")
        return 8  # Default CPU batch size

    print("="*70)
    print("BATCH SIZE OPTIMIZATION FOR RTX 5090")
    print("="*70)

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU: {gpu_name}")
    print(f"Total VRAM: {total_vram_gb:.2f} GB")
    print(f"Target: Use 80-90% VRAM for optimal throughput\n")

    # Create dummy code samples (varied lengths)
    code_samples = [
        f"def function_{i}():\n    " + "    x = " * (i % 20 + 1) + "return x\n"
        for i in range(200)
    ]

    # Test batch sizes: [8, 16, 32, 64, 128, 256]
    test_sizes = [8, 16, 32, 64, 128, 256]
    results = []

    print("Testing batch sizes:")
    print("-" * 70)

    for batch_size in test_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            print(f"\nBatch size: {batch_size}")

            # Create fresh model
            clf = create_stage2_classifier(
                'codet5',
                batch_size=batch_size,
                verbose=0
            )
            clf._initialize_model()

            # Run encoding to measure peak memory
            batch = code_samples[:batch_size]
            embeddings = clf._encode_batch(batch)

            # Force GPU sync
            torch.cuda.synchronize()

            # Measure memory
            allocated_gb = torch.cuda.memory_allocated(0) / 1024**3
            reserved_gb = torch.cuda.memory_reserved(0) / 1024**3
            peak_gb = torch.cuda.max_memory_allocated(0) / 1024**3

            vram_percent = (peak_gb / total_vram_gb) * 100

            print(f"  ✓ Success")
            print(f"    Allocated: {allocated_gb:.2f} GB")
            print(f"    Reserved:  {reserved_gb:.2f} GB")
            print(f"    Peak:      {peak_gb:.2f} GB ({vram_percent:.1f}%)")

            results.append({
                'batch_size': batch_size,
                'peak_gb': peak_gb,
                'vram_percent': vram_percent,
                'success': True
            })

            # Stop if we're using >90% VRAM (risky)
            if vram_percent > 90:
                print(f"  ⚠ Warning: Using {vram_percent:.1f}% VRAM - stopping test")
                break

        except torch.cuda.OutOfMemoryError:
            print(f"  ✗ OOM - Batch size {batch_size} too large")
            results.append({
                'batch_size': batch_size,
                'success': False
            })
            break

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'batch_size': batch_size,
                'success': False,
                'error': str(e)
            })
            break

    # Analyze results
    print("\n" + "="*70)
    print("OPTIMIZATION RESULTS")
    print("="*70)

    successful = [r for r in results if r['success']]

    if not successful:
        print("⚠ No successful batch sizes found - using default (8)")
        return 8

    # Find batch size using 70-85% VRAM (optimal range)
    optimal = None
    for result in reversed(successful):
        if 70 <= result['vram_percent'] <= 85:
            optimal = result
            break

    # If none in optimal range, take largest successful that's <85%
    if optimal is None:
        safe_results = [r for r in successful if r['vram_percent'] < 85]
        if safe_results:
            optimal = safe_results[-1]
        else:
            optimal = successful[0]  # Fallback to smallest

    print(f"\n✓ RECOMMENDED BATCH SIZE: {optimal['batch_size']}")
    print(f"  Peak VRAM: {optimal['peak_gb']:.2f} GB ({optimal['vram_percent']:.1f}%)")
    print(f"  Speedup vs batch_size=8: ~{optimal['batch_size']/8:.1f}x")
    print(f"\nUpdate tests/test_stage2_codet5_small.py with:")
    print(f"  batch_size={optimal['batch_size']}")
    print("="*70)

    return optimal['batch_size']


if __name__ == "__main__":
    optimal_batch_size = test_batch_sizes()
    sys.exit(0)
