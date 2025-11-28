#!/usr/bin/env python3
"""
GPU Acceleration Setup and Verification

Tests GPU availability and compatibility for:
- XGBoost (gpu_hist)
- LightGBM (GPU device)

Provides installation instructions if GPU support is missing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    print("\n" + "="*80)
    print("CHECKING NVIDIA GPU")
    print("="*80)

    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5
        )

        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(',')
            print(f"✓ GPU Detected: {gpu_info[0].strip()}")
            print(f"  VRAM: {gpu_info[1].strip()}")
            print(f"  Driver: {gpu_info[2].strip()}")
            return True
        else:
            print("✗ No NVIDIA GPU detected")
            return False

    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ nvidia-smi not found or timeout")
        return False


def check_cuda():
    """Check CUDA availability"""
    print("\n" + "="*80)
    print("CHECKING CUDA")
    print("="*80)

    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'],
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = [line for line in result.stdout.split('\n')
                      if 'release' in line.lower()]
            if version:
                print(f"✓ CUDA Toolkit: {version[0].strip()}")
                return True
        print("⚠ CUDA toolkit not found (optional for pre-built wheels)")
        return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("⚠ nvcc not found (optional for pre-built wheels)")
        return False


def check_xgboost_gpu():
    """Check if XGBoost has GPU support"""
    print("\n" + "="*80)
    print("CHECKING XGBOOST GPU SUPPORT")
    print("="*80)

    try:
        import xgboost as xgb
        print(f"✓ XGBoost version: {xgb.__version__}")

        # Try to use GPU
        try:
            import numpy as np
            # Create small test dataset
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 2, 100)

            # Try GPU training (XGBoost 3.1+ API)
            dtrain = xgb.DMatrix(X, label=y)
            params = {
                'device': 'cuda',  # XGBoost 3.1+ uses 'device' instead of 'gpu_id'
                'tree_method': 'hist',  # Auto-uses GPU when device='cuda'
                'objective': 'binary:logistic'
            }

            model = xgb.train(params, dtrain, num_boost_round=1, verbose_eval=False)
            print("✓ GPU training successful!")
            print(f"  Tree method: gpu_hist")
            print(f"  GPU acceleration: ENABLED")
            return True

        except Exception as e:
            print(f"✗ GPU training failed: {str(e)}")
            print("  XGBoost is installed but GPU support is not available")
            return False

    except ImportError:
        print("✗ XGBoost not installed")
        return False


def check_lightgbm_gpu():
    """Check if LightGBM has GPU support"""
    print("\n" + "="*80)
    print("CHECKING LIGHTGBM GPU SUPPORT")
    print("="*80)

    try:
        import lightgbm as lgb
        print(f"✓ LightGBM version: {lgb.__version__}")

        # Try to use GPU
        try:
            import numpy as np
            # Create small test dataset
            X = np.random.rand(100, 10)
            y = np.random.randint(0, 2, 100)

            # Try GPU training
            dtrain = lgb.Dataset(X, label=y)
            params = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'objective': 'binary',
                'verbosity': -1
            }

            model = lgb.train(params, dtrain, num_boost_round=1)
            print("✓ GPU training successful!")
            print(f"  Device: gpu")
            print(f"  GPU acceleration: ENABLED")
            return True

        except Exception as e:
            print(f"⚠ GPU training failed: {str(e)}")
            print("  LightGBM is installed but GPU support may not be available")
            print("  (This is optional - CPU version will work)")
            return False

    except ImportError:
        print("✗ LightGBM not installed")
        return False


def print_setup_instructions(xgb_gpu, lgb_gpu):
    """Print installation instructions for missing GPU support"""
    print("\n" + "="*80)
    print("SETUP INSTRUCTIONS")
    print("="*80)

    if not xgb_gpu:
        print("\n📦 XGBoost GPU Setup:")
        print("-" * 80)
        print("Current XGBoost installation doesn't support GPU.")
        print("\nOption 1: Install from pip (recommended)")
        print("  uv pip uninstall xgboost")
        print("  uv pip install xgboost")
        print("  # Recent XGBoost wheels include GPU support")
        print("\nOption 2: Build from source (advanced)")
        print("  # Requires CUDA toolkit and CMake")
        print("  pip install xgboost --no-binary xgboost")

    if not lgb_gpu:
        print("\n📦 LightGBM GPU Setup (Optional):")
        print("-" * 80)
        print("Current LightGBM installation doesn't support GPU.")
        print("Note: This is OPTIONAL - CPU version works fine.")
        print("\nIf you want GPU acceleration:")
        print("  uv pip uninstall lightgbm")
        print("  uv pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON")
        print("  # Requires: CUDA toolkit, CMake, and OpenCL")

    if xgb_gpu and lgb_gpu:
        print("\n✅ All GPU acceleration is ENABLED!")
        print("\nYou can now use the GPU-optimized search:")
        print("  uv run python scripts/train_stage1_model_search.py --gpu")


def benchmark_gpu_vs_cpu():
    """Quick benchmark to show GPU speedup"""
    print("\n" + "="*80)
    print("GPU vs CPU BENCHMARK")
    print("="*80)

    try:
        import xgboost as xgb
        import numpy as np
        import time

        # Create larger test dataset
        print("\nCreating test dataset (10k samples, 100 features)...")
        X = np.random.rand(10000, 100)
        y = np.random.randint(0, 2, 10000)
        dtrain = xgb.DMatrix(X, label=y)

        # CPU benchmark
        print("\n⏱ CPU Training (tree_method='hist')...")
        params_cpu = {
            'tree_method': 'hist',
            'objective': 'binary:logistic',
            'max_depth': 6,
            'n_jobs': -1
        }
        start = time.time()
        xgb.train(params_cpu, dtrain, num_boost_round=100, verbose_eval=False)
        cpu_time = time.time() - start
        print(f"  Time: {cpu_time:.2f} seconds")

        # GPU benchmark (XGBoost 3.1+ API)
        print("\n⚡ GPU Training (device='cuda')...")
        params_gpu = {
            'device': 'cuda',  # XGBoost 3.1+ uses 'device' instead of 'gpu_id'
            'tree_method': 'hist',  # Auto-uses GPU when device='cuda'
            'objective': 'binary:logistic',
            'max_depth': 6
        }
        start = time.time()
        xgb.train(params_gpu, dtrain, num_boost_round=100, verbose_eval=False)
        gpu_time = time.time() - start
        print(f"  Time: {gpu_time:.2f} seconds")

        # Speedup
        speedup = cpu_time / gpu_time
        print(f"\n🚀 GPU Speedup: {speedup:.2f}x faster")

        if speedup > 3:
            print("  ✓ Excellent GPU acceleration!")
        elif speedup > 1.5:
            print("  ✓ Good GPU acceleration")
        else:
            print("  ⚠ Limited speedup (may need driver updates)")

    except Exception as e:
        print(f"\n⚠ Benchmark failed: {str(e)}")


def main():
    """Run all checks"""
    print("\n" + "="*80)
    print("GPU ACCELERATION SETUP AND VERIFICATION")
    print("="*80)
    print("\nThis script will check if your system is ready for GPU-accelerated")
    print("model training with XGBoost and LightGBM.")

    # Check hardware
    has_gpu = check_nvidia_gpu()
    has_cuda = check_cuda()

    if not has_gpu:
        print("\n" + "="*80)
        print("❌ NO GPU DETECTED")
        print("="*80)
        print("\nGPU acceleration requires an NVIDIA GPU.")
        print("You can still run the model search on CPU (it will just take longer).")
        print("\nEstimated times:")
        print("  CPU only: 6-10 hours")
        print("  With GPU: 2-4 hours")
        return

    # Check software
    xgb_gpu = check_xgboost_gpu()
    lgb_gpu = check_lightgbm_gpu()

    # Print instructions if needed
    if not xgb_gpu or not lgb_gpu:
        print_setup_instructions(xgb_gpu, lgb_gpu)
    else:
        print("\n" + "="*80)
        print("✅ GPU ACCELERATION READY")
        print("="*80)
        print("\nBoth XGBoost and LightGBM have GPU support enabled!")
        print("\nRunning quick benchmark...")
        benchmark_gpu_vs_cpu()

        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Test GPU-accelerated search:")
        print("   uv run python scripts/train_stage1_model_search.py --test-mode --gpu")
        print("\n2. Run full GPU-accelerated search:")
        print("   uv run python scripts/train_stage1_model_search.py --gpu")
        print("\n3. Expected performance:")
        print("   - XGBoost: ~10-15 minutes (vs 90-120 min CPU)")
        print("   - LightGBM: ~15-20 minutes (vs 60-90 min CPU)")
        print("   - Total: ~2-4 hours (vs 6-10 hours CPU)")


if __name__ == '__main__':
    main()
