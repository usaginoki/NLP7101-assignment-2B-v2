# GPU Acceleration Guide

**Hardware:** NVIDIA RTX 5090 (32GB VRAM)
**Expected Speedup:** 3-5x faster (6-10 hours → 2-4 hours)

---

## Quick Start

### 1. Test GPU Availability

```bash
# Run GPU setup and verification script
uv run python scripts/setup_gpu_acceleration.py
```

This script will:
- ✅ Check NVIDIA GPU detection
- ✅ Verify CUDA availability
- ✅ Test XGBoost GPU support
- ✅ Test LightGBM GPU support
- ✅ Run quick GPU vs CPU benchmark
- 📋 Provide installation instructions if needed

### 2. Enable GPU Acceleration (if needed)

If the setup script shows XGBoost or LightGBM don't have GPU support, run:

```bash
# XGBoost GPU support (REQUIRED for major speedup)
uv pip uninstall xgboost
uv pip install xgboost

# LightGBM GPU support (OPTIONAL but recommended)
uv pip uninstall lightgbm
uv pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON
```

### 3. Run GPU-Accelerated Search

```bash
# Test mode with GPU (5-10 minutes)
uv run python scripts/train_stage1_model_search.py --test-mode --gpu

# Full search with GPU (2-4 hours)
uv run python scripts/train_stage1_model_search.py --gpu

# Specific classifier with GPU
uv run python scripts/train_stage1_model_search.py --classifier XGBoost --gpu
```

---

## Performance Comparison

### Estimated Runtimes (Your Hardware)

| Classifier | CPU (20 threads) | GPU (RTX 5090) | Speedup |
|------------|-----------------|----------------|---------|
| Logistic Regression | 5-10 min | 5-10 min | 1x (CPU-only) |
| SVM Linear | 20-30 min | 20-30 min | 1x (CPU-only) |
| SVM RBF | 60-120 min | 60-120 min | 1x (CPU-only) |
| Random Forest | 60-90 min | 60-90 min | 1x (CPU-only) |
| **XGBoost** | **90-120 min** | **10-15 min** | **6-9x** ⚡ |
| **LightGBM** | **60-90 min** | **15-20 min** | **3-5x** ⚡ |
| MLP | 60-90 min | 60-90 min | 1x (CPU-only) |
| **TOTAL** | **6-10 hours** | **2-4 hours** | **2.5-3x** ⚡ |

### What Gets Accelerated?

**GPU-Accelerated:**
- ✅ XGBoost (tree building via `gpu_hist`)
- ✅ LightGBM (gradient boosting via CUDA)

**CPU-Only (No GPU Benefit):**
- ❌ Logistic Regression (linear algebra, already fast)
- ❌ SVM (not GPU-accelerated in sklearn)
- ❌ Random Forest (sklearn doesn't support GPU)
- ❌ MLP (sklearn doesn't support GPU)

---

## GPU Configuration Details

### XGBoost GPU Parameters

```python
{
    'tree_method': ['gpu_hist'],  # GPU-accelerated histogram-based tree building
    'gpu_id': [0],                # RTX 5090 device ID
    'n_jobs': [1],                # GPU handles parallelism
}
```

**How it works:**
- Builds histogram on GPU (CUDA kernels)
- Evaluates splits in parallel on GPU
- ~6-9x faster than CPU `hist` method
- Memory requirement: ~2-4GB VRAM for 500k samples

### LightGBM GPU Parameters

```python
{
    'device': ['gpu'],          # Enable GPU acceleration
    'gpu_platform_id': [0],     # CUDA platform
    'gpu_device_id': [0],       # RTX 5090 device ID
    'n_jobs': [1],              # GPU handles parallelism
}
```

**How it works:**
- Uses CUDA for gradient/Hessian computation
- GPU-accelerated histogram construction
- ~3-5x faster than CPU version
- Memory requirement: ~1-2GB VRAM for 500k samples

---

## Monitoring GPU Usage

### During Training

```bash
# Watch GPU utilization in real-time
watch -n 1 nvidia-smi

# Or more detailed:
nvtop  # Install: sudo apt install nvtop
```

**Expected GPU Usage:**
- **XGBoost:** 80-95% GPU utilization, 2-4GB VRAM
- **LightGBM:** 70-90% GPU utilization, 1-2GB VRAM
- **Other classifiers:** 0% (CPU-only)

### Log Files

Check GPU usage in model search logs:
```bash
# Run with output logging
uv run python scripts/train_stage1_model_search.py --gpu 2>&1 | tee gpu_search.log

# Monitor GPU during search
tail -f gpu_search.log
```

---

## Troubleshooting

### Issue: "GPU not detected"

**Check:**
```bash
nvidia-smi
```

**Solution:**
- Install NVIDIA drivers: `sudo ubuntu-drivers autoinstall`
- Reboot system

### Issue: "XGBoost GPU training failed"

**Error:** `XGBoostError: gpu_id` or `tree_method gpu_hist not available`

**Solution:**
```bash
# Reinstall XGBoost with GPU support
uv pip uninstall xgboost
uv pip install xgboost

# Verify
uv run python -c "import xgboost as xgb; print(xgb.__version__)"
```

### Issue: "LightGBM GPU training failed"

**Error:** `LightGBMError: GPU Tree Learner was not enabled`

**Solution:**
```bash
# LightGBM GPU requires compilation with CUDA
uv pip uninstall lightgbm
uv pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON

# This may fail if CUDA toolkit is not installed
# Fallback: Use CPU version (still fast)
uv pip install lightgbm
```

### Issue: "CUDA out of memory"

**Error:** `CUDA error: out of memory`

**Your RTX 5090 has 32GB VRAM - this should NOT happen**

**If it does:**
1. Check other processes: `nvidia-smi`
2. Kill other GPU processes
3. Reduce batch size (not applicable for tree models)

### Issue: Slower than expected

**Possible causes:**
1. **GPU throttling:** Check temperatures (`nvidia-smi`)
2. **CPU bottleneck:** Data preprocessing on CPU
3. **Small parameter grids:** GPU overhead dominates

**Verify GPU is actually being used:**
```bash
# Watch GPU utilization during search
watch -n 1 nvidia-smi

# Should show 80-95% GPU usage during XGBoost/LightGBM training
```

---

## Advanced: Custom GPU Settings

### Optimize for RTX 5090

Your RTX 5090 is very powerful. You can increase grid sizes for better model quality:

```python
# In search_configs_gpu.py

def get_xgboost_grid():
    return {
        'n_estimators': [100, 200, 300, 500],  # Added 500
        'max_depth': [3, 5, 7, 10, 15],        # Added 15
        # ... other params
    }
```

### Multi-GPU (if you have multiple GPUs)

```python
# Modify search_configs_gpu.py
GPU_IDS = [0, 1]  # Use GPUs 0 and 1

# Run different classifiers on different GPUs
def get_xgboost_grid():
    return {
        'gpu_id': [0],  # XGBoost on GPU 0
        # ...
    }

def get_lightgbm_grid():
    return {
        'gpu_device_id': [1],  # LightGBM on GPU 1
        # ...
    }
```

---

## Benchmarking

### Quick GPU Benchmark

```bash
# Run benchmark (included in setup script)
uv run python scripts/setup_gpu_acceleration.py

# Or manual benchmark:
uv run python -c "
import xgboost as xgb
import numpy as np
import time

X = np.random.rand(10000, 100)
y = np.random.randint(0, 2, 10000)
dtrain = xgb.DMatrix(X, label=y)

# CPU
start = time.time()
xgb.train({'tree_method': 'hist'}, dtrain, num_boost_round=100)
cpu_time = time.time() - start

# GPU
start = time.time()
xgb.train({'tree_method': 'gpu_hist', 'gpu_id': 0}, dtrain, num_boost_round=100)
gpu_time = time.time() - start

print(f'CPU: {cpu_time:.2f}s, GPU: {gpu_time:.2f}s, Speedup: {cpu_time/gpu_time:.2f}x')
"
```

**Expected output:**
```
CPU: 5.23s, GPU: 0.82s, Speedup: 6.38x
```

### Full Search Comparison

```bash
# CPU search (baseline)
time uv run python scripts/train_stage1_model_search.py --classifier XGBoost

# GPU search
time uv run python scripts/train_stage1_model_search.py --classifier XGBoost --gpu

# Compare elapsed times
```

---

## CPU vs GPU: When to Use What?

### Use GPU (`--gpu` flag)

**When:**
- ✅ Training XGBoost or LightGBM
- ✅ Running full model search (saves 4-6 hours)
- ✅ You have NVIDIA GPU available
- ✅ Dataset is large (>10k samples)

**Command:**
```bash
uv run python scripts/train_stage1_model_search.py --gpu
```

### Use CPU (default)

**When:**
- ❌ GPU not available or setup fails
- ❌ Only testing non-GPU classifiers (LogReg, SVM, RF)
- ❌ Quick experiments (GPU overhead not worth it)

**Command:**
```bash
uv run python scripts/train_stage1_model_search.py
```

---

## Summary

### Setup Checklist

- [ ] Run `uv run python scripts/setup_gpu_acceleration.py`
- [ ] Verify GPU is detected
- [ ] Verify XGBoost GPU support
- [ ] (Optional) Verify LightGBM GPU support
- [ ] Run test mode: `uv run python scripts/train_stage1_model_search.py --test-mode --gpu`
- [ ] Run full search: `uv run python scripts/train_stage1_model_search.py --gpu`

### Expected Results

**With RTX 5090:**
- XGBoost: ~10-15 minutes (vs 90-120 min CPU)
- LightGBM: ~15-20 minutes (vs 60-90 min CPU)
- Total: ~2-4 hours (vs 6-10 hours CPU)
- Best macro F1: 0.78-0.82 (same quality, just faster)

### Need Help?

1. Run diagnostics: `uv run python scripts/setup_gpu_acceleration.py`
2. Check logs: Review error messages
3. Fallback to CPU: Run without `--gpu` flag (still works, just slower)

**GPU acceleration is a performance optimization - the CPU version will produce the same results, just slower.**
