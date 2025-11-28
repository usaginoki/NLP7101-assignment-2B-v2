# Hardware Optimization Analysis

**System:** Intel Core i9-10900X (10 cores, 20 threads @ 3.70GHz)
**RAM:** 125GB (120GB available)
**Date:** November 27, 2025

---

## Current Optimization Status: ✅ GOOD (with room for improvement)

### What's Already Optimized

#### 1. CPU Parallelization ✅
**Current Setting:** `n_jobs=-1` (uses all available cores)
**Utilization:** All 20 threads will be used

```python
# In search_configs.py
'n_jobs': [-1]  # LogisticRegression, RandomForest, XGBoost, LightGBM

# In search_runner.py (GridSearchCV/RandomizedSearchCV)
n_jobs=-1  # Parallel parameter search
```

**Impact:**
- Random Forest: Parallelizes tree building across 20 cores → **10-20x speedup**
- XGBoost/LightGBM: Parallelizes node splitting → **5-10x speedup**
- GridSearchCV: Parallelizes parameter combinations → **up to 20x speedup**

#### 2. Memory Utilization ✅
**Dataset Size:**
- Training: 500k × 36 features × 8 bytes = ~144 MB (float64)
- Validation: 100k × 36 features × 8 bytes = ~29 MB
- Total: ~173 MB

**Your Available RAM:** 120 GB
**Memory Safety Margin:** 700x headroom

**Verdict:** Zero memory constraints. Can easily handle:
- Full dataset in memory
- Multiple model copies during grid search
- Intermediate results and buffers

#### 3. Algorithm-Specific Optimizations ✅

**XGBoost:**
```python
'tree_method': ['hist']  # Histogram-based (optimized for large datasets)
```
- Uses binning for faster splits
- Reduces memory footprint
- 3-5x faster than exact method on 500k samples

**LightGBM:**
```python
# Already optimized by default
# Uses Gradient-based One-Side Sampling (GOSS)
# Uses Exclusive Feature Bundling (EFB)
```
- Designed for large datasets
- Faster than XGBoost on 500k samples

---

## Recommendations for Your Hardware

### Recommended Improvements

#### 1. **Enable Intel MKL for NumPy/SciPy** (Potential 2-3x speedup)

Your i9-10900X supports AVX-512 instructions. Intel MKL can accelerate linear algebra operations.

**Check Current BLAS Backend:**
```bash
python -c "import numpy; numpy.show_config()"
```

**If not using MKL, install:**
```bash
uv pip install mkl mkl-service
```

**Expected Impact:**
- LogisticRegression: 2-3x faster (BLAS-heavy)
- SVM: 2-4x faster (matrix operations)
- Minimal impact on tree-based models (XGBoost, LightGBM, RF)

---

#### 2. **Optimize Thread Count for Hyperthreading**

**Current:** `n_jobs=-1` (uses all 20 threads)
**Issue:** Hyperthreading can cause thread contention for CPU-bound tasks

**Recommendation:** Test with physical cores only (10 threads)

**Create optimized config:**
```python
# src/models/model_search/search_configs_optimized.py

import os

# Detect physical cores (exclude hyperthreading)
PHYSICAL_CORES = 10  # Your CPU: 10 physical cores
OPTIMAL_JOBS = PHYSICAL_CORES  # Use physical cores for CPU-bound tasks

def get_logistic_regression_grid():
    return {
        # ... existing parameters ...
        'n_jobs': [OPTIMAL_JOBS],  # Changed from -1 to 10
    }
```

**Why?**
- Tree-based models (RF, XGBoost, LightGBM) are CPU-bound
- Hyperthreading provides ~30% benefit for I/O-bound tasks
- For CPU-bound tasks: 10 physical cores ≈ 13-15 hyperthreaded cores (not 20)

**A/B Test Recommendation:**
```bash
# Test with all threads (current)
export OMP_NUM_THREADS=20
uv run python scripts/train_stage1_model_search.py --classifier RandomForest

# Test with physical cores only
export OMP_NUM_THREADS=10
uv run python scripts/train_stage1_model_search.py --classifier RandomForest

# Compare elapsed times
```

**Expected Impact:** 0-15% speedup (depends on workload)

---

#### 3. **Enable XGBoost GPU Acceleration** (Potential 5-10x speedup)

**Check if NVIDIA GPU is available:**
```bash
nvidia-smi
```

**If GPU available, modify XGBoost config:**
```python
# In search_configs.py
def get_xgboost_grid():
    return {
        # ... existing parameters ...
        'tree_method': ['hist'],  # Keep this for CPU
        # OR (if GPU available):
        # 'tree_method': ['gpu_hist'],
        # 'gpu_id': [0],
    }
```

**Install GPU-enabled XGBoost:**
```bash
uv pip install xgboost[gpu]
```

**Expected Impact:** 5-10x speedup for XGBoost searches

---

#### 4. **Optimize SVM Sampling Strategy**

**Current Sampling:**
- Linear SVM: 20% (100k samples)
- RBF SVM: 10% (50k samples)

**Your Hardware:** Can handle more samples efficiently

**Recommended Adjustment:**
```python
# In search_configs.py
'SVM_Linear': {
    'sampling_fraction': 0.3,  # Increase from 0.2 to 0.3 (150k samples)
    'timeout_seconds': 1800
},
'SVM_RBF': {
    'sampling_fraction': 0.15,  # Increase from 0.1 to 0.15 (75k samples)
    'timeout_seconds': 7200
}
```

**Rationale:**
- Your CPU can handle larger sample sizes
- More training data → better hyperparameter estimates
- Still avoids O(n²) explosion (RBF SVM)

**Trade-off:**
- +50% training time for SVM
- Better final model quality

---

#### 5. **Increase GridSearchCV Verbosity for Monitoring**

**Current:** `verbose=2` (moderate output)
**Recommendation:** Keep as-is, but add progress monitoring

**Optional: Add progress tracking:**
```python
# In search_runner.py
from tqdm.auto import tqdm

# Wrap GridSearchCV with progress bar
# (requires tqdm: uv add tqdm)
```

---

#### 6. **Optimize I/O for Data Loading**

**Current:** Loads from CSV files (pandas default)

**Check if using fast CSV parser:**
```python
# In feature_loader.py
# Ensure using C engine (default, but verify)
pd.read_csv(..., engine='c')  # Fastest
```

**Expected Impact:** Minimal (data loading is <1% of total time)

---

## Hardware-Specific Tuning

### Classifier-Specific Recommendations

#### Random Forest (Your Hardware)
```python
'n_estimators': [200, 300, 500],  # ✅ Current is good
'n_jobs': [10],  # Use physical cores (test vs -1)
'max_features': ['sqrt'],  # ✅ Already optimized
```

**Expected Runtime on Your Hardware:**
- 100 iterations × 500k samples: **~60 minutes**
- With 20 threads: **~45 minutes** (current)
- With 10 threads: **~50 minutes** (may vary)

#### XGBoost (Your Hardware)
```python
'tree_method': ['hist'],  # ✅ Already optimized
'n_jobs': [10],  # Use physical cores
# If GPU available:
# 'tree_method': ['gpu_hist'],
```

**Expected Runtime on Your Hardware:**
- 150 iterations × 500k samples: **~90 minutes** (CPU)
- With GPU: **~15-20 minutes**

#### LightGBM (Your Hardware)
```python
'n_jobs': [10],  # Use physical cores
'num_threads': [10],  # LightGBM-specific
```

**Expected Runtime on Your Hardware:**
- 200 iterations × 500k samples: **~60 minutes**

---

## Memory Optimization (Not Needed, but Good to Know)

### Current Memory Usage (Estimated)

| Component | Memory Usage |
|-----------|--------------|
| Training data (500k × 36) | 144 MB |
| Validation data (100k × 36) | 29 MB |
| Combined data (600k × 36) | 173 MB |
| Random Forest (200 trees) | ~500 MB |
| XGBoost (300 trees) | ~300 MB |
| GridSearchCV buffer | ~1-2 GB |
| **Total Peak Usage** | **~3-4 GB** |

**Your Available RAM:** 120 GB
**Safety Margin:** 30-40x headroom

**Verdict:** Memory is not a bottleneck. No optimizations needed.

---

## Disk I/O Optimization

### Model Saving (Joblib)

**Current:** Default compression
**Recommendation:** Use faster compression for large models

```python
# In save_best_model() function
joblib.dump(best_model, model_path, compress=3)  # Current (likely)

# For faster saves:
joblib.dump(best_model, model_path, compress=1)  # Faster, larger files

# For smaller files:
joblib.dump(best_model, model_path, compress=9)  # Slower, smaller files
```

**Trade-off:** Compression level 1 → 3-5x faster saves, ~30% larger files

---

## Benchmark: Expected Runtimes on Your Hardware

### Full Search (All 7 Classifiers)

| Classifier | Original Estimate | Optimized Estimate | Notes |
|------------|------------------|-------------------|-------|
| Logistic Regression | 5-10 min | **3-6 min** | With MKL |
| SVM Linear | 20-30 min | **15-25 min** | 30% sampling |
| SVM RBF | 60-120 min | **45-90 min** | 15% sampling |
| Random Forest | 60-90 min | **45-70 min** | Physical cores |
| XGBoost | 90-120 min | **60-90 min** (CPU) / **15-20 min** (GPU) | Hist method / GPU |
| LightGBM | 60-90 min | **40-60 min** | Already optimized |
| MLP | 60-90 min | **50-80 min** | Early stopping |

**Total (Conservative):** 6-10 hours → **4-7 hours** (CPU only)
**Total (With GPU):** → **3-5 hours**

---

## Quick Optimization Script

Create an optimized version with hardware-specific settings:

```bash
# Create optimized config
cat > src/models/model_search/search_configs_hw_optimized.py << 'EOF'
"""
Hardware-Optimized Search Configurations
Tuned for Intel i9-10900X (10 cores, 20 threads, 125GB RAM)
"""

# Use physical cores for CPU-bound tasks
OPTIMAL_THREADS = 10

def get_logistic_regression_grid():
    return {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2', 'l1'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [200, 500, 1000],
        'class_weight': ['balanced'],
        'random_state': [42],
        'n_jobs': [OPTIMAL_THREADS]  # Optimized for your CPU
    }

# ... (copy other functions with OPTIMAL_THREADS)
EOF
```

---

## Action Items (Ordered by Impact)

### High Impact (Recommended)
1. ✅ **Verify MKL is installed** (2-3x speedup for LogReg/SVM)
   ```bash
   python -c "import numpy; numpy.show_config()"
   ```

2. ⚡ **Check for GPU availability** (5-10x speedup for XGBoost)
   ```bash
   nvidia-smi
   ```

3. 🔧 **Increase SVM sampling** (better model quality)
   - Linear: 0.2 → 0.3
   - RBF: 0.1 → 0.15

### Medium Impact (Optional)
4. 🧵 **A/B test thread count** (0-15% speedup)
   - Test `n_jobs=10` vs `n_jobs=-1`

5. 📊 **Add progress monitoring** (visibility, no speedup)
   ```bash
   uv add tqdm
   ```

### Low Impact (Not Worth It)
6. ❌ **Disk I/O optimization** (negligible impact)
7. ❌ **Memory optimization** (not needed with 120GB RAM)

---

## Final Verdict

### Current Implementation: ✅ **WELL-OPTIMIZED**

**What's Already Great:**
- ✅ Parallel processing enabled (`n_jobs=-1`)
- ✅ Efficient algorithms (XGBoost hist, LightGBM defaults)
- ✅ Reasonable timeouts
- ✅ Memory-efficient data structures

**Potential Improvements:**
- 🎯 **MKL for NumPy** → 2-3x speedup for linear models
- 🎯 **GPU for XGBoost** → 5-10x speedup (if available)
- 🎯 **More SVM samples** → Better model quality

**Bottom Line:**
Your hardware is **excellent** for this workload. The implementation will run efficiently as-is. The suggested optimizations could reduce total runtime from **6-10 hours** to **3-5 hours** (with GPU) or **4-7 hours** (CPU only).

---

**Recommendation:** Run the full search as-is first. If you want faster iterations, implement the MKL and GPU optimizations afterward.
