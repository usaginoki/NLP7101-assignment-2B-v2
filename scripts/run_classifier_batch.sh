#!/bin/bash
# Run 4 classifiers sequentially with GPU acceleration
# Logs saved to outputs/model_search/logs/

set -e  # Exit on error

LOGDIR="outputs/model_search/logs"
mkdir -p "$LOGDIR"

echo "========================================================================"
echo "GPU-ACCELERATED HYPERPARAMETER SEARCH BATCH"
echo "========================================================================"
echo ""
echo "Running 5 classifiers sequentially:"
echo "  1. Logistic Regression (CPU, ~7-10 min)"
echo "  2. SVM Linear (CPU, ~20-30 min)"
echo "  3. XGBoost (GPU, ~2-3 min)"
echo "  4. LightGBM (GPU, ~7-8 min)"
echo "  5. MLP (GPU, ~5-10 min)"
echo ""
echo "Total estimated time: ~40-60 minutes"
echo "Logs: $LOGDIR/"
echo "========================================================================"
echo ""

# 1. Logistic Regression
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Logistic Regression..."
uv run python scripts/train_stage1_model_search.py \
    --classifier LogisticRegression \
    --gpu \
    2>&1 | tee "$LOGDIR/logistic_regression.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Logistic Regression complete"
echo ""

# 2. SVM Linear
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting SVM Linear..."
uv run python scripts/train_stage1_model_search.py \
    --classifier SVM_Linear \
    --gpu \
    2>&1 | tee "$LOGDIR/svm_linear.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ SVM Linear complete"
echo ""

# 3. XGBoost (GPU-accelerated)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting XGBoost (GPU)..."
uv run python scripts/train_stage1_model_search.py \
    --classifier XGBoost \
    --gpu \
    2>&1 | tee "$LOGDIR/xgboost.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ XGBoost complete"
echo ""

# 4. LightGBM (GPU-accelerated)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting LightGBM (GPU)..."
uv run python scripts/train_stage1_model_search.py \
    --classifier LightGBM \
    --gpu \
    2>&1 | tee "$LOGDIR/lightgbm.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ LightGBM complete"
echo ""

# 5. MLP (GPU-accelerated PyTorch, ~5-10 min)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MLP..."
uv run python scripts/train_stage1_model_search.py \
    --classifier MLP \
    --gpu \
    2>&1 | tee "$LOGDIR/mlp.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ MLP complete"
echo ""

echo "========================================================================"
echo "ALL CLASSIFIERS COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved to:"
echo "  - Experiments: outputs/model_search/experiments.json"
echo "  - Comparison: outputs/model_search/best_model_comparison.json"
echo "  - Logs: $LOGDIR/"
echo ""
echo "View results:"
echo "  cat outputs/model_search/experiments.json | jq '.[] | {classifier, val_macro_f1, elapsed_time_seconds}'"
echo ""
