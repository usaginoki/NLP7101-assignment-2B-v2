#!/bin/bash

# Run selected classifiers: LogisticRegression, XGBoost, SVM_Linear, MLP

LOGDIR="outputs/model_search/logs"
mkdir -p "$LOGDIR"

echo "========================================================================"
echo "SELECTED CLASSIFIERS - GPU-ACCELERATED SEARCH"
echo "========================================================================"
echo ""
echo "Running 4 classifiers sequentially:"
echo "  1. Logistic Regression (CPU, ~10-15 min)"
echo "  2. XGBoost (GPU, ~2-3 min)"
echo "  3. SVM Linear (CPU, ~15-20 min)"
echo "  4. MLP (GPU, ~5-10 min)"
echo ""
echo "Total estimated time: ~35-50 minutes"
echo "Logs: $LOGDIR/"
echo "========================================================================"
echo ""

# 1. Logistic Regression (CPU-only, ~10-15 min)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Logistic Regression..."
uv run python scripts/train_stage1_model_search.py \
    --classifier LogisticRegression \
    --gpu \
    2>&1 | tee "$LOGDIR/logistic_regression.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Logistic Regression complete"
echo ""

# 2. XGBoost (GPU-accelerated, ~2-3 min)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting XGBoost..."
uv run python scripts/train_stage1_model_search.py \
    --classifier XGBoost \
    --gpu \
    2>&1 | tee "$LOGDIR/xgboost.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ XGBoost complete"
echo ""

# 3. SVM Linear (CPU-only, ~15-20 min)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting SVM Linear..."
uv run python scripts/train_stage1_model_search.py \
    --classifier SVM_Linear \
    --gpu \
    2>&1 | tee "$LOGDIR/svm_linear.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ SVM Linear complete"
echo ""

# 4. MLP (GPU-accelerated PyTorch, ~5-10 min)
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting MLP..."
uv run python scripts/train_stage1_model_search.py \
    --classifier MLP \
    --gpu \
    2>&1 | tee "$LOGDIR/mlp.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ MLP complete"
echo ""

echo "========================================================================"
echo "ALL CLASSIFIERS COMPLETE"
echo "========================================================================"
echo ""
echo "Results:"
cat outputs/model_search/experiments.json | python -m json.tool 2>/dev/null || echo "experiments.json not found"
echo ""
echo "Best model comparison:"
cat outputs/model_search/best_model_comparison.json | python -m json.tool 2>/dev/null || echo "best_model_comparison.json not found"
