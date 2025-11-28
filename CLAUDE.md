# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a 2-stage cascaded classification pipeline for AI-generated code detection (SemEval 2026 Task 13 Subtask B). The system achieves **90.35% accuracy** combining XGBoost (Stage 1) and CodeT5-220m (Stage 2).

**Architecture**: Binary detection (Human vs AI) → Multi-class classification (10 AI families)

## Essential Commands

### Setup
```bash
# Install dependencies
uv sync

# All Python commands must be run with uv
uv run python <script>
```

### Training Pipeline

**Stage 1 (XGBoost Binary Classifier)**:
```bash
uv run python src/training/train_stage1.py
# Output: models/stage1/XGBoost/classifier.pkl (96.6% accuracy)
# Creates: models/stage1/scaler.pkl (shared with Stage 2)
```

**Stage 2 (CodeT5 Multi-class Classifier)**:
```bash
# IMPORTANT: Train Stage 1 first! Stage 2 requires the scaler.

# With GPU optimization (recommended for RTX 5090 or similar):
uv run python src/training/train_stage2_codet5.py \
    --batch-size 256 \
    --max-epochs 5 \
    --learning-rate 2e-4

# For smaller GPUs:
uv run python src/training/train_stage2_codet5.py --batch-size 64

# Output: models/stage2_codet5/classifier.pth (0.77 MB, only classification head)
# Training time: ~16 min (batch_size=256), ~40 min (batch_size=32)
```

### Evaluation
```bash
# Evaluate full pipeline (XGBoost + CodeT5)
uv run python src/evaluation/evaluate_pipeline_codet5.py --split validation

# Individual stages
uv run python src/evaluation/evaluate_stage1.py --split validation
uv run python src/evaluation/evaluate_stage2.py --split validation
```

### Testing
```bash
# Test CodeT5 implementation (loads, trains, saves correctly)
uv run python tests/test_codet5_verbose.py

# Compare CodeT5-220m vs CodeT5-770m loading
uv run python tests/test_220m_model.py
```

### Submission Generation
```bash
# Generate predictions for test set
uv run python scripts/generate_submission_codet5.py

# Output: outputs/submission.csv (1,000 predictions)
# Format: CSV with 'ID' and 'label' columns (labels 0-10)
# Processing time: ~20 seconds

# With custom options:
uv run python scripts/generate_submission_codet5.py \
    --test-file data/test.parquet \
    --output outputs/submission.csv
```

**What it does**:
1. Loads test.parquet (1,000 samples with 'ID' and 'code' columns)
2. Extracts 36 engineered features on-the-fly (basic stats, complexity, lexical)
3. Loads Stage 1 scaler and transforms features
4. Runs Stage 1 (XGBoost) to predict Human vs AI
5. Runs Stage 2 (CodeT5) on AI samples to predict family (1-10)
6. Combines predictions and saves to CSV

**Important**: Both Stage 1 and Stage 2 models must be trained before generating submissions.

## Architecture Overview

### Two-Stage Cascaded Design

```
Input: 36 engineered features + raw code string
    ↓
Stage 1: XGBoost (Binary: Human=0 vs AI=1)
    ├─ If pred=0 → Output: Label 0 (Human)
    └─ If pred=1 → Pass to Stage 2
              ↓
Stage 2: CodeT5-220m (Multi-class: AI families 1-10)
    ↓
Final Output: Labels 0-10
```

**Key Insight**: Stage 1 filters ~70% samples as Human, Stage 2 processes remaining ~30% to identify AI family.

### Stage 1: XGBoost Binary Classifier

**Implementation**: `src/models/stage1_classifier.py` (wrapper around XGBoost/SVM)

**Best Model**: XGBoost with:
- `n_estimators=300`, `max_depth=10`, `learning_rate=0.3`
- `scale_pos_weight=7.5` (handles 88% Human vs 12% AI imbalance)
- `device='cuda'` for GPU training (~2.5 min vs 10+ min on CPU)

**Performance**: 96.6% accuracy, F1=0.9259

**Critical Behavior**:
- `prepare_labels(y)`: Converts labels 0-10 → binary 0 (Human) or 1 (AI)
- Fits StandardScaler on training data → saved to `models/stage1/scaler.pkl`
- This scaler is **reused by Stage 2** (never create a new one!)

### Stage 2: CodeT5-220m Transformer

**Implementation**: `src/models/stage2_codet5.py` (full) and `stage2_codet5_simple.py` (minimal)

**Architecture**:
```
Code String → Tokenizer (max_length=512)
    ↓
CodeT5p-220m Encoder (FROZEN, 109.6M params)
    ↓
Mean Pooling → 768-dim embeddings
    ↓
Classification Head (TRAINABLE, 200K params):
    Linear(768, 256) → LayerNorm → ReLU → Dropout(0.3) → Linear(256, 10)
    ↓
Softmax → 10-way probabilities (AI families 1-10)
```

**Critical Behaviors**:
- `fit(X, y, code_strings)`: Internally filters `y > 0` to train only on AI samples
- Requires `code_strings` parameter (raw code) in addition to feature matrix `X`
- Uses `class_weight='balanced'` for CrossEntropyLoss (handles AI family imbalance)
- Early stopping with validation split (default 10%, patience=3)
- Saves only classification head (~1MB), encoder reloaded from HuggingFace

**Performance**: 45.6% accuracy on AI family detection, F1=0.4114

**Why CodeT5-220m not 770m?**
- CodeT5p-770m hangs indefinitely during model loading (confirmed in testing)
- CodeT5p-220m loads in ~70 seconds, same 768-dim embeddings, works perfectly

### Feature Pipeline

**Feature Loading** (`src/features/feature_loader.py`):
- Merges 3 precomputed CSVs: `basic_stats` (9 features), `complexity` (19 features), `lexical` (8 features)
- LEFT merge on `['code', 'generator', 'label', 'language']`
- Fills missing values with 0 (490k/500k samples lack complexity/lexical features)
- Returns DataFrame with 36 features total

**Feature Processing** (`src/features/feature_processor.py`):
- `fit_transform(df)`: Fits StandardScaler on training, returns scaled X and y
- `transform(df)`: Applies fitted scaler (for validation/test) - **never refits**
- Handles NaN/Inf with `np.nan_to_num()`
- Saves scaler as dict: `{'scaler': StandardScaler, 'feature_names': [...], ...}`

**Critical**: Stage 1 fits scaler, Stage 2 loads it. Same instance for both stages ensures feature space consistency.

### Pipeline Orchestration

**Not Yet Fully Implemented**: The `TwoStagePipeline` in `src/models/pipeline.py` is designed for Random Forest Stage 2. For CodeT5, use evaluation scripts directly.

**Manual Pipeline Flow** (see `src/evaluation/evaluate_pipeline_codet5.py`):
1. Load features and code strings
2. Load Stage 1 model, predict binary labels
3. Filter samples where Stage 1 pred=1 (AI)
4. Load Stage 2 CodeT5 model
5. Predict AI families with `stage2.predict(X_ai, code_strings=code_ai)`
6. Combine: Final preds = Stage 1 preds; overwrite AI indices with Stage 2 preds

## Critical Implementation Details

### Model Selection Priority

**Stage 1**: Pipeline auto-selects best model:
1. Check `models/stage1/XGBoost/classifier.pkl` (preferred, 96.6% acc)
2. Fallback to `models/stage1/classifier.pkl` (SVM, 75.1% acc)

**Stage 2**: Use CodeT5 for production:
- `models/stage2_codet5/classifier.pth` (45.6% acc, transformer-based)
- Older Random Forest variant exists but is inferior

### Training Order (Critical!)

```bash
# 1. Train Stage 1 (creates scaler)
uv run python src/training/train_stage1.py

# 2. Train Stage 2 (loads Stage 1's scaler)
uv run python src/training/train_stage2_codet5.py
```

**Why this order?**
- Stage 1 fits StandardScaler on training data → saves to `models/stage1/scaler.pkl`
- Stage 2 **must** load this exact scaler to ensure feature space alignment
- If scaler is missing, Stage 2 training will fail

### GPU Optimization

**Batch Size Impact** (CodeT5 training):
```
batch_size=32  → 73% GPU util,  2.7GB VRAM, 1629 batches/epoch, ~6 min/epoch
batch_size=256 → 100% GPU util, 13.5GB VRAM,  204 batches/epoch, ~2 min/epoch
```

**Recommendation**: Use `--batch-size 256` for GPUs with 32GB+ VRAM (8x faster training!)

**For Smaller GPUs**:
- 16GB VRAM: `--batch-size 128`
- 8GB VRAM: `--batch-size 64`
- 4GB VRAM: Use CPU or reduce max_length

### Class Imbalance Handling

**Dataset Distribution**:
- Human (label 0): 88.42% (442k samples)
- AI (labels 1-10): 11.58% (58k samples across 10 families)

**Stage 1**: `scale_pos_weight=7.5` in XGBoost
**Stage 2**: `class_weight='balanced'` in CrossEntropyLoss

**Primary Metric**: Macro F1 (treats all classes equally, not biased by majority class)

### Code Strings Requirement

**CodeT5 models require raw code in addition to features**:
```python
# ✅ Correct
clf.fit(X, y, code_strings=code_list)
clf.predict(X, code_strings=code_list)

# ❌ Wrong - will raise ValueError
clf.fit(X, y)
```

**Why?** CodeT5 tokenizes raw code to 768-dim embeddings; engineered features (X) are only for sklearn compatibility.

### Model Persistence

**Stage 1**:
- `models/stage1/XGBoost/classifier.pkl` (5.9 MB)
- `models/stage1/scaler.pkl` (StandardScaler object or dict)

**Stage 2 CodeT5**:
- `models/stage2_codet5/classifier.pth` (0.77 MB - only classification head)
- `models/stage2_codet5/config.json` (hyperparameters)
- Encoder reloaded from HuggingFace on load (not saved to save space)

**Loading with PyTorch 2.6+**: Use `weights_only=False` in `torch.load()` (already implemented in `stage2_codet5.py` and `stage2_codet5_simple.py`)

## Common Pitfalls

### Don't Do This:
- ❌ Train Stage 2 before Stage 1 → fails (no scaler)
- ❌ Create new scaler for Stage 2 → breaks feature consistency
- ❌ Use `python script.py` → use `uv run python script.py`
- ❌ Forget `code_strings` parameter with CodeT5 → ValueError
- ❌ Refit scaler on validation/test → data leakage
- ❌ Use CodeT5p-770m → hangs during loading (use 220m)

### Do This:
- ✅ Always train Stage 1 first
- ✅ Use `uv run python` for all commands
- ✅ Monitor macro F1 (primary metric)
- ✅ Use batch_size=256 for fast GPU training
- ✅ Load precomputed features from `data/reports/`
- ✅ Use CodeT5p-220m (loads reliably in ~70 seconds)

## Data Files

**Precomputed Features** (required for training):
- `data/reports/train_basic_stats.csv` (500k rows, 9 features)
- `data/reports/train_complexity_features.csv` (10k rows, 19 features)
- `data/reports/train_lexical_features.csv` (10k rows, 8 features)
- Corresponding `validation_*.csv` files

**Raw Data** (required for CodeT5):
- `data/train.parquet` (500k samples with code strings)
- `data/validation.parquet` (100k samples)
- `data/test.parquet` (1k samples, NO labels)

## Performance Metrics

**Overall Pipeline** (XGBoost + CodeT5 on 100K validation):
- Accuracy: 90.35%
- Macro F1: 0.4180
- Weighted F1: 0.9105

**Stage 1 (XGBoost Binary)**:
- Accuracy: 96.60%
- Macro F1: 0.9259

**Stage 2 (CodeT5 Multi-class on AI samples)**:
- Accuracy: 45.59%
- Macro F1: 0.4114

**Best Performing Classes**:
- Human: F1=0.98 (excellent)
- AI-10: F1=0.65 (best AI family)
- AI-8: F1=0.58
- AI-6: F1=0.50

## Troubleshooting

**"Module not found" errors**: Always use `uv run python` not just `python`

**"Stage 1 scaler not found"**: Train Stage 1 first with `uv run python src/training/train_stage1.py`

**"code_strings is required for CodeT5 classifier"**: Pass `code_strings=code_list` to `fit()` and `predict()`

**"ChunkedEncodingError" in logs**: Harmless transformers library warning during model download, can be ignored

**"CUDA out of memory"**: Reduce batch size: `--batch-size 64` or `--batch-size 32`

**CodeT5 model hangs at loading**: You're using CodeT5p-770m; switch to CodeT5p-220m (default in updated code)

**Low Stage 2 accuracy**: Expected - 10-way classification on imbalanced AI families is hard. Focus on overall pipeline metrics.

**Slow training**:
- Stage 1 XGBoost: ~2.5 min with GPU, ~10 min with CPU
- Stage 2 CodeT5: ~16 min (batch_size=256), ~40 min (batch_size=32)
- Use larger batch sizes on capable GPUs

## Output Artifacts

**After Training**:
- `models/stage1/XGBoost/classifier.pkl` + `metadata.json`
- `models/stage1/scaler.pkl`
- `models/stage2_codet5/classifier.pth` + `config.json`
- `outputs/metrics/*.json` (detailed metrics)

**After Evaluation**:
- `outputs/metrics/pipeline_codet5_validation.json` (full pipeline results)
- Confusion matrices (11x11: Human + 10 AI families)
- Per-class precision/recall/F1

**After Submission Generation**:
- `outputs/submission.csv` (1,000 test predictions, ready for upload)
- Format: CSV with 'ID' and 'label' columns (labels 0-10)

## Key Files to Understand

**Models**:
- `src/models/stage1_classifier.py` - Binary classifier wrapper
- `src/models/stage2_codet5.py` - Full CodeT5 implementation (use this)
- `src/models/stage2_codet5_simple.py` - Minimal CodeT5 (for reference)

**Training**:
- `src/training/train_stage1.py` - Trains XGBoost/SVM Stage 1
- `src/training/train_stage2_codet5.py` - Trains CodeT5 Stage 2 (NEW)

**Evaluation**:
- `src/evaluation/evaluate_pipeline_codet5.py` - Full pipeline eval (XGBoost + CodeT5)

**Submission**:
- `scripts/generate_submission_codet5.py` - Generate test predictions for competition

**Features**:
- `src/features/feature_loader.py` - Merges 3 CSV feature files
- `src/features/feature_processor.py` - StandardScaler fitting/loading

**Tests**:
- `tests/test_codet5_verbose.py` - Comprehensive CodeT5 validation
