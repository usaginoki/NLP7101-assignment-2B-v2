# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This is a 2-stage cascaded classification pipeline for AI-generated code detection (SemEval 2026 Task 13 Subtask B). The system detects whether code is human-written or AI-generated, then identifies which of 10 AI model families generated it.

## Essential Commands

### Setup
```bash
# Install dependencies
uv sync

# All Python commands must be run with uv
uv run python <script>
```

### Training Pipeline
```bash
# Train complete 2-stage pipeline (recommended)
uv run python scripts/train_pipeline.py

# Train stages individually
uv run python src/training/train_stage1.py
uv run python src/training/train_stage2.py  # Requires Stage 1 trained first

# Stage 2 MUST be trained after Stage 1 because it reuses the scaler from Stage 1
```

### Evaluation
```bash
# Evaluate full pipeline
uv run python src/evaluation/evaluate_pipeline.py --split validation

# Evaluate individual stages
uv run python src/evaluation/evaluate_stage1.py --split validation
uv run python src/evaluation/evaluate_stage2.py --split validation
```

### Submission Generation
```bash
# Generate competition submission file
uv run python scripts/generate_submission.py

# Output: outputs/submission.csv
```

## Architecture Overview

### Two-Stage Cascaded Design

The pipeline uses a hierarchical classification approach:

1. **Stage 1 (Binary)**: Human (label 0) vs AI-generated (labels 1-10)
   - Random Forest with `class_weight='balanced'` to handle 88% vs 12% imbalance
   - If predicted as Human (0), final output is 0
   - If predicted as AI (1), sample passes to Stage 2

2. **Stage 2 (Multi-class)**: Identify AI model family (labels 1-10)
   - Random Forest with `class_weight='balanced'` for 10-way classification
   - Only processes samples predicted as AI by Stage 1
   - Outputs final label 1-10

**Key Architectural Decision**: Stage 2 is trained on ALL training data (500k samples) but only fits on AI samples (y > 0). This is critical - it sees all data to establish the same feature space as Stage 1, but `fit()` internally filters to AI samples only.

### Feature Pipeline

**Feature Loading** (`src/features/feature_loader.py`):
- Merges 3 precomputed CSV files: basic_stats, complexity, lexical
- Uses LEFT merge on `['code', 'generator', 'label', 'language']`
- Fills missing values with 0 (represents 490k samples without complexity/lexical features)
- Returns unified DataFrame with 36 features

**Feature Processing** (`src/features/feature_processor.py`):
- `fit_transform()`: Fits StandardScaler on training data, transforms features
- `transform()`: Uses fitted scaler (for validation/test) - never refits
- Saves/loads scaler using joblib for consistency across stages

**Critical**: The same scaler instance is used for both stages. Stage 1 fits it, Stage 2 loads it.

### Model Components

**Stage 1 Classifier** (`src/models/stage1_classifier.py`):
- `prepare_labels(y)`: Static method converts 0-10 labels to binary (0=Human, 1=AI)
- `fit()`: Trains on binary transformed labels
- `predict()`: Returns 0 or 1

**Stage 2 Classifier** (`src/models/stage2_classifier.py`):
- Factory pattern: `create_stage2_classifier(type='random_forest')`
- `Stage2RandomForest.fit(X, y)`: Internally filters for `y > 0` before training
- `Stage2CodeT5`: Placeholder for future transformer-based implementation
- Extensible via abstract base class `Stage2ClassifierBase`

**Pipeline** (`src/models/pipeline.py`):
- `predict(X)`: Orchestrates 2-stage flow
  - Runs Stage 1 on all samples
  - Filters AI samples (stage1_preds == 1)
  - Runs Stage 2 only on filtered samples
  - Combines results into final predictions (0-10)
- `evaluate(X, y_true)`: Returns metrics for Stage 1, Stage 2, and overall
- `load(models_dir)`: Class method to load complete pipeline from disk

### Data Dependencies

**Required Precomputed Features** (from EDA phase):
- `data/reports/train_basic_stats.csv` (500k rows, 9 features)
- `data/reports/train_complexity_features.csv` (10k rows, 19 features)
- `data/reports/train_lexical_features.csv` (10k rows, 8 features)
- Corresponding `validation_*.csv` files

**Raw Data**:
- `data/train.parquet` (500k samples)
- `data/validation.parquet` (100k samples)
- `data/test.parquet` (1k samples, NO labels)

**Feature Extraction Dependencies**:
- `data/modules/feature_extractors.py` contains `ComplexityFeatureExtractor` and `LexicalFeatureExtractor`
- Used by `generate_submission.py` to extract features from test set at inference time

## Critical Implementation Details

### Class Imbalance Handling
- Dataset is 88.42% Human (label 0), 11.58% AI (labels 1-10)
- Both stages use `class_weight='balanced'` in RandomForest
- Macro F1 is the primary metric (treats all classes equally)

### Feature Scaling
- StandardScaler fitted ONLY on training data
- Same scaler used for validation and test
- Scaler saved with Stage 1 models and loaded by Stage 2
- **Never refit the scaler** on validation/test data

### Model Persistence
- Models saved using joblib (`.pkl` files)
- Stage 1: `models/stage1/classifier.pkl` + `models/stage1/scaler.pkl`
- Stage 2: `models/stage2/classifier.pkl`
- Pipeline loading: `TwoStagePipeline.load('models/')` loads all components

### Training Order
1. Stage 1 must be trained first (creates scaler)
2. Stage 2 loads Stage 1's scaler (does NOT create new scaler)
3. If Stage 1 scaler is missing, Stage 2 training will fail

## Common Pitfalls

### Don't Do This:
- ❌ Train Stage 2 before Stage 1 (will fail - no scaler)
- ❌ Create new scaler for Stage 2 (breaks feature consistency)
- ❌ Train Stage 2 only on AI samples from original data (loses feature alignment)
- ❌ Refit StandardScaler on validation/test data (data leakage)
- ❌ Use accuracy as primary metric (misleading with class imbalance)

### Do This:
- ✅ Train Stage 1 first, then Stage 2
- ✅ Use `uv run python` for all commands
- ✅ Monitor macro F1 score (primary metric)
- ✅ Let Stage 2's `fit()` method handle AI filtering internally
- ✅ Load precomputed features before training (saves hours of computation)

## Output Artifacts

**After Training**:
- `models/stage1/classifier.pkl` - Stage 1 Random Forest
- `models/stage1/scaler.pkl` - Feature StandardScaler
- `models/stage2/classifier.pkl` - Stage 2 Random Forest
- `outputs/metrics/*.json` - Evaluation metrics (F1, accuracy, confusion matrices)

**After Submission Generation**:
- `outputs/submission.csv` - Competition submission (1000 rows, single 'label' column with values 0-10)

## Extension Points

### Adding CodeT5 Embeddings (Future)
1. Install: `uv add torch>=2.0.0 transformers>=4.40.0`
2. Implement `Stage2CodeT5` class in `src/models/stage2_classifier.py`
3. Use factory: `create_stage2_classifier('codet5')`
4. Architecture: Frozen Salesforce/codet5-base encoder → Linear(768,256) → ReLU → Dropout → Linear(256,10)

### Adding New Features
1. Add feature extraction in `data/modules/feature_extractors.py`
2. Regenerate feature CSVs (or extract on-the-fly)
3. Update `FeatureLoader` merge logic if new CSV files
4. Retrain both stages from scratch (features change)

## Dataset Characteristics

- **500k training samples**: Label 0 (Human) = 442k (88%), Labels 1-10 (AI) = 58k (12%)
- **8 programming languages**: Java (27%), Python (27%), C# (13%), JavaScript (8%), C++ (7%), Go (6%), PHP (6%), C (5%)
- **36 engineered features**: 9 basic stats + 19 complexity + 8 lexical diversity
- **Expected performance**: 70-80% macro F1 (with current RF approach), ~82% with CodeT5

## Troubleshooting

**"Module not found" errors**: Always use `uv run python` not just `python`

**"File not found: train_basic_stats.csv"**: Precomputed feature CSVs must exist in `data/reports/`. Run EDA scripts first if missing.

**Stage 2 fails to load scaler**: Train Stage 1 first. Stage 2 depends on `models/stage1/scaler.pkl`.

**Low validation scores**: Check that `class_weight='balanced'` is set in both RandomForest classifiers. Review feature importance to identify overfitting.

**Slow training**: Random Forest uses all CPU cores (`n_jobs=-1`). Stage 1 takes ~10-20 min (500k samples), Stage 2 takes ~5-10 min (58k AI samples).
