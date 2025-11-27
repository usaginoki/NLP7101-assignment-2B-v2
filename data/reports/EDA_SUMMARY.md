# Exploratory Data Analysis Summary
## SemEval 2026 Task 13 Subtask B - AI-Generated Code Detection

**Date**: November 25, 2025
**Analysis Completed**: All 4 phases of EDA

---

## Dataset Overview

### Dataset Statistics
- **Training Set**: 500,000 samples
- **Validation Set**: 100,000 samples
- **Test Set**: 1,000 samples (no labels)

### Columns
- `code`: Source code samples
- `generator`: Name of the AI model/generator
- `label`: Numeric label (0-10) representing 11 classes
- `language`: Programming language

### Programming Languages (8 total)
| Language | Count (Train) | Percentage |
|----------|---------------|------------|
| Java | 137,076 | 27.42% |
| Python | 136,709 | 27.34% |
| C# | 62,781 | 12.56% |
| JavaScript | 41,780 | 8.36% |
| C++ | 36,581 | 7.32% |
| Go | 29,685 | 5.94% |
| PHP | 29,198 | 5.84% |
| C | 26,190 | 5.24% |

### Class Distribution
**Major Class Imbalance Observed!**

| Label | Count (Train) | Percentage | Interpretation |
|-------|---------------|------------|----------------|
| 0 | 442,096 | 88.42% | **Human-written code** |
| 10 | 10,810 | 2.16% | AI Model Family 10 |
| 2 | 8,993 | 1.80% | AI Model Family 2 |
| 7 | 8,197 | 1.64% | AI Model Family 7 |
| 8 | 8,127 | 1.63% | AI Model Family 8 |
| 6 | 5,783 | 1.16% | AI Model Family 6 |
| 9 | 4,608 | 0.92% | AI Model Family 9 |
| 1 | 4,162 | 0.83% | AI Model Family 1 |
| 3 | 3,029 | 0.61% | AI Model Family 3 |
| 4 | 2,227 | 0.45% | AI Model Family 4 |
| 5 | 1,968 | 0.39% | AI Model Family 5 |

**Key Observation**: Class 0 (Human) dominates at 88.42%, while AI-generated classes range from 0.39% to 2.16%. This severe imbalance will require special handling (oversampling, class weights, etc.).

---

## Phase 1: Basic Statistical Analysis

### Code Length Statistics (Training Set)

| Metric | Mean | Median | Std Dev |
|--------|------|--------|---------|
| Character Count | 1,410 | 812 | 1,662 |
| Line Count | 42 | 25 | 47 |
| Word Count | 128 | 75 | 151 |
| Avg Line Length | 34.16 | 32.75 | 12.26 |

### Key Findings by Label

#### Character Count
- **Label 0 (Human)**: 1,443 chars (baseline)
- **Label 5**: **2,694 chars** ⚠️ Significantly longer (+87%)
- **Label 4**: 842 chars (shortest among AI)

#### Average Line Length
- **Label 0 (Human)**: 34.77 chars/line
- **Label 4**: **22.65 chars/line** ⚠️ Much shorter (-35%)
- **Label 8**: 33.69 chars/line (closest to human)

#### Special Character Ratio
- **Label 0 (Human)**: 0.1373
- **Label 4**: **0.2004** ⚠️ Highest (+46%)
- **Label 8**: 0.1050 (lowest)

#### Alphanumeric Ratio
- **Label 0 (Human)**: 0.5970
- **Label 8**: **0.6501** ⚠️ Highest (+8.9%)
- **Label 4**: **0.5050** ⚠️ Lowest (-15.4%)

**Discriminative Features Identified**:
- Label 5 generates much longer code
- Label 4 has shorter lines and more special characters
- Labels 8 & 9 have cleaner code (higher alphanum ratio)

---

## Phase 2: Complexity Feature Analysis

Analyzed 10,000 samples from training set (for computational efficiency)

### Cyclomatic Complexity (Control Flow Complexity)

| Label | cc_mean | cc_max | Interpretation |
|-------|---------|--------|----------------|
| 0 (Human) | 1.049 | 1.119 | Moderate complexity |
| 6 | **1.976** | **2.107** | Most complex control flow |
| 3 | 1.580 | 1.660 | High complexity |
| 1 | 1.054 | 1.227 | Similar to human |
| 8 | 0.902 | 1.011 | Simple control flow |

**Key Finding**: Label 6 has nearly **2x the cyclomatic complexity** of human code, suggesting more branching and conditional logic.

### Lines of Code (LOC)

| Label | LOC Mean | SLOC Mean | Comments | Blank |
|-------|----------|-----------|----------|-------|
| 0 (Human) | 20.45 | 17.78 | 0.38 | 1.93 |
| 5 | **43.58** | 30.15 | **2.30** | 9.21 |
| 4 | **40.64** | 23.90 | **11.21** | 5.54 |
| 6 | 26.12 | 18.97 | 2.69 | 4.49 |
| 8 | 19.77 | 14.03 | 1.06 | 3.22 |

**Key Findings**:
- **Label 5**: Longest code (2.1x human LOC), many comments
- **Label 4**: Second longest, **highest comment count** (11.21 vs 0.38)
- **Labels 8, 9, 10**: Shorter code than human baseline

### Halstead Metrics (Program Complexity)

| Label | Volume | Difficulty | Effort | Bugs (est.) |
|-------|--------|------------|--------|-------------|
| 0 (Human) | 24.83 | 0.577 | 128.67 | 0.008 |
| 4 | **120.34** | **2.718** | **868.38** | **0.040** |
| 6 | 64.61 | 1.609 | 312.30 | 0.022 |
| 5 | 18.31 | 0.376 | 41.57 | 0.006 |
| 8 | 4.16 | 0.184 | 8.62 | 0.001 |

**Key Findings**:
- **Label 4**: **3.4x higher effort** than human code - most complex to write/maintain
- **Label 6**: Also shows high complexity (2.4x human effort)
- **Labels 8, 9, 10**: Very simple programs (low effort/volume)

### Maintainability Index

Higher is better (0-100 scale, >20 is acceptable)

| Label | Maintainability Index |
|-------|----------------------|
| 6 | **43.17** (Best) |
| 4 | 40.61 |
| 3 | 30.10 |
| 1 | 26.74 |
| 2 | 27.22 |
| 0 (Human) | 17.51 |
| 9 | **14.08** (Worst) |

**Surprising Finding**: Some AI models (6, 4) produce more maintainable code than humans!

---

## Phase 3: Lexical Diversity Analysis

Analyzed lexical patterns and vocabulary usage across 10,000 samples.

### Type-Token Ratio (TTR)
Higher TTR = more diverse vocabulary

| Label | TTR | Unique Tokens | Total Tokens |
|-------|-----|---------------|--------------|
| 8 | **0.4646** | 58.59 | 127.35 |
| 1 | **0.4648** | 61.95 | 144.75 |
| 0 (Human) | 0.4504 | 54.80 | 142.21 |
| 5 | **0.3568** | 91.45 | **243.55** |
| 4 | **0.3600** | 33.10 | 128.82 |

**Key Findings**:
- **Labels 1 & 8**: Most diverse vocabulary (higher TTR)
- **Labels 4 & 5**: Less diverse despite different code lengths
  - Label 5: Long code, many tokens, but repetitive
  - Label 4: Shorter unique vocabulary

### MTLD (Measure of Textual Lexical Diversity)
Length-independent diversity measure

| Label | MTLD | Interpretation |
|-------|------|----------------|
| 1 | **25.13** | Most diverse |
| 8 | **24.20** | Very diverse |
| 5 | 20.93 | Moderate |
| 0 (Human) | 20.12 | Baseline |
| 4 | **12.99** | Least diverse |

**Key Finding**: Labels 1 & 8 show **significantly higher lexical diversity** than human code.

### MATTR (Moving Average TTR)
Stable diversity measure across window sizes

| Label | MATTR |
|-------|-------|
| 8 | **0.5837** |
| 1 | 0.5867 |
| 5 | 0.5533 |
| 0 (Human) | 0.4994 |
| 4 | **0.3030** |
| 7 | 0.3870 |

**Key Finding**: Label 4 shows consistently **low diversity** across all metrics.

### Keyword Usage

| Label | Keyword Count | Keyword Ratio |
|-------|---------------|---------------|
| 5 | **31.91** | 0.1315 |
| 0 (Human) | 17.58 | 0.1377 |
| 7 | 20.59 | **0.1606** |
| 2 | 16.86 | **0.1634** |
| 10 | 17.87 | **0.1629** |
| 1 | 16.59 | 0.1334 |

**Key Findings**:
- **Label 5**: Uses most keywords (1.8x human) due to longer code
- **Labels 2, 7, 10**: **Higher keyword density** (ratio) - more language constructs per token

---

## Summary of Discriminative Features

### Label Characterization

#### Label 0 (Human - 88.42%)
- **Baseline** for comparison
- Moderate complexity and diversity
- avg_line_length: 34.77
- cc_mean: 1.049
- TTR: 0.4504

#### Label 1 (0.83%)
- **High lexical diversity** (MTLD: 25.13, highest)
- Similar complexity to human
- Slightly longer code
- **Distinguishing**: Highest MATTR (0.5867)

#### Label 2 (1.80%)
- Shorter code (986 chars)
- **High keyword ratio** (0.1634)
- Moderate complexity
- **Distinguishing**: High keyword density

#### Label 3 (0.61%)
- Short code (898 chars)
- Higher cyclomatic complexity (cc_mean: 1.58)
- Moderate diversity
- **Distinguishing**: Complex logic in short code

#### Label 4 (0.45%)
- **Shortest avg line length** (22.65)
- **Highest special char ratio** (0.2004)
- **Lowest lexical diversity** (TTR: 0.36, MTLD: 12.99)
- **Highest Halstead metrics** (effort: 868.38)
- **Most comments** (11.21)
- **Distinguishing**: Very repetitive, complex, heavily commented code

#### Label 5 (0.39%)
- **Longest code** (2,694 chars, 86 lines)
- **Most tokens** (243.55)
- **Lowest TTR** (0.3568) - repetitive
- Many comments (2.30)
- **Most keywords** (31.91)
- **Distinguishing**: Verbose, repetitive code

#### Label 6 (1.16%)
- **Highest cyclomatic complexity** (cc_mean: 1.976)
- **Highest maintainability index** (43.17)
- High Halstead difficulty (1.609)
- **Distinguishing**: Complex control flow, well-structured

#### Label 7 (1.64%)
- Moderate code length (1,018 chars)
- **High keyword ratio** (0.1606)
- Low diversity (MATTR: 0.387)
- **Distinguishing**: Keyword-heavy, structured code

#### Label 8 (1.63%)
- **Highest alphanum ratio** (0.6501)
- **Highest TTR** (0.4646) & MTLD (24.20)
- **Lowest special char ratio** (0.1050)
- Very simple (Halstead volume: 4.16)
- **Distinguishing**: Clean, diverse, simple code

#### Label 9 (0.92%)
- Similar to Label 8 but slightly less diverse
- High alphanum ratio (0.6444)
- Low complexity metrics
- **Distinguishing**: Clean, simple code

#### Label 10 (2.16%)
- **High keyword ratio** (0.1629)
- Moderate complexity and diversity
- Balanced metrics
- **Distinguishing**: Keyword-dense, balanced code

---

## Critical Insights for Model Development

### 1. Class Imbalance Challenge
- **88.42% human vs. <3% per AI class** requires:
  - Class-weighted loss functions
  - Oversampling minority classes (SMOTE, ADASYN)
  - Stratified sampling
  - Consider focal loss or balanced cross-entropy

### 2. Most Discriminative Features

#### High Importance (strong differentiators):
1. **avg_line_length**: Label 4 is 35% shorter
2. **special_char_ratio**: Label 4 is 46% higher
3. **char_count/LOC**: Label 5 is 87% longer
4. **cc_mean**: Label 6 is 88% higher
5. **halstead_effort**: Label 4 is 574% higher
6. **TTR/MTLD**: Labels 1 & 8 are 15-25% higher
7. **MATTR**: Label 4 is 39% lower
8. **keyword_ratio**: Labels 2, 7, 10 are 15-19% higher

#### Moderate Importance:
- alphanum_ratio
- comment counts
- halstead_volume
- maintainability_index

#### Lower Importance (high overlap):
- word_count (too correlated with char_count)
- token_count (similar issue)
- unique_tokens (redundant with TTR)

### 3. Feature Engineering Recommendations

**Create composite features**:
1. `code_density = sloc / loc` (code vs whitespace ratio)
2. `comment_ratio = comments / sloc` (documentation level)
3. `complexity_per_loc = cc_total / loc` (complexity density)
4. `diversity_index = (TTR + MTLD/50 + MATTR) / 3` (combined diversity)
5. `structural_complexity = cc_mean * halstead_difficulty` (combined complexity)

**Language-specific features**:
- Normalize metrics by programming language
- Language interaction terms (e.g., `Python * cc_mean`)

### 4. Multi-Language Considerations
- 8 different languages with different syntax characteristics
- Consider:
  - Separate models per language
  - Language embeddings
  - Language as categorical feature
  - Language-normalized features

### 5. Model Architecture Recommendations

**Baseline Approaches**:
1. **Random Forest** with class weights
   - Use top 20 features
   - Handle class imbalance with class_weight='balanced'

2. **XGBoost** with scale_pos_weight
   - Better for imbalanced data
   - Feature importance analysis

3. **LightGBM** with is_unbalance=True
   - Fast training on 500k samples
   - Categorical feature support for language

**Advanced Approaches**:
1. **Hybrid Model**: Static features + Code Embeddings
   - Extract CodeBERT/GraphCodeBERT embeddings
   - Concatenate with engineered features
   - SOTA: 82.55% F1 score

2. **Ensemble**:
   - Language-specific models
   - Combine predictions with meta-learner

3. **Two-Stage Classification**:
   - Stage 1: Human vs AI (binary)
   - Stage 2: AI model attribution (11-class)

---

## Files Generated

### Visualizations (47 files)
- **Distributions**: 16 plots (class distribution, char count, line count, etc.)
- **Complexity**: 12 plots (LOC, CC, Halstead, correlations)
- **Lexical**: 14 plots (TTR, MTLD, MATTR, token stats)
- **Correlations**: 4 heatmaps

### Reports (10 CSV files)
- `train_basic_stats.csv`: 500k samples with 9 basic features
- `train_complexity_features.csv`: 10k samples with 19 complexity features
- `train_lexical_features.csv`: 10k samples with 8 lexical features
- Corresponding validation files (5 files)
- Summary statistics (2 files)

---

## Next Steps

1. **Feature Selection**
   - Correlation analysis to remove redundant features
   - Recursive feature elimination
   - SHAP value analysis

2. **Handle Class Imbalance**
   - Implement SMOTE or ADASYN
   - Use class weights
   - Try focal loss

3. **Baseline Models**
   - Train Random Forest, XGBoost, LightGBM
   - Establish baseline performance
   - Feature importance analysis

4. **Advanced Features**
   - Extract code embeddings (CodeBERT)
   - AST-based features
   - N-gram patterns

5. **Ensemble Strategy**
   - Combine multiple models
   - Language-specific models
   - Two-stage classification

6. **Evaluation**
   - Multi-class F1 score (macro and weighted)
   - Per-class precision/recall
   - Confusion matrix analysis
   - Cross-validation with stratification

---

## Conclusion

The EDA reveals **clear discriminative patterns** across the 11 classes:
- **Label 4**: Repetitive, complex, heavily commented
- **Label 5**: Verbose, longest code
- **Label 6**: High control flow complexity
- **Labels 1 & 8**: High lexical diversity
- **Labels 2, 7, 10**: Keyword-dense

The severe **class imbalance** (88% human) is the primary challenge and must be addressed in model training.

With the **36+ engineered features** extracted, we have a strong foundation for building discriminative models to achieve competitive performance on this challenging multi-class attribution task.

**Target Performance**: 82.55% F1 score (current SOTA with embeddings)
