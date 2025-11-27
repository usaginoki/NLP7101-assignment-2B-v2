# Literature Review: AI-Generated Code Detection and Attribution
## SemEval 2026 Task 13 Subtask B

**Date**: November 25, 2025
**Task**: Multi-class classification to identify AI model families that generated code

---

## 1. Task Overview

### SemEval 2026 Task 13
- **Focus**: Detecting machine-generated code across multiple programming languages, generators, and application scenarios
- **Task Organizers**: Daniil Orel, Dilshod Azizov, Indraneil Paul, Yuxia Wang, Iryna Gurevych, and Preslav Nakov
- **Subtask B**: Multi-class classification to identify which AI model family (Gemini, Qwen, etc.) generated the code
- **Challenge**: Model attribution is significantly harder than binary detection (human vs AI)

**Source**: [SemEval-2026 Tasks](https://semeval.github.io/SemEval2026/tasks.html)

---

## 2. State-of-the-Art Approaches

### 2.1 Machine Learning with Code Embeddings (SOTA)
- **Performance**: 82.55% F1 score (mean average across models)
- **Approach**: Uses semantic code embeddings as features for classification
- **Advantages**: Captures semantic patterns beyond surface-level syntax
- **Method**: Code is converted to embeddings, then fed into classifiers (RF, GB, LR)

**Source**: [An Empirical Study on Automatically Detecting AI-Generated Source Code](https://arxiv.org/abs/2411.04299)

### 2.2 Machine Learning with Static Code Metrics
- **Features Used**:
  - Cyclomatic Complexity
  - Lines of Code (LOC, SLOC)
  - Halstead Metrics (volume, difficulty, effort)
  - Code structure metrics (nesting, function counts)
- **Performance**: Competitive but below embedding-based approaches
- **Advantages**:
  - Interpretable features
  - Language-agnostic when properly designed
  - Fast extraction

**Source**: [An Empirical Study on Automatically Detecting AI-Generated Source Code](https://arxiv.org/html/2411.04299v1)

### 2.3 Commercial Detection Tools (2025)
Several commercial tools have emerged:

**Span AI Code Detector**:
- Uses machine learning classifier trained on millions of samples
- Analyzes semantic "chunks" of code
- Looks at style, syntax, and structure patterns

**BlueOptima's Code Author Detector**:
- Leverages 16+ billion static source code metric observations
- Advanced algorithms and machine learning

**GPTSniffer**:
- State-of-the-art baseline for AI code detection
- Outperformed by recent embedding-based approaches

**Sources**:
- [AI Code Detector by Span](https://code-detector.ai/)
- [Top 10 AI Code Detector Tools](https://vertu.com/ai-tools/top-10-ai-code-detector-tools-for-developers-2025/)

---

## 3. Code Complexity Metrics

### 3.1 Cyclomatic Complexity
- **Definition**: Measures control flow complexity by counting decision points
- **Formula**: M = E - N + 2P (edges, nodes, connected components)
- **Application**: Higher complexity may indicate human-written code with more nuanced logic
- **Tool**: Radon (Python library)

**Sources**:
- [Cyclomatic Complexity - GeeksforGeeks](https://www.geeksforgeeks.org/dsa/cyclomatic-complexity/)
- [Code Complexity Explained 2025](https://www.qodo.ai/blog/code-complexity/)

### 3.2 Halstead Metrics
- **Vocabulary**: Number of unique operators and operands
- **Length**: Total operators and operands
- **Volume**: Vocabulary × log2(Vocabulary)
- **Difficulty**: Measures how difficult the code is to write
- **Effort**: Volume × Difficulty
- **Bugs**: Estimated number of bugs (Effort^(2/3) / 3000)

### 3.3 Lines of Code (LOC) Metrics
- **LOC**: Total lines
- **SLOC**: Source lines (excluding comments, blanks)
- **Comment Lines**: Documentation
- **Blank Lines**: Whitespace
- **Code-to-Comment Ratio**: Indicator of code documentation

**Tool**: Radon library provides comprehensive LOC analysis

---

## 4. Lexical Diversity Metrics

### 4.1 Type-Token Ratio (TTR)
- **Definition**: Ratio of unique words (types) to total words (tokens)
- **Range**: 0 to 1
- **Application**: AI models may have different lexical diversity patterns
- **Limitation**: Sensitive to text length

**Source**: [Type-Token Ratio in NLP](https://medium.com/@rajeswaridepala/empirical-laws-ttr-cc9f826d304d)

### 4.2 Moving Average Type-Token Ratio (MATTR)
- **Improvement over TTR**: Addresses length sensitivity
- **Method**: Calculates TTR over moving windows
- **Stability**: More stable across different text lengths

### 4.3 Measure of Textual Lexical Diversity (MTLD)
- **Method**: Measures how far one must proceed before TTR drops below threshold
- **Advantage**: Length-independent measure
- **Threshold**: Typically 0.72

### 4.4 Hypergeometric Distribution D (HD-D)
- **Statistical basis**: Uses hypergeometric distribution
- **Robustness**: Most robust to sample size variations

**Tools**:
- Python: `lexicalrichness` library
- R: `quanteda` package

**Sources**:
- [LexicalRichness Documentation](https://www.lucasshen.com/software/lexicalrichness/doc)
- [GitHub - LexicalRichness](https://github.com/LSYS/LexicalRichness)

---

## 5. Key Findings from Literature

### 5.1 AI-Generated Code Characteristics
1. **Consistency**: AI models tend to produce more consistent code patterns
2. **Style**: Different model families have distinct coding styles
3. **Complexity**: May differ in average cyclomatic complexity
4. **Naming**: Variable and function naming conventions vary by model
5. **Comments**: Comment density and style differ between models

### 5.2 Challenges in Model Attribution
1. **Fine-grained classification**: Distinguishing between model families is harder than binary detection
2. **Language diversity**: Patterns may vary across programming languages
3. **Model updates**: AI models evolve, changing their generation patterns
4. **Training data**: Different models trained on different code corpuses

### 5.3 Effective Feature Categories
1. **Static code metrics**: Reliable and interpretable
2. **Lexical features**: Capture vocabulary and style differences
3. **Structural patterns**: Control flow and code organization
4. **Token-level statistics**: Granular analysis of code tokens
5. **Embeddings**: Capture semantic meaning (SOTA)

**Source**: [An Empirical Study on Automatically Detecting AI-Generated Source Code](https://arxiv.org/html/2411.04299v1)

---

## 6. Relevant Tools and Libraries

### 6.1 Code Analysis
- **Radon**: Python library for complexity metrics
  - Cyclomatic Complexity
  - Halstead Metrics
  - Maintainability Index
  - LOC analysis

### 6.2 Lexical Analysis
- **lexicalrichness**: Python library for lexical diversity
  - TTR, RTTR, CTTR
  - MATTR, MTLD
  - HD-D

### 6.3 Visualization
- **Seaborn**: Statistical data visualization (box plots, heatmaps)
- **Plotly**: Interactive visualizations (scatter plots, 3D plots)
- **Matplotlib**: Foundational plotting library

---

## 7. Recommended Approach for This Task

### Phase 1: Feature Extraction
1. **Complexity metrics** (Cyclomatic, Halstead, LOC)
2. **Lexical diversity** (TTR, MATTR, MTLD)
3. **Token statistics** (vocabulary size, keyword usage)
4. **Structural features** (nesting depth, function counts)

### Phase 2: Exploratory Analysis
1. Distribution analysis per model family
2. Feature correlation analysis
3. Statistical significance testing (ANOVA)
4. Dimensionality reduction (PCA, t-SNE)

### Phase 3: Pattern Identification
1. Identify distinguishing characteristics per model family
2. Language-specific patterns
3. Feature importance analysis
4. Model family "fingerprints"

### Phase 4: Model Development
1. Baseline: Traditional ML with static features
2. Advanced: Hybrid approach (features + embeddings)
3. Ensemble methods for robustness

---

## 8. Expected Challenges

1. **Class imbalance**: Some model families may be underrepresented
2. **Language diversity**: Different languages may require different features
3. **Computational cost**: Feature extraction on large datasets is time-intensive
4. **Syntax errors**: Code samples may have syntax errors affecting metric extraction
5. **Generalization**: Models must generalize to unseen AI model versions

---

## 9. Success Metrics

- **Primary**: Multi-class F1 score
- **Secondary**: Per-class precision and recall
- **Interpretability**: Feature importance and model explanation
- **Efficiency**: Computational cost vs. accuracy trade-off

---

## 10. References

1. [SemEval-2026 Tasks](https://semeval.github.io/SemEval2026/tasks.html)
2. [An Empirical Study on Automatically Detecting AI-Generated Source Code](https://arxiv.org/abs/2411.04299)
3. [Code Complexity Explained (2025)](https://www.qodo.ai/blog/code-complexity/)
4. [Cyclomatic Complexity - Wikipedia](https://en.wikipedia.org/wiki/Cyclomatic_complexity)
5. [LexicalRichness Documentation](https://www.lucasshen.com/software/lexicalrichness/doc)
6. [Type-Token Ratio in NLP](https://medium.com/@rajeswaridepala/empirical-laws-ttr-cc9f826d304d)
7. [AI Code Detector by Span](https://code-detector.ai/)
8. [State of AI Code Quality in 2025](https://www.qodo.ai/reports/state-of-ai-code-quality/)

---

## Conclusion

The task of AI-generated code attribution requires a multi-faceted approach combining:
- **Static code metrics** for interpretable, language-agnostic features
- **Lexical diversity measures** to capture stylistic differences
- **Semantic embeddings** for state-of-the-art performance
- **Careful feature engineering** tailored to the specific model families in the dataset

The current state-of-the-art achieves ~82.55% F1 score using code embeddings, but there is significant room for improvement, especially in the multi-class attribution setting of this task.
