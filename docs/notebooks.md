# Notebooks Overview

The project includes 11 comprehensive Jupyter notebooks covering the complete machine learning pipeline from getting started through production-ready hybrid models. Each notebook has been enhanced with professional visualizations, comprehensive markdown analysis, and practical insights.

---

## Notebook Sequence

### 0. 00_getting_started_tutorial.ipynb - Interactive Getting Started Guide ⭐ **START HERE**

**Purpose**: Beginner-friendly interactive tutorial introducing AlertSage fundamentals for new users and contributors.

**Key Features**:

- **Environment Setup**: Verification of Python, packages, models, and dataset
- **Model Loading**: Load pre-trained TF-IDF vectorizer and baseline logistic regression
- **First Prediction**: Step-by-step walkthrough of single incident analysis
- **Batch Processing**: Analyze 30 diverse incidents across all 10 event types
- **4 Interactive Visualizations**:
  - Class distribution bar chart
  - Confidence score histogram with uncertainty threshold
  - Confidence by event type box plots
  - Confusion matrix heatmap
- **Uncertainty Analysis**: Understanding confidence thresholds (50%, 60%, 75%)
- **LLM Integration**: Conceptual overview of ML+LLM hybrid approach
- **3 Hands-On Exercises**: Custom incident analysis, threshold experimentation, problematic case identification
- **Next Steps Guide**: Links to advanced notebooks, CLI usage, Streamlit UI, and documentation

**Learning Outcomes**: Understand incident triage workflow, interpret confidence scores, create visualizations, recognize when LLM assistance is needed, practice with real scenarios.


### 1. 01_explore_dataset.ipynb - Dataset Exploration & Quality Assessment

**Purpose**: Comprehensive exploratory data analysis (EDA) of the synthetic cybersecurity incident dataset.

**Key Features**:

- **Event Type Distribution**: Color-coded bar charts showing class balance
- **Temporal Analysis**: Incident distribution across 2024 with seasonal patterns
- **Severity Assessment**: Stacked bar charts showing severity levels per event type
- **Log Source Analysis**: Detection system coverage visualizations
- **Geographic Distribution**: Source/destination country analysis
- **MITRE ATT&CK Mapping**: Technique distribution across incident classes

**Learning Outcomes**: Understand dataset balance, temporal patterns, severity biasing, and log source diversity.

---

### 2. 02_prepare_text_and_features.ipynb - Feature Engineering & Text Preprocessing

**Purpose**: Transform raw text descriptions into TF-IDF feature matrices.

**Key Features**:

- Text cleaning pipeline (lowercase, punctuation removal)
- TF-IDF vectorization (max_features=3000, min_df=2)
- Feature sparsity analysis and vocabulary composition charts
- Stratified 80/20 train/test split
- Artifact export (vectorizer, matrices) to `models/`

**Learning Outcomes**: Master TF-IDF sparse matrices, scikit-learn pipelines, and joblib serialization.

---

### 3. 03_baseline_model.ipynb - Logistic Regression Baseline

**Purpose**: Train and evaluate baseline Logistic Regression classifier.

**Key Features**:

- LogisticRegression with multinomial objective
- Dual confusion matrices (raw counts + normalized percentages)
- Per-class performance bar charts
- Model persistence for CLI/UI

**Performance**: 92-95% overall accuracy, strong on malware/data_exfiltration, weaker on web_attack/access_abuse confusion.

---

### 4. 04_model_interpretability.ipynb - Feature Importance Analysis

**Purpose**: Explain model decisions through coefficient analysis.

**Key Features**:

- Top predictive terms per class (coefficient heatmaps)
- Weight distribution histograms
- Domain knowledge validation (sensible vs spurious correlations)

**Learning Outcomes**: Understand linear model coefficients as feature importance, validate model reasoning.

---

### 5. 05_inference_and_cli.ipynb - Prediction Workflow & CLI Testing

**Purpose**: Demonstrate inference and validate CLI consistency.

**Key Features**:

- Notebook inference with probability outputs
- CLI testing (`python -m triage.cli`)
- Interactive session with Ctrl+C handling
- Prediction comparison (notebook vs CLI)

**Learning Outcomes**: Master joblib loading, predict_proba() output, command-line packaging.

---

### 6. 06_model_visualization_and_insights.ipynb - Performance Analysis

**Purpose**: Comprehensive visualization and results interpretation.

**Key Features**:

- Probability distribution histograms
- Per-class metrics grouped bar charts
- Confidence calibration analysis
- **Comprehensive markdown**: Overall performance, confusion patterns, feature validation, deployment readiness, future enhancements

**Learning Outcomes**: Interpret performance in business context, assess calibration, identify improvement opportunities.

---

### 7. 07_scenario_based_evaluation.ipynb - Edge Case Testing

**Purpose**: Validate model on curated test scenarios.

**Key Features**:

- Handcrafted incident scenarios (clear-cut, ambiguous, adversarial, novel phrasing)
- Expected vs predicted comparison
- Failure analysis with confidence scoring

**Learning Outcomes**: Understand model limitations, vocabulary gaps, misleading confidence.

---

### 8. 08_model_comparison.ipynb - Multi-Model Benchmarking

**Purpose**: Compare Logistic Regression, Linear SVM, and Random Forest.

**Key Features**:

- Side-by-side performance metrics
- Enhanced confusion matrices (dual plots + comparative heatmaps)
- Model agreement analysis
- Training time comparison
- **Section 8.6 comprehensive analysis**: Performance ranking, per-class winners, deployment recommendation

**Learning Outcomes**: Algorithm selection tradeoffs, when accuracy gains don't justify complexity, ensemble opportunities.

---

### 9. 09_operational_decision_support.ipynb - Uncertainty & Thresholds

**Purpose**: Operationalize model with uncertainty quantification.

**Key Features**:

- Uncertainty metrics (confidence, entropy, top-2 gap)
- ROC/PR curves per class
- Confidence calibration diagrams
- SOC analyst escalation framework (auto-triage >0.9, review 0.7-0.9, escalate <0.7)

**Learning Outcomes**: Quantify prediction uncertainty, set confidence thresholds, design human-in-the-loop workflows.

---

### 10. 10_hybrid_model.ipynb - Text + Metadata Feature Fusion (NEW)

**Purpose**: Combine TF-IDF with structured metadata.

**Key Features**:

- Enhanced preprocessor with detailed text summaries
- Feature engineering: TF-IDF (3000) + structured (severity, log_source, protocol, ports, is_true_positive)
- ColumnTransformer for heterogeneous features
- 4-model comparison (TF-IDF only, metadata only, hybrid LogReg, hybrid RF)
- Performance decomposition and interpretation

**Learning Outcomes**: Master ColumnTransformer, understand feature fusion tradeoffs, design multi-input pipelines.

---

## Getting Started

```bash
# Install dependencies
pip install -e ".[dev]"
pip install jupyterlab

# Launch Jupyter Lab
jupyter lab notebooks/
```

### Recommended Reading Order

**New users**: **00** → 01 → 02 → 03 → 05 → 06-09 → 10  
**SOC analysts**: **00** → 01 → 05 → 09  
**ML engineers**: **00** → 03 → 04 → 08 → 10  
**Contributors**: **00** → 01 → 02 → 03

---

## Notebook Enhancements

✅ **Professional Visualizations**: Custom colormaps, dual confusion matrices, grouped bar charts  
✅ **Comprehensive Markdown**: Results interpretation, deployment readiness, future enhancements  
✅ **Bug Fixes**: Notebook 01 alignment, 05 duplicate output/Ctrl+C, 09 histogram 'kde' parameter, 10 enhanced preprocessor outputs  
✅ **Code Quality**: Consistent styling, reproducible seeds (random_state=42), modular cells

---

## Common Issues & Solutions

**"Model file not found"**: Run notebooks 02-03 to generate artifacts  
**"Dataset not found"**: Generate with `./launch_generator.sh` or download pre-generated  
**Memory issues**: Load subset with `pd.read_csv(..., nrows=10000)`  
**Plots not displaying**: Add `%matplotlib inline` to first cell  
**Kernel crashes**: Reduce `max_features` in TF-IDF vectorizer

---

## Best Practices

1. Run notebooks in order first time (artifact dependencies)
2. Clear outputs before committing (`jupyter nbconvert --clear-output`)
3. Set random seeds for reproducibility
4. Regenerate artifacts if dataset changes
5. Test on subset before full 100K dataset

---

## Additional Resources

- **Dataset Details**: [Data & Generator Guide](data-and-generator.md)
- **Production Scripts**: [Production Generation Guide](production-generation.md)
