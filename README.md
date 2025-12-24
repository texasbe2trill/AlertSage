# 🛡️ AlertSage: NLP-Driven Incident Triage


![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![NLP](https://img.shields.io/badge/NLP-Cybersecurity%20Triage-orange)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-green)
![Dataset](https://img.shields.io/badge/Data-Synthetic%20500k%20Incidents-purple)
![Notebooks](https://img.shields.io/badge/Jupyter-Workflow-important)
![CLI](https://img.shields.io/badge/CLI-Incident%20Triage-lightgrey)
![LLM](https://img.shields.io/badge/LLM-Enhanced%20Generation-blueviolet)
![UI](https://img.shields.io/badge/Streamlit-Interactive%20UI-red)

An educational/research platform demonstrating intelligent cybersecurity incident triage through NLP. Features synthetic dataset generation with LLM enhancement, uncertainty-aware classification, interactive Streamlit UI, and production-grade monitoring tools for SOC automation research.

> **⚠️ Not production incident response tooling**  
> This repo is for learning, experimentation, and portfolio use. It is not intended to run unsupervised in a live SOC.

---

## Quick links

- **Docs site**: published with MkDocs Material (see [docs/index.md](docs/index.md) locally or `https://texasbe2trill.github.io/AlertSage/` after deployment)
- **Getting started**: [docs/getting-started.md](docs/getting-started.md)
- **CLI usage**: [docs/cli.md](docs/cli.md)
- **Data & generator**: [docs/data-and-generator.md](docs/data-and-generator.md)
- **Development workflow**: [docs/development.md](docs/development.md)

---

## 🌟 Highlights

- **🤖 LLM-Enhanced Dataset Generation** with local llama.cpp models, intelligent rewriting, sanitization, and production monitoring scripts
- **⚡ GPU Acceleration**: Automatic Metal/CUDA/Vulkan detection for ~10x faster LLM inference (6-7s vs 60s per incident on CPU)
- **🎯 Interactive Streamlit UI** with real-time classification, bulk analysis, LLM second-opinion integration, and visual analytics
- **🚀 Smart LLM Mode**: Two-pass optimization strategy for 60-80% faster batch processing (baseline analysis → selective LLM on uncertain cases)
- **📊 Synthetic SOC Dataset** (100k incidents) with multi-perspective narratives, MITRE ATT&CK enrichment, and realistic label noise
- **🧠 Uncertainty-Aware Classification** using TF-IDF + Logistic Regression with configurable thresholds and intelligent fallback handling
- **🔍 LLM Second-Opinion Engine** for uncertain cases with JSON guardrails, SOC keyword validation, and hallucination prevention
- **📈 Production Monitoring Tools** for dataset generation with progress tracking, ETA calculation, and resource efficiency metrics
- **🛠️ Rich CLI Experience** with ASCII banners, probability tables, bulk processing, and JSON output modes
- **📚 Comprehensive Notebooks** (01-10) covering exploration, training, interpretability, hybrid models, and operational decision support
- **🔬 Shared Preprocessing Pipeline** ensuring training/inference alignment across CLI, UI, and notebooks
- **📦 Professional Packaging** with pytest suite, MkDocs documentation, and CI/CD workflows

## 📋 Feature Overview

| Category                | Feature                       | Description                                                  |
| ----------------------- | ----------------------------- | ------------------------------------------------------------ |
| 🎨 **User Interface**   | Streamlit Web Application     | Interactive triage with visual analytics and bulk processing |
| 🚀 **Performance**      | Smart LLM Mode                | Two-pass optimization: 60-80% faster than full LLM analysis  |
| ⚡ **GPU Acceleration** | Automatic GPU Detection       | Auto-enables Metal/CUDA/Vulkan for ~10x LLM speedup          |
| 🤖 **LLM Integration**  | Local llama.cpp Models        | Privacy-first LLM for generation and second opinions         |
| 📊 **Dataset**          | Synthetic SOC Corpus (500k)   | Realistic incidents with noise, typos, conflicting signals   |
| 🔄 **Rewrite Engine**   | LLM-Enhanced Generation       | Intelligent rewriting with sanitization & caching            |
| 🛠️ **CLI Tools**        | nlp-triage Command            | Rich-formatted CLI with uncertainty logic and JSON modes     |
| 📈 **Monitoring**       | Production Generation Scripts | Real-time progress, ETA, resource tracking, auto-refresh     |
| 🧠 **Modeling**         | TF-IDF + Logistic Regression  | Reproducible baseline with uncertainty-aware predictions     |
| 🔍 **Second Opinion**   | LLM Triage Assistant          | Guardrails against hallucinations, SOC keyword intelligence  |
| 📚 **Analysis**         | 10 Jupyter Notebooks          | End-to-end workflow from exploration to decision support     |
| 🎯 **MITRE Mapping**    | ATT&CK Framework Integration  | Technique enrichment and adversary behavior modeling         |
| 📖 **Documentation**    | MkDocs Material Site          | Comprehensive docs with API reference and tutorials          |
| ✅ **Quality**          | CI/CD + Testing               | Automated tests, GitHub Actions, professional packaging      |

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYNTHETIC DATA GENERATION                        │
│  ┌────────────────────┐    ┌──────────────────┐                     │
│  │ Generator Scripts  │───▶│  LLM Rewriter    │                     │
│  │ • MITRE enrichment │    │  • llama.cpp     │                     │
│  │ • Label noise      │    │  • Sanitization  │                     │
│  │ • Realistic typos  │    │  • Caching       │                     │
│  └────────────────────┘    └──────────────────┘                     │
│           │                         │                               │
│           └─────────┬───────────────┘                               │
│                     ▼                                               │
│         ┌──────────────────────┐                                    │
│         │  cyber_incidents.csv │                                    │
│         │  (100k+ incidents)   │                                    │
│         └──────────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                           │
│         ┌─────────────────────────────────────────┐                 │
│         │  clean_description (shared module)      │                 │
│         │  • Unicode normalization                │                 │
│         │  • Lowercase + punctuation cleanup      │                 │
│         │  • TF-IDF vectorization                 │                 │
│         └─────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    BASELINE CLASSIFIER                              │
│              ┌─────────────────────────────┐                        │
│              │ Logistic Regression Model   │                        │
│              │ • TF-IDF features           │                        │
│              │ • Class-balanced training   │                        │
│              │ • Uncertainty threshold     │                        │
│              └─────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                           ▼
┌──────────────────────────┐   ┌────────────────────────────┐
│  HIGH CONFIDENCE PATH    │   │  UNCERTAIN PATH            │
│  • Direct classification │   │  • Low confidence score    │
│  • MITRE mapping         │   │  • Ambiguous signals       │
│  • SOC rationale         │   └────────────┬───────────────┘
└──────────────────────────┘                │
                ▲                           ▼
                │              ┌────────────────────────────┐
                │              │  LLM SECOND OPINION        │
                │              │  • JSON parsing guardrails │
                │              │  • SOC keyword validation  │
                │              │  • Label normalization     │
                │              │  • Rationale generation    │
                │              └────────────┬───────────────┘
                │                           │
                └───────────┬───────────────┘
                            ▼
        ┌──────────────────────────────────────────┐
        │         INTERFACES & OUTPUT              │
        ├──────────────────┬───────────────────────┤
        │  Rich CLI        │  Streamlit UI         │
        │  • Interactive   │  • Visual analytics   │
        │  • Bulk mode     │  • Bulk upload        │
        │  • JSON output   │  • LLM integration    │
        └──────────────────┴───────────────────────┘
```

---

## 🚀 Quick Start

Get AlertSage running in 5 minutes.

### Prerequisites
- Python 3.11+ (`python --version`)
- Git (`git --version`)

### 1) Install
```bash
# Clone
git clone https://github.com/texasbe2trill/AlertSage.git
cd AlertSage

# Create venv
# macOS/Linux
python3.11 -m venv .venv && source .venv/bin/activate
# Windows
# python -m venv .venv && .venv\Scripts\activate

# Install dev extras (editable)
pip install -e ".[dev]"
```

### 2) Verify
```bash
# Run tests (first run auto-downloads model artifacts ~10 MB)
pytest

# Optional: coverage (readable report)
pytest --cov=src/triage --cov-report=term-missing
```

### 3) Use the UI
```bash
streamlit run ui_premium.py
# If 8501 is busy:
streamlit run ui_premium.py --server.port 8502
```

### 4) Use the CLI
```bash
# Basic classification
nlp-triage "User reported suspicious email with attachment"

# JSON output for scripting
nlp-triage --json "Multiple failed login attempts detected"

# Bulk processing with LLM assistance
nlp-triage --llm-second-opinion \
  --input-file data/incidents.txt \
  --output-file results.jsonl

# High-difficulty mode for ambiguous cases
nlp-triage --difficulty soc-hard "Website experiencing slowdowns"

```

### Hosted Hugging Face Inference (no local model)

Prefer not to download a GGUF model? Use Hugging Face Inference instead:

1. Export `HF_TOKEN` (or `TRIAGE_HF_TOKEN`) with your personal token.
2. Optional: set `TRIAGE_HF_MODEL`/`HF_MODEL` (default `mistralai/Mistral-7B-Instruct-v0.3`).
3. The CLI automatically routes LLM second-opinion calls to Hugging Face when a token is present—no local model required.
4. In the Streamlit UI, enable **LLM Enhancement**, choose **Hugging Face Inference**, and supply your token in the sidebar. The UI enforces a 5-requests/60s session limit, an 8k-character input cap, and a 512 max-token generation limit per request.

Tokens are never hardcoded; keep them in environment variables or a local secrets manager.

### LLM Model Setup (required for LLM features)

LLM-assisted features use local llama.cpp models in GGUF format. Download a model and point the app to it:

```bash
# 1) Create a models directory
mkdir -p models

# 2) Install llama-cpp-python with GPU acceleration (macOS Metal)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# Alternative: CPU-only
# pip install llama-cpp-python

# 3) Download a GGUF model via Hugging Face (choose ONE)

# Option A: Llama 3.1 8B Instruct (quality, larger)
huggingface-cli download TheBloke/Llama-3.1-8B-Instruct-GGUF \
  Llama-3.1-8B-Instruct-Q6_K.gguf --local-dir models

# Option B: Mistral 7B Instruct v0.2 (solid, mid-size)
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q6_K.gguf --local-dir models

# Option C: TinyLlama 1.1B Chat (very small, CPU-friendly)
huggingface-cli download TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF \
  TinyLlama-1.1B-Chat-v1.0.Q6_K.gguf --local-dir models

# 4) Point AlertSage at your model path
export TRIAGE_LLM_MODEL="$(pwd)/models/Llama-3.1-8B-Instruct-Q6_K.gguf"  # or whichever you downloaded

# 5) Test CLI LLM second opinion
nlp-triage --llm-second-opinion "Server began encrypting shared folders."

# Optional: Streamlit UI with LLM enabled (toggle in sidebar)
streamlit run ui_premium.py

# Debug/disable GPU
export TRIAGE_LLM_DEBUG=1
export LLAMA_N_GPU_LAYERS=0   # force CPU if needed
```

Notes:
- The default fallback path is models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf; using `TRIAGE_LLM_MODEL` is preferred.
- Some models (e.g., Llama 3.x) require license acceptance on Hugging Face; authenticate before download.
- GPU backends are auto-enabled (Metal/CUDA) when available; see docs for tuning.
```

### 5) Docs
```bash
mkdocs serve
```

### Dataset (Release Asset)
- Canonical CSV is distributed as a compressed release asset: [GitHub Releases](https://github.com/texasbe2trill/AlertSage/releases).
- Tests and notebooks auto-download artifacts on first run.
- To generate locally:
```bash
# Quick generation (1000 incidents)
python generator/generate_cyber_incidents.py --n-events 1000
```
For larger runs and monitoring, see [docs/data-and-generator.md](docs/data-and-generator.md).

---

## 📖 Documentation

- **🏠 Home**: [Docs Site](https://texasbe2trill.github.io/AlertSage/)
- **🚀 Getting Started**: [docs/getting-started.md](docs/getting-started.md)
- **💻 CLI Usage**: [docs/cli.md](docs/cli.md)
- **🎨 Streamlit UI Guide**: [docs/ui-guide.md](docs/ui-guide.md)
- **📊 Data & Generator**: [docs/data-and-generator.md](docs/data-and-generator.md)
- **🏭 Production Generation**: [docs/production-generation.md](docs/production-generation.md)
- **🔬 Development Workflow**: [docs/development.md](docs/development.md)
- **📚 Notebooks**: [docs/notebooks.md](docs/notebooks.md)
- **🎯 Modeling & Eval**: [docs/modeling-and-eval.md](docs/modeling-and-eval.md)

---

## 🛠️ Production Dataset Generation

The project includes professional-grade bash orchestration scripts for large-scale dataset generation with LLM enhancement and real-time monitoring:

### Launch Generator

```bash
# Generate 100k incidents with default settings (LLM enhancement enabled)
./generator/launch_generator.sh

# Custom size and name
./generator/launch_generator.sh 50000 training_data

# Fresh start (delete existing files)
./generator/launch_generator.sh 100000 my_dataset --fresh
```

**Features:**

- 🤖 **LLM Enhancement**: Optional Llama-2-13B-Chat for realistic narrative rewrites (1% of events by default)
- 💾 **Checkpointing**: Automatic progress saving every 100 events - resume interrupted generations seamlessly
- 🔄 **Background Processing**: Uses `nohup` for SSH-safe, unattended operation (6-9 hours for 100K events)
- ⚙️ **Environment Configuration**: Automatically sets LLM model paths, rewrite probability, temperature

### Monitor Progress

```bash
# Single snapshot
./generator/monitor_generation.sh

# Auto-refresh every 30 seconds (recommended)
./generator/monitor_generation.sh --watch

# Custom refresh interval (10 seconds)
./generator/monitor_generation.sh --watch 10

# Simple mode for SSH/tmux
./generator/monitor_generation.sh --simple-color --watch
```

**Dashboard features:**

- 📊 Real-time progress tracking with visual progress bar and ETA
- 💻 CPU/memory usage with per-core utilization
- 🎮 **GPU Acceleration Metrics** (Apple Silicon): Metal detection, LLM throughput (~18.5 tokens/sec)
- 📈 Throughput trend analysis (accelerating/declining/steady)
- ⚡ Resource efficiency (events/CPU%, events/GB RAM)
- 🔄 Chunk timing and interval analysis
- 📝 Last 5 log entries tail
- 🎯 Quick action commands (kill process, check progress, view logs)

**Performance:**

- **GPU-accelerated (auto-enabled)**: ~10x faster LLM inference (6-7s per rewrite vs 60s on CPU)
- With LLM (1% rewrite, GPU): ~3-5 events/sec (100K in 6-9 hours)
- Without LLM: ~50-100 events/sec (100K in 20-30 minutes)

**GPU Acceleration:**

All LLM features automatically detect and enable GPU acceleration (Metal for Apple Silicon, CUDA for NVIDIA, Vulkan fallback):

- ✅ No configuration required - auto-enabled in CLI, UI, and generator
- ✅ ~10x speedup: 6-7 seconds per incident vs 60+ seconds on CPU
- ✅ Debug mode shows GPU configuration: `export TRIAGE_LLM_DEBUG=1`
- ✅ Disable if needed: `export LLAMA_N_GPU_LAYERS=0`

See [docs/production-generation.md](docs/production-generation.md) for comprehensive guide including troubleshooting, performance optimization, and advanced workflows.

---

## 🎨 Streamlit UI Features

The interactive web interface (`ui_premium.py`) provides a comprehensive triage experience:

### Single Incident Analysis

- Real-time classification with confidence visualization
- LLM second-opinion toggle for uncertain cases
- MITRE ATT&CK technique mapping
- Kill chain phase analysis
- Interactive probability distributions
- SOC triage recommendations

### Bulk Analysis

- CSV file upload and processing
- **Smart LLM Mode** (⚡ 60-80% faster): Two-pass strategy that runs baseline analysis on all incidents, then applies expensive LLM inference only to uncertain cases (confidence < threshold)
- Session state caching: Results persist across downloads and UI interactions
- Batch LLM enhancement for uncertain predictions
- Comprehensive threat intelligence briefs
- Advanced analytics dashboard
- Export results to CSV/JSON/Markdown
- Interactive filtering and exploration

### Visualization & Insights

- Label distribution charts
- Confidence heatmaps
- MITRE technique coverage
- Time-based analysis
- Risk scoring
- Uncertainty analysis

---

## CLI overview

`nlp-triage` is exposed via `pyproject.toml` entry points. Key capabilities:

- Shared text cleaning and TF–IDF vectorization.
- Uncertainty-aware predictions with configurable threshold (`--threshold`, default 0.50) and `uncertain` fallback.
- Rich formatting: ASCII banner, cleaned-text panel, probability tables, progress bar.
- JSON output (`--json`) for scripts and tests.
- Interactive loop when called without a positional argument.
- **SOC Difficulty Mode**: creates intentionally ambiguous test cases (`nlp-triage --difficulty soc-hard`) for scenario-based evaluation.
- Optional **LLM second-opinion mode** (`--llm-second-opinion`) with JSON parsing hardening, SOC keyword intelligence, and deterministic rationale generation.

Help menu:

```bash
nlp-triage -h
```

```text
usage: nlp-triage [-h] [--json] [--threshold THRESHOLD] [--max-classes MAX_CLASSES] [--difficulty {default,soc-medium,soc-hard}]
                  [--input-file INPUT_FILE] [--output-file OUTPUT_FILE] [--llm-second-opinion]
                  [text]

Cybersecurity Incident NLP Triage CLI

positional arguments:
  text                  Incident description

options:
  -h, --help            show this help message and exit
  --json                Return raw JSON output instead of formatted text
  --threshold THRESHOLD
                        Uncertainty threshold (default=0.5)
  --max-classes MAX_CLASSES
                        Maximum number of classes to display in the probability table
  --difficulty {default,soc-medium,soc-hard}
                        Difficulty / strictness mode for uncertainty handling. Use 'soc-hard' to mark more cases as 'uncertain'.
  --input-file INPUT_FILE
                        Optional path to a text file for bulk mode; each non-empty line is treated as an incident description.
  --output-file OUTPUT_FILE
                        Optional path to write JSONL predictions for bulk mode. Each line will contain one JSON object.
  --llm-second-opinion  If set, call a local LLM (e.g., Llama-2-7B-GGUF via llama-cpp-python) to provide a second opinion when the baseline model is
                        uncertain.
```

---

## 💡 CLI Usage Examples

### Basic Operations

```bash
nlp-triage "User reported a suspicious popup on FINANCE-WS-07."
```

Outputs include:

- Cleaned text
- Probability table
- Final label (or `uncertain`)
- MITRE ATT&CK® mapping
- Deterministic SOC rationale

---

### JSON Mode

Machine-readable output:

```bash
nlp-triage --json "Large data transfer from laptop to Dropbox."
```

---

### Bulk Mode

Process multiple incidents from a file:

```bash
nlp-triage --input-file data/incidents.txt
```

Write JSONL output:

```bash
nlp-triage --input-file data/incidents.txt --output-file data/results.jsonl
```

---

### Threshold Controls

```bash
# More uncertain
nlp-triage --threshold 0.70 "Website slowdowns after hours."

# Less uncertain
nlp-triage --threshold 0.30 "Single phishing email blocked."
```

Limit displayed classes:

```bash
nlp-triage --max-classes 3 "Multiple failed logins."
```

---

### Difficulty Modes

```bash
nlp-triage --difficulty default "Employee copied files to USB."
nlp-triage --difficulty soc-medium "Odd login patterns during travel."
nlp-triage --difficulty soc-hard "Website slow after patching."
```

---

### LLM Second-Opinion Mode

Enable LLM review on uncertain predictions:

```bash
nlp-triage --llm-second-opinion "Server began encrypting shared folders."
```

Bulk mode:

```bash
nlp-triage --llm-second-opinion \
  --input-file data/incidents.txt \
  --output-file data/triage_llm.jsonl
```

Debug logs (includes GPU configuration):

```bash
export TRIAGE_LLM_DEBUG=1
nlp-triage --llm-second-opinion --input-file data/incidents.txt
```

GPU acceleration is **auto-enabled** for ~10x speedup. Disable if needed:

```bash
export LLAMA_N_GPU_LAYERS=0  # Force CPU-only mode
nlp-triage --llm-second-opinion "Server encrypting files."
```

---

### Combined Examples

**One-off with pretty output:**

```bash
nlp-triage "Multiple VPN credential failures followed by a successful login."
```

**Bulk triage with JSONL output:**

```bash
nlp-triage \
  --difficulty default \
  --threshold 0.50 \
  --input-file data/incidents.txt \
  --output-file data/results.jsonl
```

**SOC-hard + LLM second opinion:**

```bash
nlp-triage \
  --difficulty soc-hard \
  --llm-second-opinion \
  --input-file data/incidents.txt \
  --output-file data/results_soc_hard_llm.jsonl
```

**Script-friendly JSON mode:**

```bash
nlp-triage --json --threshold 0.6 \
  "Unusual data transfer to unfamiliar cloud provider from FINANCE-WS-07."
```

---

## 📊 Dataset & Generator

### Automatic Dataset Management

- **Auto-download**: Dataset automatically downloads when running tests or notebooks if not present
- **Location**: `data/cyber_incidents_simulated.csv` (~107MB, excluded from git)
- **Size**: 100k incidents by default, configurable up to millions
- **Rich Schema**: Multiple narrative perspectives (`description`, `description_short`, `description_user_report`, `short_log`)

### Generation Features

- **MITRE ATT&CK Integration**: Realistic adversary behavior descriptions
- **Label Noise**: Configurable confusion between similar threat types
- **LLM Enhancement**: Optional rewriting with local llama.cpp models
- **Typos & Abbreviations**: Realistic SOC analyst writing patterns
- **Checkpointing**: Resume interrupted generations
- **Monitoring**: Real-time progress tracking and ETA calculation

### Quick Generation

```bash
# Basic generation (1000 incidents)
python generator/generate_cyber_incidents.py --n-events 1000

# With LLM enhancement
python generator/generate_cyber_incidents.py \
  --n-events 5000 \
  --use-llm \
  --rewrite-report audit.json

# Production mode with monitoring
./generator/launch_generator.sh 50000 my_dataset
./generator/monitor_generation.sh my_dataset --watch
```

See [docs/data-and-generator.md](docs/data-and-generator.md) for schema details, customization options, and advanced configuration.

---

## 🧠 Modeling & Evaluation

### Model Architecture

- **Text Representation**: TF-IDF (unigrams + bigrams, ~5k features, stopword removal)
- **Baseline Classifier**: Logistic Regression with class balancing (`max_iter=2000`)
- **Uncertainty Handling**: Configurable confidence threshold with `uncertain` fallback
- **Additional Experiments**: Linear SVM, Random Forest, ensemble methods

### Saved Artifacts

```
models/
├── vectorizer.joblib           # TF-IDF vectorizer
├── baseline_logreg.joblib      # Trained classifier
├── X_train_tfidf.joblib        # Cached train features
├── X_test_tfidf.joblib         # Cached test features
├── y_train.joblib              # Train labels
└── y_test.joblib               # Test labels
```

### Notebook Workflow

The repository includes **10 comprehensive Jupyter notebooks** covering the complete ML pipeline from exploration to production-ready hybrid models:

1. **01_explore_dataset.ipynb** - Dataset exploration & quality assessment (EDA with enhanced visualizations)
2. **02_prepare_text_and_features.ipynb** - Feature engineering & text preprocessing (TF-IDF pipeline)
3. **03_baseline_model.ipynb** - Logistic Regression baseline training (92-95% accuracy, dual confusion matrices)
4. **04_model_interpretability.ipynb** - Feature importance & coefficient analysis (model explainability)
5. **05_inference_and_cli.ipynb** - Prediction workflow & CLI testing (with Ctrl+C handling)
6. **06_model_visualization_and_insights.ipynb** - Performance analysis & comprehensive interpretation
7. **07_scenario_based_evaluation.ipynb** - Edge case & scenario testing (adversarial examples)
8. **08_model_comparison.ipynb** - Multi-model benchmarking (LogReg, SVM, Random Forest + comparative heatmaps)
9. **09_operational_decision_support.ipynb** - Uncertainty & threshold analysis (SOC workflow integration)
10. **10_hybrid_model.ipynb** - Text + metadata feature fusion (ColumnTransformer, 4-model comparison) **NEW**

**All notebooks enhanced with:**

- ✅ Professional visualizations (custom colormaps, dual confusion matrices, grouped bar charts)
- ✅ Comprehensive markdown analysis (results interpretation, deployment readiness, future enhancements)
- ✅ Bug fixes (alignment, duplicate output, parameter errors, preprocessor outputs)
- ✅ Code quality improvements (consistent styling, reproducible seeds, modular cells)

See [docs/notebooks.md](docs/notebooks.md) for detailed features, learning outcomes, and troubleshooting guide.

---

## 🎯 Realistic Model Behavior

This project models the ambiguity and uncertainty inherent in real-world SOC (Security Operations Center) operations. Rather than producing "perfect" predictions, the system exhibits realistic behaviors that mirror how human analysts triage incidents under time pressure and incomplete information.

### Key Characteristics

✅ **Confident on Clear Cases**

- Obvious malware infections
- Textbook phishing attempts
- Direct web attacks
- Clear policy violations

⚠️ **Appropriately Uncertain on Edge Cases**

- Impossible travel vs. legitimate VPN/sync behavior
- Large file transfers vs. normal business workflows
- Website slowdowns (DDoS vs. maintenance vs. load)
- Credential reuse vs. compromise vs. password manager

🧠 **Human-Like Decision Patterns**

- Probabilistic reasoning with confidence scores
- `uncertain` fallback when evidence is ambiguous
- Multiple plausible explanations for single symptoms
- Scenario-driven behavior matching SOC reality

### Example CLI Output

![CLI Output](cli_output.png)

_The system correctly identifies high-confidence threats while marking ambiguous cases as "uncertain" for human review_

### Additional Examples (v0.2.0+)

**Bulk Analysis Dashboard**  
![Bulk Summary](docs/images/cli_output_bulk.png)
![Bulk Final Summary](docs/images/cli_output_bulk_2.png)

**Difficulty Mode Demonstrations**  
![Difficulty Mode](docs/images/cli_output_v2_soc.png)

📸 More screenshots available in `docs/images/`  
📖 Full CLI documentation: [docs/cli.md](docs/cli.md)

---

## ⚠️ Limitations & Disclaimers

### Technical Limitations

- **Synthetic Training Only**: Model trained exclusively on generated data; real-world accuracy unknown
- **No Live Integrations**: No SIEM/EDR/SOAR connections; designed for research and learning
- **Edge Case Sensitivity**: May misclassify novel or unusual incident patterns
- **Confidence Calibration**: Probability scores are relative, not absolute measures of certainty

### Operational Considerations

- **Decision Support Only**: Outputs should inform, not replace, human analyst judgment
- **Not Production-Ready**: Requires evaluation on real data before operational deployment
- **Privacy Awareness**: LLM features use local models; no data sent to external APIs
- **Resource Requirements**: LLM generation/inference requires sufficient CPU/RAM

### Recommended Use

✅ Educational exploration of NLP in cybersecurity  
✅ Research prototypes and experimentation  
✅ Portfolio demonstrations and skill development  
✅ Testing SOC automation concepts

❌ Unsupervised production incident response  
❌ Replacing trained security analysts  
❌ Critical decision-making without human oversight

More details: [docs/limitations.md](docs/limitations.md)

---

## 📜 MITRE ATT&CK® Attribution

This project references **MITRE ATT&CK®** techniques (T1078, T1190, T1486, T1566, etc.) and incorporates paraphrased descriptions from the public ATT&CK knowledge base for educational and research purposes.

**MITRE ATT&CK®** and **ATT&CK®** are registered trademarks of The MITRE Corporation.

ATT&CK data provided by The MITRE Corporation is licensed under:  
**Creative Commons Attribution-ShareAlike 4.0 International License**

🔗 Official source: [attack.mitre.org](https://attack.mitre.org)

---

## 🛠️ Development & Deployment

This project models the ambiguity and noise found in real SOC (Security Operations Center) investigations.  
Rather than producing “perfect” predictions, the classifier shows realistic uncertainties, overlaps, and  
borderline classifications that resemble how analysts triage incidents under time pressure.

### Key characteristics

- **Correct on clear-cut cases** (malware, phishing, web attacks, policy violations)
- **Ambiguity on edge cases**, such as:
  - Impossible travel vs benign sync behavior
  - Large file transfers vs normal business workflow
  - Website slowdown vs DDoS vs system maintenance
- **Human-like uncertainty** using an `uncertain` fallback when confidence is too low
- **Scenario-driven behavior**, matching SOC reality where multiple causes can explain one symptom

Example prediction from the CLI:

![CLI Output](cli_output.png)

### Additional Examples v0.2.0

- **Bulk Analysis Summary**  
  ![Bulk Summary](docs/images/cli_output_bulk.png)
  ![Bulk Final Summary](docs/images/cli_output_bulk_2.png)

- **Difficulty Mode Examples**  
  ![Difficulty Mode](docs/images/cli_output_v2_soc.png)

Screenshots live in `docs/images/`. Full breakdown: [docs/cli.md](docs/cli.md).

---

## Limitations

- Synthetic-only training – accuracy on real SOC data is unknown.
- Edge-case narratives may be misclassified or produce low confidence.
- No live integrations (SIEM/EDR/SOAR); CLI is decision-support only.
- Treat outputs as advisory; pair with human analyst review.

More context: [docs/limitations.md](docs/limitations.md).

## MITRE ATT&CK® Attribution

This project references MITRE ATT&CK® techniques (for example, IDs such as T1078, T1190,
T1486, and T1566) and paraphrases portions of public ATT&CK technique descriptions for
educational and research purposes.

MITRE ATT&CK® and ATT&CK® are registered trademarks of The MITRE Corporation.

ATT&CK data is provided by The MITRE Corporation and is licensed under the Creative Commons
Attribution-ShareAlike 4.0 International License.

---

## Documentation & deployment

## 🛠️ Development & Deployment

### Local Documentation Preview

```bash
pip install mkdocs-material
mkdocs serve
```

Visit `http://127.0.0.1:8000` to preview the documentation site locally.

### Deploy to GitHub Pages

```bash
mkdocs gh-deploy --force
```

Automated deployment via `.github/workflows/docs.yml` runs on pushes to `main` and can be triggered manually via **Run workflow** in the Actions tab.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_cli.py -v

# With coverage
pytest tests/ --cov=src/triage --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ tests/

# Type checking (if configured)
mypy src/

# Lint
flake8 src/ tests/
```

---

## 🚀 Future Roadmap

### Planned Features

- [x] **Notebook 10**: Hybrid model with text + metadata features ✅ **COMPLETED**
- [ ] **Transformer Models**: BERT/RoBERTa variants for improved accuracy
- [ ] **Deep Learning Pipeline**: Fine-tuning with MITRE-enriched text
- [ ] **Active Learning**: Interactive labeling interface for model improvement
- [ ] **Multi-language Support**: Non-English incident descriptions
- [ ] **API Service**: REST API for programmatic access

### Infrastructure Enhancements

- [ ] **Docker Deployment**: Containerized setup with docker-compose
- [ ] **Expanded CI/CD**: More linters, security scanning, automated releases
- [ ] **Performance Optimization**: Caching, batch processing improvements
- [ ] **Real-world Evaluation**: Testing on publicly available SOC datasets

### Documentation & Community

- [ ] **Video Tutorials**: Setup and usage walkthroughs
- [ ] **Blog Posts**: Deep dives into specific features
- [ ] **Contribution Guide**: Detailed guidelines for contributors
- [ ] **Example Integrations**: Sample SIEM/SOAR connectors

**Contributions Welcome!** Issues, pull requests, and feedback are encouraged.

---

## 📄 License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) for the full text and [NOTICE](NOTICE) for attribution.

## 🙌 Attribution

- Primary authorship: Chris Campbell and contributors.
- Retain the project name “AlertSage” and include the NOTICE file in redistributed builds.
- Keep third-party licenses and notices with any vendored assets or code.



## 🙏 Acknowledgments

- **MITRE Corporation** for the ATT&CK® framework
- **Streamlit** for the excellent UI framework
- **scikit-learn** for machine learning tools
- **llama.cpp** for local LLM inference
- Open-source cybersecurity community for inspiration and knowledge sharing

## 📧 Contact & Support

- **Issues**: [GitHub Issues](https://github.com/texasbe2trill/AlertSage/issues)
- **Discussions**: [GitHub Discussions](https://github.com/texasbe2trill/AlertSage/discussions)
- **Documentation**: [https://texasbe2trill.github.io/AlertSage/](https://texasbe2trill.github.io/AlertSage/)
- **Security**: See [SECURITY.md](SECURITY.md) for reporting vulnerabilities

**⭐ If you find this project useful, please consider giving it a star!**
