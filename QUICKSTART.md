# Quick Start Guide

Get AlertSage running in 5 minutes!

## Prerequisites

- **Python 3.11+** (check with `python --version`)
- **Git** (check with `git --version`)
- **~500 MB free disk space** for models and dataset

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/texasbe2trill/AlertSage.git
cd AlertSage
```

### 2. Create Virtual Environment

```bash
# macOS/Linux
python3.11 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Package

```bash
pip install -e ".[dev]"
```

This installs AlertSage in editable mode with all dependencies.

## Verify Installation

### Run Tests

```bash
pytest
```

You should see all 9 tests pass. The first run will automatically download the model artifacts (~10 MB).

### Test CLI

```bash
nlp-triage "User reported suspicious email with attachment"
```

You should see a formatted output with classification results.

## Next Steps

### Try the Streamlit UI

```bash
streamlit run app.py
```

Your browser will open to `http://localhost:8501` with an interactive dashboard.

### Explore Notebooks

```bash
jupyter notebook notebooks/
```

Start with `01_explore_dataset.ipynb` to see the full workflow.

### Generate Custom Dataset

```bash
# Generate 1000 incidents (quick test)
python generator/generate_cyber_incidents.py --n-events 1000
```

## Common Issues

### "Module 'triage' not found"

Make sure you ran `pip install -e ".[dev]"` and your virtual environment is activated.

### "No such file: cyber_incidents_simulated.csv"

The dataset auto-downloads when you run tests or notebooks. Manually download if needed:

```bash
pytest tests/test_model_artifacts.py
```

### "Python 3.11 not found"

Install Python 3.11+ from [python.org](https://www.python.org/downloads/) or use a version manager like `pyenv`.

### Port 8501 already in use (Streamlit)

```bash
streamlit run app.py --server.port 8502
```

## Quick Command Reference

```bash
# CLI with threshold adjustment
nlp-triage --threshold 0.7 "Website experiencing slowdowns"

# JSON output for scripting
nlp-triage --json "Multiple failed login attempts"

# LLM second opinion (requires LLM model)
nlp-triage --llm-second-opinion "Server encrypting files"

# Run all tests
pytest -v

# Check code coverage
pytest --cov=src/triage --cov-report=term-missing

# Preview documentation
mkdocs serve
```

## Documentation

- **Full Documentation**: [https://texasbe2trill.github.io/AlertSage/](https://texasbe2trill.github.io/AlertSage/)
- **CLI Guide**: [docs/cli.md](docs/cli.md)
- **UI Guide**: [docs/ui-guide.md](docs/ui-guide.md)
- **Development**: [docs/development.md](docs/development.md)

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/texasbe2trill/AlertSage/issues)
- **Discussions**: [GitHub Discussions](https://github.com/texasbe2trill/AlertSage/discussions)
- **Contributing**: [CONTRIBUTING.md](CONTRIBUTING.md)

## What's Next?

1. ✅ Read the [full README](README.md)
2. ✅ Try different CLI options and thresholds
3. ✅ Explore the Streamlit UI features
4. ✅ Walk through the Jupyter notebooks
5. ✅ Generate your own synthetic dataset
6. ✅ Read the [documentation site](https://texasbe2trill.github.io/AlertSage/)

Happy triaging! 🛡️
