# AlertSage

> A SOC-style incident triage console that classifies free-text security alerts, maps them to MITRE ATT&CK, and routes them through your LLM of choice. Open-source, dark-mode-first, modeled on production SIEM consoles.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.39%2B-FF4B4B.svg)](https://streamlit.io/)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen)](https://alertsage.streamlit.app/)
[![Tests](https://img.shields.io/badge/tests-9%20passing-green.svg)](tests/)

AlertSage takes a security analyst's most boring fifteen minutes (read the alert, decide a label, map to ATT&CK, write the rationale, paste actions into the ticket) and turns it into thirty seconds. It ships as a Streamlit-based SOC console plus a CLI; both run on the same TF-IDF + sentence-transformer + Logistic Regression pipeline, with an optional LLM second opinion routed through the provider you configure (Hugging Face, OpenAI, Anthropic, or local llama.cpp).

---

## Table of contents

- [What's new in v3.1.0](#whats-new-in-v310)
- [Showcase features](#showcase-features)
- [Live demo](#live-demo)
- [Quick start](#quick-start)
- [SOC console pages](#soc-console-pages)
- [LLM provider configuration (BYOK)](#llm-provider-configuration-byok)
- [Demo data generator](#demo-data-generator)
- [Architecture](#architecture)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Documentation](#documentation)
- [Contributing, license, security](#contributing-license-security)

---

## What's new in v3.1.0

A complete rewrite of the Streamlit UI, modeled on Splunk Enterprise Security and Elastic Security. Dark theme, severity as the primary color signal, JetBrains Mono for IDs and timestamps, all styling consolidated into one external stylesheet.

| Capability | Status |
|---|---|
| SOC-style six-page console (Overview, Investigate, Hunt, Batch, Bookmarks, Settings) | New |
| MITRE ATT&CK kill chain visualization on Investigate | New |
| Auto-extracting IOC panel with VirusTotal enrichment + external pivots | New |
| Case status workflow (New / Triaging / Contained / Closed) | New |
| Case timeline that stitches creation, status changes, notes, bookmarks | New |
| MITRE ATT&CK heatmap on Overview | New |
| Brushable Splunk-style timechart with range selectors | New |
| Auto-refreshing live data panels (KPIs, charts, MITRE, recent events) | New |
| Auto-refreshing live tail of incoming events | New |
| Saved searches pinned to the sidebar | New |
| MITRE coverage report + three CSV exports from Batch | New |
| Anomaly score column on Hunt and Overview | New |
| Demo data generator (synthetic events on a 6 second timer) | New |
| BYOK panels for OpenAI, Anthropic, Hugging Face, VirusTotal | New |
| Local (GGUF) provider hidden when prerequisites are missing | New |
| Per-provider sliding-window rate limiter | New |

The classifier, MITRE mapping, and database stack are unchanged. The CLI (`nlp-triage`) is unchanged.

---

## Showcase features

### Mission control dashboard

- Six-tile KPI strip with severity-colored borders (total analyzed, critical+high count, last 24 hours, average classifier confidence, bookmarks, analyst notes). Auto-refreshes every 6 seconds.
- Stacked bar timechart of triage volume over 30 days, with a Splunk-style range slider and `1d / 7d / 14d / 30d / All` buttons. Auto-refreshes.
- Classifier confidence histogram in 20 buckets, color-coded by reliability band.
- Severity distribution donut.
- MITRE ATT&CK tactic-by-technique heatmap with cell intensity scaling from cool to hot. Auto-refreshes.
- Threat intelligence feed panel (TAXII-shaped mock data, swap in a real feed by replacing one constant).
- Auto-refreshing live tail with a pulsing live dot. New events stream in within ~5 seconds of being written.
- Recent events table with severity pill, **case status pill, anomaly score pill**, mono ID column, and truncated narrative. Auto-refreshes.

### Investigate (single-incident triage)

The headline showcase surface. After running a triage you see, top to bottom:

1. Event head card with mono event ID (`AS-000123`), classifier and LLM timing in milliseconds, severity pill, confidence pill, anomaly score pill, uncertainty pill.
2. **Case status stepper** with four-stage workflow: New → Triaging → Contained → Closed. Click any stage to advance; status persists across reloads via `db.save_setting`.
3. **MITRE ATT&CK kill chain** visualization. All 13 enterprise tactics render as cells in a horizontal flow. Tactics whose techniques the classifier mapped light up with the accent color and an indigo top stripe; matched technique IDs render as monospace chips inside the cell.
4. **Indicators and enrichment** panel. Auto-extracted IOCs (IPv4, IPv6, MD5, SHA1, SHA256, URL, email, domain, CVE, hostnames). Each IOC is an expander with verdict / score / first-seen / sources, plus per-IOC pivot links to VirusTotal, AbuseIPDB, Shodan, GreyNoise, Censys, URLhaus, MalwareBazaar, NVD, MITRE CVE.
5. **Case timeline** as a vertical narrative: triage created, LLM rationale, status changes, analyst notes, bookmarks. Each event has a colored dot, a kind label, and a JetBrains Mono timestamp.
6. **Class probabilities** with severity-colored progress bars (top N candidates configurable in the sidebar).
7. **MITRE techniques** as monospace chips.
8. **Analyst rationale** (LLM-authored or deterministic fallback) with severity-toned left border.
9. **Playbook hint** with recommended queue, priority, and a checkbox-style action list.
10. Footer actions: bookmark, add note, re-run.

### Hunt

Full-text search across triage history. Filters: free-text query, classification multiselect, severity multiselect, minimum confidence slider, **minimum anomaly score slider**, time-window selector (`Last hour / 24 hours / 7 days / 30 days / All time`). Results render in the same SOC table as Overview's recent events. Save the current filter set as a named search; it appears in the sidebar with a one-click apply.

### Batch

CSV upload (auto-detects `incident_text` / `description` / `narrative` / `alert` / `text` columns), runs the full pipeline on up to 500 rows with a progress bar, then summarizes:

- KPI strip (processed, critical+high, medium, benign+unknown, wall-clock elapsed).
- Label distribution panel with severity bars.
- **MITRE coverage report**: tactic-by-tactic event volume bars, plus a detailed expander showing every (tactic, technique, label) cell with severity breakdown.
- **Three CSV exports**: triage results, MITRE coverage, executive tactic rollup.

### Bookmarks

Saved investigations as expander cards. Each carries the severity pill, current case status pill, narrative quote (severity-toned), four-button case status stepper, optional analyst note, and the full case timeline.

### Settings

Provider radio (Local hidden when prerequisites fail). Per-provider configuration panels for OpenAI, Anthropic, Hugging Face, and VirusTotal, all with **password-masked Bring Your Own Key** fields. Demo data generator panel with toggle, "Emit one now" button, counter, last-emit timestamp, and last-error display. Triage default sliders. About panel.

---

## Live demo

A hosted demo runs on Streamlit Community Cloud. The backing classifier and HF provider are the same; demo runs there have the demo generator on by default so the dashboard is populated.

[Open the demo](https://alertsage.streamlit.app/)

---

## Quick start

### Prerequisites

- Python 3.12 (pinned via `runtime.txt`)
- `git` and a clone of the repo

### Install

```bash
git clone https://github.com/texasbe2trill/AlertSage.git
cd AlertSage

python3.12 -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate            # Windows

pip install -r requirements.txt
```

For development, tests, and notebooks:

```bash
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

### Launch the SOC console

```bash
streamlit run app.py
```

Opens at <http://localhost:8501>. First boot is faster than the previous build because the inline CSS, the 1900-line CLI module, and the unconditional 789 MB metrics joblib are all out of the cold-start path.

### Run the CLI

```bash
# Single classification
nlp-triage --text "User clicked a phishing link in their inbox"

# JSON output for scripting
nlp-triage --text "..." --json

# Bulk with LLM second opinion
nlp-triage --bulk incidents.csv --use-llm --difficulty soc-medium
```

---

## SOC console pages

| Page | Purpose |
|---|---|
| **Overview** | Mission control dashboard. Auto-refreshing KPIs, charts, MITRE heatmap, live tail, threat feed, recent events table. |
| **Investigate** | Triage one incident end-to-end with kill chain, IOC enrichment, case timeline, classification probabilities, LLM rationale, playbook hint. |
| **Hunt** | Search past triage results with full filter set + saved searches pinned to the sidebar. |
| **Batch** | CSV ingest with MITRE coverage report and three CSV exports. |
| **Bookmarks** | Saved investigations with case status workflow and timeline. |
| **Settings** | Provider configuration, BYOK, demo generator, triage defaults. |

Detailed walkthrough: [`docs/ui-guide.md`](docs/ui-guide.md).

---

## LLM provider configuration (BYOK)

AlertSage routes the LLM second opinion through whichever provider you select. **Keys live in session state only; they are never written to `data/triage.db` or any other file on disk.**

### Supported providers

| Provider | Default model | When to use |
|---|---|---|
| Hugging Face Inference | `meta-llama/Llama-3.1-8B-Instruct:cerebras` | Quick demo, free-tier or paid HF account |
| OpenAI | `gpt-4o-mini` | Best rationale quality on commodity hardware |
| Anthropic | `claude-haiku-4-5` | Best rationale quality plus longer context |
| Local llama.cpp | local `.gguf` file | Air-gapped, no network egress |

### Where to set credentials

1. **Sidebar / Settings BYOK fields** (session only, in-memory). Paste once per session.
2. `~/.streamlit/secrets.toml` or `<repo>/.streamlit/secrets.toml`:
   ```toml
   HF_TOKEN = "hf_xxx"
   HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
   ```
3. Environment variables: `TRIAGE_HF_TOKEN`, `TRIAGE_HF_MODEL`, `HF_TOKEN`, `HF_MODEL`, `VIRUSTOTAL_API_KEY`.

### Fallback chain

The dispatcher (`_build_llm_kwargs`) routes intelligently:

- OpenAI selected, no key → Hugging Face (if HF token present) → Local (if GGUF available).
- Anthropic selected, no key → same chain.
- Local selected, GGUF runtime not available → Hugging Face.

The sidebar caption shows the actually-active backend after fallbacks. Per-provider sliding-window rate limits keep one provider's quota from blocking the others.

### VirusTotal IOC enrichment

The IOC panel on Investigate is mocked by default. Paste a VirusTotal API key in **Settings → Threat intel enrichment** and the panel switches to live VT lookups (cached for 15 minutes per indicator). Free-tier keys work.

---

## Demo data generator

For public demos, fresh installs, or showcasing the console to a stakeholder, flip on **Settings → Demo data generator**. Every ~6 seconds AlertSage emits one synthetic incident drawn from a curated set of phishing, malware, access abuse, web attack, exfil, and benign narratives, runs it through the full pipeline, and seeds the case timeline. The Overview panels auto-refresh and pick it up.

Manual controls in the same panel:

- **Emit one now**: bypasses the timer for an immediate test.
- **Clear demo events**: removes synthetic rows (`batch_id = 'demo'`) without touching real triage history.
- Status row shows emit counter, last emit timestamp, and last error message in red when present.

---

## Architecture

```
                ┌───────────────────────────────────────────────────┐
                │                   app.py                          │
                │  Streamlit router + 6 pages + design system       │
                │  CSS lives in assets/styles.css                   │
                └─────────────────────────────┬─────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              ▼                               ▼                               ▼
  ┌─────────────────────┐       ┌─────────────────────┐         ┌─────────────────────┐
  │   src/triage/       │       │   data/triage.db    │         │   models/           │
  │  classifier         │       │  SQLite             │         │  vectorizer.joblib  │
  │  embeddings         │       │  history            │         │  enhanced_logreg    │
  │  llm_helpers        │       │  bookmarks          │         │  baseline_logreg    │
  │  llm_client (HF/    │       │  notes              │         │  Llama-3.1-8B GGUF  │
  │   OpenAI/Anthropic) │       │  case status        │         │   (optional)        │
  │  database           │       │  case timeline      │         └─────────────────────┘
  │  preprocess         │       │  saved searches     │
  │  cli.py             │       └─────────────────────┘
  └─────────────────────┘
```

The classifier is **TF-IDF (5000 dims) + sentence-transformer embeddings (384 dims) → Logistic Regression** over a fixed taxonomy of incident labels. Every UI prediction goes through `predict()` in `app.py` which concatenates TF-IDF and embeddings horizontally before calling `model.predict()`; this matches the training feature space of `enhanced_logreg.joblib`. The fallback `baseline_logreg.joblib` (TF-IDF only) is also supported.

The LLM second opinion is provider-routed. Each provider client (`HuggingFaceInferenceClient`, `OpenAIClient`, `AnthropicClient`, `LocalLLMClient`) implements the same `generate_json(prompt) -> dict` interface so the dispatcher in `llm_second_opinion()` is provider-agnostic.

---

## Project layout

```
AlertSage/
├── app.py                       # SOC console entry point (Streamlit)
├── assets/
│   ├── styles.css               # Single-source design system
│   └── icons/                   # SVG severity + brand icons
├── src/triage/
│   ├── cli.py                   # nlp-triage CLI entry point
│   ├── llm_helpers.py           # LLM dispatch, MITRE map, SOC playbook hints
│   ├── llm_client.py            # HF / OpenAI / Anthropic / local clients
│   ├── model.py                 # vectorizer + classifier loader
│   ├── embeddings.py            # sentence-transformer wrapper
│   ├── database.py              # SQLite schema + accessors
│   └── preprocess.py            # text cleaning
├── models/                      # classifier artifacts (logreg, vectorizer)
├── notebooks/                   # 12 educational notebooks
├── docs/                        # mkdocs site sources
├── tests/                       # pytest
├── data/                        # bundled synthetic dataset (gitignored: triage.db)
├── generator/                   # synthetic data generator
├── requirements.txt             # runtime dependencies (lean)
├── requirements-dev.txt         # dev, tests, docs
├── runtime.txt                  # python-3.12
└── pyproject.toml
```

---

## Tests

```bash
pytest tests/ -v
```

Currently 9 tests across the CLI, model artifacts, and preprocessing modules. Tests do not require the LLM provider or the GGUF model.

For coverage:

```bash
pytest tests/ --cov=src/triage --cov-report=term-missing
```

---

## Documentation

The full mkdocs site lives in `docs/`. Highlights:

| Page | Topic |
|---|---|
| [`docs/ui-guide.md`](docs/ui-guide.md) | SOC console walkthrough |
| [`docs/quickstart.md`](docs/quickstart.md) | Install + first triage |
| [`docs/cli.md`](docs/cli.md) | `nlp-triage` reference |
| [`docs/architecture.md`](docs/architecture.md) | Module map and data flow |
| [`docs/llm-integration.md`](docs/llm-integration.md) | Provider details |
| [`docs/modeling-and-eval.md`](docs/modeling-and-eval.md) | Classifier training and evaluation |
| [`docs/mitre-attribution.md`](docs/mitre-attribution.md) | MITRE ATT&CK license attribution |
| [`docs/release-notes.md`](docs/release-notes.md) | Versioned changelog |

Build the site locally:

```bash
mkdocs serve
```

---

## Contributing, license, security

- Contributions: see [`CONTRIBUTING.md`](CONTRIBUTING.md). PRs welcome.
- Security disclosures: see [`SECURITY.md`](SECURITY.md).
- License: Apache 2.0, see [`LICENSE`](LICENSE).
- MITRE ATT&CK marks and content used under MITRE's [terms of use](https://attack.mitre.org/resources/terms-of-use/); see [`docs/mitre-attribution.md`](docs/mitre-attribution.md).

---

## A note on scope

AlertSage is **research and demonstration software**. The classifier was trained on synthetic incident narratives. It maps to MITRE ATT&CK and produces SOC-style playbook hints, but it is not a substitute for production security tooling, threat intel feeds, or analyst judgment. Treat its output the way you would treat a junior analyst's first pass: a useful starting point that needs human review.
