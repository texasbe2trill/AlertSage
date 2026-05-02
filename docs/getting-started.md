# Getting started

Install AlertSage, launch the SOC console, and run your first triage.

---

## Prerequisites

- Python 3.12 (pinned via `runtime.txt`)
- Git
- A virtual environment tool such as `venv`

---

## 1. Clone the repository

```bash
git clone https://github.com/texasbe2trill/AlertSage.git
cd AlertSage
```

---

## 2. Create and activate a virtual environment

```bash
python3.12 -m venv .venv

source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate            # Windows PowerShell
```

---

## 3. Install dependencies

For just running the SOC console:

```bash
pip install -r requirements.txt
```

For development (tests, notebooks, mkdocs, the optional `llama-cpp-python`):

```bash
pip install -r requirements-dev.txt
pip install -e ".[dev]"
```

The editable install exposes the `nlp-triage` CLI.

---

## 4. Launch the SOC console

```bash
streamlit run app.py
```

Opens at <http://localhost:8501>. You should see:

- A dark-mode mission control dashboard
- KPI strip across the top
- Empty events-over-time chart (until you triage something)
- Threat intel feed panel on the right
- Empty live tail

The first launch loads the classifier and embedder into memory; subsequent launches are faster because of `@st.cache_resource`.

---

## 5. Triage your first incident

In the sidebar, click **Investigate**. Paste an incident or click one of the example buttons:

- Phishing
- Data exfiltration
- Malware
- Access abuse
- Web attack
- Benign activity

Hit **Triage**. Within a second or two you'll see:

1. The event head card with severity pill, confidence, anomaly score, and uncertainty.
2. A four-stage **case status stepper** (New, Triaging, Contained, Closed).
3. The MITRE ATT&CK **kill chain visualization**.
4. The **Indicators and enrichment** panel with auto-extracted IOCs.
5. The **case timeline** with the creation event and (when LLM is enabled) the rationale event.
6. Class probabilities, MITRE techniques, analyst rationale, playbook hint.

If the LLM second opinion is enabled in the sidebar, the rationale comes from your configured provider. Without keys, you get a deterministic rule-based rationale; the rest of the pipeline still works.

---

## 6. (Optional) Populate the dashboard with demo data

Without history, the Overview charts and panels are mostly empty. Three ways to populate:

### A. Manual triage

Run a few incidents through Investigate. Each one writes to `data/triage.db` and the Overview panels auto-refresh within ~6 seconds.

### B. Demo data generator

Click **Settings -> Demo data generator -> Run demo generator**. Synthetic events stream every six seconds. Click **Emit one now** for an immediate one. Click **Clear demo events** when you're done.

### C. CSV batch upload

Click **Batch**, drop in a CSV with an `incident_text` (or `description` / `narrative` / `alert` / `text`) column, hit **Run batch**. Up to 500 rows.

---

## 7. (Optional) Configure an LLM provider

The hosted demo uses Hugging Face Inference. For local runs, four providers are supported:

| Provider | How to set up |
|---|---|
| Hugging Face | Set `HF_TOKEN` env var, or paste in Settings -> LLM provider -> Hugging Face |
| OpenAI | Paste key in Settings -> LLM provider -> OpenAI |
| Anthropic | Paste key in Settings -> LLM provider -> Anthropic |
| Local llama.cpp | `pip install llama-cpp-python` and place a `.gguf` file in `models/` |

See [LLM integration](llm-integration.md) for the full provider matrix and fallback chain.

---

## 8. (Optional) Configure threat intel enrichment

The IOC panel on Investigate uses mocked enrichment by default. Paste a free-tier VirusTotal API key in **Settings -> Threat intel enrichment** to switch to live VT lookups (cached for 15 minutes per indicator).

---

## 9. Run tests

```bash
pytest tests/ -v
```

Currently nine tests across CLI, model artifacts, and preprocessing. They do not require any LLM provider or the GGUF model.

For coverage:

```bash
pytest tests/ --cov=src/triage --cov-report=term-missing
```

---

## 10. Try the CLI

```bash
# Single classification
nlp-triage --text "User reported suspicious email with attachment"

# JSON for scripting
nlp-triage --text "..." --json

# Bulk processing with LLM second opinion
nlp-triage --bulk incidents.csv --use-llm --difficulty soc-medium
```

See [CLI usage](cli.md) for the full flag set.

---

## Where to next

- [SOC console walkthrough](ui-guide.md) for a deeper tour of every page.
- [LLM integration](llm-integration.md) for provider details and the BYOK model.
- [Configuration](configuration.md) for environment variables, secrets paths, and the demo generator.
- [Architecture](architecture.md) for the module map and end-to-end data flow.
- [Notebooks](notebooks.md) to retrace the modeling workflow in Jupyter.
