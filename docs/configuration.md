# Configuration

Everything you can tune, where to put it, and what takes precedence.

## Configuration sources

AlertSage reads configuration from three sources, in order of precedence:

1. **Sidebar / Settings** in the SOC console (highest, session only)
2. **`.streamlit/secrets.toml`** in `~/` or in the repo root
3. **Environment variables** (lowest, persistent)

API keys for OpenAI and Anthropic are session-only **by design** and only ever read from source 1. Hugging Face and VirusTotal can come from any of the three.

---

## LLM provider configuration

### Hugging Face Inference

```bash
export TRIAGE_HF_TOKEN="hf_..."
export TRIAGE_HF_MODEL="meta-llama/Llama-3.1-8B-Instruct:cerebras"
# or
export HF_TOKEN="hf_..."
export HF_MODEL="meta-llama/Llama-3.1-8B-Instruct:cerebras"
```

Or via `secrets.toml`:

```toml
HF_TOKEN = "hf_..."
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
```

Or paste in Settings -> LLM provider.

### OpenAI

Settings -> LLM provider -> OpenAI -> Bring my own key.

```python
# Default model id is gpt-4o-mini, override in Settings
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
```

Recommended models: `gpt-4o-mini` (default, cheap), `gpt-4o`, `gpt-4.1-mini`.

### Anthropic

Settings -> LLM provider -> Anthropic -> Bring my own key.

```python
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
```

Recommended models: `claude-haiku-4-5` (default), `claude-sonnet-4-6`, `claude-opus-4-7`.

### Local llama.cpp

```bash
# Path to GGUF model file
export TRIAGE_LLM_MODEL=/absolute/path/to/your-model.gguf

# Context window
export TRIAGE_LLM_CTX=8192

# Max tokens for generation
export TRIAGE_LLM_MAX_TOKENS=1024

# Temperature
export TRIAGE_LLM_TEMP=0.1
```

The Local provider option in the sidebar is **automatically hidden** when the `llama_cpp` Python package is missing or no `.gguf` file is present in `models/`. The dispatcher then falls back to Hugging Face. You don't need to do anything to disable Local on hosts that can't run it.

### LLM debugging

```bash
export TRIAGE_LLM_DEBUG=1
streamlit run app.py
```

Verbose LLM logs go to stderr (CLI JSON on stdout stays clean).

---

## Threat intel enrichment

### VirusTotal

```bash
export VIRUSTOTAL_API_KEY="..."
# or
export VT_API_KEY="..."
```

Or paste in Settings -> Threat intel enrichment.

When a key is present, the **Investigate -> Indicators panel** switches from the deterministic mock to live VT lookups (cached for 15 minutes per indicator). Without a key, the panel shows mock verdicts so the demo works out of the box.

---

## Triage knobs

Configurable from the sidebar in the console, and from CLI flags:

### Confidence threshold

Below this, the classifier returns `uncertain`.

- Sidebar slider: `0.0` to `1.0`, step `0.05`
- CLI: `nlp-triage --threshold 0.70 "..."`
- Default: `0.50`

### Probability rows

How many candidate classes to surface in the result panel.

- Sidebar slider: `1` to `10`
- Default: `5`

### Text preprocessing

Toggle whether `clean_description()` runs before TF-IDF. Off when you want the raw narrative to flow through (rare).

- Sidebar checkbox
- Default: on

### LLM second opinion

Toggle whether the LLM is called. Off, the classifier and rule-based fallback rationale still produce a result.

- Sidebar checkbox
- Default: on (in the SOC console)

### Difficulty modes (CLI only)

- `default`: standard uncertainty handling (`threshold = 0.50`, top 5)
- `soc-medium`: moderate strictness (`threshold = 0.60`, top 5)
- `soc-hard`: maximum strictness (`threshold = 0.75`, top 3)

```bash
nlp-triage --difficulty soc-hard "..."
```

---

## Demo data generator

### Toggle

Settings -> Demo data generator -> Run demo generator.

When on, an `st.fragment(run_every="6s")` emits one synthetic incident every six seconds drawn from the curated example set, runs it through the classifier, persists it to `data/triage.db` with `batch_id = "demo"`, and seeds the case timeline.

### Manual emission

Settings -> Demo data generator -> Emit one now.

Bypasses the timer for an immediate test. Useful when validating the pipeline (predict + persist) without waiting for the next tick.

### Cleanup

Settings -> Demo data generator -> Clear demo events.

Deletes rows where `batch_id = "demo"` from `analysis_history`. Bookmarks, notes, and case status are not affected.

---

## Database

### Default path

`data/triage.db` at the repo root.

### Override

```bash
# Currently the path is hard-coded in src/triage/database.py
# Override by passing db_path to TriageDatabase() at construction time.
```

The schema rebuilds on first launch; deleting `data/triage.db` is a clean reset. The file is gitignored.

---

## UI defaults

### Streamlit page config

Set in `app.py` at module load:

```python
st.set_page_config(
    page_title="AlertSage SOC",
    page_icon=_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
```

### Theme

Single dark theme. All styling lives in `assets/styles.css`. To retheme, edit the CSS custom properties on `:root` (`--soc-bg`, `--soc-accent`, severity tiers).

```css
:root {
    --soc-bg: #0a0e1a;
    --soc-accent: #3b82f6;
    /* ... */
}
```

### Auto-refresh cadences

Set per-fragment in `app.py`:

| Fragment | Default |
|---|---|
| Live tail | 5 s |
| KPI strip | 6 s |
| Charts row (events-over-time, histogram, donut) | 6 s |
| MITRE heatmap | 8 s |
| Top classifications | 8 s |
| Recent events table | 6 s |
| Demo data generator | 6 s |

History is cached for 4 seconds across all fragments so a refresh cycle hits the database at most once.

---

## Dataset generation

```bash
python generator/generate_cyber_incidents.py \
    --n-events 50000 \
    --chunk-size 1000 \
    --start-date 2024-01-01 \
    --end-date 2024-12-31 \
    --use-llm \
    --rewrite-report audit.json
```

See [Dataset generation](data-and-generator.md) and [Production generation](production-generation.md) for the full set of flags.

---

## Streamlit deployment

### Streamlit Community Cloud

The hosted demo at <https://alertsage.streamlit.app/> reads:

- `HF_TOKEN`, `HF_MODEL` from Streamlit Cloud secrets
- `runtime.txt` pinning Python 3.12
- `requirements.txt` (slim, runtime only)

The Local provider is automatically hidden because `llama-cpp-python` is not in `requirements.txt` and there's no `.gguf` file deployed.

### Self-hosting

Any host that can run Python 3.12 + Streamlit. The console expects:

- `app.py` and `assets/styles.css` co-located
- `models/vectorizer.joblib` and `models/baseline_logreg.joblib` (or `enhanced_logreg.joblib`) present
- A writable `data/` directory for SQLite
- Optional: `models/*.gguf` for the Local provider
- Optional: HF / OpenAI / Anthropic / VirusTotal credentials via secrets or env

Memory: enhanced model + sentence-transformer embedder needs about 1 GB. With local llama.cpp, add the model size (about 6 GB for Llama 3.1 8B Q6_K).
