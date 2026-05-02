# LLM integration

AlertSage routes the LLM second opinion through whichever provider you select. Four providers are supported out of the box:

| Provider | Default model | When to use |
|---|---|---|
| **Hugging Face Inference** | `meta-llama/Llama-3.1-8B-Instruct:cerebras` | Quick demo, free-tier or paid HF account, hosted demo on Streamlit Cloud. |
| **OpenAI** | `gpt-4o-mini` | Best rationale quality on commodity hardware, low cost. |
| **Anthropic** | `claude-haiku-4-5` | Best rationale quality with longer context. |
| **Local llama.cpp** | local `.gguf` file | Air-gapped deployments, no network egress. |

All providers implement the same `generate_json(prompt) -> dict` interface so the dispatcher in `llm_helpers.llm_second_opinion()` is provider-agnostic. Adding a fifth provider is a small, isolated patch in `src/triage/llm_client.py`.

## Bring Your Own Key

!!! info "Keys never touch disk"
    API keys live in `st.session_state` only. They are never written to `data/triage.db` and never persisted to any other file. Closing the browser tab clears them.

The console exposes a Bring Your Own Key panel for each remote provider in **Settings**. Paste a key once, it stays for the rest of the session. The same provider also accepts the key from `~/.streamlit/secrets.toml` or environment variables; precedence is sidebar/Settings > secrets.toml > env vars.

### Where keys are read

| Provider | Sidebar/Settings | Secrets file key | Env var |
|---|---|---|---|
| Hugging Face | yes (token) | `HF_TOKEN`, `HF_MODEL` | `TRIAGE_HF_TOKEN`, `HF_TOKEN`, `TRIAGE_HF_MODEL`, `HF_MODEL` |
| OpenAI | yes (key) | not read | session-only |
| Anthropic | yes (key) | not read | session-only |
| Local | n/a | n/a | `TRIAGE_LLM_MODEL` (path to .gguf) |
| VirusTotal | yes (key) | `VIRUSTOTAL_API_KEY`, `VT_API_KEY` | `VIRUSTOTAL_API_KEY`, `VT_API_KEY` |

OpenAI and Anthropic keys are **session only by design**. If you want to persist them for a deployment, export them in your shell or pass via your hosting platform's secrets mechanism.

## Fallback chain

The dispatcher (`_build_llm_kwargs` in `app.py`) routes intelligently when keys are missing or providers are unavailable:

```text
Pick OpenAI:
  has key?      -> OpenAI
  no key, hf    -> Hugging Face Router (if HF_TOKEN)
  no key, local -> local llama.cpp (if GGUF available)

Pick Anthropic:
  has key?      -> Anthropic
  no key, hf    -> Hugging Face Router (if HF_TOKEN)
  no key, local -> local llama.cpp (if GGUF available)

Pick Local:
  GGUF + llama_cpp present? -> local llama.cpp
  otherwise                  -> Hugging Face Router
```

The sidebar caption shows the actually-active backend after fallbacks resolve.

## Per-provider rate limits

Each provider has its own sliding-window limiter, default 5 requests per 60 seconds. Burning OpenAI quota does not lock out Anthropic or Hugging Face. The window and request count come from `RATE_LIMIT_REQS` and `RATE_LIMIT_WINDOW_S` in `app.py`.

When the limit is hit, the UI surfaces a friendly "wait Ns" message instead of a stack trace.

## Hugging Face Inference

Default provider for the hosted demo. Uses the HF Router REST endpoint (`https://router.huggingface.co/v1/chat/completions`), so you can target any model the router supports including provider-suffixed variants like `meta-llama/Llama-3.1-8B-Instruct:cerebras`.

```bash
# Read once and cache
export HF_TOKEN="hf_..."

# Or via secrets.toml
cat > .streamlit/secrets.toml <<EOF
HF_TOKEN = "hf_..."
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
EOF
```

Token has Inference scope (free-tier works for low-volume demos).

## OpenAI

Uses the official `openai` Python SDK with the Chat Completions endpoint and `response_format={"type": "json_object"}` to enforce JSON output.

```python
# What the dispatcher calls under the hood
client = OpenAIClient(
    api_key="sk-...",
    model="gpt-4o-mini",
    max_new_tokens=512,
    rate_limiter=...,
)
response = client.generate_json(prompt)  # returns dict
```

Recommended models:

- `gpt-4o-mini` (default): cheap, fast, good enough for SOC rationale
- `gpt-4o`: better quality, more expensive
- `gpt-4.1-mini`: middle ground

The system prompt (`SOC_SYSTEM_PROMPT` in `llm_client.py`) instructs the model to return strict JSON with `label`, `mitre_ids`, and `rationale`.

## Anthropic

Uses the official `anthropic` Python SDK with the Messages API. The system prompt is set via `system=`, the user prompt via `messages=[{"role": "user", "content": ...}]`.

Recommended models:

- `claude-haiku-4-5` (default): cheap, fast, longer context than gpt-4o-mini
- `claude-sonnet-4-6`: higher quality
- `claude-opus-4-7`: highest quality, most expensive

Anthropic responses come back as a list of content blocks; the client extracts `block.text` for `block.type == "text"` and concatenates.

## Local llama.cpp

For air-gapped or zero-egress deployments. The Local (GGUF) provider option in the sidebar is **automatically hidden** when its prerequisites are missing:

1. The `llama_cpp` Python package must be importable.
2. A `.gguf` file must be present in `models/`.

When either check fails, the option does not appear and the dispatcher's fallback routes to Hugging Face instead.

### Install llama-cpp-python

```bash
# Apple Silicon (Metal acceleration)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# CUDA
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python

# CPU only
pip install llama-cpp-python
```

Or use the dev extras: `pip install -e ".[dev]"` includes a pinned `llama-cpp-python==0.3.16`.

### Download a GGUF model

```bash
mkdir -p models

# Llama 3.1 8B Instruct, Q6_K (about 6 GB, recommended)
huggingface-cli download TheBloke/Llama-3.1-8B-Instruct-GGUF \
  Llama-3.1-8B-Instruct-Q6_K.gguf --local-dir models

# Mistral 7B Instruct v0.2, Q6_K (about 5.5 GB)
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  mistral-7b-instruct-v0.2.Q6_K.gguf --local-dir models

# TinyLlama 1.1B, Q4_K_M (about 700 MB, CPU-friendly)
huggingface-cli download TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF \
  tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf --local-dir models
```

### Configure the path

Default path: `models/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf`. Override with `TRIAGE_LLM_MODEL`:

```bash
export TRIAGE_LLM_MODEL="/absolute/path/to/your-model.gguf"
```

GPU acceleration is auto-enabled at first call:

- `LLAMA_N_GPU_LAYERS=999` (offload all layers)
- `LLAMA_METAL=1` on macOS
- `LLAMA_CUDA=1` on NVIDIA

To force CPU-only, set these to `0` in your environment before launching.

## VirusTotal IOC enrichment

The IOC panel on **Investigate** auto-extracts indicators from the narrative (IPv4, IPv6, MD5, SHA1, SHA256, URL, email, domain, CVE, hostnames). Without a VirusTotal key it shows a **deterministic mock**: the same indicator always returns the same verdict, score, and source list, so the demo is reproducible.

Paste a VirusTotal API key in **Settings -> Threat intel enrichment** and the panel switches to **live VT lookups**. Free-tier keys work; results are cached for 15 minutes per indicator to stay under the 4 req/min free-tier limit.

The enrichment shape is provider-agnostic so swapping in AbuseIPDB, OTX, GreyNoise, or an aggregator is a one-function change. See `_vt_lookup` and `_enrich_ioc_real_or_mock` in `app.py`.

### External pivot links

Each IOC row also exposes click-through pivot links to:

| IOC type | Pivots |
|---|---|
| ipv4 / ipv6 | VirusTotal, AbuseIPDB, Shodan, GreyNoise |
| domain | VirusTotal, URLhaus, Censys |
| md5 / sha1 / sha256 | VirusTotal, MalwareBazaar |
| url | URLhaus, VirusTotal |
| cve | NVD, MITRE CVE |

These open in a new tab and require no API keys.

## Prompt engineering

The triage prompt is a single function (`llm_second_opinion` in `src/triage/llm_helpers.py`). Two key elements:

1. **Strict JSON output** with three keys: `label`, `mitre_ids`, `rationale`. The system prompt explicitly tells the model to refuse free-form prose and only emit the JSON object.
2. **Hallucination guardrails**: extracted IOCs from the narrative are compared against the model's rationale. If the model invents an IOC that wasn't in the source narrative, the label is downgraded to `uncertain` and the rationale is replaced with a deterministic fallback. See `_extract_indicators` in `llm_helpers.py`.

A second pass applies **keyword validation per label**. If the model returns `data_exfiltration` but the narrative contains no exfil-style keywords, the label downgrades to `uncertain`. This is a deliberate conservatism trade-off: better to mark uncertain than to confidently misclassify.

## Debugging

Enable verbose LLM logs by setting `TRIAGE_LLM_DEBUG=1` before launching:

```bash
TRIAGE_LLM_DEBUG=1 streamlit run app.py
```

Debug output goes to stderr (so JSON output on stdout from the CLI stays clean).

The Settings -> Demo data generator panel also surfaces the **last LLM error** if the demo fragment hits one. Use this to diagnose key issues, model mismatches, or rate-limit blowback without digging through logs.

## Cost notes

The LLM second opinion runs once per triage. With a 200-row Batch and OpenAI gpt-4o-mini at typical pricing, a full run is fractions of a cent. Anthropic Haiku is similar. Hugging Face free-tier covers small demos; paid is per-token.

For larger deployments, set the per-provider rate limit and watch the sidebar's rate-limit caption. Or skip the LLM entirely by leaving the **LLM second opinion** checkbox off in the sidebar; the classifier alone still produces a label, MITRE mapping, and a deterministic rule-based rationale.
