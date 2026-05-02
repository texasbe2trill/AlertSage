# SOC Console Guide

AlertSage ships with a Streamlit-based SOC console designed to feel like a
production SIEM front-end (think Splunk Enterprise Security or Elastic
Security): dark-mode-first, dense, severity as the primary color signal.

## Launch

```bash
streamlit run app.py
```

The console opens at <http://localhost:8501>.

## Layout

The console is split into a sidebar and a main panel.

The sidebar carries:

- **Brand block** with the AlertSage wordmark and a `SOC console` tag.
- **Navigate** rail with six pages: Overview, Investigate, Hunt, Batch,
  Bookmarks, Settings.
- **Triage** controls: confidence threshold, probability rows, text
  preprocessing toggle, and the LLM second-opinion toggle.
- **Provider** snapshot (one line summarizing the active LLM backend and
  model).

The top bar above the main panel shows the brand again, a centered UTC
clock with the current page label, and a status strip on the right: a
green dot when the classifier is loaded and a separate pill for the
active LLM provider.

## Pages

### Overview

The default landing page. Reads as a SOC mission control screen with six
sections.

1. **KPI strip.** Six tiles: total analyzed, critical-or-high count, last
   24 hours, average classifier confidence, bookmarks, analyst notes.
   Each tile carries a left-edge severity bar and a small trend line.
2. **Events over time.** A 14-day stacked bar chart, one segment per
   severity tier (critical, high, medium, low, info).
3. **Classifier confidence histogram.** Twenty buckets across `[0, 1]`
   with green / yellow / orange / red shading so an analyst can see
   immediately whether the classifier is operating in its high-confidence
   band.
4. **Severity distribution.** Donut chart of the same data, grouped by
   tier.
5. **Threat intel feed.** Right-rail panel listing recent CISA, MS-ISAC,
   Mandiant, OTX, and MSRC entries. Currently a static demo feed; the
   data shape mirrors a TAXII collection so a real feed can drop in.
6. **Live tail.** Auto-refreshing event stream that re-queries the
   database every 5 seconds via `st.fragment`. New triage events appear
   without a page reload.
7. **MITRE ATT&CK coverage.** A tactic-by-technique heatmap built from
   the labeled history. Cells scale from translucent indigo (low) to
   saturated red (high).
8. **Top classifications.** Distribution rows colored by the severity
   tier the label maps to.
9. **Recent events table.** Latest ten triage runs with mono ID column,
   classification pill, status pill, confidence, anomaly score, and a
   truncated narrative.

### Investigate

Triage one incident end to end. The page is organized into:

- **Examples** column. One-click sample narratives (phishing, data
  exfiltration, malware, access abuse, web attack, benign activity).
- **Narrative** textarea. Free-form text; the classifier preprocesses
  internally.
- **Triage** button. Runs the TF-IDF + LogReg classifier and, when
  enabled, the LLM second opinion.

After triage the result panel renders top to bottom:

- **Event head.** Mono event id (`AS-000123`), classifier and LLM timing
  badges, severity pill, confidence pill, anomaly score, uncertainty
  badge.
- **Case status stepper.** Four-stage workflow: New, Triaging, Contained,
  Closed. Click the next-stage button to advance; status persists across
  reloads via the application database.
- **MITRE ATT&CK kill chain.** Horizontal flow of all 13 enterprise
  tactics. Tactics whose techniques the classifier surfaced light up in
  indigo with the matching technique IDs as monospace chips.
- **Indicators and enrichment.** Auto-extracted IOCs (IPv4, IPv6, MD5,
  SHA1, SHA256, URL, email, domain, CVE, hostname). Each indicator gets a
  mocked verdict (`clean` / `unknown` / `suspicious` / `malicious`),
  reputation score, first-seen estimate, and source list. The shape
  matches an aggregated VirusTotal / AbuseIPDB / OTX response.
- **Class probabilities.** Top N candidate classes (set by the sidebar
  slider) with severity-colored progress bars.
- **MITRE techniques.** Mapped technique IDs as monospace chips.
- **Analyst rationale.** LLM-authored or rule-based fallback narrative
  with severity-toned left border.
- **Playbook hint.** Recommended queue, priority, and a checkbox-style
  action list.

Footer actions: bookmark, add note, re-run.

### Hunt

Full-text search across triage history with filters:

- Free-text query against the narrative.
- Classification multiselect.
- Severity multiselect.
- Minimum confidence slider.
- Minimum anomaly-score slider.
- Time window selector (last hour, 24 hours, 7 days, 30 days, all time).

Results render as the same SOC table used on Overview. Up to 200 matches
are shown per render.

### Batch

CSV upload that triages many events at once. The console auto-detects the
text column from a fixed list (`incident_text`, `description`,
`narrative`, `alert`, `text`). Up to 500 rows per batch.

After processing:

- **KPI strip.** Processed count, critical-and-high count, medium count,
  benign-or-unknown count, plus wall-clock elapsed time.
- **Distribution panel.** Per-label event counts with severity bars.
- **MITRE coverage.** Tactic-by-tactic coverage bars, plus a detailed
  expander showing every (tactic, technique, label) cell with severity
  breakdown.
- **Exports.** Three download buttons: triage results CSV,
  MITRE coverage CSV, tactic rollup CSV (executive summary, one row per
  tactic).

### Bookmarks

Saved investigations. Each entry expands to show the severity pill, the
current case status, the narrative quote (severity-toned), the four-step
case status stepper with advance buttons, and any analyst note.

### Settings

LLM provider configuration plus triage defaults.

- **Provider** radio: Local llama.cpp (only shown when the
  `llama-cpp-python` package and a `.gguf` file in `models/` are both
  present), Hugging Face Inference, OpenAI, Anthropic.
- **Per-provider configuration** panel: model id text input and a
  password-masked Bring Your Own Key field. **Keys live in session
  state only and are never written to disk.**
- **Triage defaults** sliders: confidence threshold, probability rows,
  text-preprocessing toggle.

If you select OpenAI or Anthropic but leave the key blank, the
dispatcher silently falls back to the Hugging Face demo (when an
`HF_TOKEN` is available in environment or `.streamlit/secrets.toml`),
so first-time visitors still get a working triage. The status caption
in the sidebar shows the actually-active backend.

## Provider configuration

Three ways to set credentials, in order of precedence:

1. Sidebar / Settings BYOK fields (session only, in-memory).
2. `~/.streamlit/secrets.toml` or `<repo>/.streamlit/secrets.toml`
   with keys `HF_TOKEN`, `HF_MODEL`.
3. Environment variables: `TRIAGE_HF_TOKEN`, `TRIAGE_HF_MODEL`,
   `HF_TOKEN`, `HF_MODEL`.

OpenAI and Anthropic keys are session-only by design.

## Privacy

API keys are never persisted to `data/triage.db` or anywhere else on
disk. The Settings panel "Apply" action writes to `st.session_state`
keys only. The local TOML loader bypasses `st.secrets` so a missing
secrets file does not surface a UI toast on every read.

## Data lifecycle

Every triage run writes a row to `data/triage.db` with the narrative,
final label, max probability, uncertainty band, and a JSON blob of the
top-N probabilities, MITRE techniques, optional LLM opinion, and timing
breakdown. Bookmarks, notes, and case status are stored in the same
database via `db.save_setting` (case status is keyed by analysis id).

The database file is gitignored. Delete it any time to reset the
console; the schema rebuilds on next launch.

## Customizing

All styling lives in `assets/styles.css` and is loaded once at module
import. Tokens at the top of the file (`--soc-bg`, `--soc-accent`,
severity tiers) drive every panel; change one variable to retheme the
whole console. The Python side (`app.py`) is the router and view
layer; primitive helpers like `render_kpi`, `severity_pill`,
`render_case_stepper`, and `render_kill_chain` are reusable across
pages.
