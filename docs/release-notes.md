

# AlertSage v3.1.0 - SOC console rewrite

The Streamlit UI was rewritten end-to-end to look and behave like a
production SOC console (Splunk Enterprise Security, Elastic Security).
The functional surface (classifier, MITRE mapping, LLM second opinion,
database) is unchanged; everything you see is new.

## What changed

**Layout.** Dark-mode-first design system. Slate-900 backgrounds,
muted chrome, severity as the only loud color. JetBrains Mono for IDs,
timestamps, IOCs; Inter for prose. All styling consolidated into
`assets/styles.css` and loaded once at module import.

**Six pages.** Overview (mission control), Investigate (single-incident
triage), Hunt (search with anomaly + time-window filters), Batch (CSV
ingest with MITRE coverage report), Bookmarks (saved investigations
with case status), Settings (provider configuration).

**New components.**

- MITRE ATT&CK kill chain visualization on Investigate (13-tactic
  horizontal flow, lit-up cells for matched techniques).
- IOC extraction and enrichment panel on Investigate (IPv4/IPv6, MD5/
  SHA1/SHA256, URL, email, domain, CVE, hostname; mocked verdict +
  reputation in the shape of a real TI lookup).
- MITRE ATT&CK heatmap on Overview (tactic by technique density).
- Auto-refreshing live tail panel via `st.fragment(run_every="5s")`.
- Threat intel feed panel (demo entries; shape matches a TAXII feed).
- Confidence histogram on Overview.
- Anomaly score column on Hunt and Overview events tables.
- Case status workflow (New / Triaging / Contained / Closed) persisted
  via `db.save_setting`. Surfaced on Investigate, Bookmarks, and the
  events tables.
- MITRE coverage report on Batch (tactic-by-technique aggregation, with
  three CSV exports: triage results, MITRE coverage, tactic rollup).

**Bring Your Own Key.** OpenAI, Anthropic, and Hugging Face provider
panels accept a session-only API key in Settings. Keys are never written
to `data/triage.db` or any other on-disk file. Without a key, calls
fall back to the Hugging Face demo (when an `HF_TOKEN` is configured)
or to the local llama.cpp runtime.

**Capability gating.** The Local (GGUF) provider is hidden when
`llama-cpp-python` is missing or no `.gguf` file is present in
`models/`. Cloud deployments don't see a dead option.

**Per-provider rate limits.** Sliding-window limiter is now per
provider so burning OpenAI quota does not lock out Anthropic or HF.

**Performance.** The 789 MB joblib metrics file is no longer loaded on
every dashboard render; the homepage gates on file existence and shows
a tasteful inline note when missing. Heavy loaders (`vectorizer`,
`embedder`, `database`) are wrapped in `@st.cache_resource`.

## What was removed

- The previous purple-gradient "Premium" UI.
- Light/dark mode toggle (single dark theme).
- Experimental Lab tab.
- Inline `apply_theme_mode_css` runtime CSS injection.
- `ui_premium.py` (renamed to `app.py`).
- Stale UI screenshots (`docs/images/ui-*.png`) that depicted the old
  UI; new screenshots can be added when the SOC console settles.

## Breaking changes for downstream consumers

- Module rename: `ui_premium.py` is gone, use `app.py`.
- The old PREMIUM_CSS string is gone; use `assets/styles.css` for
  styling overrides.
- `THEME_OPTIONS` and dual-theme branches are removed.

---

# 🚀 NLP-Driven Incident Triage v0.2.0 Release Notes

This release delivers a major leap forward in realism, robustness, and usability.  
With enriched MITRE ATT&CK® narratives, an upgraded CLI, batch processing, improved documentation, and enhanced testing, the project now behaves much closer to a lightweight NLP SOC analyst assistant.

---

## 🔥 Major Enhancements

### 🧠 MITRE ATT&CK® Narrative Enrichment
- Incident generator now embeds realistic MITRE techniques across all event types:
  - Phishing → T1566 (various subtypes)
  - Malware → T1486, T1059 (PowerShell), etc.
  - Access Abuse → T1078, T1110
  - Web Attack → T1190, T1110
  - Policy Violations → mapped where relevant
- Added `mitre_clause` generation per event.
- Documentation updated with required MITRE license attribution.

---

## 💻 CLI Upgrades

### ✨ Rich UI & Banner
- New ASCII NLPTriage banner on start.
- Colorized output, aligned columns, and better readability.
- Uses `rich` for tables, highlighting, and labeling.

### 🤖 Difficulty Modes (Uncertainty Handling)
New flag:
```
--difficulty {default, soc-medium, soc-hard}
```
- Adjusts the strictness for marking predictions as `uncertain`.
- `soc-hard` simulates cautious SOC analyst behavior.

### 📂 Bulk Mode (New!)
New flags:
```
--input-file incidents.txt
--output-file results.jsonl
```
- Supports batch-classifying hundreds of incidents.
- Writes results as JSONL.
- Includes an **automated summary**:
  - event-type distribution
  - uncertainty rate
  - MITRE technique counts (from generator)
  - suggested analyst review priorities

### 🎯 Prediction Enhancements
- Cleaner uncertainty threshold logic.
- Better sorting of probabilities.
- Improved preprocessing alignment between training and inference.

---

## 🧱 Data & Modeling Improvements
- More realistic SOC narratives with ATT&CK technique references.
- Expanded variation across event types.
- Added ambiguous real-world-like descriptions for robustness.
- Updated dataset to align with generator improvements.

---

## 📘 Documentation & Website (MkDocs)
- All docs updated to reflect new CLI, features, and MITRE attribution.
- New or updated pages:
  - CLI Usage
  - Modeling & Evaluation
  - Getting Started
  - Limitations + MITRE License
  - Realistic Model Behavior

---

## 🧪 Tests & CI
- Expanded pytest suite:
  - prediction structure tests
  - artifact loading tests
  - uncertainty logic tests
  - CLI helper tests
- Fixed issues with test imports and artifact loading.
- GitHub CI workflow updated to validate on PRs.

---

## 📦 Packaging & Structure
- Project supports:
  - `pip install -e .`
  - `nlp-triage` console entry point
- Improved `pyproject.toml`, `README.md`, and MkDocs structure.

---

## 🛠️ Bug Fixes
- Fixed issues related to path imports in CLI.
- Resolved LFS model load errors.
- Fixed probability length assumptions in tests.
- Corrected documentation sync issues.

---

## 🏁 Summary
**v0.2.0** transforms the project from a baseline demo into a far more realistic SOC triage assistant.  
With MITRE integration, batch mode, enhanced CLI, and polished documentation, the project is now ready for broader use, portfolio presentation, and future extensions.

---

## 🏷️ Upgrade Instructions
To install or upgrade locally:

```bash
pip install -e .
```

If you're using editable mode and updated the CLI, reinstall:

```bash
pip install -e . --force-reinstall
```

---

## 📎 MITRE ATT&CK® Notice
This project includes derived technique names and references from the  
MITRE ATT&CK® framework.  
ATT&CK® is licensed under CC BY-NC-SA 4.0.  
See: https://attack.mitre.org/resources/terms-of-use/