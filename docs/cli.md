# CLI Usage (NLPTriage)

The project ships with an **enhanced CLI** that behaves like a mini SOC assistant on the command line.

It supports:

- NLPTriage ASCII banner
- Text cleaning preview
- Uncertainty-aware predictions (`uncertain` fallback)
- Difficulty modes (`--difficulty default|soc-medium|soc-hard`)
- MITRE ATT&CK technique mapping panel
- Top‑k probability table (Rich table)
- Bulk file processing (`--file incidents.txt`)
- Batch summary report + recommendations
- Optional JSON output for automation
- Progress bar for all operations
- Interactive SOC‑style loop

!!! tip "Install first"
Make sure you’ve run:

```bash
pip install -e ".[dev]"

```

from the project root so the package and entry points are available.

---

## Basic single-shot usage

From the repo root:

```bash
nlp-triage "EDR detected powershell.exe spawning from Outlook on LAPTOP-093 and reaching out to 185.22.11.4 over port 443."
```

### Example output

![](images/cli_output_single.png)

## Interactive mode usage

```bash
nlp-triage
```

### Example output

![](images/cli_output.png)

## Bulk file processing

Provide a plain‑text file where each line is an incident description:

```bash
nlp-triage --file incidents.txt
```

At the end of processing, NLPTriage prints a **batch summary** including:
- Per‑class distribution
- Most frequent MITRE techniques observed
- Uncertain predictions count
- Recommendations (e.g., raise threshold, retrain, add rules)

## How it works

Each input is:
- **Cleaned** using training‑consistent normalization
- **Vectorized** using TF‑IDF model
- **Classified** with uncertainty handling and difficulty rules
- **Mapped to MITRE ATT&CK** via keyword + heuristic extraction
- **Displayed** using Rich panels for quick SOC triage

---

## Uncertainty & thresholds

- `predict_with_uncertainty` compares the **maximum class probability** against `--threshold` (default **0.50**).
- If the max probability is below the threshold, the CLI returns `final_label = "uncertain"` while still showing the `base_label`.
- A Rich panel labels each prediction as `low`, `medium`, or `high` certainty (color coded red/yellow/green) so analysts can gauge trust quickly.
- Use `--threshold 0.7` (or higher) when you only want confident predictions in automation workflows.

---

## Difficulty Modes

You can control how strict MITRE matching and uncertainty logic behaves:

```bash
nlp-triage --difficulty soc-medium "Suspicious PowerShell execution on host."
```

Modes:
- **default**: balanced mode (recommended)
- **soc-medium**: medium‑strict SOC mode (higher uncertainty marking)
- **soc-hard**: strict SOC mode (very conservative, many predictions become 'uncertain')

---

## JSON payload schema

`--json` skips all Rich formatting and prints a dict with the following shape:

```json
{
  "raw_text": "...",
  "cleaned": "...",
  "base_label": "phishing",
  "final_label": "phishing",
  "max_prob": 0.83,
  "threshold": 0.5,
  "uncertainty_level": "medium",
  "difficulty": "default",
  "mitre_techniques": ["T1566.002"],
  "probs_sorted": [...]
}
```

This payload is what `tests/test_cli.py` asserts against, so you can rely on the keys staying stable even if the formatting evolves.

---

## MITRE ATT&CK Mapping

The CLI enriches each prediction with a lightweight MITRE ATT&CK® mapping based on the predicted `event_type`.  
This mapping is **illustrative**, not exhaustive, and is meant to provide quick pivot points for further analysis.

| event_type        | ATT&CK technique IDs      |
|-------------------|---------------------------|
| `phishing`        | T1566                     |
| `malware`         | T1204, T1059, T1486      |
| `web_attack`      | T1190, T1110             |
| `access_abuse`    | T1078, T1110             |
| `data_exfiltration` | T1041, T1567           |
| `policy_violation`| T1052                    |
| `benign_activity` | _None (non-security / operational)_ |
| `uncertain`       | _None (requires analyst review)_    |

In the CLI:

- The **“Top Class Probabilities”** table shows a `MITRE Techniques` column populated from this mapping.
- JSON / JSONL output includes:
  - `probs_sorted[*].mitre_techniques`
  - `final_label_mitre_techniques`
so downstream tools can pivot directly to ATT&CK documentation.

---

## Running without installing entry points

If `pip install -e .` is not available (e.g., in a quick experiment), you can invoke the CLI module directly after exporting `PYTHONPATH`:

```bash
PYTHONPATH=src python -m triage.cli "Short incident description here."
```

Dependencies (`joblib`, `rich`, `scikit-learn`, etc.) still need to be available in the environment.

## Help menu

```bash
nlp-triage -h
usage: nlp-triage [-h] [--json] [--threshold THRESHOLD] [--max-classes MAX_CLASSES] [--difficulty {default,soc-medium,soc-hard}] [--input-file INPUT_FILE] [--output-file OUTPUT_FILE]
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
```
