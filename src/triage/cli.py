#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter

# Suppress tokenizers parallelism warning when using sentence-transformers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.status import Status

console = Console()

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_mitre_from_json() -> dict:
    """
    Load MITRE ATT&CK technique mappings from JSON file.
    Returns dict with technique IDs as keys and technique details as values.
    """
    try:
        mitre_json_path = PROJECT_ROOT / "data" / "mitre_techniques_snippets.json"
        if mitre_json_path.exists():
            with open(mitre_json_path, "r") as f:
                return json.load(f)
        return {}
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent  # repo root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.preprocess import clean_description  # type: ignore
from src.triage.embeddings import get_embedder  # type: ignore
from src.triage.llm_helpers import (  # type: ignore
    HF_DEFAULT_MODEL,
    HF_ENDPOINT,
    HF_RATE_LIMIT_MAX,
    HF_RATE_LIMIT_WINDOW,
    HF_TOKEN_ENV,
    LLM_CTX_SIZE,
    LLM_DEBUG,
    LLM_MAX_TOKENS,
    LLM_MODEL_PATH,
    LLM_TEMP,
    MITRE_MAPPING,
    _llm_debug,
    build_llm_rationale,
    get_llm,
    llm_second_opinion,
    soc_triage_hint,
)

# -----------------------------------------------------------------------------
# Globals
# -----------------------------------------------------------------------------
console = Console()
DEFAULT_UNCERTAINTY_THRESHOLD = 0.50

DIFFICULTY_MODES = {
    "default": {"threshold": DEFAULT_UNCERTAINTY_THRESHOLD, "max_classes": 5},
    "soc-medium": {"threshold": 0.60, "max_classes": 5},
    "soc-hard": {"threshold": 0.75, "max_classes": 3},
}

BANNER = r"""
      __  __    ___  _____      _                  
  ╱╲ ╲ ╲╱ ╱   ╱ _ ╲╱__   ╲_ __(_) __ _  __ _  ___ 
 ╱  ╲╱ ╱ ╱   ╱ ╱_)╱  ╱ ╱╲╱ '__│ │╱ _` │╱ _` │╱ _ ╲
╱ ╱╲  ╱ ╱___╱ ___╱  ╱ ╱  │ │  │ │ (_│ │ (_│ │  __╱
╲_╲ ╲╱╲____╱╲╱      ╲╱   │_│  │_│╲__,_│╲__, │╲___│
                                       │___╱      
NLP-Driven Cyber Incident Triage
"""



def print_llm_panel(result: dict) -> None:
    """
    Pretty-print the LLM second opinion as a Rich panel, with context
    from the baseline classifier (so the analyst can see how the
    second opinion compares to the original decision).
    """
    llm_result = result.get("llm_second_opinion") or {}
    if not llm_result:
        return

    label = llm_result.get("label", "uncertain")
    mitre_ids = llm_result.get("mitre_ids", [])
    rationale = llm_result.get("rationale", "")

    mitre_text = ", ".join(mitre_ids) if mitre_ids else "-"

    # Baseline model context
    base_label = result.get("final_label", result.get("base_label", "unknown"))
    max_prob = result.get("max_prob", None)

    # Relationship between baseline and LLM opinion
    if base_label == "uncertain" and label != "uncertain":
        relation = (
            f"Baseline model was [bold]uncertain[/bold]; "
            f"LLM suggests [bold]{label}[/bold] as a second opinion."
        )
    elif base_label != "uncertain" and label == base_label:
        relation = (
            f"LLM second opinion [bold]agrees[/bold] with baseline label "
            f"[bold]{base_label}[/bold]."
        )
    elif base_label != "uncertain" and label != base_label:
        relation = (
            f"LLM suggests an [bold]alternative[/bold] label: "
            f"baseline [bold]{base_label}[/bold] → LLM [bold]{label}[/bold]."
        )
    else:
        # both uncertain, or anything unexpected
        relation = (
            "Both baseline model and LLM remain [bold]uncertain[/bold]; "
            "treat this as an ambiguous signal and prioritize manual review."
        )

    prob_line = (
        f"[bold white]Baseline max probability:[/] {max_prob:.3f}\n"
        if isinstance(max_prob, (int, float))
        else ""
    )

    body = (
        f"[bold white]Baseline final label:[/] {base_label}\n"
        f"{prob_line}"
        f"[bold white]LLM suggested label:[/] {label}\n"
        f"[bold white]Suggested MITRE IDs:[/] {mitre_text}\n\n"
        f"[bold white]How this compares to baseline:[/]\n{relation}\n\n"
        f"[bold white]Rationale & Next Steps (LLM):[/]\n{rationale}"
    )

    console.print(
        Panel(
            body,
            title="LLM Assist (Second Opinion)",
            border_style="bright_magenta",
        )
    )


# -----------------------------------------------------------------------------
# Loading artifacts
# -----------------------------------------------------------------------------
def load_artifacts():
    vectorizer = joblib.load(PROJECT_ROOT / "models/vectorizer.joblib")
    clf = joblib.load(PROJECT_ROOT / "models/enhanced_logreg.joblib")
    embedder = get_embedder()
    classes = clf.classes_
    return vectorizer, clf, embedder, classes


# -----------------------------------------------------------------------------
# Uncertainty helpers
# -----------------------------------------------------------------------------
def categorize_uncertainty(max_prob: float, threshold: float) -> str:
    """
    Map max_prob into three coarse bands:
      - 'low'    -> below threshold
      - 'medium' -> between threshold and 0.80
      - 'high'   -> above 0.80
    """
    if max_prob < threshold:
        return "low"
    elif max_prob < 0.80:
        return "medium"
    return "high"


import time
from rich.progress import Progress, BarColumn, TextColumn


def show_progress_bar(duration: float = 0.4, length: int = 24) -> None:
    """
    Small animated progress bar for CLI polish, with color.
    duration: total animation time (seconds)
    length: number of characters in the bar
    """
    with Progress(
        TextColumn("[bold green]Running NLP classifier...[/bold green]"),
        BarColumn(
            bar_width=length, complete_style="bright_cyan", finished_style="bold green"
        ),
        transient=True,
    ) as progress:
        task = progress.add_task("nlp", total=100)
        start = time.time()
        while not progress.finished:
            elapsed = time.time() - start
            frac = min(1.0, elapsed / duration)
            progress.update(task, completed=int(frac * 100))
            if frac >= 1.0:
                break
            time.sleep(0.04)


def predict_with_uncertainty(
    text: str,
    vectorizer,
    clf,
    embedder,
    classes,
    threshold: float = DEFAULT_UNCERTAINTY_THRESHOLD,
    max_classes: int = 5,
):
    """
    Run a single prediction with:
      - text cleaning
      - TF–IDF vectorization
      - sentence embeddings
      - feature fusion (TF-IDF + embeddings)
      - max-probability classification
      - simple uncertainty handling
    """
    from scipy.sparse import hstack, csr_matrix

    cleaned = clean_description(text)

    # Get TF-IDF features
    X_tfidf = vectorizer.transform([cleaned])

    # Get sentence embeddings
    embedding = embedder.encode(cleaned, normalize=True)
    embedding_sparse = csr_matrix(embedding)

    # Combine features (TF-IDF + Embeddings)
    X_vec = hstack([X_tfidf, embedding_sparse])

    proba = clf.predict_proba(X_vec)[0]

    base_idx = int(np.argmax(proba))
    base_label = classes[base_idx]
    max_prob = float(proba[base_idx])

    final_label = base_label if max_prob >= threshold else "uncertain"
    uncertainty_level = categorize_uncertainty(max_prob, threshold)

    probs_sorted = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[
        :max_classes
    ]

    return {
        "raw_text": text,
        "cleaned": cleaned,
        "base_label": base_label,
        "final_label": final_label,
        "max_prob": max_prob,
        "threshold": threshold,
        "uncertainty_level": uncertainty_level,
        "probs_sorted": probs_sorted,
    }


# -----------------------------------------------------------------------------
# Pretty printing / JSON output
# -----------------------------------------------------------------------------
def prob_color(prob: float) -> str:
    """
    Color ramp for probabilities when shown in the table.
    """
    if prob >= 0.90:
        return "bold bright_green"
    if prob >= 0.75:
        return "green"
    if prob >= 0.50:
        return "yellow"
    if prob >= 0.25:
        return "dark_orange"
    return "dim"



def build_analyst_note(result: dict, triage: dict) -> str:
    """
    Build a short, ticket-ready analyst note summarizing the model decision
    and suggested handling.
    """
    final_label = result["final_label"]
    max_prob = result["max_prob"]
    uncertainty = result["uncertainty_level"]
    queue = triage["queue"]

    if final_label == "uncertain":
        return (
            "Model could not confidently assign a specific event_type. "
            "Treat this as an ambiguous signal: gather additional telemetry, "
            "review user reports, and route to the triage queue for manual review."
        )

    return (
        f"Model assessed this narrative as '{final_label}' "
        f"with max class probability {max_prob:.3f} "
        f"and '{uncertainty}' confidence. "
        f"Suggested routing: {queue}. Use this as a decision-support signal, "
        f"not an automated decision, and validate with additional context "
        f"(EDR, proxy, auth logs, and user history) before taking action."
    )


def print_pretty(result: dict) -> None:
    console.rule("[bold cyan]Incident Triage Result")

    # Panel color by uncertainty band
    panel_color = {
        "high": "green",
        "medium": "yellow",
        "low": "red",
    }.get(result["uncertainty_level"], "white")

    # Prediction summary panel
    summary_text = (
        f"[bold white]Base label:[/] {result['base_label']}\n"
        f"[bold white]Final label:[/] "
        f"[{panel_color}]{result['final_label']}[/{panel_color}]\n"
        f"[bold white]Max probability:[/] {result['max_prob']:.3f} "
        f"(threshold={result['threshold']:.2f})\n"
        f"[bold white]Uncertainty level:[/] {result['uncertainty_level']}"
    )

    console.print(
        Panel.fit(
            summary_text,
            title="Classification",
            border_style=panel_color,
        )
    )

    # Cleaned text panel
    console.print(
        Panel(
            f"[white]{result['cleaned']}[/white]",
            title="Cleaned Text",
            border_style="dim",
        )
    )

    # Probabilities table
    table = Table(title="Top Class Probabilities")
    table.add_column("Class", style="cyan", no_wrap=True)
    table.add_column("Probability", style="magenta")
    table.add_column("MITRE Techniques", style="dim")

    base_label = result["base_label"]

    for cls, p in result["probs_sorted"]:
        style = prob_color(p)
        if cls == base_label:
            cls_display = f"[bold]{cls}[/bold]"
        else:
            cls_display = cls
        mitre_ids = ", ".join(MITRE_MAPPING.get(cls, [])) or "-"
        table.add_row(cls_display, f"[{style}]{p:.3f}[/{style}]", mitre_ids)

    console.print(table)

    # SOC-style triage hint
    triage = soc_triage_hint(result["final_label"], result["uncertainty_level"])
    actions_bullets = "\n".join(f"- {a}" for a in triage["actions"])
    triage_text = (
        f"[bold white]Suggested queue:[/] {triage['queue']}\n"
        f"[bold white]Suggested priority:[/] {triage['priority']}\n\n"
        f"[bold white]First actions:[/]\n{actions_bullets}"
    )

    console.print(
        Panel(
            triage_text,
            title="SOC Triage Hint",
            border_style="blue",
        )
    )

    # Analyst-facing note suitable for tickets or handoff
    analyst_note = build_analyst_note(result, triage)
    console.print(
        Panel(
            analyst_note,
            title="Analyst Note",
            border_style="magenta",
        )
    )

    console.rule()


def print_json(result: dict) -> None:
    # Make JSON-serializable
    json_ready = {k: v for k, v in result.items() if k != "probs_sorted"}
    json_ready["probs_sorted"] = [
        {
            "class": cls,
            "probability": float(p),
            "mitre_techniques": MITRE_MAPPING.get(cls, []),
        }
        for cls, p in result["probs_sorted"]
    ]
    json_ready["final_label_mitre_techniques"] = MITRE_MAPPING.get(
        result["final_label"], []
    )
    console.print_json(json.dumps(json_ready))


# -----------------------------------------------------------------------------
# Bulk results summary and recommendations
# -----------------------------------------------------------------------------
def summarize_bulk_results(results: list[dict]) -> None:
    """
    Print a data-enriched overview of bulk predictions with high-level
    recommendations for SOC-style review.
    """
    if not results:
        return

    total = len(results)
    label_counts = Counter(r["final_label"] for r in results)
    base_counts = Counter(r["base_label"] for r in results)

    uncertain_count = label_counts.get("uncertain", 0)
    certain_count = total - uncertain_count
    uncertain_ratio = uncertain_count / total

    avg_max_prob = sum(r["max_prob"] for r in results) / total
    llm_count = sum(1 for r in results if r.get("llm_second_opinion"))

    # Summary table of final labels
    table = Table(title="Bulk Triage Summary")
    table.add_column("Final Label", style="cyan", no_wrap=True)
    table.add_column("Count", justify="right")
    table.add_column("Percent", justify="right")

    for label, count in label_counts.most_common():
        pct = 100.0 * count / total
        table.add_row(label, str(count), f"{pct:.1f}%")

    console.print()
    console.print(table)

    # Quick MITRE coverage summary (based on base labels)
    mitre_set = set()
    for lbl, count in base_counts.items():
        for tech in MITRE_MAPPING.get(lbl, []):
            mitre_set.add(tech)

    mitre_text = ", ".join(sorted(mitre_set)) if mitre_set else "None"

    # Build recommendations text
    rec_records = [
        f"Total records processed: {total}",
        f"Certain vs. uncertain: {certain_count} certain / {uncertain_count} uncertain "
        f"({uncertain_ratio:.1%} uncertain)",
        f"Average max probability across records: {avg_max_prob:.3f}",
        (
            f"records with LLM second opinions: {llm_count} "
            f"({llm_count / total:.1%} of batch)"
            if total > 0
            else "records with LLM second opinions: 0"
        ),
        "",
        f"MITRE technique coverage (by model base labels): {mitre_text}",
    ]

    # Heuristic recommendations
    if uncertain_ratio > 0.25:
        rec_records.append(
            "- A relatively high fraction of incidents are flagged as 'uncertain'. "
            "Consider routing these to an L2 triage queue and reviewing difficulty/threshold settings."
        )
    else:
        rec_records.append(
            "- Most incidents received confident classifications. Use the model as a decision-support signal, "
            "but still validate high-impact cases with additional telemetry."
        )

    if "data_exfiltration" in label_counts and label_counts["data_exfiltration"] > 0:
        rec_records.append(
            "- At least one potential data exfiltration pattern was detected. "
            "Ensure data protection and DLP queues are monitoring these hosts and users."
        )

    if "malware" in label_counts and label_counts["malware"] > 0:
        rec_records.append(
            "- Malware-related narratives are present. Confirm EDR containment status and review recent hunts."
        )

    if "web_attack" in label_counts and label_counts["web_attack"] > 0:
        rec_records.append(
            "- Web attack activity appears in this batch. Check WAF telemetry and customer-facing impact."
        )

    # LLM second-opinion summary (if used in this batch)
    # We also report which record indices received which LLM labels so the
    # analyst can quickly jump back to specific records in the bulk file.
    llm_records_indexed: list[tuple[int, dict]] = [
        (idx, r)
        for idx, r in enumerate(results, start=1)
        if r.get("llm_second_opinion")
    ]
    if llm_records_indexed:
        llm_total = len(llm_records_indexed)
        total_uncertain = label_counts.get("uncertain", 0)
        llm_labels = Counter(
            (rec["llm_second_opinion"].get("label") or "uncertain")
            for _, rec in llm_records_indexed
        )
        concrete_count = sum(c for lbl, c in llm_labels.items() if lbl != "uncertain")
        unresolved_count = llm_labels.get("uncertain", 0)

        # Map label -> list of line indices for quick reference
        records_by_label: dict[str, list[int]] = {}
        for idx, rec in llm_records_indexed:
            lbl = rec["llm_second_opinion"].get("label") or "uncertain"
            records_by_label.setdefault(lbl, []).append(idx)

        llm_records = [
            f"Total records with LLM second opinion: {llm_total}",
            f"Uncertain records in batch: {total_uncertain}",
            f"LLM provided a concrete label for {concrete_count} of {llm_total} LLM-reviewed records "
            f"({(concrete_count / llm_total):.1%})",
        ]

        if unresolved_count:
            llm_records.append(
                f"LLM left {unresolved_count} record(s) as 'uncertain' after review."
            )

        llm_records.extend(
            [
                "",
                "LLM suggested labels (second-opinion distribution):",
            ]
        )
        for lbl, count in llm_labels.most_common():
            llm_records.append(f"- {lbl}: {count}")

        # Compact high-impact view: highlight only the most critical
        # LLM labels on uncertain records, by record number.
        llm_records.extend(
            [
                "",
                "High-impact LLM suggestions on uncertain records (by record number):",
            ]
        )
        high_impact_labels = ("data_exfiltration", "access_abuse", "web_attack")
        any_highlighted = False
        for lbl in high_impact_labels:
            idx_list = records_by_label.get(lbl)
            if not idx_list:
                continue
            idx_str = ", ".join(str(i) for i in sorted(idx_list))
            pretty_lbl = lbl.replace("_", " ")
            llm_records.append(f"- {pretty_lbl}: records {idx_str}")
            any_highlighted = True
        if not any_highlighted:
            llm_records.append(
                "- none (no high-impact second-opinion labels on uncertain records)."
            )

        llm_records.append("")
        if concrete_count == 0:
            llm_records.append(
                "Observation: In this batch, the LLM second opinion did not promote any 'uncertain' records "
                "to a concrete label. This can happen when narratives are short, low-signal, or genuinely ambiguous. "
                "If this pattern persists across larger batches, consider tuning the prompt/model or temporarily "
                "disabling LLM assist until retrained with SOC-specific data."
            )
        else:
            llm_records.extend(
                [
                    "Observation:",
                    "- LLM second opinion helped convert some 'uncertain' cases into concrete labels.",
                    "- Prioritize manual review of uncertain incidents where the LLM suggests high-impact labels "
                    "such as 'data_exfiltration', 'access_abuse', or 'web_attack'.",
                ]
            )

        console.print(
            Panel(
                "\n".join(llm_records),
                title="Bulk LLM Second-Opinion Summary",
                border_style="bright_magenta",
            )
        )

        # Enrich bulk review recommendations with high-level LLM context
        rec_records.append("")
        rec_records.append("LLM Assist Highlights:")
        rec_records.append(
            f"- LLM second opinions were used on {llm_total} record(s); "
            f"{concrete_count} of those received a concrete label suggestion."
        )
        if unresolved_count:
            rec_records.append(
                f"- {unresolved_count} record(s) remain 'uncertain' even after LLM review; "
                "treat these as priority candidates for manual triage."
            )
        if any_highlighted:
            rec_records.append(
                "- Pay special attention to uncertain incidents where the LLM suggested "
                "high-impact labels such as 'data_exfiltration', 'access_abuse', or "
                "'web_attack' (see the Bulk LLM Second-Opinion Summary for record numbers)."
            )

    console.print(
        Panel(
            "\n".join(rec_records),
            title="Bulk Review Recommendations",
            border_style="bright_blue",
        )
    )


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Cybersecurity Incident NLP Triage CLI"
    )
    parser.add_argument("text", nargs="?", help="Incident description")
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Return raw JSON output instead of formatted text",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=None,
        help=(
            "Uncertainty threshold. If omitted, it is derived from --difficulty "
            f"(default={DEFAULT_UNCERTAINTY_THRESHOLD} for 'default')."
        ),
    )
    parser.add_argument(
        "-k",
        "--max-classes",
        type=int,
        default=None,
        help=(
            "Maximum number of classes to display in the probability table. "
            "If omitted, it is derived from --difficulty."
        ),
    )
    parser.add_argument(
        "-d",
        "--difficulty",
        choices=["default", "soc-medium", "soc-hard"],
        default="default",
        help=(
            "Difficulty / strictness mode for uncertainty handling. "
            "Use 'soc-hard' to mark more cases as 'uncertain'."
        ),
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help=(
            "Optional path to a text file for bulk mode; each non-empty line "
            "is treated as an incident description."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        help=(
            "Optional path to write JSONL predictions for bulk mode. "
            "Each line will contain one JSON object."
        ),
    )
    parser.add_argument(
        "-l",
        "--llm-second-opinion",
        action="store_true",
        help=(
            "If set, call a local LLM (e.g., Llama-2-7B-GGUF via llama-cpp-python) "
            "to provide a second opinion when the baseline model is uncertain."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve effective threshold/max_classes based on difficulty mode.
    mode = DIFFICULTY_MODES.get(args.difficulty, DIFFICULTY_MODES["default"])

    if args.threshold is None:
        effective_threshold = mode["threshold"]
    else:
        effective_threshold = args.threshold

    if args.max_classes is None:
        effective_max_classes = mode["max_classes"]
    else:
        effective_max_classes = args.max_classes

    # Banner only once per run
    console.print(f"[bold bright_cyan]{BANNER}[/bold bright_cyan]")
    console.print(
        f"[dim]Difficulty mode:[/] [bold]{args.difficulty}[/bold]  "
        f"(threshold={effective_threshold:.2f}, max_classes={effective_max_classes})\n"
    )

    vectorizer, clf, embedder, classes = load_artifacts()

    # Bulk mode: process input file if provided
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            console.print(f"[red]Input file not found: {input_path}[/red]")
            raise SystemExit(1)

        with input_path.open("r", encoding="utf-8") as f:
            records = [line.strip() for line in f]

        records = [
            line for line in records if line and not line.lstrip().startswith("#")
        ]
        if not records:
            console.print(
                "[yellow]No non-empty records to process in input file.[/yellow]"
            )
            return

        results = []
        total_records = len(records)
        for idx, text in enumerate(records, start=1):
            result = predict_with_uncertainty(
                text,
                vectorizer,
                clf,
                embedder,
                classes,
                effective_threshold,
                effective_max_classes,
            )

            # Optional LLM second opinion in bulk mode
            if args.llm_second_opinion:
                try:
                    status_msg = (
                        f"[bold magenta]Requesting LLM second opinion for line "
                        f"{idx}/{total_records}...[/bold magenta]"
                    )
                    with console.status(status_msg, spinner="dots"):
                        _llm_debug(
                            "Requesting LLM second opinion in bulk mode "
                            f"for line {idx}/{total_records}."
                        )
                        llm_result = llm_second_opinion(result["raw_text"])
                    result["llm_second_opinion"] = llm_result
                except Exception as exc:
                    _llm_debug(f"LLM second opinion failed in bulk mode: {exc!r}")

            results.append(result)

        # If an output file is provided, write JSONL; otherwise pretty-print
        if args.output_file:
            out_path = Path(args.output_file)
            with out_path.open("w", encoding="utf-8") as out_f:
                for r in results:
                    json_ready = {k: v for k, v in r.items() if k != "probs_sorted"}
                    json_ready["probs_sorted"] = [
                        {
                            "class": cls,
                            "probability": float(p),
                            "mitre_techniques": MITRE_MAPPING.get(cls, []),
                        }
                        for cls, p in r["probs_sorted"]
                    ]
                    json_ready["final_label_mitre_techniques"] = MITRE_MAPPING.get(
                        r["final_label"], []
                    )
                    # Include LLM second opinion in JSONL if present
                    if "llm_second_opinion" in r:
                        json_ready["llm_second_opinion"] = r["llm_second_opinion"]
                    out_f.write(json.dumps(json_ready) + "\n")
            console.print(
                f"[green]Wrote {len(results)} predictions to {out_path} (JSONL).[/green]"
            )
            summarize_bulk_results(results)
        else:
            for idx, r in enumerate(results, start=1):
                console.rule(f"[bold cyan]Record {idx}/{len(results)}")
                print_pretty(r)
                # If we have an LLM second opinion for this record, print it as well
                if r.get("llm_second_opinion"):
                    print_llm_panel(r)
            summarize_bulk_results(results)
        return

    # Single-shot mode
    if args.text:
        show_progress_bar()
        result = predict_with_uncertainty(
            args.text,
            vectorizer,
            clf,
            embedder,
            classes,
            effective_threshold,
            effective_max_classes,
        )

        # Optional LLM second opinion
        if args.llm_second_opinion:
            with console.status(
                "[bold magenta]Requesting LLM second opinion...[/bold magenta]",
                spinner="dots",
            ):
                llm_result = llm_second_opinion(result["raw_text"])
            result["llm_second_opinion"] = llm_result

        if args.json:
            print_json(result)
        else:
            print_pretty(result)
            if result.get("llm_second_opinion"):
                print_llm_panel(result)
        return

    # Interactive mode
    console.print("[bold cyan]Interactive Incident Triage CLI[/bold cyan]")
    console.print("Type 'exit' or 'quit' to stop.\n")

    while True:
        text = console.input("[bold yellow]Enter incident text: [/bold yellow]")
        if not text.strip():
            break
        if text.lower().strip() in {"exit", "quit"}:
            break
        show_progress_bar()
        result = predict_with_uncertainty(
            text,
            vectorizer,
            clf,
            embedder,
            classes,
            effective_threshold,
            effective_max_classes,
        )

        # Optional LLM second opinion in interactive mode
        if args.llm_second_opinion:
            with console.status(
                "[bold magenta]Requesting LLM second opinion...[/bold magenta]",
                spinner="dots",
            ):
                llm_result = llm_second_opinion(result["raw_text"])
            result["llm_second_opinion"] = llm_result

        if args.json:
            print_json(result)
        else:
            print_pretty(result)
            if result.get("llm_second_opinion"):
                print_llm_panel(result)


if __name__ == "__main__":
    main()
