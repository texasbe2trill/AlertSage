#!/usr/bin/env python3
import argparse
import json
import os
import sys
import re
import contextlib
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
# Optional: local LLM backend (e.g., Llama-2-7B-GGUF via llama-cpp-python)
# -----------------------------------------------------------------------------
try:
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Llama = None

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent  # repo root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.triage.preprocess import clean_description  # type: ignore
from src.triage.embeddings import get_embedder  # type: ignore
from src.triage.llm_client import (  # type: ignore
    HuggingFaceInferenceClient,
    RateLimiter,
    resolve_hf_credentials,
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

# Mapping of event types to MITRE ATT&CK techniques
MITRE_MAPPING = {
    "phishing": ["T1566"],
    "malware": ["T1204", "T1059", "T1486"],
    "web_attack": ["T1190", "T1110"],
    "access_abuse": ["T1078", "T1110"],
    "data_exfiltration": ["T1041", "T1567"],
    "policy_violation": ["T1052"],
    "benign_activity": [],
    "uncertain": [],
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

# -----------------------------------------------------------------------------
# Optional local LLM backend (Llama-2-7B-GGUF via llama-cpp-python)
# -----------------------------------------------------------------------------

LLM_MODEL_PATH = os.environ.get(
    "TRIAGE_LLM_MODEL",
    str(PROJECT_ROOT / "models" / "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"),
)
# Llama 3.1 supports 128k context, we use 8k for efficiency
LLM_CTX_SIZE = int(os.environ.get("TRIAGE_LLM_CTX", "8192"))
LLM_MAX_TOKENS = int(os.environ.get("TRIAGE_LLM_MAX_TOKENS", "1024"))
LLM_TEMP = float(os.environ.get("TRIAGE_LLM_TEMP", "0.1"))

HF_DEFAULT_MODEL = os.environ.get(
    "TRIAGE_HF_MODEL_DEFAULT",
    os.environ.get(
        "TRIAGE_HF_MODEL",
        os.environ.get(
            "HF_MODEL",
            "meta-llama/Llama-3.1-8B-Instruct:cerebras",
        ),
    ),
)
HF_ENDPOINT = os.environ.get(
    "TRIAGE_HF_ENDPOINT", "https://router.huggingface.co"
)
HF_TOKEN_ENV = os.environ.get("TRIAGE_HF_TOKEN") or os.environ.get("HF_TOKEN") or ""
HF_RATE_LIMIT_MAX = int(os.environ.get("TRIAGE_HF_MAX_REQUESTS", "5"))
HF_RATE_LIMIT_WINDOW = int(os.environ.get("TRIAGE_HF_WINDOW_SECONDS", "60"))

# -----------------------------------------------------------------------------
# LLM debug flag and helper
# -----------------------------------------------------------------------------
LLM_DEBUG = os.environ.get("TRIAGE_LLM_DEBUG", "0") == "1"


def _llm_debug(msg: str) -> None:
    """
    Lightweight, optionally-enabled debug logging for LLM assist.
    Enable by setting TRIAGE_LLM_DEBUG=1 in the environment.
    Writes to stderr to avoid contaminating JSON output on stdout.
    """
    if LLM_DEBUG:
        import sys

        print(f"[LLM DEBUG] {msg}", file=sys.stderr, flush=True)


def _resolve_llm_provider(provider: str | None, hf_available: bool) -> str:
    env_choice = (os.environ.get("TRIAGE_LLM_PROVIDER") or "").lower()
    requested = (provider or env_choice).lower()

    if requested in {"local", "gguf", "llama", "llama.cpp"}:
        return "local"

    if hf_available:
        return "huggingface"

    return "local"


def _resolve_hf_settings(model: str | None, token: str | None) -> tuple[str, str, bool]:
    resolved_model, resolved_token, has_token = resolve_hf_credentials(model, token)
    return resolved_model, resolved_token, has_token


def _get_hf_client(model: str, token: str, max_tokens: int):
    global _hf_rate_limiter
    if _hf_rate_limiter is None:
        _hf_rate_limiter = RateLimiter(
            max_requests=HF_RATE_LIMIT_MAX,
            window_seconds=HF_RATE_LIMIT_WINDOW,
        )

    return HuggingFaceInferenceClient(
        model=model,
        token=token,
        endpoint=HF_ENDPOINT,
        max_new_tokens=max_tokens,
        rate_limiter=_hf_rate_limiter,
    )


# -----------------------------------------------------------------------------
# Helper: Extract simple IOC-style indicators from text
# -----------------------------------------------------------------------------
def _extract_indicators(text: str) -> set[str]:
    """
    Extract simple IOC-style indicators (URLs, domains, emails, IPv4s) from text.
    Used to sanity-check LLM rationales for hallucinated entities that do not
    appear in the original incident narrative.
    """
    if not text:
        return set()

    indicators: set[str] = set()
    # Very lightweight regexes; we deliberately keep them simple and conservative.
    url_pattern = r"https?://[^\s]+"
    domain_pattern = (
        r"\b[a-zA-Z0-9.-]+\.(com|net|org|io|gov|edu|co|biz|info|cloud|xyz)\b"
    )
    email_pattern = r"\b\S+@\S+\b"
    ipv4_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

    for pattern in (url_pattern, domain_pattern, email_pattern, ipv4_pattern):
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                # For grouped patterns like domain_pattern, take the full match[0]
                indicators.add(str(match[0]).lower())
            else:
                indicators.add(str(match).lower())

    return indicators


_llm_instance = None  # cached singleton
_hf_rate_limiter: RateLimiter | None = None


def get_llm():
    """
    Lazily initialize and cache the local LLM.

    This requires:
      - `pip install llama-cpp-python`
      - the GGUF model file at LLM_MODEL_PATH (or TRIAGE_LLM_MODEL env var)
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is not installed or import failed. "
            "Install it with `pip install llama-cpp-python` and ensure "
            "a GGUF model exists at TRIAGE_LLM_MODEL or the default path."
        )

    # Auto-enable GPU acceleration if not explicitly configured
    # This provides 5-10x speedup on Apple Silicon and NVIDIA GPUs
    if "LLAMA_N_GPU_LAYERS" not in os.environ:
        os.environ["LLAMA_N_GPU_LAYERS"] = "999"  # Offload all layers to GPU
        _llm_debug("Auto-enabled GPU acceleration (LLAMA_N_GPU_LAYERS=999)")

    if "LLAMA_METAL" not in os.environ:
        os.environ["LLAMA_METAL"] = "1"  # Enable Metal for Apple Silicon
        _llm_debug("Auto-enabled Metal GPU backend (LLAMA_METAL=1)")

    if "LLAMA_CUDA" not in os.environ:
        os.environ["LLAMA_CUDA"] = "1"  # Enable CUDA for NVIDIA
        _llm_debug("Auto-enabled CUDA GPU backend (LLAMA_CUDA=1)")

    # Log GPU environment variables for debugging
    _llm_debug(f"GPU Environment Variables:")
    _llm_debug(f"  LLAMA_METAL={os.environ.get('LLAMA_METAL', 'not set')}")
    _llm_debug(f"  LLAMA_CUDA={os.environ.get('LLAMA_CUDA', 'not set')}")
    _llm_debug(f"  LLAMA_VULKAN={os.environ.get('LLAMA_VULKAN', 'not set')}")
    _llm_debug(
        f"  GGML_METAL_PATH_RESOURCES={os.environ.get('GGML_METAL_PATH_RESOURCES', 'not set')}"
    )
    _llm_debug(f"Model path: {LLM_MODEL_PATH}")
    _llm_debug(f"Context size: {LLM_CTX_SIZE}")
    _llm_debug(f"CPU threads: {os.cpu_count() or 8}")

    # Try to initialize with GPU layers if available
    n_gpu_layers = int(os.environ.get("LLAMA_N_GPU_LAYERS", "0"))
    if n_gpu_layers > 0:
        _llm_debug(f"Attempting to use {n_gpu_layers} GPU layers")
    else:
        _llm_debug("No GPU layers configured (n_gpu_layers=0, running on CPU)")

    # Temporarily don't suppress stderr/stdout during init so we can see GPU messages
    if LLM_DEBUG:
        _llm_debug("Initializing LLM with verbose output enabled...")
        _llm_instance = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_CTX_SIZE,
            n_threads=os.cpu_count() or 8,
            n_gpu_layers=n_gpu_layers,
            verbose=True,  # Enable verbose to see GPU backend info
        )
    else:
        with (
            open(os.devnull, "w") as devnull,
            contextlib.redirect_stderr(devnull),
            contextlib.redirect_stdout(devnull),
        ):
            _llm_instance = Llama(
                model_path=LLM_MODEL_PATH,
                n_ctx=LLM_CTX_SIZE,
                n_threads=os.cpu_count() or 8,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )

    _llm_debug("LLM initialization complete")
    return _llm_instance


def build_llm_rationale(label: str, incident_text: str) -> str:
    """
    Build a comprehensive, grounded rationale that summarizes the incident text
    and provides detailed SOC actions. We do NOT trust or reuse the LLM's
    narrative, only its label.

    Note: incident_text should already be in the desired format (raw or preprocessed)
    as determined by the caller. This function will NOT preprocess it again.
    """
    summary_text = incident_text.strip()
    if not summary_text:
        summary_text = "Incident narrative was provided but could not be parsed."

    triage = soc_triage_hint(label, "medium")
    actions = triage.get("actions", [])

    # Build comprehensive next steps with at least 5 actions
    if len(actions) < 5:
        # Add generic investigation steps
        generic_actions = [
            "Gather additional telemetry and review user history",
            "Check for similar patterns across other systems",
            "Document findings and timeline in case management system",
            "Coordinate with relevant teams (security, IT, legal)",
            "Monitor for continued suspicious activity",
        ]
        actions = actions + generic_actions

    # Take first 5-7 actions for comprehensive guidance
    action_count = min(len(actions), 7)
    next_steps = " ".join(
        f"{idx}) {action}" for idx, action in enumerate(actions[:action_count], start=1)
    )

    # Assess severity based on label
    severity_map = {
        "malware": "HIGH - Active threat requiring immediate containment",
        "data_exfiltration": "CRITICAL - Data loss in progress or completed",
        "phishing": "MEDIUM-HIGH - Credential compromise risk",
        "web_attack": "HIGH - Application security breach attempt",
        "access_abuse": "MEDIUM-HIGH - Unauthorized access detected",
        "policy_violation": "LOW-MEDIUM - Policy compliance issue",
        "benign_activity": "LOW - No immediate threat detected",
        "uncertain": "UNKNOWN - Requires manual analyst review",
    }
    impact_assessment = severity_map.get(label, "UNKNOWN - Unable to assess")

    return (
        f"Summary: {summary_text} "
        f"Impact: {impact_assessment}. "
        f"Model label (LLM second opinion): '{label}'. "
        f"Next steps: {next_steps}"
    )


def _lenient_extract_llm_fields(raw_text: str) -> dict:
    """
        Lenient fallback parser for nearly-JSON LLM output.

    Extracts only 'label' and 'mitre_ids' using regex, ignoring
    the 'rationale' text (we will rebuild our own rationale downstream).
    """
    label_match = re.search(r'"label"\s*:\s*"([^"]+)"', raw_text)
    label = label_match.group(1).strip() if label_match else "uncertain"

    mitre_ids: list[str] = []
    mitre_match = re.search(r'"mitre_ids"\s*:\s*\[(.*?)\]', raw_text, re.DOTALL)
    if mitre_match:
        inner = mitre_match.group(1)
        for token in inner.split(","):
            token = token.strip().strip('"').strip("'")
            if token:
                mitre_ids.append(token)

    return {"label": label, "mitre_ids": mitre_ids, "rationale": ""}


def llm_second_opinion(
    text: str,
    skip_preprocessing: bool = False,
    *,
    provider: str | None = None,
    hf_model: str | None = None,
    hf_token: str | None = None,
    max_tokens: int | None = None,
) -> dict:
    """
    Use a local LLM or Hugging Face Inference as a *second opinion* on the incident narrative.

    Args:
        text: The incident narrative text
        skip_preprocessing: If True, pass original text to LLM without normalization.
                          If False (default), apply clean_description() for consistency.
                          Set via TRIAGE_LLM_RAW_TEXT=1 environment variable.

        Returns a dict with:
      - label: suggested event_type or 'uncertain'
      - mitre_ids: list of ATT&CK technique IDs (best-effort)
      - rationale: short SOC-style explanation

        If the configured backend is unavailable, returns a safe placeholder so the
        CLI never crashes.
    """
    # Check environment variable for raw text mode (used by UI)
    if not skip_preprocessing:
        skip_preprocessing = os.environ.get("TRIAGE_LLM_RAW_TEXT", "0") == "1"

    # Resolve HF credentials (Streamlit secrets → env → provided args)
    hf_model_resolved, hf_token_resolved, hf_available = _resolve_hf_settings(
        hf_model, hf_token
    )

    provider_choice = _resolve_llm_provider(provider, hf_available)
    max_gen_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

    # Optionally preprocess the text for LLM (default behavior for CLI consistency)
    llm_text = text if skip_preprocessing else clean_description(text)

    if skip_preprocessing:
        _llm_debug(f"Using RAW text for LLM (length: {len(text)} chars)")
    else:
        _llm_debug(f"Using PREPROCESSED text for LLM (length: {len(llm_text)} chars)")

    system_instructions = (
        "You are assisting with SOC incident triage. "
        "You MUST respond with a single valid JSON object only, "
        "with keys: 'label', 'mitre_ids', 'rationale'. "
        "The 'label' must be one of: phishing, malware, access_abuse, "
        "data_exfiltration, policy_violation, web_attack, benign_activity, uncertain. "
        "The 'mitre_ids' must be a list of ATT&CK technique IDs like ['T1566']. "
        "You must ground your answer ONLY in the 'Incident narrative' text I provide. "
        "Do NOT invent any prior infection chain, phishing emails, malware, or other attack steps "
        "that are not explicitly stated in the narrative. If the narrative does not specify how access "
        "was obtained or what happened before/after, explicitly say that it is unknown. "
        "When the narrative clearly describes movement of internal or corporate data to a personal or external "
        "cloud storage location (for example, 'user transferred internal company data to a personal Dropbox account'), "
        "you should normally classify this as 'data_exfiltration' unless the narrative clearly states the activity "
        "is authorized and benign. "
        "Do NOT output headings, notes, or multiple examples. Do NOT use the word 'Example' in your output. "
        "If you are unsure, still return JSON with label 'uncertain' and an empty "
        "mitre_ids list. The 'rationale' must be detailed and comprehensive (3–6 sentences), SOC-focused, "
        "and MUST include: 1) A thorough summary of what happened, 2) Assessment of threat severity and potential impact, "
        "3) At least 3-5 specific, actionable next steps or recommended actions for the SOC analyst with technical details. "
        "Format the rationale as: 'Summary: [detailed description]. Impact: [severity assessment]. Next steps: 1) [detailed action] 2) [detailed action] 3) [detailed action]...' "
        "Provide specific commands, log locations, or investigation techniques where applicable."
    )

    # Prefer chat-style API if available, since the model is recognized as llama-2.
    messages = [
        {
            "role": "system",
            "content": system_instructions,
        },
        {
            "role": "user",
            "content": (
                f"Incident narrative:\n{llm_text}\n\n"
                "Return JSON ONLY (no extra commentary)."
            ),
        },
    ]

    prompt = f"""
{system_instructions}

Incident narrative:
{llm_text}

Now respond with a single valid JSON object ONLY, with keys "label", "mitre_ids", and "rationale" for this specific incident.
Do not include any explanations, headings, notes, or examples.
Do not repeat these instructions.
""".strip()
    data: dict | None = None

    if provider_choice == "huggingface":
        if hf_token_resolved:
            try:
                hf_client = _get_hf_client(
                    hf_model_resolved, hf_token_resolved, max_gen_tokens
                )
                data = hf_client.generate_json(prompt, max_tokens=max_gen_tokens)
                _llm_debug("HF inference completed successfully.")
            except Exception as exc:  # pragma: no cover - network dependent
                _llm_debug(
                    f"HF inference failed: {exc!r}; falling back to local if available."
                )
        else:
            _llm_debug(
                "HF provider selected but no token provided; falling back to local."
            )

    if data is None:
        if Llama is None:
            _llm_debug(
                "llama-cpp-python is not available; returning uncertain placeholder."
            )
            return {
                "label": "uncertain",
                "mitre_ids": [],
                "rationale": (
                    "LLM assist is not configured. Set HF_TOKEN (and optional HF_MODEL) in Streamlit secrets, "
                    "or run locally with llama.cpp + GGUF."
                ),
            }

        try:
            llm = get_llm()
            _llm_debug("Successfully initialized LLM backend.")
        except Exception as exc:
            _llm_debug(f"Failed to initialize LLM backend: {exc!r}")
            return {
                "label": "uncertain",
                "mitre_ids": [],
                "rationale": (
                    "LLM assist could not be initialized. "
                    f"Details: {exc}. Proceed with standard SOC triage without LLM."
                ),
            }

        try:
            # Prefer chat-style API with messages (works better for chat-tuned models)
            _llm_debug("Starting LLM inference...")
            import time

            start_time = time.time()

            # Build a simple chat-style prompt string for llama_cpp
            prompt_text = "\n".join(
                f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
                for m in messages
            )

            try:
                output = llm(
                    prompt=prompt_text,
                    max_tokens=max_gen_tokens,
                    temperature=0.05,
                    top_p=0.5,
                    top_k=20,
                )
            except TypeError:
                # Fallback for llama_cpp versions that use positional prompt
                output = llm(
                    prompt_text,
                    max_tokens=max_gen_tokens,
                    temperature=0.05,
                    top_p=0.5,
                    top_k=20,
                )

            elapsed = time.time() - start_time
            _llm_debug(f"LLM inference completed in {elapsed:.2f} seconds")

            choice = output.get("choices", [{}])[0] if isinstance(output, dict) else {}
            raw_text = (
                choice.get("message", {}).get("content", "")
                or choice.get("text", "")
            ).strip()
            _llm_debug(f"Raw LLM output: {raw_text!r}")

            # Preprocess and normalize LLM output for JSON parsing
            text_for_json = raw_text.strip()
            # Remove code block fences if present
            if text_for_json.startswith("```"):
                lines = text_for_json.splitlines()
                if lines:
                    lines = lines[1:]  # drop opening ```
                    if lines and lines[-1].strip().startswith("```"):
                        lines = lines[:-1]  # drop closing ```
                    text_for_json = "\n".join(lines).strip()
            # Remove single quotes around the entire output if present
            if text_for_json.startswith("'") and text_for_json.endswith("'"):
                candidate = text_for_json[1:-1].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    text_for_json = candidate
            _llm_debug(f"Normalized text for JSON parsing: {text_for_json!r}")

            # Quick sanity check for obviously-wrong template echoes or empty output
            if not text_for_json or "[INST]" in text_for_json:
                _llm_debug(
                    "LLM output is empty or appears to be a chat template echo; "
                    "treating as invalid and falling back."
                )
                raise ValueError(
                    f"Invalid non-JSON output from LLM: {text_for_json!r}"
                )

            # First try direct JSON parse; if it fails, fall back to lenient extraction.
            try:
                _llm_debug("Attempting direct JSON parse of LLM output.")
                data = json.loads(text_for_json)
            except Exception as parse_exc:
                _llm_debug(
                    f"Direct JSON parse failed: {parse_exc!r}; falling back to lenient field extraction for label/mitre_ids only."
                )
                # We only need 'label' and 'mitre_ids'; ignore malformed rationale text.
                data = _lenient_extract_llm_fields(text_for_json)

        except Exception as exc:
            # Fall back to a conservative, non-breaking structure
            _llm_debug(f"LLM assist failed or returned invalid JSON: {exc!r}")
            safe_label = "uncertain"
            safe_rationale = build_llm_rationale(safe_label, llm_text)
            return {
                "label": safe_label,
                "mitre_ids": [],
                "rationale": safe_rationale,
            }

    if data is None:
        _llm_debug("LLM output was empty after all backends; returning uncertain.")
        safe_label = "uncertain"
        safe_rationale = build_llm_rationale(safe_label, llm_text)
        return {
            "label": safe_label,
            "mitre_ids": [],
            "rationale": safe_rationale,
        }

    # Normalise and sanity-check the response
    label = data.get("label", "uncertain")
    _llm_debug(f"Parsed LLM JSON: {data!r}")
    _llm_debug(f"LLM-suggested label before normalization: {label!r}")
    # Normalize common non-canonical labels into our fixed taxonomy
    synonym_map = {
        "ransomware": "malware",
        "brute_force_attack": "access_abuse",
    }
    if label in synonym_map:
        canonical = synonym_map[label]
        _llm_debug(f"Normalizing LLM label {label!r} to canonical label {canonical!r}.")
        label = canonical
    if label not in MITRE_MAPPING.keys() and label != "uncertain":
        label = "uncertain"

    raw_mitre_ids = data.get("mitre_ids", [])
    if not isinstance(raw_mitre_ids, list):
        raw_mitre_ids = []

    lower_text = text.lower()

    exfil_keywords = [
        # Generic data movement
        "exfil",
        "exfiltration",
        "data exfil",
        "data leak",
        "data theft",
        "download",
        "downloaded",
        "upload",
        "uploaded",
        "transfer",
        "transferred",
        "copied",
        "moved",
        "synced",
        "synchronized",
        "archive",
        "archived",
        "compressed",
        "zip",
        "tar.gz",
        "7z",
        "export",
        "exported",
        "dump",
        "database dump",
        "db dump",
        # Channels / destinations
        "dropbox",
        "google drive",
        "gdrive",
        "onedrive",
        "box.com",
        "box drive",
        "sharefile",
        "sharepoint",
        "share point",
        "wetransfer",
        "mega.nz",
        "mega.io",
        "cloud storage",
        "object storage",
        "s3",
        "s3 bucket",
        "ftp",
        "sftp",
        "scp",
        "rsync",
        "rclone",
        # Removable media
        "usb",
        "thumb drive",
        "flash drive",
        "removable media",
        "external drive",
        "external disk",
        "burned to dvd",
        # Email / messaging exfil
        "sent to personal email",
        "personal email account",
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "protonmail",
        "forwarded externally",
        "emailed externally",
        "sent outside organization",
    ]

    malware_keywords = [
        # Generic malware families
        "malware",
        "ransomware",
        "trojan",
        "virus",
        "worm",
        "backdoor",
        "remote access trojan",
        "rat",
        "infostealer",
        "info stealer",
        "keylogger",
        "key logger",
        "spyware",
        "adware",
        "crypto-miner",
        "cryptominer",
        "coinminer",
        # Behavior / artifacts
        "malicious payload",
        "payload dropped",
        "dropped file",
        "suspicious process",
        "unknown binary",
        "unsigned binary",
        "persistence",
        "autorun",
        "runkey",
        "scheduled task",
        "schtasks.exe",
        "registry run key",
        "dll sideloading",
        "sideloading",
        "code injection",
        "shellcode",
        "beacon",
        "c2",
        "command and control",
        "callback domain",
        # Script / LOLBAS patterns
        "powershell",
        "powershell.exe",
        "wscript.exe",
        "cscript.exe",
        "mshta.exe",
        "rundll32.exe",
        "regsvr32.exe",
        "living off the land",
        "lolbin",
        # Ransomware / encryption language
        "ransom",
        "ransom note",
        "decrypt",
        "decryptor",
        "encrypting",
        "encrypted",
        "encryption",
        "files renamed",
        "file extension changed",
        # Remote access tool / RAT language
        "remote access tool",
        "remote administration tool",
        "unapproved remote access",
        "unauthorized remote access",
        "screen sharing tool",
        "remote desktop tool",
        # EDR / AV detections
        "edr alert",
        "edr detection",
        "av alert",
        "antivirus alert",
        "detected malware",
        "blocked malware",
        "malicious hash",
        "malicious executable",
    ]

    web_keywords = [
        # Generic web app / http semantics
        "web application",
        "web app",
        "web server",
        "website",
        "portal",
        "api endpoint",
        "rest api",
        "graphql",
        "http",
        "https",
        "url path",
        "endpoint",
        "uri",
        # Common web infra
        "apache",
        "nginx",
        "iis",
        "tomcat",
        "reverse proxy",
        "load balancer",
        "waf",
        "web application firewall",
        # Web attack patterns
        "webshell",
        "web shell",
        "file upload handler",
        "upload handler",
        "sql injection",
        "sql-injection",
        "sqli",
        "xss",
        "cross-site scripting",
        "csrf",
        "cross-site request forgery",
        "ssrf",
        "server-side request forgery",
        "lfi",
        "rfi",
        "path traversal",
        # DoS / DDoS language
        "http flood",
        "layer 7 ddos",
        "ddos",
        "denial of service",
        "distributed denial-of-service",
        "spike in http requests",
        "excessive http requests",
        "botnet traffic",
        "suspicious user agents",
        # Login pages / auth endpoints
        "/login",
        "/signin",
        "/auth",
        "login page",
        "authentication endpoint",
    ]

    access_keywords = [
        # Auth / identity concepts
        "unauthorized",
        "unauthorised",
        "suspicious login",
        "suspicious logon",
        "login",
        "logon",
        "sign-in",
        "signin",
        "authentication",
        "auth failure",
        "failed login",
        "failed logon",
        "failed authentication",
        "account",
        "user account",
        "service account",
        "privileged account",
        "admin account",
        # Credential / password language
        "credential",
        "credentials",
        "password",
        "passphrase",
        "password reset",
        "password change",
        "password spray",
        "brute force",
        "dictionary attack",
        "credential stuffing",
        "compromised credentials",
        # Access control
        "mfa",
        "multi-factor",
        "otp",
        "one-time passcode",
        "sso",
        "single sign-on",
        "okta",
        "entra id",
        "azure ad",
        "pingfederate",
        "ping federate",
        "duo",
        "vpn",
        "remote access vpn",
        "citrix",
        "rdp",
        "remote desktop",
        "beyondtrust",
        "privilege",
        "role",
        "entitlement",
        "elevated rights",
        "access",
        "session",
        "session hijack",
        # Account status
        "account lockout",
        "locked out",
        "disabled account",
        "new account created",
        "suspicious account creation",
    ]

    policy_keywords = [
        # Generic policy / governance language
        "policy",
        "corporate policy",
        "company policy",
        "policy violation",
        "policy breach",
        "violated policy",
        "acceptable use",
        "acceptable use policy",
        "aup",
        "code of conduct",
        "code-of-conduct",
        "data handling standard",
        "information security policy",
        # Org structures / ownership
        "hr",
        "human resources",
        "compliance",
        "governance",
        "grc",
        "legal",
        # Insider risk / misuse
        "insider risk",
        "misuse of resources",
        "misuse of company resources",
        "inappropriate content",
        "inappropriate use",
        "shadow it",
        "unsanctioned application",
        "unsanctioned cloud service",
        # DLP / data handling
        "dlp alert",
        "data loss prevention",
        "classified data",
        "sensitive data",
        "confidential data",
        "handling of pii",
        "handling of phi",
        # HR / disciplinary
        "hr case opened",
        "hr investigation",
        "written warning",
        "disciplinary action",
    ]

    def _has_any(text_lc: str, keywords: list[str]) -> bool:
        return any(k in text_lc for k in keywords)

    # ------------------------------------------------------------------
    # 1) Hallucination guardrail: reject rationales that introduce new
    #    IOC-like indicators (URLs/domains/emails/IPs) not present in
    #    the original incident narrative.
    # ------------------------------------------------------------------
    raw_rationale = str(data.get("rationale", "") or "")
    incident_iocs = _extract_indicators(text)
    rationale_iocs = _extract_indicators(raw_rationale)
    extra_iocs = rationale_iocs - incident_iocs
    if extra_iocs:
        _llm_debug(
            f"LLM rationale introduced IOC-like indicators not present in the narrative: {sorted(extra_iocs)!r}; "
            "treating LLM output as hallucinated and downgrading to 'uncertain'."
        )
        # Return a conservative, grounded uncertain result using our own rationale builder.
        safe_label = "uncertain"
        safe_rationale = build_llm_rationale(safe_label, llm_text)
        return {
            "label": safe_label,
            "mitre_ids": [],
            "rationale": safe_rationale,
        }

    # ------------------------------------------------------------------
    # 2) Keyword-based validation for specific labels. If the narrative
    #    does not contain basic supporting terms, downgrade to 'uncertain'.
    # ------------------------------------------------------------------
    if label == "data_exfiltration":
        if not _has_any(lower_text, exfil_keywords):
            _llm_debug(
                "LLM suggested 'data_exfiltration' but no obvious exfiltration "
                "keywords found in narrative; downgrading label to 'uncertain'."
            )
            label = "uncertain"

    elif label == "malware":
        if not _has_any(lower_text, malware_keywords):
            _llm_debug(
                "LLM suggested 'malware' but no clear malware-related terms "
                "found in narrative; downgrading label to 'uncertain'."
            )
            label = "uncertain"

    elif label == "web_attack":
        if not _has_any(lower_text, web_keywords):
            _llm_debug(
                "LLM suggested 'web_attack' but no web-app related indicators "
                "found in narrative; downgrading label to 'uncertain'."
            )
            label = "uncertain"

    elif label == "access_abuse":
        if not _has_any(lower_text, access_keywords):
            _llm_debug(
                "LLM suggested 'access_abuse' but no identity/access-related "
                "terms found in narrative; downgrading label to 'uncertain'."
            )
            label = "uncertain"

    elif label == "policy_violation":
        if not _has_any(lower_text, policy_keywords):
            _llm_debug(
                "LLM suggested 'policy_violation' but no policy/HR/compliance "
                "language found in narrative; downgrading label to 'uncertain'."
            )
            label = "uncertain"

    # Sanity check: if phishing label but no email indicators, downgrade.
    if label == "phishing" and not re.search(
        r"\b(email|mailbox|inbox|message|phishing|link|url|clicked)\b",
        lower_text,
    ):
        _llm_debug(
            "LLM suggested 'phishing' but no email-related indicators found "
            "in narrative; downgrading label to 'uncertain'."
        )
        label = "uncertain"

    # ------------------------------------------------------------------
    # 3) Second-opinion bias: if we still have 'uncertain', try to promote
    #    to a concrete label based on the same keyword heuristics. This
    #    keeps the LLM acting as a true "second opinion" on uncertain cases.
    # ------------------------------------------------------------------
    if label == "uncertain":
        heuristic_label: str | None = None
        if _has_any(lower_text, exfil_keywords):
            heuristic_label = "data_exfiltration"
        elif _has_any(lower_text, malware_keywords):
            heuristic_label = "malware"
        elif _has_any(lower_text, web_keywords):
            heuristic_label = "web_attack"
        elif _has_any(lower_text, access_keywords):
            heuristic_label = "access_abuse"
        elif _has_any(lower_text, policy_keywords):
            heuristic_label = "policy_violation"
        elif re.search(
            r"\b(email|mailbox|inbox|message|phishing|link|url|clicked)\b", lower_text
        ):
            heuristic_label = "phishing"

        if heuristic_label:
            _llm_debug(
                f"LLM returned 'uncertain'; promoting to heuristic label {heuristic_label!r} "
                "based on narrative keywords."
            )
            label = heuristic_label

    # Now that we've finalized the label, derive MITRE techniques.
    # First try to use LLM-suggested MITRE IDs if valid, otherwise fall back to label mapping
    canonical_mitre = MITRE_MAPPING.get(label, [])
    if raw_mitre_ids:
        # Use LLM's MITRE suggestions if provided
        mitre_ids = raw_mitre_ids
    elif canonical_mitre:
        # Fall back to label-based mapping
        mitre_ids = canonical_mitre
    else:
        mitre_ids = []

    _llm_debug(f"Final normalized label: {label!r}, mitre_ids: {mitre_ids!r}")

    rationale = build_llm_rationale(label, llm_text)

    return {
        "label": label,
        "mitre_ids": mitre_ids,
        "rationale": rationale,
    }


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
    """
    Load ML models and embedder.
    
    Uses cached loader from model.py to prevent repeated disk I/O.
    Models are loaded once and cached in memory for the process lifetime.
    """
    import sys
    # Get the current model module (handles module reloading in tests)
    # Check src.triage.model first (used by tests), then triage.model (installed package)
    model_module = sys.modules.get("src.triage.model") or sys.modules.get("triage.model")
    if model_module is None:
        try:
            from triage import model as model_module
        except ImportError:
            from src.triage import model as model_module
    
    vectorizer, clf = model_module.load_vectorizer_and_model()
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


# -----------------------------------------------------------------------------
# SOC triage hint and analyst note helpers
# -----------------------------------------------------------------------------
def soc_triage_hint(label: str, uncertainty_level: str) -> dict:
    """
    Map (event_type, uncertainty_level) to SOC-style guidance:
      - recommended queue / owner
      - rough priority
      - 1–3 suggested first actions
    """
    base = {
        "access_abuse": {
            "queue": "Identity / IAM",
            "priority": "High",
            "actions": [
                "Review recent sign-in locations and device fingerprints.",
                "Force password reset and invalidate active sessions.",
                "Check MFA enrollment, recent changes, and delegated access.",
            ],
        },
        "benign_activity": {
            "queue": "Service Desk / Monitoring",
            "priority": "Low",
            "actions": [
                "Confirm maintenance window, known outages, or deployments.",
                "Document as a non-security incident if impact is benign.",
            ],
        },
        "data_exfiltration": {
            "queue": "Data Protection / DLP",
            "priority": "High",
            "actions": [
                "Identify file types and data classifications involved.",
                "Confirm user intent with the manager and HR if appropriate.",
                "Block or quarantine the exfiltration channel if still active.",
            ],
        },
        "malware": {
            "queue": "Endpoint / Incident Response",
            "priority": "High",
            "actions": [
                "Isolate the affected host from the network.",
                "Collect EDR timeline, process tree, and artifact details.",
                "Hunt for similar indicators across the environment.",
            ],
        },
        "phishing": {
            "queue": "Email / Threat Intel",
            "priority": "Medium",
            "actions": [
                "Collect full message headers and original phishing artifact.",
                "Search for similar messages across user mailboxes.",
                "Update email gateway rules and block IOCs if confirmed.",
            ],
        },
        "policy_violation": {
            "queue": "GRC / Insider Risk",
            "priority": "Medium",
            "actions": [
                "Validate applicable corporate policies for the behavior.",
                "Notify manager or HR for repeated or severe violations.",
                "Coordinate with IR if potential data misuse is suspected.",
            ],
        },
        "web_attack": {
            "queue": "Network / AppSec",
            "priority": "High",
            "actions": [
                "Review WAF and load balancer logs around the timeframe.",
                "Confirm impact on customer-facing services and SLAs.",
                "Identify attacker IP ranges and consider blocking or rate limits.",
            ],
        },
        "uncertain": {
            "queue": "Triage / L2 Review",
            "priority": "Review",
            "actions": [
                "Gather additional context (EDR, proxy, and auth logs).",
                "Clarify any available user report or ticket history.",
            ],
        },
    }

    info = base.get(label, base["uncertain"]).copy()

    # Tweak priority wording based on uncertainty band
    if uncertainty_level == "low":
        info["priority"] = (
            f"{info['priority']} (model confidence: low, manual review recommended)"
        )
    elif uncertainty_level == "medium":
        info["priority"] = f"{info['priority']} (model confidence: medium)"

    return info


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
