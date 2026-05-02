"""LLM helpers used by the Streamlit UI and the CLI.

Previously these functions lived in `cli.py`. They were moved here so the
UI does not have to import the entire 1900-line CLI module (and its
argparse, rich console, and other startup-time side effects) just to use
the LLM second-opinion logic.

Public surface (import-safe):
    - llm_second_opinion(...)
    - build_llm_rationale(...)
    - get_llm()
    - MITRE_MAPPING
    - soc_triage_hint(...)
    - constants: LLM_MODEL_PATH, LLM_CTX_SIZE, LLM_MAX_TOKENS, LLM_TEMP,
      HF_DEFAULT_MODEL, HF_ENDPOINT, HF_TOKEN_ENV,
      HF_RATE_LIMIT_MAX, HF_RATE_LIMIT_WINDOW, LLM_DEBUG
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from .llm_client import (
    HuggingFaceInferenceClient,
    RateLimiter,
    resolve_hf_credentials,
)
from .preprocess import clean_description

# Optional local LLM backend. Importing llama_cpp is expensive and pulls
# in a native extension; keep it optional so the UI does not pay the
# cost on Streamlit Cloud where the GGUF path is not used.
try:  # pragma: no cover - optional dependency
    from llama_cpp import Llama  # type: ignore
except Exception:  # pragma: no cover
    Llama = None  # type: ignore


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Mapping of event types to MITRE ATT&CK techniques.
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

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

LLM_MODEL_PATH = os.environ.get(
    "TRIAGE_LLM_MODEL",
    str(PROJECT_ROOT / "models" / "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf"),
)
# Llama 3.1 supports 128k context, we use 8k for efficiency.
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

LLM_DEBUG = os.environ.get("TRIAGE_LLM_DEBUG", "0") == "1"


def _llm_debug(msg: str) -> None:
    """Lightweight debug logger. Enabled by TRIAGE_LLM_DEBUG=1.

    Writes to stderr to avoid contaminating JSON output on stdout.
    """
    if LLM_DEBUG:
        print(f"[LLM DEBUG] {msg}", file=sys.stderr, flush=True)


# -----------------------------------------------------------------------------
# Provider routing helpers
# -----------------------------------------------------------------------------

def _resolve_llm_provider(provider: str | None, hf_available: bool) -> str:
    env_choice = (os.environ.get("TRIAGE_LLM_PROVIDER") or "").lower()
    requested = (provider or env_choice).lower()

    if requested in {"local", "gguf", "llama", "llama.cpp"}:
        return "local"

    if hf_available:
        return "huggingface"

    return "local"


def _resolve_hf_settings(model: str | None, token: str | None) -> tuple[str, str, bool]:
    return resolve_hf_credentials(model, token)


_llm_instance = None  # cached singleton for local Llama
_hf_rate_limiter: RateLimiter | None = None


def _get_hf_client(model: str, token: str, max_tokens: int) -> HuggingFaceInferenceClient:
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
# Hallucination guard helpers
# -----------------------------------------------------------------------------

def _extract_indicators(text: str) -> set[str]:
    """Extract simple IOC-style indicators from text.

    Used to sanity-check LLM rationales for hallucinated entities that do
    not appear in the original incident narrative.
    """
    if not text:
        return set()

    indicators: set[str] = set()
    url_pattern = r"https?://[^\s]+"
    domain_pattern = (
        r"\b[a-zA-Z0-9.-]+\.(com|net|org|io|gov|edu|co|biz|info|cloud|xyz)\b"
    )
    email_pattern = r"\b\S+@\S+\b"
    ipv4_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"

    for pattern in (url_pattern, domain_pattern, email_pattern, ipv4_pattern):
        for match in re.findall(pattern, text):
            if isinstance(match, tuple):
                indicators.add(str(match[0]).lower())
            else:
                indicators.add(str(match).lower())

    return indicators


def _lenient_extract_llm_fields(raw_text: str) -> dict:
    """Lenient fallback parser for nearly-JSON LLM output.

    Extracts only 'label' and 'mitre_ids' using regex, ignoring the
    'rationale' text (we will rebuild our own rationale downstream).
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


# -----------------------------------------------------------------------------
# Local Llama backend (optional)
# -----------------------------------------------------------------------------

def get_llm():
    """Lazily initialize and cache the local LLM.

    Requires:
      - `pip install llama-cpp-python`
      - the GGUF model file at LLM_MODEL_PATH (or the TRIAGE_LLM_MODEL env var)
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is not installed or import failed. "
            "Install it with `pip install llama-cpp-python` and ensure a "
            "GGUF model exists at TRIAGE_LLM_MODEL or the default path."
        )

    if "LLAMA_N_GPU_LAYERS" not in os.environ:
        os.environ["LLAMA_N_GPU_LAYERS"] = "999"
        _llm_debug("Auto-enabled GPU acceleration (LLAMA_N_GPU_LAYERS=999)")

    if "LLAMA_METAL" not in os.environ:
        os.environ["LLAMA_METAL"] = "1"
        _llm_debug("Auto-enabled Metal GPU backend (LLAMA_METAL=1)")

    if "LLAMA_CUDA" not in os.environ:
        os.environ["LLAMA_CUDA"] = "1"
        _llm_debug("Auto-enabled CUDA GPU backend (LLAMA_CUDA=1)")

    _llm_debug("GPU Environment Variables:")
    _llm_debug(f"  LLAMA_METAL={os.environ.get('LLAMA_METAL', 'not set')}")
    _llm_debug(f"  LLAMA_CUDA={os.environ.get('LLAMA_CUDA', 'not set')}")
    _llm_debug(f"  LLAMA_VULKAN={os.environ.get('LLAMA_VULKAN', 'not set')}")
    _llm_debug(
        f"  GGML_METAL_PATH_RESOURCES={os.environ.get('GGML_METAL_PATH_RESOURCES', 'not set')}"
    )
    _llm_debug(f"Model path: {LLM_MODEL_PATH}")
    _llm_debug(f"Context size: {LLM_CTX_SIZE}")
    _llm_debug(f"CPU threads: {os.cpu_count() or 8}")

    n_gpu_layers = int(os.environ.get("LLAMA_N_GPU_LAYERS", "0"))
    if n_gpu_layers > 0:
        _llm_debug(f"Attempting to use {n_gpu_layers} GPU layers")
    else:
        _llm_debug("No GPU layers configured (n_gpu_layers=0, running on CPU)")

    if LLM_DEBUG:
        _llm_debug("Initializing LLM with verbose output enabled...")
        _llm_instance = Llama(
            model_path=LLM_MODEL_PATH,
            n_ctx=LLM_CTX_SIZE,
            n_threads=os.cpu_count() or 8,
            n_gpu_layers=n_gpu_layers,
            verbose=True,
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


# -----------------------------------------------------------------------------
# SOC triage hints (used by the rationale builder and the CLI)
# -----------------------------------------------------------------------------

def soc_triage_hint(label: str, uncertainty_level: str) -> dict:
    """Map (event_type, uncertainty_level) to SOC-style guidance."""
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

    if uncertainty_level == "low":
        info["priority"] = (
            f"{info['priority']} (model confidence: low, manual review recommended)"
        )
    elif uncertainty_level == "medium":
        info["priority"] = f"{info['priority']} (model confidence: medium)"

    return info


# -----------------------------------------------------------------------------
# Rationale builder (does not call any LLM)
# -----------------------------------------------------------------------------

def build_llm_rationale(label: str, incident_text: str) -> str:
    """Build a grounded rationale string given the LLM's label.

    We do NOT trust or reuse the LLM's narrative, only its label.
    `incident_text` should already be in the desired format (raw or
    preprocessed) as determined by the caller.
    """
    summary_text = incident_text.strip()
    if not summary_text:
        summary_text = "Incident narrative was provided but could not be parsed."

    triage = soc_triage_hint(label, "medium")
    actions = triage.get("actions", [])

    if len(actions) < 5:
        generic_actions = [
            "Gather additional telemetry and review user history",
            "Check for similar patterns across other systems",
            "Document findings and timeline in case management system",
            "Coordinate with relevant teams (security, IT, legal)",
            "Monitor for continued suspicious activity",
        ]
        actions = actions + generic_actions

    action_count = min(len(actions), 7)
    next_steps = " ".join(
        f"{idx}) {action}" for idx, action in enumerate(actions[:action_count], start=1)
    )

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


# -----------------------------------------------------------------------------
# Provider-aware LLM second opinion
# -----------------------------------------------------------------------------

# Maps stringy provider identifiers used by the UI and CLI to the
# canonical names this module understands.
_PROVIDER_ALIASES = {
    "local": "local",
    "gguf": "local",
    "llama": "local",
    "llama.cpp": "local",
    "huggingface": "huggingface",
    "hf": "huggingface",
    "openai": "openai",
    "gpt": "openai",
    "anthropic": "anthropic",
    "claude": "anthropic",
}


def _normalize_provider(provider: str | None) -> str:
    if not provider:
        return ""
    return _PROVIDER_ALIASES.get(provider.strip().lower(), provider.strip().lower())


def llm_second_opinion(
    text: str,
    skip_preprocessing: bool = False,
    *,
    provider: str | None = None,
    hf_model: str | None = None,
    hf_token: str | None = None,
    openai_model: str | None = None,
    openai_api_key: str | None = None,
    anthropic_model: str | None = None,
    anthropic_api_key: str | None = None,
    max_tokens: int | None = None,
) -> dict:
    """Use a local LLM, Hugging Face, OpenAI, or Anthropic as a second
    opinion on the incident narrative.

    Returns a dict with `label`, `mitre_ids`, and `rationale`. When the
    configured backend is unavailable, returns a safe placeholder so the
    caller never crashes.

    Provider selection:
      - Explicit `provider` argument wins (case insensitive, accepts
        aliases like "gpt" or "claude").
      - When unset, the function falls back to Hugging Face if a token is
        available, otherwise to the local Llama backend.
      - Provider-specific keys (`openai_api_key`, `anthropic_api_key`,
        `hf_token`) are session-scoped and never persisted by this module.
    """
    if not skip_preprocessing:
        skip_preprocessing = os.environ.get("TRIAGE_LLM_RAW_TEXT", "0") == "1"

    hf_model_resolved, hf_token_resolved, hf_available = _resolve_hf_settings(
        hf_model, hf_token
    )

    requested_provider = _normalize_provider(provider)
    if requested_provider == "openai":
        provider_choice = "openai"
    elif requested_provider == "anthropic":
        provider_choice = "anthropic"
    elif requested_provider == "local":
        provider_choice = "local"
    elif requested_provider == "huggingface":
        provider_choice = "huggingface"
    else:
        provider_choice = _resolve_llm_provider(provider, hf_available)

    max_gen_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS

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
        "mitre_ids list. The 'rationale' must be detailed and comprehensive (3 to 6 sentences), SOC-focused, "
        "and MUST include: 1) A thorough summary of what happened, 2) Assessment of threat severity and potential impact, "
        "3) At least 3-5 specific, actionable next steps or recommended actions for the SOC analyst with technical details. "
        "Format the rationale as: 'Summary: [detailed description]. Impact: [severity assessment]. Next steps: 1) [detailed action] 2) [detailed action] 3) [detailed action]...' "
        "Provide specific commands, log locations, or investigation techniques where applicable."
    )

    messages = [
        {"role": "system", "content": system_instructions},
        {
            "role": "user",
            "content": (
                f"Incident narrative:\n{llm_text}\n\n"
                "Return JSON ONLY (no extra commentary)."
            ),
        },
    ]

    prompt = (
        f"{system_instructions}\n\n"
        f"Incident narrative:\n{llm_text}\n\n"
        'Now respond with a single valid JSON object ONLY, with keys "label", "mitre_ids", and "rationale" for this specific incident.\n'
        "Do not include any explanations, headings, notes, or examples.\n"
        "Do not repeat these instructions."
    ).strip()

    data: dict | None = None

    if provider_choice == "openai":
        from .llm_client import OpenAIClient  # local import to keep cold start light

        if not openai_api_key:
            _llm_debug("OpenAI provider selected but no API key supplied.")
            return _placeholder_result(
                "OpenAI provider selected but no API key was supplied. "
                "Paste a key in the sidebar (Bring Your Own Key) to proceed."
            )
        try:
            client = OpenAIClient(
                api_key=openai_api_key,
                model=openai_model or "gpt-4o-mini",
                max_new_tokens=max_gen_tokens,
                rate_limiter=_get_provider_rate_limiter("openai"),
            )
            data = client.generate_json(prompt, max_tokens=max_gen_tokens)
            _llm_debug("OpenAI inference completed successfully.")
        except Exception as exc:  # pragma: no cover - network dependent
            _llm_debug(f"OpenAI inference failed: {exc!r}")
            return _placeholder_result(
                f"OpenAI inference failed: {exc}. Check the API key, model id, and quota."
            )

    elif provider_choice == "anthropic":
        from .llm_client import AnthropicClient

        if not anthropic_api_key:
            _llm_debug("Anthropic provider selected but no API key supplied.")
            return _placeholder_result(
                "Anthropic provider selected but no API key was supplied. "
                "Paste a key in the sidebar (Bring Your Own Key) to proceed."
            )
        try:
            client = AnthropicClient(
                api_key=anthropic_api_key,
                model=anthropic_model or "claude-haiku-4-5",
                max_new_tokens=max_gen_tokens,
                rate_limiter=_get_provider_rate_limiter("anthropic"),
            )
            data = client.generate_json(prompt, max_tokens=max_gen_tokens)
            _llm_debug("Anthropic inference completed successfully.")
        except Exception as exc:  # pragma: no cover - network dependent
            _llm_debug(f"Anthropic inference failed: {exc!r}")
            return _placeholder_result(
                f"Anthropic inference failed: {exc}. Check the API key, model id, and quota."
            )

    elif provider_choice == "huggingface":
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
            _llm_debug("HF provider selected but no token provided; falling back to local.")

    if data is None and provider_choice in {"local", "huggingface"}:
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
            _llm_debug("Starting LLM inference...")
            start_time = time.time()

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
                choice.get("message", {}).get("content", "") or choice.get("text", "")
            ).strip()
            _llm_debug(f"Raw LLM output: {raw_text!r}")

            text_for_json = raw_text.strip()
            if text_for_json.startswith("```"):
                lines = text_for_json.splitlines()
                if lines:
                    lines = lines[1:]
                    if lines and lines[-1].strip().startswith("```"):
                        lines = lines[:-1]
                    text_for_json = "\n".join(lines).strip()

            if text_for_json.startswith("'") and text_for_json.endswith("'"):
                candidate = text_for_json[1:-1].strip()
                if candidate.startswith("{") and candidate.endswith("}"):
                    text_for_json = candidate
            _llm_debug(f"Normalized text for JSON parsing: {text_for_json!r}")

            if not text_for_json or "[INST]" in text_for_json:
                _llm_debug(
                    "LLM output is empty or appears to be a chat template echo; "
                    "treating as invalid and falling back."
                )
                raise ValueError(f"Invalid non-JSON output from LLM: {text_for_json!r}")

            try:
                _llm_debug("Attempting direct JSON parse of LLM output.")
                data = json.loads(text_for_json)
            except Exception as parse_exc:
                _llm_debug(
                    f"Direct JSON parse failed: {parse_exc!r}; falling back to lenient extraction."
                )
                data = _lenient_extract_llm_fields(text_for_json)

        except Exception as exc:
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

    label = data.get("label", "uncertain")
    _llm_debug(f"Parsed LLM JSON: {data!r}")
    _llm_debug(f"LLM-suggested label before normalization: {label!r}")

    synonym_map = {
        "ransomware": "malware",
        "brute_force_attack": "access_abuse",
    }
    if label in synonym_map:
        canonical = synonym_map[label]
        _llm_debug(f"Normalizing LLM label {label!r} to canonical {canonical!r}.")
        label = canonical
    if label not in MITRE_MAPPING.keys() and label != "uncertain":
        label = "uncertain"

    raw_mitre_ids = data.get("mitre_ids", [])
    if not isinstance(raw_mitre_ids, list):
        raw_mitre_ids = []

    lower_text = text.lower()

    exfil_keywords = _EXFIL_KEYWORDS
    malware_keywords = _MALWARE_KEYWORDS
    web_keywords = _WEB_KEYWORDS
    access_keywords = _ACCESS_KEYWORDS
    policy_keywords = _POLICY_KEYWORDS

    def _has_any(text_lc: str, keywords: list[str]) -> bool:
        return any(k in text_lc for k in keywords)

    raw_rationale = str(data.get("rationale", "") or "")
    incident_iocs = _extract_indicators(text)
    rationale_iocs = _extract_indicators(raw_rationale)
    extra_iocs = rationale_iocs - incident_iocs
    if extra_iocs:
        _llm_debug(
            f"LLM rationale introduced new IOC-like indicators: {sorted(extra_iocs)!r}; "
            "treating as hallucinated and downgrading to 'uncertain'."
        )
        safe_label = "uncertain"
        safe_rationale = build_llm_rationale(safe_label, llm_text)
        return {
            "label": safe_label,
            "mitre_ids": [],
            "rationale": safe_rationale,
        }

    if label == "data_exfiltration" and not _has_any(lower_text, exfil_keywords):
        _llm_debug("Downgrading 'data_exfiltration' (no exfil keywords).")
        label = "uncertain"
    elif label == "malware" and not _has_any(lower_text, malware_keywords):
        _llm_debug("Downgrading 'malware' (no malware keywords).")
        label = "uncertain"
    elif label == "web_attack" and not _has_any(lower_text, web_keywords):
        _llm_debug("Downgrading 'web_attack' (no web indicators).")
        label = "uncertain"
    elif label == "access_abuse" and not _has_any(lower_text, access_keywords):
        _llm_debug("Downgrading 'access_abuse' (no identity terms).")
        label = "uncertain"
    elif label == "policy_violation" and not _has_any(lower_text, policy_keywords):
        _llm_debug("Downgrading 'policy_violation' (no policy/HR language).")
        label = "uncertain"

    if label == "phishing" and not re.search(
        r"\b(email|mailbox|inbox|message|phishing|link|url|clicked)\b",
        lower_text,
    ):
        _llm_debug("Downgrading 'phishing' (no email indicators).")
        label = "uncertain"

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
                f"Promoting 'uncertain' to heuristic label {heuristic_label!r}."
            )
            label = heuristic_label

    canonical_mitre = MITRE_MAPPING.get(label, [])
    if raw_mitre_ids:
        mitre_ids = raw_mitre_ids
    elif canonical_mitre:
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


# -----------------------------------------------------------------------------
# Per-provider rate limiters
# -----------------------------------------------------------------------------

_provider_rate_limiters: dict[str, RateLimiter] = {}


def _get_provider_rate_limiter(provider: str) -> RateLimiter:
    """Each remote provider gets its own sliding-window budget so that
    burning OpenAI quota does not lock out Anthropic or Hugging Face.
    """
    if provider not in _provider_rate_limiters:
        _provider_rate_limiters[provider] = RateLimiter(
            max_requests=HF_RATE_LIMIT_MAX,
            window_seconds=HF_RATE_LIMIT_WINDOW,
        )
    return _provider_rate_limiters[provider]


def _placeholder_result(rationale: str) -> dict:
    return {
        "label": "uncertain",
        "mitre_ids": [],
        "rationale": rationale,
    }


# -----------------------------------------------------------------------------
# Keyword tables (kept private; only used by llm_second_opinion)
# -----------------------------------------------------------------------------

_EXFIL_KEYWORDS = [
    "exfil", "exfiltration", "data exfil", "data leak", "data theft",
    "download", "downloaded", "upload", "uploaded", "transfer",
    "transferred", "copied", "moved", "synced", "synchronized",
    "archive", "archived", "compressed", "zip", "tar.gz", "7z",
    "export", "exported", "dump", "database dump", "db dump",
    "dropbox", "google drive", "gdrive", "onedrive", "box.com",
    "box drive", "sharefile", "sharepoint", "share point",
    "wetransfer", "mega.nz", "mega.io", "cloud storage",
    "object storage", "s3", "s3 bucket", "ftp", "sftp", "scp",
    "rsync", "rclone", "usb", "thumb drive", "flash drive",
    "removable media", "external drive", "external disk",
    "burned to dvd", "sent to personal email", "personal email account",
    "gmail.com", "yahoo.com", "outlook.com", "protonmail",
    "forwarded externally", "emailed externally", "sent outside organization",
]

_MALWARE_KEYWORDS = [
    "malware", "ransomware", "trojan", "virus", "worm", "backdoor",
    "remote access trojan", "rat", "infostealer", "info stealer",
    "keylogger", "key logger", "spyware", "adware", "crypto-miner",
    "cryptominer", "coinminer", "malicious payload", "payload dropped",
    "dropped file", "suspicious process", "unknown binary",
    "unsigned binary", "persistence", "autorun", "runkey",
    "scheduled task", "schtasks.exe", "registry run key",
    "dll sideloading", "sideloading", "code injection", "shellcode",
    "beacon", "c2", "command and control", "callback domain",
    "powershell", "powershell.exe", "wscript.exe", "cscript.exe",
    "mshta.exe", "rundll32.exe", "regsvr32.exe", "living off the land",
    "lolbin", "ransom", "ransom note", "decrypt", "decryptor",
    "encrypting", "encrypted", "encryption", "files renamed",
    "file extension changed", "remote access tool",
    "remote administration tool", "unapproved remote access",
    "unauthorized remote access", "screen sharing tool",
    "remote desktop tool", "edr alert", "edr detection", "av alert",
    "antivirus alert", "detected malware", "blocked malware",
    "malicious hash", "malicious executable",
]

_WEB_KEYWORDS = [
    "web application", "web app", "web server", "website", "portal",
    "api endpoint", "rest api", "graphql", "http", "https", "url path",
    "endpoint", "uri", "apache", "nginx", "iis", "tomcat",
    "reverse proxy", "load balancer", "waf", "web application firewall",
    "webshell", "web shell", "file upload handler", "upload handler",
    "sql injection", "sql-injection", "sqli", "xss",
    "cross-site scripting", "csrf", "cross-site request forgery",
    "ssrf", "server-side request forgery", "lfi", "rfi",
    "path traversal", "http flood", "layer 7 ddos", "ddos",
    "denial of service", "distributed denial-of-service",
    "spike in http requests", "excessive http requests",
    "botnet traffic", "suspicious user agents", "/login", "/signin",
    "/auth", "login page", "authentication endpoint",
]

_ACCESS_KEYWORDS = [
    "unauthorized", "unauthorised", "suspicious login", "suspicious logon",
    "login", "logon", "sign-in", "signin", "authentication",
    "auth failure", "failed login", "failed logon", "failed authentication",
    "account", "user account", "service account", "privileged account",
    "admin account", "credential", "credentials", "password", "passphrase",
    "password reset", "password change", "password spray", "brute force",
    "dictionary attack", "credential stuffing", "compromised credentials",
    "mfa", "multi-factor", "otp", "one-time passcode", "sso",
    "single sign-on", "okta", "entra id", "azure ad", "pingfederate",
    "ping federate", "duo", "vpn", "remote access vpn", "citrix",
    "rdp", "remote desktop", "beyondtrust", "privilege", "role",
    "entitlement", "elevated rights", "access", "session",
    "session hijack", "account lockout", "locked out",
    "disabled account", "new account created", "suspicious account creation",
]

_POLICY_KEYWORDS = [
    "policy", "corporate policy", "company policy", "policy violation",
    "policy breach", "violated policy", "acceptable use",
    "acceptable use policy", "aup", "code of conduct", "code-of-conduct",
    "data handling standard", "information security policy", "hr",
    "human resources", "compliance", "governance", "grc", "legal",
    "insider risk", "misuse of resources", "misuse of company resources",
    "inappropriate content", "inappropriate use", "shadow it",
    "unsanctioned application", "unsanctioned cloud service",
    "dlp alert", "data loss prevention", "classified data",
    "sensitive data", "confidential data", "handling of pii",
    "handling of phi", "hr case opened", "hr investigation",
    "written warning", "disciplinary action",
]


__all__ = [
    "MITRE_MAPPING",
    "soc_triage_hint",
    "build_llm_rationale",
    "llm_second_opinion",
    "get_llm",
    "LLM_MODEL_PATH",
    "LLM_CTX_SIZE",
    "LLM_MAX_TOKENS",
    "LLM_TEMP",
    "HF_DEFAULT_MODEL",
    "HF_ENDPOINT",
    "HF_TOKEN_ENV",
    "HF_RATE_LIMIT_MAX",
    "HF_RATE_LIMIT_WINDOW",
    "LLM_DEBUG",
]
