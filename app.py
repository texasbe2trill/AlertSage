"""AlertSage SOC console.

A focused SIEM-style frontend for the AlertSage incident triage stack.
The visual language is modeled on production SOC consoles (Splunk
Enterprise Security, Elastic Security): dark-mode-first, dense, severity
as the primary color signal. All styling lives in `assets/styles.css`.

The classifier, embedder, database, and LLM helpers are imported from
`src/triage/` and reused through cached wrappers.
"""

from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover - py3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found]

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)
# setLevel alone is not enough: Python's root logger has no INFO-level
# handler by default and Streamlit only configures handlers under its
# own 'streamlit.*' logger namespace. Without a handler attached to
# OUR logger (or to root), every logger.info call propagates and gets
# silently dropped, which is what hid the LLM-path diagnostics during
# the batch debug session. Attaching a StreamHandler here writes to
# stderr, which streamlit's nohup redirect captures into the
# /tmp/alertsage-streamlit.log file we tail for diagnostics.
if not logger.handlers:
    _diag_handler = logging.StreamHandler()
    _diag_handler.setLevel(logging.INFO)
    _diag_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.addHandler(_diag_handler)
    # Stop double-printing if Streamlit ever adds a root handler later.
    logger.propagate = False
logger.setLevel(logging.INFO)

from src.triage.database import TriageDatabase
from src.triage.embeddings import get_embedder
from src.triage.hunt_query import (
    FIELDS as HUNT_FIELDS,
    ParseError as HuntParseError,
    compile_query as compile_hunt_query,
    field_spec as hunt_field_spec,
)
from src.triage.llm_client import list_anthropic_models, list_openai_models
from src.triage.llm_helpers import (
    LLM_ASSIST_FALLBACK,
    LLM_ASSIST_MODES,
    LLM_ASSIST_OFF,
    LLM_ASSIST_OVERRIDE,
    MITRE_MAPPING,
    apply_llm_override,
    build_llm_rationale,
    effective_rate_window,
    llm_second_opinion,
    soc_triage_hint,
    with_forced_fallback,
)
from src.triage.model import load_vectorizer_and_model
from src.triage.preprocess import clean_description


# =============================================================================
# PAGE CONFIG  (must come before any other st.* call)
# =============================================================================

_LOGO_PATH = Path(__file__).parent / "assets" / "icons" / "alertsage-logo.svg"
_FAVICON = "🛡"
if _LOGO_PATH.exists():
    _logo_b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
    _FAVICON = f"data:image/svg+xml;base64,{_logo_b64}"

st.set_page_config(
    page_title="AlertSage SOC",
    page_icon=_FAVICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# DESIGN SYSTEM
# =============================================================================

_STYLES_FILE = Path(__file__).parent / "assets" / "styles.css"
if _STYLES_FILE.exists():
    st.markdown(
        f"<style>{_STYLES_FILE.read_text()}</style>",
        unsafe_allow_html=True,
    )


# =============================================================================
# CONSTANTS
# =============================================================================

NAV_ITEMS = [
    ("overview",   "Overview",    "Mission control"),
    ("investigate","Investigate", "Triage one incident"),
    ("hunt",       "Hunt",        "Search history"),
    ("batch",      "Batch",       "Bulk analysis"),
    ("bookmarks",  "Bookmarks",   "Saved investigations"),
    ("settings",   "Settings",    "Providers and profiles"),
]

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5"
HF_DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct:cerebras"
UI_LLM_MAX_TOKENS = 512
UI_LLM_MAX_INPUT_CHARS = 8000
# Per-provider rate limits now live in llm_helpers.effective_rate_window
# (BYOK gets 60/60s, bundled HF demo gets the original 5/60s, local
# llama.cpp is effectively unlimited). The constants below are kept
# only as the shared-quota defaults used when the diagnostic caption
# wants a labeled "demo" number to show; the live cap comes from
# _provider_rate_window per call.
RATE_LIMIT_REQS = 5
RATE_LIMIT_WINDOW_S = 60

# Map of canonical triage labels to the severity tier surfaced in the UI.
LABEL_SEVERITY = {
    "data_exfiltration":  "critical",
    "malware":            "critical",
    "credential_compromise": "high",
    "access_abuse":       "high",
    "web_attack":         "high",
    "phishing":           "medium",
    "policy_violation":   "medium",
    "suspicious_network_activity": "medium",
    "insider_threat":     "medium",
    "benign_activity":    "low",
    "uncertain":          "info",
}

LABEL_DESCRIPTIONS = {
    "data_exfiltration":  "Data leaving the environment via network, cloud, email, or removable media.",
    "malware":            "Malicious code execution, persistence, or command and control activity.",
    "credential_compromise": "Indicators of credential theft, replay, or unauthorized authentication.",
    "access_abuse":       "Anomalous identity behavior, brute force, or privilege misuse.",
    "web_attack":         "Application-layer attack patterns against web infrastructure.",
    "phishing":           "Social engineering targeting users via email or messaging.",
    "policy_violation":   "Acceptable-use, DLP, or compliance policy breach.",
    "suspicious_network_activity": "Anomalous lateral movement, scanning, or beaconing.",
    "insider_threat":     "Authorized user acting against the organization's interests.",
    "benign_activity":    "No malicious activity identified.",
    "uncertain":          "Insufficient signal to commit to a label; analyst review required.",
}

# MITRE ATT&CK kill chain (Enterprise tactics, in order). Each entry maps
# to the technique IDs that the AlertSage classifier surfaces. The viz on
# the Investigate page lights up cells whose tactic is implicated by the
# techniques returned for the current incident.
KILL_CHAIN_STAGES: list[tuple[str, str, list[str]]] = [
    ("TA0043", "Reconnaissance",      ["T1595", "T1598"]),
    ("TA0001", "Initial Access",      ["T1566", "T1190", "T1078"]),
    ("TA0002", "Execution",           ["T1059", "T1059.007", "T1204", "T1053"]),
    ("TA0003", "Persistence",         ["T1556", "T1098", "T1547"]),
    ("TA0004", "Privilege Escalation",["T1068", "T1078"]),
    ("TA0005", "Defense Evasion",     ["T1027", "T1070", "T1036"]),
    ("TA0006", "Credential Access",   ["T1110", "T1539", "T1528", "T1556"]),
    ("TA0007", "Discovery",           ["T1087", "T1082", "T1046", "T1057"]),
    ("TA0008", "Lateral Movement",    ["T1021", "T1570"]),
    ("TA0009", "Collection",          ["T1213", "T1530", "T1119"]),
    ("TA0011", "Command & Control",   ["T1071", "T1573", "T1105"]),
    ("TA0010", "Exfiltration",        ["T1041", "T1048", "T1567", "T1020", "T1052"]),
    ("TA0040", "Impact",              ["T1486", "T1490", "T1499"]),
]

# A small mock threat-intel feed surfaced on Overview. In production this
# would come from a TAXII collection or a vendor feed; we ship a static
# list so the showcase has something to point at.
THREAT_FEED: list[dict[str, str]] = [
    {
        "ts":     "2026-04-30 09:14",
        "source": "CISA",
        "tag":    "ransomware",
        "title":  "Akira ransomware exploiting SonicWall SSL VPN",
        "ref":    "AA26-120A",
    },
    {
        "ts":     "2026-04-29 16:42",
        "source": "MS-ISAC",
        "tag":    "phishing",
        "title":  "Phishing campaign impersonating DocuSign with OAuth abuse",
        "ref":    "TLP:AMBER",
    },
    {
        "ts":     "2026-04-29 11:08",
        "source": "Mandiant",
        "tag":    "infostealer",
        "title":  "Lumma Stealer v4 distributed via fake browser update",
        "ref":    "M-ALERT-3148",
    },
    {
        "ts":     "2026-04-28 20:19",
        "source": "AlienVault OTX",
        "tag":    "c2",
        "title":  "Cobalt Strike beacons hosted on Cloudflare R2 buckets",
        "ref":    "OTX-PULSE-22481",
    },
    {
        "ts":     "2026-04-28 07:55",
        "source": "Microsoft",
        "tag":    "0day",
        "title":  "CVE-2026-1142 SharePoint deserialization, exploited in the wild",
        "ref":    "MSRC-13921",
    },
]

# Maps the curated EXAMPLE_INCIDENTS keys to the canonical taxonomy
# label so seed_historical_events can synthesize realistic-looking
# events without invoking the classifier on every row (which would
# block the request thread long enough to trip the WebSocket
# heartbeat).
EXAMPLE_LABEL_MAP: dict[str, str] = {
    "Phishing":          "phishing",
    "Data exfiltration": "data_exfiltration",
    "Malware":           "malware",
    "Access abuse":      "access_abuse",
    "Web attack":        "web_attack",
    "Benign activity":   "benign_activity",
}


EXAMPLE_INCIDENTS = {
    "Phishing": (
        "Multiple users in the finance department received an email "
        "purporting to be from the CFO requesting urgent invoice payment "
        "processing. The link redirects to a credential harvesting page "
        "imitating Microsoft 365 login. Three users reported clicking the "
        "link and one entered credentials before noticing the URL."
    ),
    "Data exfiltration": (
        "User account jdoe@corp uploaded 2.3 GB of compressed archives to "
        "a personal Dropbox account between 23:14 and 23:42 UTC. Files "
        "originated from the legal/M&A folder which is classified "
        "Confidential. The user accessed the share for the first time at "
        "23:09 from a VPN session."
    ),
    "Malware": (
        "EDR alerted on encoded PowerShell execution from "
        "WINWORD.EXE on host FIN-WS-042. Process tree shows mshta.exe "
        "spawning rundll32.exe with an unsigned DLL dropped to "
        "C:\\\\Users\\\\Public\\\\update.dll. Outbound TLS to "
        "85.193.14.221:443 with non-standard SNI. Hash matches a known "
        "Emotet variant."
    ),
    "Access abuse": (
        "Service account svc-backup attempted 412 Kerberos pre-auth "
        "failures across 17 domain controllers within 90 seconds, then "
        "successfully authenticated against DC-EAST-03. Source: jump host "
        "10.34.12.5 which is normally idle on weekends."
    ),
    "Web attack": (
        "WAF blocked 3,200 requests against the customer portal login "
        "endpoint over 10 minutes. Payloads contain SQL injection probes "
        "(' OR 1=1 -- and UNION SELECT variants). Source IPs cluster in "
        "two ASNs known for residential proxy services."
    ),
    "Benign activity": (
        "Routine software update package deployed to 200 endpoints "
        "flagged by scanner due to expected registry changes under "
        "HKLM\\\\Software\\\\Microsoft\\\\Updates. Maintenance window "
        "approved by change request CHG-2024-1031."
    ),
}


# =============================================================================
# RESOURCE CACHES
# =============================================================================

@st.cache_resource(show_spinner=False)
def _classifier():
    return load_vectorizer_and_model()


@st.cache_resource(show_spinner=False)
def _embedder():
    return get_embedder()


@st.cache_resource(show_spinner=False)
def _db() -> TriageDatabase:
    return TriageDatabase()


@st.cache_resource(show_spinner=False)
def local_gguf_available() -> bool:
    """True when llama-cpp-python imports AND a .gguf file exists.

    Used to decide whether the Local provider option is shown.
    """
    try:
        import llama_cpp  # type: ignore  # noqa: F401
    except Exception:
        return False
    models_dir = Path(__file__).parent / "models"
    if not models_dir.is_dir():
        return False
    try:
        return any(p.suffix.lower() == ".gguf" for p in models_dir.iterdir())
    except OSError:
        return False


# =============================================================================
# SESSION-STATE BOOTSTRAP
# =============================================================================

# Hosted demo detection. Setting `IS_HOSTED_DEMO=1` (or any truthy
# value) in Streamlit Community Cloud secrets / env flips the console
# into demo-friendly defaults: auto-seed history on cold start, demo
# generator on by default, slightly more chatty captions. Local runs
# stay quiet.
def _is_hosted_demo() -> bool:
    if os.environ.get("IS_HOSTED_DEMO"):
        return True
    if os.environ.get("STREAMLIT_SHARING_MODE"):
        return True
    return False


_DEFAULTS: dict[str, Any] = {
    "view": "overview",
    "selected_bookmark": None,
    "current_analysis": None,
    "investigate_text": "",
    # LLM provider state (kept in session only; never persisted)
    "llm_provider": None,
    # use_llm is the legacy boolean; llm_assist_mode is the 3-way
    # selector that supersedes it (off / fallback / override). The
    # boolean is derived from the mode at sidebar render time so any
    # legacy reads keep working.
    "use_llm": True,
    "llm_assist_mode": LLM_ASSIST_FALLBACK,
    "hf_model_id": HF_DEFAULT_MODEL,
    "selected_hf_token": "",
    "hf_byo_token": False,
    "openai_model_id": DEFAULT_OPENAI_MODEL,
    "selected_openai_api_key": "",
    "openai_byo_key": False,
    "anthropic_model_id": DEFAULT_ANTHROPIC_MODEL,
    "selected_anthropic_api_key": "",
    "anthropic_byo_key": False,
    # Triage knobs
    "threshold": 0.50,
    "max_classes": 5,
    "use_preprocessing": True,
    # Demo generator: off by default everywhere. Was previously on for
    # the hosted demo, but that auto-mounted an every-8s rerun fragment
    # which interacted badly with sidebar rendering on newer Streamlit
    # versions during Cloud cold start. Visitors still land on a
    # populated dashboard (ensure_demo_data_seeded handles that on cold
    # start). The live tail just stops growing automatically; users who
    # want it can toggle it in Settings.
    "demo_generator_on": False,
}
for _key, _default in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# =============================================================================
# SECRETS / ENV HELPERS
# =============================================================================

_SECRETS_PATHS = [
    Path.cwd() / ".streamlit" / "secrets.toml",
    Path.home() / ".streamlit" / "secrets.toml",
]
_SECRETS_CACHE: dict[str, Any] | None = None


def _load_secrets() -> dict[str, Any]:
    """Read .streamlit/secrets.toml directly.

    We bypass `st.secrets` because the Streamlit API surfaces a noisy
    "No secrets found" toast on every read when the file is missing; here
    we just return an empty dict and move on.
    """
    global _SECRETS_CACHE
    if _SECRETS_CACHE is not None:
        return _SECRETS_CACHE
    for path in _SECRETS_PATHS:
        try:
            if path.is_file():
                _SECRETS_CACHE = tomllib.loads(path.read_text())
                return _SECRETS_CACHE
        except Exception:
            _SECRETS_CACHE = {}
            return _SECRETS_CACHE
    _SECRETS_CACHE = {}
    return _SECRETS_CACHE


def _secret(key: str, default: str = "") -> str:
    return str(_load_secrets().get(key, default)).strip()


def _env(*names: str) -> str:
    for n in names:
        val = os.environ.get(n)
        if val:
            return val
    return ""


# =============================================================================
# LLM PROVIDER PLUMBING
# =============================================================================

def _should_invoke_llm(result: dict, mode: str, threshold: float) -> bool:
    """Decide whether the LLM should run for this single classifier result.

    - off: never call.
    - override: always call. The LLM classifies every event; sklearn
      stays as the fast pre-pass that we keep for diagnostics and as
      the fallback when the LLM declines or errors.
    - fallback: call only when sklearn looks shaky. 'Shaky' means the
      label landed on 'uncertain' OR the top-class probability barely
      crossed the threshold (within +0.1). The cushion catches the
      'just over the line so labeled but probably wrong' cases that
      would otherwise sail past the existing 'uncertain'-only check.
    """
    if mode == LLM_ASSIST_OFF:
        logger.info("LLM gate: SKIP mode=off")
        return False
    if mode == LLM_ASSIST_OVERRIDE:
        logger.info(
            "LLM gate: INVOKE mode=override label=%s conf=%.3f",
            result.get("final_label"),
            float(result.get("max_prob") or 0.0),
        )
        return True
    label = result.get("final_label")
    max_prob = float(result.get("max_prob") or 0.0)
    invoke = label == "uncertain" or max_prob < (threshold + 0.1)
    logger.info(
        "LLM gate: %s mode=fallback label=%s conf=%.3f cushion_threshold=%.3f",
        "INVOKE" if invoke else "SKIP",
        label,
        max_prob,
        threshold + 0.1,
    )
    return invoke


def _resolve_llm_settings() -> dict[str, Any]:
    """Snapshot the provider configuration from session/secrets/env.

    All three hosted providers fall back through the same chain:
      1. Streamlit session state (BYOK fields in Settings)
      2. .streamlit/secrets.toml
      3. environment variables
    Whichever yields a non-empty value wins. This lets a developer drop
    a token in their shell and have the recorder / a local run light
    up without typing anything into the BYOK panel.
    """
    hf_secret_token = _secret("HF_TOKEN")
    hf_secret_model = _secret("HF_MODEL")
    hf_env_token = _env("TRIAGE_HF_TOKEN", "HF_TOKEN")
    hf_env_model = _env("TRIAGE_HF_MODEL", "HF_MODEL")

    openai_secret_key = _secret("OPENAI_API_KEY")
    openai_secret_model = _secret("OPENAI_MODEL")
    openai_env_key = _env("OPENAI_API_KEY")
    openai_env_model = _env("OPENAI_MODEL")

    anthropic_secret_key = _secret("ANTHROPIC_API_KEY")
    anthropic_secret_model = _secret("ANTHROPIC_MODEL")
    anthropic_env_key = _env("ANTHROPIC_API_KEY")
    anthropic_env_model = _env("ANTHROPIC_MODEL")

    return {
        "provider": st.session_state.get("llm_provider") or _default_provider(),
        "hf_model": (
            st.session_state.get("hf_model_id")
            or hf_secret_model or hf_env_model or HF_DEFAULT_MODEL
        ),
        "hf_token": (
            st.session_state.get("selected_hf_token") or hf_secret_token
            or hf_env_token or ""
        ),
        "openai_model": (
            st.session_state.get("openai_model_id")
            or openai_secret_model or openai_env_model or DEFAULT_OPENAI_MODEL
        ),
        "openai_api_key": (
            st.session_state.get("selected_openai_api_key")
            or openai_secret_key or openai_env_key or ""
        ),
        "anthropic_model": (
            st.session_state.get("anthropic_model_id")
            or anthropic_secret_model or anthropic_env_model
            or DEFAULT_ANTHROPIC_MODEL
        ),
        "anthropic_api_key": (
            st.session_state.get("selected_anthropic_api_key")
            or anthropic_secret_key or anthropic_env_key or ""
        ),
    }


def _default_provider() -> str:
    """Pick a sensible default provider on first load.

    Priority order matches typical preference: Anthropic > OpenAI > HF >
    local. The first provider with a discoverable token in secrets or
    env wins, so a recorder run with only ANTHROPIC_API_KEY exported
    routes there automatically without the user having to flip the
    radio in Settings.
    """
    if _secret("ANTHROPIC_API_KEY") or _env("ANTHROPIC_API_KEY"):
        return "anthropic"
    if _secret("OPENAI_API_KEY") or _env("OPENAI_API_KEY"):
        return "openai"
    if _secret("HF_TOKEN") or _env("TRIAGE_HF_TOKEN", "HF_TOKEN"):
        return "huggingface"
    return "local" if local_gguf_available() else "huggingface"


def _build_llm_kwargs(settings: dict[str, Any]) -> dict[str, Any]:
    """Translate settings into kwargs for llm_second_opinion.

    Implements the "Bring Your Own Key, fall back to demo" policy:
      OpenAI/Anthropic without a key, but HF token present, route to HF.
      Provider 'local' on a host without llama_cpp falls back to HF.
    """
    provider = settings.get("provider", "local")
    local_ok = local_gguf_available()

    def _fallback() -> str:
        if settings.get("hf_token"):
            return "huggingface"
        return "local" if local_ok else "huggingface"

    if provider == "openai" and not settings.get("openai_api_key"):
        provider = _fallback()
    elif provider == "anthropic" and not settings.get("anthropic_api_key"):
        provider = _fallback()
    if provider == "local" and not local_ok:
        provider = "huggingface"

    kwargs: dict[str, Any] = {"provider": provider}
    if provider == "huggingface":
        kwargs["hf_model"] = settings.get("hf_model")
        kwargs["hf_token"] = settings.get("hf_token")
    elif provider == "openai":
        kwargs["openai_model"] = settings.get("openai_model")
        kwargs["openai_api_key"] = settings.get("openai_api_key")
    elif provider == "anthropic":
        kwargs["anthropic_model"] = settings.get("anthropic_model")
        kwargs["anthropic_api_key"] = settings.get("anthropic_api_key")
    return kwargs


def _byok_present(provider: str) -> bool:
    """Has the user pasted their own API key for this provider?

    Reads only session_state, never secrets/env, because BYOK status
    is what determines who pays for the call. A bundled secret token
    is shared across every visitor and stays under the modest demo
    cap; a session-state key is the user's own quota and gets the
    higher BYOK cap.
    """
    if provider == "openai":
        return bool(st.session_state.get("selected_openai_api_key"))
    if provider == "anthropic":
        return bool(st.session_state.get("selected_anthropic_api_key"))
    if provider == "huggingface":
        return bool(st.session_state.get("selected_hf_token"))
    return False


def _provider_rate_window(provider: str) -> tuple[int, int]:
    """Live (cap, window_seconds) for the current session and provider."""
    return effective_rate_window(provider, byok_present=_byok_present(provider))


def _provider_rate_check(provider: str) -> tuple[bool, float]:
    """Per-provider sliding-window rate limiter (session scoped).

    The cap and window come from _provider_rate_window so BYOK calls
    are not throttled at the demo-fallback rate. The previous global
    5/60s starved batch runs against hosted providers; see the
    earlier user-reported 'rate-limited: 20 of 25' diagnostic.
    """
    cap, window = _provider_rate_window(provider)
    bucket_key = f"_rl_{provider}"
    now = datetime.now(timezone.utc).timestamp()
    window_start = now - window
    timestamps = [t for t in st.session_state.get(bucket_key, []) if t >= window_start]
    if len(timestamps) >= cap:
        retry_after = window - (now - timestamps[0])
        st.session_state[bucket_key] = timestamps
        return False, max(retry_after, 0.0)
    timestamps.append(now)
    st.session_state[bucket_key] = timestamps
    return True, 0.0


def run_llm_second_opinion(
    text: str,
    *,
    skip_preprocessing: bool = False,
    force_classification: bool = False,
) -> tuple[dict | None, str | None]:
    """Single dispatch helper used by every page that calls the LLM.

    force_classification swaps the prompt so 'uncertain' is forbidden;
    used by run_llm_with_forced_fallback for the second-pass retry
    when the first pass hedged.
    """
    settings = _resolve_llm_settings()
    kwargs = _build_llm_kwargs(settings)
    provider = kwargs["provider"]
    allowed, retry_after = _provider_rate_check(provider)
    if not allowed:
        return None, (
            f"{provider.title()} rate limit reached. "
            f"Wait {retry_after:.0f}s and try again."
        )
    try:
        opinion = llm_second_opinion(
            text,
            skip_preprocessing=skip_preprocessing,
            max_tokens=UI_LLM_MAX_TOKENS,
            force_classification=force_classification,
            **kwargs,
        )
        return opinion, None
    except Exception as exc:  # pragma: no cover - network dependent
        return None, str(exc)


def run_llm_with_forced_fallback(
    text: str, *, skip_preprocessing: bool = False
) -> tuple[dict | None, str | None, dict]:
    """Rate-limited wrapper around the pure with_forced_fallback
    orchestrator. The orchestrator lives in llm_helpers.py so it can
    be unit-tested with a stub classifier; this function wires in the
    real, rate-limited classifier and emits a diagnostic log line so
    we can grep one event end-to-end in /tmp/alertsage-streamlit.log.
    """
    opinion, err, details = with_forced_fallback(
        run_llm_second_opinion,
        text,
        skip_preprocessing=skip_preprocessing,
    )
    logger.info(
        "LLM call: first_pass=%s err=%r force_attempted=%s force_pass=%s "
        "force_err=%r final_label=%s",
        details.get("first_pass_label"),
        details.get("first_pass_err"),
        details.get("force_pass_attempted"),
        details.get("force_pass_label"),
        details.get("force_pass_err"),
        opinion.get("label") if opinion else None,
    )
    return opinion, err, details


# =============================================================================
# CLASSIFIER FRONT-END
# =============================================================================

def _build_feature_matrix(cleaned_text: str, model_n_features: int):
    """Build the feature matrix the loaded classifier was trained on.

    The enhanced classifier was trained on TF-IDF (5000 dims) plus
    sentence-transformer embeddings (384 dims) concatenated horizontally
    (5384 dims total). The baseline classifier uses TF-IDF only. We
    branch on the model's `n_features_in_` so both checkpoints work.
    """
    from scipy import sparse
    vectorizer, _ = _classifier()
    X_tfidf = vectorizer.transform([cleaned_text])
    n_tfidf = X_tfidf.shape[1]
    if model_n_features == n_tfidf:
        return X_tfidf
    extra = model_n_features - n_tfidf
    if extra > 0:
        # Embedding dim should match the gap (384 for all-MiniLM-L6-v2)
        emb = _embedder().encode(cleaned_text, normalize=True)
        if hasattr(emb, "ndim") and emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if emb.shape[1] != extra:
            raise ValueError(
                f"feature mismatch: classifier expects {model_n_features} "
                f"features but TF-IDF gives {n_tfidf} and embedder gives "
                f"{emb.shape[1]}"
            )
        return sparse.hstack([X_tfidf, sparse.csr_matrix(emb)], format="csr")
    raise ValueError(
        f"classifier expects {model_n_features} features but TF-IDF alone "
        f"gives {n_tfidf}; cannot reconcile."
    )


def predict(text: str, *, threshold: float, max_classes: int) -> dict[str, Any]:
    """Run the loaded classifier and return a normalized triage result."""
    _, model = _classifier()
    cleaned = clean_description(text)
    n_features = int(getattr(model, "n_features_in_", 0))
    X = _build_feature_matrix(cleaned, n_features)
    label = model.predict(X)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        proba_dict = dict(zip(model.classes_, proba))
        sorted_probs = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
        top = sorted_probs[:max_classes]
    else:
        top = [(label, 1.0)]
    max_prob = float(top[0][1]) if top else 0.0
    if max_prob < threshold:
        final_label = "uncertain"
        uncertainty = "low" if max_prob < threshold * 0.7 else "medium"
    else:
        final_label = label
        uncertainty = "high" if max_prob >= 0.85 else "medium"
    return {
        "incident_text": text,
        "base_label": label,
        "final_label": final_label,
        "max_prob": max_prob,
        "uncertainty_level": uncertainty,
        "probabilities": top,
        "mitre_techniques": MITRE_MAPPING.get(final_label, []),
        "timestamp": datetime.now().isoformat(),
    }


# =============================================================================
# UI PRIMITIVES
# =============================================================================

def humanize(label: str) -> str:
    return (label or "uncertain").replace("_", " ").title()


def severity_for(label: str) -> str:
    return LABEL_SEVERITY.get(label, "info")


def severity_pill(label: str, *, fallback_text: str | None = None) -> str:
    """Render a severity pill for a triage label."""
    sev = severity_for(label)
    text = fallback_text or humanize(label)
    return f'<span class="soc-pill {sev}">{text}</span>'


def render_page_header(
    title: str, subtitle: str, breadcrumb: str = "", action_html: str = ""
) -> None:
    crumb_html = (
        f'<div class="soc-page__breadcrumb">{breadcrumb}</div>' if breadcrumb else ""
    )
    st.markdown(
        f"""
        <div class="soc-page">
            <div>
                {crumb_html}
                <h1 class="soc-page__title">{title}</h1>
                <p class="soc-page__subtitle">{subtitle}</p>
            </div>
            <div>{action_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str, *, sub: str = "", tone: str = "info",
               trend: str = "") -> str:
    trend_cls = "flat"
    trend_html = ""
    if trend:
        if trend.startswith("+"):
            trend_cls = "up"
        elif trend.startswith("-"):
            trend_cls = "down"
        trend_html = f'<div class="soc-kpi__sub {trend_cls}">{trend}</div>'
    elif sub:
        trend_html = f'<div class="soc-kpi__sub flat">{sub}</div>'
    return (
        f'<div class="soc-kpi tone-{tone} fade-in">'
        f'<div class="soc-kpi__label">{label}</div>'
        f'<div class="soc-kpi__value">{value}</div>'
        f'{trend_html}'
        '</div>'
    )


def render_panel(title: str, body_html: str, meta: str = "") -> str:
    meta_html = f'<span class="soc-meta">{meta}</span>' if meta else ""
    return (
        f'<div class="soc-panel">'
        f'<div class="soc-panel__title">{title}{meta_html}</div>'
        f'{body_html}'
        '</div>'
    )


def render_empty(title: str, hint: str) -> None:
    st.markdown(
        f'<div class="soc-empty">'
        f'<div class="soc-empty__title">{title}</div>'
        f'<div class="soc-empty__hint">{hint}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_section_head(title: str, action: str = "") -> None:
    action_html = f'<span class="soc-section-head__action">{action}</span>' if action else ""
    st.markdown(
        f'<div class="soc-section-head">'
        f'<h2 class="soc-section-head__title">{title}</h2>'
        f'{action_html}'
        '</div>',
        unsafe_allow_html=True,
    )


def time_ago(ts: str | None) -> str:
    if not ts:
        return "n/a"
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return "n/a"
    delta = datetime.now() - dt
    if delta.days >= 1:
        return f"{delta.days}d"
    if delta.seconds >= 3600:
        return f"{delta.seconds // 3600}h"
    if delta.seconds >= 60:
        return f"{delta.seconds // 60}m"
    return f"{max(delta.seconds, 1)}s"


# =============================================================================
# PLOTLY THEMING
# =============================================================================

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", size=11, color="#cbd5e1"),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(showgrid=False, color="#94a3b8", linecolor="#1f2a44"),
    yaxis=dict(showgrid=True, gridcolor="#1f2a44", color="#94a3b8"),
)
SEVERITY_COLOR_HEX = {
    "critical": "#ef4444",
    "high":     "#f97316",
    "medium":   "#eab308",
    "low":      "#22c55e",
    "info":     "#06b6d4",
}


# =============================================================================
# KILL CHAIN RENDERER
# =============================================================================

def render_kill_chain(active_techniques: list[str]) -> str:
    """Render an HTML horizontal kill chain visualization.

    Stages whose technique set intersects `active_techniques` are
    highlighted with the accent color and a count badge. Inactive
    stages render dimmed so the analyst can see the full chain context.

    The active set is expanded to include parent technique IDs of any
    subtechniques so e.g. an LLM that returns "T1566.001" still lights
    up the same cell as a bare "T1566". Without this normalization,
    LLMs that prefer subtechnique granularity produced a "3 techniques
    mapped" caption with zero visible hits.
    """
    # Two sets: `original_set` is what the caption counts (so the user
    # sees "3 techniques mapped" when the LLM returned 3 IDs), and
    # `match_set` includes parent IDs of any subtechniques so cell hit
    # detection still fires for "T1566.001".
    original_set: set[str] = set()
    match_set: set[str] = set()
    for t in active_techniques or []:
        normalized = (t or "").upper().strip()
        if not normalized:
            continue
        original_set.add(normalized)
        match_set.add(normalized)
        if "." in normalized:
            match_set.add(normalized.split(".", 1)[0])

    cells = []
    for tactic_id, tactic_name, techs in KILL_CHAIN_STAGES:
        hits = [t for t in techs if t.upper() in match_set]
        is_hit = bool(hits)
        cell_cls = "soc-kc__cell hit" if is_hit else "soc-kc__cell"
        chips = "".join(
            f'<span class="soc-kc__tech">{t}</span>' for t in hits
        ) if hits else '<span class="soc-kc__tech soc-kc__tech--idle">--</span>'
        cells.append(
            f'<div class="{cell_cls}">'
            f'<div class="soc-kc__id">{tactic_id}</div>'
            f'<div class="soc-kc__name">{tactic_name}</div>'
            f'<div class="soc-kc__row">{chips}</div>'
            '</div>'
        )
    return (
        '<div class="soc-panel">'
        '<div class="soc-panel__title">Kill chain · MITRE ATT&CK '
        f'<span class="soc-meta">{len(original_set)} technique'
        f'{"s" if len(original_set) != 1 else ""} mapped</span></div>'
        f'<div class="soc-kc">{"".join(cells)}</div>'
        '</div>'
    )


# =============================================================================
# CHARTS: MITRE HEATMAP, CONFIDENCE HISTOGRAM
# =============================================================================

def _mitre_heatmap_figure(history: list[dict]):
    """Build a heatmap of (tactic, technique) cell density across history."""
    counts: dict[tuple[str, str], int] = {}
    technique_to_tactic: dict[str, str] = {}
    for tactic_id, tactic_name, techs in KILL_CHAIN_STAGES:
        for t in techs:
            technique_to_tactic[t.upper()] = tactic_name

    for h in history:
        for t in MITRE_MAPPING.get(h.get("final_label", ""), []):
            tactic = technique_to_tactic.get(t.upper(), "Unmapped")
            key = (tactic, t)
            counts[key] = counts.get(key, 0) + 1

    if not counts:
        return None

    tactics_in_order = [name for _, name, _ in KILL_CHAIN_STAGES]
    if any(tactic == "Unmapped" for tactic, _ in counts.keys()):
        tactics_in_order.append("Unmapped")

    techniques = sorted({tech for _, tech in counts.keys()})

    z = []
    text = []
    for tactic in tactics_in_order:
        row_z = []
        row_text = []
        for tech in techniques:
            v = counts.get((tactic, tech), 0)
            row_z.append(v)
            row_text.append(f"{v}" if v else "")
        z.append(row_z)
        text.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z,
        x=techniques,
        y=tactics_in_order,
        text=text,
        texttemplate="%{text}",
        textfont=dict(family="JetBrains Mono", size=10, color="#f8fafc"),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z} events<extra></extra>",
        colorscale=[
            [0.0, "rgba(59, 130, 246, 0.05)"],
            [0.25, "rgba(59, 130, 246, 0.25)"],
            [0.5, "rgba(59, 130, 246, 0.55)"],
            [0.75, "rgba(239, 68, 68, 0.65)"],
            [1.0, "rgba(239, 68, 68, 0.95)"],
        ],
        showscale=False,
        zmin=0,
    ))
    fig.update_layout(
        height=max(220, 24 * len(tactics_in_order) + 80),
        **{
            **PLOT_LAYOUT,
            "xaxis": dict(
                showgrid=False,
                color="#94a3b8",
                tickangle=-30,
                tickfont=dict(family="JetBrains Mono", size=10),
            ),
            "yaxis": dict(
                showgrid=False,
                color="#cbd5e1",
                autorange="reversed",
                tickfont=dict(size=11),
            ),
            "margin": dict(l=10, r=10, t=10, b=40),
        },
    )
    return fig


def _confidence_histogram_figure(history: list[dict]):
    confidences = []
    for h in history:
        try:
            confidences.append(float(h.get("max_prob") or 0))
        except Exception:
            pass
    if not confidences:
        return None
    bins = np.linspace(0, 1, 21)
    counts, edges = np.histogram(confidences, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = []
    for c in centers:
        if c >= 0.8:
            colors.append("#22c55e")
        elif c >= 0.6:
            colors.append("#eab308")
        elif c >= 0.4:
            colors.append("#f97316")
        else:
            colors.append("#ef4444")
    fig = go.Figure(go.Bar(
        x=centers,
        y=counts,
        marker=dict(color=colors),
        hovertemplate="%{x:.0%} bucket<br>%{y} events<extra></extra>",
    ))
    fig.update_layout(
        height=180,
        bargap=0.08,
        **{
            **PLOT_LAYOUT,
            "xaxis": dict(
                showgrid=False, color="#94a3b8", tickformat=".0%",
                range=[0, 1],
            ),
            "yaxis": dict(
                showgrid=True, gridcolor="#1f2a44", color="#94a3b8",
                rangemode="tozero",
            ),
        },
    )
    return fig


# =============================================================================
# ANOMALY SCORE
# =============================================================================

def _anomaly_score(label: str, confidence: float) -> int:
    """Cheap proxy for anomaly score (0 to 100).

    Combines uncertainty with severity and unmapped-label flags. Higher
    numbers = stranger event. This is intentionally heuristic; a future
    iteration can swap in an embedding-distance-based score.
    """
    sev = severity_for(label)
    sev_weight = {
        "critical": 30,
        "high":     20,
        "medium":   10,
        "info":     25,  # 'uncertain' counts as anomalous
        "low":      0,
    }.get(sev, 0)
    unconfidence = max(0.0, 1.0 - float(confidence or 0))
    unconfidence_pts = int(unconfidence * 60)
    label_pts = 10 if label == "uncertain" else 0
    score = min(100, sev_weight + unconfidence_pts + label_pts)
    return int(score)


def _anomaly_pill(score: int) -> str:
    if score >= 75:
        tone = "critical"
    elif score >= 50:
        tone = "high"
    elif score >= 25:
        tone = "medium"
    else:
        tone = "low"
    return f'<span class="soc-pill {tone}">{score}</span>'


# =============================================================================
# THREAT FEED + LIVE TAIL RENDERERS
# =============================================================================

def render_threat_feed() -> str:
    items = []
    for entry in THREAT_FEED:
        items.append(
            '<div class="soc-feed__item">'
            f'<div class="soc-feed__time soc-mono">{entry["ts"]} · {entry["source"]}</div>'
            f'<div class="soc-feed__title">{entry["title"]}</div>'
            '<div class="soc-feed__meta">'
            f'<span class="soc-tag accent">{entry["tag"]}</span>'
            f'<span class="soc-tag soc-mono">{entry["ref"]}</span>'
            '</div>'
            '</div>'
        )
    return (
        '<div class="soc-panel">'
        '<div class="soc-panel__title">Threat intel feed '
        '<span class="soc-meta">curated · last 72h</span></div>'
        f'<div class="soc-feed">{"".join(items)}</div>'
        '</div>'
    )


@st.fragment(run_every="8s")
def render_live_tail_fragment(n: int = 6) -> None:
    """Auto-refreshing live tail.

    Re-queries the database every 8 seconds and re-renders independently
    of the rest of the page. The user sees new triage events flow in
    without a manual refresh. Wrapped in try/except so a transient DB
    lock can't crash the page.
    """
    try:
        history = _db().get_analysis_history(limit=200) or []
        st.markdown(render_live_tail(history, n=n), unsafe_allow_html=True)
    except Exception as exc:
        _fragment_error_box("Live tail", exc)


def render_live_tail(history: list[dict], n: int = 8) -> str:
    if not history:
        return (
            '<div class="soc-panel">'
            '<div class="soc-panel__title">Live tail '
            '<span class="soc-live-dot"></span></div>'
            '<div class="soc-empty" style="padding: 1.25rem;">'
            '<div class="soc-empty__hint">Quiet on the wire. '
            'Triage an incident to start the stream.</div>'
            '</div>'
            '</div>'
        )
    rows = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)[:n]
    lines = []
    for r in rows:
        dt = _safe_dt(r.get("timestamp"))
        when = dt.strftime("%H:%M:%S") if dt else "--:--:--"
        ago = time_ago(r.get("timestamp"))
        label = r.get("final_label", "uncertain")
        sev = severity_for(label)
        body = (r.get("incident_text") or "").strip().replace("\n", " ")
        body_short = body[:96] + ("..." if len(body) > 96 else "")
        try:
            conf = float(r.get("max_prob") or 0)
        except Exception:
            conf = 0
        lines.append(
            '<div class="soc-tail__line">'
            f'<span class="soc-tail__time soc-mono">{when}</span>'
            f'<span class="soc-tail__sev soc-pill {sev}">{label.upper().replace("_", " ")}</span>'
            f'<span class="soc-tail__conf soc-mono">{conf:.0%}</span>'
            f'<span class="soc-tail__body">{body_short}</span>'
            f'<span class="soc-tail__ago soc-mono">{ago}</span>'
            '</div>'
        )
    return (
        '<div class="soc-panel">'
        '<div class="soc-panel__title">Live tail  '
        '<span class="soc-live-dot"></span>  '
        '<span class="soc-meta">latest events</span></div>'
        f'<div class="soc-tail">{"".join(lines)}</div>'
        '</div>'
    )


# =============================================================================
# CASE STATUS WORKFLOW
# =============================================================================

# Stages an analyst walks an event through. We persist these via
# db.save_setting (key = "case_status::{analysis_id}") so we never have
# to touch the schema. UI render is a horizontal stepper.
CASE_STATUSES = [
    ("new",        "New",        "info"),
    ("triaging",   "Triaging",   "medium"),
    ("contained",  "Contained",  "high"),
    ("closed",     "Closed",     "low"),
]
_CASE_STATUS_KEYS = [k for k, _, _ in CASE_STATUSES]


def _case_status_key(analysis_id: int | str) -> str:
    return f"case_status::{analysis_id}"


def get_case_status(analysis_id: int | str | None) -> str:
    """Return the persisted case status for an analysis, defaulting to 'new'."""
    if analysis_id in (None, "", 0):
        return "new"
    try:
        v = _db().get_setting(_case_status_key(analysis_id), default="new")
        return v if v in _CASE_STATUS_KEYS else "new"
    except Exception:
        return "new"


def set_case_status(analysis_id: int | str | None, status: str) -> None:
    if analysis_id in (None, "", 0) or status not in _CASE_STATUS_KEYS:
        return
    try:
        previous = get_case_status(analysis_id)
        _db().save_setting(_case_status_key(analysis_id), status)
        if previous != status:
            label_for = dict((k, l) for k, l, _ in CASE_STATUSES)
            append_timeline_event(
                analysis_id,
                "status",
                f"Status changed from <strong>{label_for.get(previous, previous)}</strong> "
                f"to <strong>{label_for.get(status, status)}</strong>.",
            )
    except Exception as exc:
        st.warning(f"Could not save status: {exc}")


def render_case_stepper(analysis_id: int | str | None, current: str) -> str:
    """Render a horizontal status stepper. Past stages are filled."""
    current = current if current in _CASE_STATUS_KEYS else "new"
    current_idx = _CASE_STATUS_KEYS.index(current)
    cells = []
    for idx, (key, label, tone) in enumerate(CASE_STATUSES):
        if idx < current_idx:
            cls = "soc-step done"
            mark = "&#10003;"
        elif idx == current_idx:
            cls = f"soc-step active tone-{tone}"
            mark = str(idx + 1)
        else:
            cls = "soc-step pending"
            mark = str(idx + 1)
        cells.append(
            f'<div class="{cls}">'
            f'<div class="soc-step__num">{mark}</div>'
            f'<div class="soc-step__label">{label}</div>'
            '</div>'
        )
    return f'<div class="soc-stepper">{"".join(cells)}</div>'


# =============================================================================
# IOC EXTRACTION + ENRICHMENT
# =============================================================================

import re as _re

_IOC_PATTERNS: list[tuple[str, str]] = [
    ("ipv4",   r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"),
    ("ipv6",   r"\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{1,4}\b"),
    ("md5",    r"\b[A-Fa-f0-9]{32}\b"),
    ("sha1",   r"\b[A-Fa-f0-9]{40}\b"),
    ("sha256", r"\b[A-Fa-f0-9]{64}\b"),
    ("url",    r"https?://[^\s,'\"<>)]+"),
    ("email",  r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("domain", r"\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+(?:com|net|org|io|gov|edu|co|biz|info|cloud|xyz|nz|ru|cn|uk|us|de|fr)\b"),
    ("cve",    r"\bCVE-\d{4}-\d{4,7}\b"),
    ("hostname", r"\b(?:[A-Z]{2,6}-(?:WS|SRV|DC|FIN|HR|EXEC|VPN|JMP|DB)-\d{2,4})\b"),
]

# Words we don't want to flag as domains
_DOMAIN_BLOCKLIST = {"e.g", "i.e", "etc"}


def extract_iocs(text: str) -> list[dict[str, str]]:
    """Extract IOC-like indicators from free text.

    Returns deduped list of {indicator, type} dicts. Order is preserved
    based on first appearance.
    """
    if not text:
        return []
    seen: set[tuple[str, str]] = set()
    results: list[dict[str, str]] = []
    for ioc_type, pattern in _IOC_PATTERNS:
        for match in _re.findall(pattern, text):
            if isinstance(match, tuple):
                match = match[0]
            indicator = str(match).strip().rstrip(".,;:")
            if not indicator:
                continue
            if ioc_type == "domain":
                # Don't flag the FQDN inside emails or URLs we already caught
                if any(indicator.lower() in r["indicator"].lower()
                       for r in results
                       if r["type"] in {"email", "url"}):
                    continue
                # Drop common false positives
                stem = indicator.split(".")[0].lower()
                if stem in _DOMAIN_BLOCKLIST:
                    continue
            key = (ioc_type, indicator.lower())
            if key in seen:
                continue
            seen.add(key)
            results.append({"indicator": indicator, "type": ioc_type})
    return results


def _mock_enrich(ioc: dict[str, str]) -> dict[str, Any]:
    """Deterministic mock enrichment (so the same indicator yields the
    same enrichment between renders).

    A real deployment would hit VirusTotal, AbuseIPDB, OTX, etc. The
    return shape mirrors a typical aggregated-intel response.
    """
    h = abs(hash(ioc["indicator"])) % 100
    reputation = h
    first_seen_days = (h % 365) + 1
    sources = []
    if h > 70: sources.append("VirusTotal")
    if h > 55: sources.append("AbuseIPDB")
    if h > 40: sources.append("OTX")
    if h > 25: sources.append("URLhaus")
    if h <= 15: sources.append("none")
    if reputation >= 70:
        verdict = "malicious"
        verdict_tone = "critical"
    elif reputation >= 45:
        verdict = "suspicious"
        verdict_tone = "high"
    elif reputation >= 20:
        verdict = "unknown"
        verdict_tone = "medium"
    else:
        verdict = "clean"
        verdict_tone = "low"
    return {
        "reputation": reputation,
        "verdict": verdict,
        "verdict_tone": verdict_tone,
        "first_seen": f"{first_seen_days}d ago",
        "sources": ", ".join(sources) if sources else "none",
    }


def render_ioc_panel(text: str) -> None:
    """Render the IOC panel with enrichment + per-IOC pivot expanders.

    This is now a Streamlit component (not pure HTML) because we want
    each row to expand into a real VirusTotal pivot panel when the
    analyst clicks. The header row stays semantic; pivots open below.
    """
    iocs = extract_iocs(text)
    has_vt_key = bool(_vt_api_key())
    enrichment_label = "VirusTotal live" if has_vt_key else "demo enrichment"

    if not iocs:
        st.markdown(
            '<div class="soc-panel">'
            '<div class="soc-panel__title">Indicators '
            '<span class="soc-meta">no observables found</span></div>'
            '<div style="color: var(--soc-text-muted); font-size: 0.85rem;">'
            'No IPs, hashes, domains, URLs, emails, CVEs, or hostnames '
            'detected in this narrative.</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    note = f' · showing first 30 of {len(iocs)}' if len(iocs) > 30 else ""
    st.markdown(
        '<div class="soc-panel" style="margin-bottom: 0.4rem;">'
        '<div class="soc-panel__title">Indicators &amp; enrichment '
        f'<span class="soc-meta">{len(iocs)} observable{"s" if len(iocs) != 1 else ""} · {enrichment_label}{note}</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Render each IOC as a SOC table row + collapsible pivot
    for idx, ioc in enumerate(iocs[:30]):
        enrichment = _enrich_ioc_real_or_mock(ioc)
        verdict = enrichment.get("verdict", "unknown")
        verdict_tone = enrichment.get("verdict_tone", "medium")
        reputation = enrichment.get("reputation", "-")
        first_seen = enrichment.get("first_seen", "-")
        sources = enrichment.get("sources", "")
        attributes = enrichment.get("vt_attributes")

        title = (
            f"{ioc['indicator']}  ·  {ioc['type']}  ·  "
            f"{verdict.upper()}  ·  score: {reputation}"
        )
        with st.expander(title, expanded=False):
            cols = st.columns([1, 1, 1, 1])
            cols[0].markdown(
                f'<div class="soc-panel__title" style="margin: 0;">Type</div>'
                f'<span class="soc-tag soc-mono">{ioc["type"]}</span>',
                unsafe_allow_html=True,
            )
            cols[1].markdown(
                f'<div class="soc-panel__title" style="margin: 0;">Verdict</div>'
                f'<span class="soc-pill {verdict_tone}">{verdict}</span>',
                unsafe_allow_html=True,
            )
            cols[2].markdown(
                f'<div class="soc-panel__title" style="margin: 0;">Score</div>'
                f'<span class="soc-cell-mono">{reputation}</span>',
                unsafe_allow_html=True,
            )
            cols[3].markdown(
                f'<div class="soc-panel__title" style="margin: 0;">First seen</div>'
                f'<span class="soc-cell-mono">{first_seen}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="soc-panel__title" style="margin-top: 0.6rem;">'
                f'Sources</div>'
                f'<span class="soc-cell-mono" style="color: var(--soc-text-secondary);">{sources}</span>',
                unsafe_allow_html=True,
            )

            # External pivot links
            pivots = _ioc_pivot_links(ioc)
            if pivots:
                pivot_html = " &nbsp; ".join(
                    f'<a href="{u}" target="_blank" rel="noopener">{lbl}</a>'
                    for lbl, u in pivots
                )
                st.markdown(
                    f'<div class="soc-panel__title" style="margin-top: 0.7rem;">'
                    'Pivot</div>'
                    f'<div style="font-size: 0.85rem;">{pivot_html}</div>',
                    unsafe_allow_html=True,
                )

            # Live VT attributes. Streamlit forbids nested expanders, so
            # we use a toggle inside the parent IOC expander rather than
            # opening a second one. The toggle key is per-IOC so each
            # row's state is independent.
            if isinstance(attributes, dict):
                show_raw = st.toggle(
                    "Show VirusTotal raw attributes",
                    key=f"vt_raw_{idx}_{ioc['indicator']}",
                )
                if show_raw:
                    st.json(attributes)


def _ioc_pivot_links(ioc: dict[str, str]) -> list[tuple[str, str]]:
    """External pivot URLs (VT, Shodan, AbuseIPDB, GreyNoise, MITRE)."""
    indicator = ioc["indicator"]
    ioc_type = ioc["type"]
    links: list[tuple[str, str]] = []
    if ioc_type in ("ipv4", "ipv6"):
        links += [
            ("VirusTotal",  f"https://www.virustotal.com/gui/ip-address/{indicator}"),
            ("AbuseIPDB",   f"https://www.abuseipdb.com/check/{indicator}"),
            ("Shodan",      f"https://www.shodan.io/host/{indicator}"),
            ("GreyNoise",   f"https://viz.greynoise.io/ip/{indicator}"),
        ]
    elif ioc_type == "domain":
        links += [
            ("VirusTotal",  f"https://www.virustotal.com/gui/domain/{indicator}"),
            ("URLhaus",     f"https://urlhaus.abuse.ch/browse.php?search={indicator}"),
            ("Censys",      f"https://search.censys.io/hosts?q={indicator}"),
        ]
    elif ioc_type in ("md5", "sha1", "sha256"):
        links += [
            ("VirusTotal",  f"https://www.virustotal.com/gui/file/{indicator}"),
            ("MalwareBazaar", f"https://bazaar.abuse.ch/browse.php?search=sha256%3A{indicator}"),
        ]
    elif ioc_type == "url":
        from urllib.parse import quote
        links += [
            ("URLhaus",     f"https://urlhaus.abuse.ch/browse.php?search={quote(indicator)}"),
            ("VirusTotal",  f"https://www.virustotal.com/gui/search/{quote(indicator)}"),
        ]
    elif ioc_type == "cve":
        links += [
            ("NVD",         f"https://nvd.nist.gov/vuln/detail/{indicator}"),
            ("MITRE",       f"https://cve.mitre.org/cgi-bin/cvename.cgi?name={indicator}"),
        ]
    return links


# =============================================================================
# MITRE COVERAGE REPORT (Batch)
# =============================================================================

def _build_coverage_report(results: list[dict]) -> pd.DataFrame:
    """Aggregate per-tactic technique coverage from a batch run."""
    technique_to_tactic: dict[str, tuple[str, str]] = {}
    for tactic_id, tactic_name, techs in KILL_CHAIN_STAGES:
        for t in techs:
            technique_to_tactic[t.upper()] = (tactic_id, tactic_name)

    rows: list[dict[str, Any]] = []
    total = max(len(results), 1)
    cell_counts: dict[tuple[str, str, str, str], int] = {}
    sev_breakdown: dict[tuple[str, str], Counter] = {}

    for r in results:
        label = r.get("final_label", "uncertain")
        sev = severity_for(label)
        techs = MITRE_MAPPING.get(label, [])
        if not techs:
            # Some labels (uncertain, benign_activity) intentionally
            # have no MITRE techniques. Without a synthetic Unmapped
            # bucket, those events vanish from the coverage CSV and
            # tactic rollup, so the per-batch totals don't reconcile
            # against the triage results CSV. Group by label so the
            # user can see the breakdown of what didn't map.
            key = ("UNMAPPED", "Unmapped", "(no MITRE technique)", label)
            cell_counts[key] = cell_counts.get(key, 0) + 1
            sev_breakdown.setdefault(
                ("UNMAPPED", "(no MITRE technique)"), Counter()
            )[sev] += 1
            continue
        for tech in techs:
            tactic = technique_to_tactic.get(tech.upper(), ("UNMAPPED", "Unmapped"))
            key = (tactic[0], tactic[1], tech, label)
            cell_counts[key] = cell_counts.get(key, 0) + 1
            sev_breakdown.setdefault((tactic[0], tech), Counter())[sev] += 1

    for (tactic_id, tactic_name, tech, label), count in cell_counts.items():
        sev_dist = sev_breakdown.get((tactic_id, tech), Counter())
        rows.append({
            "tactic_id": tactic_id,
            "tactic": tactic_name,
            "technique": tech,
            "label": humanize(label),
            "events": count,
            "pct_of_batch": round(count / total * 100, 2),
            "critical": sev_dist.get("critical", 0),
            "high":     sev_dist.get("high", 0),
            "medium":   sev_dist.get("medium", 0),
            "low":      sev_dist.get("low", 0),
            "info":     sev_dist.get("info", 0),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["tactic_id", "events"], ascending=[True, False])
    return df


def render_coverage_summary(df: pd.DataFrame) -> str:
    """Compact tactic-level coverage summary used above the export button."""
    if df.empty:
        return (
            '<div class="soc-empty"><div class="soc-empty__title">No coverage data</div>'
            '<div class="soc-empty__hint">No labeled techniques in this batch.</div></div>'
        )
    by_tactic = df.groupby("tactic", as_index=False).agg(
        events=("events", "sum"),
        techniques=("technique", "nunique"),
    ).sort_values("events", ascending=False)
    max_events = by_tactic["events"].max() if not by_tactic.empty else 1
    rows = []
    for _, row in by_tactic.iterrows():
        bar_pct = (row["events"] / max_events * 100) if max_events else 0
        rows.append(
            f'<div class="soc-coverage__row">'
            f'<div class="soc-coverage__name">{row["tactic"]}</div>'
            '<div class="soc-coverage__bar-wrap">'
            f'<div class="soc-coverage__bar-fill" style="width: {bar_pct:.1f}%;"></div>'
            '</div>'
            f'<div class="soc-coverage__count soc-mono">{int(row["events"])} ev</div>'
            f'<div class="soc-coverage__count soc-mono" style="opacity: 0.7;">{int(row["techniques"])} tech</div>'
            '</div>'
        )
    return (
        '<div class="soc-panel">'
        '<div class="soc-panel__title">MITRE coverage by tactic '
        f'<span class="soc-meta">{by_tactic["events"].sum()} total event-technique pairs</span></div>'
        f'<div class="soc-coverage">{"".join(rows)}</div>'
        '</div>'
    )


# =============================================================================
# CASE TIMELINE
# =============================================================================
# Events are persisted as a JSON-encoded list under the settings key
# "case_timeline::{analysis_id}". Adding a status change, a note, or the
# initial creation appends to this list. The Investigate result and the
# Bookmarks expander each render the timeline as a vertical narrative.

import json as _json


_TIMELINE_KIND_LABELS = {
    "created":   ("Triage created",  "info"),
    "llm":       ("LLM rationale",   "accent"),
    "status":    ("Status change",   "medium"),
    "note":      ("Analyst note",    "info"),
    "bookmark":  ("Bookmarked",      "accent"),
}


def _timeline_key(analysis_id: int | str) -> str:
    return f"case_timeline::{analysis_id}"


def get_case_timeline(analysis_id: int | str | None) -> list[dict]:
    if analysis_id in (None, "", 0):
        return []
    try:
        raw = _db().get_setting(_timeline_key(analysis_id), default="[]")
        if isinstance(raw, str):
            entries = _json.loads(raw)
        elif isinstance(raw, list):
            entries = raw
        else:
            entries = []
        return entries if isinstance(entries, list) else []
    except Exception:
        return []


def append_timeline_event(
    analysis_id: int | str | None,
    kind: str,
    details: str,
    *,
    extra: dict | None = None,
) -> None:
    if analysis_id in (None, "", 0):
        return
    entries = get_case_timeline(analysis_id)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "details": details or "",
    }
    if extra:
        entry["extra"] = extra
    entries.append(entry)
    try:
        _db().save_setting(_timeline_key(analysis_id), _json.dumps(entries))
    except Exception as exc:
        st.warning(f"Could not append to case timeline: {exc}")


def render_case_timeline(analysis_id: int | str | None) -> str:
    entries = get_case_timeline(analysis_id)
    if not entries:
        return (
            '<div class="soc-panel">'
            '<div class="soc-panel__title">Case timeline '
            '<span class="soc-meta">no events yet</span></div>'
            '<div style="color: var(--soc-text-muted); font-size: 0.85rem;">'
            'Status changes and analyst notes will appear here.</div>'
            '</div>'
        )
    items = []
    for entry in entries:
        kind = entry.get("kind", "")
        label, tone = _TIMELINE_KIND_LABELS.get(
            kind, ("Event", "muted")
        )
        try:
            ts = datetime.fromisoformat(entry.get("ts", "")).astimezone()
            ts_human = ts.strftime("%b %d %H:%M")
        except Exception:
            ts_human = entry.get("ts", "-")[:16]
        details = (entry.get("details") or "").replace("\n", "<br>")
        items.append(
            '<li class="soc-timeline__item">'
            f'<span class="soc-timeline__dot tone-{tone}"></span>'
            '<div class="soc-timeline__body">'
            '<div class="soc-timeline__head">'
            f'<span class="soc-timeline__kind">{label}</span>'
            f'<span class="soc-timeline__time soc-mono">{ts_human}</span>'
            '</div>'
            f'<div class="soc-timeline__details">{details}</div>'
            '</div>'
            '</li>'
        )
    return (
        '<div class="soc-panel">'
        '<div class="soc-panel__title">Case timeline '
        f'<span class="soc-meta">{len(entries)} event{"s" if len(entries) != 1 else ""}</span></div>'
        f'<ul class="soc-timeline">{"".join(items)}</ul>'
        '</div>'
    )


# =============================================================================
# SAVED SEARCHES
# =============================================================================
# Persisted as a JSON list under the settings key "saved_searches".

_SAVED_SEARCHES_KEY = "saved_searches"


def get_saved_searches() -> list[dict]:
    try:
        raw = _db().get_setting(_SAVED_SEARCHES_KEY, default="[]")
        entries = _json.loads(raw) if isinstance(raw, str) else raw
        return entries if isinstance(entries, list) else []
    except Exception:
        return []


def save_search(name: str, payload: dict) -> None:
    if not name.strip():
        return
    entries = get_saved_searches()
    # Replace existing entry with same name (idempotent save)
    entries = [e for e in entries if e.get("name") != name.strip()]
    entries.append({
        "name": name.strip(),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "filters": payload,
    })
    try:
        _db().save_setting(_SAVED_SEARCHES_KEY, _json.dumps(entries))
    except Exception as exc:
        st.warning(f"Could not save search: {exc}")


def delete_saved_search(name: str) -> None:
    entries = [e for e in get_saved_searches() if e.get("name") != name]
    try:
        _db().save_setting(_SAVED_SEARCHES_KEY, _json.dumps(entries))
    except Exception as exc:
        st.warning(f"Could not delete saved search: {exc}")


# =============================================================================
# DEMO DATA GENERATOR
# =============================================================================
# When toggled on, a fragment fires every few seconds, picks a random
# sample narrative, runs it through the classifier, and persists the
# result. The Overview live tail picks it up automatically and the page
# starts to feel like a real SOC console even on a fresh database.

_DEMO_FLAG_KEY = "demo_generator_on"
_DEMO_LAST_KEY = "demo_generator_last"
_DEMO_COUNT_KEY = "demo_emitted_count"
_DEMO_LAST_ERR_KEY = "demo_last_error"
_DEMO_LAST_EMIT_KEY = "demo_last_emit_iso"


def demo_generator_active() -> bool:
    return bool(st.session_state.get(_DEMO_FLAG_KEY, False))


def seed_historical_events(days: int = 30, count: int = 150) -> tuple[int, str | None]:
    """Backfill `count` synthetic events spread across the last `days`.

    Used to populate the Overview charts on a fresh install or after a
    Clear demo events so the dashboard looks lived-in immediately.

    Performance note: this used to call `predict()` per row, which loads
    sentence-transformer embeddings and ran 5 to 30 seconds blocking
    long enough to trip Streamlit's WebSocket heartbeat. The classifier
    isn't adding signal here (we picked the example's category, so we
    know its label), so this version uses `EXAMPLE_LABEL_MAP` for
    deterministic labels and a single batched INSERT. Total runtime is
    under a second for 200 rows.
    """
    import random as _random

    db = _db()
    rng = _random.Random(time.time_ns())
    examples = list(EXAMPLE_INCIDENTS.items())

    suffix_pool = [
        " Source IP: 10.{a}.{b}.{c}.",
        " Asset: WS-{tag}-{n:02d}.",
        " Sensor cluster {region}-{idx}.",
        " EDR alert id: ED-{ts}.",
        " Detected by sensor {region}-soc-{idx}.",
    ]
    regions = ["us-east", "eu-west", "ap-south", "us-west", "eu-north"]
    tags = ["FIN", "HR", "EXEC", "IT", "OPS", "DEV"]

    rows: list[tuple] = []
    now = datetime.now()
    threshold_value = float(st.session_state.get("threshold", 0.5))

    for _ in range(count):
        # Skewed distribution: bias toward recent days
        day_offset = int(rng.triangular(0, days - 1, 0))
        hour = rng.randint(7, 21) if rng.random() > 0.15 else rng.randint(0, 23)
        minute = rng.randint(0, 59)
        second = rng.randint(0, 59)
        event_dt = now - timedelta(days=day_offset)
        event_dt = event_dt.replace(
            hour=hour, minute=minute, second=second, microsecond=0
        )

        example_name, body = rng.choice(examples)
        canonical_label = EXAMPLE_LABEL_MAP.get(example_name, "uncertain")

        # Synthesize a confidence score with a realistic distribution.
        # Most events are mid-confidence; a few drop into the uncertain
        # band so the histogram + uncertainty stats look honest.
        roll = rng.random()
        if roll < 0.05:
            # Low-confidence: triangular [0.30, 0.55] biased toward 0.45
            max_prob = rng.triangular(0.30, 0.55, 0.45)
            final_label = "uncertain"
            uncertainty = "low"
        elif roll < 0.20:
            max_prob = rng.triangular(0.55, 0.75, 0.65)
            final_label = canonical_label
            uncertainty = "medium"
        else:
            max_prob = rng.triangular(0.70, 0.95, 0.85)
            final_label = canonical_label
            uncertainty = "high"

        suffix_template = rng.choice(suffix_pool)
        suffix = suffix_template.format(
            a=rng.randint(1, 250),
            b=rng.randint(1, 250),
            c=rng.randint(2, 254),
            tag=rng.choice(tags),
            n=rng.randint(10, 99),
            region=rng.choice(regions),
            idx=rng.randint(1, 4),
            ts=rng.randint(10000, 99999),
        )
        text = f"[demo] {body}{suffix}"

        rows.append((
            event_dt.isoformat(),
            text,
            final_label,
            float(max_prob),
            uncertainty,
            "demo",
            "default",
            threshold_value,
            0,
            None,
            "demo",
        ))

    inserted = 0
    try:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.executemany(
                """
                INSERT INTO analysis_history
                (timestamp, incident_text, final_label, max_prob,
                 uncertainty_level, analysis_mode, difficulty,
                 threshold, use_llm, raw_result, batch_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            inserted = cur.rowcount or len(rows)
    except Exception as exc:
        return inserted, f"persist failed: {exc}"
    finally:
        # Bust the cached history snapshots so the Overview fragments
        # repaint with the new rows on the next refresh.
        try:
            _overview_history_snapshot.clear()
            _overview_meta_snapshot.clear()
        except Exception:
            pass

    return inserted, None


@st.cache_resource(show_spinner=False)
def _auto_seed_marker() -> dict[str, bool]:
    """Module-scoped marker that the auto-seeder has already run.

    Cached as a resource so the seed only fires once per process.
    Returns a dict so we can flip flags without invalidating the cache.
    """
    return {"seeded": False}


def ensure_demo_data_seeded() -> None:
    """Auto-seed synthetic data when the database is empty on cold start.

    Runs at most once per process. Only fires when:
      1. The history is empty (or nearly so), AND
      2. We're either on the hosted demo OR the user just toggled the
         demo generator on for the first time.

    This is what makes a fresh deploy land on a populated dashboard
    instead of an empty one. After this fires once, subsequent boots
    find rows already there and skip.
    """
    marker = _auto_seed_marker()
    if marker.get("seeded"):
        return
    # Only auto-seed on the hosted demo. Local runs stay unmolested
    # unless the user explicitly hits "Backfill history" in Settings.
    if not _is_hosted_demo():
        marker["seeded"] = True
        return
    try:
        history = _db().get_analysis_history(limit=5)
        if len(history) >= 5:
            marker["seeded"] = True
            return
        n, err = seed_historical_events(days=30, count=180)
        marker["seeded"] = True
        if err:
            st.session_state[_DEMO_LAST_ERR_KEY] = f"auto-seed: {err}"
        else:
            st.session_state[_DEMO_COUNT_KEY] = (
                int(st.session_state.get(_DEMO_COUNT_KEY, 0)) + n
            )
    except Exception as exc:
        marker["seeded"] = True
        st.session_state[_DEMO_LAST_ERR_KEY] = f"auto-seed: {exc}"


def _clear_demo_events() -> int:
    """Remove demo-generated rows (batch_id = 'demo') from history.

    Best-effort; returns the number of rows deleted. The bookmark/note/
    settings rows are left intact since they are unlikely to be tied to
    demo events.
    """
    db = _db()
    try:
        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM analysis_history WHERE batch_id = ?",
                ("demo",),
            )
            n = cur.rowcount or 0
        st.session_state[_DEMO_COUNT_KEY] = 0
        return n
    except Exception:
        return 0


def emit_demo_event() -> tuple[int | None, str | None]:
    """Emit one synthetic incident through the full pipeline.

    Used by both the periodic fragment and the manual "emit now" button
    in Settings, so we have one path to debug and one path to harden.
    Returns (analysis_id, error_message). The error string is None on
    success.
    """
    import random as _random

    rng = _random.Random(time.time_ns())
    label, body = rng.choice(list(EXAMPLE_INCIDENTS.items()))
    suffix_pool = [
        f" Source IP: 10.{rng.randint(1, 250)}.{rng.randint(1, 250)}.{rng.randint(2, 254)}.",
        f" Detection at {datetime.now().strftime('%H:%M:%S')} UTC.",
        f" Asset: WS-{rng.choice(['FIN', 'HR', 'EXEC', 'IT'])}-{rng.randint(10, 99):02d}.",
        f" Sensor cluster {rng.choice(['us-east', 'eu-west', 'ap-south'])}-{rng.randint(1, 4)}.",
    ]
    text = f"[demo] {body}{rng.choice(suffix_pool)}"

    try:
        result = predict(
            text,
            threshold=float(st.session_state.get("threshold", 0.5)),
            max_classes=int(st.session_state.get("max_classes", 5)),
        )
    except Exception as exc:
        msg = f"predict() failed: {exc}"
        st.session_state[_DEMO_LAST_ERR_KEY] = msg
        return None, msg

    try:
        aid = _persist_analysis(result, batch_id="demo")
    except Exception as exc:
        msg = f"persist failed: {exc}"
        st.session_state[_DEMO_LAST_ERR_KEY] = msg
        return None, msg

    if aid:
        try:
            append_timeline_event(
                aid,
                "created",
                f"Synthetic event emitted by demo generator. Predicted "
                f"label: {humanize(result['final_label'])} at "
                f"{float(result.get('max_prob', 0)):.0%} confidence.",
            )
        except Exception:
            # Timeline seeding is best-effort; the row is already saved.
            pass
        st.session_state[_DEMO_COUNT_KEY] = (
            int(st.session_state.get(_DEMO_COUNT_KEY, 0)) + 1
        )
        st.session_state[_DEMO_LAST_EMIT_KEY] = datetime.now(timezone.utc).isoformat()
        st.session_state[_DEMO_LAST_ERR_KEY] = None
        return aid, None

    msg = "save_analysis returned no id"
    st.session_state[_DEMO_LAST_ERR_KEY] = msg
    return None, msg


@st.fragment(run_every="8s")
def demo_generator_fragment() -> None:
    """Periodic synthetic-event emitter.

    Mounted globally from main() so it fires regardless of which page
    the user is on; without that, flipping the toggle in Settings would
    do nothing until the user navigated to Overview.

    Wrapped in a try/except so a network blip on the LLM provider or a
    transient DB lock cannot bubble up and tear down the page.
    """
    if not demo_generator_active():
        return
    # Once the user opts to clear demo data before a batch run, stay
    # paused for the rest of the session so the emitter doesn't
    # immediately reseed one row at a time and re-pollute the
    # dashboards the user just cleaned up.
    if st.session_state.get("_demo_emitter_paused"):
        return
    last = st.session_state.get(_DEMO_LAST_KEY, 0.0)
    now_ts = time.time()
    if now_ts - last < 7:
        return
    st.session_state[_DEMO_LAST_KEY] = now_ts
    try:
        emit_demo_event()
    except Exception as exc:
        st.session_state[_DEMO_LAST_ERR_KEY] = f"fragment: {exc}"


# =============================================================================
# REAL IOC ENRICHMENT (VirusTotal)
# =============================================================================

_VT_KEY_SETTING = "vt_api_key"


def _vt_api_key() -> str:
    """Pull a VirusTotal API key from session, then env, then secrets."""
    return (
        st.session_state.get(_VT_KEY_SETTING, "")
        or _env("VIRUSTOTAL_API_KEY", "VT_API_KEY")
        or _secret("VIRUSTOTAL_API_KEY", "")
        or _secret("VT_API_KEY", "")
    )


@st.cache_data(ttl=900, show_spinner=False)
def _vt_lookup(indicator: str, ioc_type: str, key: str) -> dict[str, Any] | None:
    """Best-effort VirusTotal lookup. Returns None on any failure.

    Cached for 15 minutes so repeat hits on the same indicator do not
    burn API quota. The cache key includes the API key so swapping keys
    in Settings does not return stale data.
    """
    if not key:
        return None
    import requests
    endpoint = None
    if ioc_type in ("ipv4", "ipv6"):
        endpoint = f"https://www.virustotal.com/api/v3/ip_addresses/{indicator}"
    elif ioc_type == "domain":
        endpoint = f"https://www.virustotal.com/api/v3/domains/{indicator}"
    elif ioc_type == "url":
        url_id = base64.urlsafe_b64encode(indicator.encode()).decode().strip("=")
        endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
    elif ioc_type in ("md5", "sha1", "sha256"):
        endpoint = f"https://www.virustotal.com/api/v3/files/{indicator}"
    if not endpoint:
        return None
    try:
        resp = requests.get(
            endpoint,
            headers={"x-apikey": key, "Accept": "application/json"},
            timeout=8,
        )
        if resp.status_code == 404:
            return {"verdict": "unknown", "stats": {}, "raw": "not found in VT"}
        if resp.status_code != 200:
            return {"verdict": "unknown", "stats": {}, "raw": f"VT {resp.status_code}"}
        data = resp.json().get("data", {}).get("attributes", {})
        stats = data.get("last_analysis_stats", {})
        malicious = int(stats.get("malicious", 0))
        suspicious = int(stats.get("suspicious", 0))
        if malicious >= 5:
            verdict = "malicious"
        elif malicious >= 1:
            verdict = "suspicious"
        elif suspicious >= 1:
            verdict = "suspicious"
        else:
            verdict = "clean"
        return {
            "verdict": verdict,
            "stats": stats,
            "raw": data,
        }
    except Exception:
        return None


def _enrich_ioc_real_or_mock(ioc: dict[str, str]) -> dict[str, Any]:
    """Combine VT (when key is configured) with the deterministic mock."""
    mock = _mock_enrich(ioc)
    key = _vt_api_key()
    real = _vt_lookup(ioc["indicator"], ioc["type"], key) if key else None
    if real:
        verdict = real["verdict"]
        verdict_tone = {
            "malicious":  "critical",
            "suspicious": "high",
            "unknown":    "medium",
            "clean":      "low",
        }.get(verdict, "medium")
        stats = real.get("stats") or {}
        malicious = stats.get("malicious")
        sources = ["VirusTotal"]
        score_text = (
            f"{malicious}/{sum(int(v or 0) for v in stats.values())}"
            if stats and malicious is not None
            else "live"
        )
        return {
            "reputation": score_text,
            "verdict": verdict,
            "verdict_tone": verdict_tone,
            "first_seen": "live",
            "sources": ", ".join(sources),
            "vt_attributes": real.get("raw") if isinstance(real.get("raw"), dict) else None,
        }
    return mock


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar() -> None:
    # Brand block
    logo_html = ""
    if _LOGO_PATH.exists():
        b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
        logo_html = (
            f'<img src="data:image/svg+xml;base64,{b64}" '
            'style="height: 26px; width: auto;" alt="AlertSage" />'
        )
    st.sidebar.markdown(
        '<div style="display:flex; align-items:center; gap:0.55rem; '
        'padding: 0.4rem 0 0.65rem 0;">'
        f'{logo_html}'
        '<div>'
        '<div style="font-size: 1.05rem; font-weight: 800; letter-spacing: -0.01em;">'
        'AlertSage</div>'
        '<div style="font-size: 0.7rem; color: var(--soc-text-muted); '
        'letter-spacing: 0.06em; text-transform: uppercase; margin-top: -2px;">'
        'SOC console</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<div style="font-size: 0.66rem; font-weight: 700; '
        'text-transform: uppercase; letter-spacing: 0.08em; '
        'color: var(--soc-text-muted); margin: 0.5rem 0 0.4rem 0;">Navigate</div>',
        unsafe_allow_html=True,
    )

    # Nav buttons styled as a clean nav. The active class uses a wrapper div.
    for key, label, hint in NAV_ITEMS:
        active = st.session_state.get("view") == key
        wrapper = "soc-nav-button active" if active else "soc-nav-button"
        st.sidebar.markdown(f'<div class="{wrapper}">', unsafe_allow_html=True)
        if st.sidebar.button(
            label,
            key=f"nav_{key}",
            use_container_width=True,
            help=hint,
        ):
            st.session_state["view"] = key
            st.rerun()
        st.sidebar.markdown("</div>", unsafe_allow_html=True)

    st.sidebar.markdown("---")

    # Triage knobs
    st.sidebar.markdown(
        '<div style="font-size: 0.66rem; font-weight: 700; '
        'text-transform: uppercase; letter-spacing: 0.08em; '
        'color: var(--soc-text-muted); margin: 0.25rem 0 0.4rem 0;">Triage</div>',
        unsafe_allow_html=True,
    )
    st.session_state["threshold"] = st.sidebar.slider(
        "Confidence threshold",
        0.0, 1.0,
        float(st.session_state.get("threshold", 0.50)),
        0.05,
        help="Below this, the classifier returns 'uncertain'.",
    )
    st.session_state["max_classes"] = st.sidebar.slider(
        "Probability rows",
        1, 10,
        int(st.session_state.get("max_classes", 5)),
        1,
        help="How many candidate classes to show.",
    )
    st.session_state["use_preprocessing"] = st.sidebar.checkbox(
        "Text preprocessing",
        value=bool(st.session_state.get("use_preprocessing", True)),
    )
    # LLM assist mode replaces the old boolean checkbox. Off keeps the
    # sklearn output untouched. Fallback (default, matches the prior
    # checkbox-on behavior) calls the LLM only when sklearn looks
    # shaky. Override calls the LLM on every event and lets it set
    # the label whenever the LLM commits to a canonical class.
    _existing_mode = st.session_state.get("llm_assist_mode")
    if _existing_mode not in LLM_ASSIST_MODES:
        # Migrate from legacy boolean if no mode is set yet.
        _existing_mode = (
            LLM_ASSIST_FALLBACK
            if bool(st.session_state.get("use_llm", True))
            else LLM_ASSIST_OFF
        )
    _mode_labels = {
        LLM_ASSIST_OFF: "Off",
        LLM_ASSIST_FALLBACK: "Fallback",
        LLM_ASSIST_OVERRIDE: "Override",
    }
    st.session_state["llm_assist_mode"] = st.sidebar.radio(
        "LLM assist mode",
        options=list(LLM_ASSIST_MODES),
        index=list(LLM_ASSIST_MODES).index(_existing_mode),
        format_func=lambda m: _mode_labels[m],
        help=(
            "Off: sklearn classifier only. "
            "Fallback: LLM runs only when sklearn is uncertain or "
            "barely above threshold. "
            "Override: LLM classifies every event."
        ),
    )
    # Keep the legacy boolean in sync so any code path that still reads
    # use_llm (DB persistence, older callers) stays consistent.
    st.session_state["use_llm"] = (
        st.session_state["llm_assist_mode"] != LLM_ASSIST_OFF
    )

    # Effective rate-limit hint for the active provider, so the user
    # can see at a glance whether they're on the BYOK budget or the
    # shared demo cap. Suppressed when LLM mode is Off because no
    # calls will be made anyway.
    if st.session_state["llm_assist_mode"] != LLM_ASSIST_OFF:
        _rl_provider = (
            _resolve_llm_settings().get("provider") or _default_provider()
        )
        _rl_cap, _rl_window = _provider_rate_window(_rl_provider)
        if _rl_provider == "local":
            _rl_label = "Local: unlimited"
        elif _byok_present(_rl_provider):
            _rl_label = f"BYOK: {_rl_cap} calls/{_rl_window}s"
        else:
            _rl_label = f"Demo token: {_rl_cap} calls/{_rl_window}s"
        st.sidebar.caption(_rl_label)

    st.sidebar.markdown("---")

    # Quick provider snapshot
    settings = _resolve_llm_settings()
    provider = settings.get("provider", "local")
    provider_text = {
        "local": f"Local llama.cpp · GGUF",
        "huggingface": f"Hugging Face · {settings['hf_model']}",
        "openai": f"OpenAI · {settings['openai_model']}",
        "anthropic": f"Anthropic · {settings['anthropic_model']}",
    }.get(provider, "")
    st.sidebar.markdown(
        '<div style="font-size: 0.66rem; font-weight: 700; '
        'text-transform: uppercase; letter-spacing: 0.08em; '
        'color: var(--soc-text-muted); margin: 0.25rem 0 0.35rem 0;">Provider</div>'
        f'<div style="font-family: \'JetBrains Mono\', monospace; font-size: 0.78rem; '
        'color: var(--soc-text-secondary); line-height: 1.45; word-break: break-all;">'
        f'{provider_text}</div>'
        '<div style="font-size: 0.72rem; color: var(--soc-text-muted); '
        'margin-top: 0.3rem;">Configure in Settings.</div>',
        unsafe_allow_html=True,
    )

    # Saved searches pinned to the sidebar. Clicking applies the filter
    # set to Hunt and routes the user there.
    saved = get_saved_searches()
    if saved:
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            '<div style="font-size: 0.66rem; font-weight: 700; '
            'text-transform: uppercase; letter-spacing: 0.08em; '
            'color: var(--soc-text-muted); margin: 0.25rem 0 0.4rem 0;">'
            'Saved searches</div>',
            unsafe_allow_html=True,
        )
        for entry in saved[-6:]:
            name = entry.get("name", "Unnamed")
            cols = st.sidebar.columns([5, 1], gap="small")
            with cols[0]:
                if st.button(
                    name,
                    key=f"saved_{name}",
                    use_container_width=True,
                    help="Apply to Hunt",
                ):
                    st.session_state["pending_search_filters"] = entry.get(
                        "filters", {}
                    )
                    st.session_state["view"] = "hunt"
                    st.rerun()
            with cols[1]:
                if st.button(
                    "x",
                    key=f"saved_del_{name}",
                    use_container_width=True,
                    help="Delete saved search",
                ):
                    delete_saved_search(name)
                    st.rerun()

    # Demo data generator status (visible when active so users see why
    # the live tail keeps moving).
    if demo_generator_active():
        st.sidebar.markdown(
            '<div style="margin-top: 0.6rem; padding: 0.5rem 0.65rem; '
            'background: rgba(34, 197, 94, 0.10); border: 1px solid '
            'rgba(34, 197, 94, 0.30); border-radius: 6px;">'
            '<div style="display:flex; align-items:center; gap:0.45rem; '
            'font-size: 0.72rem; font-weight: 700; '
            'color: var(--soc-low); letter-spacing: 0.06em; '
            'text-transform: uppercase;">'
            '<span class="soc-live-dot"></span> Demo generator</div>'
            '<div style="font-size: 0.74rem; color: var(--soc-text-secondary); '
            'margin-top: 0.25rem;">Synth events every ~7s. Toggle off in Settings.</div>'
            '</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# OVERVIEW PAGE
# =============================================================================

# --- Cached snapshot helpers (shared across all live Overview fragments) ---

@st.cache_data(ttl=8, show_spinner=False)
def _overview_history_snapshot(_cache_bust: float) -> list[dict]:
    """Return the most recent triage history.

    Cached for 8 seconds so the data fragments on Overview share one DB
    hit per refresh cycle. The argument is a cache buster the fragments
    pass through so the cache invalidates when needed (used by the seed
    function to force a refresh).
    """
    try:
        return _db().get_analysis_history(limit=2000) or []
    except Exception:
        return []


@st.cache_data(ttl=15, show_spinner=False)
def _overview_meta_snapshot() -> dict[str, int]:
    try:
        return {
            "bookmarks": len(_db().get_bookmarks() or []),
            "notes":     len(_db().get_all_notes() or []),
        }
    except Exception:
        return {"bookmarks": 0, "notes": 0}


def _history_bucket_key() -> float:
    """Bucket the cache key into 8 second windows so fragments collide."""
    return float(int(time.time() // 8))


def _compute_overview_stats(history: list[dict]) -> dict[str, Any]:
    now = datetime.now()
    def _within(hours: int) -> int:
        cutoff = now - timedelta(hours=hours)
        return sum(
            1 for h in history
            if "timestamp" in h
            and _safe_dt(h["timestamp"]) and _safe_dt(h["timestamp"]) > cutoff
        )
    sev_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    confidences: list[float] = []
    for h in history:
        label = h.get("final_label", "unknown")
        label_counts[label] += 1
        sev_counts[severity_for(label)] += 1
        try:
            confidences.append(float(h.get("max_prob") or 0))
        except Exception:
            pass
    return {
        "total": len(history),
        "h_24": _within(24),
        "h_7d": _within(24 * 7),
        "h_30d": _within(24 * 30),
        "sev_counts": sev_counts,
        "label_counts": label_counts,
        "high_severity": sev_counts.get("critical", 0) + sev_counts.get("high", 0),
        "avg_conf": float(np.mean(confidences)) if confidences else 0.0,
    }


# --- Live Overview fragments ----------------------------------------------

def _fragment_error_box(name: str, exc: Exception) -> None:
    """Render a small error block inside a fragment body.

    Used so an exception in one fragment doesn't tear down the page.
    """
    st.markdown(
        '<div class="soc-empty">'
        f'<div class="soc-empty__title">{name} unavailable</div>'
        f'<div class="soc-empty__hint">{type(exc).__name__}: {exc}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every="10s")
def _overview_kpi_fragment() -> None:
    try:
        history = _overview_history_snapshot(_history_bucket_key())
        meta = _overview_meta_snapshot()
        stats = _compute_overview_stats(history)
    except Exception as exc:
        _fragment_error_box("KPI strip", exc)
        return
    total = stats["total"]
    high_severity = stats["high_severity"]
    h_24 = stats["h_24"]
    h_7d = stats["h_7d"]
    avg_conf = stats["avg_conf"]
    avg_tone = (
        "low" if avg_conf >= 0.8
        else "medium" if avg_conf >= 0.6
        else "high" if total else "info"
    )
    cols = st.columns(6, gap="small")
    cols[0].markdown(render_kpi(
        "Total analyzed", f"{total:,}",
        sub=f"+{h_7d} last 7d" if h_7d else "no recent activity",
        tone="info",
    ), unsafe_allow_html=True)
    cols[1].markdown(render_kpi(
        "Critical / high", f"{high_severity:,}",
        sub=f"{(high_severity / total * 100):.0f}% of corpus" if total else "n/a",
        tone="critical" if high_severity else "low",
    ), unsafe_allow_html=True)
    cols[2].markdown(render_kpi(
        "Last 24h", f"{h_24:,}",
        sub=f"vs {h_7d - h_24:,} prior 6d" if h_7d else "no events",
        tone="medium" if h_24 > 0 else "low",
    ), unsafe_allow_html=True)
    cols[3].markdown(render_kpi(
        "Avg confidence",
        f"{avg_conf:.0%}" if total else "n/a",
        sub="classifier output", tone=avg_tone,
    ), unsafe_allow_html=True)
    cols[4].markdown(render_kpi(
        "Bookmarks", f"{meta['bookmarks']:,}",
        sub="saved investigations", tone="info",
    ), unsafe_allow_html=True)
    cols[5].markdown(render_kpi(
        "Analyst notes", f"{meta['notes']:,}",
        sub="across history", tone="info",
    ), unsafe_allow_html=True)


@st.fragment(run_every="10s")
def _overview_charts_fragment() -> None:
    """Events-over-time + confidence histogram + severity donut."""
    try:
        history = _overview_history_snapshot(_history_bucket_key())
        stats = _compute_overview_stats(history)
    except Exception as exc:
        _fragment_error_box("Charts", exc)
        return

    render_section_head(
        "Events over time",
        f"Last 30 days, brush the slider to focus a window · "
        f"updated {datetime.now().strftime('%H:%M:%S')}",
    )
    fig = _events_over_time_figure(history, days=30)
    if fig is None:
        render_empty(
            "No events yet",
            "Run an Investigate triage or enable the demo generator in "
            "Settings to populate this view.",
        )
    else:
        st.plotly_chart(fig, use_container_width=True, key="ovw_chart_main")

    sub_left, sub_right = st.columns(2, gap="medium")
    with sub_left:
        render_section_head("Classifier confidence", "histogram")
        chist = _confidence_histogram_figure(history)
        if chist is None:
            render_empty("No confidence data", "Triage events to populate.")
        else:
            st.plotly_chart(chist, use_container_width=True, key="ovw_chart_hist")
    with sub_right:
        render_section_head("Severity distribution", "donut")
        if stats["total"] == 0:
            render_empty("No data", "Severity tiers populate as you triage.")
        else:
            st.plotly_chart(
                _severity_donut(stats["sev_counts"]),
                use_container_width=True,
                key="ovw_chart_donut",
            )


@st.fragment(run_every="15s")
def _overview_mitre_fragment() -> None:
    try:
        history = _overview_history_snapshot(_history_bucket_key())
    except Exception as exc:
        _fragment_error_box("MITRE heatmap", exc)
        return
    render_section_head("MITRE ATT&CK coverage", "tactic x technique density")
    heatmap_fig = _mitre_heatmap_figure(history)
    if heatmap_fig is None:
        render_empty(
            "No technique coverage",
            "Techniques map automatically once incidents are classified.",
        )
    else:
        st.plotly_chart(heatmap_fig, use_container_width=True, key="ovw_chart_mitre")


@st.fragment(run_every="12s")
def _overview_top_labels_fragment() -> None:
    try:
        history = _overview_history_snapshot(_history_bucket_key())
        stats = _compute_overview_stats(history)
    except Exception as exc:
        _fragment_error_box("Top classifications", exc)
        return
    label_counts: Counter[str] = stats["label_counts"]
    render_section_head("Top classifications", "by count")
    if not label_counts:
        render_empty("No classifications", "Run a triage to track labels.")
        return
    top = label_counts.most_common(8)
    max_count = top[0][1] if top else 1
    rows_html = "".join(
        f'''<div class="soc-dist__row">
            <div class="soc-dist__name">
                {severity_pill(lbl, fallback_text=humanize(lbl))}
            </div>
            <div class="soc-dist__bar-wrap">
                <div class="soc-dist__bar-fill" style="width: {(c / max_count * 100):.1f}%; '''
        f'''background: {SEVERITY_COLOR_HEX.get(severity_for(lbl), "#3b82f6")};"></div>
            </div>
            <div class="soc-dist__count">{c:,}</div>
        </div>'''
        for lbl, c in top
    )
    st.markdown(
        f'<div class="soc-panel">'
        f'<div class="soc-panel__title">Triage labels '
        f'<span class="soc-meta">{len(label_counts)} unique</span></div>'
        f'<div class="soc-dist">{rows_html}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


@st.fragment(run_every="10s")
def _overview_recent_table_fragment() -> None:
    try:
        history = _overview_history_snapshot(_history_bucket_key())
    except Exception as exc:
        _fragment_error_box("Recent events", exc)
        return
    render_section_head("Recent events", action="latest 10")
    recent = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)[:10]
    if not recent:
        render_empty(
            "Quiet on the wire",
            "No triage runs yet. Click <strong>Investigate</strong> in the sidebar to analyze your first incident, "
            "or enable the demo generator in Settings to backfill synthetic events.",
        )
    else:
        st.markdown(_recent_events_table(recent), unsafe_allow_html=True)


def view_overview() -> None:
    render_page_header(
        title="Mission control",
        subtitle="Triage volume, classification distribution, and recent activity across the AlertSage corpus. All data panels auto-refresh.",
        breadcrumb="Dashboards / Overview",
    )

    # Empty-state explainer: if the user just cleared demo data and ran
    # a small CSV, the dashboard will look almost-empty. A one-line
    # caption is gentler than letting the panels render with sparse
    # data and no context. Threshold of 5 picks up the realistic
    # "fresh after demo clear" state without false-firing for a
    # populated install.
    try:
        _hist_count = _db().count_history()
    except Exception:
        _hist_count = None
    if _hist_count is not None and _hist_count < 5:
        st.caption(
            f"Only {_hist_count} analyses in history. Run a few more from "
            "Investigate or Batch, or backfill demo data from Settings then "
            "Backfill history to populate the dashboard."
        )

    # KPI strip (auto-refresh)
    _overview_kpi_fragment()

    # Row 1: charts on the left, threat feed + live tail on the right.
    # The left column is one big fragment so the chart, histogram, and
    # donut all repaint on the same cadence.
    left, right = st.columns([5, 3], gap="large")
    with left:
        _overview_charts_fragment()
    with right:
        st.markdown(render_threat_feed(), unsafe_allow_html=True)
        render_live_tail_fragment(n=6)

    # Row 2: MITRE heatmap + top classifications, each in its own fragment.
    left2, right2 = st.columns([5, 3], gap="large")
    with left2:
        _overview_mitre_fragment()
    with right2:
        _overview_top_labels_fragment()

    # Recent events table (auto-refresh)
    _overview_recent_table_fragment()


def _safe_dt(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _count_by_day(history: list[dict], days: int = 14) -> dict[str, int]:
    cutoff = datetime.now() - timedelta(days=days)
    bucket: Counter[str] = Counter()
    for h in history:
        dt = _safe_dt(h.get("timestamp"))
        if not dt or dt < cutoff:
            continue
        bucket[dt.date().isoformat()] += 1
    return dict(bucket)


def _events_over_time_figure(history: list[dict], days: int = 30):
    """Stacked bar timechart with a Splunk-style brushable range slider.

    The full window is `days` deep, but the initial visible range is the
    last 14 days; the analyst can drag the slider handles or shift-drag
    the chart itself to refocus. Plotly persists the selected range in
    the URL so deep-links work.
    """
    cutoff = datetime.now() - timedelta(days=days)
    sev_per_day: dict[str, Counter[str]] = {}
    for h in history:
        dt = _safe_dt(h.get("timestamp"))
        if not dt or dt < cutoff:
            continue
        d = dt.date().isoformat()
        sev = severity_for(h.get("final_label", "uncertain"))
        sev_per_day.setdefault(d, Counter())[sev] += 1
    if not sev_per_day:
        return None
    days_sorted = sorted(sev_per_day.keys())
    severities_order = ["critical", "high", "medium", "low", "info"]
    fig = go.Figure()
    for sev in severities_order:
        y = [sev_per_day[d].get(sev, 0) for d in days_sorted]
        if all(v == 0 for v in y):
            continue
        fig.add_trace(go.Bar(
            x=days_sorted,
            y=y,
            name=sev.upper(),
            marker=dict(color=SEVERITY_COLOR_HEX[sev]),
            hovertemplate=f"<b>{sev.upper()}</b><br>%{{x}}: %{{y}} events<extra></extra>",
        ))

    layout = {**PLOT_LAYOUT}
    # Replace the default xaxis with a brushable one. The visible range
    # extends one day past "now" so today's events (which sit at midnight
    # on the bucket date) are not clipped by the right edge of the
    # plotting area.
    visible_start = (
        datetime.now() - timedelta(days=14)
    ).date().isoformat()
    visible_end = (
        datetime.now() + timedelta(days=1)
    ).date().isoformat()
    layout["xaxis"] = dict(
        showgrid=False,
        color="#94a3b8",
        linecolor="#1f2a44",
        rangeslider=dict(
            visible=True,
            thickness=0.10,
            bgcolor="#0f172a",
            bordercolor="#1f2a44",
            borderwidth=1,
        ),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="7d", step="day", stepmode="backward"),
                dict(count=14, label="14d", step="day", stepmode="backward"),
                dict(count=30, label="30d", step="day", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#111827",
            activecolor="#1e293b",
            bordercolor="#1f2a44",
            font=dict(color="#cbd5e1", size=10, family="JetBrains Mono"),
            x=0,
            xanchor="left",
            y=1.18,
            yanchor="top",
        ),
        range=[visible_start, visible_end],
        type="date",
    )
    fig.update_layout(
        barmode="stack",
        height=320,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.04, x=0.55,
            xanchor="left",
            font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)",
        ),
        **layout,
    )
    return fig


def _severity_donut(sev_counts: Counter[str]):
    severities_order = ["critical", "high", "medium", "low", "info"]
    labels = [s.upper() for s in severities_order if sev_counts.get(s)]
    values = [sev_counts.get(s, 0) for s in severities_order if sev_counts.get(s)]
    colors = [SEVERITY_COLOR_HEX[s] for s in severities_order if sev_counts.get(s)]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.65,
        marker=dict(colors=colors, line=dict(color="#0a0e1a", width=2)),
        textfont=dict(family="JetBrains Mono", size=11, color="#f8fafc"),
        textinfo="label+percent",
        hovertemplate="<b>%{label}</b><br>%{value} events (%{percent})<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(height=240, **{**PLOT_LAYOUT, "xaxis": dict(visible=False), "yaxis": dict(visible=False)})
    return fig


def _status_pill(status: str) -> str:
    tone = {
        "new": "info",
        "triaging": "medium",
        "contained": "high",
        "closed": "low",
    }.get(status, "muted")
    label = dict((k, l) for k, l, _ in CASE_STATUSES).get(status, status).upper()
    return f'<span class="soc-pill {tone}">{label}</span>'


def _hydrate_history_to_analysis(row: dict) -> dict:
    """Map a get_analysis_history row into the dict shape that
    render_analysis_result expects.

    Demo-seeded rows have raw_result=None (we skip the classifier on
    seed because the example label map already gives us the verdict),
    so probabilities / mitre_techniques / llm_opinion will be empty.
    render_analysis_result already falls back to MITRE_MAPPING for
    techniques and tolerates an empty probabilities list, so the
    rehydrated card just shows reduced detail rather than failing.
    """
    raw = row.get("raw_result")
    if isinstance(raw, str):
        try:
            raw = _json.loads(raw)
        except Exception:
            raw = None
    raw = raw or {}
    label = row.get("final_label", "uncertain")
    aid = int(row.get("id") or row.get("analysis_id") or 0)
    return {
        "analysis_id": aid,
        "incident_text": row.get("incident_text", ""),
        "final_label": label,
        "max_prob": float(row.get("max_prob") or 0),
        "uncertainty_level": row.get("uncertainty_level"),
        "probabilities": raw.get("probabilities") or [],
        "mitre_techniques": raw.get("mitre_techniques") or [],
        "llm_opinion": raw.get("llm_opinion"),
        "classifier_ms": raw.get("classifier_ms"),
        "llm_ms": raw.get("llm_ms"),
    }


def _recent_events_table(rows: list[dict]) -> str:
    body_rows = []
    for r in rows:
        dt = _safe_dt(r.get("timestamp"))
        when = dt.strftime("%b %d %H:%M") if dt else "-"
        ago = time_ago(r.get("timestamp"))
        label = r.get("final_label", "uncertain")
        try:
            conf = float(r.get("max_prob") or 0)
        except Exception:
            conf = 0.0
        anomaly = _anomaly_score(label, conf)
        body = (r.get("incident_text") or "").strip()
        body_short = body[:140] + ("..." if len(body) > 140 else "")
        analysis_id = r.get("analysis_id", r.get("id", ""))
        status = get_case_status(analysis_id) if analysis_id else "new"
        body_rows.append(
            f"<tr>"
            f'<td class="soc-cell-time">{when}<br><span style="opacity:0.7;">{ago} ago</span></td>'
            f'<td class="soc-cell-mono">#{analysis_id}</td>'
            f"<td>{severity_pill(label)}</td>"
            f"<td>{_status_pill(status)}</td>"
            f'<td class="soc-cell-mono">{conf:.0%}</td>'
            f"<td>{_anomaly_pill(anomaly)}</td>"
            f'<td class="soc-cell-truncate">{body_short}</td>'
            f"</tr>"
        )
    return (
        '<table class="soc-table">'
        "<thead><tr>"
        "<th>Time</th>"
        "<th>ID</th>"
        "<th>Classification</th>"
        "<th>Status</th>"
        "<th>Confidence</th>"
        "<th>Anomaly</th>"
        "<th>Narrative</th>"
        "</tr></thead>"
        f'<tbody>{"".join(body_rows)}</tbody>'
        "</table>"
    )


# =============================================================================
# INVESTIGATE PAGE
# =============================================================================

def view_investigate() -> None:
    render_page_header(
        title="Investigate",
        subtitle="Run an incident through the AlertSage classifier and (optionally) an LLM second opinion.",
        breadcrumb="Console / Investigate",
    )

    # Input row: example dropdown + textarea + run
    cols = st.columns([1, 4], gap="medium")

    with cols[0]:
        st.markdown(
            '<div class="soc-panel__title" style="margin-bottom: 0.45rem;">Examples</div>',
            unsafe_allow_html=True,
        )
        for ex_label, ex_body in EXAMPLE_INCIDENTS.items():
            if st.button(ex_label, key=f"ex_{ex_label}", use_container_width=True):
                st.session_state["investigate_text"] = ex_body
                st.session_state["current_analysis"] = None
                st.rerun()

    with cols[1]:
        st.markdown(
            '<div class="soc-panel__title" style="margin-bottom: 0.45rem;">'
            'Incident narrative</div>',
            unsafe_allow_html=True,
        )
        text = st.text_area(
            "Incident narrative",
            value=st.session_state.get("investigate_text", ""),
            height=180,
            placeholder=(
                "Paste alert text, EDR detection summary, or analyst notes. "
                "Free-form text works; the classifier preprocesses internally."
            ),
            key="investigate_textarea",
            label_visibility="collapsed",
        )
        st.session_state["investigate_text"] = text

        run_cols = st.columns([1, 1, 4], gap="small")
        with run_cols[0]:
            run = st.button("Triage", type="primary", use_container_width=True, key="btn_triage")
        with run_cols[1]:
            if st.button("Clear", type="secondary", use_container_width=True, key="btn_clear"):
                st.session_state["investigate_text"] = ""
                st.session_state["current_analysis"] = None
                st.rerun()
        with run_cols[2]:
            if text:
                st.caption(f"{len(text):,} characters · {len(text.split()):,} words")

    if run and text.strip():
        if len(text) > UI_LLM_MAX_INPUT_CHARS:
            st.error(
                f"Narrative too long ({len(text):,} chars). "
                f"Limit is {UI_LLM_MAX_INPUT_CHARS:,}."
            )
            return

        threshold = float(st.session_state.get("threshold", 0.5))
        max_classes = int(st.session_state.get("max_classes", 5))
        with st.spinner("Classifying..."):
            t0 = time.time()
            result = predict(text, threshold=threshold, max_classes=max_classes)
            classifier_ms = int((time.time() - t0) * 1000)
        result["classifier_ms"] = classifier_ms

        opinion = None
        opinion_ms = None
        mode = st.session_state.get("llm_assist_mode", LLM_ASSIST_FALLBACK)
        if _should_invoke_llm(result, mode, threshold):
            with st.spinner("Querying LLM second opinion..."):
                t0 = time.time()
                opinion, err, _details = run_llm_with_forced_fallback(
                    text,
                    skip_preprocessing=not st.session_state.get("use_preprocessing", True),
                )
                opinion_ms = int((time.time() - t0) * 1000)
            if err:
                st.warning(err)
        result["llm_opinion"] = opinion
        result["llm_ms"] = opinion_ms

        # When the LLM commits to a concrete canonical label, let it
        # drive the label and MITRE techniques. The LLM has the full
        # narrative in front of it and is the smarter model; the
        # TF-IDF classifier is a fast first-pass filter whose
        # vocabulary is fixed at training time, so it routinely
        # misroutes events that use slightly different wording.
        # apply_llm_override is the shared helper that gates on label
        # vocab so out-of-vocab LLM responses can never make the
        # result worse than sklearn alone.
        apply_llm_override(result, opinion)

        result["analysis_id"] = _persist_analysis(result, batch_id=None)
        st.session_state["current_analysis"] = result

    if st.session_state.get("current_analysis"):
        render_analysis_result(st.session_state["current_analysis"])


def _persist_analysis(result: dict, batch_id: str | None) -> int | None:
    """Save an analysis to the DB. Best-effort; returns id or None.

    On success, also seeds the case timeline with a 'created' event and
    a 'llm' event (when an LLM opinion is present), so the timeline
    panel always has something to show on the result card.
    """
    try:
        db = _db()
        analysis_id = db.save_analysis(
            incident_text=result["incident_text"],
            final_label=result["final_label"],
            max_prob=float(result.get("max_prob", 0)),
            uncertainty_level=result.get("uncertainty_level"),
            analysis_mode="single" if not batch_id else "batch",
            difficulty="default",
            threshold=float(st.session_state.get("threshold", 0.5)),
            use_llm=bool(result.get("llm_opinion")),
            raw_result={
                "probabilities": [(c, float(p)) for c, p in result.get("probabilities", [])],
                "mitre_techniques": result.get("mitre_techniques", []),
                "llm_opinion": result.get("llm_opinion"),
                "classifier_ms": result.get("classifier_ms"),
                "llm_ms": result.get("llm_ms"),
            },
            batch_id=batch_id,
        )
        if analysis_id and not batch_id:
            try:
                conf = float(result.get("max_prob", 0))
                append_timeline_event(
                    analysis_id,
                    "created",
                    f"Triage created · {humanize(result['final_label'])} "
                    f"at {conf:.0%} confidence "
                    f"(classifier {result.get('classifier_ms', '-')}ms).",
                )
                if result.get("llm_opinion"):
                    rationale = (
                        result["llm_opinion"].get("rationale", "")
                        or "(no rationale returned)"
                    )
                    rationale_short = (
                        rationale[:280] + ("..." if len(rationale) > 280 else "")
                    )
                    append_timeline_event(
                        analysis_id,
                        "llm",
                        f"LLM second opinion: {rationale_short}",
                    )
            except Exception:
                pass
        return analysis_id
    except Exception as exc:
        st.warning(f"Could not persist to history: {exc}")
        return None


def render_analysis_result(result: dict) -> None:
    """Render the SOC-style event detail card."""
    label = result["final_label"]
    sev = severity_for(label)
    conf = float(result.get("max_prob") or 0)
    techniques = result.get("mitre_techniques", []) or MITRE_MAPPING.get(label, [])

    # Header strip
    aid = result.get("analysis_id")
    aid_str = f"AS-{aid:06d}" if isinstance(aid, int) else "AS-PREVIEW"
    classifier_ms = result.get("classifier_ms")
    llm_ms = result.get("llm_ms")
    timing_meta = []
    if classifier_ms is not None:
        timing_meta.append(f"classifier {classifier_ms}ms")
    if llm_ms is not None:
        timing_meta.append(f"llm {llm_ms}ms")
    timing_html = " · ".join(timing_meta) or "no telemetry"

    # Case status stepper sits inside the head card, below the title.
    current_status = get_case_status(aid)
    anomaly = _anomaly_score(label, conf)
    head_html = f"""
    <div class="soc-event-head">
        <div class="soc-event-head__left">
            <div class="soc-event-id">{aid_str}  ·  {timing_html}</div>
            <div class="soc-event-title">{humanize(label)}</div>
            <div class="soc-event-meta">
                {severity_pill(label)}
                <span class="soc-tag accent">confidence {conf:.0%}</span>
                <span class="soc-tag">anomaly {anomaly}/100</span>
                <span class="soc-tag">uncertainty {result.get('uncertainty_level','-')}</span>
            </div>
        </div>
    </div>
    {render_case_stepper(aid, current_status)}
    """
    st.markdown(
        f'<div class="soc-panel">{head_html}</div>',
        unsafe_allow_html=True,
    )

    # Status-advance buttons (only when the analysis is persisted)
    if isinstance(aid, int):
        cur_idx = _CASE_STATUS_KEYS.index(current_status)
        status_cols = st.columns([1, 1, 1, 1, 4], gap="small")
        for idx, (key, label_text, _) in enumerate(CASE_STATUSES):
            with status_cols[idx]:
                disabled = (idx == cur_idx)
                if st.button(
                    label_text,
                    key=f"status_{key}_{aid}",
                    use_container_width=True,
                    type="primary" if idx == cur_idx else "secondary",
                    disabled=disabled,
                ):
                    set_case_status(aid, key)
                    st.rerun()

    # Kill chain stretches the full width above the two-column body.
    st.markdown(render_kill_chain(techniques), unsafe_allow_html=True)

    # IOC enrichment panel (full width, below kill chain). This is now a
    # Streamlit component (with per-IOC pivot expanders), not pure HTML.
    render_ioc_panel(result["incident_text"])

    # Case timeline runs the full width above the two-column body.
    st.markdown(render_case_timeline(aid), unsafe_allow_html=True)

    # Two columns: probabilities + LLM panel
    col_a, col_b = st.columns([5, 6], gap="large")

    with col_a:
        prob_rows = []
        max_p = max((p for _, p in result.get("probabilities", [])), default=1.0) or 1.0
        for cls_name, p in result.get("probabilities", []):
            bar_pct = (p / max_p) * 100
            prob_rows.append(
                f'''<div class="soc-prob-row">
                    <div class="soc-prob-row__name">
                        {severity_pill(cls_name, fallback_text=humanize(cls_name))}
                    </div>
                    <div class="soc-prob-row__pct">{p:.1%}</div>
                    <div class="soc-prob-row__bar">
                        <div class="soc-prob-row__bar-fill" '''
                f'''style="width: {bar_pct:.1f}%; background: {SEVERITY_COLOR_HEX.get(severity_for(cls_name), "#3b82f6")};"></div>
                    </div>
                </div>'''
            )
        st.markdown(
            '<div class="soc-panel">'
            '<div class="soc-panel__title">Class probabilities '
            f'<span class="soc-meta">{len(result.get("probabilities", []))} candidates</span></div>'
            f'{"".join(prob_rows)}'
            '</div>',
            unsafe_allow_html=True,
        )

        # MITRE techniques
        if techniques:
            chips = "".join(f'<span class="soc-mitre">{t}</span>' for t in techniques)
            st.markdown(
                '<div class="soc-panel">'
                '<div class="soc-panel__title">MITRE ATT&CK techniques</div>'
                f'<div class="soc-mitre-grid">{chips}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="soc-panel">'
                '<div class="soc-panel__title">MITRE ATT&CK techniques</div>'
                '<div style="color: var(--soc-text-muted); font-size: 0.85rem;">'
                'No mapped techniques for this label.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

    with col_b:
        # LLM rationale
        opinion = result.get("llm_opinion") or {}
        llm_label = opinion.get("label", label)
        rationale = opinion.get("rationale") or build_llm_rationale(label, result["incident_text"])
        st.markdown(
            '<div class="soc-panel">'
            '<div class="soc-panel__title">Analyst rationale '
            f'<span class="soc-meta">{"LLM" if opinion else "rule-based fallback"}</span></div>'
            f'<div class="soc-narrative tone-{sev}">{rationale}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # SOC playbook hint
        triage = soc_triage_hint(label, result.get("uncertainty_level", "medium"))
        actions_html = "".join(f"<li>{a}</li>" for a in triage.get("actions", []))
        st.markdown(
            '<div class="soc-panel">'
            '<div class="soc-panel__title">Playbook hint '
            f'<span class="soc-meta">queue: {triage.get("queue", "-")}  ·  '
            f'priority: {triage.get("priority", "-")}</span></div>'
            f'<ul class="soc-action-list">{actions_html}</ul>'
            '</div>',
            unsafe_allow_html=True,
        )

    # Footer: actions
    action_cols = st.columns([1, 1, 1, 6], gap="small")
    with action_cols[0]:
        if st.button("Bookmark", key="btn_bookmark", use_container_width=True):
            _bookmark_current(result)
    with action_cols[1]:
        if st.button("Add note", key="btn_add_note", use_container_width=True):
            st.session_state["adding_note"] = True
    with action_cols[2]:
        if st.button("Re-run", key="btn_rerun", type="secondary", use_container_width=True):
            st.session_state["current_analysis"] = None
            st.rerun()

    if st.session_state.get("adding_note"):
        with st.expander("Add analyst note", expanded=True):
            note_text = st.text_area("Note", key="note_input", height=80)
            note_cols = st.columns([1, 1, 4])
            with note_cols[0]:
                if st.button("Save note", key="save_note"):
                    aid = result.get("analysis_id")
                    if aid:
                        try:
                            _db().add_note(int(aid), note_text)
                            append_timeline_event(
                                int(aid), "note", note_text or "(empty note)"
                            )
                            st.success("Note saved.")
                            st.session_state["adding_note"] = False
                        except Exception as exc:
                            st.error(f"Could not save: {exc}")
                    else:
                        st.warning("No analysis id; cannot save note.")
            with note_cols[1]:
                if st.button("Cancel", key="cancel_note"):
                    st.session_state["adding_note"] = False


def _bookmark_current(result: dict) -> None:
    aid = result.get("analysis_id")
    if not aid:
        st.warning("No analysis id; cannot bookmark.")
        return
    try:
        # add_bookmark's first positional argument is incident_text, NOT
        # analysis_id. Calling add_bookmark(aid, note="") historically
        # stuffed the integer id into the incident_text column and left
        # analysis_id / final_label NULL, so the Bookmarks page rendered
        # only the number. Passing all four named args wires it up
        # correctly.
        _db().add_bookmark(
            incident_text=result.get("incident_text", ""),
            final_label=result.get("final_label"),
            note="",
            analysis_id=int(aid),
        )
        append_timeline_event(int(aid), "bookmark", "Saved as a bookmark.")
        st.success("Bookmarked.")
    except Exception as exc:
        st.error(f"Could not bookmark: {exc}")


# =============================================================================
# HUNT PAGE
# =============================================================================

_HUNT_QUERY_KEY = "hunt_query"
_HUNT_DEFAULT_QUERY = "last:7d"


def _legacy_payload_to_dsl(payload: dict) -> str:
    """Convert an old saved-search filter payload to a DSL query string.

    Older saved searches stored individual filter widgets (label_filter,
    sev_filter, min_conf, ...) instead of a single query string. This
    rebuilds an equivalent DSL expression so existing pinned searches
    keep working after the Hunt-tab redesign.
    """
    q = payload.get("query")
    if isinstance(q, str) and q.strip():
        # New format already stores a DSL string; pass through unless it
        # looks like a bare narrative term (no DSL operators).
        if any(tok in q for tok in (":", " AND ", " OR ", " NOT ", "(", "[")):
            return q
        parts = [f'narrative:"{q.strip()}"']
    else:
        parts = []

    for label in payload.get("label_filter") or []:
        parts.append(f"label:{label}")
    sevs = payload.get("sev_filter") or []
    if len(sevs) == 1:
        parts.append(f"severity:{sevs[0]}")
    elif len(sevs) > 1:
        parts.append("(" + " OR ".join(f"severity:{s}" for s in sevs) + ")")
    try:
        mc = float(payload.get("min_conf") or 0)
    except (TypeError, ValueError):
        mc = 0.0
    if mc > 0:
        parts.append(f"confidence:>={mc}")
    try:
        ma = int(payload.get("min_anomaly") or 0)
    except (TypeError, ValueError):
        ma = 0
    if ma > 0:
        parts.append(f"anomaly:>={ma}")
    tw_map = {
        "Last hour": "last:1h",
        "Last 24 hours": "last:24h",
        "Last 7 days": "last:7d",
        "Last 30 days": "last:30d",
    }
    tw = payload.get("time_window")
    if tw in tw_map:
        parts.append(tw_map[tw])
    return " AND ".join(parts)


def _decorate_history_row(row: dict) -> dict:
    """Pre-compute derived fields the DSL evaluator reads.

    Reuses severity_for / _anomaly_score / get_case_status so the Hunt
    DSL stays consistent with what the result card and history table
    show elsewhere.
    """
    label = row.get("final_label") or ""
    try:
        conf = float(row.get("max_prob") or 0)
    except (TypeError, ValueError):
        conf = 0.0
    aid = row.get("id") or row.get("analysis_id")
    row["_severity"] = severity_for(label)
    row["_anomaly"] = _anomaly_score(label, conf)
    row["_status"] = get_case_status(aid) if aid is not None else "new"
    row["_dt"] = _safe_dt(row.get("timestamp"))
    return row


def _append_to_query(snippet: str) -> None:
    """Append a snippet to the hunt query in session state.

    Designed to be wired up as a Streamlit ``on_click`` callback: those
    callbacks run *before* widgets are instantiated on the next rerun,
    which is the only safe time to mutate ``st.session_state`` for a
    key that a widget (here, the Query ``st.text_input``) owns.
    Calling this function from inside a ``if st.button(...):`` block
    instead would raise ``StreamlitAPIException`` because the text
    input has already claimed the key earlier in the same run. Do not
    call ``st.rerun()`` here -- the click that triggered the callback
    already produces a rerun, and an explicit rerun inside a callback
    raises ``NoSessionContext``.
    """
    if not snippet:
        return
    current = (st.session_state.get(_HUNT_QUERY_KEY) or "").strip()
    st.session_state[_HUNT_QUERY_KEY] = (
        f"{current} {snippet}".strip() if current else snippet
    )


def _clear_query() -> None:
    """Empty the hunt query.

    Same on_click callback contract as ``_append_to_query`` -- runs
    before the Query text_input is re-instantiated, so it can write
    to its session-state key without tripping
    ``StreamlitAPIException``.
    """
    st.session_state[_HUNT_QUERY_KEY] = ""


def _format_field_snippet(spec, value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    needs_quotes = (" " in value) and not (value.startswith('"') and value.endswith('"'))
    quoted = f'"{value}"' if needs_quotes else value
    return f"{spec.name}:{quoted}"


_HUNT_QUICK_INSERTS = (
    "label:malware",
    "label:phishing",
    "label:data_exfiltration",
    "severity:critical",
    "confidence:>=0.8",
    "anomaly:>=80",
    "last:1h",
    "last:24h",
    "last:7d",
    "status:new",
    "status:contained",
    "mitre:T1566",
    "NOT label:benign_activity",
    "NOT status:closed",
)


def _render_hunt_helpers() -> None:
    """Field/value picker + quick-insert chips that mutate the query."""
    st.markdown(
        '<div style="font-size: 0.72rem; font-weight: 700; '
        'text-transform: uppercase; letter-spacing: 0.08em; '
        'color: var(--soc-text-muted); margin: 0.6rem 0 0.35rem 0;">'
        'Build a clause</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([1.5, 2, 0.9], gap="small")
    field_names = [s.name for s in HUNT_FIELDS]
    with cols[0]:
        field_choice = st.selectbox(
            "Field",
            field_names,
            key="hunt_helper_field",
            help="Pick a field, then a value, then click Insert.",
        )
    spec = hunt_field_spec(field_choice)
    # Per-field scoped keys keep each field's selection independent.
    # Without scoping, the dropdown's saved option leaks to a new field
    # whose suggestions don't include it, and the two text_inputs (custom
    # branch + no-suggestions branch) share state across switches.
    pick_key = f"hunt_helper_value_pick_{field_choice}"
    text_key = f"hunt_helper_value_text_{field_choice}"
    with cols[1]:
        if spec and spec.suggestions:
            options = ["(custom)", *spec.suggestions]
            picked = st.selectbox(
                "Value",
                options,
                key=pick_key,
                help=spec.help or "",
            )
            if picked == "(custom)":
                value = st.text_input(
                    "Custom value",
                    key=text_key,
                    placeholder="Type a value",
                    label_visibility="collapsed",
                )
            else:
                value = picked
        else:
            value = st.text_input(
                "Value",
                key=text_key,
                placeholder=(spec.help or "Value") if spec else "Value",
            )
    with cols[2]:
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        # Compute the snippet up front so it can be passed to the
        # on_click callback. Streamlit captures callback kwargs at
        # widget-registration time, which is exactly what we want: the
        # snippet reflects the field/value selected in this same render.
        # Mutating st.session_state[_HUNT_QUERY_KEY] inside an
        # ``if st.button(...):`` block fails because the Query
        # text_input has already claimed that key earlier in this run.
        pending_snippet = _format_field_snippet(spec, value)
        st.button(
            "Insert",
            key="hunt_helper_insert",
            use_container_width=True,
            disabled=not pending_snippet,
            on_click=_append_to_query,
            kwargs={"snippet": pending_snippet},
        )

    st.markdown(
        '<div style="font-size: 0.72rem; font-weight: 700; '
        'text-transform: uppercase; letter-spacing: 0.08em; '
        'color: var(--soc-text-muted); margin: 0.6rem 0 0.35rem 0;">'
        'Quick inserts</div>',
        unsafe_allow_html=True,
    )
    chip_cols = st.columns(7, gap="small")
    for idx, snippet in enumerate(_HUNT_QUICK_INSERTS):
        with chip_cols[idx % 7]:
            # Same widget-key-after-instantiation hazard as the Insert
            # button above: route through on_click so the mutation
            # happens before the Query text_input claims its key on
            # the next rerun.
            st.button(
                snippet,
                key=f"hunt_quick_{idx}",
                use_container_width=True,
                help="Append to query",
                on_click=_append_to_query,
                kwargs={"snippet": snippet},
            )


def _render_hunt_cheatsheet() -> None:
    with st.expander("DSL cheat sheet", expanded=False):
        st.markdown(
            "**Syntax**\n"
            "- `field:value` — exact / substring / numeric match depending on field.\n"
            "- `field:\"two words\"` — quote values containing spaces.\n"
            "- `field:>x  field:<x  field:>=x  field:<=x` — comparisons (numeric / time fields).\n"
            "- `field:[a TO b]` — inclusive range (numeric / time fields).\n"
            "- `AND`, `OR`, `NOT`, parentheses, and a leading `-` for NOT.\n"
            "- A bare term with no field matches the incident narrative.\n\n"
            "**Examples**\n"
            "```\n"
            "label:malware AND confidence:>=0.8\n"
            "(label:phishing OR label:malware) AND last:24h\n"
            "severity:critical AND -status:closed\n"
            "narrative:\"personal dropbox\" confidence:[0.5 TO 1.0]\n"
            "mitre:T1566 OR mitre:T1190\n"
            "```\n\n"
            "**Fields**"
        )
        rows = [
            (s.name, ", ".join(s.aliases) or "—", s.kind, s.help)
            for s in HUNT_FIELDS
        ]
        st.dataframe(
            pd.DataFrame(rows, columns=["Field", "Aliases", "Kind", "Description"]),
            hide_index=True,
            use_container_width=True,
        )


def view_hunt() -> None:
    render_page_header(
        title="Hunt",
        subtitle="Lucene-style query language across past triage results.",
        breadcrumb="Console / Hunt",
    )

    db = _db()
    try:
        history = db.get_analysis_history(limit=10000) or []
    except Exception as exc:
        st.error(f"Could not load history: {exc}")
        return

    if not history:
        render_empty(
            "No history yet",
            "Run an Investigate or Batch analysis to populate the hunt index.",
        )
        return

    # Saved searches arrive as a "pending_search_filters" payload from
    # the sidebar. Translate legacy filter dicts into a DSL string so
    # old pins keep working, then drop the payload.
    pending = st.session_state.pop("pending_search_filters", None)
    if pending:
        st.session_state[_HUNT_QUERY_KEY] = _legacy_payload_to_dsl(pending)

    if _HUNT_QUERY_KEY not in st.session_state:
        st.session_state[_HUNT_QUERY_KEY] = _HUNT_DEFAULT_QUERY

    # 0.6-wide column reserves room for the Clear button next to the
    # Query input without crowding the Save-as field on the right.
    query_row = st.columns([3.4, 0.6, 2], gap="medium")
    with query_row[0]:
        st.text_input(
            "Query",
            key=_HUNT_QUERY_KEY,
            placeholder='label:malware AND confidence:>=0.8 AND last:24h',
            help="Lucene-style. AND / OR / NOT, parens, ranges supported.",
        )
    with query_row[1]:
        # Spacer matches the height of the Query label so the button
        # baseline aligns with the text input. Disabled when the query
        # is already empty so the click is a no-op rather than a rerun.
        st.markdown("<div style='height: 1.7rem;'></div>", unsafe_allow_html=True)
        st.button(
            "Clear",
            key="hunt_clear_btn",
            use_container_width=True,
            help="Empty the query box",
            on_click=_clear_query,
            disabled=not (st.session_state.get(_HUNT_QUERY_KEY) or "").strip(),
        )
    with query_row[2]:
        save_name = st.text_input(
            "Save as",
            value="",
            placeholder="e.g. 'High-severity last 24h'",
            key="hunt_save_name",
        )
        if st.button(
            "Save query",
            use_container_width=True,
            key="hunt_save_btn",
            disabled=not save_name.strip(),
        ):
            save_search(save_name, {"query": st.session_state[_HUNT_QUERY_KEY]})
            st.toast(f"Saved: {save_name}", icon=None)

    _render_hunt_helpers()
    _render_hunt_cheatsheet()

    query = st.session_state.get(_HUNT_QUERY_KEY, "")
    decorated = [_decorate_history_row(dict(h)) for h in history]

    try:
        predicate = compile_hunt_query(query)
    except HuntParseError as exc:
        caret = " " * exc.pos + "^"
        st.error(
            f"Query error: {exc.msg} (column {exc.pos + 1})\n\n"
            f"```\n{query}\n{caret}\n```"
        )
        return
    except Exception as exc:
        st.error(f"Query error: {exc}")
        return

    try:
        rows = [r for r in decorated if predicate(r)]
    except HuntParseError as exc:
        st.error(f"Query error: {exc.msg}")
        return

    render_section_head(
        "Results",
        action=f"{len(rows):,} of {len(history):,} events",
    )

    if not rows:
        render_empty(
            "No matches",
            "Loosen the query, drop a clause, or widen the time window.",
        )
        return

    rows = sorted(rows, key=lambda x: x.get("timestamp", ""), reverse=True)

    # Per-row interactive layout. Each row carries the same pills as the
    # legacy HTML table plus two action buttons: View opens the row in
    # Investigate (hydrated from the history record), Bookmark calls
    # add_bookmark directly. Capped at the first 100 rows because every
    # row creates two button widgets and Streamlit's per-rerun overhead
    # adds up; users with broader queries should narrow with the
    # filters above.
    LIMIT = 100
    visible = rows[:LIMIT]

    # Header strip
    h = st.columns([1.3, 0.8, 1.4, 1.1, 0.8, 1.0, 3.6, 0.9, 1.1], gap="small")
    headers = ["Time", "ID", "Class", "Status", "Conf", "Anomaly", "Narrative", "", ""]
    for col, label_txt in zip(h, headers):
        col.markdown(
            f'<div class="soc-hunt-th">{label_txt}</div>',
            unsafe_allow_html=True,
        )

    for r in visible:
        aid_raw = r.get("id") or r.get("analysis_id")
        if aid_raw is None:
            continue
        try:
            aid = int(aid_raw)
        except (TypeError, ValueError):
            continue
        dt = _safe_dt(r.get("timestamp"))
        when = dt.strftime("%b %d %H:%M") if dt else "-"
        label = r.get("final_label", "uncertain")
        try:
            conf = float(r.get("max_prob") or 0)
        except Exception:
            conf = 0.0
        anomaly = _anomaly_score(label, conf)
        body = (r.get("incident_text") or "").strip()
        body_short = body[:130] + ("..." if len(body) > 130 else "")
        status = get_case_status(aid)

        c = st.columns([1.3, 0.8, 1.4, 1.1, 0.8, 1.0, 3.6, 0.9, 1.1], gap="small")
        c[0].markdown(f'<div class="soc-hunt-cell soc-mono">{when}</div>', unsafe_allow_html=True)
        c[1].markdown(f'<div class="soc-hunt-cell soc-mono">#{aid}</div>', unsafe_allow_html=True)
        c[2].markdown(f'<div class="soc-hunt-cell">{severity_pill(label)}</div>', unsafe_allow_html=True)
        c[3].markdown(f'<div class="soc-hunt-cell">{_status_pill(status)}</div>', unsafe_allow_html=True)
        c[4].markdown(f'<div class="soc-hunt-cell soc-mono">{conf:.0%}</div>', unsafe_allow_html=True)
        c[5].markdown(f'<div class="soc-hunt-cell">{_anomaly_pill(anomaly)}</div>', unsafe_allow_html=True)
        c[6].markdown(f'<div class="soc-hunt-cell soc-hunt-narr">{body_short}</div>', unsafe_allow_html=True)
        if c[7].button("View", key=f"hunt_view_{aid}", help="Open in Investigate", use_container_width=True):
            st.session_state["current_analysis"] = _hydrate_history_to_analysis(r)
            st.session_state["view"] = "investigate"
            st.rerun()
        if c[8].button("Bookmark", key=f"hunt_bm_{aid}", help="Bookmark this incident", use_container_width=True):
            try:
                # add_bookmark's first positional arg is incident_text,
                # not analysis_id. Pass everything by keyword so the row
                # carries narrative + label + the link back to the
                # original analysis when we render Bookmarks later.
                _db().add_bookmark(
                    incident_text=r.get("incident_text", ""),
                    final_label=label,
                    note="",
                    analysis_id=aid,
                )
                append_timeline_event(aid, "bookmark", "Bookmarked from Hunt.")
                st.toast(f"#{aid} bookmarked")
            except Exception as exc:
                st.error(f"Could not bookmark: {exc}")

    if len(rows) > LIMIT:
        st.caption(
            f"Showing first {LIMIT} interactive of {len(rows):,} matches. "
            "Tighten the query (e.g. add `confidence:>=0.9` or `last:1h`) "
            "to drill into more."
        )


# =============================================================================
# BATCH PAGE
# =============================================================================


def _batch_settings_snapshot() -> dict[str, Any]:
    """Snapshot the sidebar inputs that materially shape batch output.

    Compared on every render against the snapshot taken when the
    cached results were produced; a divergence drives the drift
    warning at the top of the dashboard. Only the fields the analysis
    loop actually reads are included, so a no-op sidebar tweak (e.g.
    LLM provider switched while use_llm is off) won't trip the warning.
    API keys are intentionally excluded: rotating a key for the same
    provider/model produces equivalent output and shouldn't pester
    the user.
    """
    snap: dict[str, Any] = {
        "threshold": float(st.session_state.get("threshold", 0.5)),
        "max_classes": int(st.session_state.get("max_classes", 5)),
        "llm_assist_mode": st.session_state.get(
            "llm_assist_mode", LLM_ASSIST_FALLBACK
        ),
    }
    if snap["llm_assist_mode"] != LLM_ASSIST_OFF:
        llm = _resolve_llm_settings()
        provider = llm.get("provider")
        snap["llm_provider"] = provider
        if provider == "openai":
            snap["llm_model"] = llm.get("openai_model")
        elif provider == "anthropic":
            snap["llm_model"] = llm.get("anthropic_model")
        else:
            snap["llm_model"] = llm.get("hf_model")
    return snap


_BATCH_SETTING_LABELS = {
    "threshold": "threshold",
    "max_classes": "max classes",
    "llm_assist_mode": "LLM assist mode",
    "llm_provider": "LLM provider",
    "llm_model": "LLM model",
}


def view_batch() -> None:
    render_page_header(
        title="Batch",
        subtitle="Upload a CSV with an 'incident_text' or 'description' column to triage many events at once.",
        breadcrumb="Console / Batch",
    )

    cols = st.columns([3, 2], gap="large")

    with cols[0]:
        st.markdown(
            '<div class="soc-panel">'
            '<div class="soc-panel__title">Upload</div>',
            unsafe_allow_html=True,
        )
        uploaded = st.file_uploader(
            "CSV file",
            type=["csv"],
            help="Required column: incident_text (or description). Optional: id, source, severity.",
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with cols[1]:
        st.markdown(
            '<div class="soc-panel">'
            '<div class="soc-panel__title">Tips</div>'
            '<div style="font-size: 0.85rem; color: var(--soc-text-secondary); line-height: 1.55;">'
            '<ul style="margin: 0.25rem 0 0 1.1rem; padding: 0;">'
            '<li>One incident per row.</li>'
            '<li>Up to 500 rows per batch.</li>'
            '<li>LLM toggle in the sidebar applies per row.</li>'
            '</ul></div></div>',
            unsafe_allow_html=True,
        )

    if uploaded is None:
        # Upload cleared or never set: drop any stale results so the
        # next upload starts clean and download buttons from a prior
        # batch don't keep dangling on screen.
        for key in (
            "batch_results",
            "batch_id",
            "batch_elapsed",
            "batch_text_col",
            "batch_settings",
            "batch_llm_invoked",
            "batch_llm_overrode",
            "batch_llm_mode",
            "batch_force_pass_fired",
            "batch_rate_limited",
            "batch_provider_failures",
            "batch_provider_failure_messages",
            "batch_still_uncertain",
        ):
            st.session_state.pop(key, None)
        return

    # Streamlit's UploadedFile is BytesIO-backed and the same instance
    # is reused across reruns. pandas.read_csv advances the position to
    # EOF; without seeking back to 0 first, every rerun after the
    # initial parse (e.g. a download_button click) would re-read from
    # EOF, raise EmptyDataError, and early-return before the dashboard
    # render block could pick up the cached batch_results.
    try:
        uploaded.seek(0)
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    text_col = None
    for candidate in ("incident_text", "description", "narrative", "alert", "text"):
        if candidate in df.columns:
            text_col = candidate
            break
    if text_col is None:
        st.error("No usable text column found. Expected one of: incident_text, description, narrative, alert, text.")
        st.dataframe(df.head(5), use_container_width=True)
        return

    df = df.head(500).copy()
    st.caption(f"Detected {len(df)} rows. Using column: `{text_col}`.")

    # Demo data detected: surface the option to wipe synthetic rows
    # before the user batch lands, so Overview / Hunt / Settings views
    # don't mix demo and user data. Default-checked because the user
    # is past the demo phase by virtue of uploading their own CSV;
    # they can always rebuild via Settings then Backfill history.
    try:
        _demo_count = _db().count_demo_events()
    except Exception:
        _demo_count = 0
    if _demo_count > 0:
        st.info(
            f"Demo data detected: **{_demo_count:,}** synthetic incidents are "
            "in your database. Running batch analysis on your CSV will mix "
            "your results with demo data in the Overview, Hunt, and Settings "
            "views."
        )
        st.checkbox(
            "Clear demo data before running this batch (recommended)",
            value=True,
            key="batch_clear_demo_before_run",
            help=(
                "Removes the synthetic events from analysis history. "
                "Bookmarks of demo events stay (they keep their incident "
                "text) but lose their link to the analysis row. "
                "Reversible: rebuild from Settings then Backfill history."
            ),
        )

    # Run is a one-shot trigger: st.button only returns True on the
    # rerun that immediately follows the click. The analysis output and
    # the download buttons used to live below an `if not st.button(...):
    # return` guard, so any subsequent rerun (e.g. a download_button
    # click) returned early and wiped the whole batch dashboard. Now
    # the analysis runs once on the click and stashes its outputs in
    # session_state; the dashboard renders from session_state on every
    # rerun that has data.
    if st.button("Run batch", type="primary", key="batch_run"):
        # Clear demo data first if the user opted in via the checkbox
        # above. Done before the redo-deletion block so the order of
        # operations is: demo wipe -> redo wipe -> fresh analysis.
        # Setting _demo_emitter_paused stops the periodic 8s emitter
        # from immediately reseeding one row at a time for the rest
        # of this Streamlit session; a fresh app reload restarts it.
        if (
            _demo_count > 0
            and st.session_state.get("batch_clear_demo_before_run", True)
        ):
            try:
                cleared = _clear_demo_events()
                st.session_state["_demo_emitter_paused"] = True
                logger.info(
                    "Demo clear before user batch: removed %d demo rows; "
                    "emitter paused for the rest of this session.",
                    cleared,
                )
            except Exception as exc:
                logger.warning(
                    "Demo clear before user batch failed: %s", exc
                )

        # If a prior batch is still cached in session state, this click
        # is a redo on the same upload (a fresh upload would have run
        # the pop loop above and cleared batch_id). Drop the prior
        # batch's analysis_history rows so the redo replaces them
        # instead of stacking next to obsolete records.
        prior_id = st.session_state.get("batch_id")
        if prior_id:
            try:
                deleted = _db().delete_batch(prior_id)
                logger.info(
                    "Batch redo: deleted %d prior rows for batch %s",
                    deleted,
                    prior_id,
                )
            except Exception as exc:
                logger.warning(
                    "Batch redo: failed to delete prior batch %s: %s",
                    prior_id,
                    exc,
                )

        progress = st.progress(0)
        status = st.empty()
        batch_id = str(uuid.uuid4())[:8]
        threshold = float(st.session_state.get("threshold", 0.5))
        max_classes = int(st.session_state.get("max_classes", 5))
        mode = st.session_state.get("llm_assist_mode", LLM_ASSIST_FALLBACK)

        results = []
        llm_invoked = 0
        llm_overrode = 0
        force_pass_fired = 0
        rate_limited = 0
        provider_failures = 0
        provider_failure_messages: list[str] = []
        still_uncertain = 0
        total = len(df)
        t_start = time.time()
        logger.info("Batch start: id=%s mode=%s rows=%d", batch_id, mode, total)
        for i, row in enumerate(df[text_col].fillna("").astype(str).tolist()):
            status.markdown(
                f'<div class="soc-mono" style="color: var(--soc-text-secondary); font-size: 0.85rem;">'
                f'Processing {i+1:,} / {total:,}  ·  batch {batch_id}</div>',
                unsafe_allow_html=True,
            )
            if not row.strip():
                continue
            result = predict(row, threshold=threshold, max_classes=max_classes)
            logger.info(
                "[batch=%s][row=%d] sklearn label=%s conf=%.3f",
                batch_id, i + 1,
                result.get("final_label"),
                float(result.get("max_prob") or 0.0),
            )
            # Mirror Investigate's flow so batch outputs reflect LLM
            # judgment too. Without apply_llm_override here the LLM
            # was being called and billed in the prior code, but its
            # label was never used: the persisted final_label stayed
            # as sklearn's best guess (often 'uncertain'), which is
            # what made the user's 25-event sample show 14 unmapped.
            if _should_invoke_llm(result, mode, threshold):
                opinion, err, details = run_llm_with_forced_fallback(
                    row, skip_preprocessing=False
                )
                result["llm_opinion"] = opinion
                llm_invoked += 1
                if details.get("force_pass_attempted"):
                    force_pass_fired += 1
                # Single side-effecting call. apply_llm_override returns
                # False without mutating when opinion is None or label
                # is 'uncertain', so calling it once and branching on
                # the return is safe and avoids the double-mutation
                # hazard of calling it inside two if-branches.
                overrode = apply_llm_override(result, opinion)
                # Distinguish rate-limit drops from genuine parse/auth
                # failures so the diagnostics caption can point to the
                # right knob to turn. effective_err covers both the
                # force-pass error and the first-pass error (the
                # wrapper returns the first-pass opinion + force-pass
                # err on rate-limit during the second call).
                effective_err = err or details.get("first_pass_err")
                if overrode:
                    llm_overrode += 1
                elif effective_err and "rate limit" in effective_err.lower():
                    rate_limited += 1
                elif effective_err or not opinion:
                    provider_failures += 1
                    # Stash distinct error messages so the post-batch
                    # warning surfaces what actually went wrong (auth
                    # missing, model loading, content policy, etc.)
                    # instead of just a counter. Bounded by the
                    # warning's display cap of 3, but we keep more
                    # in case we want to expand later.
                    if (
                        effective_err
                        and effective_err not in provider_failure_messages
                        and len(provider_failure_messages) < 12
                    ):
                        provider_failure_messages.append(effective_err)
                # else: LLM responded with 'uncertain' that the
                # force-pass also couldn't escape. That row falls
                # through and shows up in the still_uncertain bucket
                # below.
            if result.get("final_label") == "uncertain":
                still_uncertain += 1
                logger.info(
                    "[batch=%s][row=%d] STILL_UNCERTAIN after LLM path",
                    batch_id, i + 1,
                )
            result["analysis_id"] = _persist_analysis(result, batch_id=batch_id)
            results.append(result)
            progress.progress((i + 1) / total)

        elapsed = time.time() - t_start
        progress.empty()
        status.empty()

        # Decorate so the Hunt DSL can filter batch results just like
        # history rows (`_severity`, `_anomaly`, `_status`, `_dt`).
        for r in results:
            _decorate_history_row(r)

        st.session_state["batch_results"] = results
        st.session_state["batch_id"] = batch_id
        st.session_state["batch_elapsed"] = elapsed
        st.session_state["batch_text_col"] = text_col
        st.session_state["batch_llm_invoked"] = llm_invoked
        st.session_state["batch_llm_overrode"] = llm_overrode
        st.session_state["batch_llm_mode"] = mode
        st.session_state["batch_force_pass_fired"] = force_pass_fired
        st.session_state["batch_rate_limited"] = rate_limited
        st.session_state["batch_provider_failures"] = provider_failures
        st.session_state["batch_provider_failure_messages"] = (
            provider_failure_messages
        )
        st.session_state["batch_still_uncertain"] = still_uncertain
        st.session_state["batch_settings"] = _batch_settings_snapshot()
        logger.info(
            "Batch end: id=%s invoked=%d overrode=%d force=%d "
            "rate_limited=%d provider_failures=%d still_uncertain=%d",
            batch_id, llm_invoked, llm_overrode, force_pass_fired,
            rate_limited, provider_failures, still_uncertain,
        )

    if "batch_results" not in st.session_state:
        return

    # Pull from session_state on every render so download_button clicks
    # (which trigger a full page rerun) and DSL filter edits don't lose
    # the analysis output. The CSV byte payloads passed into the three
    # download_buttons below get re-derived from these results on each
    # render; the derivation is pure pandas with no IO and produces
    # identical bytes for the same inputs, so the click that triggered
    # the rerun still resolves against stable data.
    results = st.session_state["batch_results"]
    batch_id = st.session_state["batch_id"]
    elapsed = st.session_state["batch_elapsed"]

    # Drift warning: if any sidebar input that materially affects the
    # analysis has changed since the cached run, name the diff and
    # prompt the user to click Run again. We don't auto-rerun because
    # that would silently re-bill the LLM and could surprise the user
    # mid-investigation; the warning leaves the prior dashboard fully
    # usable (downloads still work) and just explains why the chart
    # doesn't reflect the current sliders.
    cached_settings = st.session_state.get("batch_settings") or {}
    current_settings = _batch_settings_snapshot()
    drift_keys = [
        k for k in (set(cached_settings) | set(current_settings))
        if cached_settings.get(k) != current_settings.get(k)
    ]
    if drift_keys:
        labels = ", ".join(
            _BATCH_SETTING_LABELS.get(k, k) for k in sorted(drift_keys)
        )
        st.warning(
            f"Settings changed since last run: {labels}. "
            "Click Run batch to re-analyze with the current settings."
        )

    st.markdown("</br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size: 0.72rem; font-weight: 700; '
        'text-transform: uppercase; letter-spacing: 0.08em; '
        'color: var(--soc-text-muted); margin: 0.4rem 0 0.35rem 0;">'
        'Filter results (DSL)</div>',
        unsafe_allow_html=True,
    )
    batch_query = st.text_input(
        "Filter",
        value="",
        placeholder='label:malware AND confidence:>=0.8',
        help="Lucene-style query. Same DSL as the Hunt tab. Leave empty to keep all rows.",
        key=f"batch_query_{batch_id}",
        label_visibility="collapsed",
    )

    filtered = results
    if batch_query.strip():
        try:
            predicate = compile_hunt_query(batch_query)
            filtered = [r for r in results if predicate(r)]
        except HuntParseError as exc:
            caret = " " * exc.pos + "^"
            st.error(
                f"Query error: {exc.msg} (column {exc.pos + 1})\n\n"
                f"```\n{batch_query}\n{caret}\n```"
            )
            filtered = results
        except Exception as exc:
            st.error(f"Query error: {exc}")
            filtered = results

    st.caption(
        f"Showing {len(filtered):,} of {len(results):,} results."
        if batch_query.strip()
        else f"{len(results):,} rows in this batch."
    )

    # Aggregates downstream operate on the filtered subset so KPIs,
    # distribution, MITRE coverage, and exports all stay consistent
    # with what the analyst is currently looking at.
    results = filtered

    sev_counts: Counter[str] = Counter(severity_for(r["final_label"]) for r in results)
    label_counts: Counter[str] = Counter(r["final_label"] for r in results)

    cols = st.columns(4, gap="small")
    cols[0].markdown(
        render_kpi("Processed", f"{len(results):,}", sub=f"{elapsed:.1f}s wall", tone="info"),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        render_kpi(
            "Critical / high",
            f"{sev_counts.get('critical', 0) + sev_counts.get('high', 0):,}",
            sub="needs review", tone="critical",
        ),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        render_kpi(
            "Medium",
            f"{sev_counts.get('medium', 0):,}",
            sub="standard queue", tone="medium",
        ),
        unsafe_allow_html=True,
    )
    cols[3].markdown(
        render_kpi(
            "Benign / unknown",
            f"{sev_counts.get('low', 0) + sev_counts.get('info', 0):,}",
            sub="auto-close eligible", tone="low",
        ),
        unsafe_allow_html=True,
    )

    render_section_head("Distribution", "by classification")
    if label_counts:
        max_count = max(label_counts.values())
        rows_html = "".join(
            f'''<div class="soc-dist__row">
                <div class="soc-dist__name">{severity_pill(lbl)}</div>
                <div class="soc-dist__bar-wrap">
                    <div class="soc-dist__bar-fill" '''
            f'''style="width: {(c / max_count * 100):.1f}%; background: {SEVERITY_COLOR_HEX.get(severity_for(lbl), "#3b82f6")};"></div>
                </div>
                <div class="soc-dist__count">{c:,}</div>
            </div>'''
            for lbl, c in label_counts.most_common()
        )
        st.markdown(
            '<div class="soc-panel">'
            f'<div class="soc-panel__title">Labels  '
            f'<span class="soc-meta">batch {batch_id}</span></div>'
            f'<div class="soc-dist">{rows_html}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # MITRE coverage report
    render_section_head(
        "MITRE coverage",
        action="tactic-by-technique aggregation",
    )
    coverage_df = _build_coverage_report(results)
    st.markdown(render_coverage_summary(coverage_df), unsafe_allow_html=True)

    # Detailed coverage table (tactic, technique, label, count)
    if not coverage_df.empty:
        with st.expander("Detailed coverage table", expanded=False):
            st.dataframe(
                coverage_df,
                use_container_width=True,
                hide_index=True,
            )

    # ---- Exports ----
    render_section_head("Exports", action="download artifacts")
    # Reconciliation caption: the triage CSV has one row per event, but
    # the MITRE coverage and tactic rollup exports group by technique,
    # which can multiply rows (events with multiple techniques) and
    # historically dropped rows (events whose label has no technique).
    # The Unmapped bucket added in _build_coverage_report keeps the
    # coverage totals reconcilable; this caption summarizes the split
    # so the user can sanity-check at a glance.
    _mapped_events = sum(
        1 for r in results if MITRE_MAPPING.get(r.get("final_label", "uncertain"))
    )
    _unmapped_events = len(results) - _mapped_events
    st.caption(
        f"Total events: {len(results):,}  ·  "
        f"Mapped to MITRE: {_mapped_events:,}  ·  "
        f"Unmapped: {_unmapped_events:,}"
    )
    # Cost-transparency line: shows how many events the LLM was actually
    # called for and how many of those calls produced a label override.
    # In Override mode "called" equals total; in Fallback mode it's the
    # subset where sklearn was uncertain or barely above threshold; in
    # Off mode this line is suppressed because there's nothing to say.
    _llm_mode = st.session_state.get("batch_llm_mode", LLM_ASSIST_OFF)
    if _llm_mode != LLM_ASSIST_OFF:
        _llm_invoked = int(st.session_state.get("batch_llm_invoked", 0))
        _llm_overrode = int(st.session_state.get("batch_llm_overrode", 0))
        _force_fired = int(st.session_state.get("batch_force_pass_fired", 0))
        _rate_limited = int(st.session_state.get("batch_rate_limited", 0))
        _provider_fail = int(
            st.session_state.get("batch_provider_failures", 0)
        )
        _provider_fail_msgs = list(
            st.session_state.get("batch_provider_failure_messages", []) or []
        )
        _still_unc = int(st.session_state.get("batch_still_uncertain", 0))
        _mode_label = {
            LLM_ASSIST_FALLBACK: "Fallback mode",
            LLM_ASSIST_OVERRIDE: "Override mode",
        }.get(_llm_mode, _llm_mode)
        st.caption(
            f"LLM-assisted: {_llm_invoked:,} of {len(results):,} events  ·  "
            f"Override fired: {_llm_overrode:,}  ·  "
            f"Force-pass fired: {_force_fired:,}  ·  {_mode_label}"
        )
        # Diagnostics line: surfaces rate-limit drops, real upstream
        # provider failures, and events that ended up uncertain
        # anyway. A high rate-limited count points at the per-
        # provider cap in llm_helpers.effective_rate_window; a high
        # provider-failures count points at upstream auth / network
        # / model-loading / content-policy issues (see the warning
        # below for the actual error strings); a high
        # still_uncertain with both other counts low points at the
        # LLM model genuinely returning uncertain even after force-
        # pass.
        st.caption(
            f"Diagnostics: rate-limited: {_rate_limited:,}  ·  "
            f"provider-failures: {_provider_fail:,}  ·  "
            f"uncertain after force: {_still_unc:,}"
        )
        # Surface the actual upstream error strings so the user can
        # tell auth issues from rate limits from model-loading delays
        # without grepping the streamlit log. Cap at 3 distinct
        # messages to keep the warning readable; the full set sits in
        # session_state for anyone who wants to inspect via st.write.
        if _provider_fail > 0 and _provider_fail_msgs:
            _shown = _provider_fail_msgs[:3]
            _bullet_lines = "\n".join(f'- "{m}"' for m in _shown)
            _suffix = (
                f" (showing {len(_shown)} of {_provider_fail})"
                if len(_provider_fail_msgs) < _provider_fail
                or len(_shown) < len(_provider_fail_msgs)
                else ""
            )
            st.warning(
                f"Provider failures observed{_suffix}:\n{_bullet_lines}"
            )
    export_cols = st.columns(3, gap="small")

    out_df = pd.DataFrame([
        {
            "analysis_id": r.get("analysis_id"),
            "label": r["final_label"],
            "severity": severity_for(r["final_label"]),
            "confidence": float(r.get("max_prob", 0)),
            "anomaly_score": _anomaly_score(
                r["final_label"], float(r.get("max_prob", 0))
            ),
            "incident_text": r["incident_text"],
        }
        for r in results
    ])
    with export_cols[0]:
        st.download_button(
            "Triage results CSV",
            data=out_df.to_csv(index=False).encode("utf-8"),
            file_name=f"alertsage-triage-{batch_id}.csv",
            mime="text/csv",
            key="batch_download",
            use_container_width=True,
        )
    with export_cols[1]:
        st.download_button(
            "MITRE coverage CSV",
            data=coverage_df.to_csv(index=False).encode("utf-8"),
            file_name=f"alertsage-mitre-coverage-{batch_id}.csv",
            mime="text/csv",
            key="coverage_download",
            disabled=coverage_df.empty,
            use_container_width=True,
        )
    with export_cols[2]:
        # Tactic-level rollup for executive consumption
        if not coverage_df.empty:
            rollup = coverage_df.groupby(
                ["tactic_id", "tactic"], as_index=False
            ).agg(
                events=("events", "sum"),
                techniques=("technique", "nunique"),
                critical=("critical", "sum"),
                high=("high", "sum"),
                medium=("medium", "sum"),
                low=("low", "sum"),
                info=("info", "sum"),
            )
        else:
            rollup = pd.DataFrame()
        st.download_button(
            "Tactic rollup CSV",
            data=rollup.to_csv(index=False).encode("utf-8"),
            file_name=f"alertsage-tactic-rollup-{batch_id}.csv",
            mime="text/csv",
            key="rollup_download",
            disabled=rollup.empty,
            use_container_width=True,
        )


# =============================================================================
# BOOKMARKS PAGE
# =============================================================================

def view_bookmarks() -> None:
    render_page_header(
        title="Bookmarks",
        subtitle="Saved investigations and analyst notes for quick recall.",
        breadcrumb="Console / Bookmarks",
    )

    db = _db()
    try:
        bookmarks = db.get_bookmarks(limit=200) or []
    except Exception as exc:
        st.error(f"Could not load bookmarks: {exc}")
        return

    if not bookmarks:
        render_empty(
            "No bookmarks yet",
            "Hit <strong>Bookmark</strong> on an analysis to save it here.",
        )
        return

    render_section_head("Saved", action=f"{len(bookmarks):,} entries")
    for bm in bookmarks:
        # The bookmarks table stores incident_text, final_label, and
        # created_at. max_prob / uncertainty / mitre live on the linked
        # analysis_history row, so fetch that when analysis_id is set.
        bm_id = bm.get("id")
        analysis_id_raw = bm.get("analysis_id")
        analysis_id = int(analysis_id_raw) if analysis_id_raw else None

        linked = None
        if analysis_id is not None:
            try:
                linked = db.get_analysis_by_id(analysis_id)
            except Exception:
                linked = None

        label = (
            bm.get("final_label")
            or (linked or {}).get("final_label")
            or "uncertain"
        )
        body = (
            bm.get("incident_text")
            or (linked or {}).get("incident_text")
            or ""
        ).strip()
        try:
            conf = float((linked or {}).get("max_prob") or 0)
        except Exception:
            conf = 0.0
        # Prefer the original analysis timestamp; fall back to when the
        # bookmark itself was created.
        ts_str = (linked or {}).get("timestamp") or bm.get("created_at")
        ts = _safe_dt(ts_str)
        when = ts.strftime("%b %d, %Y at %H:%M") if ts else "Unknown time"
        note = (bm.get("note") or "").strip()
        status = get_case_status(analysis_id) if analysis_id else "new"
        anomaly = _anomaly_score(label, conf)

        title = f"{humanize(label)}  ·  {when}  ·  status: {status.upper()}"
        with st.expander(title, expanded=False):
            head_cols = st.columns([6, 1])
            with head_cols[0]:
                st.markdown(
                    f'<div style="display:flex; gap:0.4rem; align-items:center;">'
                    f'{severity_pill(label)}'
                    f'{_status_pill(status)}'
                    f'<span class="soc-tag accent">conf {conf:.0%}</span>'
                    f'<span class="soc-tag">anomaly {anomaly}</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            with head_cols[1]:
                if st.button(
                    "Remove", key=f"rm_{bm_id}",
                    use_container_width=True, type="secondary",
                ):
                    try:
                        db.delete_bookmark(int(bm_id))
                        st.session_state["cached_bookmarks"] = None
                        st.success("Removed.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not remove: {exc}")

            st.markdown(
                f'<div class="soc-narrative tone-{severity_for(label)}" '
                'style="margin-top: 0.55rem;">'
                f'{body}</div>',
                unsafe_allow_html=True,
            )

            # Status stepper + advance buttons
            if analysis_id:
                st.markdown(
                    render_case_stepper(analysis_id, status),
                    unsafe_allow_html=True,
                )
                cur_idx = _CASE_STATUS_KEYS.index(status)
                status_cols = st.columns([1, 1, 1, 1, 4], gap="small")
                for idx, (key, label_text, _) in enumerate(CASE_STATUSES):
                    with status_cols[idx]:
                        if st.button(
                            label_text,
                            key=f"bm_status_{key}_{analysis_id}",
                            use_container_width=True,
                            type="primary" if idx == cur_idx else "secondary",
                            disabled=(idx == cur_idx),
                        ):
                            set_case_status(analysis_id, key)
                            st.rerun()

            if note:
                st.markdown(
                    '<div style="margin-top: 0.6rem; font-size: 0.78rem; '
                    'color: var(--soc-text-muted); text-transform: uppercase; '
                    'letter-spacing: 0.06em;">Analyst note</div>'
                    f'<div style="font-size: 0.92rem; line-height: 1.55;">{note}</div>',
                    unsafe_allow_html=True,
                )

            # Case timeline (status changes + notes + bookmarks)
            if analysis_id:
                st.markdown(
                    render_case_timeline(analysis_id),
                    unsafe_allow_html=True,
                )


# =============================================================================
# SETTINGS PAGE
# =============================================================================

def view_settings() -> None:
    render_page_header(
        title="Settings",
        subtitle="LLM provider, models, and Bring Your Own Key. Nothing here is persisted to disk.",
        breadcrumb="Console / Settings",
    )

    # Provider picker
    local_ok = local_gguf_available()
    st.markdown(
        '<div class="soc-panel"><div class="soc-panel__title">LLM provider</div>',
        unsafe_allow_html=True,
    )

    options = []
    if local_ok:
        options.append(("local", "Local llama.cpp"))
    options.append(("huggingface", "Hugging Face Inference"))
    options.append(("openai", "OpenAI"))
    options.append(("anthropic", "Anthropic"))

    current_provider = st.session_state.get("llm_provider") or _default_provider()
    available_keys = [v for v, _ in options]
    if current_provider not in available_keys:
        current_provider = available_keys[0]

    selected = st.radio(
        "Provider",
        options=[v for v, _ in options],
        format_func=lambda v: dict(options)[v],
        index=available_keys.index(current_provider),
        horizontal=True,
        key="settings_provider_radio",
    )
    st.session_state["llm_provider"] = selected
    if not local_ok:
        st.caption("Local provider hidden: install llama-cpp-python and place a .gguf in models/ to enable.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Per-provider configuration panels
    if selected == "huggingface":
        _settings_panel_huggingface()
    elif selected == "openai":
        _settings_panel_openai()
    elif selected == "anthropic":
        _settings_panel_anthropic()
    else:
        st.markdown(
            '<div class="soc-panel">'
            '<div class="soc-panel__title">Local llama.cpp</div>'
            '<div style="color: var(--soc-text-secondary); font-size: 0.9rem; line-height: 1.55;">'
            'Local mode loads the .gguf model from <code>models/</code> on first call. '
            'No external network. Use this when you need full air-gap.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    # Triage defaults
    st.markdown(
        '<div class="soc-panel"><div class="soc-panel__title">Triage defaults</div>',
        unsafe_allow_html=True,
    )
    cc = st.columns(3, gap="medium")
    with cc[0]:
        st.session_state["threshold"] = st.slider(
            "Confidence threshold",
            0.0, 1.0,
            float(st.session_state.get("threshold", 0.5)),
            0.05,
            key="settings_threshold",
        )
    with cc[1]:
        st.session_state["max_classes"] = st.slider(
            "Probability rows",
            1, 10,
            int(st.session_state.get("max_classes", 5)),
            1,
            key="settings_max_classes",
        )
    with cc[2]:
        st.session_state["use_preprocessing"] = st.checkbox(
            "Text preprocessing",
            value=bool(st.session_state.get("use_preprocessing", True)),
            key="settings_preproc",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Threat intel enrichment ----
    st.markdown(
        '<div class="soc-panel">'
        '<div class="soc-panel__title">Threat intel enrichment</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([3, 2], gap="medium")
    with cols[0]:
        st.markdown(
            'Pasting a VirusTotal API key here switches the IOC '
            'enrichment panel from the deterministic mock to a live '
            'lookup. Free-tier keys work; results are cached for 15 '
            'minutes per indicator.'
        )
    with cols[1]:
        existing = st.session_state.get(_VT_KEY_SETTING, "")
        st.session_state["vt_byo_key"] = st.checkbox(
            "Bring my own VT key",
            value=bool(st.session_state.get("vt_byo_key", False)),
            key="settings_vt_byo",
        )
        if st.session_state["vt_byo_key"]:
            entered = st.text_input(
                "VirusTotal API key",
                value="",
                type="password",
                placeholder="Paste key, session only",
                key="settings_vt_key",
            )
            if entered:
                st.session_state[_VT_KEY_SETTING] = entered.strip()
                st.toast("VT key set for this session.", icon=None)
        st.caption(
            "Live VT lookup active." if existing
            else "Mocked enrichment. Add a key to enable live VT."
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Demo data generator ----
    st.markdown(
        '<div class="soc-panel">'
        '<div class="soc-panel__title">Demo data generator '
        '<span class="soc-meta">populates the live tail with synthetic events</span></div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([3, 2], gap="medium")
    with cols[0]:
        st.markdown(
            'When on, AlertSage emits one synthetic incident every '
            '~6 seconds drawn from a curated set of phishing, malware, '
            'access abuse, web attack, exfiltration, and benign '
            'narratives. Each event is classified, persisted, and '
            'seeded into the case timeline. The Overview live tail '
            'picks them up automatically.'
        )
    with cols[1]:
        active = st.checkbox(
            "Run demo generator",
            value=bool(st.session_state.get(_DEMO_FLAG_KEY, False)),
            key="settings_demo_flag",
        )
        st.session_state[_DEMO_FLAG_KEY] = active
        st.caption(
            "Auto-emitting every ~6s." if active
            else "Off. Only your own triage runs will appear."
        )

        emit_cols = st.columns(2, gap="small")
        with emit_cols[0]:
            if st.button(
                "Emit one now",
                use_container_width=True,
                key="settings_demo_emit_one",
                type="primary",
            ):
                aid, err = emit_demo_event()
                if err:
                    st.error(err)
                else:
                    st.toast(f"Emitted event #{aid}.", icon=None)
                    st.rerun()
        with emit_cols[1]:
            if st.button(
                "Clear demo events",
                use_container_width=True,
                key="settings_demo_clear",
                type="secondary",
            ):
                n = _clear_demo_events()
                st.toast(f"Removed {n} demo events.", icon=None)

        # Backfill row: seed 30 days of synthetic events so charts look
        # populated immediately. This is the demo-appearance lever.
        backfill_cols = st.columns([2, 1, 1], gap="small")
        with backfill_cols[0]:
            backfill_count = st.slider(
                "Backfill volume",
                min_value=30, max_value=400, value=150, step=10,
                key="settings_backfill_count",
                help="Number of synthetic events to spread across the last 30 days.",
            )
        with backfill_cols[1]:
            backfill_days = st.selectbox(
                "Window",
                options=[7, 14, 30, 60],
                index=2,
                key="settings_backfill_days",
                format_func=lambda d: f"{d} days",
            )
        with backfill_cols[2]:
            st.markdown('<div style="height: 1.7rem;"></div>', unsafe_allow_html=True)
            if st.button(
                "Backfill history",
                use_container_width=True,
                key="settings_demo_seed",
                type="primary",
            ):
                with st.spinner(
                    f"Seeding {backfill_count} events across {backfill_days} days..."
                ):
                    n, err = seed_historical_events(
                        days=int(backfill_days),
                        count=int(backfill_count),
                    )
                if err:
                    st.error(f"Seeded {n} rows then failed: {err}")
                else:
                    st.session_state[_DEMO_COUNT_KEY] = (
                        int(st.session_state.get(_DEMO_COUNT_KEY, 0)) + n
                    )
                    st.toast(
                        f"Seeded {n} synthetic events across the last "
                        f"{backfill_days} days.",
                        icon=None,
                    )
                    st.rerun()

    # Status row: count + last emit + last error
    count = int(st.session_state.get(_DEMO_COUNT_KEY, 0))
    last_emit = st.session_state.get(_DEMO_LAST_EMIT_KEY)
    last_err = st.session_state.get(_DEMO_LAST_ERR_KEY)

    last_emit_str = "never"
    if last_emit:
        try:
            ts = datetime.fromisoformat(last_emit).astimezone()
            last_emit_str = ts.strftime("%H:%M:%S")
        except Exception:
            last_emit_str = str(last_emit)[:19]

    status_cols = st.columns(3, gap="small")
    status_cols[0].markdown(
        '<div class="soc-panel__title" style="margin: 0;">Emitted this session</div>'
        f'<span class="soc-cell-mono" style="font-size: 1.1rem; '
        f'color: var(--soc-text-strong);">{count}</span>',
        unsafe_allow_html=True,
    )
    status_cols[1].markdown(
        '<div class="soc-panel__title" style="margin: 0;">Last emit</div>'
        f'<span class="soc-cell-mono" style="color: var(--soc-text-secondary);">{last_emit_str}</span>',
        unsafe_allow_html=True,
    )
    status_cols[2].markdown(
        '<div class="soc-panel__title" style="margin: 0;">Last error</div>'
        f'<span class="soc-cell-mono" style="color: '
        f'{"var(--soc-danger)" if last_err else "var(--soc-text-muted)"};">'
        f'{last_err or "none"}</span>',
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # About
    st.markdown(
        '<div class="soc-panel">'
        '<div class="soc-panel__title">About</div>'
        '<div style="font-size: 0.88rem; line-height: 1.6; color: var(--soc-text-secondary);">'
        '<strong>AlertSage</strong> classifies free-text security incidents into '
        'a fixed taxonomy (phishing, malware, access abuse, data exfiltration, '
        'and so on), maps them to MITRE ATT&CK, and optionally requests an LLM '
        'second opinion through your provider of choice.<br><br>'
        '<strong>Stack:</strong> Logistic regression + TF-IDF for the classifier, '
        'sentence-transformers for similarity, SQLite for history, '
        'Streamlit for the UI.<br>'
        '<strong>Data privacy:</strong> Bring-Your-Own-Key fields live in session '
        'state only, never written to disk.</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def _settings_panel_huggingface() -> None:
    st.markdown(
        '<div class="soc-panel">'
        '<div class="soc-panel__title">Hugging Face Inference</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([3, 2], gap="medium")
    with cols[0]:
        st.session_state["hf_model_id"] = st.text_input(
            "Model id",
            value=st.session_state.get("hf_model_id", HF_DEFAULT_MODEL),
            help="Examples: meta-llama/Llama-3.1-8B-Instruct:cerebras",
            key="settings_hf_model",
        )
    with cols[1]:
        st.session_state["hf_byo_token"] = st.checkbox(
            "Bring my own token",
            value=bool(st.session_state.get("hf_byo_token", False)),
            key="settings_hf_byo",
        )
        if st.session_state["hf_byo_token"]:
            entered = st.text_input(
                "HF token",
                value="",
                type="password",
                placeholder="hf_xxx",
                help="Session only. Never written to disk.",
                key="settings_hf_token",
            )
            if entered:
                st.session_state["selected_hf_token"] = entered.strip()
    st.markdown("</div>", unsafe_allow_html=True)


def _fetch_provider_models(
    provider: str, api_key: str
) -> tuple[list[str] | None, str | None]:
    """Fetch the model list for a given provider, cached per session+key.

    The cache is keyed on the API key so rotating keys forces a refresh.
    Returns (model_ids, error_message). One of the two is always None.
    """
    cache = st.session_state.setdefault("_provider_model_cache", {})
    cache_key = (provider, api_key)
    if cache_key in cache:
        return cache[cache_key]

    try:
        if provider == "anthropic":
            models = list_anthropic_models(api_key)
        elif provider == "openai":
            models = list_openai_models(api_key)
        else:
            raise ValueError(f"Unknown provider for model fetch: {provider}")
        result: tuple[list[str] | None, str | None] = (models, None)
    except Exception as exc:
        result = (None, str(exc))

    cache[cache_key] = result
    return result


def _model_picker(
    provider: str,
    *,
    api_key: str,
    session_state_key: str,
    settings_widget_key: str,
    default_model: str,
    text_placeholder: str,
) -> None:
    """Render the model id selector.

    Auto-fetches the provider's model list when a key is present and
    renders a selectbox; otherwise falls back to a manual text input.
    """
    if not api_key:
        st.session_state[session_state_key] = st.text_input(
            "Model id",
            value=st.session_state.get(session_state_key, default_model),
            help=f"Examples: {text_placeholder}. Add a key to auto-load available models.",
            key=settings_widget_key,
        )
        return

    refresh_key = f"{settings_widget_key}_refresh"
    cols = st.columns([5, 1], gap="small")
    with cols[1]:
        st.markdown('<div style="height: 1.7rem;"></div>', unsafe_allow_html=True)
        if st.button("Refresh", key=refresh_key, help="Refetch model list"):
            cache = st.session_state.get("_provider_model_cache", {})
            cache.pop((provider, api_key), None)

    models, err = _fetch_provider_models(provider, api_key)
    with cols[0]:
        if err or not models:
            st.session_state[session_state_key] = st.text_input(
                "Model id",
                value=st.session_state.get(session_state_key, default_model),
                help=f"Could not auto-load models ({err or 'empty list'}). Enter manually.",
                key=settings_widget_key,
            )
            return

        current = st.session_state.get(session_state_key, default_model)
        if current not in models:
            models = [current, *models] if current else models
        try:
            idx = models.index(current)
        except ValueError:
            idx = 0
        st.session_state[session_state_key] = st.selectbox(
            "Model",
            options=models,
            index=idx,
            help=f"Auto-loaded from {provider.title()} for the supplied key.",
            key=settings_widget_key,
        )


def _settings_panel_openai() -> None:
    st.markdown(
        '<div class="soc-panel">'
        '<div class="soc-panel__title">OpenAI</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([3, 2], gap="medium")
    with cols[1]:
        st.session_state["openai_byo_key"] = st.checkbox(
            "Bring my own key",
            value=bool(st.session_state.get("openai_byo_key", False)),
            key="settings_oa_byo",
        )
        if st.session_state["openai_byo_key"]:
            entered = st.text_input(
                "OpenAI key",
                value="",
                type="password",
                placeholder="sk-...",
                help="Session only. Never written to disk.",
                key="settings_oa_key",
            )
            if entered:
                st.session_state["selected_openai_api_key"] = entered.strip()
    with cols[0]:
        _model_picker(
            "openai",
            api_key=st.session_state.get("selected_openai_api_key", "")
            or _secret("OPENAI_API_KEY") or _env("OPENAI_API_KEY") or "",
            session_state_key="openai_model_id",
            settings_widget_key="settings_oa_model",
            default_model=DEFAULT_OPENAI_MODEL,
            text_placeholder="gpt-4o-mini, gpt-4o, gpt-4.1-mini",
        )
    has_key = bool(st.session_state.get("selected_openai_api_key"))
    st.caption(
        "Key set for this session." if has_key
        else "No key set. Calls will fall back to the Hugging Face demo."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _settings_panel_anthropic() -> None:
    st.markdown(
        '<div class="soc-panel">'
        '<div class="soc-panel__title">Anthropic</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([3, 2], gap="medium")
    with cols[1]:
        st.session_state["anthropic_byo_key"] = st.checkbox(
            "Bring my own key",
            value=bool(st.session_state.get("anthropic_byo_key", False)),
            key="settings_an_byo",
        )
        if st.session_state["anthropic_byo_key"]:
            entered = st.text_input(
                "Anthropic key",
                value="",
                type="password",
                placeholder="sk-ant-...",
                help="Session only. Never written to disk.",
                key="settings_an_key",
            )
            if entered:
                st.session_state["selected_anthropic_api_key"] = entered.strip()
    with cols[0]:
        _model_picker(
            "anthropic",
            api_key=st.session_state.get("selected_anthropic_api_key", "")
            or _secret("ANTHROPIC_API_KEY") or _env("ANTHROPIC_API_KEY") or "",
            session_state_key="anthropic_model_id",
            settings_widget_key="settings_an_model",
            default_model=DEFAULT_ANTHROPIC_MODEL,
            text_placeholder="claude-haiku-4-5, claude-sonnet-4-6",
        )
    has_key = bool(st.session_state.get("selected_anthropic_api_key"))
    st.caption(
        "Key set for this session." if has_key
        else "No key set. Calls will fall back to the Hugging Face demo."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# ROUTER
# =============================================================================

def main() -> None:
    # Render the sidebar FIRST so navigation paints before any work that
    # could delay first paint. Previously ensure_demo_data_seeded ran up
    # top and on Streamlit Cloud (with IS_HOSTED_DEMO=1) the synchronous
    # batch insert plus the auto-mounted fragment rerun storm during
    # cold start broke sidebar visibility entirely. Sidebar now always
    # renders, and seed/fragment work happens after the user already has
    # navigation chrome.
    render_sidebar()
    view = st.session_state.get("view", "overview")

    # First-cold-start auto-seed for the hosted demo. Idempotent and
    # cached so this fires once per process at most. Runs AFTER sidebar
    # so a slow seed cannot block the sidebar from rendering.
    ensure_demo_data_seeded()

    # Demo data generator: only mount the every-8s rerun fragment if the
    # user has explicitly toggled it on in Settings. Auto-mounting it on
    # the hosted demo previously caused continuous reruns during cold
    # start that interacted badly with sidebar state in newer Streamlit
    # versions. Auto-seed (above) still populates the dashboard for
    # visitors; the live tail just stops growing on its own.
    if st.session_state.get("demo_generator_on", False):
        demo_generator_fragment()

    if view == "overview":
        view_overview()
    elif view == "investigate":
        view_investigate()
    elif view == "hunt":
        view_hunt()
    elif view == "batch":
        view_batch()
    elif view == "bookmarks":
        view_bookmarks()
    elif view == "settings":
        view_settings()
    else:
        view_overview()


main()
