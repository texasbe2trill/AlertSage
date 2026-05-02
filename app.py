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

from src.triage.database import TriageDatabase
from src.triage.embeddings import get_embedder
from src.triage.llm_helpers import (
    MITRE_MAPPING,
    build_llm_rationale,
    llm_second_opinion,
    soc_triage_hint,
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
    ("TA0010", "Exfiltration",        ["T1041", "T1048", "T1567", "T1020"]),
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

_DEFAULTS: dict[str, Any] = {
    "view": "overview",
    "selected_bookmark": None,
    "current_analysis": None,
    "investigate_text": "",
    # LLM provider state (kept in session only; never persisted)
    "llm_provider": None,
    "use_llm": True,
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

def _resolve_llm_settings() -> dict[str, Any]:
    """Snapshot the provider configuration from session/secrets/env."""
    hf_secret_token = _secret("HF_TOKEN")
    hf_secret_model = _secret("HF_MODEL")
    hf_env_token = _env("TRIAGE_HF_TOKEN", "HF_TOKEN")
    hf_env_model = _env("TRIAGE_HF_MODEL", "HF_MODEL")

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
        "openai_model": st.session_state.get("openai_model_id", DEFAULT_OPENAI_MODEL),
        "openai_api_key": st.session_state.get("selected_openai_api_key", ""),
        "anthropic_model": st.session_state.get(
            "anthropic_model_id", DEFAULT_ANTHROPIC_MODEL
        ),
        "anthropic_api_key": st.session_state.get(
            "selected_anthropic_api_key", ""
        ),
    }


def _default_provider() -> str:
    """Pick a sensible default provider on first load."""
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


def _provider_rate_check(provider: str) -> tuple[bool, float]:
    """Per-provider sliding-window rate limiter (session scoped)."""
    bucket_key = f"_rl_{provider}"
    now = datetime.now(timezone.utc).timestamp()
    window_start = now - RATE_LIMIT_WINDOW_S
    timestamps = [t for t in st.session_state.get(bucket_key, []) if t >= window_start]
    if len(timestamps) >= RATE_LIMIT_REQS:
        retry_after = RATE_LIMIT_WINDOW_S - (now - timestamps[0])
        st.session_state[bucket_key] = timestamps
        return False, max(retry_after, 0.0)
    timestamps.append(now)
    st.session_state[bucket_key] = timestamps
    return True, 0.0


def run_llm_second_opinion(
    text: str, *, skip_preprocessing: bool = False
) -> tuple[dict | None, str | None]:
    """Single dispatch helper used by every page that calls the LLM."""
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
            **kwargs,
        )
        return opinion, None
    except Exception as exc:  # pragma: no cover - network dependent
        return None, str(exc)


# =============================================================================
# CLASSIFIER FRONT-END
# =============================================================================

def predict(text: str, *, threshold: float, max_classes: int) -> dict[str, Any]:
    """Run the TF-IDF + LogReg classifier and return a normalized result."""
    vectorizer, model = _classifier()
    cleaned = clean_description(text)
    X = vectorizer.transform([cleaned])
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


def render_topbar(active_view: str) -> None:
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    settings = _resolve_llm_settings()
    provider = settings.get("provider", "local")
    provider_label = {
        "local": "Local llama.cpp",
        "huggingface": "Hugging Face",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
    }.get(provider, "Local")

    page_meta = {
        "overview": "Mission control",
        "investigate": "Triage console",
        "hunt": "Hunt and search",
        "batch": "Batch processor",
        "bookmarks": "Bookmarks",
        "settings": "Configuration",
    }.get(active_view, "")

    logo_html = ""
    if _LOGO_PATH.exists():
        b64 = base64.b64encode(_LOGO_PATH.read_bytes()).decode()
        logo_html = (
            f'<img src="data:image/svg+xml;base64,{b64}" '
            'style="height: 22px; width: auto;" alt="AlertSage" />'
        )

    st.markdown(
        f"""
        <div class="soc-topbar">
            <div class="soc-topbar__brand">
                {logo_html}
                <span>AlertSage</span>
                <span class="soc-topbar__brand-tag">SOC</span>
            </div>
            <div class="soc-topbar__center soc-mono">{page_meta}  ·  {now_iso}</div>
            <div class="soc-topbar__status">
                <span class="soc-status-pill"><span class="dot"></span>classifier ready</span>
                <span class="soc-status-pill info"><span class="dot"></span>llm: {provider_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    """
    active_set = {t.upper() for t in active_techniques or []}
    cells = []
    for tactic_id, tactic_name, techs in KILL_CHAIN_STAGES:
        hits = [t for t in techs if t.upper() in active_set]
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
        f'<span class="soc-meta">{len(active_set)} technique'
        f'{"s" if len(active_set) != 1 else ""} mapped</span></div>'
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


@st.fragment(run_every="5s")
def render_live_tail_fragment(n: int = 6) -> None:
    """Auto-refreshing live tail.

    Re-queries the database every 5 seconds and re-renders independently
    of the rest of the page. The user sees new triage events flow in
    without a manual refresh.
    """
    try:
        history = _db().get_analysis_history(limit=200) or []
    except Exception:
        history = []
    st.markdown(render_live_tail(history, n=n), unsafe_allow_html=True)


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
        _db().save_setting(_case_status_key(analysis_id), status)
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


def render_ioc_panel(text: str) -> str:
    iocs = extract_iocs(text)
    if not iocs:
        return (
            '<div class="soc-panel">'
            '<div class="soc-panel__title">Indicators '
            '<span class="soc-meta">no observables found</span></div>'
            '<div style="color: var(--soc-text-muted); font-size: 0.85rem;">'
            'No IPs, hashes, domains, URLs, emails, CVEs, or hostnames '
            'detected in this narrative.</div>'
            '</div>'
        )
    rows = []
    for ioc in iocs[:30]:
        enrichment = _mock_enrich(ioc)
        rows.append(
            "<tr>"
            f'<td class="soc-cell-mono">{ioc["indicator"]}</td>'
            f'<td><span class="soc-tag soc-mono">{ioc["type"]}</span></td>'
            f'<td><span class="soc-pill {enrichment["verdict_tone"]}">{enrichment["verdict"]}</span></td>'
            f'<td class="soc-cell-mono">{enrichment["reputation"]}</td>'
            f'<td class="soc-cell-mono">{enrichment["first_seen"]}</td>'
            f'<td class="soc-cell-truncate soc-cell-mono">{enrichment["sources"]}</td>'
            "</tr>"
        )
    note = ""
    if len(iocs) > 30:
        note = f' · showing first 30 of {len(iocs)}'
    return (
        '<div class="soc-panel">'
        '<div class="soc-panel__title">Indicators &amp; enrichment '
        f'<span class="soc-meta">{len(iocs)} observable{"s" if len(iocs) != 1 else ""} · demo enrichment{note}</span></div>'
        '<table class="soc-table">'
        "<thead><tr>"
        "<th>Indicator</th><th>Type</th><th>Verdict</th>"
        "<th>Score</th><th>First seen</th><th>Sources</th>"
        "</tr></thead>"
        f'<tbody>{"".join(rows)}</tbody>'
        "</table>"
        '</div>'
    )


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
        for tech in MITRE_MAPPING.get(label, []):
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
    st.session_state["use_llm"] = st.sidebar.checkbox(
        "LLM second opinion",
        value=bool(st.session_state.get("use_llm", True)),
        help="Adds a provider-routed LLM rationale on top of the classifier.",
    )

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


# =============================================================================
# OVERVIEW PAGE
# =============================================================================

def view_overview() -> None:
    render_page_header(
        title="Mission control",
        subtitle="Triage volume, classification distribution, and recent activity across the AlertSage corpus.",
        breadcrumb="Dashboards / Overview",
    )

    db = _db()
    history = []
    bookmarks = []
    notes = []
    try:
        history = db.get_analysis_history(limit=10000) or []
        bookmarks = db.get_bookmarks() or []
        notes = db.get_all_notes() or []
    except Exception as exc:
        st.error(f"Could not load dashboard data: {exc}")

    now = datetime.now()
    # Time windows
    def _within(hours: int) -> list:
        cutoff = now - timedelta(hours=hours)
        return [
            h for h in history
            if "timestamp" in h
            and _safe_dt(h["timestamp"]) and _safe_dt(h["timestamp"]) > cutoff
        ]
    h_24 = _within(24)
    h_7d = _within(24 * 7)
    h_30d = _within(24 * 30)

    total = len(history)
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
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    high_severity = sev_counts.get("critical", 0) + sev_counts.get("high", 0)

    # ---- KPI strip ----
    cols = st.columns(6, gap="small")
    cols[0].markdown(
        render_kpi(
            "Total analyzed", f"{total:,}",
            sub=f"+{len(h_7d)} last 7d" if h_7d else "no recent activity",
            tone="info",
        ),
        unsafe_allow_html=True,
    )
    cols[1].markdown(
        render_kpi(
            "Critical / high", f"{high_severity:,}",
            sub=f"{(high_severity / total * 100):.0f}% of corpus" if total else "n/a",
            tone="critical" if high_severity else "low",
        ),
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        render_kpi(
            "Last 24h", f"{len(h_24):,}",
            sub=f"vs {len(h_7d) - len(h_24):,} prior 6d" if h_7d else "no events",
            tone="medium" if len(h_24) > 0 else "low",
        ),
        unsafe_allow_html=True,
    )
    avg_tone = (
        "low" if avg_conf >= 0.8
        else "medium" if avg_conf >= 0.6
        else "high" if total else "info"
    )
    cols[3].markdown(
        render_kpi(
            "Avg confidence",
            f"{avg_conf:.0%}" if total else "n/a",
            sub="classifier output", tone=avg_tone,
        ),
        unsafe_allow_html=True,
    )
    cols[4].markdown(
        render_kpi(
            "Bookmarks", f"{len(bookmarks):,}",
            sub="saved investigations", tone="info",
        ),
        unsafe_allow_html=True,
    )
    cols[5].markdown(
        render_kpi(
            "Analyst notes", f"{len(notes):,}",
            sub="across history", tone="info",
        ),
        unsafe_allow_html=True,
    )

    # ---- Row 1: events-over-time (wide) + threat feed (rail) ----
    left, right = st.columns([5, 3], gap="large")

    with left:
        render_section_head(
            "Events over time",
            f"Last 14 days · {sum(_count_by_day(history, days=14).values())} events",
        )
        fig = _events_over_time_figure(history, days=14)
        if fig is None:
            render_empty(
                "No events yet",
                "Triage your first incident in the Investigate tab. It will "
                "appear here once classified.",
            )
        else:
            st.plotly_chart(fig, use_container_width=True)

        # Confidence histogram + severity donut side by side
        sub_left, sub_right = st.columns(2, gap="medium")
        with sub_left:
            render_section_head("Classifier confidence", "histogram")
            chist = _confidence_histogram_figure(history)
            if chist is None:
                render_empty("No confidence data", "Triage events to populate.")
            else:
                st.plotly_chart(chist, use_container_width=True)
        with sub_right:
            render_section_head("Severity distribution", "donut")
            if total == 0:
                render_empty("No data", "Severity tiers populate as you triage.")
            else:
                st.plotly_chart(_severity_donut(sev_counts), use_container_width=True)

    with right:
        st.markdown(render_threat_feed(), unsafe_allow_html=True)
        # Live tail auto-refreshes every 5 seconds independent of the rest
        # of the page (st.fragment).
        render_live_tail_fragment(n=6)

    # ---- Row 2: MITRE heatmap (wide) + top classifications ----
    left2, right2 = st.columns([5, 3], gap="large")

    with left2:
        render_section_head("MITRE ATT&CK coverage", "tactic x technique density")
        heatmap_fig = _mitre_heatmap_figure(history)
        if heatmap_fig is None:
            render_empty(
                "No technique coverage",
                "Techniques map automatically once incidents are classified.",
            )
        else:
            st.plotly_chart(heatmap_fig, use_container_width=True)

    with right2:
        render_section_head("Top classifications", "by count")
        if not label_counts:
            render_empty("No classifications", "Run a triage to track labels.")
        else:
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

    # ---- Recent events table ----
    render_section_head("Recent events", action="latest 10")
    recent = sorted(history, key=lambda x: x.get("timestamp", ""), reverse=True)[:10]
    if not recent:
        render_empty(
            "Quiet on the wire",
            "No triage runs yet. Click <strong>Investigate</strong> in the sidebar to analyze your first incident.",
        )
    else:
        st.markdown(_recent_events_table(recent), unsafe_allow_html=True)


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


def _events_over_time_figure(history: list[dict], days: int = 14):
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
    fig.update_layout(
        barmode="stack",
        height=240,
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            font=dict(size=10, color="#94a3b8"), bgcolor="rgba(0,0,0,0)",
        ),
        **PLOT_LAYOUT,
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
        if st.session_state.get("use_llm", True):
            with st.spinner("Querying LLM second opinion..."):
                t0 = time.time()
                opinion, err = run_llm_second_opinion(
                    text,
                    skip_preprocessing=not st.session_state.get("use_preprocessing", True),
                )
                opinion_ms = int((time.time() - t0) * 1000)
            if err:
                st.warning(err)
        result["llm_opinion"] = opinion
        result["llm_ms"] = opinion_ms
        result["analysis_id"] = _persist_analysis(result, batch_id=None)
        st.session_state["current_analysis"] = result

    if st.session_state.get("current_analysis"):
        render_analysis_result(st.session_state["current_analysis"])


def _persist_analysis(result: dict, batch_id: str | None) -> int | None:
    """Save an analysis to the DB. Best-effort; returns id or None."""
    try:
        db = _db()
        return db.save_analysis(
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

    # IOC enrichment panel (full width, below kill chain).
    st.markdown(render_ioc_panel(result["incident_text"]), unsafe_allow_html=True)

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
        _db().add_bookmark(int(aid), note="")
        st.success("Bookmarked.")
    except Exception as exc:
        st.error(f"Could not bookmark: {exc}")


# =============================================================================
# HUNT PAGE
# =============================================================================

def view_hunt() -> None:
    render_page_header(
        title="Hunt",
        subtitle="Search past triage results by classification, narrative, or confidence.",
        breadcrumb="Console / Hunt",
    )

    db = _db()
    history = []
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

    # Filters: row 1 (query, classification, severity)
    row1 = st.columns([3, 2, 2], gap="small")
    with row1[0]:
        query = st.text_input(
            "Search narrative",
            value="",
            placeholder="Free text matched against the narrative...",
        )
    with row1[1]:
        all_labels = sorted({h.get("final_label", "uncertain") for h in history})
        label_filter = st.multiselect(
            "Classification", all_labels, default=[], placeholder="All classifications"
        )
    with row1[2]:
        sev_filter = st.multiselect(
            "Severity",
            ["critical", "high", "medium", "low", "info"],
            default=[],
            placeholder="All severities",
        )

    # Filters: row 2 (confidence, anomaly score, time window)
    row2 = st.columns([2, 2, 2], gap="small")
    with row2[0]:
        min_conf = st.slider("Min confidence", 0.0, 1.0, 0.0, 0.05)
    with row2[1]:
        min_anomaly = st.slider("Min anomaly score", 0, 100, 0, 5)
    with row2[2]:
        time_window = st.selectbox(
            "Time window",
            ["All time", "Last hour", "Last 24 hours", "Last 7 days", "Last 30 days"],
            index=0,
        )

    # Apply filters
    now = datetime.now()
    cutoff: datetime | None = None
    if time_window == "Last hour":
        cutoff = now - timedelta(hours=1)
    elif time_window == "Last 24 hours":
        cutoff = now - timedelta(hours=24)
    elif time_window == "Last 7 days":
        cutoff = now - timedelta(days=7)
    elif time_window == "Last 30 days":
        cutoff = now - timedelta(days=30)

    rows = []
    for h in history:
        if query and query.lower() not in (h.get("incident_text") or "").lower():
            continue
        if label_filter and h.get("final_label") not in label_filter:
            continue
        if sev_filter and severity_for(h.get("final_label", "")) not in sev_filter:
            continue
        try:
            conf = float(h.get("max_prob") or 0)
        except Exception:
            conf = 0
        if conf < min_conf:
            continue
        anomaly = _anomaly_score(h.get("final_label", ""), conf)
        if anomaly < min_anomaly:
            continue
        if cutoff is not None:
            dt = _safe_dt(h.get("timestamp"))
            if not dt or dt < cutoff:
                continue
        rows.append(h)

    render_section_head(
        "Results",
        action=f"{len(rows):,} of {len(history):,} events",
    )

    if not rows:
        render_empty("No matches", "Loosen your filters to see results.")
        return

    rows = sorted(rows, key=lambda x: x.get("timestamp", ""), reverse=True)
    st.markdown(_recent_events_table(rows[:200]), unsafe_allow_html=True)
    if len(rows) > 200:
        st.caption(f"Showing first 200 of {len(rows):,} matches.")


# =============================================================================
# BATCH PAGE
# =============================================================================

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
        return

    try:
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

    if not st.button("Run batch", type="primary", key="batch_run"):
        return

    progress = st.progress(0)
    status = st.empty()
    batch_id = str(uuid.uuid4())[:8]
    threshold = float(st.session_state.get("threshold", 0.5))
    max_classes = int(st.session_state.get("max_classes", 5))
    use_llm = bool(st.session_state.get("use_llm", False))

    results = []
    total = len(df)
    t_start = time.time()
    for i, row in enumerate(df[text_col].fillna("").astype(str).tolist()):
        status.markdown(
            f'<div class="soc-mono" style="color: var(--soc-text-secondary); font-size: 0.85rem;">'
            f'Processing {i+1:,} / {total:,}  ·  batch {batch_id}</div>',
            unsafe_allow_html=True,
        )
        if not row.strip():
            continue
        result = predict(row, threshold=threshold, max_classes=max_classes)
        if use_llm:
            opinion, _ = run_llm_second_opinion(row, skip_preprocessing=False)
            result["llm_opinion"] = opinion
        result["analysis_id"] = _persist_analysis(result, batch_id=batch_id)
        results.append(result)
        progress.progress((i + 1) / total)

    elapsed = time.time() - t_start
    progress.empty()
    status.empty()

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
        label = bm.get("final_label", "uncertain")
        try:
            conf = float(bm.get("max_prob") or 0)
        except Exception:
            conf = 0.0
        body = (bm.get("incident_text") or "").strip()
        ts = _safe_dt(bm.get("timestamp"))
        when = ts.strftime("%b %d, %Y at %H:%M") if ts else "Unknown time"
        note = (bm.get("note") or "").strip()
        bm_id = bm.get("id")
        analysis_id = bm.get("analysis_id") or bm.get("id")
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


def _settings_panel_openai() -> None:
    st.markdown(
        '<div class="soc-panel">'
        '<div class="soc-panel__title">OpenAI</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([3, 2], gap="medium")
    with cols[0]:
        st.session_state["openai_model_id"] = st.text_input(
            "Model id",
            value=st.session_state.get("openai_model_id", DEFAULT_OPENAI_MODEL),
            help="Examples: gpt-4o-mini, gpt-4o, gpt-4.1-mini",
            key="settings_oa_model",
        )
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
    with cols[0]:
        st.session_state["anthropic_model_id"] = st.text_input(
            "Model id",
            value=st.session_state.get("anthropic_model_id", DEFAULT_ANTHROPIC_MODEL),
            help="Examples: claude-haiku-4-5, claude-sonnet-4-6",
            key="settings_an_model",
        )
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
    render_sidebar()
    view = st.session_state.get("view", "overview")
    render_topbar(view)

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
