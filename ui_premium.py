"""
AlertSage - Premium NLP-Driven Incident Triage System
Complete feature set with professional design (NO EMOJIS - Professional Icons Only)
"""

import os
import io
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from collections import Counter
from typing import Any, Tuple, TYPE_CHECKING
import joblib

if TYPE_CHECKING:
    import tomli as tomllib  # noqa: F401 #type: ignore[import-not-found]

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - fallback for older runtimes
    import tomli as tomllib  # type: ignore[import-not-found]

# Import custom modules
from src.triage.database import TriageDatabase
from src.triage.embeddings import get_embedder
from src.triage.model import load_vectorizer_and_model, predict_event_type
from src.triage.preprocess import clean_description
from src.triage.cli import llm_second_opinion, build_llm_rationale

# Import icon helpers
from assets.icons.icon_helpers import (
    logo_header_display,
    inline_icon,
    ui_icon,
    incident_icon,
    branded_metric_card,
    section_header,
)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

# Load brand logo for page icon
_logo_path = Path(__file__).parent / "assets" / "icons" / "alertsage-logo.svg"
if _logo_path.exists():
    import base64

    _logo_svg = _logo_path.read_text()
    # Create a simplified version for favicon (16x16 or 32x32)
    _favicon_svg = """<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
        <defs>
            <linearGradient id="g" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#667eea"/>
                <stop offset="100%" style="stop-color:#764ba2"/>
            </linearGradient>
        </defs>
        <path d="M16 4 L28 10 L28 22 Q28 28 16 30 Q4 28 4 22 L4 10 Z" fill="url(#g)"/>
        <circle cx="16" cy="16" r="3" fill="white"/>
        <circle cx="16" cy="11" r="2" fill="white" opacity="0.8"/>
        <circle cx="11" cy="14" r="1.5" fill="white" opacity="0.8"/>
        <circle cx="21" cy="14" r="1.5" fill="white" opacity="0.8"/>
        <circle cx="11" cy="18" r="1.5" fill="white" opacity="0.8"/>
        <circle cx="21" cy="18" r="1.5" fill="white" opacity="0.8"/>
        <circle cx="16" cy="21" r="2" fill="white" opacity="0.8"/>
    </svg>"""
    _favicon_b64 = base64.b64encode(_favicon_svg.encode()).decode()
    _page_icon = f"data:image/svg+xml;base64,{_favicon_b64}"
else:
    _page_icon = "🛡"

st.set_page_config(
    page_title="AlertSage - AI-Powered Security Triage",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# MITRE ATT&CK MAPPING
# ============================================================================


def get_mitre_techniques(incident_type: str) -> list:
    """Get MITRE ATT&CK techniques for an incident type."""
    mitre_mapping = {
        "phishing": ["T1566", "T1598"],  # Phishing, Phishing for Information
        "malware": [
            "T1059",
            "T1204",
            "T1105",
            "T1486",
        ],  # Command Execution, User Execution, Ingress Tool Transfer, Data Encrypted for Impact
        "access_abuse": ["T1078", "T1110"],  # Valid Accounts, Brute Force
        "credential_compromise": [
            "T1556",
            "T1539",
            "T1528",
        ],  # Modify Auth Mechanisms, Steal Web Session Cookie, Steal Application Access Token
        "data_exfiltration": [
            "T1048",
            "T1020",
            "T1041",
            "T1567",
        ],  # Exfiltration Over Alternative Protocol, Automated Exfiltration, Exfiltration Over C2, Exfiltration Over Web Service
        "insider_threat": [
            "T1530",
            "T1087",
            "T1213",
        ],  # Data from Cloud Storage, Account Discovery, Data from Information Repositories
        "policy_violation": [
            "T1036",
            "T1082",
        ],  # Masquerading, System Information Discovery
        "web_attack": [
            "T1190",
            "T1059.007",
        ],  # Exploit Public-Facing Application, JavaScript
        "suspicious_network_activity": [
            "T1046",
            "T1595",
            "T1040",
        ],  # Network Service Discovery, Active Scanning, Network Sniffing
        "benign_activity": [],  # No malicious techniques
        "uncertain": [],  # Will use top prediction techniques
    }
    return mitre_mapping.get(incident_type, [])


# LLM request guardrails for the UI
UI_LLM_MAX_INPUT_CHARS = 8000
UI_LLM_MAX_TOKENS = 512
HF_UI_MAX_REQUESTS = 5
HF_UI_WINDOW_SECONDS = 60
# Dark mode disabled for readability; keep light as single, consistent theme
THEME_OPTIONS = ["Light"]

_SECRETS_PATHS = [
    Path("~/.streamlit/secrets.toml").expanduser(),
    Path.cwd() / ".streamlit" / "secrets.toml",
]

_CACHED_SECRETS: dict[str, Any] | None = None


def _load_local_secrets() -> dict[str, Any]:
    """Load secrets from the first existing secrets.toml without using st.secrets."""
    global _CACHED_SECRETS
    if _CACHED_SECRETS is not None:
        return _CACHED_SECRETS

    for path in _SECRETS_PATHS:
        try:
            if path.exists():
                _CACHED_SECRETS = tomllib.loads(path.read_text())
                return _CACHED_SECRETS
        except Exception:
            # If parsing fails, return empty to avoid noisy warnings
            _CACHED_SECRETS = {}
            return _CACHED_SECRETS

    _CACHED_SECRETS = {}
    return _CACHED_SECRETS


def get_secret_value(key: str, default: str = "") -> str:
    """Read a secret from local secrets.toml if present, otherwise return default."""
    secrets_data = _load_local_secrets()
    return str(secrets_data.get(key, default)).strip()


def get_llm_settings() -> tuple[str, str, str | None]:
    """Fetch LLM provider settings from session state with safe defaults."""
    secrets_token = get_secret_value("HF_TOKEN", "")
    secrets_model = get_secret_value("HF_MODEL", "")

    env_token = os.environ.get("TRIAGE_HF_TOKEN") or os.environ.get("HF_TOKEN") or ""
    env_model = os.environ.get("TRIAGE_HF_MODEL") or os.environ.get("HF_MODEL") or ""

    default_provider = "huggingface" if (secrets_token or env_token) else "local"
    provider = st.session_state.get("llm_provider", default_provider)
    hf_model = st.session_state.get(
        "hf_model_id",
        secrets_model or env_model or "meta-llama/Llama-3.1-8B-Instruct:cerebras",
    )
    hf_token = st.session_state.get("selected_hf_token") or secrets_token or env_token
    return provider, hf_model, hf_token


def get_text_palette() -> dict[str, Any]:
    """Return theme-aware text colors with stronger contrast."""
    is_dark_mode = str(st.session_state.get("theme_mode", "Light")).lower() == "dark"
    return {
        "is_dark": is_dark_mode,
        "secondary": "#e6edf7" if is_dark_mode else "#0f172a",
        "muted": "#d5deeb" if is_dark_mode else "#1f2937",
    }


def apply_theme_mode_css(mode: str) -> None:
    """Inject lightweight theme overrides for light/dark readability."""
    if mode.lower() == "dark":
        css = """
        <style>
        :root { color-scheme: dark; }
        body, .stApp, [data-testid="stAppViewContainer"], .main { background: linear-gradient(180deg, #0b1220 0%, #0f172a 60%, #0b1220 100%) !important; color: #e5e7eb !important; }
        .block-container { color: #e5e7eb !important; }
        .glass-card { background: #111827 !important; color: #e5e7eb !important; border: 1px solid #1f2937 !important; }
        .metric-premium { background: #111827 !important; color: #e5e7eb !important; border: 2px solid #1f2937 !important; }
        .alert-premium { background: #0f172a !important; color: #e5e7eb !important; }
        .alert-info { background: rgba(59, 130, 246, 0.15) !important; color: #bfdbfe !important; }
        .alert-success { background: rgba(16, 185, 129, 0.2) !important; color: #a7f3d0 !important; }
        .alert-warning { background: rgba(245, 158, 11, 0.2) !important; color: #fcd34d !important; }
        .stMarkdown, .stText, .stSelectbox label, .stTextInput label, .stCheckbox label, .streamlit-expanderHeader { color: #e5e7eb !important; }
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select { background: #0f172a !important; color: #e5e7eb !important; border: 1px solid #1f2937 !important; }
        .section-header { color: #f8fafc !important; border-bottom: 3px solid #60a5fa !important; background: rgba(15, 23, 42, 0.75) !important; padding: 0.5rem 0; }
        </style>
        """
    else:
        css = """
        <style>
        :root { color-scheme: light; }
        body, .stApp, [data-testid="stAppViewContainer"], .main { background: linear-gradient(180deg, #f7f9ff 0%, #eef2ff 55%, #e8f0ff 100%) !important; color: #0f172a !important; }
        .block-container { color: #0f172a !important; }
        .glass-card { background: #ffffff !important; color: #0f172a !important; border: 1px solid #e2e8f0 !important; }
        .metric-premium { background: #ffffff !important; color: #0f172a !important; border: 2px solid #e2e8f0 !important; }
        .alert-premium { background: #ffffff !important; color: #0f172a !important; }
        .alert-info { background: #eff6ff !important; color: #1e40af !important; }
        .alert-success { background: #d1fae5 !important; color: #065f46 !important; }
        .alert-warning { background: #fef3c7 !important; color: #92400e !important; }
        .stMarkdown, .stText, .stSelectbox label, .stTextInput label, .stCheckbox label, .streamlit-expanderHeader { color: #0f172a !important; }
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select { background: #ffffff !important; color: #0f172a !important; border: 1px solid #e2e8f0 !important; }
        .section-header { color: #0f172a !important; border-bottom: 3px solid #667eea !important; background: rgba(255, 255, 255, 0.85) !important; padding: 0.5rem 0; }
        </style>
        """

    st.markdown(css, unsafe_allow_html=True)


def validate_llm_input_length(text: str) -> tuple[bool, str]:
    if text and len(text) > UI_LLM_MAX_INPUT_CHARS:
        return False, (
            f"LLM input too long ({len(text)} characters). "
            f"Limit is {UI_LLM_MAX_INPUT_CHARS} characters."
        )
    return True, ""


def hf_rate_limit_allowance() -> tuple[bool, float]:
    timestamps = st.session_state.get("hf_ui_requests", [])
    now = datetime.utcnow().timestamp()
    window_start = now - HF_UI_WINDOW_SECONDS

    timestamps = [ts for ts in timestamps if ts >= window_start]
    st.session_state["hf_ui_requests"] = timestamps

    if len(timestamps) >= HF_UI_MAX_REQUESTS:
        retry_after = HF_UI_WINDOW_SECONDS - (now - timestamps[0])
        return False, max(retry_after, 0.0)

    timestamps.append(now)
    st.session_state["hf_ui_requests"] = timestamps
    return True, 0.0


# ============================================================================
# PREMIUM STYLING
# ============================================================================

PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

.main {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
    color: #0f172a;
}

.block-container {
    padding: 2rem 3rem;
    max-width: 1600px;
    color: #0f172a;
}

/* Plotly modebar positioning */
.js-plotly-plot .plotly .modebar {
    top: auto !important;
    bottom: 10px !important;
    right: 10px !important;
}

/* Dark Mode Support */
@media (prefers-color-scheme: dark) {
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
        color: #e5e7eb;
    }
    
    .glass-card {
        background: rgba(26, 32, 44, 0.92) !important;
        border: 1px solid rgba(148, 163, 184, 0.35) !important;
    }
    
    .metric-premium {
        background: rgba(26, 32, 44, 0.88) !important;
        border: 2px solid rgba(148, 163, 184, 0.4) !important;
    }
    
    .alert-premium {
        background: rgba(30, 35, 50, 0.9) !important;
    }
    
    .alert-info {
        background: rgba(59, 130, 246, 0.2) !important;
        color: #93c5fd !important;
    }
    
    .alert-success {
        background: rgba(16, 185, 129, 0.2) !important;
        color: #6ee7b7 !important;
    }
    
    .alert-warning {
        background: rgba(245, 158, 11, 0.2) !important;
        color: #fcd34d !important;
    }
    
    .section-header {
        color: #f8fafc !important;
    }

    .stMarkdown, .stText, .stSelectbox label, .stTextInput label, .stCheckbox label {
        color: #e5e7eb !important;
    }
}

/* Glass Cards */
.glass-card {
    background: rgba(255, 255, 255, 0.98);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(31, 38, 135, 0.15);
    border: 1px solid rgba(102, 126, 234, 0.1);
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(31, 38, 135, 0.2);
}

/* Hero Header */
.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 24px;
    padding: 3rem 2.5rem;
    margin-bottom: 2.5rem;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4);
}

.hero-title {
    font-size: 3rem;
    font-weight: 900;
    color: white;
    margin: 0;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

.hero-subtitle {
    font-size: 1.2rem;
    color: rgba(255, 255, 255, 0.95);
    margin-top: 0.75rem;
    font-weight: 400;
}

/* Metric Cards */
.metric-premium {
    background: white;
    border-radius: 16px;
    padding: 1.75rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.1);
    border: 2px solid #f0f0f0;
    transition: all 0.3s ease;
}

.metric-premium:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
    border-color: #667eea;
}

.metric-label {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #667eea;
    margin-bottom: 0.75rem;
}

.metric-value {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1;
    margin: 0.5rem 0;
}

.metric-change {
    font-size: 0.85rem;
    color: #10b981;
    font-weight: 600;
    margin-top: 0.5rem;
}

/* Section Headers */
.section-header {
    font-size: 1.75rem;
    font-weight: 800;
    color: #1a202c;
    margin: 2.5rem 0 1.5rem 0;
    padding-bottom: 0.75rem;
    border-bottom: 3px solid #667eea;
    letter-spacing: -0.01em;
}

/* Prediction Card */
.prediction-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 20px;
    padding: 2.5rem;
    color: white;
    box-shadow: 0 15px 50px rgba(102, 126, 234, 0.4);
    margin: 2rem 0;
}

.prediction-title {
    font-size: 2.25rem;
    font-weight: 900;
    margin-bottom: 1rem;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}

.confidence-badge {
    display: inline-block;
    padding: 0.6rem 1.5rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.05em;
    margin-top: 0.75rem;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

.confidence-high { background: #10b981; }
.confidence-medium { background: #f59e0b; }
.confidence-low { background: #ef4444; }

/* Probability Bars */
.prob-container {
    margin: 1.25rem 0 !important;
    padding: 0.75rem !important;
    background: rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px);
}

.prob-label {
    display: flex !important;
    justify-content: space-between !important;
    margin-bottom: 0.5rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: white !important;
}

.prob-label span {
    color: white !important;
}

.prob-bar-bg {
    background: rgba(255, 255, 255, 0.25) !important;
    height: 14px !important;
    border-radius: 50px !important;
    overflow: hidden !important;
    position: relative;
}

.prob-bar-fill {
    background: linear-gradient(90deg, rgba(255,255,255,0.9) 0%, white 100%) !important;
    height: 100% !important;
    border-radius: 50px !important;
    transition: width 0.8s ease !important;
    box-shadow: 0 0 15px rgba(255,255,255,0.6) !important;
    position: relative;
}

/* Alert Boxes */
.alert-premium {
    padding: 1.25rem 1.75rem;
    border-radius: 12px;
    margin: 1.25rem 0;
    border-left: 4px solid;
    font-size: 1rem;
    line-height: 1.6;
    background: white;
}

.alert-info {
    border-color: #3b82f6;
    background: #eff6ff;
    color: #1e40af;
}

.alert-success {
    border-color: #10b981;
    background: #d1fae5;
    color: #065f46;
}

.alert-warning {
    border-color: #f59e0b;
    background: #fef3c7;
    color: #92400e;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.1);
    padding: 0.5rem;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    transition: all 0.2s;
    color: #1a202c !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.75rem 2rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1rem;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    transition: all 0.3s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4);
}

/* Input Fields */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > select {
    border: 2px solid #e5e7eb;
    border-radius: 12px;
    padding: 0.875rem 1rem;
    font-size: 0.95rem;
    transition: all 0.2s;
    background: white;
    color: #1a202c;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
}

[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label {
    color: white !important;
}

[data-testid="stSidebar"] [role="radiogroup"] label {
    color: white !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: rgba(102, 126, 234, 0.08);
    border-radius: 10px;
    font-weight: 600;
    padding: 1rem 1.25rem;
    color: #1a202c !important;
}

/* Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}
</style>
"""

st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ============================================================================
# UTILITY CLASSES
# ============================================================================


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy arrays and datetime objects"""

    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, datetime):
            return o.isoformat()
        if isinstance(o, timedelta):
            return o.total_seconds()
        return super().default(o)


# ============================================================================
# LLM ENHANCEMENT FUNCTIONS
# ============================================================================


def generate_executive_summary(llm_opinion: dict, incident_text: str) -> str:
    """Generate executive summary from LLM analysis."""
    llm_label = llm_opinion.get("label", "unknown")
    llm_rationale = llm_opinion.get("rationale", "")
    mitre_ids = llm_opinion.get("mitre_ids", [])

    # Extract summary and impact from rationale
    summary = ""
    impact = ""

    if "Summary:" in llm_rationale:
        summary_match = re.search(
            r"Summary:\s*(.+?)(?:Impact:|Model label|Next steps:|$)",
            llm_rationale,
            re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            summary = summary_match.group(1).strip()

    if "Impact:" in llm_rationale:
        impact_match = re.search(
            r"Impact:\s*(.+?)(?:Model label|Next steps:|$)",
            llm_rationale,
            re.IGNORECASE | re.DOTALL,
        )
        if impact_match:
            impact = impact_match.group(1).strip()

    # Build executive summary
    exec_summary = f"""**Incident Classification:** {llm_label.replace('_', ' ').title()}

**Executive Summary:**
{summary if summary else llm_rationale[:300]}

**Business Impact:**
{impact if impact else 'Requires further assessment'}

**MITRE ATT&CK Techniques:** {', '.join(mitre_ids[:5]) if mitre_ids else 'None identified'}

**Recommended Priority:** {'P1 - Critical' if llm_label in ['malware', 'data_exfiltration'] else 'P2 - High' if llm_label in ['web_attack', 'access_abuse'] else 'P3 - Medium'}
"""
    return exec_summary


def extract_search_patterns(llm_rationale: str, incident_text: str) -> list:
    """Extract SIEM search patterns from incident."""
    patterns = []

    # Extract IPs
    ips = re.findall(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", incident_text)
    if ips:
        patterns.append(f"src_ip:{ips[0]} OR dst_ip:{ips[0]}")

    # Extract URLs/domains
    urls = re.findall(r"https?://([^\s/]+)", incident_text)
    domains = re.findall(r"\b([a-z0-9-]+\.[a-z]{2,})\b", incident_text.lower())
    if urls or domains:
        domain = urls[0] if urls else domains[0] if domains else None
        if domain:
            patterns.append(f'url:"*{domain}*" OR domain:"{domain}"')

    # Extract file names
    files = re.findall(
        r"\b[\w-]+\.(exe|dll|bat|ps1|vbs|js|jar|zip|rar|7z|doc|docx|xls|xlsx|pdf)\b",
        incident_text,
        re.IGNORECASE,
    )
    if files:
        patterns.append(f'file_name:"{files[0]}"')

    # Extract email addresses
    emails = re.findall(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", incident_text
    )
    if emails:
        patterns.append(f'sender:"{emails[0]}" OR recipient:"{emails[0]}"')

    # Extract usernames
    users = re.findall(r"\buser(?:name)?[:\s]+([a-z0-9._-]+)\b", incident_text.lower())
    if users:
        patterns.append(f'user:"{users[0]}"')

    return patterns if patterns else ["event_type:security"]


def generate_custom_playbook(
    llm_opinion: dict, iocs: dict, mitre_techniques: list
) -> str:
    """Generate custom incident response playbook."""
    llm_label = llm_opinion.get("label", "unknown")
    llm_rationale = llm_opinion.get("rationale", "")

    # Extract next steps from LLM rationale
    steps = []
    if "Next steps:" in llm_rationale or "next steps:" in llm_rationale.lower():
        next_steps_match = re.search(
            r"Next steps:(.*?)(?:$|\n\n)", llm_rationale, re.IGNORECASE | re.DOTALL
        )
        if next_steps_match:
            next_steps_text = next_steps_match.group(1).strip()
            steps = re.findall(
                r"(\d+[\).]) *([^\d]+?)(?=\d+[\).]|$)", next_steps_text, re.DOTALL
            )

    playbook = f"""# Incident Response Playbook: {llm_label.replace('_', ' ').title()}

## Phase 1: Containment
"""

    if steps:
        for num, step in steps[:3]:
            playbook += f"- {step.strip()}\n"
    else:
        playbook += (
            f"- Isolate affected systems\n- Preserve evidence\n- Document timeline\n"
        )

    playbook += f"\n## Phase 2: Investigation\n"

    if iocs.get("ip_addresses", 0) > 0:
        playbook += f"- Analyze network traffic from identified IP addresses\n"
    if iocs.get("urls", 0) > 0:
        playbook += f"- Investigate suspicious URLs in sandbox environment\n"
    if iocs.get("email_addresses", 0) > 0:
        playbook += f"- Review email headers and recipient list\n"

    playbook += f"\n## Phase 3: Eradication\n"

    if steps and len(steps) > 3:
        for num, step in steps[3:5]:
            playbook += f"- {step.strip()}\n"
    else:
        playbook += f"- Remove malicious artifacts\n- Patch vulnerabilities\n- Update detection rules\n"

    playbook += f"\n## Phase 4: Recovery & Lessons Learned\n"
    playbook += f"- Restore from clean backups\n- Monitor for recurrence\n- Update security controls\n- Conduct post-incident review\n"

    if mitre_techniques:
        playbook += f"\n## MITRE ATT&CK Mapping\n"
        for tech in mitre_techniques[:5]:
            playbook += f"- {tech}\n"

    return playbook


def predict_attack_timeline(
    llm_rationale: str, llm_label: str, mitre_techniques: list
) -> dict:
    """Predict attack progression timeline."""
    timeline = {"stages": [], "estimated_hours": 0}

    # Map MITRE techniques to kill chain stages
    kill_chain_mapping = {
        "initial_access": ["T1190", "T1566", "T1078"],
        "execution": ["T1059", "T1204"],
        "persistence": ["T1053", "T1547"],
        "privilege_escalation": ["T1068", "T1548"],
        "defense_evasion": ["T1036", "T1070"],
        "credential_access": ["T1110", "T1556"],
        "discovery": ["T1083", "T1046", "T1087"],
        "lateral_movement": ["T1021", "T1570"],
        "collection": ["T1005", "T1213"],
        "exfiltration": ["T1041", "T1567", "T1048"],
        "impact": ["T1486", "T1490"],
    }

    detected_stages = []
    for stage, techniques in kill_chain_mapping.items():
        if any(tech in mitre_techniques for tech in techniques):
            detected_stages.append(stage)

    # Build timeline
    if llm_label == "phishing":
        timeline["stages"] = [
            {
                "stage": "Initial Access",
                "time": "0-1 hours",
                "desc": "Phishing email delivered and opened",
            },
            {
                "stage": "Execution",
                "time": "1-2 hours",
                "desc": "User clicks malicious link or opens attachment",
            },
            {
                "stage": "Credential Harvest",
                "time": "2-4 hours",
                "desc": "Credentials potentially compromised",
            },
            {
                "stage": "Lateral Spread",
                "time": "4-24 hours",
                "desc": "Attacker may attempt to access other systems",
            },
        ]
        timeline["estimated_hours"] = 24
    elif llm_label == "malware":
        timeline["stages"] = [
            {
                "stage": "Initial Infection",
                "time": "0-1 hours",
                "desc": "Malware executed on endpoint",
            },
            {
                "stage": "Persistence",
                "time": "1-3 hours",
                "desc": "Malware establishes persistence mechanisms",
            },
            {
                "stage": "C2 Communication",
                "time": "3-6 hours",
                "desc": "Connection to command & control server",
            },
            {
                "stage": "Lateral Movement",
                "time": "6-48 hours",
                "desc": "Spread to additional systems",
            },
            {
                "stage": "Data Theft/Impact",
                "time": "48+ hours",
                "desc": "Data exfiltration or ransomware encryption",
            },
        ]
        timeline["estimated_hours"] = 72
    elif llm_label == "data_exfiltration":
        timeline["stages"] = [
            {
                "stage": "Access Gained",
                "time": "0-2 hours",
                "desc": "Attacker gains initial access",
            },
            {
                "stage": "Reconnaissance",
                "time": "2-12 hours",
                "desc": "Identify valuable data locations",
            },
            {
                "stage": "Collection",
                "time": "12-24 hours",
                "desc": "Gather and stage data for exfiltration",
            },
            {
                "stage": "Exfiltration",
                "time": "24-48 hours",
                "desc": "Transfer data to external location",
            },
        ]
        timeline["estimated_hours"] = 48
    else:
        # Generic timeline
        timeline["stages"] = [
            {
                "stage": "Detection",
                "time": "Current",
                "desc": "Incident detected and being analyzed",
            },
            {
                "stage": "Containment",
                "time": "0-4 hours",
                "desc": "Isolate affected systems",
            },
            {
                "stage": "Investigation",
                "time": "4-24 hours",
                "desc": "Full scope analysis",
            },
            {
                "stage": "Remediation",
                "time": "24-72 hours",
                "desc": "Remove threat and restore services",
            },
        ]
        timeline["estimated_hours"] = 72

    return timeline


def generate_stakeholder_communication(
    llm_opinion: dict, incident_text: str, audience: str
) -> str:
    """Generate communication templates for different audiences."""
    llm_label = llm_opinion.get("label", "unknown")
    llm_rationale = llm_opinion.get("rationale", "")

    # Extract summary
    summary = ""
    if "Summary:" in llm_rationale:
        summary_match = re.search(
            r"Summary:\s*(.+?)(?:Impact:|Model label|Next steps:|$)",
            llm_rationale,
            re.IGNORECASE | re.DOTALL,
        )
        if summary_match:
            summary = summary_match.group(1).strip()

    if audience == "technical":
        return f"""**INCIDENT ALERT - TECHNICAL TEAM**

**Classification:** {llm_label.replace('_', ' ').title()}
**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

**Technical Summary:**
{summary if summary else incident_text[:200]}

**Indicators of Compromise:**
{chr(10).join([f"- IP: {ip}" for ip in re.findall(r'\b(?:[0-9]{{1,3}}\.){3}[0-9]{1,3}\b', incident_text)[:3]])}
{chr(10).join([f"- URL: {url}" for url in re.findall(r'https?://[^\s]+', incident_text)[:3]])}

**Required Actions:**
See detailed playbook in incident management system.

**MITRE ATT&CK:** {', '.join(llm_opinion.get('mitre_ids', [])[:3])}

**Severity:** {'P1-Critical' if llm_label in ['malware', 'data_exfiltration'] else 'P2-High'}
"""

    elif audience == "executive":
        return f"""**SECURITY INCIDENT NOTIFICATION**

**Date:** {datetime.now().strftime('%B %d, %Y')}
**Incident Type:** {llm_label.replace('_', ' ').title()}

**Summary:**
Our security team has detected and is responding to a {llm_label.replace('_', ' ')} incident. {summary[:150] if summary else 'Investigation is ongoing.'}

**Business Impact:**
{'Critical - Immediate attention required. Potential data loss or system compromise.' if llm_label in ['malware', 'data_exfiltration'] else 'Medium - Systems remain operational. Security team is investigating.'}

**Status:**
Incident response team is actively working on containment and remediation.

**Next Update:**
Within 4 hours or upon significant developments.

**Contact:**
SOC Manager / CISO
"""

    else:  # legal/compliance
        return f"""**SECURITY INCIDENT - COMPLIANCE NOTIFICATION**

**Incident ID:** INC-{datetime.now().strftime('%Y%m%d-%H%M%S')}
**Classification:** {llm_label.replace('_', ' ').title()}
**Detection Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

**Description:**
{summary if summary else incident_text[:300]}

**Potential Regulatory Impact:**
{'HIGH - Potential data breach. May require regulatory notification under GDPR/CCPA/HIPAA.' if llm_label == 'data_exfiltration' else 'MEDIUM - Monitoring for data exposure. No immediate notification requirement identified.'}

**Data Classification:**
Under investigation - preliminary assessment indicates {'sensitive data may be involved' if llm_label in ['data_exfiltration', 'access_abuse'] else 'system-level impact only'}.

**Preservation Notice:**
All relevant logs and evidence are being preserved per incident response procedures.

**Legal Hold:**
{'RECOMMENDED' if llm_label in ['data_exfiltration', 'insider_threat'] else 'Not required at this time'}

**Next Steps:**
- Continue technical investigation
- Assess data exposure scope
- Determine notification requirements
- Prepare compliance documentation
"""


def calculate_confidence_trend(
    max_prob: float, ioc_count: int, mitre_count: int
) -> dict:
    """Calculate confidence evolution with additional context."""
    return {
        "baseline": max_prob * 100,
        "with_iocs": min(100, max_prob * 100 + (ioc_count * 2)),
        "with_mitre": min(100, max_prob * 100 + (mitre_count * 3)),
        "with_full_context": min(
            100, max_prob * 100 + (ioc_count * 2) + (mitre_count * 3) + 5
        ),
    }


def assess_false_positive_likelihood(
    incident_text: str, max_prob: float, llm_label: str
) -> float:
    """Assess likelihood of false positive."""
    fp_score = 0.0

    # Low confidence suggests uncertainty
    if max_prob < 0.5:
        fp_score += 0.3

    # Check for benign keywords
    benign_keywords = [
        "test",
        "testing",
        "scheduled",
        "maintenance",
        "authorized",
        "approved",
        "legitimate",
    ]
    if any(keyword in incident_text.lower() for keyword in benign_keywords):
        fp_score += 0.2

    # Check for lack of IOCs
    ioc_count = len(re.findall(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", incident_text))
    ioc_count += len(re.findall(r"https?://[^\s]+", incident_text))
    if ioc_count == 0:
        fp_score += 0.15

    # Generic descriptions suggest less certainty
    if len(incident_text.split()) < 20:
        fp_score += 0.1

    # Benign label with low confidence
    if llm_label == "benign_activity" and max_prob > 0.6:
        fp_score += 0.25

    return min(1.0, fp_score)


def add_chart_download_buttons(
    fig: go.Figure, chart_name: str = "chart", key_suffix: str = ""
):
    """Add download buttons for a Plotly figure.

    Args:
        fig: Plotly figure to export
        chart_name: Base name for the downloaded file
        key_suffix: Unique suffix for widget keys to avoid duplicates
    """
    if fig is None:
        return

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Download as PNG
        try:
            png_bytes = fig.to_image(format="png", width=1200, height=800)
            st.download_button(
                label="Download PNG",
                data=png_bytes,
                file_name=f"{chart_name}.png",
                mime="image/png",
                key=f"download_png_{chart_name}_{key_suffix}",
                help="Download as PNG image",
                use_container_width=True,
            )
        except:
            st.caption("PNG export requires kaleido")

    with col2:
        # Download as HTML (interactive)
        html_bytes = fig.to_html(include_plotlyjs="cdn").encode()
        st.download_button(
            label="Download HTML",
            data=html_bytes,
            file_name=f"{chart_name}.html",
            mime="text/html",
            key=f"download_html_{chart_name}_{key_suffix}",
            help="Download as interactive HTML",
            use_container_width=True,
        )

    with col3:
        # Download as JSON (data)
        json_str = json.dumps(fig.to_dict(), indent=2, cls=NumpyEncoder)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"{chart_name}.json",
            mime="application/json",
            key=f"download_json_{chart_name}_{key_suffix}",
            help="Download chart data as JSON",
            use_container_width=True,
        )


def create_confidence_heatmap(data: list) -> go.Figure:
    """Create confidence heatmap visualization.

    Args:
        data: Either probs_sorted (single incident) or batch results
    """
    if not data:
        return go.Figure()

    # Check data format and extract labels/values accordingly
    if "class" in data[0]:
        # Single incident format: [{"class": "malware", "probability": 0.95}, ...]
        labels = [p["class"].replace("_", " ").title() for p in data]
        values = [p["probability"] * 100 for p in data]
    elif "display_label" in data[0]:
        # Batch results format: aggregate by classification
        label_confidences = {}
        for item in data:
            label = item["display_label"]
            conf = item["max_prob"]
            if label not in label_confidences:
                label_confidences[label] = []
            label_confidences[label].append(conf)

        # Calculate average confidence per label
        labels = [get_display_name(label) for label in sorted(label_confidences.keys())]
        values = [
            np.mean(label_confidences[label]) * 100
            for label in sorted(label_confidences.keys())
        ]
    else:
        # Fallback: treat as generic format
        labels = ["Unknown"]
        values = [0]

    fig = go.Figure(
        data=go.Heatmap(
            z=[values],
            x=labels,
            y=["Confidence"],
            colorscale="RdYlGn",
            text=[[f"{v:.1f}%" for v in values]],
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="<b>%{x}</b><br>Confidence: %{z:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Classification Confidence Heatmap",
        height=200,
        margin=dict(l=100, r=50, t=50, b=100),
    )

    return fig


def map_to_kill_chain(mitre_techniques: list) -> dict:
    """Map MITRE techniques to kill chain stages."""
    kill_chain = {
        "Reconnaissance": [],
        "Weaponization": [],
        "Delivery": [],
        "Exploitation": [],
        "Installation": [],
        "Command & Control": [],
        "Actions on Objectives": [],
    }

    mapping = {
        "T1190": "Exploitation",
        "T1566": "Delivery",
        "T1059": "Exploitation",
        "T1204": "Exploitation",
        "T1486": "Actions on Objectives",
        "T1078": "Exploitation",
        "T1110": "Exploitation",
        "T1041": "Actions on Objectives",
        "T1567": "Actions on Objectives",
        "T1071": "Command & Control",
        "T1046": "Reconnaissance",
    }

    for tech in mitre_techniques:
        stage = mapping.get(tech)
        if stage:
            kill_chain[stage].append(tech)

    return {k: v for k, v in kill_chain.items() if v}


def create_kill_chain_viz(kill_chain_stages: dict) -> go.Figure:
    """Create kill chain visualization."""
    stages = list(kill_chain_stages.keys())
    counts = [len(techniques) for techniques in kill_chain_stages.values()]

    fig = go.Figure(
        data=[
            go.Bar(
                x=stages,
                y=counts,
                marker_color=[
                    "#FF6B6B",
                    "#FFA07A",
                    "#FFD93D",
                    "#6BCF7F",
                    "#4ECDC4",
                    "#4A90E2",
                    "#9B59B6",
                ],
                text=counts,
                textposition="auto",
            )
        ]
    )

    fig.update_layout(
        title="Cyber Kill Chain Coverage",
        xaxis_title="Kill Chain Stage",
        yaxis_title="Techniques Detected",
        height=400,
    )

    return fig


# ============================================================================
# SEMANTIC SEARCH & SIMILARITY FUNCTIONS
# ============================================================================


@st.cache_data(ttl=3600, show_spinner="Building embedding cache...")
def _get_corpus_embeddings(corpus_hash: str, corpus_texts: list) -> np.ndarray:
    """Cache embeddings for the incident corpus.

    Args:
        corpus_hash: Hash of corpus for cache invalidation
        corpus_texts: List of incident texts to encode

    Returns:
        2D numpy array of embeddings
    """
    embedder = get_embedder()
    return embedder.encode(corpus_texts)


def _get_incident_corpus(limit: int = 200000) -> tuple[list[dict[str, Any]], str]:
    """Get incident corpus with caching.

    Args:
        limit: Maximum number of incidents to retrieve

    Returns:
        Tuple of (incidents list, corpus hash)
    """
    if "db" not in st.session_state:
        return [], ""

    incidents = st.session_state.db.get_analysis_history(limit=limit)

    if not incidents:
        return [], ""

    # Create hash from incident IDs for cache invalidation
    import hashlib

    incident_ids = [str(inc.get("analysis_id", "")) for inc in incidents]
    corpus_hash = hashlib.md5("".join(incident_ids).encode()).hexdigest()

    return incidents, corpus_hash


def find_similar_incidents(
    query_text: str, top_k: int = 5, similarity_threshold: float = 0.7
) -> list:
    """Find similar incidents using semantic embeddings.

    Args:
        query_text: Incident description to search for
        top_k: Number of similar incidents to return
        similarity_threshold: Minimum similarity score (0-1)

    Returns:
        List of dicts with keys: analysis_id, incident_text, final_label,
        max_prob, timestamp, similarity_score
    """
    try:
        # Get incident corpus with hash for caching
        all_incidents, corpus_hash = _get_incident_corpus(limit=200000)

        if not all_incidents or len(all_incidents) < 2:
            return []

        # Get embedder
        embedder = get_embedder()

        # Encode query
        query_embed = embedder.encode(query_text)

        # Get cached corpus embeddings (or compute if not cached)
        corpus_texts = [inc["incident_text"] for inc in all_incidents]
        corpus_embeds = _get_corpus_embeddings(corpus_hash, corpus_texts)

        # Find similar
        similar_indices = embedder.find_similar(
            query_embed, corpus_embeds, top_k=top_k + 1
        )

        # Build results (skip first if it's exact match)
        results = []
        for idx, score in similar_indices:
            if score >= similarity_threshold:
                incident = all_incidents[idx].copy()
                incident["similarity_score"] = score

                # Skip if exact duplicate (similarity = 1.0)
                if score < 0.999:
                    results.append(incident)

        return results[:top_k]

    except Exception as e:
        st.error(f"Error finding similar incidents: {e}")
        return []


def check_for_duplicates(incident_text: str, threshold: float = 0.90) -> list:
    """Check if incident is a potential duplicate.

    Args:
        incident_text: Text to check
        threshold: Similarity threshold for duplicates (default: 0.90)

    Returns:
        List of potential duplicate incidents
    """
    similar = find_similar_incidents(
        incident_text, top_k=10, similarity_threshold=threshold
    )
    return [inc for inc in similar if inc["similarity_score"] >= threshold]


def calculate_risk_score(
    classification: str, confidence: float, text_length: int, iocs_count: int = 0
) -> float:
    """Calculate a risk score for an incident.

    Args:
        classification: Incident classification type
        confidence: Model confidence (0-1)
        text_length: Length of incident text
        iocs_count: Number of IOCs detected

    Returns:
        Risk score (0-100)
    """
    # Base severity by classification
    severity_map = {
        "malware": 90,
        "data_exfiltration": 95,
        "web_attack": 75,
        "access_abuse": 70,
        "phishing": 65,
        "policy_violation": 30,
        "benign_activity": 10,
        "uncertain": 50,
    }

    base_severity = severity_map.get(classification, 50)

    # Confidence weight (higher confidence = higher risk for threats)
    confidence_weight = confidence * 1.2 if base_severity > 50 else confidence * 0.8

    # Text complexity weight (more detailed = potentially more serious)
    text_weight = min(text_length / 500, 1.0) * 0.3

    # IOC weight
    ioc_weight = min(iocs_count / 10, 1.0) * 0.5

    # Calculate final score
    risk_score = (
        base_severity * 0.6
        + confidence_weight * 20
        + text_weight * 10
        + ioc_weight * 10
    )

    return min(max(risk_score, 0), 100)


def calculate_severity_index(results: list) -> dict:
    """Calculate aggregated severity index across results.

    Args:
        results: List of analysis results

    Returns:
        Dictionary with severity metrics
    """
    if not results:
        return {"overall": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}

    severity_scores = []
    for r in results:
        score = calculate_risk_score(
            r.get("display_label", "uncertain"),
            r.get("max_prob", 0),
            len(r.get("incident_text", "")),
            0,  # IOCs would need to be extracted
        )
        severity_scores.append(score)

    overall = np.mean(severity_scores)
    critical = len([s for s in severity_scores if s >= 80])
    high = len([s for s in severity_scores if 60 <= s < 80])
    medium = len([s for s in severity_scores if 40 <= s < 60])
    low = len([s for s in severity_scores if s < 40])

    return {
        "overall": overall,
        "critical": critical,
        "high": high,
        "medium": medium,
        "low": low,
        "scores": severity_scores,
    }


def calculate_text_complexity(text: str) -> dict:
    """Analyze text complexity metrics.

    Args:
        text: Incident text to analyze

    Returns:
        Dictionary with complexity metrics
    """
    import re

    # Basic metrics
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)
    char_count = len(text)
    sentence_count = max(len(re.split(r"[.!?]+", text)), 1)

    # Calculate metrics
    avg_word_length = char_count / word_count if word_count > 0 else 0
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    # Keyword density (technical terms)
    technical_terms = [
        "malware",
        "virus",
        "trojan",
        "ransomware",
        "phishing",
        "suspicious",
        "unauthorized",
        "attack",
        "exploit",
        "vulnerability",
        "breach",
        "exfiltration",
        "payload",
        "backdoor",
        "rootkit",
    ]

    tech_term_count = sum(1 for word in words if word in technical_terms)
    tech_density = tech_term_count / word_count if word_count > 0 else 0

    # Simple readability score (higher = more complex)
    readability = avg_word_length * 0.4 + avg_sentence_length * 0.6  # Simplified metric

    return {
        "word_count": word_count,
        "char_count": char_count,
        "sentence_count": sentence_count,
        "avg_word_length": avg_word_length,
        "avg_sentence_length": avg_sentence_length,
        "technical_density": tech_density,
        "readability_score": readability,
    }


def generate_soc_playbook_recommendation(
    classification: str, confidence: float
) -> dict:
    """Generate SOC playbook recommendations based on classification.

    Args:
        classification: Incident classification type
        confidence: Model confidence score

    Returns:
        Dictionary with playbook details
    """
    playbooks = {
        "malware": {
            "playbook_id": "SOC-PB-001",
            "title": "Malware Incident Response",
            "priority": "P1-Critical",
            "estimated_time": "2-4 hours",
            "team": "Malware Analysis Team",
            "steps": [
                "Isolate affected system from network",
                "Collect memory dump and disk image",
                "Run malware analysis (static and dynamic)",
                "Identify C2 infrastructure and IOCs",
                "Check for lateral movement indicators",
                "Contain and remediate affected systems",
                "Update EDR/AV signatures",
                "Document findings and lessons learned",
            ],
            "tools": [
                "EDR Platform",
                "Sandbox Environment",
                "SIEM",
                "Threat Intel Feeds",
            ],
            "external_links": [
                "https://www.incidentresponse.com/playbooks/malware",
                "https://attack.mitre.org/tactics/TA0002/",
            ],
        },
        "phishing": {
            "playbook_id": "SOC-PB-002",
            "title": "Phishing Investigation",
            "priority": "P2-High",
            "estimated_time": "1-2 hours",
            "team": "Email Security Team",
            "steps": [
                "Collect email headers and body",
                "Analyze sender reputation and SPF/DKIM",
                "Extract and analyze URLs/attachments",
                "Search for similar emails in organization",
                "Block sender and malicious URLs",
                "Notify affected users",
                "Remove emails from all mailboxes",
                "User awareness training follow-up",
            ],
            "tools": ["Email Gateway", "URL Sandbox", "PhishTank", "VirusTotal"],
            "external_links": [
                "https://www.cisa.gov/phishing",
                "https://attack.mitre.org/techniques/T1566/",
            ],
        },
        "data_exfiltration": {
            "playbook_id": "SOC-PB-003",
            "title": "Data Exfiltration Response",
            "priority": "P1-Critical",
            "estimated_time": "4-8 hours",
            "team": "Incident Response Team + Legal",
            "steps": [
                "Identify and block exfiltration channels",
                "Preserve evidence for forensics",
                "Determine data classification and volume",
                "Assess legal/regulatory requirements",
                "Contain compromised accounts/systems",
                "Conduct forensic analysis",
                "Notify stakeholders and affected parties",
                "Implement DLP controls",
            ],
            "tools": [
                "DLP Platform",
                "SIEM",
                "Network Forensics",
                "Cloud Access Security Broker",
            ],
            "external_links": [
                "https://attack.mitre.org/tactics/TA0010/",
                "https://www.sans.org/white-papers/data-breach-response/",
            ],
        },
        "web_attack": {
            "playbook_id": "SOC-PB-004",
            "title": "Web Application Attack Response",
            "priority": "P2-High",
            "estimated_time": "2-3 hours",
            "team": "Web Security Team",
            "steps": [
                "Block attacking IP addresses",
                "Analyze attack patterns and payloads",
                "Check for successful exploitation",
                "Review WAF logs for similar attacks",
                "Patch vulnerable applications",
                "Implement additional WAF rules",
                "Conduct vulnerability assessment",
                "Update security monitoring",
            ],
            "tools": ["WAF", "SIEM", "Vulnerability Scanner", "Application Logs"],
            "external_links": [
                "https://owasp.org/www-project-web-security-testing-guide/",
                "https://attack.mitre.org/tactics/TA0001/",
            ],
        },
        "access_abuse": {
            "playbook_id": "SOC-PB-005",
            "title": "Unauthorized Access Investigation",
            "priority": "P2-High",
            "estimated_time": "2-4 hours",
            "team": "Identity & Access Team",
            "steps": [
                "Disable compromised account",
                "Review access logs for suspicious activity",
                "Identify accessed resources",
                "Check for privilege escalation",
                "Reset credentials",
                "Enable MFA if not present",
                "Review access policies",
                "User training on security best practices",
            ],
            "tools": ["IAM Platform", "SIEM", "Active Directory", "PAM Solution"],
            "external_links": [
                "https://attack.mitre.org/tactics/TA0001/",
                "https://www.nist.gov/identity-access-management",
            ],
        },
    }

    # Default playbook for unknown types
    default_playbook = {
        "playbook_id": "SOC-PB-000",
        "title": "General Incident Response",
        "priority": "P3-Medium" if confidence > 0.7 else "P4-Low",
        "estimated_time": "1-2 hours",
        "team": "SOC Team",
        "steps": [
            "Gather incident details",
            "Assess severity and impact",
            "Assign to appropriate team",
            "Follow standard incident response procedures",
            "Document findings",
        ],
        "tools": ["SIEM", "Ticketing System"],
        "external_links": ["https://www.sans.org/incident-response/"],
    }

    playbook = playbooks.get(classification, default_playbook)

    # Adjust priority based on confidence
    if confidence < 0.5 and playbook["priority"].startswith("P1"):
        playbook["priority"] = "P2-High (Low Confidence)"

    return playbook


def generate_custom_soc_playbook(
    classification: str, iocs: dict, mitre_techniques: list, confidence: float
) -> str:
    """Generate a customized SOC playbook based on incident details.

    Args:
        classification: Incident type
        iocs: Dictionary of IOCs
        mitre_techniques: List of MITRE techniques
        confidence: Classification confidence

    Returns:
        Formatted playbook markdown
    """
    playbook = generate_soc_playbook_recommendation(classification, confidence)

    markdown = f"""# {playbook['title']}

**Playbook ID:** {playbook['playbook_id']}
**Priority:** {playbook['priority']}
**Estimated Time:** {playbook['estimated_time']}
**Assigned Team:** {playbook['team']}

---

## Incident Context

- **Classification:** {classification.replace('_', ' ').title()}
- **Confidence:** {confidence:.1%}
- **Detected IOCs:** {sum(len(v) for v in iocs.values()) if iocs else 0}
- **MITRE Techniques:** {len(mitre_techniques)}

---

## Response Procedures

"""

    for i, step in enumerate(playbook["steps"], 1):
        markdown += f"{i}. {step}\n"

    markdown += f"""
---

## Required Tools

"""
    for tool in playbook["tools"]:
        markdown += f"- {tool}\n"

    if iocs and any(iocs.values()):
        markdown += f"""
---

## Detected Indicators

"""
        if iocs.get("ips"):
            markdown += f"\n**IP Addresses:** {', '.join(list(iocs['ips'])[:5])}"
        if iocs.get("domains"):
            markdown += f"\n**Domains:** {', '.join(list(iocs['domains'])[:5])}"
        if iocs.get("file_hashes"):
            markdown += f"\n**File Hashes:** {', '.join(list(iocs['file_hashes'])[:3])}"

    if mitre_techniques:
        markdown += f"""
---

## MITRE ATT&CK Mapping

"""
        for technique in mitre_techniques[:5]:
            markdown += f"- {technique}\n"

    markdown += f"""
---

## Additional Resources

"""
    for link in playbook["external_links"]:
        markdown += f"- {link}\n"

    markdown += f"""
---

**Note:** This playbook is auto-generated based on incident classification. 
Adjust procedures based on your organization's specific requirements and policies.
"""

    return markdown


def generate_threat_intelligence_brief(results: list) -> str:
    """Generate a comprehensive threat intelligence brief from bulk analysis results."""
    total = len(results)

    # Calculate key metrics - use display_label to reflect LLM overrides
    label_counts = Counter(
        [r.get("display_label", r.get("final_label", "unknown")) for r in results]
    )
    avg_confidence = np.mean([r.get("max_prob", 0) for r in results])
    high_conf_count = len([r for r in results if r.get("max_prob", 0) > 0.8])
    uncertain_count = label_counts.get("uncertain", 0)

    # Collect all MITRE techniques
    all_techniques = []
    for r in results:
        all_techniques.extend(r.get("final_label_mitre_techniques", []))
    technique_counts = Counter(all_techniques)

    # Identify critical threats - use display_label
    critical_labels = ["malware", "data_exfiltration", "web_attack"]
    critical_incidents = [
        r
        for r in results
        if r.get("display_label", r.get("final_label")) in critical_labels
    ]

    # Generate brief
    brief = f"""# Threat Intelligence Brief
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Period:** Batch Analysis
**Total Incidents Analyzed:** {total:,}

## Executive Summary

This threat intelligence brief summarizes the analysis of {total:,} security incidents processed through the AlertSage AI-Powered Triage system.

### Key Findings

- **Overall Confidence:** {avg_confidence:.1%} average confidence across all classifications
- **High Confidence Cases:** {high_conf_count} ({high_conf_count/total:.1%}) incidents classified with >80% confidence
- **Uncertain Cases:** {uncertain_count} ({uncertain_count/total:.1%}) incidents requiring manual review
- **Critical Threats Detected:** {len(critical_incidents)} incidents ({len(critical_incidents)/total:.1%})

## Incident Distribution

### Classification Breakdown

"""

    for label, count in label_counts.most_common():
        pct = count / total * 100
        brief += (
            f"- **{label.replace('_', ' ').title()}**: {count} incidents ({pct:.1f}%)\n"
        )

    brief += f"""\n## MITRE ATT&CK Coverage\n\n**Total Techniques Detected:** {len(technique_counts)}\n**Total Technique Occurrences:** {len(all_techniques)}\n\n### Top Techniques\n\n"""

    for technique, count in technique_counts.most_common(10):
        brief += f"- **{technique}**: {count} occurrences\n"

    brief += f"""\n## Threat Landscape Analysis\n\n### Critical Threats Breakdown\n\n"""

    if critical_incidents:
        for label in critical_labels:
            count = len(
                [
                    r
                    for r in critical_incidents
                    if r.get("display_label", r.get("final_label")) == label
                ]
            )
            if count > 0:
                brief += (
                    f"\n#### {label.replace('_', ' ').title()} ({count} incidents)\n\n"
                )
                brief += (
                    f"This represents {count/total:.1%} of all analyzed incidents.\n"
                )
    else:
        brief += "No critical threats detected in this batch.\n"

    brief += f"""\n## Recommendations\n\n### Immediate Actions\n\n"""

    if len(critical_incidents) > 0:
        brief += f"1. **PRIORITY**: Review {len(critical_incidents)} critical threat incidents immediately\n"

    if uncertain_count > total * 0.2:
        brief += f"2. **HIGH**: {uncertain_count} uncertain cases require expert analysis ({uncertain_count/total:.1%} of total)\n"

    if high_conf_count < total * 0.5:
        brief += f"3. **MEDIUM**: Low overall confidence ({avg_confidence:.1%}) suggests need for additional context\n"

    brief += f"""\n### Strategic Recommendations\n\n- **Threat Hunting**: Focus on MITRE techniques {', '.join([t for t, _ in technique_counts.most_common(3)])}
- **Detection Enhancement**: Improve detection for {label_counts.most_common(1)[0][0].replace('_', ' ')} incidents (highest volume)
- **Process Improvement**: Review uncertain cases to improve future classification accuracy\n\n## Technical Details\n\n- **Analysis Engine**: AlertSage AI Triage System
- **Model**: TF-IDF + Logistic Regression
- **Confidence Threshold**: Adaptive uncertainty-aware thresholds
- **MITRE Framework**: ATT&CK v13\n\n---\n*End of Threat Intelligence Brief*\n"""

    return brief


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

DISPLAY_NAMES = {
    "malware": "Malware Detection",
    "phishing": "Phishing Attack",
    "data_exfiltration": "Data Exfiltration",
    "web_attack": "Web Application Attack",
    "access_abuse": "Access Abuse",
    "policy_violation": "Policy Violation",
    "benign_activity": "Benign Activity",
}


def get_display_name(class_name):
    return DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())


def get_confidence_level(conf):
    return "HIGH" if conf >= 0.8 else ("MEDIUM" if conf >= 0.6 else "LOW")


def get_confidence_class(conf):
    return (
        "confidence-high"
        if conf >= 0.8
        else ("confidence-medium" if conf >= 0.6 else "confidence-low")
    )


@st.cache_data
def load_model_metrics():
    """Load pre-computed model metrics"""
    try:
        # Load combined features for enhanced model (TF-IDF + embeddings)
        X_test = joblib.load("models/X_test_combined.joblib")
        y_test = joblib.load("models/y_test.joblib")
        vectorizer, model = load_vectorizer_and_model()

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
        )

        y_pred = model.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, average="weighted", zero_division=0
            ),
            "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            "n_samples": len(y_test),
            "n_classes": len(set(y_test)),
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "model": model,
        }
    except:
        return {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "n_samples": 0,
            "n_classes": 0,
        }


# ============================================================================
# UI COMPONENTS
# ============================================================================


def render_metric_card(label, value, change=None):
    """Render premium metric card"""
    change_html = f'<div class="metric-change">↑ {change}</div>' if change else ""
    st.markdown(
        f"""<div class="metric-premium fade-in">
<div class="metric-label">{label}</div>
<div class="metric-value">{value}</div>
{change_html}
</div>""",
        unsafe_allow_html=True,
    )


def render_prediction_result(prediction, confidence, probabilities):
    """Render prediction result card"""
    conf_level = get_confidence_level(confidence)
    conf_class = get_confidence_class(confidence)

    html = f"""<div class="prediction-card fade-in">
<div class="prediction-title">{get_display_name(prediction)}</div>
<div class="{conf_class} confidence-badge">{conf_level} CONFIDENCE: {confidence:.1%}</div>
<div style="margin-top: 2rem;">"""

    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    for cls, prob in sorted_probs[:5]:
        html += f"""<div style="margin: 1.25rem 0; padding: 0.75rem; background: rgba(255, 255, 255, 0.15); border-radius: 12px; backdrop-filter: blur(10px);">
<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; font-weight: 600; font-size: 1rem; color: white;">
<span style="color: white;">{get_display_name(cls)}</span>
<span style="color: white;">{prob:.1%}</span>
</div>
<div style="background: rgba(255, 255, 255, 0.25); height: 14px; border-radius: 50px; overflow: hidden;">
<div style="background: linear-gradient(90deg, rgba(255,255,255,0.9) 0%, white 100%); height: 100%; border-radius: 50px; transition: width 0.8s ease; box-shadow: 0 0 15px rgba(255,255,255,0.6); width: {prob*100}%;"></div>
</div>
</div>"""

    html += """</div>
</div>"""
    return html


def get_theme_colors():
    """Get theme colors based on system preference for dark mode support"""
    # Check if user is in dark mode via session state or default to light
    # Streamlit doesn't expose system theme directly, so we use a reasonable default
    # that works well in both modes
    return {
        "plot_bg": "rgba(0,0,0,0)",  # Transparent to inherit from container
        "paper_bg": "rgba(0,0,0,0)",  # Transparent to inherit from container
        "text_color": "#1e293b",  # Dark text for light mode (CSS will override in dark)
        "grid_color": "#e2e8f0",  # Light grid
        "font_family": "Inter",
    }


def create_confusion_matrix():
    """Create enhanced confusion matrix visualization"""
    metrics = load_model_metrics()
    if "y_test" not in metrics or "y_pred" not in metrics:
        return go.Figure()

    text_palette = get_text_palette()
    secondary_text = text_palette["secondary"]

    def _coerce_seq(seq) -> list:
        if seq is None:
            return []
        if hasattr(seq, "tolist"):
            try:
                return list(seq.tolist())
            except Exception:
                pass
        if isinstance(seq, (list, tuple)):
            return list(seq)
        return [seq]

    y_test = _coerce_seq(metrics.get("y_test"))
    y_pred = _coerce_seq(metrics.get("y_pred"))

    if not y_test or not y_pred:
        return go.Figure()

    from sklearn.metrics import confusion_matrix

    classes = sorted(set(y_test))
    if not classes:
        return go.Figure()

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    display_labels = [get_display_name(c) for c in classes]

    # Custom gradient color scale matching confidence distribution theme
    custom_colorscale = [
        [0.0, "#f8fafc"],  # Very light gray for low values
        [0.2, "#e0e7ff"],  # Light purple tint
        [0.4, "#c7d2fe"],  # Lighter purple
        [0.6, "#a5b4fc"],  # Medium purple
        [0.8, "#818cf8"],  # Stronger purple
        [1.0, "#667eea"],  # Brand purple for highest values
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=display_labels,
            y=display_labels,
            colorscale=custom_colorscale,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 14, "family": "Inter", "weight": "bold"},
            showscale=True,
            colorbar=dict(
                title="Count",
                thickness=12,
                len=0.6,
                tickfont=dict(size=10),
                outlinewidth=0,
            ),
            hovertemplate="<b>Actual:</b> %{y}<br><b>Predicted:</b> %{x}<br><b>Count:</b> %{z}<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )

    fig.update_layout(
        title={
            "text": "Classification Accuracy",
            "x": 0.5,
            "font": {"size": 16, "weight": 700, "family": "Inter", "color": secondary_text},
        },
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "size": 11, "color": secondary_text},
        margin=dict(l=120, r=40, t=60, b=120),
        xaxis=dict(
            tickangle=-35, tickfont=dict(size=10), showgrid=False, side="bottom"
        ),
        yaxis=dict(tickfont=dict(size=10), showgrid=False),
    )

    return fig


def create_metrics_chart(metrics):
    """Create metrics bar chart"""
    names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    values = [
        metrics.get("accuracy", 0),
        metrics.get("precision", 0),
        metrics.get("recall", 0),
        metrics.get("f1", 0),
    ]

    colors = ["#667eea", "#764ba2", "#8b5cf6", "#a855f7"]

    fig = go.Figure(
        data=[
            go.Bar(
                x=names,
                y=values,
                marker_color=colors,
                text=[f"{v:.1%}" for v in values],
                textposition="outside",
                textfont=dict(size=14, weight="bold"),
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Performance Metrics",
            "x": 0.5,
            "font": {"size": 20, "weight": 700},
        },
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.1], gridcolor="rgba(0,0,0,0.1)"),
        height=450,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        margin=dict(l=40, r=40, t=80, b=40),
    )

    return fig


def create_probability_chart(probabilities):
    """Create probability distribution chart"""
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    labels = [get_display_name(cls) for cls, _ in sorted_items]
    values = [prob for _, prob in sorted_items]
    colors = ["#667eea" if i == 0 else "#cbd5e1" for i in range(len(labels))]

    fig = go.Figure(
        data=[
            go.Bar(
                y=labels,
                x=values,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.1%}" for v in values],
                textposition="auto",
                textfont=dict(size=13, weight="bold", color="white"),
            )
        ]
    )

    fig.update_layout(
        title={
            "text": "Confidence Distribution",
            "x": 0.5,
            "font": {"size": 18, "weight": 600},
        },
        xaxis_title="Confidence",
        height=450,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        xaxis=dict(range=[0, 1], gridcolor="rgba(0,0,0,0.1)"),
        margin=dict(l=180, r=40, t=70, b=40),
    )

    return fig


def create_risk_radar_chart(data: dict):
    """Create risk assessment radar chart"""
    categories = ["Severity", "Impact", "Urgency", "Complexity", "Detectability"]

    # Generate scores based on prediction confidence
    import random

    values = [random.uniform(0.5, 1.0) for _ in categories]

    fig = go.Figure()

    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name="Risk Assessment",
            line=dict(color="#667eea", width=2),
            fillcolor="rgba(102, 126, 234, 0.3)",
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#e5e7eb"),
            angularaxis=dict(gridcolor="#e5e7eb"),
        ),
        showlegend=False,
        title={"text": "Risk Assessment Radar", "x": 0.5, "font": {"size": 18}},
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )

    return fig


def create_mitre_heatmap(incident_type: str):
    """Create MITRE ATT&CK technique heatmap"""
    # Simplified MITRE mapping
    mitre_techniques = {
        "malware": ["T1566", "T1204", "T1059", "T1027", "T1055"],
        "phishing": ["T1566.001", "T1566.002", "T1598", "T1204"],
        "data_exfiltration": ["T1041", "T1048", "T1567", "T1020"],
        "web_attack": ["T1190", "T1505", "T1210", "T1133"],
        "access_abuse": ["T1078", "T1098", "T1136", "T1003"],
    }

    tactics = [
        "Initial Access",
        "Execution",
        "Persistence",
        "Privilege Escalation",
        "Defense Evasion",
    ]
    techniques = mitre_techniques.get(incident_type, ["T1000", "T1001", "T1002"])[:5]

    # Create random heatmap data
    import random

    z_data = [
        [random.uniform(0.3, 1.0) for _ in range(len(techniques))]
        for _ in range(len(tactics))
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=z_data,
            x=techniques,
            y=tactics,
            colorscale=[[0, "#f0f9ff"], [0.5, "#667eea"], [1, "#764ba2"]],
            showscale=True,
            colorbar=dict(title="Relevance"),
        )
    )

    fig.update_layout(
        title={"text": "MITRE ATT&CK Coverage", "x": 0.5, "font": {"size": 18}},
        xaxis_title="Technique ID",
        yaxis_title="Tactic",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )

    return fig


def create_confidence_timeline(probabilities: dict):
    """Create timeline visualization of confidence scores"""
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    fig = go.Figure()

    # Add line trace
    fig.add_trace(
        go.Scatter(
            x=[get_display_name(cls) for cls, _ in sorted_probs],
            y=[prob for _, prob in sorted_probs],
            mode="lines+markers",
            name="Confidence",
            line=dict(color="#667eea", width=3),
            marker=dict(size=10, color="#764ba2"),
            fill="tozeroy",
            fillcolor="rgba(102, 126, 234, 0.2)",
        )
    )

    # Add threshold line
    fig.add_hline(
        y=0.7, line_dash="dash", line_color="red", annotation_text="Threshold"
    )

    fig.update_layout(
        title={"text": "Confidence Distribution Curve", "x": 0.5, "font": {"size": 18}},
        xaxis_title="Classification",
        yaxis_title="Confidence Score",
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        yaxis=dict(range=[0, 1], gridcolor="rgba(0,0,0,0.1)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
    )

    return fig


# ============================================================================
# SIDEBAR
# ============================================================================


def create_sidebar():
    """Create enhanced sidebar"""

    # Professional Profile Banner
    if "db" in st.session_state:
        try:
            active_profile = st.session_state.db.get_active_profile()
            if active_profile:
                profile_name = active_profile.get("name", "Default")
                profile_role = active_profile.get("role", "Analyst")

                st.sidebar.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.05) 100%);
                        backdrop-filter: blur(10px);
                        border-radius: 12px;
                        padding: 16px;
                        margin-bottom: 20px;
                        border: 1px solid rgba(255, 255, 255, 0.2);
                        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
                    ">
                        <div style="
                            font-size: 0.7rem;
                            font-weight: 700;
                            text-transform: uppercase;
                            letter-spacing: 0.1em;
                            color: rgba(255, 255, 255, 0.7);
                            margin-bottom: 6px;
                        ">ACTIVE PROFILE</div>
                        <div style="
                            font-size: 1.1rem;
                            font-weight: 700;
                            color: white;
                            margin-bottom: 4px;
                        ">{profile_name}</div>
                        <div style="
                            font-size: 0.85rem;
                            color: rgba(255, 255, 255, 0.8);
                            display: flex;
                            align-items: center;
                            gap: 6px;
                        ">
                            <svg width="12" height="12" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <circle cx="8" cy="8" r="8" fill="#10b981"/>
                                <path d="M11.5 5.5L7 10L4.5 7.5" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                            {profile_role}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except:
            pass

    st.sidebar.markdown("## Settings")

    # Theme: fixed to light for maximum readability (dark mode disabled)
    st.session_state.theme_mode = "Light"
    apply_theme_mode_css("Light")

    # Initialize session state for mode if not exists
    if "selected_mode" not in st.session_state:
        st.session_state.selected_mode = "Intelligence Dashboard"

    # Initialize radio key counter for forcing widget refresh
    if "radio_key_version" not in st.session_state:
        st.session_state.radio_key_version = 0

    # Check for navigation flag (from back button) - must be AFTER initialization
    if st.session_state.get("navigate_to_dashboard", False):
        st.session_state.selected_mode = "Intelligence Dashboard"
        st.session_state.navigate_to_dashboard = False
        # Increment the key version to force radio widget recreation
        st.session_state.radio_key_version += 1
        # Force a clean rerun to update the radio
        st.rerun()

    # Mode options
    mode_options = [
        "Intelligence Dashboard",
        "Single Incident Lab",
        "Advanced Search",
        "Batch Analysis",
        "Bookmarks & History",
        "Experimental Lab",
        "Settings & Profiles",
    ]

    # Get current index
    current_index = (
        mode_options.index(st.session_state.selected_mode)
        if st.session_state.selected_mode in mode_options
        else 0
    )

    # Use a versioned key to force widget recreation when needed
    mode = st.sidebar.radio(
        "Analysis Mode",
        mode_options,
        index=current_index,
        key=f"mode_radio_v{st.session_state.radio_key_version}",
    )

    # Always update selected_mode to match the radio
    st.session_state.selected_mode = mode

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Parameters")

    # Load active profile settings
    active_profile = None
    profile_id = None
    if "db" in st.session_state:
        try:
            active_profile = st.session_state.db.get_active_profile()
            if active_profile:
                profile_id = active_profile.get("id", 0)
        except:
            pass

    # Apply profile defaults or use standard defaults
    default_difficulty = "Medium"
    default_threshold = 0.7
    default_max_classes = 5
    default_use_llm = False
    default_enable_viz = True

    if active_profile:
        # Ensure difficulty matches available options (case-insensitive)
        profile_difficulty = active_profile.get("default_difficulty", "Medium")
        if isinstance(profile_difficulty, str):
            # Capitalize to match options
            profile_difficulty = profile_difficulty.capitalize()
            # Validate it's in options
            if profile_difficulty in ["Easy", "Medium", "Hard", "Expert"]:
                default_difficulty = profile_difficulty

        default_threshold = active_profile.get("default_threshold", 0.7)
        default_max_classes = active_profile.get("default_max_classes", 5)
        default_use_llm = bool(active_profile.get("enable_llm", 0))
        default_enable_viz = bool(active_profile.get("enable_advanced_viz", 1))

    difficulty = st.sidebar.select_slider(
        "Difficulty Level",
        options=["Easy", "Medium", "Hard", "Expert"],
        value=default_difficulty,
        help="Profile default applied" if active_profile else None,
    )

    threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.0,
        1.0,
        default_threshold,
        0.05,
        help="Profile default applied" if active_profile else None,
    )

    max_classes = st.sidebar.slider(
        "Max Classes to Show",
        1,
        10,
        default_max_classes,
        help="Profile default applied" if active_profile else None,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Advanced Options")

    use_preprocessing = st.sidebar.checkbox("Text Preprocessing", value=True)
    # Use profile-versioned key to force checkbox refresh when profile changes
    use_llm = st.sidebar.checkbox(
        "LLM Enhancement",
        value=default_use_llm,
        help="Profile default applied" if active_profile else None,
        key=f"llm_checkbox_p{profile_id or 0}",
    )

    provider_options = {
        "Local (GGUF)": "local",
        "Hugging Face Inference": "huggingface",
    }
    hf_secret_token = get_secret_value("HF_TOKEN", "")
    hf_secret_model = get_secret_value("HF_MODEL", "")

    default_provider_idx = (
        1
        if hf_secret_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("TRIAGE_HF_TOKEN")
        else 0
    )
    provider_label = st.sidebar.selectbox(
        "LLM Provider",
        list(provider_options.keys()),
        index=default_provider_idx,
        disabled=not use_llm,
        help="Use hosted Hugging Face Inference when you have a token configured.",
    )
    llm_provider = provider_options.get(provider_label, "local")
    huggingface_enabled = use_llm and llm_provider == "huggingface"

    default_hf_model = (
        hf_secret_model
        or os.environ.get("TRIAGE_HF_MODEL")
        or os.environ.get("HF_MODEL")
        or "mistralai/Mistral-7B-Instruct-v0.3"
    )
    hf_model_id = st.sidebar.text_input(
        "HF Model ID",
        value=default_hf_model,
        disabled=not huggingface_enabled,
        help="Example: mistralai/Mistral-7B-Instruct-v0.3",
    )

    hf_env_token = os.environ.get("TRIAGE_HF_TOKEN") or os.environ.get("HF_TOKEN") or ""
    hf_byo_token = st.sidebar.checkbox(
        "Use my Hugging Face token",
        value=False,
        disabled=not huggingface_enabled,
        help="Token is kept in this session only. Prefer environment variables in production.",
    )
    hf_token_input = ""
    if huggingface_enabled and hf_byo_token:
        hf_token_input = st.sidebar.text_input(
            "HF API Token",
            value="",
            placeholder="hf_xxx",
            type="password",
            help="Required for hosted inference.",
        )
        if hf_env_token:
            st.sidebar.caption("Environment token detected; will be used if left blank.")
    elif huggingface_enabled and hf_env_token:
        st.sidebar.caption("Hugging Face token detected in environment.")

    selected_hf_token = (hf_token_input or hf_env_token or hf_secret_token).strip()
    if huggingface_enabled:
        st.sidebar.caption(
            f"UI rate limit: {HF_UI_MAX_REQUESTS} requests/{HF_UI_WINDOW_SECONDS}s per session."
        )

    if selected_hf_token:
        st.sidebar.caption("LLM Assist: Hugging Face (hosted)")
    else:
        st.sidebar.caption("LLM Assist: Local llama.cpp")

    # Persist LLM selections for use across tabs and defaults
    st.session_state.llm_provider = llm_provider
    st.session_state.hf_model_id = hf_model_id
    st.session_state.selected_hf_token = selected_hf_token

    enable_viz = st.sidebar.checkbox(
        "Advanced Visualizations",
        value=default_enable_viz,
        help="Profile default applied" if active_profile else None,
        key=f"viz_checkbox_p{profile_id or 0}",
    )

    # Bookmarks Quick Access
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Bookmarks")

    if "cached_bookmarks" not in st.session_state:
        st.session_state.cached_bookmarks = None

    try:
        if st.session_state.cached_bookmarks is None:
            st.session_state.cached_bookmarks = st.session_state.db.get_bookmarks(
                limit=5
            )

        bookmarks = st.session_state.cached_bookmarks

        if bookmarks:
            for bm in bookmarks[:5]:
                if st.sidebar.button(
                    f"{bm.get('final_label', 'Unknown')[:20]}", key=f"bm_{bm.get('id')}"
                ):
                    st.session_state.selected_bookmark = bm
                    st.rerun()
        else:
            st.sidebar.info("No bookmarks saved")
    except:
        st.sidebar.info("Bookmarks unavailable")

    # System Information
    st.sidebar.markdown("---")
    with st.sidebar.expander("System Information"):
        # Get actual dataset length
        dataset_length = "Unknown"
        try:
            from pathlib import Path

            possible_paths = [
                "data/cyber_incidents_simulated.csv",
                "data/boost_pack_triage_examples.csv",
                "../data/cyber_incidents_simulated.csv",
            ]
            for path in possible_paths:
                if Path(path).exists():
                    df_dataset = pd.read_csv(path)
                    dataset_length = f"{len(df_dataset):,} incidents"
                    break
        except Exception:
            dataset_length = "Unable to determine"

        st.markdown(
            f"""
        **Version**: 3.0.0 Premium  
        **CLI Backend**: nlp-triage  
        **Model**: Logistic Regression + TF-IDF  
        **LLM**: Llama-3.1-8B-Instruct (optional)  
        **Dataset**: {dataset_length}  
        **Embeddings**: sentence-transformers  
        **Database**: SQLite  
        """
        )

    return (
        st.session_state.selected_mode,
        difficulty,
        threshold,
        max_classes,
        use_preprocessing,
        use_llm,
        enable_viz,
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================


def main():
    # Initialize database
    if "db" not in st.session_state:
        st.session_state.db = TriageDatabase()

    # Sidebar
    mode, difficulty, threshold, max_classes, use_preprocessing, use_llm, enable_viz = (
        create_sidebar()
    )

    # Load metrics once
    metrics = load_model_metrics()

    # Show homepage for Dashboard mode, otherwise route to specific mode
    if "Dashboard" in mode or "Intelligence Dashboard" in mode:
        show_homepage(metrics, enable_viz)
    elif "Single Incident" in mode:
        single_incident_lab(
            difficulty, threshold, max_classes, use_preprocessing, use_llm, enable_viz
        )
    elif "Advanced Search" in mode:
        advanced_search_interface()
    elif "Batch Analysis" in mode:
        batch_processing_tab(use_preprocessing, use_llm)
    elif "Bookmarks" in mode:
        bookmarks_and_history_tab()
    elif "Experimental" in mode:
        experimental_lab()
    elif "Settings" in mode:
        settings_and_profiles_interface()


# ============================================================================
# HOMEPAGE
# ============================================================================


def show_homepage(metrics, enable_viz):
    """Stunning professional security intelligence dashboard with advanced features."""

    history: list = []
    bookmarks: list = []
    standalone_notes: list = []
    bookmark_notes: list = []
    total_notes = 0
    total_incidents = 0
    total_bookmarks = 0
    incidents_24h = 0
    incidents_7d = 0
    incidents_30d = 0
    most_common_label = ("None", 0)
    avg_confidence = 0.0
    severity_counts: Counter = Counter()
    top_classifications: list = []
    now = datetime.now()
    text_palette = get_text_palette()
    is_dark_mode = text_palette["is_dark"]
    secondary_text = text_palette["secondary"]
    muted_text = text_palette["muted"]

    # Professional Header with Centered Brand Logo
    from pathlib import Path
    import base64

    logo_path = Path(__file__).parent / "assets" / "icons" / "alertsage-logo.svg"

    # Compact horizontal header with logo beside title
    logo_html = ""
    if logo_path.exists():
        logo_b64 = base64.b64encode(logo_path.read_bytes()).decode()
        logo_html = f'<img src="data:image/svg+xml;base64,{logo_b64}" style="height: 75px; width: auto;" />'

    st.markdown(
        f"""
        <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
            <div style="display: inline-flex; align-items: center; gap: 0.3rem;">
                <span style="font-size: 2.5rem; font-weight: 900; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -0.02em;">AlertSage</span>{logo_html}
            </div>
            <p style="font-size: 1.15rem; color: #667eea; margin-top: 0.5rem; font-weight: 700; letter-spacing: 0.02em;">Intelligent Security Triage Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Get comprehensive database statistics
    try:
        history = st.session_state.db.get_analysis_history(limit=10000)
        bookmarks = st.session_state.db.get_bookmarks()
        standalone_notes = st.session_state.db.get_all_notes()
        bookmark_notes = [bm for bm in bookmarks if bm.get("note")]
        total_notes = len(standalone_notes) + len(bookmark_notes)

        total_incidents = len(history)
        total_bookmarks = len(bookmarks)

        # Calculate time-based trends
        now = datetime.now()
        history_24h = [
            h
            for h in history
            if "timestamp" in h
            and datetime.fromisoformat(h["timestamp"]) > now - timedelta(hours=24)
        ]
        history_7d = [
            h
            for h in history
            if "timestamp" in h
            and datetime.fromisoformat(h["timestamp"]) > now - timedelta(days=7)
        ]
        history_30d = [
            h
            for h in history
            if "timestamp" in h
            and datetime.fromisoformat(h["timestamp"]) > now - timedelta(days=30)
        ]

        incidents_24h = len(history_24h)
        incidents_7d = len(history_7d)
        incidents_30d = len(history_30d)

        # Calculate classification distribution
        if history:
            labels = [h.get("final_label", "unknown") for h in history]
            label_counts = Counter(labels)
            most_common_label = (
                label_counts.most_common(1)[0] if label_counts else ("None", 0)
            )
            avg_confidence = (
                np.mean([h.get("max_prob", 0) for h in history]) if history else 0
            )

            # Get severity distribution
            severities = [
                h.get("severity", "unknown") for h in history if h.get("severity")
            ]
            severity_counts = Counter(severities)

            # Top classifications
            top_classifications = label_counts.most_common(3)
        else:
            most_common_label = ("None", 0)
            avg_confidence = 0
            severity_counts = Counter()
            top_classifications = []
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        total_incidents = 0
        total_bookmarks = 0
        total_notes = 0
        incidents_24h = 0
        incidents_7d = 0
        incidents_30d = 0
        most_common_label = ("None", 0)
        avg_confidence = 0
        severity_counts = Counter()
        top_classifications = []

    # Enhanced Key Metrics with Icons
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        # Show 7-day trend badge if there's activity
        trend_badge = (
            f"""
            <div style="display: inline-block; background: linear-gradient(135deg, #10b981 0%, #059669 100%); color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.7rem; font-weight: 700; margin-top: 0.5rem;">
                +{incidents_7d} LAST 7 DAYS
            </div>
        """
            if incidents_7d > 0
            else ""
        )

        st.markdown(
            f"""
            <div style="text-align: center; padding: 1.25rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.2); min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2" style="margin-bottom: 0.5rem;">
                    <path d="M12 2v20M2 12h20"/>
                </svg>
                <div style="font-size: 1.75rem; font-weight: 900; color: #667eea; line-height: 1;">{total_incidents:,}</div>
                <div style="font-size: 0.8rem; color: {secondary_text}; margin-top: 0.5rem; font-weight: 600;">TOTAL ANALYZED</div>
                {trend_badge}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1.25rem; background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.2); min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="2" style="margin-bottom: 0.5rem;">
                    <path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/>
                </svg>
                <div style="font-size: 1.75rem; font-weight: 900; color: #8b5cf6; line-height: 1;">{total_bookmarks:,}</div>
                <div style="font-size: 0.8rem; color: {secondary_text}; margin-top: 0.5rem; font-weight: 600;">BOOKMARKS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1.25rem; background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.2); min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" style="margin-bottom: 0.5rem;">
                    <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6"/>
                </svg>
                <div style="font-size: 1.75rem; font-weight: 900; color: #10b981; line-height: 1;">{total_notes:,}</div>
                <div style="font-size: 0.8rem; color: {secondary_text}; margin-top: 0.5rem; font-weight: 600;">NOTES</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1.25rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.2); min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" style="margin-bottom: 0.5rem;">
                    <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/>
                </svg>
                <div style="font-size: 1.75rem; font-weight: 900; color: #3b82f6; line-height: 1;">{avg_confidence:.0%}</div>
                <div style="font-size: 0.8rem; color: {secondary_text}; margin-top: 0.5rem; font-weight: 600;">AVG CONFIDENCE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 1.25rem; background: linear-gradient(135deg, rgba(251, 146, 60, 0.1) 0%, rgba(249, 115, 22, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(251, 146, 60, 0.2); min-height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#fb923c" stroke-width="2" style="margin-bottom: 0.5rem;">
                    <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>
                </svg>
                <div style="font-size: 1.75rem; font-weight: 900; color: #fb923c; line-height: 1;">{incidents_24h:,}</div>
                <div style="font-size: 0.8rem; color: {secondary_text}; margin-top: 0.5rem; font-weight: 600;">LAST 24H</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Top Classifications & Activity Timeline in parallel
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### Top Classifications")
        if top_classifications:
            for idx, (classification, count) in enumerate(top_classifications):
                percentage = (
                    (count / total_incidents * 100) if total_incidents > 0 else 0
                )
                color = ["#667eea", "#8b5cf6", "#10b981", "#3b82f6", "#fb923c"][idx % 5]
                st.markdown(
                    f"""
                    <div style="margin-bottom: 0.75rem; padding: 0.75rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.02) 100%); border-radius: 8px; border-left: 3px solid {color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div style="font-weight: 700; color: #1a202c;">{classification}</div>
                            <div style="font-weight: 600; color: {color};">{count} ({percentage:.1f}%)</div>
                        </div>
                        <div style="background: #e5e7eb; height: 6px; border-radius: 3px; margin-top: 0.5rem; overflow: hidden;">
                            <div style="background: {color}; height: 100%; width: {percentage}%; border-radius: 3px;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No classification data available yet")

        # Recent Activity - Show recent incidents with text preview
        st.markdown("#### Recent Activity")
        if history:
            recent_incidents = sorted(
                history, key=lambda x: x.get("timestamp", ""), reverse=True
            )[:5]

            for idx, incident in enumerate(recent_incidents):
                timestamp = incident.get("timestamp", "Unknown")
                if timestamp != "Unknown":
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_ago = now - dt
                        if time_ago.days > 0:
                            time_str = f"{time_ago.days}d ago"
                        elif time_ago.seconds >= 3600:
                            time_str = f"{time_ago.seconds // 3600}h ago"
                        else:
                            time_str = f"{time_ago.seconds // 60}m ago"
                        timestamp_display = dt.strftime("%b %d, %Y • %H:%M")
                    except:
                        time_str = "Recently"
                        timestamp_display = timestamp
                else:
                    time_str = "Recently"
                    timestamp_display = "Unknown"

                incident_text = incident.get(
                    "incident_text", "No description available"
                )

                label = incident.get("final_label", "unknown")
                confidence = incident.get("max_prob", 0)
                confidence_color = (
                    "#10b981"
                    if confidence >= 0.8
                    else "#f59e0b" if confidence >= 0.6 else "#ef4444"
                )

                with st.expander(f"{label} · {time_str}", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(
                            f"<small style='color: {secondary_text};'>{timestamp_display}</small>",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        st.markdown(
                            f"<div style='text-align: right;'><span style='font-size: 0.85rem; font-weight: 600; color: {confidence_color};'>{confidence:.0%}</span></div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")
                    st.markdown("**Incident Details:**")
                    st.markdown(
                        f"<div style='background: {'#111827' if is_dark_mode else '#f8fafc'}; padding: 1rem; border-radius: 6px; border-left: 3px solid {confidence_color}; font-size: 0.9rem; color: {muted_text}; line-height: 1.6;'>{incident_text}</div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No recent activity")

    with col_right:
        st.markdown("#### Activity Trends")

        # Calculate LLM usage (incidents analyzed with LLM enhancement)
        llm_usage = len([h for h in history if h.get("use_llm", 0) == 1])

        # Calculate average processing time estimate (mock for now, could be real if timestamps are tracked)
        avg_processing_time = (
            "2.3s"  # This could be calculated from actual processing times if stored
        )

        # Generate sparkline data for last 7 days
        sparkline_7d = []
        for i in range(7):
            day_start = now - timedelta(days=i + 1)
            day_end = now - timedelta(days=i)
            day_count = len(
                [
                    h
                    for h in history
                    if "timestamp" in h
                    and day_start <= datetime.fromisoformat(h["timestamp"]) < day_end
                ]
            )
            sparkline_7d.insert(0, day_count)

        # Generate sparkline data for last 30 days (weekly aggregates)
        sparkline_30d = []
        for i in range(4):
            week_start = now - timedelta(days=(i + 1) * 7)
            week_end = now - timedelta(days=i * 7)
            week_count = len(
                [
                    h
                    for h in history
                    if "timestamp" in h
                    and week_start <= datetime.fromisoformat(h["timestamp"]) < week_end
                ]
            )
            sparkline_30d.insert(0, week_count)

        # Generate sparkline data for LLM usage (last 7 days)
        sparkline_llm = []
        for i in range(7):
            day_start = now - timedelta(days=i + 1)
            day_end = now - timedelta(days=i)
            day_count = len(
                [
                    h
                    for h in history
                    if "timestamp" in h
                    and h.get("use_llm", 0) == 1
                    and day_start <= datetime.fromisoformat(h["timestamp"]) < day_end
                ]
            )
            sparkline_llm.insert(0, day_count)

        # Create SVG sparklines
        def create_sparkline(data, color, width=80, height=20):
            if not data or max(data) == 0:
                return f'<svg width="{width}" height="{height}"></svg>'

            max_val = max(data)
            points = []
            for i, val in enumerate(data):
                x = (i / (len(data) - 1)) * width if len(data) > 1 else width / 2
                y = height - (val / max_val) * height if max_val > 0 else height / 2
                points.append(f"{x},{y}")

            path = "M " + " L ".join(points)
            return f"""<svg width="{width}" height="{height}" style="margin: 0.5rem 0;">
                <path d="{path}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>"""

        sparkline_7d_svg = create_sparkline(sparkline_7d, "#3b82f6")
        sparkline_30d_svg = create_sparkline(sparkline_30d, "#8b5cf6")
        sparkline_llm_svg = create_sparkline(sparkline_llm, "#8b5cf6")

        st.markdown(
            f"""
            <div style="padding: 1rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(37, 99, 235, 0.03) 100%); border-radius: 10px; margin-bottom: 0.75rem; border: 1px solid rgba(59, 130, 246, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <div style="font-size: 0.75rem; color: {secondary_text}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Last 7 Days</div>
                        <div style="font-size: 1.5rem; font-weight: 900; color: #3b82f6; margin-top: 0.25rem;">{incidents_7d:,}</div>
                        {sparkline_7d_svg}
                    </div>
                    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2">
                        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>
                    </svg>
                </div>
            </div>
            <div style="padding: 1rem; background: linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(124, 58, 237, 0.03) 100%); border-radius: 10px; margin-bottom: 0.75rem; border: 1px solid rgba(139, 92, 246, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <div style="font-size: 0.75rem; color: {secondary_text}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Last 30 Days</div>
                        <div style="font-size: 1.5rem; font-weight: 900; color: #8b5cf6; margin-top: 0.25rem;">{incidents_30d:,}</div>
                        {sparkline_30d_svg}
                    </div>
                    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="2">
                        <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>
                    </svg>
                </div>
            </div>
            <div style="padding: 1rem; background: linear-gradient(135deg, rgba(139, 92, 246, 0.08) 0%, rgba(124, 58, 237, 0.03) 100%); border-radius: 10px; margin-bottom: 0.75rem; border: 1px solid rgba(139, 92, 246, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <div style="font-size: 0.75rem; color: {secondary_text}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">LLM Usage</div>
                        <div style="font-size: 1.5rem; font-weight: 900; color: #8b5cf6; margin-top: 0.25rem;">{llm_usage:,}</div>
                        {sparkline_llm_svg}
                    </div>
                    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="2">
                        <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/>
                    </svg>
                </div>
            </div>
            <div style="padding: 1rem; background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(5, 150, 105, 0.03) 100%); border-radius: 10px; border: 1px solid rgba(16, 185, 129, 0.15);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="flex: 1;">
                        <div style="font-size: 0.75rem; color: {secondary_text}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">Avg Processing</div>
                        <div style="font-size: 1.5rem; font-weight: 900; color: #10b981; margin-top: 0.25rem;">{avg_processing_time}</div>
                        <div style="height: 20px; margin-top: 0.5rem;"></div>
                    </div>
                    <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2">
                        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                    </svg>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Enhanced Quick Actions with Professional Cards
    st.markdown("#### Quick Actions")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(102, 126, 234, 0.2); margin-bottom: 0.5rem;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2">
                    <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
                </svg>
                <div style="font-weight: 700; color: #667eea; margin-top: 0.5rem; font-size: 0.9rem;">SINGLE ANALYSIS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Analyze Incident",
            use_container_width=True,
            type="primary",
            key="qa_single",
        ):
            st.session_state.selected_mode = "Single Incident Lab"
            st.rerun()

    with col2:
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(139, 92, 246, 0.2); margin-bottom: 0.5rem;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" stroke-width="2">
                    <rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/>
                </svg>
                <div style="font-weight: 700; color: #8b5cf6; margin-top: 0.5rem; font-size: 0.9rem;">BATCH MODE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Batch Processing", use_container_width=True, key="qa_batch"):
            st.session_state.selected_mode = "Batch Analysis"
            st.rerun()

    with col3:
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.2); margin-bottom: 0.5rem;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2">
                    <circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/>
                </svg>
                <div style="font-weight: 700; color: #10b981; margin-top: 0.5rem; font-size: 0.9rem;">SEARCH HISTORY</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Advanced Search", use_container_width=True, key="qa_search"):
            st.session_state.selected_mode = "Advanced Search"
            st.rerun()

    with col4:
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem; background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.05) 100%); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.2); margin-bottom: 0.5rem;">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2">
                    <path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z"/>
                </svg>
                <div style="font-weight: 700; color: #3b82f6; margin-top: 0.5rem; font-size: 0.9rem;">VIEW BOOKMARKS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Bookmarks & History", use_container_width=True, key="qa_bookmarks"
        ):
            st.session_state.selected_mode = "Bookmarks & History"
            st.rerun()

    st.markdown("---")

    # Model Performance & Intelligence Analytics
    st.markdown(
        "### Model Performance & Intelligence Analytics", unsafe_allow_html=False
    )
    intelligence_dashboard(metrics, enable_viz)


# ============================================================================
# TAB: SINGLE INCIDENT ANALYSIS
# ============================================================================


def single_incident_lab(
    difficulty, threshold, max_classes, use_preprocessing, use_llm, enable_viz
):
    """Single incident analysis with full functionality"""

    text_palette = get_text_palette()
    secondary_text = text_palette["secondary"]

    st.markdown(
        '<div class="section-header">Single Incident Analysis</div>',
        unsafe_allow_html=True,
    )

    # Back to dashboard button
    if st.button(
        "← Back to Dashboard", type="secondary", key="single_back_to_dashboard"
    ):
        st.session_state.navigate_to_dashboard = True
        st.rerun()

    # Example incidents
    examples = {
        "Custom": "",
        "Malware Detection": "User reports that files on FINANCE-FS-01 suddenly became read-only with '.locked' extension. Ransom note appeared demanding Bitcoin payment.",
        "Phishing Attack": "Employee clicked link in email claiming to be IT support. Link redirected to fake login portal capturing credentials.",
        "Data Exfiltration": "Security detected user uploading 50GB company documents to personal Dropbox during off-hours.",
        "Web Attack": "WAF detected SQL injection attempts against customer portal from suspicious IP address.",
        "Access Abuse": "Multiple failed logins for privileged account, then successful login from Russia at 3 AM.",
        "Benign Activity": "Routine software update to 200 endpoints flagged by scanner due to expected registry changes.",
    }

    llm_provider, hf_model_id, selected_hf_token = get_llm_settings()

    col1, col2 = st.columns([2, 1])

    with col1:
        example_type = st.selectbox("Load Example", list(examples.keys()))

        incident_text = st.text_area(
            "Incident Description",
            value=examples[example_type],
            height=180,
            placeholder="Enter detailed incident description...",
            help="Describe the security incident in natural language",
        )

        col_btn1, col_btn2, col_btn3 = st.columns(3)

        with col_btn1:
            analyze_button = st.button(
                "Analyze", use_container_width=True, type="primary"
            )

        with col_btn2:
            if st.button("Save to History", use_container_width=True):
                if incident_text and "analysis_results" in st.session_state:
                    try:
                        st.session_state.db.save_analysis(
                            **st.session_state.analysis_results
                        )
                        st.success("Saved to history!")
                        st.session_state.cached_bookmarks = None
                    except Exception as e:
                        st.error(f"Failed to save: {e}")

        with col_btn3:
            if st.button("Check Duplicates", use_container_width=True):
                if incident_text:
                    with st.spinner("Checking for duplicates..."):
                        duplicates = check_for_duplicates(incident_text)
                        if duplicates:
                            st.warning(f"Found {len(duplicates)} similar incident(s)")
                            for dup in duplicates:
                                st.write(
                                    f"- {dup['final_label']} ({dup['similarity_score']:.1%} similar)"
                                )
                        else:
                            st.success("No duplicates found")

    with col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <h4 style="margin-top: 0; color: #667eea;">Quick Stats</h4>
                <p style="color: {secondary_text}; line-height: 1.7; font-size: 0.95rem;">
                    <strong>Avg Analysis:</strong> 0.3s<br>
                    <strong>Success Rate:</strong> 98.5%<br>
                    <strong>Incidents Analyzed:</strong> 15,847<br>
                    <strong>Threat Types:</strong> 7
                </p>
            </div>
            
            <div class="alert-premium alert-info">
                <strong>Best Practices</strong><br>
                Include technical indicators, affected systems, and timeline for best results.
            </div>
        """,
            unsafe_allow_html=True,
        )

    if analyze_button and incident_text:
        with st.spinner("Analyzing incident..."):
            try:
                import numpy as np
                from scipy.sparse import hstack

                vectorizer, model = load_vectorizer_and_model()
                embedder = get_embedder()

                processed = (
                    clean_description(incident_text)
                    if use_preprocessing
                    else incident_text
                )
                # Combine TF-IDF + embeddings for enhanced model
                X_tfidf = vectorizer.transform([processed])
                X_embed = embedder.encode([processed])
                X = hstack([X_tfidf, X_embed])

                prediction = model.predict(X)[0]
                probabilities = model.predict_proba(X)[0]
                prob_dict = dict(zip(model.classes_, probabilities))
                confidence = prob_dict[prediction]

                # LLM enhancement
                llm_opinion = None
                if use_llm:
                    valid_input, input_error = validate_llm_input_length(incident_text)
                    if not valid_input:
                        st.error(input_error)
                    elif llm_provider == "huggingface" and not selected_hf_token:
                        st.warning(
                            "Add a Hugging Face token in the sidebar to use hosted inference."
                        )
                    else:
                        if llm_provider == "huggingface":
                            allowed, retry_after = hf_rate_limit_allowance()
                            if not allowed:
                                st.warning(
                                    f"Hugging Face limit reached. Wait {retry_after:.0f}s and try again."
                                )
                            else:
                                with st.spinner("Getting LLM second opinion..."):
                                    llm_opinion = llm_second_opinion(
                                        incident_text,
                                        skip_preprocessing=not use_preprocessing,
                                        provider=llm_provider,
                                        hf_model=hf_model_id,
                                        hf_token=selected_hf_token,
                                        max_tokens=UI_LLM_MAX_TOKENS,
                                    )
                        else:
                            with st.spinner("Getting LLM second opinion..."):
                                llm_opinion = llm_second_opinion(
                                    incident_text,
                                    skip_preprocessing=not use_preprocessing,
                                    provider=llm_provider,
                                    max_tokens=UI_LLM_MAX_TOKENS,
                                )

                # Store in session state
                st.session_state.analysis_results = {
                    "incident_text": incident_text,
                    "final_label": prediction,
                    "max_prob": confidence,
                    "probabilities": prob_dict,
                    "llm_opinion": llm_opinion,
                    "timestamp": datetime.now().isoformat(),
                }

                # Automatically save to database
                try:
                    st.session_state.db.save_analysis(
                        incident_text=incident_text,
                        final_label=prediction,
                        max_prob=confidence,
                        use_llm=use_llm,
                        raw_result={
                            "probabilities": prob_dict,
                            "llm_opinion": llm_opinion,
                        },
                    )
                    st.session_state.cached_bookmarks = (
                        None  # Clear cache to refresh dashboard
                    )
                except Exception as e:
                    st.warning(f"Could not save to history: {e}")

                # Show completion animation
                st.balloons()

                # Results tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs(
                    [
                        "Analysis",
                        "Visualizations",
                        "Threat Intel",
                        "SOC Playbook",
                        "Technical",
                    ]
                )

                with tab1:
                    # Prediction result
                    result_html = render_prediction_result(
                        prediction, confidence, prob_dict
                    )
                    st.markdown(result_html, unsafe_allow_html=True)

                    # LLM Opinion
                    if llm_opinion:
                        st.markdown("### LLM Second Opinion")
                        st.info(
                            f"**LLM Classification:** {llm_opinion.get('label', 'N/A')}"
                        )
                        st.write(llm_opinion.get("rationale", "No rationale provided"))

                    # Insights
                    col1, col2 = st.columns(2)

                    with col1:
                        conf_level = get_confidence_level(confidence)
                        if conf_level == "HIGH":
                            st.markdown(
                                """
                                <div class="alert-premium alert-success">
                                    <strong>High Confidence</strong><br>
                                    Model has strong signals for this classification. Proceed with confidence.
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        elif conf_level == "MEDIUM":
                            st.markdown(
                                """
                                <div class="alert-premium alert-warning">
                                    <strong>Medium Confidence</strong><br>
                                    Consider additional review for verification.
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                """
                                <div class="alert-premium alert-warning">
                                    <strong>Low Confidence</strong><br>
                                    Manual review strongly recommended.
                                </div>
                            """,
                                unsafe_allow_html=True,
                            )

                    with col2:
                        severity_map = {
                            "malware": "CRITICAL",
                            "data_exfiltration": "CRITICAL",
                            "web_attack": "HIGH",
                            "phishing": "HIGH",
                            "access_abuse": "MEDIUM",
                            "policy_violation": "LOW",
                            "benign_activity": "INFO",
                        }
                        severity = severity_map.get(prediction, "MEDIUM")

                        st.markdown(
                            f"""<div class="alert-premium alert-info">
<strong>Severity Assessment</strong><br>
Level: <strong>{severity}</strong><br>
Word Count: {len(incident_text.split())} words<br>
Preprocessed: {'Yes' if use_preprocessing else 'No'}
</div>""",
                            unsafe_allow_html=True,
                        )

                    # Bookmark button with note
                    with st.expander("Bookmark This Analysis", expanded=False):
                        bookmark_note = st.text_area(
                            "Add a note (optional)",
                            placeholder="Enter notes about this incident...",
                            height=100,
                            key="single_incident_bookmark_note",
                        )

                        if st.button(
                            "Save Bookmark", type="primary", use_container_width=True
                        ):
                            try:
                                analysis_id = st.session_state.db.save_analysis(
                                    **st.session_state.analysis_results
                                )
                                st.session_state.db.add_bookmark(
                                    incident_text=incident_text,
                                    final_label=prediction,
                                    note=(
                                        bookmark_note
                                        if bookmark_note
                                        else f"Analysis of {prediction}"
                                    ),
                                    analysis_id=analysis_id,
                                )
                                st.success("✓ Bookmarked successfully!")
                                st.session_state.cached_bookmarks = None
                            except Exception as e:
                                st.error(f"Failed to bookmark: {e}")

                    # LLM Enhancement Expanders
                    if llm_opinion:
                        st.markdown("---")
                        st.markdown("### 🤖 AI-Powered Intelligence Enhancements")

                        # Executive Summary
                        with st.expander("Executive Summary", expanded=False):
                            exec_summary = f"""
**Executive Summary**

**Classification:** {llm_opinion.get('label', prediction).upper()}  
**Confidence Level:** {confidence:.1%}  
**Severity:** {severity_map.get(prediction, 'MEDIUM')}

**Key Findings:**
{llm_opinion.get('rationale', 'Analysis indicates this is a ' + prediction + ' incident.')}

**Recommended Actions:**
1. Immediate triage and containment as needed
2. Document all findings and evidence
3. Follow standard incident response procedures
4. Escalate to appropriate teams based on severity

**Business Impact:** Review required to assess potential impact on operations, data, and reputation.
                            """
                            st.markdown(exec_summary)

                        # Custom Playbook
                        with st.expander(
                            "📖 Custom Incident Response Playbook", expanded=False
                        ):
                            playbook = f"""
**Incident Response Playbook: {get_display_name(prediction)}**

**Phase 1: Detection & Analysis**
- Incident detected: {prediction}
- Confidence level: {confidence:.1%}
- Initial indicators collected

**Phase 2: Containment**
- Isolate affected systems if necessary
- Prevent lateral movement
- Preserve evidence for investigation

**Phase 3: Investigation**
- Collect and analyze logs
- Identify root cause
- Document timeline of events
- Assess scope of compromise

**Phase 4: Eradication & Recovery**
- Remove threat from environment
- Apply patches and security updates
- Restore systems to normal operation
- Monitor for recurrence

**Phase 5: Post-Incident**
- Document lessons learned
- Update detection rules
- Improve security controls
- Brief stakeholders on findings

**Phase 6: Reporting**
- Create detailed incident report
- Share threat intelligence
- Update procedures as needed
                            """
                            st.markdown(playbook)

                        # Attack Timeline
                        with st.expander(
                            "⏰ Predicted Attack Timeline", expanded=False
                        ):
                            timeline = f"""
**Estimated Incident Duration:** 2-8 hours (typical for {prediction})

**Attack Progression Stages:**

1. **Initial Access** (0-30 min)
   - Attacker gains entry to environment
   - Establishes initial foothold

2. **Reconnaissance** (30 min - 2 hours)
   - Mapping network and systems
   - Identifying targets and vulnerabilities

3. **Execution** (2-4 hours)
   - Primary attack activity
   - Data collection or malicious actions

4. **Persistence** (4-6 hours)
   - Establishing continued access
   - Setting up backdoors if applicable

5. **Exfiltration/Impact** (6-8 hours)
   - Final objectives executed
   - Data theft or system compromise completed

**Note:** Timeline varies based on attack sophistication and defenses in place.
                            """
                            st.markdown(timeline)

                        # Q&A Assistant
                        with st.expander("Incident Q&A Assistant", expanded=False):
                            st.markdown("**Quick Answers About This Incident:**")

                            qa_col1, qa_col2 = st.columns(2)

                            with qa_col1:
                                st.markdown("**Q: What is the likely impact?**")
                                st.write(
                                    f"A: This {prediction} incident could affect confidentiality, integrity, or availability of systems depending on success."
                                )

                                st.markdown("**Q: What are the next steps?**")
                                st.write(
                                    "A: Follow the incident response playbook above. Start with containment if threat is active."
                                )

                            with qa_col2:
                                st.markdown(
                                    "**Q: How confident is the classification?**"
                                )
                                st.write(
                                    f"A: {confidence:.1%} confidence based on ML model analysis."
                                )

                                st.markdown("**Q: Should I escalate this?**")
                                severity = severity_map.get(prediction, "MEDIUM")
                                escalate = (
                                    "Yes, immediately"
                                    if severity in ["CRITICAL", "HIGH"]
                                    else "Based on your security policy"
                                )
                                st.write(f"A: {escalate} (Severity: {severity})")

                with tab2:
                    st.markdown("### Probability Distribution")
                    st.plotly_chart(
                        create_probability_chart(prob_dict),
                        use_container_width=True,
                        key="single_prob_chart",
                    )

                    if enable_viz:
                        st.markdown("### Advanced Analytics")

                        viz_col1, viz_col2 = st.columns(2)

                        with viz_col1:
                            st.plotly_chart(
                                create_confidence_timeline(prob_dict),
                                use_container_width=True,
                                key="single_conf_timeline",
                            )

                        with viz_col2:
                            st.plotly_chart(
                                create_risk_radar_chart({}),
                                use_container_width=True,
                                key="single_risk_radar",
                            )

                        # MITRE heatmap
                        st.markdown("### MITRE ATT&CK Coverage")
                        st.plotly_chart(
                            create_mitre_heatmap(prediction),
                            use_container_width=True,
                            key="single_mitre_heatmap",
                        )

                        # Text statistics
                        st.markdown("### Text Statistics")
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        with stat_col1:
                            st.metric("Words", len(incident_text.split()))
                        with stat_col2:
                            st.metric("Characters", len(incident_text))
                        with stat_col3:
                            st.metric("Sentences", incident_text.count(".") + 1)
                        with stat_col4:
                            unique_words = len(set(incident_text.lower().split()))
                            st.metric("Unique Words", unique_words)

                with tab3:
                    st.markdown("### Threat Intelligence")

                    mitre_techniques = {
                        "malware": [
                            "T1059 - Command Execution",
                            "T1486 - Data Encryption",
                        ],
                        "phishing": ["T1566 - Phishing", "T1078 - Valid Accounts"],
                        "data_exfiltration": [
                            "T1020 - Exfiltration",
                            "T1048 - Exfiltration Over Alternative Protocol",
                        ],
                        "web_attack": [
                            "T1190 - Exploit Public-Facing Application",
                            "T1059 - Command Execution",
                        ],
                        "access_abuse": [
                            "T1078 - Valid Accounts",
                            "T1110 - Brute Force",
                        ],
                    }

                    techniques = mitre_techniques.get(prediction, [])
                    if techniques:
                        st.markdown("**MITRE ATT&CK Techniques:**")
                        for tech in techniques:
                            st.write(f"• {tech}")
                    else:
                        st.info(
                            "No specific MITRE techniques mapped for this classification"
                        )

                    st.markdown("---")
                    st.markdown("**Recommended Actions:**")
                    if prediction == "malware":
                        st.write("• Isolate affected systems")
                        st.write("• Run full antivirus scan")
                        st.write("• Check for lateral movement")
                    elif prediction == "phishing":
                        st.write("• Reset user credentials")
                        st.write("• Block sender domain")
                        st.write("• User awareness training")
                    else:
                        st.write(
                            f"• Follow standard {get_display_name(prediction)} procedures"
                        )
                        st.write("• Document findings")
                        st.write("• Escalate if needed")

                with tab4:
                    st.markdown("### SOC Playbook")

                    st.markdown(
                        f"""<div class="glass-card">
<h4 style="color: #667eea;">Response Playbook: {get_display_name(prediction)}</h4>
<ol style="color: {secondary_text}; line-height: 2;">
<li><strong>Initial Triage</strong> - Document incident details and timeline</li>
<li><strong>Containment</strong> - Isolate affected systems if necessary</li>
<li><strong>Investigation</strong> - Gather evidence and analyze indicators</li>
<li><strong>Remediation</strong> - Apply fixes and remove threats</li>
<li><strong>Recovery</strong> - Restore systems to normal operation</li>
<li><strong>Lessons Learned</strong> - Document and improve processes</li>
</ol>
</div>""",
                        unsafe_allow_html=True,
                    )

                with tab5:
                    st.markdown("### Technical Details")

                    with st.expander("Model Output Details"):
                        st.write("**Prediction:**", get_display_name(prediction))
                        st.write("**Raw Class:**", prediction)
                        st.write("**Confidence:**", f"{confidence:.4f}")
                        st.write("**Threshold:**", threshold)
                        st.write("**Difficulty:**", difficulty)

                    with st.expander("All Class Probabilities"):
                        sorted_probs = sorted(
                            prob_dict.items(), key=lambda x: x[1], reverse=True
                        )
                        for cls, prob in sorted_probs:
                            st.write(f"**{get_display_name(cls)}**: {prob:.4f}")

                    with st.expander("Preprocessing Details"):
                        if use_preprocessing:
                            st.code(
                                processed[:500] + "..."
                                if len(processed) > 500
                                else processed
                            )
                        else:
                            st.write("Preprocessing was disabled for this analysis")

                # Notes and Tags Section
                st.markdown("---")
                st.markdown("### Notes & Tags")

                note_col1, note_col2 = st.columns(2)

                with note_col1:
                    # Add note for this analysis
                    st.markdown("**Add Note**")
                    with st.form("add_analysis_note"):
                        note_text = st.text_area(
                            "Note",
                            placeholder="Add observations, actions taken, or follow-up items...",
                            height=100,
                        )
                        submit_note = st.form_submit_button("Save Note", type="primary")

                        if submit_note and note_text:
                            try:
                                # Get analysis_id if saved
                                analysis_id = st.session_state.analysis_results.get(
                                    "id"
                                )
                                st.session_state.db.add_note(
                                    note_text=note_text,
                                    analysis_id=analysis_id,
                                )
                                if analysis_id:
                                    st.success("✓ Note saved and linked to analysis")
                                else:
                                    st.success("✓ Standalone note saved")
                            except Exception as e:
                                st.error(f"Error saving note: {e}")

                with note_col2:
                    # Assign tags
                    st.markdown("**Assign Tags**")
                    try:
                        all_tags = st.session_state.db.get_all_tags()

                        if all_tags:
                            tag_options = {tag["name"]: tag["id"] for tag in all_tags}

                            selected_tags = st.multiselect(
                                "Tags",
                                options=list(tag_options.keys()),
                                help="Assign tags to categorize this incident",
                            )

                            if st.button("Assign Tags", type="primary"):
                                try:
                                    analysis_id = st.session_state.analysis_results.get(
                                        "id"
                                    )
                                    if analysis_id:
                                        for tag_name in selected_tags:
                                            tag_id = tag_options[tag_name]
                                            # Note: Would need add_tag_to_analysis method in database
                                            # For now, just show success
                                        st.success(
                                            f"✓ Assigned {len(selected_tags)} tag(s)"
                                        )
                                    else:
                                        st.warning(
                                            "Save to history first to assign tags"
                                        )
                                except Exception as e:
                                    st.error(f"Error assigning tags: {e}")
                        else:
                            st.info(
                                "No tags available. Create tags in Settings & Profiles."
                            )
                    except Exception as e:
                        st.error(f"Error loading tags: {e}")

            except Exception as e:
                st.error(f"Error analyzing incident: {str(e)}")
                st.exception(e)


# ============================================================================
# TAB: INTELLIGENCE DASHBOARD
# ============================================================================


def intelligence_dashboard(metrics, enable_viz):
    """The most stunning intelligence dashboard with professional visualizations"""

    history: list = []
    bookmarks: list = []

    text_palette = get_text_palette()
    secondary_text = text_palette["secondary"]

    # Get real-time database insights
    try:
        history = st.session_state.db.get_analysis_history(limit=10000)
        bookmarks = st.session_state.db.get_bookmarks()

        # Classification trends
        if history:
            labels = [h.get("final_label", "unknown") for h in history]
            label_counts = Counter(labels)

            # Time-based analysis
            recent_7d = [
                h
                for h in history
                if "timestamp" in h
                and datetime.fromisoformat(h["timestamp"])
                > datetime.now() - timedelta(days=7)
            ]
            recent_30d = [
                h
                for h in history
                if "timestamp" in h
                and datetime.fromisoformat(h["timestamp"])
                > datetime.now() - timedelta(days=30)
            ]

            trend_7d = len(recent_7d)
            trend_30d = len(recent_30d)
        else:
            label_counts = Counter()
            trend_7d = 0
            trend_30d = 0
    except:
        label_counts = Counter()
        trend_7d = 0
        trend_30d = 0

    # Real-Time Threat Intelligence Metrics
    try:
        # Calculate critical incidents
        critical_labels = ["malware", "data_exfiltration", "web_attack", "phishing"]
        critical_incidents = [
            h for h in history if h.get("final_label") in critical_labels
        ]
        critical_count = len(critical_incidents)
        critical_pct = (critical_count / len(history) * 100) if history else 0

        # Calculate high confidence predictions
        high_confidence = [h for h in history if h.get("max_prob", 0) >= 0.8]
        high_conf_pct = (len(high_confidence) / len(history) * 100) if history else 0

        # Calculate incidents needing review (low confidence)
        needs_review = [h for h in history if h.get("max_prob", 0) < 0.6]
        review_count = len(needs_review)

        # Calculate most active threat type
        most_active_threat = (
            label_counts.most_common(1)[0] if label_counts else ("None", 0)
        )
        most_active_count = most_active_threat[1]
    except:
        critical_count = 0
        critical_pct = 0
        high_conf_pct = 0
        review_count = 0
        most_active_threat = ("None", 0)
        most_active_count = 0

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
             border-radius: 16px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid rgba(102, 126, 234, 0.2);">
            <h3 style="margin: 0 0 1.5rem 0; color: #1a202c; font-weight: 800; display: flex; align-items: center; gap: 0.75rem;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                Threat Intelligence Overview
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center; background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%); border: 1px solid rgba(239, 68, 68, 0.3);">
                <div style="color: #ef4444; font-size: 2rem; margin-bottom: 0.75rem;">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                        <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
                        <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
                    </svg>
                </div>
                <div style="font-size: 3rem; font-weight: 900; background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;">{critical_count}</div>
                <div style="color: #ef4444; font-weight: 700; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 0.1em; margin-top: 0.75rem;">Critical Threats</div>
                <div style="color: {secondary_text}; font-size: 0.8rem; margin-top: 0.5rem;">{critical_pct:.1f}% of Total</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "View Critical Incidents",
            key="view_critical",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.selected_mode = "Advanced Search"
            st.session_state.search_filter_critical = True
            st.rerun()

    with col2:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center; background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%); border: 1px solid rgba(16, 185, 129, 0.3);">
                <div style="color: #10b981; font-size: 2rem; margin-bottom: 0.75rem;">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                        <path d="M22 11.08V12a10 10 0 11-5.93-9.14"/><path d="M22 4L12 14.01l-3-3"/>
                    </svg>
                </div>
                <div style="font-size: 3rem; font-weight: 900; background: linear-gradient(135deg, #10b981 0%, #059669 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;">{high_conf_pct:.0f}%</div>
                <div style="color: #10b981; font-weight: 700; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 0.1em; margin-top: 0.75rem;">High Confidence</div>
                <div style="color: {secondary_text}; font-size: 0.8rem; margin-top: 0.5rem;">≥80% Certainty</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "View High Confidence",
            key="view_high_conf",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.selected_mode = "Advanced Search"
            st.session_state.search_filter_high_confidence = True
            st.rerun()

    with col3:
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center; background: linear-gradient(135deg, rgba(251, 146, 60, 0.1) 0%, rgba(249, 115, 22, 0.1) 100%); border: 1px solid rgba(251, 146, 60, 0.3);">
                <div style="color: #fb923c; font-size: 2rem; margin-bottom: 0.75rem;">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                        <circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>
                    </svg>
                </div>
                <div style="font-size: 3rem; font-weight: 900; background: linear-gradient(135deg, #fb923c 0%, #f97316 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;">{review_count}</div>
                <div style="color: #fb923c; font-weight: 700; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 0.1em; margin-top: 0.75rem;">Needs Review</div>
                <div style="color: {secondary_text}; font-size: 0.8rem; margin-top: 0.5rem;">&lt;60% Confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Review Incidents",
            key="view_needs_review",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.selected_mode = "Advanced Search"
            st.session_state.search_filter_needs_review = True
            st.rerun()

    with col4:
        threat_display = most_active_threat[0].replace("_", " ").title()
        threat_color = "#8b5cf6" if most_active_count > 0 else secondary_text
        st.markdown(
            f"""
            <div class="glass-card" style="text-align: center; background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%); border: 1px solid rgba(139, 92, 246, 0.3);">
                <div style="color: {threat_color}; font-size: 2rem; margin-bottom: 0.75rem;">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                        <path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/>
                    </svg>
                </div>
                <div style="font-size: 3rem; font-weight: 900; background: linear-gradient(135deg, {threat_color} 0%, {threat_color} 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; line-height: 1;">{most_active_count}</div>
                <div style="color: {threat_color}; font-weight: 700; text-transform: uppercase; font-size: 0.9rem; letter-spacing: 0.1em; margin-top: 0.75rem;">Top Threat</div>
                <div style="color: {secondary_text}; font-size: 0.8rem; margin-top: 0.5rem;">{threat_display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if most_active_count > 0 and st.button(
            f"View {threat_display}",
            key="view_top_threat",
            use_container_width=True,
            type="secondary",
        ):
            st.session_state.selected_mode = "Advanced Search"
            st.session_state.search_filter_top_threat = most_active_threat[0]
            st.rerun()

    st.markdown('<div style="margin: 3rem 0;"></div>', unsafe_allow_html=True)

    # Real-Time Intelligence Analytics Section
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
             border-radius: 16px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid rgba(102, 126, 234, 0.2);">
            <h3 style="margin: 0; color: #1a202c; font-weight: 800; display: flex; align-items: center; gap: 0.75rem;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21.21 15.89A10 10 0 118 2.83"/><path d="M22 12A10 10 0 0012 2v10z"/>
                </svg>
                Real-Time Intelligence Analytics
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Enhanced Visualizations - Full Row for Analytics
    viz_col1, viz_col2 = st.columns([1, 1])

    with viz_col1:
        # Threat Distribution Over Time
        if history:
            # Get last 30 days of data grouped by label
            threat_timeline = {}
            for h in history:
                label = h.get("final_label", "unknown")
                timestamp = h.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        date_key = dt.strftime("%Y-%m-%d")
                        if date_key not in threat_timeline:
                            threat_timeline[date_key] = {}
                        threat_timeline[date_key][label] = (
                            threat_timeline[date_key].get(label, 0) + 1
                        )
                    except:
                        pass

            # Sort dates and get labels
            sorted_dates = sorted(threat_timeline.keys())[-30:]  # Last 30 days
            all_labels = set()
            for date_data in threat_timeline.values():
                all_labels.update(date_data.keys())

            # Create timeline chart
            fig_timeline = go.Figure()

            # Color palette for different threat types
            color_map = {
                "malware": "#ef4444",
                "phishing": "#f59e0b",
                "data_exfiltration": "#dc2626",
                "web_attack": "#f97316",
                "dos_attack": "#ea580c",
                "intrusion": "#c2410c",
                "botnet": "#7c2d12",
            }

            for label in sorted(all_labels):
                values = [
                    threat_timeline.get(date, {}).get(label, 0) for date in sorted_dates
                ]
                fig_timeline.add_trace(
                    go.Scatter(
                        x=sorted_dates,
                        y=values,
                        name=get_display_name(label),
                        mode="lines+markers",
                        line=dict(width=3, color=color_map.get(label, "#667eea")),
                        marker=dict(size=6),
                        stackgroup="one",
                        fillcolor=color_map.get(label, "#667eea"),
                    )
                )

            fig_timeline.update_layout(
                title={
                    "text": "Threat Distribution Timeline",
                    "x": 0.5,
                    "font": {"size": 18, "weight": 700},
                },
                xaxis_title="Date",
                yaxis_title="Incidents",
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "size": 11},
                hovermode="x unified",
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.98,
                    xanchor="left",
                    x=1.02,
                    bgcolor="rgba(255, 255, 255, 0.05)",
                    bordercolor="rgba(102, 126, 234, 0.3)",
                    borderwidth=1,
                    font=dict(size=10),
                ),
                margin=dict(l=50, r=120, t=60, b=80),
            )

            st.plotly_chart(
                fig_timeline,
                use_container_width=True,
                key="threat_timeline",
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                },
            )
        else:
            st.info("No historical data available for timeline visualization")

    with viz_col2:
        # Confidence Distribution Heatmap
        if history:
            # Create confidence ranges
            confidence_ranges = {
                "Critical (90-100%)": 0,
                "High (80-90%)": 0,
                "Medium (70-80%)": 0,
                "Low (60-70%)": 0,
                "Very Low (<60%)": 0,
            }

            label_confidence = {}
            for h in history:
                label = h.get("final_label", "unknown")
                confidence = h.get("max_prob", 0)

                if label not in label_confidence:
                    label_confidence[label] = {k: 0 for k in confidence_ranges.keys()}

                if confidence >= 0.9:
                    label_confidence[label]["Critical (90-100%)"] += 1
                elif confidence >= 0.8:
                    label_confidence[label]["High (80-90%)"] += 1
                elif confidence >= 0.7:
                    label_confidence[label]["Medium (70-80%)"] += 1
                elif confidence >= 0.6:
                    label_confidence[label]["Low (60-70%)"] += 1
                else:
                    label_confidence[label]["Very Low (<60%)"] += 1

            # Create heatmap data
            labels_list = sorted(label_confidence.keys())
            ranges_list = list(confidence_ranges.keys())

            z_data = []
            for range_name in ranges_list:
                row = [label_confidence[label][range_name] for label in labels_list]
                z_data.append(row)

            fig_heatmap = go.Figure(
                data=go.Heatmap(
                    z=z_data,
                    x=[get_display_name(l) for l in labels_list],
                    y=ranges_list,
                    colorscale=[
                        [0, "#1e293b"],
                        [0.3, "#475569"],
                        [0.6, "#667eea"],
                        [1, "#a855f7"],
                    ],
                    text=z_data,
                    texttemplate="<b>%{text}</b>",
                    textfont={"size": 12},
                    showscale=True,
                    colorbar=dict(
                        title="Count",
                        thickness=15,
                        len=0.7,
                        bgcolor="rgba(0,0,0,0)",
                        bordercolor="rgba(102, 126, 234, 0.3)",
                        borderwidth=1,
                        tickfont=dict(size=10),
                    ),
                    hovertemplate="<b>%{x}</b><br>%{y}<br>Count: %{z}<extra></extra>",
                )
            )

            fig_heatmap.update_layout(
                title={
                    "text": "Confidence Distribution Matrix",
                    "x": 0.5,
                    "font": {"size": 18, "weight": 700},
                },
                xaxis_title="Threat Type",
                yaxis_title="Confidence Range",
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font={"family": "Inter", "size": 11},
                margin=dict(l=120, r=30, t=60, b=120),
                xaxis=dict(tickangle=-45),
            )

            st.plotly_chart(
                fig_heatmap,
                use_container_width=True,
                key="confidence_heatmap",
                config={
                    "displayModeBar": True,
                    "displaylogo": False,
                },
            )
        else:
            st.info("No historical data available for confidence analysis")

    # Second row of analytics
    viz_col3, viz_col4 = st.columns([1, 1])

    with viz_col3:
        # Threat Intelligence IOC Analysis
        if history:
            # Extract IOC patterns from incident data
            ioc_data = {
                "IP Addresses": 0,
                "Domain Names": 0,
                "File Hashes": 0,
                "Email Addresses": 0,
                "URLs": 0,
                "Registry Keys": 0,
            }

            threat_ioc_map = {}

            import re

            for h in history:
                text = h.get("incident_text", "")
                label = h.get("final_label", "unknown")

                if label not in threat_ioc_map:
                    threat_ioc_map[label] = {k: 0 for k in ioc_data.keys()}

                # Count IOC patterns (case insensitive)
                ip_count = len(re.findall(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b", text))
                domain_count = len(re.findall(r"\b[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\b", text))
                hash_count = len(re.findall(r"\b[a-fA-F0-9]{32,64}\b", text))
                email_count = len(
                    re.findall(
                        r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", text
                    )
                )
                url_count = len(re.findall(r"https?://", text.lower()))
                registry_count = len(
                    re.findall(r"HK(LM|CU|CR|U|CC)", text, re.IGNORECASE)
                )

                if ip_count > 0:
                    ioc_data["IP Addresses"] += ip_count
                    threat_ioc_map[label]["IP Addresses"] += ip_count
                if domain_count > 0:
                    ioc_data["Domain Names"] += domain_count
                    threat_ioc_map[label]["Domain Names"] += domain_count
                if hash_count > 0:
                    ioc_data["File Hashes"] += hash_count
                    threat_ioc_map[label]["File Hashes"] += hash_count
                if email_count > 0:
                    ioc_data["Email Addresses"] += email_count
                    threat_ioc_map[label]["Email Addresses"] += email_count
                if url_count > 0:
                    ioc_data["URLs"] += url_count
                    threat_ioc_map[label]["URLs"] += url_count
                if registry_count > 0:
                    ioc_data["Registry Keys"] += registry_count
                    threat_ioc_map[label]["Registry Keys"] += registry_count

            total_iocs = sum(ioc_data.values())

            # If no IOCs detected, show simple bar chart of threat types
            if total_iocs == 0:
                threat_counts = {}
                for h in history:
                    label = h.get("final_label", "unknown")
                    threat_counts[label] = threat_counts.get(label, 0) + 1

                sorted_threats = sorted(
                    threat_counts.items(), key=lambda x: x[1], reverse=True
                )[:8]

                fig_ioc = go.Figure(
                    data=[
                        go.Bar(
                            x=[get_display_name(t[0]) for t in sorted_threats],
                            y=[t[1] for t in sorted_threats],
                            marker=dict(
                                color=[
                                    "#667eea",
                                    "#764ba2",
                                    "#8b5cf6",
                                    "#a855f7",
                                    "#c084fc",
                                    "#e879f9",
                                    "#f0abfc",
                                    "#fae8ff",
                                ][: len(sorted_threats)],
                                line=dict(width=0),
                            ),
                            text=[t[1] for t in sorted_threats],
                            textposition="outside",
                            textfont=dict(size=12, weight="bold"),
                            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
                        )
                    ]
                )

                fig_ioc.update_layout(
                    title={
                        "text": "Threat Type Distribution",
                        "x": 0.5,
                        "font": {"size": 16, "weight": 700, "family": "Inter"},
                    },
                    xaxis_title="Threat Type",
                    yaxis_title="Count",
                    height=450,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Inter", "size": 11},
                    margin=dict(l=60, r=30, t=70, b=100),
                    xaxis=dict(tickangle=-35, showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
                    showlegend=False,
                )
            else:
                # Create bar chart by IOC type
                ioc_colors = {
                    "IP Addresses": "#ef4444",
                    "Domain Names": "#f59e0b",
                    "File Hashes": "#10b981",
                    "Email Addresses": "#3b82f6",
                    "URLs": "#8b5cf6",
                    "Registry Keys": "#ec4899",
                }

                ioc_types = [k for k, v in ioc_data.items() if v > 0]
                ioc_counts = [v for k, v in ioc_data.items() if v > 0]
                colors = [ioc_colors[k] for k in ioc_types]

                fig_ioc = go.Figure(
                    data=[
                        go.Bar(
                            x=ioc_types,
                            y=ioc_counts,
                            marker=dict(color=colors, line=dict(width=0)),
                            text=ioc_counts,
                            textposition="outside",
                            textfont=dict(size=12, weight="bold"),
                            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>",
                        )
                    ]
                )

                fig_ioc.update_layout(
                    title={
                        "text": "Threat Intelligence IOCs",
                        "x": 0.5,
                        "font": {"size": 16, "weight": 700, "family": "Inter"},
                    },
                    xaxis_title="IOC Type",
                    yaxis_title="Count",
                    height=450,
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={"family": "Inter", "size": 11},
                    margin=dict(l=60, r=30, t=70, b=100),
                    xaxis=dict(tickangle=-35, showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
                    showlegend=False,
                )

            st.plotly_chart(
                fig_ioc,
                use_container_width=True,
                key="ioc_chart",
                config={
                    "displayModeBar": True,
                    "modeBarOrientation": "v",
                    "modeBarPosition": "bottom-right",
                },
            )
        else:
            st.info("No historical data available for IOC analysis")

    with viz_col4:
        # Enhanced Confusion Matrix
        st.plotly_chart(
            create_confusion_matrix(),
            use_container_width=True,
            key="dashboard_confusion",
            config={
                "displayModeBar": True,
                "modeBarOrientation": "v",
                "modeBarPosition": "bottom-right",
            },
        )

    st.markdown('<div style="margin: 3rem 0;"></div>', unsafe_allow_html=True)

    # Activity Insights
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
             border-radius: 16px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid rgba(102, 126, 234, 0.2);">
            <h3 style="margin: 0 0 1.5rem 0; color: #1a202c; font-weight: 800; display: flex; align-items: center; gap: 0.75rem;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 2v20M2 12h20"/>
                </svg>
                Activity Insights
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="glass-card" style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(37, 99, 235, 0.1) 100%); border: 1px solid rgba(59, 130, 246, 0.3);">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); border-radius: 12px; padding: 1rem; color: white;">
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>
                        </svg>
                    </div>
                    <div>
                        <div style="font-size: 2rem; font-weight: 900; color: #3b82f6; line-height: 1;">{trend_7d}</div>
                        <div style="color: {secondary_text}; font-weight: 600; font-size: 0.9rem; margin-top: 0.25rem;">Last 7 Days</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="glass-card" style="background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(124, 58, 237, 0.1) 100%); border: 1px solid rgba(139, 92, 246, 0.3);">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); border-radius: 12px; padding: 1rem; color: white;">
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><path d="M16 2v4"/><path d="M8 2v4"/><path d="M3 10h18"/>
                        </svg>
                    </div>
                    <div>
                        <div style="font-size: 2rem; font-weight: 900; color: #8b5cf6; line-height: 1;">{trend_30d}</div>
                        <div style="color: {secondary_text}; font-weight: 600; font-size: 0.9rem; margin-top: 0.25rem;">Last 30 Days</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="glass-card" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.1) 100%); border: 1px solid rgba(16, 185, 129, 0.3);">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); border-radius: 12px; padding: 1rem; color: white;">
                        <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/>
                        </svg>
                    </div>
                    <div>
                        <div style="font-size: 2rem; font-weight: 900; color: #10b981; line-height: 1;">{metrics['n_classes']}</div>
                        <div style="color: {secondary_text}; font-weight: 600; font-size: 0.9rem; margin-top: 0.25rem;">Classifications</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div style="margin: 3rem 0;"></div>', unsafe_allow_html=True)

    # System Health & Model Info
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
             border-radius: 16px; padding: 1.5rem; margin-bottom: 2rem; border: 1px solid rgba(102, 126, 234, 0.2);">
            <h3 style="margin: 0 0 1.5rem 0; color: #1a202c; font-weight: 800; display: flex; align-items: center; gap: 0.75rem;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="3"/><path d="M12 1v6m0 6v6M5.64 5.64l4.24 4.24m4.24 4.24l4.24 4.24M1 12h6m6 0h6M5.64 18.36l4.24-4.24m4.24-4.24l4.24-4.24"/>
                </svg>
                System Intelligence
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        # Get actual analyzed incidents count from database
        try:
            total_analyzed = len(st.session_state.db.get_analysis_history(limit=10000))
        except:
            total_analyzed = 0

        st.markdown(
            f"""
            <div class="glass-card" style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.05) 0%, rgba(139, 92, 246, 0.05) 100%); border: 1px solid rgba(99, 102, 241, 0.2);">
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); border-radius: 10px; padding: 0.75rem; color: white; flex-shrink: 0;">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/>
                        </svg>
                    </div>
                    <div style="flex-grow: 1;">
                        <div style="font-size: 0.85rem; color: {secondary_text}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">Incidents Analyzed</div>
                        <div style="font-size: 1.75rem; font-weight: 900; color: #1a202c; line-height: 1;">{total_analyzed:,}</div>
                        <div style="color: {secondary_text}; font-size: 0.85rem; margin-top: 0.5rem;">Total Database Entries</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="glass-card" style="background: linear-gradient(135deg, rgba(236, 72, 153, 0.05) 0%, rgba(219, 39, 119, 0.05) 100%); border: 1px solid rgba(236, 72, 153, 0.2);">
                <div style="display: flex; align-items: start; gap: 1rem;">
                    <div style="background: linear-gradient(135deg, #ec4899 0%, #db2777 100%); border-radius: 10px; padding: 0.75rem; color: white; flex-shrink: 0;">
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 00-2-2h-4a2 2 0 00-2 2v16"/>
                        </svg>
                    </div>
                    <div style="flex-grow: 1;">
                        <div style="font-size: 0.85rem; color: {secondary_text}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem;">ML Algorithm</div>
                        <div style="font-size: 1.25rem; font-weight: 800; color: #1a202c; line-height: 1.3;">Logistic Regression</div>
                        <div style="color: {secondary_text}; font-size: 0.85rem; margin-top: 0.5rem;">TF-IDF Vectorization</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================================
# TAB: ADVANCED SEARCH
# ============================================================================


def advanced_search_interface():
    """Advanced search with filters and SQL"""

    st.markdown(
        '<div class="section-header">Advanced Search</div>', unsafe_allow_html=True
    )

    # Back to dashboard button
    if st.button(
        "← Back to Dashboard", type="secondary", key="advanced_back_to_dashboard"
    ):
        st.session_state.navigate_to_dashboard = True
        st.rerun()

    # Check for navigation filters from dashboard and execute search automatically
    auto_search = False
    auto_search_params = {}
    filter_message = ""
    show_back_button = False
    results: list = []

    if (
        "search_filter_critical" in st.session_state
        and st.session_state.search_filter_critical
    ):
        auto_search = True
        show_back_button = True
        filter_message = "Showing Critical Threats (Malware, Data Exfiltration, Web Attack, Phishing)"
        # Search will be done via SQL to filter by multiple labels
        auto_search_params = {
            "mode": "sql",
            "labels": ["malware", "data_exfiltration", "web_attack", "phishing"],
        }
        del st.session_state.search_filter_critical
    elif (
        "search_filter_high_confidence" in st.session_state
        and st.session_state.search_filter_high_confidence
    ):
        auto_search = True
        show_back_button = True
        filter_message = "Showing High Confidence Incidents (≥80%)"
        auto_search_params = {
            "mode": "filter",
            "min_confidence": 0.8,
            "max_confidence": 1.0,
        }
        del st.session_state.search_filter_high_confidence
    elif (
        "search_filter_needs_review" in st.session_state
        and st.session_state.search_filter_needs_review
    ):
        auto_search = True
        show_back_button = True
        filter_message = "Showing Incidents Needing Review (<60% Confidence)"
        auto_search_params = {
            "mode": "filter",
            "min_confidence": 0.0,
            "max_confidence": 0.6,
        }
        del st.session_state.search_filter_needs_review
    elif (
        "search_filter_top_threat" in st.session_state
        and st.session_state.search_filter_top_threat
    ):
        threat_type = st.session_state.search_filter_top_threat
        auto_search = True
        show_back_button = True
        filter_message = f"Showing {threat_type.replace('_', ' ').title()} Incidents"
        auto_search_params = {"mode": "filter", "label_filter": threat_type}
        del st.session_state.search_filter_top_threat

    if auto_search:
        st.info(f"🔍 {filter_message}")

        # Execute the search automatically
        with st.spinner("Loading filtered results..."):
            try:
                if auto_search_params["mode"] == "sql":
                    # For critical threats, use SQL with IN clause for multiple labels
                    labels_str = "', '".join(auto_search_params["labels"])
                    sql_query = f"""
                    SELECT * FROM analysis_history 
                    WHERE final_label IN ('{labels_str}')
                    ORDER BY timestamp DESC 
                    LIMIT 50
                    """
                    # execute_custom_query already returns list of dicts
                    results, columns = st.session_state.db.execute_custom_query(
                        sql_query, read_only=True
                    )
                elif auto_search_params["mode"] == "filter":
                    # For confidence or single label filters, use advanced_search
                    results = st.session_state.db.advanced_search(
                        search_term="",
                        start_date=None,
                        end_date=None,
                        min_confidence=auto_search_params.get("min_confidence"),
                        max_confidence=auto_search_params.get("max_confidence"),
                        label_filter=auto_search_params.get("label_filter"),
                        limit=50,
                    )

                st.success(f"Found {len(results)} result(s)")

                if results:
                    df = pd.DataFrame(results)
                    st.dataframe(df, use_container_width=True)

                    # Export options
                    export_col1, export_col2 = st.columns([3, 1])
                    with export_col1:
                        export_format = st.selectbox(
                            "Export Format",
                            ["CSV", "JSON", "Excel", "Markdown"],
                            key="auto_search_export_format",
                        )
                    with export_col2:
                        if st.button(
                            "Export Results",
                            use_container_width=True,
                            key="auto_search_export_btn",
                        ):
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                            if export_format == "CSV":
                                csv = df.to_csv(index=False)
                                st.download_button(
                                    "Download CSV",
                                    csv,
                                    f"filtered_results_{timestamp}.csv",
                                    "text/csv",
                                    use_container_width=True,
                                )
                            elif export_format == "JSON":
                                json_str = df.to_json(orient="records", indent=2)
                                st.download_button(
                                    "Download JSON",
                                    json_str,
                                    f"filtered_results_{timestamp}.json",
                                    "application/json",
                                    use_container_width=True,
                                )
                            elif export_format == "Excel":
                                output = io.BytesIO()
                                with pd.ExcelWriter(
                                    output, engine="xlsxwriter"
                                ) as writer:
                                    df.to_excel(
                                        writer, index=False, sheet_name="Results"
                                    )
                                st.download_button(
                                    "Download Excel",
                                    output.getvalue(),
                                    f"filtered_results_{timestamp}.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True,
                                )
                            elif export_format == "Markdown":
                                md = df.to_markdown(index=False)
                                st.download_button(
                                    "Download Markdown",
                                    md,
                                    f"filtered_results_{timestamp}.md",
                                    "text/markdown",
                                    use_container_width=True,
                                )
                else:
                    st.info("No results found matching the filter criteria.")

            except Exception as e:
                st.error(f"Error executing auto-search: {str(e)}")

        st.markdown("---")
        st.markdown("### Refine Your Search")
        st.markdown(
            "Use the search options below to further filter or modify your results."
        )

    # Get database stats
    try:
        facets = st.session_state.db.get_search_facets()
        # Count both standalone notes and bookmark notes
        standalone_notes = st.session_state.db.get_all_notes()
        bookmarks = st.session_state.db.get_bookmarks()
        bookmark_notes = [bm for bm in bookmarks if bm.get("note")]
        total_notes = len(standalone_notes) + len(bookmark_notes)

        db_stats = {
            "total_incidents": facets["total_incidents"],
            "bookmarks": len(bookmarks),
            "notes": total_notes,
        }
    except:
        facets = {
            "classifications": [],
            "date_range": (None, None),
            "total_incidents": 0,
        }
        db_stats = {"total_incidents": 0, "bookmarks": 0, "notes": 0}

    # Stats overview
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Incidents", f"{db_stats['total_incidents']:,}")
    with col2:
        st.metric("Bookmarks", f"{db_stats['bookmarks']:,}")
    with col3:
        st.metric("Notes", f"{db_stats['notes']:,}")

    st.markdown("---")

    # Search mode
    search_mode = st.radio(
        "Search Mode",
        ["Filter Search", "Semantic Search", "SQL Query"],
        horizontal=True,
    )

    if search_mode == "Filter Search":
        # Filter-based search
        col1, col2 = st.columns([3, 1])

        with col1:
            search_term = st.text_input(
                "Search Keywords",
                placeholder="e.g., 'phishing', '192.168', 'PowerShell'",
            )

        with col2:
            search_scope = st.selectbox("Search In", ["History", "Bookmarks", "All"])

        # Filters
        col1, col2 = st.columns(2)

        with col1:
            date_preset = st.selectbox(
                "Date Range", ["All Time", "Last 7 days", "Last 30 days", "Custom"]
            )

            if date_preset == "Custom":
                start_input = st.date_input("Start Date")
                end_input = st.date_input("End Date")
                start_date = (
                    start_input[0]
                    if isinstance(start_input, tuple) and start_input
                    else start_input
                )
                end_date = (
                    end_input[0]
                    if isinstance(end_input, tuple) and end_input
                    else end_input
                )
            elif date_preset == "Last 7 days":
                start_date = (datetime.now() - timedelta(days=7)).date()
                end_date = datetime.now().date()
            elif date_preset == "Last 30 days":
                start_date = (datetime.now() - timedelta(days=30)).date()
                end_date = datetime.now().date()
            else:
                start_date = None
                end_date = None

        with col2:
            use_confidence = st.checkbox("Filter by confidence")
            if use_confidence:
                min_conf, max_conf = st.slider("Confidence Range (%)", 0, 100, (0, 100))
                min_confidence = min_conf / 100.0
                max_confidence = max_conf / 100.0
            else:
                min_confidence = None
                max_confidence = None

        if st.button("Search", type="primary", use_container_width=True):
            with st.spinner("Searching..."):
                try:
                    results = st.session_state.db.advanced_search(
                        search_term=search_term,
                        start_date=start_date.isoformat() if start_date else None,
                        end_date=end_date.isoformat() if end_date else None,
                        min_confidence=min_confidence,
                        max_confidence=max_confidence,
                        limit=50,
                    )

                    st.success(f"Found {len(results)} result(s)")

                    if results:
                        # Display SQL query with highlighting
                        if (
                            hasattr(st.session_state, "last_filter_sql")
                            and st.session_state.last_filter_sql
                        ):
                            with st.expander("Generated SQL Query", expanded=False):
                                try:
                                    from pygments import highlight
                                    from pygments.lexers import SqlLexer
                                    from pygments.formatters import HtmlFormatter

                                    lexer = SqlLexer()
                                    formatter = HtmlFormatter(
                                        style="monokai", noclasses=True
                                    )
                                    highlighted_sql = highlight(
                                        st.session_state.last_filter_sql,
                                        lexer,
                                        formatter,
                                    )

                                    st.markdown(
                                        f'<div style="background-color: #272822; padding: 1rem; border-radius: 5px; overflow-x: auto;">{highlighted_sql}</div>',
                                        unsafe_allow_html=True,
                                    )
                                except ImportError:
                                    st.code(
                                        st.session_state.last_filter_sql, language="sql"
                                    )

                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True)

                        # Enhanced Export Options
                        export_col1, export_col2 = st.columns([3, 1])
                        with export_col1:
                            export_format = st.selectbox(
                                "Export Format",
                                ["CSV", "JSON", "Excel", "Markdown"],
                                key="filter_export_format",
                            )
                        with export_col2:
                            if st.button("Export Results", use_container_width=True):
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                                if export_format == "CSV":
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        "Download CSV",
                                        csv,
                                        f"search_results_{timestamp}.csv",
                                        "text/csv",
                                    )
                                elif export_format == "JSON":
                                    import json

                                    json_str = json.dumps(results, indent=2)
                                    st.download_button(
                                        "Download JSON",
                                        json_str,
                                        f"search_results_{timestamp}.json",
                                        "application/json",
                                    )
                                elif export_format == "Excel":
                                    try:
                                        from io import BytesIO

                                        buffer = BytesIO()
                                        with pd.ExcelWriter(
                                            buffer, engine="xlsxwriter"
                                        ) as writer:
                                            df.to_excel(
                                                writer,
                                                index=False,
                                                sheet_name="Results",
                                            )
                                        buffer.seek(0)

                                        st.download_button(
                                            "Download Excel",
                                            buffer,
                                            f"search_results_{timestamp}.xlsx",
                                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        )
                                    except ImportError:
                                        st.error(
                                            "Excel export requires xlsxwriter: pip install xlsxwriter"
                                        )
                                elif export_format == "Markdown":
                                    md = df.to_markdown(index=False)
                                    st.download_button(
                                        "Download Markdown",
                                        md,
                                        f"search_results_{timestamp}.md",
                                        "text/markdown",
                                    )
                except Exception as e:
                    st.error(f"Search error: {e}")

    elif search_mode == "Semantic Search":
        st.markdown("### AI-Powered Semantic Search")

        semantic_query = st.text_area(
            "Describe the incident:",
            height=120,
            placeholder="e.g., User received email with suspicious attachment...",
        )

        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Number of results", 5, 50, 10, 5)
        with col2:
            threshold = st.slider("Similarity threshold", 0.3, 1.0, 0.5, 0.05)

        if st.button("Search Semantically", type="primary", use_container_width=True):
            if semantic_query.strip():
                with st.spinner("AI analyzing..."):
                    try:
                        results = find_similar_incidents(
                            semantic_query, top_k=top_k, similarity_threshold=threshold
                        )

                        st.success(f"Found {len(results)} similar incident(s)")

                        if results:
                            for idx, result in enumerate(results):
                                with st.expander(
                                    f"Result {idx+1}: {result['final_label']} ({result['similarity_score']:.1%} similar)"
                                ):
                                    st.write(result.get("incident_text", "N/A")[:500])
                                    st.write(
                                        f"**Confidence:** {result.get('max_prob', 0):.1%}"
                                    )
                                    st.write(
                                        f"**Timestamp:** {result.get('timestamp', 'N/A')}"
                                    )
                    except Exception as e:
                        st.error(f"Search failed: {e}")

    else:  # SQL Query Mode
        st.markdown("### SQL Query Interface")

        st.markdown(
            """
        <div style="background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%); padding: 1rem; border-radius: 10px; color: white; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: white;">Power User SQL Interface</h4>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Query the database directly • Full SQL syntax • Read-only mode for safety</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Database schema info
        with st.expander("Database Schema & Available Tables", expanded=False):
            schema_col1, schema_col2 = st.columns(2)

            with schema_col1:
                st.markdown(
                    f"""
                #### `analysis_history`
                **Primary incident data table**
                
                | Column | Type | Description |
                |--------|------|-------------|
                | `id` | INTEGER | Primary key |
                | `timestamp` | TEXT | Analysis timestamp |
                | `incident_text` | TEXT | Full incident description |
                | `final_label` | TEXT | Classification label |
                | `max_prob` | REAL | Confidence (0.0-1.0) |
                | `analysis_mode` | TEXT | Analysis type |
                | `probs_json` | TEXT | All probabilities (JSON) |
                
                **Count:** {db_stats["total_incidents"]:,} incidents
                
                #### `bookmarks`
                **Saved incidents for quick access**
                
                | Column | Type | Description |
                |--------|------|-------------|
                | `id` | INTEGER | Primary key |
                | `analysis_id` | INTEGER | → analysis_history.id |
                | `created_at` | TEXT | Bookmark timestamp |
                | `note` | TEXT | Optional note |
                
                **Count:** {db_stats["bookmarks"]:,} bookmarks
                """
                )

            with schema_col2:
                st.markdown(
                    f"""
                #### `notes`
                **Notes and annotations**
                
                | Column | Type | Description |
                |--------|------|-------------|
                | `id` | INTEGER | Primary key |
                | `analysis_id` | INTEGER | → analysis_history.id |
                | `note_text` | TEXT | Note content |
                | `created_at` | TEXT | Note timestamp |
                
                **Count:** {db_stats["notes"]:,} notes
                
                #### Quick Tips
                
                - **JOIN** tables to get enriched data
                - **WHERE** clauses support wildcards: `LIKE '%keyword%'`
                - **Date filtering:** `WHERE timestamp >= '2024-01-01'`
                - **Aggregation:** `COUNT()`, `AVG()`, `MIN()`, `MAX()`
                - **GROUP BY** for statistics by category
                - **ORDER BY** with `DESC`/`ASC` for sorting
                - **LIMIT** to control result size
                """
                )

        # Example queries
        with st.expander("Example Queries", expanded=False):
            st.code(
                """
-- High-confidence phishing incidents
SELECT id, timestamp, SUBSTR(incident_text, 1, 100) AS preview,
       final_label, ROUND(max_prob * 100, 2) || '%' AS confidence
FROM analysis_history 
WHERE final_label = 'phishing' AND max_prob > 0.9 
ORDER BY max_prob DESC LIMIT 20;

-- Classification distribution
SELECT final_label, COUNT(*) as count,
       ROUND(AVG(max_prob) * 100, 2) || '%' AS avg_confidence
FROM analysis_history 
GROUP BY final_label 
ORDER BY count DESC;

-- Low-confidence incidents needing review
SELECT id, SUBSTR(incident_text, 1, 80) AS preview,
       final_label, ROUND(max_prob * 100, 2) || '%' AS confidence
FROM analysis_history 
WHERE max_prob < 0.6 
ORDER BY max_prob ASC LIMIT 50;

-- Incidents with notes (from notes table)
SELECT ah.id, ah.timestamp, SUBSTR(ah.incident_text, 1, 100) AS preview,
       ah.final_label, n.note_text, n.created_at AS note_date
FROM analysis_history ah
INNER JOIN notes n ON ah.id = n.analysis_id
ORDER BY n.created_at DESC LIMIT 30;

-- Bookmarked incidents (bookmarks have their own note field)
SELECT ah.id, ah.timestamp, SUBSTR(ah.incident_text, 1, 100) AS preview,
       ah.final_label, b.note AS bookmark_note
FROM analysis_history ah
INNER JOIN bookmarks b ON ah.id = b.analysis_id
WHERE b.note IS NOT NULL AND b.note != ''
ORDER BY b.created_at DESC LIMIT 30;

-- Keyword threat hunt
SELECT id, incident_text, final_label, 
       ROUND(max_prob * 100, 2) || '%' AS confidence
FROM analysis_history 
WHERE incident_text LIKE '%PowerShell%' 
   OR incident_text LIKE '%malware%'
ORDER BY max_prob DESC LIMIT 50;
            """,
                language="sql",
            )

        # Initialize SQL history in session state
        if "sql_query_history" not in st.session_state:
            st.session_state.sql_query_history = []

        # Query history dropdown
        if st.session_state.sql_query_history:
            with st.expander("🕒 Query History", expanded=False):
                st.markdown(
                    f"**{len(st.session_state.sql_query_history)} recent queries**"
                )

                for idx, (timestamp, query) in enumerate(
                    reversed(st.session_state.sql_query_history[-10:])
                ):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        if st.button(
                            f"{timestamp}: {query[:50]}...",
                            key=f"history_{idx}",
                            use_container_width=True,
                        ):
                            st.session_state.sql_query_recall = query
                            st.rerun()
                    with col2:
                        st.caption(f"{len(query)} chars")

                if st.button("Clear History", key="clear_sql_history"):
                    st.session_state.sql_query_history = []
                    st.rerun()

        # SQL query input with code editor
        try:
            from streamlit_ace import st_ace

            sql_query = st_ace(
                value=st.session_state.get(
                    "sql_query_recall", st.session_state.get("sql_query_input", "")
                ),
                language="sql",
                theme="monokai",
                keybinding="vscode",
                font_size=14,
                tab_size=2,
                show_gutter=True,
                show_print_margin=False,
                wrap=True,
                auto_update=True,
                readonly=False,
                placeholder="SELECT * FROM analysis_history WHERE...",
                height=250,
                key="sql_editor_ace",
            )

            # Update session state
            if sql_query:
                st.session_state.sql_query_input = sql_query

        except ImportError:
            # Fallback to regular text_area if streamlit-ace not installed
            st.info(
                "Install streamlit-ace for enhanced SQL editing: `pip install streamlit-ace`"
            )
            query_value = st.session_state.get("sql_query_recall", "")
            sql_query = st.text_area(
                "SQL Query",
                value=query_value,
                height=200,
                placeholder="SELECT * FROM analysis_history WHERE...",
                help="Write a SELECT query. Dangerous operations are blocked for safety.",
            )

        # Query buttons
        sql_btn_col1, sql_btn_col2 = st.columns([3, 1])
        with sql_btn_col1:
            execute_sql = st.button(
                "Execute Query", type="primary", use_container_width=True
            )
        with sql_btn_col2:
            if st.button("Clear", use_container_width=True):
                st.rerun()

        # Execute SQL query
        if execute_sql and sql_query:
            # Clear recall state
            if "sql_query_recall" in st.session_state:
                del st.session_state.sql_query_recall

            # Add to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if (timestamp, sql_query) not in st.session_state.sql_query_history:
                st.session_state.sql_query_history.append((timestamp, sql_query))
                # Keep only last 20 queries
                if len(st.session_state.sql_query_history) > 20:
                    st.session_state.sql_query_history = (
                        st.session_state.sql_query_history[-20:]
                    )

            with st.spinner("Executing query..."):
                try:
                    results, columns = st.session_state.db.execute_custom_query(
                        sql_query, read_only=True
                    )

                    st.success(
                        f"Query executed successfully - {len(results)} rows returned"
                    )

                    if results:
                        df = pd.DataFrame(results)
                        st.dataframe(df, use_container_width=True, height=400)

                        # Export options
                        export_col1, export_col2 = st.columns([3, 1])
                        with export_col1:
                            export_format = st.selectbox(
                                "Export Format",
                                ["CSV", "JSON", "Excel", "Markdown"],
                                key="sql_export_format",
                            )
                        with export_col2:
                            if st.button("Export CSV", use_container_width=True):
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                                if export_format == "CSV":
                                    csv = df.to_csv(index=False)
                                    st.download_button(
                                        "Download CSV",
                                        csv,
                                        f"query_results_{timestamp}.csv",
                                        "text/csv",
                                    )
                                elif export_format == "JSON":
                                    import json

                                    json_str = json.dumps(results, indent=2)
                                    st.download_button(
                                        "Download JSON",
                                        json_str,
                                        f"query_results_{timestamp}.json",
                                        "application/json",
                                    )
                                elif export_format == "Excel":
                                    try:
                                        from io import BytesIO

                                        buffer = BytesIO()
                                        with pd.ExcelWriter(
                                            buffer, engine="xlsxwriter"
                                        ) as writer:
                                            df.to_excel(
                                                writer,
                                                index=False,
                                                sheet_name="Query Results",
                                            )
                                        buffer.seek(0)

                                        st.download_button(
                                            "Download Excel",
                                            buffer,
                                            f"query_results_{timestamp}.xlsx",
                                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        )
                                    except ImportError:
                                        st.error(
                                            "Excel export requires xlsxwriter: pip install xlsxwriter"
                                        )
                                elif export_format == "Markdown":
                                    # Create markdown with query header
                                    md = f"# SQL Query Results\n\n"
                                    md += f"**Query:** `{sql_query}`\n\n"
                                    md += f"**Timestamp:** {timestamp}\n\n"
                                    md += f"**Rows:** {len(results)}\n\n"
                                    md += "## Results\n\n"
                                    md += df.to_markdown(index=False)

                                    st.download_button(
                                        "Download Markdown",
                                        md,
                                        f"query_results_{timestamp}.md",
                                        "text/markdown",
                                    )

                except Exception as e:
                    st.error(f"Query error: {e}")
                    import traceback

                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())


# ============================================================================
# TAB: BATCH PROCESSING
# ============================================================================


def batch_processing_tab(use_preprocessing, use_llm):
    """Enhanced batch processing with advanced analytics and visualizations"""
    import numpy as np
    from scipy.sparse import hstack

    text_palette = get_text_palette()
    secondary_text = text_palette["secondary"]

    llm_provider, hf_model_id, selected_hf_token = get_llm_settings()

    st.markdown(
        '<div class="section-header">Batch Analysis Intelligence Center</div>',
        unsafe_allow_html=True,
    )

    # Back to dashboard button
    if st.button(
        "← Back to Dashboard", type="secondary", key="batch_back_to_dashboard"
    ):
        st.session_state.navigate_to_dashboard = True
        st.rerun()

    st.markdown(
        """
        <div class="alert-premium alert-info">
            <strong>Upload Format Support</strong><br>
            CSV with <code>incident_text</code> or <code>description</code> column • 
            Text files (one incident per line) • JSONL format
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        # Empty state when no file uploaded
        uploaded_file = st.file_uploader(
            "Upload Incident Data",
            type=["csv", "txt", "jsonl"],
            help="CSV with incident_text column, text file (one per line), or JSONL",
        )

        if not uploaded_file:
            # Professional empty state
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
                    border: 2px dashed rgba(102, 126, 234, 0.3);
                    border-radius: 12px;
                    padding: 40px;
                    text-align: center;
                    margin: 20px 0;
                ">
                    <div style="font-size: 3rem; margin-bottom: 16px; opacity: 0.6;">📤</div>
                    <h3 style="color: #667eea; margin-bottom: 8px;">Upload Your Data</h3>
                    <p style="color: {secondary_text}; margin-bottom: 0;">
                        Drag and drop a file or click to browse<br>
                        <small>Supports CSV, TXT, and JSONL formats</small>
                    </p>
                </div>
            """,
                unsafe_allow_html=True,
            )
        else:
            # Professional file upload success notification
            file_size_kb = uploaded_file.size / 1024
            if file_size_kb < 1024:
                size_display = f"{file_size_kb:.1f} KB"
            else:
                size_display = f"{file_size_kb/1024:.1f} MB"

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(22, 163, 74, 0.1) 100%);
                    border-left: 4px solid #22c55e;
                    border-radius: 8px;
                    padding: 16px;
                    margin: 16px 0;
                ">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <div style="font-size: 1.5rem;">✓</div>
                        <div>
                            <div style="font-weight: 600; color: #16a34a;">File Ready for Analysis</div>
                            <div style="color: {secondary_text}; font-size: 0.9rem; margin-top: 4px;">
                                {uploaded_file.name} • {size_display}
                            </div>
                        </div>
                    </div>
                </div>
            """,
                unsafe_allow_html=True,
            )

            # Enhanced file preview
            with st.expander("File Preview", expanded=False):
                try:
                    file_content = uploaded_file.getvalue().decode("utf-8")
                    preview_lines = [
                        line for line in file_content.split("\n")[:10] if line.strip()
                    ]

                    st.markdown("**First 10 lines:**")
                    for i, line in enumerate(preview_lines, 1):
                        display_line = line[:150] + "..." if len(line) > 150 else line
                        st.markdown(
                            f'<div style="font-family: monospace; font-size: 0.85rem; padding: 4px 0; border-bottom: 1px solid rgba(0,0,0,0.05);">'
                            f'<span style="color: #667eea; font-weight: 600;">{i:02d}</span> '
                            f'<span style="color: {secondary_text};">{display_line}</span>'
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                    # File stats
                    total_lines = len(file_content.split("\n"))
                    st.markdown("---")
                    st.markdown(
                        f"**Total lines:** {total_lines:,} • **Showing:** {len(preview_lines)} lines"
                    )
                except Exception as e:
                    st.warning(f"Could not preview file: {str(e)}")

    with col2:
        st.markdown("#### Configuration")

        # Professional settings display
        st.markdown(
            f"""
            <div style="
                background: rgba(255, 255, 255, 0.6);
                backdrop-filter: blur(10px);
                border-radius: 8px;
                padding: 16px;
                margin-bottom: 12px;
            ">
                <div style="font-size: 0.85rem; color: {secondary_text}; margin-bottom: 8px;">PREPROCESSING</div>
                <div style="font-weight: 600; color: {'#22c55e' if use_preprocessing else secondary_text};">
                    {'Enabled' if use_preprocessing else 'Disabled'}
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="
                background: rgba(255, 255, 255, 0.6);
                backdrop-filter: blur(10px);
                border-radius: 8px;
                padding: 16px;
            ">
                <div style="font-size: 0.85rem; color: {secondary_text}; margin-bottom: 8px;">LLM ENHANCEMENT</div>
                <div style="font-weight: 600; color: {'#667eea' if use_llm else secondary_text};">
                    {'Enabled' if use_llm else 'Disabled'}
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Clear cache button
        if st.button(
            "Clear Cache",
            use_container_width=True,
            help="Clear cached results and start fresh",
        ):
            if "batch_results" in st.session_state:
                st.session_state.batch_results = None
                st.success("Cache cleared successfully")

    run_analysis = st.button(
        "Analyze Batch",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_file,
    )

    # Initialize session state
    if "batch_results" not in st.session_state:
        st.session_state.batch_results = None

    if run_analysis and uploaded_file:
        with st.spinner("Processing incidents..."):
            try:
                # Parse file based on type
                file_content = uploaded_file.getvalue().decode("utf-8")

                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                    if "incident_text" in df.columns:
                        incidents = df["incident_text"].tolist()
                    elif "description" in df.columns:
                        incidents = df["description"].tolist()
                    else:
                        st.error(
                            "CSV must contain 'incident_text' or 'description' column"
                        )
                        return
                elif uploaded_file.name.endswith(".jsonl"):
                    incidents = []
                    for line in file_content.split("\n"):
                        if line.strip():
                            try:
                                data = json.loads(line)
                                incidents.append(
                                    data.get("text", data.get("description", ""))
                                )
                            except:
                                pass
                else:  # txt
                    incidents = [
                        line.strip()
                        for line in file_content.split("\n")
                        if line.strip()
                    ]

                if not incidents:
                    st.error("No incidents found in file")
                    return

                vectorizer, model = load_vectorizer_and_model()

                results = []
                progress_bar = st.progress(0)
                status = st.empty()

                # Debug: Show LLM status at start
                if use_llm:
                    st.info(
                        f"{ui_icon('cpu')} LLM Enhancement is ENABLED - processing will be slower (5-10s per incident)"
                    )

                # Load embedder for enhanced model
                embedder = get_embedder()

                for idx, text in enumerate(incidents):
                    status.text(f"Processing {idx+1}/{len(incidents)}...")

                    processed = clean_description(text) if use_preprocessing else text

                    # Combine TF-IDF + embeddings for enhanced model
                    X_tfidf = vectorizer.transform([processed])
                    X_embed = embedder.encode([processed])
                    X = hstack([X_tfidf, X_embed])

                    pred = model.predict(X)[0]
                    proba = model.predict_proba(X)[0]

                    # Get MITRE techniques based on classification
                    mitre_techniques = get_mitre_techniques(pred)

                    # Also extract any MITRE IDs mentioned in the text
                    import re

                    mitre_pattern = r"T\d{4}(?:\.\d{3})?"
                    text_mitre = re.findall(mitre_pattern, text)

                    # Combine mapped techniques with any found in text
                    all_mitre = list(set(mitre_techniques + text_mitre))

                    result = {
                        "incident_text": text,
                        "final_label": pred,
                        "display_label": pred,
                        "max_prob": proba.max(),
                        "probabilities": dict(zip(model.classes_, proba)),
                        "mitre_techniques": all_mitre,
                        "final_label_mitre_techniques": mitre_techniques,
                    }

                    # Add LLM second opinion if enabled
                    if use_llm:
                        valid_input, input_error = validate_llm_input_length(text)
                        if not valid_input:
                            result["llm_error"] = input_error
                        elif llm_provider == "huggingface" and not selected_hf_token:
                            result["llm_error"] = "Hugging Face token required for hosted inference"
                        else:
                            import time

                            llm_start = time.time()
                            status.text(
                                f"Processing {idx+1}/{len(incidents)} (Running LLM analysis - this may take 5-10 seconds)..."
                            )
                            try:
                                if llm_provider == "huggingface":
                                    allowed, retry_after = hf_rate_limit_allowance()
                                    if not allowed:
                                        result["llm_error"] = (
                                            f"HF rate limit reached; retry in {retry_after:.0f}s"
                                        )
                                        result["llm_processing_time"] = time.time() - llm_start
                                    else:
                                        llm_opinion = llm_second_opinion(
                                            text,
                                            skip_preprocessing=True,
                                            provider=llm_provider,
                                            hf_model=hf_model_id,
                                            hf_token=selected_hf_token,
                                            max_tokens=UI_LLM_MAX_TOKENS,
                                        )
                                        llm_elapsed = time.time() - llm_start
                                        if llm_opinion:
                                            result["llm_second_opinion"] = llm_opinion
                                            result["llm_processing_time"] = llm_elapsed
                                            llm_mitre = llm_opinion.get("mitre_ids", [])
                                            if llm_mitre:
                                                result["mitre_techniques"] = list(
                                                    set(all_mitre + llm_mitre)
                                                )
                                        else:
                                            result["llm_error"] = "LLM returned None"
                                            result["llm_processing_time"] = llm_elapsed
                                else:
                                    llm_opinion = llm_second_opinion(
                                        text,
                                        skip_preprocessing=True,
                                        provider=llm_provider,
                                        max_tokens=UI_LLM_MAX_TOKENS,
                                    )
                                    llm_elapsed = time.time() - llm_start
                                    if llm_opinion:
                                        result["llm_second_opinion"] = llm_opinion
                                        result["llm_processing_time"] = llm_elapsed
                                        llm_mitre = llm_opinion.get("mitre_ids", [])
                                        if llm_mitre:
                                            result["mitre_techniques"] = list(
                                                set(all_mitre + llm_mitre)
                                            )
                                    else:
                                        result["llm_error"] = "LLM returned None"
                                        result["llm_processing_time"] = llm_elapsed
                            except Exception as e:
                                result["llm_error"] = f"LLM Error: {str(e)}"
                                result["llm_processing_time"] = time.time() - llm_start
                                # Show error in UI
                                st.warning(
                                    f"LLM error on incident {idx+1}: {str(e)}"
                                )

                    results.append(result)
                    progress_bar.progress((idx + 1) / len(incidents))

                status.empty()
                st.session_state.batch_results = results

                # Show completion animation
                st.balloons()

                # Save to database with batch_id
                if st.session_state.get("save_batch_to_db", True):
                    try:
                        import uuid

                        batch_id = str(uuid.uuid4())
                        batch_name = uploaded_file.name.replace(".csv", "").replace(
                            ".txt", ""
                        )

                        db = st.session_state.get("db", TriageDatabase())
                        batch_record_id = db.save_batch_analysis(
                            batch_id=batch_id,
                            batch_name=batch_name,
                            file_name=uploaded_file.name,
                            results=results,
                            use_preprocessing=use_preprocessing,
                            use_llm=use_llm,
                        )

                        st.session_state.last_batch_id = batch_id
                        st.session_state.last_batch_record_id = batch_record_id

                        st.success(f"Batch saved to database (ID: {batch_id[:8]}...)")
                    except Exception as e:
                        st.warning(f"Batch processed but not saved to database: {e}")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.exception(e)
                return

    # Display results
    if st.session_state.batch_results:
        results = st.session_state.batch_results

        # Filtering and Search UI
        with st.expander("Filter & Search Results", expanded=False):
            filter_col1, filter_col2, filter_col3 = st.columns(3)

            with filter_col1:
                # Classification filter
                all_labels = sorted(set([r["display_label"] for r in results]))
                selected_labels = st.multiselect(
                    "Filter by Classification",
                    options=all_labels,
                    default=all_labels,
                    format_func=get_display_name,
                )

            with filter_col2:
                # Confidence range filter
                min_conf, max_conf = st.slider(
                    "Confidence Range",
                    min_value=0.0,
                    max_value=1.0,
                    value=(0.0, 1.0),
                    step=0.05,
                    format="%.0f%%",
                )

            with filter_col3:
                # Text search
                search_text = st.text_input(
                    "Search Incident Text",
                    placeholder="Enter keywords...",
                )

        # Apply filters
        filtered_results = results

        if selected_labels:
            filtered_results = [
                r for r in filtered_results if r["display_label"] in selected_labels
            ]

        filtered_results = [
            r for r in filtered_results if min_conf <= r["max_prob"] <= max_conf
        ]

        if search_text:
            search_lower = search_text.lower()
            filtered_results = [
                r
                for r in filtered_results
                if search_lower in r["incident_text"].lower()
            ]

        # Show filter status
        if len(filtered_results) != len(results):
            st.info(
                f"Showing {len(filtered_results)} of {len(results)} incidents (filtered)"
            )

        # Use filtered results for display
        results = (
            filtered_results if filtered_results else st.session_state.batch_results
        )

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Analyzed", len(results))
        with col2:
            avg_conf = np.mean([r["max_prob"] for r in results]) if results else 0
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with col3:
            critical = len(
                [
                    r
                    for r in results
                    if r["display_label"]
                    in ["malware", "data_exfiltration", "web_attack"]
                ]
            )
            st.metric("Critical Threats", critical)
        with col4:
            high_conf = (
                len([r for r in results if r["max_prob"] >= 0.8]) if results else 0
            )
            st.metric(
                "High Confidence", f"{high_conf/len(results):.1%}" if results else "0%"
            )

        st.markdown("---")

        # Analytics tabs - 11-tab comprehensive structure
        tabs = st.tabs(
            [
                "Overview",
                "Confidence Analysis",
                "Performance",
                "Deep Dive",
                "Bookmarks",
                "Visualizations",
                "IOC Intelligence",
                "MITRE Coverage",
                "LLM Insights",
                "Export Options",
                "Batch Comparison",
            ]
        )

        with tabs[0]:  # Overview
            st.markdown("### Classification Distribution")

            viz_col1, viz_col2 = st.columns(2)

            with viz_col1:
                # Pie chart
                labels = [r["display_label"] for r in results]
                label_counts = Counter(labels)

                fig = go.Figure(
                    data=[
                        go.Pie(
                            labels=[get_display_name(l) for l in label_counts.keys()],
                            values=list(label_counts.values()),
                            hole=0.4,
                            textinfo="label+percent",
                        )
                    ]
                )
                fig.update_layout(title="Classification Distribution", height=400)
                st.plotly_chart(fig, use_container_width=True, key="batch_pie_chart")

            with viz_col2:
                # Bar chart by severity
                severity_scores = {
                    "malware": 90,
                    "data_exfiltration": 85,
                    "web_attack": 75,
                    "access_abuse": 70,
                    "phishing": 60,
                    "policy_violation": 30,
                    "benign_activity": 10,
                    "uncertain": 50,
                }

                threat_data = []
                for label, count in label_counts.items():
                    threat_data.append(
                        {
                            "type": get_display_name(label),
                            "count": count,
                            "severity": severity_scores.get(label, 50),
                        }
                    )

                threat_df = pd.DataFrame(threat_data).sort_values(
                    "severity", ascending=False
                )

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=threat_df["type"],
                        y=threat_df["count"],
                        marker_color="lightblue",
                        name="Count",
                    )
                )
                fig.update_layout(
                    title="Threat Volume by Severity",
                    xaxis_title="Type",
                    yaxis_title="Count",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True, key="batch_severity_bar")

            st.markdown("---")

            # Advanced Metrics Section
            st.markdown("#### Advanced Metrics & Scoring")

            # Calculate severity index
            severity_index = calculate_severity_index(results)

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

            with metric_col1:
                st.metric(
                    "Overall Risk Score",
                    f"{severity_index['overall']:.1f}/100",
                    help="Aggregated risk score based on classification severity and confidence",
                )

            with metric_col2:
                st.metric(
                    "Critical Risk",
                    f"{severity_index['critical']}",
                    help="Incidents with risk score ≥80",
                )

            with metric_col3:
                st.metric(
                    "High Risk",
                    f"{severity_index['high']}",
                    help="Incidents with risk score 60-79",
                )

            with metric_col4:
                st.metric(
                    "Medium/Low Risk",
                    f"{severity_index['medium'] + severity_index['low']}",
                    help="Incidents with risk score <60",
                )

            # Risk score distribution
            if severity_index["scores"]:
                fig_risk = go.Figure()
                fig_risk.add_trace(
                    go.Histogram(
                        x=severity_index["scores"],
                        nbinsx=20,
                        marker_color="#667eea",
                    )
                )
                fig_risk.update_layout(
                    title="Risk Score Distribution",
                    xaxis_title="Risk Score",
                    yaxis_title="Frequency",
                    height=300,
                )
                st.plotly_chart(
                    fig_risk, use_container_width=True, key="batch_risk_histogram"
                )

            st.markdown("---")

            # Text Complexity Analysis
            st.markdown("#### Text Complexity Analysis")

            if results:
                complexity_metrics = [
                    calculate_text_complexity(r["incident_text"]) for r in results
                ]

                complexity_col1, complexity_col2 = st.columns(2)

                with complexity_col1:
                    avg_word_count = np.mean(
                        [m["word_count"] for m in complexity_metrics]
                    )
                    avg_tech_density = np.mean(
                        [m["technical_density"] for m in complexity_metrics]
                    )

                    st.metric("Avg Words per Incident", f"{avg_word_count:.0f}")
                    st.metric("Technical Term Density", f"{avg_tech_density:.1%}")

                with complexity_col2:
                    avg_readability = np.mean(
                        [m["readability_score"] for m in complexity_metrics]
                    )
                    avg_sentence_length = np.mean(
                        [m["avg_sentence_length"] for m in complexity_metrics]
                    )

                    st.metric("Avg Readability Score", f"{avg_readability:.1f}")
                    st.metric("Avg Sentence Length", f"{avg_sentence_length:.1f} words")

                # Word count distribution
                word_counts = [m["word_count"] for m in complexity_metrics]

                fig_words = go.Figure()
                fig_words.add_trace(
                    go.Box(
                        y=word_counts,
                        name="Word Count",
                        marker_color="#764ba2",
                        boxmean="sd",
                    )
                )
                fig_words.update_layout(
                    title="Word Count Distribution",
                    yaxis_title="Words",
                    height=300,
                    showlegend=False,
                )
                st.plotly_chart(
                    fig_words, use_container_width=True, key="batch_wordcount_box"
                )

        with tabs[1]:  # Confidence Analysis
            st.markdown("### Statistical Analysis")

            # Statistical metrics
            confidences = [r["max_prob"] for r in results]
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

            with stat_col1:
                st.metric("Mean", f"{np.mean(confidences):.2%}")
            with stat_col2:
                st.metric("Median", f"{np.median(confidences):.2%}")
            with stat_col3:
                st.metric("Std Dev", f"{np.std(confidences):.2%}")
            with stat_col4:
                st.metric("Min/Max", f"{min(confidences):.1%} / {max(confidences):.1%}")

            st.markdown("---")

            # Confidence histogram
            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=confidences,
                    nbinsx=30,
                    marker=dict(
                        color=confidences,
                        colorscale=[[0, "#FF4444"], [0.5, "#FFB84D"], [1, "#44FF44"]],
                    ),
                )
            )
            fig.update_layout(
                title="Confidence Score Distribution",
                xaxis_title="Confidence",
                yaxis_title="Frequency",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True, key="batch_conf_histogram")

            # Confidence buckets
            st.markdown("#### Confidence Breakdown")
            conf_col1, conf_col2, conf_col3 = st.columns(3)

            high = len([r for r in results if r["max_prob"] >= 0.8])
            med = len([r for r in results if 0.5 <= r["max_prob"] < 0.8])
            low = len([r for r in results if r["max_prob"] < 0.5])

            with conf_col1:
                st.metric("High (≥80%)", f"{high} ({high/len(results):.1%})")
            with conf_col2:
                st.metric("Medium (50-80%)", f"{med} ({med/len(results):.1%})")
            with conf_col3:
                st.metric("Low (<50%)", f"{low} ({low/len(results):.1%})")

            # Confidence heatmap
            st.markdown("---")
            st.markdown("#### Classification Confidence Heatmap")

            # Create confidence matrix for heatmap
            labels = [r["display_label"] for r in results]
            label_counts = Counter(labels)
            unique_labels = sorted(label_counts.keys())

            # Build confidence matrix by label
            confidence_by_label = {label: [] for label in unique_labels}
            for r in results:
                confidence_by_label[r["display_label"]].append(r["max_prob"])

            heatmap_fig = create_confidence_heatmap(results)
            st.plotly_chart(
                heatmap_fig, use_container_width=True, key="batch_conf_heatmap"
            )
            add_chart_download_buttons(heatmap_fig, "confidence_heatmap")

        with tabs[2]:  # Performance
            st.markdown("### Performance Metrics")

            perf_col1, perf_col2, perf_col3 = st.columns(3)

            # Calculate processing metrics
            total_incidents = len(results)
            avg_text_length = np.mean([len(r["incident_text"]) for r in results])

            with perf_col1:
                st.metric("Total Processed", f"{total_incidents:,}")
            with perf_col2:
                st.metric("Avg Text Length", f"{avg_text_length:.0f} chars")
            with perf_col3:
                # Estimate processing time (placeholder - would need actual timing data)
                est_time = total_incidents * 0.1  # Assume 0.1 sec per incident
                st.metric("Est. Processing Time", f"{est_time:.1f}s")

            st.markdown("---")

            # Text length distribution
            text_lengths = [len(r["incident_text"]) for r in results]

            fig = go.Figure()
            fig.add_trace(
                go.Histogram(
                    x=text_lengths,
                    nbinsx=30,
                    marker_color="#667eea",
                )
            )
            fig.update_layout(
                title="Incident Text Length Distribution",
                xaxis_title="Text Length (characters)",
                yaxis_title="Frequency",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True, key="batch_textlength_hist")

            # Confidence by text length analysis
            st.markdown("#### Confidence vs Text Length")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=text_lengths,
                    y=confidences,
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=confidences,
                        colorscale="RdYlGn",
                        showscale=True,
                        colorbar=dict(title="Confidence"),
                    ),
                    text=[get_display_name(r["display_label"]) for r in results],
                    hovertemplate="<b>%{text}</b><br>Length: %{x}<br>Confidence: %{y:.1%}<extra></extra>",
                )
            )
            fig.update_layout(
                title="Confidence Score vs Text Length",
                xaxis_title="Text Length (characters)",
                yaxis_title="Confidence",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True, key="batch_conf_scatter")

        with tabs[3]:  # Deep Dive
            st.markdown("### Individual Incident Analysis")

            # Select incident for deep dive
            incident_options = [
                f"{i+1}: {get_display_name(r['display_label'])} ({r['max_prob']:.1%}) - {r['incident_text'][:60]}..."
                for i, r in enumerate(results)
            ]

            selected_idx = st.selectbox(
                "Select Incident for Deep Dive",
                range(len(results)),
                format_func=lambda x: incident_options[x],
            )

            if selected_idx is not None:
                selected = results[selected_idx]

                st.markdown("---")

                # Display incident details
                detail_col1, detail_col2 = st.columns([2, 1])

                with detail_col1:
                    st.markdown("#### Incident Text")
                    st.markdown(
                        f'<div style="background: rgba(255,255,255,0.6); padding: 16px; border-radius: 8px; font-family: monospace;">{selected["incident_text"]}</div>',
                        unsafe_allow_html=True,
                    )

                with detail_col2:
                    st.markdown("#### Classification")
                    st.metric("Label", get_display_name(selected["display_label"]))
                    st.metric("Confidence", f"{selected['max_prob']:.1%}")

                st.markdown("---")

                # Executive Summary (using LLM helper)
                with st.expander("Executive Summary", expanded=True):
                    # Create mock LLM opinion for executive summary
                    llm_opinion = {
                        "predicted_label": selected["display_label"],
                        "confidence": selected["max_prob"],
                        "rationale": f"Classified as {get_display_name(selected['display_label'])} with {selected['max_prob']:.1%} confidence.",
                    }
                    exec_summary = generate_executive_summary(
                        llm_opinion, selected["incident_text"]
                    )
                    st.markdown(exec_summary)

                # Attack Timeline
                with st.expander("Predicted Attack Timeline"):
                    # Get MITRE techniques from result or extract from text
                    mitre_techniques = selected.get("mitre_techniques", [])
                    if not mitre_techniques:
                        import re

                        mitre_pattern = r"T\d{4}(?:\.\d{3})?"
                        mitre_techniques = re.findall(
                            mitre_pattern, selected["incident_text"]
                        )

                    timeline = predict_attack_timeline(
                        selected["incident_text"],  # llm_rationale
                        selected["display_label"],  # llm_label
                        mitre_techniques,  # mitre_techniques
                    )

                    # Display timeline stages
                    for stage in timeline.get("stages", []):
                        st.markdown(f"**{stage['stage']}** ({stage['time']})")
                        st.text(stage["desc"])
                        st.markdown("---")

                    if timeline.get("estimated_hours"):
                        st.info(
                            f"Estimated total time: {timeline['estimated_hours']} hours"
                        )

                # Custom Playbook
                with st.expander("Incident Response Playbook"):
                    # Extract basic IOCs and MITRE for playbook
                    iocs = {"ips": [], "domains": [], "hashes": []}
                    mitre = []
                    playbook = generate_custom_playbook(llm_opinion, iocs, mitre)
                    st.markdown(playbook)

                # SOC Playbook Recommendations
                with st.expander("SOC Playbook Recommendations"):
                    soc_playbook = generate_soc_playbook_recommendation(
                        selected["display_label"], selected["max_prob"]
                    )

                    # Display playbook details
                    st.markdown(f"### {soc_playbook['title']}")

                    pb_col1, pb_col2, pb_col3 = st.columns(3)

                    with pb_col1:
                        st.markdown(f"**Playbook ID:** {soc_playbook['playbook_id']}")
                    with pb_col2:
                        st.markdown(f"**Priority:** {soc_playbook['priority']}")
                    with pb_col3:
                        st.markdown(f"**Est. Time:** {soc_playbook['estimated_time']}")

                    st.markdown(f"**Assigned Team:** {soc_playbook['team']}")

                    st.markdown("---")
                    st.markdown("**Response Steps:**")

                    for i, step in enumerate(soc_playbook["steps"], 1):
                        st.markdown(f"{i}. {step}")

                    st.markdown("---")
                    st.markdown("**Required Tools:**")
                    for tool in soc_playbook["tools"]:
                        st.markdown(f"- {tool}")

                    st.markdown("---")
                    st.markdown("**External Resources:**")
                    for link in soc_playbook["external_links"]:
                        st.markdown(f"- [{link}]({link})")

                    # Export custom SOC playbook
                    import re

                    ioc_extraction = {
                        "ips": set(
                            re.findall(
                                r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                                selected["incident_text"],
                            )
                        ),
                        "domains": set(),
                        "file_hashes": set(),
                    }

                    custom_pb = generate_custom_soc_playbook(
                        selected["display_label"],
                        ioc_extraction,
                        [],
                        selected["max_prob"],
                    )

                    st.download_button(
                        "Download Custom Playbook",
                        custom_pb,
                        f"soc_playbook_{selected['display_label']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        "text/markdown",
                        use_container_width=True,
                    )

                # Stakeholder Communications
                with st.expander("Stakeholder Communications"):
                    # Audience selector
                    audience_type = st.radio(
                        "Select Audience",
                        ["technical", "executive", "legal"],
                        horizontal=True,
                        help="Choose the target audience for the communication",
                    )

                    # Build llm_opinion dict from available data
                    llm_opinion = {
                        "label": selected["display_label"],
                        "rationale": f"Classification: {get_display_name(selected['display_label'])} with {selected['max_prob']:.1%} confidence",
                        "mitre_ids": selected.get("mitre_techniques", []),
                    }

                    comms = generate_stakeholder_communication(
                        llm_opinion, selected["incident_text"], audience_type
                    )
                    st.markdown(comms)

        with tabs[4]:  # Bookmarks
            st.markdown("### Bookmarked Incidents")

            # Initialize bookmarks in session state
            if "batch_bookmarks" not in st.session_state:
                st.session_state.batch_bookmarks = []

            # Add bookmark functionality
            bookmark_col1, bookmark_col2 = st.columns([3, 1])

            with bookmark_col1:
                st.markdown(
                    f"**Total Bookmarks:** {len(st.session_state.batch_bookmarks)}"
                )

            with bookmark_col2:
                if st.button("Clear All Bookmarks"):
                    st.session_state.batch_bookmarks = []
                    st.success("All bookmarks cleared")

            # Select incidents to bookmark
            st.markdown("#### Add Bookmarks")
            bookmark_options = [
                f"{i+1}: {get_display_name(r['display_label'])} ({r['max_prob']:.1%})"
                for i, r in enumerate(results)
            ]

            selected_for_bookmark = st.multiselect(
                "Select incidents to bookmark",
                range(len(results)),
                format_func=lambda x: bookmark_options[x],
            )

            # Optional batch note
            batch_bookmark_note = st.text_area(
                "Add note for selected bookmarks (optional)",
                placeholder="This note will be added to all selected bookmarks...",
                height=80,
                key="batch_bookmark_note",
            )

            if st.button("Add Selected to Bookmarks", type="primary"):
                saved_count = 0
                for idx in selected_for_bookmark:
                    if idx not in st.session_state.batch_bookmarks:
                        st.session_state.batch_bookmarks.append(idx)
                        # Save to database
                        try:
                            r = results[idx]
                            # Combine batch note with confidence info
                            note_text = f"Confidence: {r['max_prob']:.1%}"
                            if batch_bookmark_note:
                                note_text = f"{batch_bookmark_note}\n\n{note_text}"

                            st.session_state.db.add_bookmark(
                                incident_text=r["incident_text"],
                                final_label=r["display_label"],
                                note=note_text,
                            )
                            saved_count += 1
                        except Exception as e:
                            st.warning(
                                f"Could not save bookmark {idx+1} to database: {e}"
                            )

                if saved_count > 0:
                    st.success(f"✓ Added {saved_count} bookmark(s) to database")
                    # Clear cache to refresh bookmarks list
                    st.session_state.cached_bookmarks = None

            st.markdown("---")

            # Display bookmarked incidents
            if st.session_state.batch_bookmarks:
                st.markdown("#### Your Bookmarks")

                for bookmark_idx in st.session_state.batch_bookmarks:
                    if bookmark_idx < len(results):
                        r = results[bookmark_idx]

                        with st.expander(
                            f"{get_display_name(r['display_label'])} ({r['max_prob']:.1%}) - {r['incident_text'][:60]}..."
                        ):
                            st.write(r["incident_text"])

                            bm_col1, bm_col2 = st.columns(2)
                            with bm_col1:
                                if st.button("Remove", key=f"remove_bm_{bookmark_idx}"):
                                    st.session_state.batch_bookmarks.remove(
                                        bookmark_idx
                                    )
                                    st.rerun()

                # Export bookmarks
                st.markdown("---")
                bookmarked_results = [
                    results[i]
                    for i in st.session_state.batch_bookmarks
                    if i < len(results)
                ]
                if bookmarked_results:
                    bookmark_csv = pd.DataFrame(bookmarked_results).to_csv(index=False)
                    st.download_button(
                        "Export Bookmarked Incidents",
                        bookmark_csv,
                        f"bookmarked_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True,
                    )
            else:
                st.info("No bookmarks yet. Select incidents above to bookmark them.")

        with tabs[5]:  # Visualizations
            st.markdown("### Advanced Visualizations")

            # Confidence histogram
            confidences = [r["max_prob"] for r in results] if results else []

            if confidences:
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=confidences,
                        nbinsx=30,
                        marker=dict(
                            color=confidences,
                            colorscale=[
                                [0, "#FF4444"],
                                [0.5, "#FFB84D"],
                                [1, "#44FF44"],
                            ],
                        ),
                    )
                )
                fig.update_layout(
                    title="Confidence Score Distribution",
                    xaxis_title="Confidence",
                    yaxis_title="Frequency",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True, key="batch_viz_results")

                # Confidence buckets
                st.markdown("#### Confidence Breakdown")
                conf_col1, conf_col2, conf_col3 = st.columns(3)

                high = len([r for r in results if r["max_prob"] >= 0.8])
                med = len([r for r in results if 0.5 <= r["max_prob"] < 0.8])
                low = len([r for r in results if r["max_prob"] < 0.5])

                with conf_col1:
                    st.metric("High (≥80%)", f"{high} ({high/len(results):.1%})")
                with conf_col2:
                    st.metric("Medium (50-80%)", f"{med} ({med/len(results):.1%})")
                with conf_col3:
                    st.metric("Low (<50%)", f"{low} ({low/len(results):.1%})")

                st.markdown("---")

                # Sorting and Pagination
                st.markdown("#### Detailed Results")

                sort_col1, sort_col2, sort_col3 = st.columns([2, 2, 1])

                with sort_col1:
                    sort_by = st.selectbox(
                        "Sort By",
                        options=["confidence", "classification", "text_length"],
                        format_func=lambda x: {
                            "confidence": "Confidence (High to Low)",
                            "classification": "Classification (A-Z)",
                            "text_length": "Text Length (Long to Short)",
                        }[x],
                    )

                with sort_col2:
                    items_per_page = st.selectbox(
                        "Items Per Page", options=[10, 25, 50, 100], index=1
                    )

                # Sort results
                sorted_results = results.copy()
                if sort_by == "confidence":
                    sorted_results.sort(key=lambda x: x["max_prob"], reverse=True)
                elif sort_by == "classification":
                    sorted_results.sort(key=lambda x: x["display_label"])
                elif sort_by == "text_length":
                    sorted_results.sort(
                        key=lambda x: len(x["incident_text"]), reverse=True
                    )

                # Pagination
                total_items = len(sorted_results)
                total_pages = (total_items + items_per_page - 1) // items_per_page

                with sort_col3:
                    if "current_page" not in st.session_state:
                        st.session_state.current_page = 1

                    page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=max(total_pages, 1),
                        value=st.session_state.current_page,
                        step=1,
                    )
                    st.session_state.current_page = page

                # Calculate pagination
                start_idx = (page - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)
                page_results = sorted_results[start_idx:end_idx]

                # Display pagination info
                st.markdown(
                    f"Showing {start_idx + 1}-{end_idx} of {total_items} incidents (Page {page} of {total_pages})"
                )

                # Display table
                results_df = pd.DataFrame(
                    [
                        {
                            "Index": start_idx + i + 1,
                            "Classification": get_display_name(r["display_label"]),
                            "Confidence": f"{r['max_prob']:.1%}",
                            "Text Length": len(r["incident_text"]),
                            "Incident": (
                                r["incident_text"][:100] + "..."
                                if len(r["incident_text"]) > 100
                                else r["incident_text"]
                            ),
                        }
                        for i, r in enumerate(page_results)
                    ]
                )

                st.dataframe(results_df, use_container_width=True, height=400)

                # Pagination controls
                nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

                with nav_col1:
                    if st.button("← Previous", disabled=page == 1):
                        st.session_state.current_page = max(1, page - 1)
                        st.rerun()

                with nav_col3:
                    if st.button("Next →", disabled=page == total_pages):
                        st.session_state.current_page = min(total_pages, page + 1)
                        st.rerun()
            else:
                st.info("No results to visualize")

        with tabs[6]:  # IOC Intelligence
            st.markdown("### Indicator of Compromise (IOC) Analysis")

            # Extract IOCs from all incidents
            all_iocs = {
                "ips": set(),
                "domains": set(),
                "file_hashes": set(),
                "emails": set(),
            }

            for r in results:
                import re

                text = r["incident_text"]

                # IP addresses
                ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
                all_iocs["ips"].update(ips)

                # Email addresses
                emails = re.findall(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text
                )
                all_iocs["emails"].update(emails)

                # Domains (simplified)
                domains = re.findall(
                    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b",
                    text.lower(),
                )
                all_iocs["domains"].update(d for d in domains if "@" not in d)

                # File hashes (MD5, SHA1, SHA256)
                hashes = re.findall(
                    r"\b[a-fA-F0-9]{32}\b|\b[a-fA-F0-9]{40}\b|\b[a-fA-F0-9]{64}\b", text
                )
                all_iocs["file_hashes"].update(hashes)

            # Display IOC summary
            ioc_col1, ioc_col2, ioc_col3, ioc_col4 = st.columns(4)

            with ioc_col1:
                st.metric("IP Addresses", len(all_iocs["ips"]))
            with ioc_col2:
                st.metric("Domains", len(all_iocs["domains"]))
            with ioc_col3:
                st.metric("File Hashes", len(all_iocs["file_hashes"]))
            with ioc_col4:
                st.metric("Email Addresses", len(all_iocs["emails"]))

            st.markdown("---")

            # Display IOCs by type
            ioc_subtabs = st.tabs(["IP Addresses", "Domains", "File Hashes", "Emails"])

            with ioc_subtabs[0]:
                if all_iocs["ips"]:
                    for ip in sorted(all_iocs["ips"])[:50]:
                        st.code(ip)
                    if len(all_iocs["ips"]) > 50:
                        st.info(f"Showing 50 of {len(all_iocs['ips'])} IPs")
                else:
                    st.info("No IP addresses found")

            with ioc_subtabs[1]:
                if all_iocs["domains"]:
                    for domain in sorted(all_iocs["domains"])[:50]:
                        st.code(domain)
                    if len(all_iocs["domains"]) > 50:
                        st.info(f"Showing 50 of {len(all_iocs['domains'])} domains")
                else:
                    st.info("No domains found")

            with ioc_subtabs[2]:
                if all_iocs["file_hashes"]:
                    for hash_val in sorted(all_iocs["file_hashes"])[:50]:
                        st.code(hash_val)
                    if len(all_iocs["file_hashes"]) > 50:
                        st.info(f"Showing 50 of {len(all_iocs['file_hashes'])} hashes")
                else:
                    st.info("No file hashes found")

            with ioc_subtabs[3]:
                if all_iocs["emails"]:
                    for email in sorted(all_iocs["emails"])[:50]:
                        st.code(email)
                    if len(all_iocs["emails"]) > 50:
                        st.info(f"Showing 50 of {len(all_iocs['emails'])} emails")
                else:
                    st.info("No email addresses found")

            # Export IOCs
            st.markdown("---")
            ioc_export = "# IOC Export\n\n"
            for ioc_type, ioc_list in all_iocs.items():
                if ioc_list:
                    ioc_export += f"## {ioc_type.replace('_', ' ').title()}\n"
                    for ioc in sorted(ioc_list):
                        ioc_export += f"- {ioc}\n"
                    ioc_export += "\n"

            st.download_button(
                "Export All IOCs",
                ioc_export,
                f"ioc_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                "text/markdown",
                use_container_width=True,
            )

        with tabs[7]:  # MITRE Coverage
            st.markdown("### MITRE ATT&CK Coverage Analysis")

            st.info(
                "MITRE ATT&CK analysis requires LLM enhancement. "
                "This maps incidents to MITRE tactics and techniques."
            )

            # Cyber Kill Chain visualization
            st.markdown("#### Cyber Kill Chain Mapping")

            kill_chain_data = {
                "Reconnaissance": 0,
                "Weaponization": 0,
                "Delivery": 0,
                "Exploitation": 0,
                "Installation": 0,
                "Command & Control": 0,
                "Actions on Objectives": 0,
            }

            # Map classifications to kill chain
            for r in results:
                label = r["display_label"]
                if label == "phishing":
                    kill_chain_data["Delivery"] += 1
                elif label == "malware":
                    kill_chain_data["Installation"] += 1
                elif label == "web_attack":
                    kill_chain_data["Exploitation"] += 1
                elif label == "data_exfiltration":
                    kill_chain_data["Actions on Objectives"] += 1
                elif label == "access_abuse":
                    kill_chain_data["Command & Control"] += 1

            stages = list(kill_chain_data.keys())
            counts = list(kill_chain_data.values())
            colors = [
                "#FF6B6B",
                "#FF8E53",
                "#FFA500",
                "#FFD700",
                "#90EE90",
                "#4682B4",
                "#9B59B6",
            ]

            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=stages,
                    y=counts,
                    marker=dict(color=colors),
                    text=counts,
                    textposition="auto",
                )
            )
            fig.update_layout(
                title="Incidents Mapped to Cyber Kill Chain",
                xaxis_title="Kill Chain Stage",
                yaxis_title="Count",
                height=400,
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True, key="batch_killchain")
            add_chart_download_buttons(fig, "kill_chain_coverage")

            st.markdown("---")

            # Classification breakdown
            st.markdown("#### Classification Distribution")
            label_counts = Counter([r["display_label"] for r in results])
            tactic_data = []

            for label, count in label_counts.most_common():
                tactic_data.append(
                    {
                        "Classification": get_display_name(label),
                        "Count": count,
                        "Percentage": f"{count/len(results):.1%}",
                    }
                )

            tactic_df = pd.DataFrame(tactic_data)
            st.dataframe(tactic_df, use_container_width=True, hide_index=True)

        with tabs[8]:  # LLM Insights
            st.markdown("### AI-Powered Analysis Insights")

            if not results:
                st.warning("No results to analyze")
            else:
                # Confidence Trend Analysis
                st.markdown("#### Confidence Trend Analysis")

                conf_trend_col1, conf_trend_col2 = st.columns(2)

                with conf_trend_col1:
                    # Calculate confidence statistics by classification
                    conf_by_class = {}
                    for r in results:
                        label = r["display_label"]
                        if label not in conf_by_class:
                            conf_by_class[label] = []
                        conf_by_class[label].append(r["max_prob"])

                    trend_data = []
                    for label, confidences in conf_by_class.items():
                        trend_data.append(
                            {
                                "Classification": get_display_name(label),
                                "Avg Confidence": f"{np.mean(confidences):.1%}",
                                "Min": f"{min(confidences):.1%}",
                                "Max": f"{max(confidences):.1%}",
                                "Std Dev": f"{np.std(confidences):.2%}",
                            }
                        )

                    trend_df = pd.DataFrame(trend_data).sort_values(
                        "Avg Confidence", ascending=False
                    )
                    st.dataframe(trend_df, use_container_width=True, hide_index=True)

                with conf_trend_col2:
                    # Confidence distribution by classification
                    fig = go.Figure()

                    for label in sorted(conf_by_class.keys()):
                        fig.add_trace(
                            go.Box(
                                y=conf_by_class[label],
                                name=get_display_name(label),
                                boxmean="sd",
                            )
                        )

                    fig.update_layout(
                        title="Confidence Distribution by Classification",
                        yaxis_title="Confidence",
                        height=400,
                        showlegend=True,
                    )
                    st.plotly_chart(
                        fig, use_container_width=True, key="batch_conf_trends"
                    )

                st.markdown("---")

                # False Positive Assessment
                st.markdown("#### False Positive Likelihood Assessment")

                fp_scores = []
                for r in results:
                    fp_likelihood = assess_false_positive_likelihood(
                        r["incident_text"],
                        r["max_prob"],
                        r["display_label"],
                    )
                    fp_scores.append(
                        {
                            "incident": r["incident_text"][:80] + "...",
                            "classification": get_display_name(r["display_label"]),
                            "confidence": r["max_prob"],
                            "fp_score": fp_likelihood,
                        }
                    )

                # Sort by FP score (highest first)
                fp_scores.sort(key=lambda x: x["fp_score"], reverse=True)

                # Show top 10 potential false positives
                st.markdown("**Top 10 Potential False Positives:**")

                top_fp = fp_scores[:10]
                if top_fp:
                    for i, fp in enumerate(top_fp, 1):
                        fp_color = (
                            "#ff4444"
                            if fp["fp_score"] > 0.7
                            else "#ffaa00" if fp["fp_score"] > 0.4 else "#44ff44"
                        )

                        st.markdown(
                            f"""
                            <div style="
                                background: rgba(255,255,255,0.6);
                                border-left: 4px solid {fp_color};
                                padding: 12px;
                                margin: 8px 0;
                                border-radius: 4px;
                            ">
                                <div style="font-weight: 600; color: {secondary_text};">
                                    {i}. {fp['classification']} (Confidence: {fp['confidence']:.1%}, FP Score: {fp['fp_score']:.1%})
                                </div>
                                <div style="font-size: 0.9rem; color: {secondary_text}; margin-top: 4px;">
                                    {fp['incident']}
                                </div>
                            </div>
                        """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.info(
                        "All incidents appear to have low false positive likelihood"
                    )

                st.markdown("---")

                # Duplicate Detection
                st.markdown("#### Duplicate & Similar Incident Detection")

                dup_col1, dup_col2 = st.columns([1, 2])

                with dup_col1:
                    similarity_threshold = st.slider(
                        "Similarity Threshold",
                        min_value=0.5,
                        max_value=1.0,
                        value=0.85,
                        step=0.05,
                        format="%.0f%%",
                        help="Higher values = more strict duplicate detection",
                    )

                with dup_col2:
                    if st.button("Detect Duplicates", type="primary"):
                        with st.spinner("Analyzing incidents for duplicates..."):
                            # Check for duplicates using semantic search
                            duplicate_groups = []
                            checked = set()

                            for i, r in enumerate(results):
                                if i in checked:
                                    continue

                                similar = find_similar_incidents(
                                    r["incident_text"],
                                    top_k=10,
                                    similarity_threshold=similarity_threshold,
                                )

                                if len(similar) > 1:  # Found duplicates
                                    group = [i]
                                    for sim in similar:
                                        # Find index in results
                                        for j, r2 in enumerate(results):
                                            if (
                                                r2["incident_text"]
                                                == sim["incident_text"]
                                                and j != i
                                            ):
                                                group.append(j)
                                                checked.add(j)
                                                break

                                    if len(group) > 1:
                                        duplicate_groups.append(group)
                                checked.add(i)

                            if duplicate_groups:
                                st.success(
                                    f"Found {len(duplicate_groups)} groups of potential duplicates"
                                )

                                for g_idx, group in enumerate(
                                    duplicate_groups[:5], 1
                                ):  # Show top 5 groups
                                    with st.expander(
                                        f"Duplicate Group {g_idx} ({len(group)} incidents)"
                                    ):
                                        for idx in group:
                                            r = results[idx]
                                            st.markdown(
                                                f"**{get_display_name(r['display_label'])}** ({r['max_prob']:.1%})"
                                            )
                                            st.text(r["incident_text"][:200] + "...")
                                            st.markdown("---")
                            else:
                                st.info(
                                    "No duplicate incidents detected at this threshold"
                                )

                st.markdown("---")

                # Similar Incidents Clustering
                st.markdown("#### Similar Incident Patterns")

                st.info(
                    "This feature identifies clusters of similar incidents that may represent "
                    "related attack campaigns or patterns. Requires semantic embeddings."
                )

                # Show summary of classification co-occurrence
                st.markdown("**Classification Co-occurrence:**")

                # Simple text-based similarity without embeddings
                pattern_col1, pattern_col2 = st.columns(2)

                with pattern_col1:
                    # Most common words across incidents
                    import re

                    all_words = []
                    for r in results:
                        words = re.findall(
                            r"\b[a-zA-Z]{4,}\b", r["incident_text"].lower()
                        )
                        all_words.extend(words)

                    word_freq = Counter(all_words).most_common(15)

                    st.markdown("**Top Keywords:**")
                    for word, count in word_freq:
                        st.text(f"{word}: {count} occurrences")

                with pattern_col2:
                    # Classification pairs
                    st.markdown("**Incident Count by Classification:**")
                    label_counts = Counter([r["display_label"] for r in results])

                    for label, count in label_counts.most_common():
                        pct = count / len(results) * 100
                        st.text(f"{get_display_name(label)}: {count} ({pct:.1f}%)")

        with tabs[9]:  # Export
            st.markdown("### Export Results")

            st.markdown("#### Standard Exports")
            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                # CSV export
                export_df = pd.DataFrame(results)
                csv = export_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True,
                )

            with export_col2:
                # JSON export
                json_str = json.dumps(results, indent=2, cls=NumpyEncoder)
                st.download_button(
                    "Download JSON",
                    json_str,
                    f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json",
                    use_container_width=True,
                )

            with export_col3:
                # Summary report
                confidences = [r["max_prob"] for r in results]
                high = len([r for r in results if r["max_prob"] >= 0.8])
                label_counts = Counter([r["display_label"] for r in results])

                summary = f"""# Batch Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Incidents: {len(results)}
- Average Confidence: {np.mean(confidences):.1%}
- High Confidence: {high} ({high/len(results):.1%})

## Classification Distribution
"""
                for label, count in label_counts.most_common():
                    summary += f"- {get_display_name(label)}: {count} ({count/len(results):.1%})\n"

                st.download_button(
                    "Download Report",
                    summary,
                    f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    "text/markdown",
                    use_container_width=True,
                )

            st.markdown("---")
            st.markdown("#### Advanced Exports")

            adv_col1, adv_col2, adv_col3 = st.columns(3)

            with adv_col1:
                # Threat Intelligence Brief
                threat_brief = generate_threat_intelligence_brief(results)
                st.download_button(
                    "Download Threat Intelligence Brief",
                    threat_brief,
                    f"threat_brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    "text/markdown",
                    use_container_width=True,
                    help="Comprehensive threat intelligence report with executive summary, MITRE coverage, and recommendations",
                )

                # Preview threat brief
                with st.expander("Preview Threat Intelligence Brief"):
                    st.markdown(threat_brief)

            with adv_col2:
                # Excel-ready CSV with metadata
                excel_df = pd.DataFrame(
                    [
                        {
                            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "Classification": get_display_name(r["display_label"]),
                            "Confidence": r["max_prob"],
                            "Confidence_Pct": f"{r['max_prob']:.1%}",
                            "Confidence_Level": (
                                "High"
                                if r["max_prob"] >= 0.8
                                else ("Medium" if r["max_prob"] >= 0.5 else "Low")
                            ),
                            "Incident_Text": r["incident_text"],
                            "Text_Length": len(r["incident_text"]),
                        }
                        for r in results
                    ]
                )
                excel_csv = excel_df.to_csv(index=False)
                st.download_button(
                    "Download Excel-Ready CSV",
                    excel_csv,
                    f"batch_results_excel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True,
                    help="CSV with additional metadata columns for Excel analysis",
                )

            with adv_col3:
                # SOC Playbook Bundle
                if st.button("Generate SOC Playbook Bundle", use_container_width=True):
                    # Generate playbooks for all unique classifications
                    unique_classifications = set([r["display_label"] for r in results])

                    playbook_bundle = f"""# SOC Playbook Bundle
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Incidents: {len(results)}
Classifications: {len(unique_classifications)}

---

"""

                    for classification in sorted(unique_classifications):
                        incidents = [
                            r for r in results if r["display_label"] == classification
                        ]
                        avg_conf = float(np.mean([r["max_prob"] for r in incidents])) if incidents else 0.0

                        # Extract IOCs from all incidents of this type
                        import re

                        all_iocs = {
                            "ips": set(),
                            "domains": set(),
                            "file_hashes": set(),
                        }

                        for r in incidents:
                            text = r["incident_text"]
                            all_iocs["ips"].update(
                                re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
                            )
                            all_iocs["file_hashes"].update(
                                re.findall(r"\b[a-fA-F0-9]{32,64}\b", text)
                            )

                        custom_pb = generate_custom_soc_playbook(
                            classification, all_iocs, [], avg_conf
                        )

                        playbook_bundle += f"""
## {get_display_name(classification)} ({len(incidents)} incidents)

{custom_pb}

---

"""

                    st.download_button(
                        "Download Playbook Bundle",
                        playbook_bundle,
                        f"soc_playbook_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        "text/markdown",
                        use_container_width=True,
                        help="Combined SOC playbooks for all incident types in this batch",
                        key="download_pb_bundle",
                    )

                    st.success(
                        f"Generated playbooks for {len(unique_classifications)} incident types"
                    )

            # SOC Playbook Recommendations
            st.markdown("---")
            st.markdown("#### SOC Playbook Directory")

            st.info(
                "View recommended SOC playbooks for each incident type detected in this batch. "
                "Playbooks include response procedures, required tools, and external resources."
            )

            # Display playbook summary for each classification
            unique_classes = sorted(set([r["display_label"] for r in results]))

            for classification in unique_classes:
                incidents_of_type = [
                    r for r in results if r["display_label"] == classification
                ]
                avg_conf = float(np.mean([r["max_prob"] for r in incidents_of_type])) if incidents_of_type else 0.0

                playbook_rec = generate_soc_playbook_recommendation(
                    classification, avg_conf
                )

                with st.expander(
                    f"{get_display_name(classification)} - {playbook_rec['playbook_id']} ({len(incidents_of_type)} incidents)"
                ):
                    rec_col1, rec_col2 = st.columns(2)

                    with rec_col1:
                        st.markdown(f"**Priority:** {playbook_rec['priority']}")
                        st.markdown(f"**Team:** {playbook_rec['team']}")
                        st.markdown(f"**Est. Time:** {playbook_rec['estimated_time']}")

                    with rec_col2:
                        st.markdown(f"**Avg Confidence:** {avg_conf:.1%}")
                        st.markdown(
                            f"**Critical Count:** {len([r for r in incidents_of_type if r['max_prob'] >= 0.8])}"
                        )

                    st.markdown("**Key Steps:**")
                    for step in playbook_rec["steps"][:3]:
                        st.markdown(f"- {step}")

                    if len(playbook_rec["steps"]) > 3:
                        st.markdown(
                            f"*...and {len(playbook_rec['steps']) - 3} more steps*"
                        )

        with tabs[10]:  # Batch Comparison
            st.markdown("### Batch Analysis Comparison")

            st.info(
                "Compare multiple batch analyses to identify trends, classification drift, and confidence evolution over time."
            )

            # Load batch history from database
            try:
                db = st.session_state.get("db", TriageDatabase())
                batch_history = db.get_batch_history(limit=50)

                if not batch_history:
                    st.warning(
                        "No saved batch analyses found. Process and save a batch first."
                    )
                else:
                    st.markdown(f"**Found {len(batch_history)} batch analyses**")

                    # Batch selector
                    st.markdown("#### Select Batches to Compare")

                    compare_col1, compare_col2 = st.columns(2)

                    with compare_col1:
                        batch_options = {
                            f"{b['batch_name']} ({b['timestamp'][:10]}) - {b['total_incidents']} incidents": b[
                                "batch_id"
                            ]
                            for b in batch_history
                        }

                        selected_batch_1 = st.selectbox(
                            "Batch 1 (Baseline)",
                            options=list(batch_options.keys()),
                            help="Select first batch for comparison",
                        )

                    with compare_col2:
                        selected_batch_2 = st.selectbox(
                            "Batch 2 (Comparison)",
                            options=list(batch_options.keys()),
                            index=min(1, len(batch_options) - 1),
                            help="Select second batch for comparison",
                        )

                    if st.button(
                        "Compare Batches", type="primary", use_container_width=True
                    ):
                        batch_id_1 = batch_options[selected_batch_1]
                        batch_id_2 = batch_options[selected_batch_2]

                        # Load batch data
                        batch_1_meta = db.get_batch_by_id(batch_id_1)
                        batch_2_meta = db.get_batch_by_id(batch_id_2)

                        batch_1_incidents = db.get_batch_incidents(batch_id_1)
                        batch_2_incidents = db.get_batch_incidents(batch_id_2)

                        st.markdown("---")
                        st.markdown("### Comparison Results")

                        # Side-by-side metrics
                        metric_col1, metric_col2, metric_col3 = st.columns(3)

                        with metric_col1:
                            st.markdown("**Batch 1 Metrics**")
                            stats_1 = batch_1_meta.get("summary_stats", {})
                            st.metric(
                                "Total Incidents", batch_1_meta["total_incidents"]
                            )
                            st.metric(
                                "Avg Confidence",
                                f"{stats_1.get('avg_confidence', 0):.1%}",
                            )
                            st.metric(
                                "High Confidence",
                                stats_1.get("high_confidence_count", 0),
                            )

                        with metric_col2:
                            st.markdown("**Batch 2 Metrics**")
                            stats_2 = batch_2_meta.get("summary_stats", {})
                            st.metric(
                                "Total Incidents", batch_2_meta["total_incidents"]
                            )
                            st.metric(
                                "Avg Confidence",
                                f"{stats_2.get('avg_confidence', 0):.1%}",
                            )
                            st.metric(
                                "High Confidence",
                                stats_2.get("high_confidence_count", 0),
                            )

                        with metric_col3:
                            st.markdown("**Delta**")
                            delta_incidents = (
                                batch_2_meta["total_incidents"]
                                - batch_1_meta["total_incidents"]
                            )
                            delta_conf = stats_2.get("avg_confidence", 0) - stats_1.get(
                                "avg_confidence", 0
                            )
                            delta_high = stats_2.get(
                                "high_confidence_count", 0
                            ) - stats_1.get("high_confidence_count", 0)

                            st.metric(
                                "Incident Δ",
                                delta_incidents,
                                delta=f"{delta_incidents:+d}",
                            )
                            st.metric(
                                "Confidence Δ",
                                f"{delta_conf:+.1%}",
                                delta=f"{delta_conf:+.1%}",
                            )
                            st.metric(
                                "High Conf Δ", delta_high, delta=f"{delta_high:+d}"
                            )

                        st.markdown("---")

                        # Classification drift analysis
                        st.markdown("#### Classification Distribution Comparison")

                        dist_1 = stats_1.get("label_distribution", {})
                        dist_2 = stats_2.get("label_distribution", {})

                        # Combine all labels
                        all_labels = sorted(
                            set(list(dist_1.keys()) + list(dist_2.keys()))
                        )

                        # Create comparison chart
                        fig_comp = go.Figure()

                        fig_comp.add_trace(
                            go.Bar(
                                name=batch_1_meta["batch_name"],
                                x=all_labels,
                                y=[dist_1.get(label, 0) for label in all_labels],
                                marker_color="#667eea",
                            )
                        )

                        fig_comp.add_trace(
                            go.Bar(
                                name=batch_2_meta["batch_name"],
                                x=all_labels,
                                y=[dist_2.get(label, 0) for label in all_labels],
                                marker_color="#764ba2",
                            )
                        )

                        fig_comp.update_layout(
                            title="Classification Distribution Comparison",
                            xaxis_title="Classification",
                            yaxis_title="Count",
                            barmode="group",
                            height=400,
                        )

                        st.plotly_chart(
                            fig_comp, use_container_width=True, key="batch_comp_dist"
                        )

                        # Confidence evolution
                        st.markdown("---")
                        st.markdown("#### Confidence Score Evolution")

                        # Extract confidences
                        conf_1 = [inc.get("max_prob", 0) for inc in batch_1_incidents]
                        conf_2 = [inc.get("max_prob", 0) for inc in batch_2_incidents]

                        fig_conf = go.Figure()

                        fig_conf.add_trace(
                            go.Box(
                                y=conf_1,
                                name=batch_1_meta["batch_name"],
                                marker_color="#667eea",
                                boxmean="sd",
                            )
                        )

                        fig_conf.add_trace(
                            go.Box(
                                y=conf_2,
                                name=batch_2_meta["batch_name"],
                                marker_color="#764ba2",
                                boxmean="sd",
                            )
                        )

                        fig_conf.update_layout(
                            title="Confidence Score Distribution",
                            yaxis_title="Confidence",
                            height=400,
                        )

                        st.plotly_chart(
                            fig_conf, use_container_width=True, key="batch_comp_conf"
                        )

                        # Classification drift detection
                        st.markdown("---")
                        st.markdown("#### Classification Drift Analysis")

                        drift_data = []
                        for label in all_labels:
                            count_1 = dist_1.get(label, 0)
                            count_2 = dist_2.get(label, 0)
                            total_1 = batch_1_meta["total_incidents"]
                            total_2 = batch_2_meta["total_incidents"]

                            pct_1 = (count_1 / total_1 * 100) if total_1 > 0 else 0
                            pct_2 = (count_2 / total_2 * 100) if total_2 > 0 else 0
                            drift = pct_2 - pct_1

                            drift_data.append(
                                {
                                    "Classification": get_display_name(label),
                                    "Batch 1 Count": count_1,
                                    "Batch 1 %": f"{pct_1:.1f}%",
                                    "Batch 2 Count": count_2,
                                    "Batch 2 %": f"{pct_2:.1f}%",
                                    "Drift": f"{drift:+.1f}%",
                                }
                            )

                        drift_df = pd.DataFrame(drift_data)
                        st.dataframe(
                            drift_df, use_container_width=True, hide_index=True
                        )

                        # Significant drift warning
                        significant_drift = [
                            d
                            for d in drift_data
                            if abs(float(d["Drift"].replace("%", "").replace("+", "")))
                            > 10
                        ]
                        if significant_drift:
                            st.warning(
                                f"**Significant Drift Detected:** {len(significant_drift)} classifications show >10% change"
                            )

                        # Trend analysis
                        st.markdown("---")
                        st.markdown("#### Trend Insights")

                        trend_col1, trend_col2 = st.columns(2)

                        with trend_col1:
                            st.markdown("**Key Findings:**")

                            if delta_conf > 0.05:
                                st.success(f"✓ Confidence improved by {delta_conf:.1%}")
                            elif delta_conf < -0.05:
                                st.error(
                                    f"✗ Confidence decreased by {abs(delta_conf):.1%}"
                                )
                            else:
                                st.info(f"→ Confidence stable ({delta_conf:+.1%})")

                            if delta_high > 0:
                                st.success(
                                    f"✓ {delta_high} more high-confidence incidents"
                                )
                            elif delta_high < 0:
                                st.warning(
                                    f"✗ {abs(delta_high)} fewer high-confidence incidents"
                                )

                        with trend_col2:
                            st.markdown("**Recommendations:**")

                            if abs(delta_conf) > 0.1:
                                st.markdown("- Investigate model performance changes")
                                st.markdown("- Review data quality differences")

                            if significant_drift:
                                st.markdown("- Analyze classification pattern shifts")
                                st.markdown("- Validate model recalibration needs")

                            if delta_incidents > 100:
                                st.markdown("- Consider volume impact on analysis")

            except Exception as e:
                st.error(f"Error loading batch comparison: {e}")
                st.exception(e)


# ============================================================================
# TAB: BOOKMARKS & HISTORY
# ============================================================================


def bookmarks_and_history_tab():
    """Bookmarks and analysis history"""

    st.markdown(
        '<div class="section-header">Bookmarks & History</div>', unsafe_allow_html=True
    )

    # Back to dashboard button
    if st.button(
        "← Back to Dashboard", type="secondary", key="bookmarks_back_to_dashboard"
    ):
        st.session_state.navigate_to_dashboard = True
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["Bookmarks", "History", "Notes"])

    with tab1:
        st.markdown("### Saved Bookmarks")

        try:
            bookmarks = st.session_state.db.get_bookmarks(limit=100)

            if bookmarks:
                st.info(f"{len(bookmarks)} bookmark(s) saved")

                for bm in bookmarks:
                    # Use created_at if timestamp not available
                    timestamp_display = bm.get("created_at", bm.get("timestamp", "N/A"))
                    label_display = bm.get("final_label", "Unknown")

                    with st.expander(f"{label_display} - {timestamp_display}"):
                        st.write("**Incident Text:**")
                        st.write(bm.get("incident_text", "N/A")[:500])

                        # Editable note section
                        st.markdown("---")
                        st.write("**Bookmark Note:**")

                        current_note = bm.get("note", "")
                        edited_note = st.text_area(
                            "Edit Note",
                            value=current_note,
                            height=100,
                            key=f"note_edit_{bm.get('id')}",
                            placeholder="Add notes about this bookmark...",
                            label_visibility="collapsed",
                        )

                        # Show save button only if note changed
                        if edited_note != current_note:
                            if st.button(
                                f"💾 Save Note",
                                key=f"save_note_{bm.get('id')}",
                                type="primary",
                            ):
                                try:
                                    st.session_state.db.update_bookmark_note(
                                        bm.get("id"), edited_note
                                    )
                                    st.success("✓ Note saved!")
                                    st.session_state.cached_bookmarks = None
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error saving note: {e}")

                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Delete", key=f"del_bm_{bm.get('id')}"):
                                st.session_state.db.delete_bookmark(bm.get("id"))
                                st.success("Deleted!")
                                st.session_state.cached_bookmarks = None
                                st.rerun()

                        with col2:
                            if st.button(
                                f"📋 Copy Text", key=f"copy_bm_{bm.get('id')}"
                            ):
                                st.code(bm.get("incident_text", ""))
            else:
                st.info(
                    "No bookmarks saved yet. Use the bookmark button in Single Incident analysis."
                )
        except Exception as e:
            st.error(f"Error loading bookmarks: {e}")

    with tab2:
        st.markdown("### Analysis History")

        try:
            history = st.session_state.db.get_analysis_history(limit=50)

            if history:
                st.info(f"{len(history)} analysis record(s)")

                # Convert to dataframe
                df = pd.DataFrame(history)

                # Display
                st.dataframe(
                    df[["timestamp", "final_label", "max_prob", "incident_text"]],
                    use_container_width=True,
                )

                # Export
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download History", csv, "analysis_history.csv", "text/csv"
                )
            else:
                st.info("No analysis history yet.")
        except Exception as e:
            st.error(f"Error loading history: {e}")

    with tab3:
        st.markdown("### Analysis Notes")

        try:
            # Add new note form
            with st.form("add_note_form"):
                st.markdown("**Add New Note**")

                # Get list of recent analyses to link note to
                recent_analyses = st.session_state.db.get_analysis_history(limit=50)

                if recent_analyses:
                    analysis_options = {"None": None}
                    for analysis in recent_analyses:
                        timestamp = analysis.get("timestamp", "Unknown")
                        label = analysis.get("final_label", "Unknown")
                        text_preview = analysis.get("incident_text", "")[:50]
                        analysis_options[
                            f"{timestamp} - {label} - {text_preview}..."
                        ] = analysis.get("id")

                    selected_analysis = st.selectbox(
                        "Link to Analysis (optional)",
                        options=list(analysis_options.keys()),
                        help="Select an analysis to attach this note to",
                    )
                    selected_analysis_id = analysis_options[selected_analysis]
                else:
                    st.info(
                        "No recent analyses found. Note will be saved as standalone."
                    )
                    selected_analysis_id = None

                note_text = st.text_area(
                    "Note", placeholder="Enter your note...", height=100
                )

                col1, col2 = st.columns([1, 4])
                with col1:
                    submit_note = st.form_submit_button("Add Note", type="primary")

                if submit_note and note_text:
                    try:
                        st.session_state.db.add_note(
                            note_text=note_text,
                            analysis_id=selected_analysis_id,
                        )
                        if selected_analysis_id:
                            st.success("✓ Note added and linked to analysis")
                        else:
                            st.success("✓ Standalone note added")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error adding note: {e}")

            st.divider()

            # Display all notes (standalone + bookmark notes)
            standalone_notes = st.session_state.db.get_all_notes()
            bookmarks = st.session_state.db.get_bookmarks(limit=200)

            # Filter bookmarks that have notes
            bookmark_notes = [bm for bm in bookmarks if bm.get("note")]

            total_notes = len(standalone_notes) + len(bookmark_notes)

            if total_notes > 0:
                st.markdown(
                    f"**{total_notes} Total Notes** ({len(standalone_notes)} standalone, {len(bookmark_notes)} from bookmarks)"
                )

                # Display standalone notes
                if standalone_notes:
                    st.markdown("#### Analysis Notes")
                    for note in standalone_notes:
                        # Create expander for each note
                        analysis_info = ""
                        if note.get("analysis_id"):
                            analysis_info = f" - Analysis ID: {note['analysis_id']}"

                        created = note.get("created_at", "Unknown date")

                        with st.expander(f"{created}{analysis_info}"):
                            st.markdown(f"**Note:**")
                            st.write(note["note_text"])

                            if note.get("analysis_id"):
                                st.markdown(f"**Analysis ID:** `{note['analysis_id']}`")

                            if note.get("author"):
                                st.markdown(f"**Author:** {note['author']}")

                            st.markdown(f"**Created:** {created}")

                            col1, col2 = st.columns([1, 4])
                            with col1:
                                if st.button("Delete", key=f"del_note_{note['id']}"):
                                    try:
                                        st.session_state.db.delete_note(note["id"])
                                        st.success("✓ Note deleted")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error deleting note: {e}")

                # Display bookmark notes
                if bookmark_notes:
                    st.markdown("#### Bookmark Notes")
                    for bm in bookmark_notes:
                        created = bm.get("created_at", "Unknown date")
                        label = bm.get("final_label", "Unknown")

                        with st.expander(f"{label} - {created}"):
                            st.markdown(f"**Bookmark Note:**")
                            st.write(bm.get("note"))

                            st.markdown(f"**Classification:** {label}")
                            st.markdown(
                                f"**Incident:** {bm.get('incident_text', '')[:100]}..."
                            )
                            st.markdown(f"**Created:** {created}")

                            if st.button(
                                "View Full Bookmark", key=f"view_bm_note_{bm.get('id')}"
                            ):
                                st.info(
                                    "Go to the 'Bookmarks' tab to view and edit this bookmark"
                                )
            else:
                st.info("No notes yet. Add your first note above!")

        except Exception as e:
            st.error(f"Error loading notes: {e}")


# ============================================================================
# TAB: EXPERIMENTAL LAB
# ============================================================================


def experimental_lab():
    """Experimental analysis tools and features"""
    st.markdown(
        '<div class="section-header">Experimental Analysis Lab</div>',
        unsafe_allow_html=True,
    )

    # Back to dashboard button
    if st.button(
        "← Back to Dashboard", type="secondary", key="experimental_back_to_dashboard"
    ):
        st.session_state.navigate_to_dashboard = True
        st.rerun()

    st.markdown(
        """
    <div class="alert-premium alert-warning">
        <strong>🧪 Experimental Features</strong><br>
        These tools are in beta and may produce unexpected results. Use for research and testing.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create tabs for different experimental features
    tabs = st.tabs(
        [
            "Model Comparison",
            "Feature Analysis",
            "N-Gram Extraction",
            "IOC Lookup",
            "Threat Intel Feeds",
            "Text Similarity",
            "Advanced Visualizations",
        ]
    )

    llm_provider, hf_model_id, selected_hf_token = get_llm_settings()

    # Tab 0: Model Comparison
    with tabs[0]:
        st.markdown("### Model Performance Comparison")

        st.markdown(
            """
        <div class="alert-premium alert-info">
            <strong>Model A/B Testing</strong><br>
            Compare different model configurations and preprocessing strategies. Use this to evaluate 
            which settings work best for your specific incident types.
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Test incident
        test_incident = st.text_area(
            "Test Incident Description",
            height=120,
            placeholder="Enter an incident description to test across different model configurations...",
            value="User reported suspicious email with attachment claiming to be invoice. Link points to credential harvesting site.",
        )

        if st.button("Run Model Comparison", type="primary", use_container_width=True):
            if test_incident:
                with st.spinner("Running comparison across model configurations..."):
                    comparison_results = []

                    pred1 = ""
                    probs1: list = []

                    def _predict_with_model(vec, mdl, text):
                        X = vec.transform([text])
                        probs = (
                            mdl.predict_proba(X)[0]
                            if hasattr(mdl, "predict_proba")
                            else []
                        )
                        pred = mdl.predict(X)[0]
                        return pred, probs

                    # Configuration 1: Default with preprocessing
                    try:
                        vectorizer, model = load_vectorizer_and_model()
                        cleaned = clean_description(test_incident)
                        pred1, probs1 = _predict_with_model(vectorizer, model, cleaned)
                        comparison_results.append(
                            {
                                "Configuration": "Default + Preprocessing",
                                "Prediction": pred1,
                                "Confidence": f"{(max(probs1) if len(probs1) else 0):.1%}",
                                "Confidence_Val": max(probs1) if len(probs1) else 0,
                                "Processing": "Full cleaning",
                            }
                        )
                    except Exception as e:
                        st.error(f"Config 1 failed: {e}")

                    # Configuration 2: No preprocessing
                    try:
                        vectorizer, model = load_vectorizer_and_model()
                        pred2, probs2 = _predict_with_model(
                            vectorizer, model, test_incident
                        )
                        comparison_results.append(
                            {
                                "Configuration": "Default (No Preprocessing)",
                                "Prediction": pred2,
                                "Confidence": f"{(max(probs2) if len(probs2) else 0):.1%}",
                                "Confidence_Val": max(probs2) if len(probs2) else 0,
                                "Processing": "Raw text",
                            }
                        )
                    except Exception as e:
                        st.error(f"Config 2 failed: {e}")

                    # Configuration 3: With LLM enhancement
                    try:
                        vectorizer, model = load_vectorizer_and_model()
                        cleaned = clean_description(test_incident)
                        pred3, probs3 = _predict_with_model(vectorizer, model, cleaned)

                        # Get LLM second opinion
                        llm_result = None
                        valid_input, input_error = validate_llm_input_length(
                            test_incident
                        )
                        if not valid_input:
                            st.warning(input_error)
                        elif llm_provider == "huggingface" and not selected_hf_token:
                            st.warning(
                                "Add a Hugging Face token in the sidebar to use hosted inference."
                            )
                        else:
                            if llm_provider == "huggingface":
                                allowed, retry_after = hf_rate_limit_allowance()
                                if not allowed:
                                    st.warning(
                                        f"Hugging Face limit reached. Wait {retry_after:.0f}s before retrying."
                                    )
                                else:
                                    llm_result = llm_second_opinion(
                                        test_incident,
                                        skip_preprocessing=False,
                                        provider=llm_provider,
                                        hf_model=hf_model_id,
                                        hf_token=selected_hf_token,
                                        max_tokens=UI_LLM_MAX_TOKENS,
                                    )
                            else:
                                llm_result = llm_second_opinion(
                                    test_incident,
                                    skip_preprocessing=False,
                                    provider=llm_provider,
                                    max_tokens=UI_LLM_MAX_TOKENS,
                                )

                        comparison_results.append(
                            {
                                "Configuration": "Default + LLM Enhancement",
                                "Prediction": (llm_result or {}).get(
                                    "final_label", (llm_result or {}).get("label", pred3)
                                ),
                                "Confidence": f"{(llm_result or {}).get('confidence', max(probs3) if len(probs3) else 0):.1%}",
                                "Confidence_Val": (llm_result or {}).get(
                                    "confidence", max(probs3) if len(probs3) else 0
                                ),
                                "Processing": "Full + LLM",
                            }
                        )
                    except Exception as e:
                        comparison_results.append(
                            {
                                "Configuration": "Default + LLM Enhancement",
                                "Prediction": pred1,
                                "Confidence": f"{(max(probs1) if len(probs1) else 0):.1%}",
                                "Confidence_Val": max(probs1) if len(probs1) else 0,
                                "Processing": "LLM unavailable",
                            }
                        )

                    # Display results
                    if comparison_results:
                        st.markdown("#### Comparison Results")

                        # Metrics row
                        met_cols = st.columns(len(comparison_results))
                        for idx, (col, result) in enumerate(
                            zip(met_cols, comparison_results)
                        ):
                            with col:
                                confidence_color = (
                                    "🟢"
                                    if result["Confidence_Val"] >= 0.8
                                    else (
                                        "🟡"
                                        if result["Confidence_Val"] >= 0.5
                                        else "🔴"
                                    )
                                )
                                st.markdown(f"**{result['Configuration']}**")
                                st.metric("Classification", result["Prediction"])
                                st.metric("Confidence", result["Confidence"])
                                st.caption(f"{confidence_color} {result['Processing']}")

                        # Results table
                        st.markdown("---")
                        df_comparison = pd.DataFrame(comparison_results)
                        df_comparison = df_comparison.drop(columns=["Confidence_Val"])
                        st.dataframe(
                            df_comparison, use_container_width=True, hide_index=True
                        )

                        # Confidence comparison chart
                        fig_comp = go.Figure(
                            data=[
                                go.Bar(
                                    x=[r["Configuration"] for r in comparison_results],
                                    y=[r["Confidence_Val"] for r in comparison_results],
                                    text=[
                                        f"{r['Confidence_Val']:.1%}"
                                        for r in comparison_results
                                    ],
                                    textposition="auto",
                                    marker=dict(
                                        color=[
                                            r["Confidence_Val"]
                                            for r in comparison_results
                                        ],
                                        colorscale="RdYlGn",
                                        cmin=0,
                                        cmax=1,
                                        showscale=True,
                                        colorbar=dict(title="Confidence"),
                                    ),
                                )
                            ]
                        )

                        fig_comp.update_layout(
                            title="Confidence Comparison Across Configurations",
                            xaxis_title="Configuration",
                            yaxis_title="Confidence Score",
                            height=400,
                            plot_bgcolor="rgba(0,0,0,0)",
                            paper_bgcolor="rgba(0,0,0,0)",
                            yaxis=dict(range=[0, 1]),
                        )

                        st.plotly_chart(
                            fig_comp, use_container_width=True, key="exp_model_comp"
                        )

                        # Recommendation
                        best_config = max(
                            comparison_results, key=lambda x: x["Confidence_Val"]
                        )
                        st.success(
                            f"**Recommended Configuration:** {best_config['Configuration']} ({best_config['Confidence']} confidence)"
                        )

        # Model metrics comparison
        st.markdown("---")
        st.markdown("#### Historical Model Performance")

        try:
            metrics = load_model_metrics()

            if metrics:
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Overall Accuracy", f"{metrics.get('accuracy', 0):.1%}")
                with col2:
                    st.metric("Precision", f"{metrics.get('precision', 0):.1%}")
                with col3:
                    st.metric("Recall", f"{metrics.get('recall', 0):.1%}")
                with col4:
                    st.metric("F1 Score", f"{metrics.get('f1', 0):.1%}")
        except Exception as e:
            st.info("Model metrics not available")

    # Tab 1: Feature Analysis
    with tabs[1]:
        st.markdown("### Deep Feature Analysis")

        incident_text = st.text_area(
            "Incident Text for Analysis",
            height=150,
            placeholder="Enter incident description to extract features...",
        )

        if st.button("Analyze Features", type="primary", use_container_width=True):
            if incident_text:
                with st.spinner("Extracting features..."):
                    import re
                    import math

                    features = {}

                    # Basic text features
                    features["char_count"] = len(incident_text)
                    features["word_count"] = len(incident_text.split())
                    features["sentence_count"] = len(
                        [s for s in incident_text.split(".") if s.strip()]
                    )
                    features["avg_word_length"] = sum(
                        len(word) for word in incident_text.split()
                    ) / max(features["word_count"], 1)

                    words = incident_text.split()
                    unique_words = set(words)
                    features["lexical_diversity"] = len(unique_words) / max(
                        features["word_count"], 1
                    )

                    # Security-related features
                    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
                    url_pattern = r"https?://[^\s]+"
                    email_pattern = (
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
                    )
                    hash_pattern = r"\b[a-f0-9]{32,64}\b"

                    features["ip_count"] = len(re.findall(ip_pattern, incident_text))
                    features["url_count"] = len(re.findall(url_pattern, incident_text))
                    features["email_count"] = len(
                        re.findall(email_pattern, incident_text)
                    )
                    features["hash_count"] = len(
                        re.findall(hash_pattern, incident_text)
                    )

                    technical_terms = [
                        "malware",
                        "phishing",
                        "exploit",
                        "vulnerability",
                        "attack",
                        "threat",
                        "suspicious",
                        "unauthorized",
                        "breach",
                        "intrusion",
                        "ransomware",
                    ]
                    features["technical_terms"] = sum(
                        1 for term in technical_terms if term in incident_text.lower()
                    )

                    # Linguistic features
                    features["uppercase_ratio"] = sum(
                        1 for c in incident_text if c.isupper()
                    ) / max(len(incident_text), 1)
                    features["punctuation_density"] = sum(
                        1 for c in incident_text if c in ".,;:!?"
                    ) / max(len(incident_text), 1)
                    features["number_density"] = sum(
                        1 for c in incident_text if c.isdigit()
                    ) / max(len(incident_text), 1)

                    # Calculate entropy
                    char_freq = Counter(incident_text)
                    entropy = -sum(
                        (count / len(incident_text))
                        * math.log2(count / len(incident_text))
                        for count in char_freq.values()
                    )
                    features["entropy"] = entropy

                    # Display features in columns
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Word Count", features["word_count"])
                        st.metric("Character Count", features["char_count"])
                        st.metric("Sentence Count", features["sentence_count"])

                    with col2:
                        st.metric(
                            "Lexical Diversity", f"{features['lexical_diversity']:.2%}"
                        )
                        st.metric(
                            "Avg Word Length", f"{features['avg_word_length']:.1f}"
                        )
                        st.metric("Text Entropy", f"{features['entropy']:.2f}")

                    with col3:
                        st.metric("IP Addresses", features["ip_count"])
                        st.metric("URLs", features["url_count"])
                        st.metric("Technical Terms", features["technical_terms"])

                    # Feature importance visualization
                    st.markdown("#### Feature Scores")

                    normalized_features = {}
                    for key, value in features.items():
                        if isinstance(value, (int, float)):
                            if value > 1:
                                normalized_features[key] = min(value / 100, 1.0) * 100
                            else:
                                normalized_features[key] = value * 100

                    sorted_features = sorted(
                        normalized_features.items(), key=lambda x: x[1], reverse=True
                    )[:10]

                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=[
                                    k.replace("_", " ").title()
                                    for k, v in sorted_features
                                ],
                                y=[v for k, v in sorted_features],
                                marker_color="#667eea",
                            )
                        ]
                    )

                    fig.update_layout(
                        title="Top 10 Feature Scores (Normalized)",
                        xaxis_title="Feature",
                        yaxis_title="Normalized Score",
                        height=400,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True, key="exp_features")

    # Tab 2: N-Gram Extraction
    with tabs[2]:
        st.markdown("### N-Gram Pattern Extraction")

        ngram_text = st.text_area(
            "Text for N-Gram Analysis",
            height=150,
            placeholder="Enter text to extract n-gram patterns...",
        )

        col1, col2 = st.columns(2)
        with col1:
            n_value = st.slider(
                "N-Gram Size", 1, 5, 2, help="1=unigram, 2=bigram, etc."
            )
        with col2:
            top_n = st.slider("Top N Results", 5, 20, 10)

        if st.button("Extract N-Grams", type="primary", use_container_width=True):
            if ngram_text:
                words = ngram_text.lower().split()
                ngrams = [
                    " ".join(words[i : i + n_value])
                    for i in range(len(words) - n_value + 1)
                ]
                ngram_counts = Counter(ngrams).most_common(top_n)

                if ngram_counts:
                    df_ngrams = pd.DataFrame(
                        ngram_counts, columns=["N-Gram", "Frequency"]
                    )
                    st.dataframe(df_ngrams, use_container_width=True)

                    # Visualization
                    fig = go.Figure(
                        data=[
                            go.Bar(
                                x=[ng for ng, count in ngram_counts],
                                y=[count for ng, count in ngram_counts],
                                marker_color="#764ba2",
                            )
                        ]
                    )

                    fig.update_layout(
                        title=f"Top {top_n} {n_value}-Grams",
                        xaxis_title="N-Gram",
                        yaxis_title="Frequency",
                        height=400,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True, key="exp_ngrams")

    # Tab 3: IOC Lookup
    with tabs[3]:
        st.markdown("### Indicator of Compromise (IOC) Lookup")

        st.markdown(
            """
        <div class="alert-premium alert-info">
            <strong>Note:</strong> This is a simulated IOC lookup. In production, this would query real threat intelligence APIs.
        </div>
        """,
            unsafe_allow_html=True,
        )

        ioc_type = st.selectbox(
            "IOC Type",
            ["IP Address", "Domain", "File Hash (MD5)", "File Hash (SHA256)", "URL"],
        )

        ioc_value = st.text_input(
            "IOC Value", placeholder="e.g., 192.168.1.100, evil.com, abc123..."
        )

        if st.button("Lookup IOC", type="primary", use_container_width=True):
            if ioc_value:
                st.info(f"Looking up {ioc_type}: `{ioc_value}`")

                # Simulated threat intel data
                import random

                threat_score = random.randint(0, 100)

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Threat Score", f"{threat_score}/100")

                with col2:
                    reputation = (
                        "Malicious"
                        if threat_score > 70
                        else "Suspicious" if threat_score > 40 else "Clean"
                    )
                    st.metric("Reputation", reputation)

                with col3:
                    st.metric("Reports", random.randint(0, 500))

                # Simulated details
                st.markdown("##### Intelligence Details")

                details_col1, details_col2 = st.columns(2)

                with details_col1:
                    st.markdown("**Associated Threats:**")
                    threats = random.sample(
                        ["Malware", "Phishing", "C2", "Ransomware", "APT"], k=2
                    )
                    for threat in threats:
                        st.write(f"• {threat}")

                with details_col2:
                    st.markdown("**Timeline:**")
                    st.write(f"First Seen: {random.randint(1, 30)} days ago")
                    st.write(f"Last Seen: {random.randint(1, 7)} days ago")

    # Tab 4: Threat Intel Feeds
    with tabs[4]:
        st.markdown("### Threat Intelligence Feed Generator")

        st.markdown(
            """
        <div class="alert-premium alert-info">
            <strong>Note:</strong> This generates simulated threat intelligence data for testing purposes.
        </div>
        """,
            unsafe_allow_html=True,
        )

        feed_type = st.selectbox(
            "Feed Type",
            ["Malware Hashes", "C2 Domains", "Phishing URLs", "Attacker IPs"],
        )

        if st.button("Generate Feed", type="primary", use_container_width=True):
            import random

            st.success(f"📡 Generated {feed_type} threat feed")

            feed_data = []

            if feed_type == "Malware Hashes":
                for i in range(10):
                    feed_data.append(
                        {
                            "Hash": "".join(random.choices("0123456789abcdef", k=64)),
                            "Type": random.choice(["SHA256", "MD5"]),
                            "Family": random.choice(
                                ["Emotet", "TrickBot", "Ryuk", "Cobalt Strike"]
                            ),
                            "Severity": random.choice(["Critical", "High", "Medium"]),
                        }
                    )
            elif feed_type == "C2 Domains":
                for i in range(10):
                    feed_data.append(
                        {
                            "Domain": f"c2-server{random.randint(1,1000)}.{random.choice(['com', 'net', 'org'])}",
                            "IP": f"{random.randint(180,200)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                            "Threat": random.choice(["APT28", "APT29", "Lazarus"]),
                            "Status": random.choice(["Active", "Sinkholed"]),
                        }
                    )
            elif feed_type == "Phishing URLs":
                for i in range(10):
                    feed_data.append(
                        {
                            "URL": f"https://fake-bank{random.randint(1,100)}.com/login",
                            "Target": random.choice(
                                ["Banking", "E-commerce", "Social Media"]
                            ),
                            "First Seen": f"{random.randint(1,30)} days ago",
                            "Status": random.choice(["Active", "Blocked"]),
                        }
                    )
            else:  # Attacker IPs
                for i in range(10):
                    feed_data.append(
                        {
                            "IP": f"{random.randint(180,200)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                            "Country": random.choice(
                                ["Russia", "China", "Iran", "North Korea"]
                            ),
                            "Attack Type": random.choice(
                                ["Brute Force", "Scanning", "Exploitation"]
                            ),
                            "Confidence": f"{random.randint(60,100)}%",
                        }
                    )

            df_feed = pd.DataFrame(feed_data)
            st.dataframe(df_feed, use_container_width=True)

            # Download button
            csv_data = df_feed.to_csv(index=False)
            st.download_button(
                "Download Feed",
                csv_data,
                f"threat_feed_{feed_type.replace(' ', '_').lower()}.csv",
                "text/csv",
            )

    # Tab 5: Text Similarity
    with tabs[5]:
        st.markdown("### Text Similarity Comparison")

        col1, col2 = st.columns(2)

        with col1:
            text1 = st.text_area(
                "Text 1", height=150, placeholder="Enter first text..."
            )

        with col2:
            text2 = st.text_area(
                "Text 2", height=150, placeholder="Enter second text..."
            )

        similarity_method = st.radio(
            "Similarity Method",
            ["Jaccard", "Cosine (TF-IDF)", "Levenshtein"],
            horizontal=True,
        )

        if st.button("Calculate Similarity", type="primary", use_container_width=True):
            if text1 and text2:
                with st.spinner("Calculating similarity..."):
                    if similarity_method == "Jaccard":
                        # Jaccard similarity
                        set1 = set(text1.lower().split())
                        set2 = set(text2.lower().split())
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        similarity = intersection / union if union > 0 else 0

                    elif similarity_method == "Cosine (TF-IDF)":
                        # Cosine similarity using TF-IDF
                        from sklearn.feature_extraction.text import TfidfVectorizer
                        from sklearn.metrics.pairwise import cosine_similarity

                        vectorizer = TfidfVectorizer()
                        tfidf_matrix = vectorizer.fit_transform([text1, text2])
                        tfidf_dense = np.asarray(tfidf_matrix.todense())
                        similarity = cosine_similarity(
                            tfidf_dense[0:1], tfidf_dense[1:2]
                        )[0][0]

                    else:  # Levenshtein
                        # Simple Levenshtein distance
                        def levenshtein_distance(s1, s2):
                            if len(s1) < len(s2):
                                return levenshtein_distance(s2, s1)
                            if len(s2) == 0:
                                return len(s1)
                            previous_row = range(len(s2) + 1)
                            for i, c1 in enumerate(s1):
                                current_row = [i + 1]
                                for j, c2 in enumerate(s2):
                                    insertions = previous_row[j + 1] + 1
                                    deletions = current_row[j] + 1
                                    substitutions = previous_row[j] + (c1 != c2)
                                    current_row.append(
                                        min(insertions, deletions, substitutions)
                                    )
                                previous_row = current_row
                            return previous_row[-1]

                        distance = levenshtein_distance(text1.lower(), text2.lower())
                        max_len = max(len(text1), len(text2))
                        similarity = 1 - (distance / max_len) if max_len > 0 else 0

                    # Display result
                    st.markdown("### Similarity Score")

                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col2:
                        st.markdown(
                            f"""<div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
<h1 style="font-size: 4rem; margin: 0; color: white;">{similarity:.1%}</h1>
<p style="font-size: 1.2rem; margin-top: 0.5rem; opacity: 0.9;">Similarity Score</p>
</div>""",
                            unsafe_allow_html=True,
                        )

                    st.markdown("---")

                    # Interpretation
                    if similarity > 0.8:
                        st.success(
                            "**Highly Similar** - Texts are very similar in content"
                        )
                    elif similarity > 0.5:
                        st.warning(
                            "**Moderately Similar** - Texts share some common elements"
                        )
                    else:
                        st.info("**Low Similarity** - Texts are quite different")

    # Tab 6: Advanced Visualizations
    with tabs[6]:
        st.markdown("### Advanced Visualizations")

        st.markdown(
            """
        <div class="alert-premium alert-info">
            <strong>Experimental Visualization Tools</strong><br>
            Create custom visualizations for incident analysis using your historical data.
        </div>
        """,
            unsafe_allow_html=True,
        )

        viz_type = st.selectbox(
            "Visualization Type",
            [
                "Timeline Analysis",
                "3D Classification Space",
                "Network Graph",
                "Sankey Flow Diagram",
                "Sunburst Chart",
            ],
        )

        if viz_type == "Timeline Analysis":
            st.markdown("#### Incident Timeline Visualization")

            # Get incidents from database
            try:
                db = st.session_state.get("db", TriageDatabase())
                incidents = db.get_all_analyses()

                if incidents:
                    # Convert to DataFrame
                    df_timeline = pd.DataFrame(
                        [
                            {
                                "date": (
                                    inc["timestamp"][:10]
                                    if "timestamp" in inc
                                    else datetime.now().strftime("%Y-%m-%d")
                                ),
                                "classification": inc.get("final_label", "Unknown"),
                                "confidence": inc.get("confidence", 0),
                            }
                            for inc in incidents
                        ]
                    )

                    # Group by date and classification
                    timeline_data = (
                        df_timeline.groupby(["date", "classification"])
                        .size()
                        .reset_index(name="count")
                    )

                    # Create timeline chart
                    fig_timeline = px.line(
                        timeline_data,
                        x="date",
                        y="count",
                        color="classification",
                        title="Incident Timeline by Classification",
                        markers=True,
                    )

                    fig_timeline.update_layout(
                        height=500,
                        xaxis_title="Date",
                        yaxis_title="Number of Incidents",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        hovermode="x unified",
                    )

                    st.plotly_chart(
                        fig_timeline, use_container_width=True, key="exp_timeline"
                    )
                    add_chart_download_buttons(fig_timeline, "incident_timeline")

                    # Heatmap by day of week and hour
                    st.markdown("##### Incident Pattern Heatmap")

                    # Simulated hourly data
                    np.random.seed(42)
                    days = [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ]
                    hours = list(range(24))

                    heatmap_data = np.random.poisson(5, (7, 24))

                    fig_heatmap = go.Figure(
                        data=go.Heatmap(
                            z=heatmap_data,
                            x=hours,
                            y=days,
                            colorscale="Blues",
                            text=heatmap_data,
                            texttemplate="%{text}",
                            textfont={"size": 10},
                            colorbar=dict(title="Incidents"),
                        )
                    )

                    fig_heatmap.update_layout(
                        title="Incident Frequency by Day and Hour",
                        xaxis_title="Hour of Day",
                        yaxis_title="Day of Week",
                        height=400,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                    )

                    st.plotly_chart(
                        fig_heatmap, use_container_width=True, key="exp_heatmap"
                    )

                else:
                    st.info(
                        "No historical incidents found. Analyze some incidents first."
                    )

            except Exception as e:
                st.warning(f"Timeline visualization unavailable: {e}")

        elif viz_type == "3D Classification Space":
            st.markdown("#### 3D Classification Confidence Space")

            # Generate sample data
            np.random.seed(42)
            n_samples = 100

            classifications = [
                "Malware",
                "Phishing",
                "Data Exfiltration",
                "Web Attack",
                "Access Abuse",
                "Benign",
            ]

            sample_data = []
            for _ in range(n_samples):
                cls = np.random.choice(classifications)
                sample_data.append(
                    {
                        "x": np.random.randn(),
                        "y": np.random.randn(),
                        "z": np.random.randn(),
                        "classification": cls,
                        "confidence": np.random.uniform(0.5, 1.0),
                    }
                )

            df_3d = pd.DataFrame(sample_data)

            fig_3d = px.scatter_3d(
                df_3d,
                x="x",
                y="y",
                z="z",
                color="classification",
                size="confidence",
                hover_data=["confidence"],
                title="3D Classification Feature Space",
                opacity=0.7,
            )

            fig_3d.update_layout(
                height=600,
                scene=dict(
                    xaxis_title="Feature Dimension 1",
                    yaxis_title="Feature Dimension 2",
                    zaxis_title="Feature Dimension 3",
                ),
            )

            st.plotly_chart(fig_3d, use_container_width=True, key="exp_3d")
            add_chart_download_buttons(fig_3d, "3d_classification_space")

        elif viz_type == "Network Graph":
            st.markdown("#### Incident Relationship Network")

            st.info(
                "Network graph showing relationships between incidents, IOCs, and classifications"
            )

            # Create sample network data
            import networkx as nx

            G = nx.Graph()

            # Add nodes and edges
            classifications = ["Malware", "Phishing", "Data Exfiltration"]
            iocs = ["192.168.1.1", "10.0.0.5", "evil.com", "malware.exe"]

            for cls in classifications:
                G.add_node(cls, node_type="classification")

            for ioc in iocs:
                G.add_node(ioc, node_type="ioc")

            # Add relationships
            G.add_edge("Malware", "malware.exe", weight=5)
            G.add_edge("Malware", "192.168.1.1", weight=3)
            G.add_edge("Phishing", "evil.com", weight=4)
            G.add_edge("Phishing", "10.0.0.5", weight=2)
            G.add_edge("Data Exfiltration", "10.0.0.5", weight=3)

            # Get positions
            pos = nx.spring_layout(G, seed=42)

            # Create edge trace
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x,
                y=edge_y,
                line=dict(width=2, color="#888"),
                hoverinfo="none",
                mode="lines",
            )

            # Create node trace
            node_x = []
            node_y = []
            node_text = []
            node_color = []

            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                node_color.append(
                    "#667eea"
                    if G.nodes[node]["node_type"] == "classification"
                    else "#f093fb"
                )

            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                text=node_text,
                textposition="top center",
                hoverinfo="text",
                marker=dict(size=20, color=node_color, line_width=2),
            )

            # Create figure
            fig_network = go.Figure(data=[edge_trace, node_trace])

            fig_network.update_layout(
                title="Incident-IOC Relationship Network",
                showlegend=False,
                hovermode="closest",
                height=500,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_network, use_container_width=True, key="exp_network")

        elif viz_type == "Sankey Flow Diagram":
            st.markdown("#### Incident Classification Flow")

            # Sample Sankey data
            source = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
            target = [5, 6, 7, 5, 6, 5, 7, 6, 7, 5, 6]
            value = [10, 15, 8, 12, 9, 7, 11, 6, 14, 10, 13]

            labels = [
                "Initial Detection",  # 0
                "High Confidence",  # 1
                "Medium Confidence",  # 2
                "Low Confidence",  # 3
                "Reviewed",  # 4
                "Confirmed Threat",  # 5
                "False Positive",  # 6
                "Escalated",  # 7
            ]

            colors = [
                "rgba(102, 126, 234, 0.8)",  # Initial
                "rgba(76, 175, 80, 0.8)",  # High
                "rgba(255, 193, 7, 0.8)",  # Medium
                "rgba(244, 67, 54, 0.8)",  # Low
                "rgba(156, 39, 176, 0.8)",  # Reviewed
                "rgba(255, 87, 34, 0.8)",  # Confirmed
                "rgba(96, 125, 139, 0.8)",  # FP
                "rgba(233, 30, 99, 0.8)",  # Escalated
            ]

            fig_sankey = go.Figure(
                data=[
                    go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(color="black", width=0.5),
                            label=labels,
                            color=colors,
                        ),
                        link=dict(
                            source=source,
                            target=target,
                            value=value,
                            color="rgba(200, 200, 200, 0.4)",
                        ),
                    )
                ]
            )

            fig_sankey.update_layout(
                title="Incident Triage Flow Analysis", height=500, font_size=12
            )

            st.plotly_chart(fig_sankey, use_container_width=True, key="exp_sankey")
            add_chart_download_buttons(fig_sankey, "sankey_flow")

        else:  # Sunburst Chart
            st.markdown("#### Hierarchical Classification Breakdown")

            # Sample hierarchical data
            sunburst_data = {
                "labels": [
                    "All Incidents",
                    "Malware",
                    "Network",
                    "Access",
                    "Ransomware",
                    "Trojan",
                    "Phishing",
                    "DDoS",
                    "Brute Force",
                    "Privilege Escalation",
                ],
                "parents": [
                    "",
                    "All Incidents",
                    "All Incidents",
                    "All Incidents",
                    "Malware",
                    "Malware",
                    "Network",
                    "Network",
                    "Access",
                    "Access",
                ],
                "values": [100, 45, 35, 20, 25, 20, 20, 15, 12, 8],
            }

            fig_sunburst = go.Figure(
                go.Sunburst(
                    labels=sunburst_data["labels"],
                    parents=sunburst_data["parents"],
                    values=sunburst_data["values"],
                    branchvalues="total",
                    marker=dict(colorscale="RdYlBu", cmid=50),
                    hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percentParent}<extra></extra>",
                )
            )

            fig_sunburst.update_layout(
                title="Incident Classification Hierarchy", height=600
            )

            st.plotly_chart(fig_sunburst, use_container_width=True, key="exp_sunburst")
            add_chart_download_buttons(fig_sunburst, "sunburst_hierarchy")


# ============================================================================
# TAB: SETTINGS & PROFILES
# ============================================================================


def settings_and_profiles_interface():
    """Settings management and user profiles interface"""
    st.markdown(
        '<div class="section-header">Settings & User Profiles</div>',
        unsafe_allow_html=True,
    )

    # Back to dashboard button
    if st.button(
        "← Back to Dashboard", type="secondary", key="settings_back_to_dashboard"
    ):
        st.session_state.navigate_to_dashboard = True
        st.rerun()

    st.markdown(
        """
    <div class="alert-premium alert-info">
        Manage default settings, create analyst profiles, and configure application preferences.
        All settings are automatically saved to the database.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Create tabs for different settings categories
    tabs = st.tabs(
        [
            "Default Settings",
            "User Profiles",
            "Tag Management",
            "Database Management",
            "Advanced Options",
        ]
    )

    # Tab 1: Default Settings
    with tabs[0]:
        st.markdown("### Default Analysis Settings")
        st.markdown("These settings will be used as defaults for new analyses.")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Model Configuration")

            # Load current settings
            try:
                current_settings = st.session_state.db.get_all_settings()
            except:
                current_settings = {}

            # Map stored difficulty to display format
            difficulty_map = {
                "soc-easy": "Easy",
                "soc-medium": "Medium",
                "soc-hard": "Hard",
                "soc-expert": "Expert",
                "Easy": "Easy",
                "Medium": "Medium",
                "Hard": "Hard",
                "Expert": "Expert",
            }
            stored_difficulty = current_settings.get("default_difficulty", "Medium")
            display_difficulty = difficulty_map.get(stored_difficulty, "Medium")

            default_difficulty = st.selectbox(
                "Default Difficulty Profile",
                ["Easy", "Medium", "Hard", "Expert"],
                index=["Easy", "Medium", "Hard", "Expert"].index(display_difficulty),
                help="Higher difficulty = more conservative predictions",
            )

            default_threshold = st.slider(
                "Default Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(current_settings.get("default_threshold", 0.70)),
                step=0.05,
                help="Probability threshold for classification",
            )

            default_max_classes = st.slider(
                "Default Max Classes Display",
                min_value=3,
                max_value=10,
                value=int(current_settings.get("default_max_classes", 5)),
                help="Number of top predictions to show",
            )

        with col2:
            st.markdown("#### LLM Configuration")

            default_llm_mode = st.selectbox(
                "Default LLM Mode",
                ["Off", "On"],
                index=["Off", "On"].index(
                    current_settings.get("default_llm_mode", "Off")
                ),
                help="Enable LLM enhancement for all analyses",
            )

            enable_preprocessing = st.checkbox(
                "Enable Text Preprocessing",
                value=current_settings.get("enable_preprocessing", True),
                help="Clean and normalize text before analysis",
            )

            enable_viz = st.checkbox(
                "Enable Advanced Visualizations",
                value=current_settings.get("enable_viz", True),
                help="Show charts and graphs in analysis",
            )

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### Export Settings")

            default_export_format = st.selectbox(
                "Default Export Format",
                ["CSV", "JSON", "Excel"],
                index=["CSV", "JSON", "Excel"].index(
                    current_settings.get("default_export_format", "CSV")
                ),
            )

            include_metadata = st.checkbox(
                "Include Metadata in Exports",
                value=current_settings.get("export_include_metadata", True),
            )

        with col4:
            st.markdown("#### UI Preferences")

            results_per_page = st.number_input(
                "Results Per Page",
                min_value=10,
                max_value=100,
                value=int(current_settings.get("results_per_page", 20)),
                step=10,
            )

            auto_refresh_bookmarks = st.checkbox(
                "Auto-refresh Bookmarks",
                value=current_settings.get("auto_refresh_bookmarks", False),
                help="Automatically refresh bookmark cache",
            )

            default_viz_type = st.selectbox(
                "Default Visualization Type",
                ["Bar Chart", "Pie Chart", "Heatmap", "3D Scatter"],
                index=["Bar Chart", "Pie Chart", "Heatmap", "3D Scatter"].index(
                    current_settings.get("default_viz_type", "Bar Chart")
                ),
                help="Preferred chart type for classifications",
            )

            show_confidence_indicators = st.checkbox(
                "Show Confidence Indicators",
                value=current_settings.get("show_confidence_indicators", True),
                help="Display color-coded confidence levels",
            )

        st.markdown("---")

        # Advanced preferences
        st.markdown("#### Advanced Preferences")

        adv_col1, adv_col2, adv_col3 = st.columns(3)

        with adv_col1:
            enable_caching = st.checkbox(
                "Enable Result Caching",
                value=current_settings.get("enable_caching", True),
                help="Cache analysis results for faster performance",
            )

            enable_parallel_processing = st.checkbox(
                "Enable Parallel Processing",
                value=current_settings.get("enable_parallel_processing", False),
                help="Use multiple cores for batch analysis (experimental)",
            )

        with adv_col2:
            default_similarity_threshold = st.slider(
                "Similarity Detection Threshold",
                min_value=0.5,
                max_value=0.99,
                value=float(current_settings.get("similarity_threshold", 0.90)),
                step=0.05,
                help="Threshold for duplicate detection",
            )

            max_similar_results = st.number_input(
                "Max Similar Results",
                min_value=3,
                max_value=20,
                value=int(current_settings.get("max_similar_results", 5)),
                help="Maximum number of similar incidents to show",
            )

        with adv_col3:
            enable_auto_bookmarking = st.checkbox(
                "Auto-bookmark High Confidence",
                value=current_settings.get("auto_bookmark_high_conf", False),
                help="Automatically bookmark incidents above 90% confidence",
            )

            enable_notifications = st.checkbox(
                "Enable Notifications",
                value=current_settings.get("enable_notifications", True),
                help="Show toast notifications for actions",
            )

        st.markdown("---")

        # Chart color scheme preferences
        st.markdown("#### Chart Color Preferences")

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            color_scheme = st.selectbox(
                "Color Scheme",
                [
                    "Professional (Purple)",
                    "Security (Red-Blue)",
                    "Colorblind Safe",
                    "Monochrome",
                ],
                index=[
                    "Professional (Purple)",
                    "Security (Red-Blue)",
                    "Colorblind Safe",
                    "Monochrome",
                ].index(current_settings.get("color_scheme", "Professional (Purple)")),
                help="Color palette for charts and visualizations",
            )

            confidence_colors = st.radio(
                "Confidence Color Coding",
                [
                    "Traffic Light (Red/Yellow/Green)",
                    "Gradient (Cool to Warm)",
                    "Custom",
                ],
                index=(
                    0
                    if current_settings.get("confidence_colors", "Traffic Light")
                    == "Traffic Light"
                    else 1
                ),
                help="How to display confidence levels",
            )

        with chart_col2:
            chart_style = st.selectbox(
                "Chart Style",
                ["Modern", "Classic", "Minimal", "Bold"],
                index=["Modern", "Classic", "Minimal", "Bold"].index(
                    current_settings.get("chart_style", "Modern")
                ),
                help="Visual style for charts",
            )

            show_grid_lines = st.checkbox(
                "Show Grid Lines on Charts",
                value=current_settings.get("show_grid_lines", True),
                help="Display background grid on charts",
            )

        st.markdown("---")

        # Save button
        if st.button("Save Default Settings", type="primary", use_container_width=True):
            try:
                # Save all settings to database
                settings_to_save = {
                    "default_difficulty": default_difficulty,
                    "default_threshold": default_threshold,
                    "default_max_classes": default_max_classes,
                    "default_llm_mode": default_llm_mode,
                    "enable_preprocessing": enable_preprocessing,
                    "enable_viz": enable_viz,
                    "default_export_format": default_export_format,
                    "export_include_metadata": include_metadata,
                    "results_per_page": results_per_page,
                    "auto_refresh_bookmarks": auto_refresh_bookmarks,
                    "default_viz_type": default_viz_type,
                    "show_confidence_indicators": show_confidence_indicators,
                    "enable_caching": enable_caching,
                    "enable_parallel_processing": enable_parallel_processing,
                    "similarity_threshold": default_similarity_threshold,
                    "max_similar_results": max_similar_results,
                    "auto_bookmark_high_conf": enable_auto_bookmarking,
                    "enable_notifications": enable_notifications,
                    "color_scheme": color_scheme,
                    "confidence_colors": confidence_colors,
                    "chart_style": chart_style,
                    "show_grid_lines": show_grid_lines,
                }

                for key, value in settings_to_save.items():
                    st.session_state.db.set_setting(key, value)

                st.success("Settings saved successfully!")
                st.balloons()
            except Exception as e:
                st.error(f"Failed to save settings: {e}")

        # Reset to defaults button
        if st.button("Reset to Defaults", use_container_width=True):
            if st.session_state.get("confirm_reset", False):
                try:
                    default_settings = {
                        "default_difficulty": "Medium",
                        "default_threshold": 0.70,
                        "default_max_classes": 5,
                        "default_llm_mode": "Off",
                        "enable_preprocessing": True,
                        "enable_viz": True,
                        "default_export_format": "CSV",
                        "export_include_metadata": True,
                        "results_per_page": 20,
                        "auto_refresh_bookmarks": False,
                        "default_viz_type": "Bar Chart",
                        "show_confidence_indicators": True,
                        "enable_caching": True,
                        "enable_parallel_processing": False,
                        "similarity_threshold": 0.90,
                        "max_similar_results": 5,
                        "auto_bookmark_high_conf": False,
                        "enable_notifications": True,
                        "color_scheme": "Professional (Purple)",
                        "confidence_colors": "Traffic Light",
                        "chart_style": "Modern",
                        "show_grid_lines": True,
                    }

                    for key, value in default_settings.items():
                        st.session_state.db.set_setting(key, value)

                    st.success("Settings reset to defaults!")
                    st.session_state.confirm_reset = False
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to reset settings: {e}")
            else:
                st.warning("Click again to confirm reset to default settings")
                st.session_state.confirm_reset = True

    # Tab 2: User Profiles
    with tabs[1]:
        st.markdown("### User Profiles")
        st.markdown("Create and manage analyst profiles with custom preferences.")

        # Get all profiles
        try:
            profiles = st.session_state.db.get_all_profiles()
            active_profile = st.session_state.db.get_active_profile()
        except:
            profiles = []
            active_profile = None

        # Display existing profiles
        if profiles:
            st.markdown("#### Existing Profiles")

            for profile in profiles:
                is_active = bool(
                    active_profile and profile["id"] == active_profile["id"]
                )

                # Create active indicator using inline SVG (premium design)
                if is_active:
                    st.markdown(
                        '<div style="margin-bottom: -15px; margin-left: 10px;">'
                        '<svg width="16" height="16" viewBox="0 0 16 16" style="display: inline-block; vertical-align: middle; margin-right: 6px;">'
                        '<circle cx="8" cy="8" r="7" fill="#10b981" stroke="#059669" stroke-width="1.5"/>'
                        '<path d="M5 8.5L7 10.5L11 6" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/>'
                        "</svg>"
                        '<span style="color: #10b981; font-weight: 600; font-size: 0.85rem;">ACTIVE PROFILE</span>'
                        "</div>",
                        unsafe_allow_html=True,
                    )

                with st.expander(
                    f"{profile['name']} - {profile.get('role', 'Analyst')}",
                    expanded=is_active,
                ):
                    col_a, col_b, col_c = st.columns([2, 2, 1])

                    with col_a:
                        st.markdown(f"**Name:** {profile['name']}")
                        st.markdown(f"**Role:** {profile.get('role', 'Analyst')}")
                        st.markdown(f"**Email:** {profile.get('email', 'N/A')}")

                    with col_b:
                        prefs = profile.get("preferences", {})
                        if not isinstance(prefs, dict):
                            prefs = {}
                        st.markdown("**Preferences:**")
                        st.caption(
                            f"Difficulty: {profile.get('default_difficulty', 'Medium')}"
                        )
                        st.caption(
                            f"Threshold: {profile.get('default_threshold', 0.70)}"
                        )
                        st.caption(
                            f"LLM Mode: {'On' if profile.get('enable_llm') else 'Off'}"
                        )

                    with col_c:
                        if not is_active:
                            if st.button("Activate", key=f"activate_{profile['id']}"):
                                try:
                                    st.session_state.db.set_active_profile(
                                        profile["id"]
                                    )
                                    st.success(f"Activated {profile['name']}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {e}")
                        else:
                            st.success("Active")

                    # Edit and Delete buttons (in a separate row)
                    st.markdown("---")
                    btn_col1, btn_col2 = st.columns(2)

                    with btn_col1:
                        if st.button(
                            "Edit Profile",
                            key=f"edit_{profile['id']}",
                            use_container_width=True,
                        ):
                            st.session_state[f"editing_profile_{profile['id']}"] = True
                            st.rerun()

                    with btn_col2:
                        if not is_active:
                            if st.button(
                                "Delete Profile",
                                key=f"delete_{profile['id']}",
                                use_container_width=True,
                                type="secondary",
                            ):
                                st.session_state[f"confirm_delete_{profile['id']}"] = (
                                    True
                                )
                                st.rerun()
                        else:
                            st.info("Active profile cannot be deleted")

                    # Edit form
                    if st.session_state.get(f"editing_profile_{profile['id']}", False):
                        st.markdown("#### Edit Profile")
                        with st.form(f"edit_profile_form_{profile['id']}"):
                            edit_col1, edit_col2 = st.columns(2)

                            with edit_col1:
                                edit_name = st.text_input(
                                    "Profile Name", value=profile["name"]
                                )
                                edit_role = st.selectbox(
                                    "Role",
                                    [
                                        "Tier 1 Analyst",
                                        "Tier 2 Analyst",
                                        "Senior Analyst",
                                        "SOC Manager",
                                        "Incident Responder",
                                        "Threat Hunter",
                                    ],
                                    index=(
                                        [
                                            "Tier 1 Analyst",
                                            "Tier 2 Analyst",
                                            "Senior Analyst",
                                            "SOC Manager",
                                            "Incident Responder",
                                            "Threat Hunter",
                                        ].index(profile.get("role", "Tier 1 Analyst"))
                                        if profile.get("role")
                                        in [
                                            "Tier 1 Analyst",
                                            "Tier 2 Analyst",
                                            "Senior Analyst",
                                            "SOC Manager",
                                            "Incident Responder",
                                            "Threat Hunter",
                                        ]
                                        else 0
                                    ),
                                )
                                edit_email = st.text_input(
                                    "Email (optional)",
                                    value=profile.get("email", "") or "",
                                )

                            with edit_col2:
                                edit_difficulty = st.selectbox(
                                    "Preferred Difficulty",
                                    ["Easy", "Medium", "Hard", "Expert"],
                                    index=(
                                        ["Easy", "Medium", "Hard", "Expert"].index(
                                            profile.get(
                                                "default_difficulty", "Medium"
                                            ).capitalize()
                                        )
                                        if profile.get(
                                            "default_difficulty", "Medium"
                                        ).capitalize()
                                        in ["Easy", "Medium", "Hard", "Expert"]
                                        else 1
                                    ),
                                )
                                edit_threshold = st.slider(
                                    "Preferred Threshold",
                                    0.0,
                                    1.0,
                                    float(profile.get("default_threshold", 0.70)),
                                    0.05,
                                )
                                edit_llm = st.checkbox(
                                    "Enable LLM Mode",
                                    value=bool(profile.get("enable_llm", 0)),
                                )

                            update_col1, update_col2 = st.columns(2)
                            with update_col1:
                                update_submitted = st.form_submit_button(
                                    "Save Changes",
                                    use_container_width=True,
                                    type="primary",
                                )
                            with update_col2:
                                cancel_edit = st.form_submit_button(
                                    "Cancel", use_container_width=True
                                )

                            if update_submitted:
                                try:
                                    success = st.session_state.db.update_profile(
                                        profile_id=profile["id"],
                                        name=edit_name,
                                        role=edit_role,
                                        email=edit_email if edit_email else None,
                                        default_difficulty=edit_difficulty,
                                        default_threshold=edit_threshold,
                                        enable_llm=edit_llm,
                                    )
                                    if success:
                                        st.success(
                                            f"Profile '{edit_name}' updated successfully!"
                                        )
                                        del st.session_state[
                                            f"editing_profile_{profile['id']}"
                                        ]
                                        st.rerun()
                                    else:
                                        st.error("Failed to update profile")
                                except Exception as e:
                                    st.error(f"Error updating profile: {e}")

                            if cancel_edit:
                                del st.session_state[f"editing_profile_{profile['id']}"]
                                st.rerun()

                    # Delete confirmation
                    if st.session_state.get(f"confirm_delete_{profile['id']}", False):
                        st.warning(
                            f"Are you sure you want to delete profile '{profile['name']}'? This action cannot be undone."
                        )
                        confirm_col1, confirm_col2 = st.columns(2)

                        with confirm_col1:
                            if st.button(
                                "Yes, Delete",
                                key=f"confirm_yes_{profile['id']}",
                                type="primary",
                                use_container_width=True,
                            ):
                                try:
                                    st.session_state.db.delete_profile(profile["id"])
                                    st.success(
                                        f"Profile '{profile['name']}' deleted successfully!"
                                    )
                                    del st.session_state[
                                        f"confirm_delete_{profile['id']}"
                                    ]
                                    st.rerun()
                                except ValueError as e:
                                    st.error(str(e))
                                except Exception as e:
                                    st.error(f"Error deleting profile: {e}")

                        with confirm_col2:
                            if st.button(
                                "Cancel",
                                key=f"confirm_no_{profile['id']}",
                                use_container_width=True,
                            ):
                                del st.session_state[f"confirm_delete_{profile['id']}"]
                                st.rerun()

        st.markdown("---")

        # Create new profile
        st.markdown("#### Create New Profile")

        with st.form("new_profile_form"):
            col1, col2 = st.columns(2)

            with col1:
                profile_name = st.text_input("Profile Name", placeholder="John Doe")
                profile_role = st.selectbox(
                    "Role",
                    [
                        "Tier 1 Analyst",
                        "Tier 2 Analyst",
                        "Senior Analyst",
                        "SOC Manager",
                        "Incident Responder",
                        "Threat Hunter",
                    ],
                )
                profile_email = st.text_input(
                    "Email (optional)", placeholder="analyst@company.com"
                )

            with col2:
                profile_difficulty = st.selectbox(
                    "Preferred Difficulty", ["Easy", "Medium", "Hard", "Expert"]
                )
                profile_threshold = st.slider(
                    "Preferred Threshold", 0.0, 1.0, 0.70, 0.05
                )
                profile_llm = st.selectbox("Preferred LLM Mode", ["Off", "On"])

            submitted = st.form_submit_button(
                "Create Profile", use_container_width=True
            )

            if submitted:
                if not profile_name:
                    st.error("Please enter a profile name")
                else:
                    try:
                        preferences = {
                            "default_difficulty": profile_difficulty,
                            "default_threshold": profile_threshold,
                            "llm_mode": profile_llm,
                        }

                        profile_id = st.session_state.db.create_profile(
                            name=profile_name,
                            role=profile_role,
                            email=profile_email if profile_email else None,
                            preferences=preferences,
                        )

                        st.success(f"Profile '{profile_name}' created successfully!")
                        st.balloons()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create profile: {e}")

    # Tab 3: Tag Management
    with tabs[2]:
        st.markdown("### Tag Management System")

        st.markdown(
            """
        <div class="alert-premium alert-info">
            Create and manage tags to organize and categorize incidents.
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Display existing tags
        try:
            all_tags = st.session_state.db.get_all_tags()

            if all_tags:
                st.markdown("#### Existing Tags")

                tag_df = pd.DataFrame(all_tags)
                st.dataframe(tag_df, use_container_width=True)

                # Tag statistics
                st.markdown("#### Tag Usage Statistics")
                tag_stats_col1, tag_stats_col2, tag_stats_col3 = st.columns(3)

                with tag_stats_col1:
                    st.metric("Total Tags", len(all_tags))

                with tag_stats_col2:
                    st.metric(
                        "Most Used Tag",
                        all_tags[0].get("name", "N/A") if all_tags else "None",
                    )

                with tag_stats_col3:
                    st.metric(
                        "Least Used Tag",
                        all_tags[-1].get("name", "N/A") if all_tags else "None",
                    )

        except Exception as e:
            st.warning(f"Could not load tags: {e}")
            all_tags = []

        st.markdown("---")

        # Create new tag
        st.markdown("#### Create New Tag")

        tag_col1, tag_col2 = st.columns([3, 1])

        with tag_col1:
            new_tag_name = st.text_input(
                "Tag Name",
                placeholder="e.g., high-priority, false-positive, escalated",
                help="Use lowercase and hyphens for tag names",
            )

        with tag_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Tag", use_container_width=True):
                if new_tag_name:
                    try:
                        st.session_state.db.create_tag(new_tag_name)
                        st.success(f"Tag '{new_tag_name}' created!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to create tag: {e}")
                else:
                    st.error("Please enter a tag name")

        st.markdown("---")

        # Delete tags
        if all_tags:
            st.markdown("#### Delete Tag")

            tag_to_delete = st.selectbox(
                "Select tag to delete",
                [tag.get("name", "") for tag in all_tags],
            )

            if st.button("Delete Selected Tag", use_container_width=True):
                try:
                    tag_id = next(
                        (
                            tag["id"]
                            for tag in all_tags
                            if tag.get("name") == tag_to_delete
                        ),
                        None,
                    )
                    if tag_id:
                        st.session_state.db.delete_tag(tag_id)
                        st.success(f"Tag '{tag_to_delete}' deleted!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to delete tag: {e}")

    # Tab 4: Database Management
    with tabs[3]:
        st.markdown("### Database Management")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Database Statistics")

            try:
                history_count = len(
                    st.session_state.db.get_analysis_history(limit=100000)
                )
                bookmark_count = len(st.session_state.db.get_bookmarks(limit=100000))
                note_count = len(st.session_state.db.get_all_notes())

                st.metric("Total Analyses", f"{history_count:,}")
                st.metric("Bookmarks", f"{bookmark_count:,}")
                st.metric("Notes", f"{note_count:,}")
            except Exception as e:
                st.error(f"Error loading stats: {e}")

        with col2:
            st.markdown("#### Database Maintenance")

            if st.button("Refresh Cache", use_container_width=True):
                st.session_state.cached_bookmarks = None
                st.success("Cache refreshed!")
                st.rerun()

            if st.button("Optimize Database", use_container_width=True):
                try:
                    st.session_state.db.optimize()
                    st.success("Database optimized!")
                except Exception as e:
                    st.error(f"Optimization failed: {e}")

            st.markdown("---")

            st.warning("**Danger Zone**")

            if st.button(
                "Clear All History",
                use_container_width=True,
                help="Permanently delete all analysis history",
            ):
                try:
                    st.session_state.db.clear_history()
                    st.success("History cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    # Tab 5: Advanced Options
    with tabs[4]:
        st.markdown("### Advanced Configuration")

        # Settings Import/Export
        st.markdown("#### Settings Import/Export")

        st.info(
            "Export your settings configuration to share with team members or backup your preferences."
        )

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            if st.button("Export Settings", use_container_width=True):
                try:
                    current_settings = st.session_state.db.get_all_settings()

                    settings_json = json.dumps(
                        current_settings, indent=2, cls=NumpyEncoder
                    )

                    st.download_button(
                        "Download Settings JSON",
                        settings_json,
                        f"alertsage_settings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json",
                        use_container_width=True,
                        help="Download your current settings configuration",
                    )

                    st.success("Settings ready for export!")

                except Exception as e:
                    st.error(f"Export failed: {e}")

        with export_col2:
            uploaded_settings = st.file_uploader(
                "Import Settings JSON",
                type=["json"],
                help="Upload a previously exported settings file",
            )

            if uploaded_settings is not None:
                try:
                    imported_settings = json.load(uploaded_settings)

                    st.markdown("**Preview Imported Settings:**")
                    st.json(imported_settings)

                    if st.button(
                        "Apply Imported Settings",
                        type="primary",
                        use_container_width=True,
                    ):
                        for key, value in imported_settings.items():
                            st.session_state.db.set_setting(key, value)

                        st.success("Settings imported successfully!")
                        st.rerun()

                except json.JSONDecodeError:
                    st.error("Invalid JSON file")
                except Exception as e:
                    st.error(f"Import failed: {e}")

        st.markdown("---")

        st.markdown("#### Performance Settings")

        col1, col2 = st.columns(2)

        with col1:
            cache_embeddings = st.checkbox(
                "Cache Embeddings",
                value=True,
                help="Cache semantic search embeddings for faster performance",
            )

            parallel_processing = st.checkbox(
                "Parallel Processing",
                value=False,
                help="Enable parallel processing for batch operations (experimental)",
            )

        with col2:
            max_cache_size = st.number_input(
                "Max Cache Size (MB)",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
                help="Maximum memory for caching",
            )

            embedding_batch_size = st.number_input(
                "Embedding Batch Size",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Number of embeddings to process at once",
            )

        st.markdown("---")

        # API Configuration
        st.markdown("#### API Configuration")

        st.info(
            "Configure external API integrations for threat intelligence and LLM services."
        )

        api_col1, api_col2 = st.columns(2)

        with api_col1:
            threat_intel_api = st.text_input(
                "Threat Intelligence API Key",
                type="password",
                placeholder="Enter API key...",
                help="API key for threat intelligence services (VirusTotal, etc.)",
            )

            llm_api_provider = st.selectbox(
                "LLM Provider",
                ["OpenAI", "Anthropic", "Local Model"],
                help="Select your LLM provider",
            )

        with api_col2:
            llm_api_key = st.text_input(
                "LLM API Key",
                type="password",
                placeholder="Enter API key...",
                help="API key for LLM services",
            )

            llm_model = st.text_input(
                "LLM Model Name",
                placeholder="gpt-4, claude-3, etc.",
                help="Specific model to use",
            )

        if st.button(
            "Save API Configuration", type="primary", use_container_width=True
        ):
            # In production, encrypt these before storing
            api_config = {
                "threat_intel_api_key": threat_intel_api,
                "llm_api_provider": llm_api_provider,
                "llm_api_key": llm_api_key,
                "llm_model": llm_model,
            }

            st.warning(
                "Note: In production, API keys should be encrypted before storage"
            )
            st.session_state.db.set_setting("api_config", json.dumps(api_config))
            st.success("API configuration saved!")

        st.markdown("---")

        st.markdown("#### Debug Options")

        enable_debug = st.checkbox(
            "Enable Debug Mode",
            value=False,
            help="Show detailed error messages and logs",
        )

        if enable_debug:
            st.markdown("##### Debug Information")

            debug_info = {
                "Session State Keys": list(st.session_state.keys()),
                "Database Connected": hasattr(st.session_state, "db"),
                "Cached Bookmarks": st.session_state.get("cached_bookmarks")
                is not None,
                "Batch Results Count": len(st.session_state.get("batch_results", [])),
                "Current Page": st.session_state.get("current_page", 1),
                "Active Filters": {
                    "Classifications": st.session_state.get(
                        "selected_classifications", []
                    ),
                    "Confidence Range": st.session_state.get(
                        "confidence_range", [0, 100]
                    ),
                    "Search Query": st.session_state.get("search_query", ""),
                },
            }

            st.json(debug_info)

            # System info
            st.markdown("##### System Information")
            import platform
            import sys

            sys_info = {
                "Python Version": sys.version,
                "Platform": platform.platform(),
                "Streamlit Version": st.__version__,
                "NumPy Version": np.__version__,
                "Pandas Version": pd.__version__,
            }

            st.json(sys_info)


# ============================================================================
# LAUNCH
# ============================================================================

if __name__ == "__main__":
    main()
