import os
import sys
import urllib.request
import pytest

# Ensure tests resolve `triage` from THIS checkout's src/, not from a
# stale editable install pointing at a sibling worktree. Without this
# guard the conftest in a git-worktree checkout will silently use the
# main repo's src/ tree (whatever was `pip install -e`'d into the
# venv), which means new modules added on this branch are invisible
# to pytest. Prepending the worktree's src/ to sys.path -- and
# evicting any previously-imported `triage` modules -- forces a clean
# resolution from the local files.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOCAL_SRC = os.path.join(_REPO_ROOT, "src")
if os.path.isdir(os.path.join(_LOCAL_SRC, "triage")) and _LOCAL_SRC not in sys.path:
    sys.path.insert(0, _LOCAL_SRC)
    for _mod in [m for m in list(sys.modules) if m == "triage" or m.startswith("triage.")]:
        del sys.modules[_mod]

DATA_PATH = "data/cyber_incidents_simulated.csv"
# Alternative URLs in order of preference
DATA_URLS = [
    "https://github.com/texasbe2trill/AlertSage/releases/download/v1.0/cyber_incidents_simulated.csv",
    # Fallback: generate minimal test data if download fails
]


def ensure_data():
    """Download dataset if it doesn't exist locally."""
    if not os.path.exists(DATA_PATH):
        print(f"Dataset not found at {DATA_PATH}, downloading...")
        os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

        # Try to download from available URLs
        for url in DATA_URLS:
            try:
                print(f"Attempting to download from {url}")
                urllib.request.urlretrieve(url, DATA_PATH)
                print(f"Dataset downloaded to {DATA_PATH}")
                return
            except Exception as e:
                print(f"Failed to download from {url}: {e}")
                continue

        # If all downloads fail, create minimal test data
        print("All download attempts failed. Creating minimal test data...")
        create_minimal_test_data()


def create_minimal_test_data():
    """Create minimal test data if download fails."""
    import csv

    minimal_data = [
        ["incident_id", "title", "description", "severity", "category"],
        [
            "INC-001",
            "Test Incident",
            "This is a test incident for CI/CD",
            "Medium",
            "Security Alert",
        ],
        [
            "INC-002",
            "Another Test",
            "Another test incident",
            "High",
            "Malware Detection",
        ],
    ]

    with open(DATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(minimal_data)
    print(f"Created minimal test data at {DATA_PATH}")


@pytest.fixture(scope="session", autouse=True)
def setup_data():
    """Pytest fixture to ensure data is available before running tests."""
    ensure_data()
    yield
    # Optional: cleanup code here if needed
