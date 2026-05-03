"""Record a 60-second walkthrough GIF of the AlertSage SOC console.

Drives a live Streamlit instance via Playwright, records the browser
window as webm, then converts to an optimized dark-mode GIF using
ffmpeg's two-pass palette filter.

The walkthrough:
  ~0s   Mission control loads (KPIs, charts, threat feed already populated
        by IS_HOSTED_DEMO=1 auto-seed)
  ~6s   Click Investigate, paste a Phishing example, hit Triage, wait for
        the LLM rationale to come back from Hugging Face
  ~28s  Click Hunt, the populated triage history table
  ~38s  Click Batch, the upload pane
  ~46s  Click Settings, BYOK fields and provider config
  ~58s  Back to Overview to bookend
  ~60s  Stop

This script manages the Streamlit subprocess itself so the LLM provider
token exported in the recorder's shell propagates correctly. Trying to
record against a pre-existing Streamlit started in a different shell
will look fine until the Triage call fires and the LLM rationale comes
back as the "LLM assist is not configured" placeholder.

Prerequisites:
  - Export ONE provider token in the shell you run this from. Any of:
        export ANTHROPIC_API_KEY=<your-token>     # routes to Anthropic
        export OPENAI_API_KEY=<your-token>        # routes to OpenAI
        export HF_TOKEN=<your-token>              # routes to Hugging Face
    The app's _default_provider() picks the first one it finds in
    Anthropic > OpenAI > HF order, so you do not need to also flip the
    Settings radio in the recording.
  - ffmpeg installed (brew install ffmpeg).
  - Playwright installed (already in requirements-dev.txt for dev work).

Output: docs/images/demo.gif
"""
import asyncio
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from playwright.async_api import async_playwright

URL = "http://localhost:8765/"
PORT = 8765
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "images"
OUT_GIF = OUT_DIR / "demo.gif"
VIDEO_DIR = ROOT / ".cache" / "demo-recording"

# 1920x1080 logical at scale 1 records as 1920x1080 webm. We then
# downscale to 1100 wide via ffmpeg so the final GIF is GitHub-friendly,
# but recording at full HD means text and pills stay sharp instead of
# turning into a blurry 720p mush. Tall enough to show the full result
# card on Investigate without aggressive scrolling.
VP = {"width": 1920, "height": 1080}

# Hide the sidebar for the ENTIRE recording. The user wanted a clean
# main-content recording without the SOC nav rail bleeding into every
# frame. Nav clicks still work because Streamlit keeps the buttons in
# the DOM under display:none, and we click them via JS event dispatch
# which bypasses visibility checks (Playwright's standard click fails
# on display:none elements). The block-container max-width is bumped
# so the wider HD viewport doesn't leave huge empty rails on either
# side of the content.
HIDE_SIDEBAR_CSS = """
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stMain"] {
    width: 100vw !important;
    max-width: 100vw !important;
    margin-left: 0 !important;
}
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 1700px !important;
    margin: 0 auto !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
"""


def kill_existing_streamlit():
    """Best-effort kill any streamlit serving our port. We use pkill -f
    rather than going through psutil so this stays dependency-free."""
    subprocess.run(
        ["pkill", "-f", f"streamlit run.*--server.port {PORT}"],
        check=False,
    )
    # Also catch streamlits started without explicit port (defaults to 8501,
    # but our recorder launches with --server.port so be conservative and
    # also kill anything matching app.py).
    subprocess.run(
        ["pkill", "-f", f"streamlit run.*app.py"],
        check=False,
    )
    time.sleep(2)


def wait_for_streamlit(timeout: int = 60) -> bool:
    """Poll the Streamlit port until it responds 200 or we time out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(URL, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(1)
    return False


def launch_streamlit_with_env() -> subprocess.Popen:
    """Start Streamlit as a child of THIS process so it inherits our
    environment (in particular HF_TOKEN). Returns the Popen so we can
    terminate it after recording.
    """
    env = os.environ.copy()
    env.setdefault("IS_HOSTED_DEMO", "1")
    # Find the streamlit binary. Prefer the project venv if present.
    venv_streamlit = ROOT / ".venv" / "bin" / "streamlit"
    streamlit_bin = str(venv_streamlit) if venv_streamlit.exists() else "streamlit"
    return subprocess.Popen(
        [
            streamlit_bin, "run", "app.py",
            "--server.port", str(PORT),
            "--server.headless", "true",
            "--server.runOnSave", "false",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # New process group so we can clean up cleanly on shutdown.
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )


def stop_streamlit(proc: subprocess.Popen):
    if proc.poll() is not None:
        return
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except Exception:
            pass


async def click_nav(page, label: str, wait_text: str):
    """Click a sidebar nav button via direct DOM dispatch.

    The sidebar is hidden via CSS for the duration of the recording, so
    Playwright's standard click would refuse (element-not-visible). We
    pick the largest-bbox button matching the accessible name (Streamlit
    renders an aria-only 0x0 duplicate alongside each visible button)
    and call .click() on it directly. Streamlit's React onClick handler
    fires on the synthetic event, the rerun starts.
    """
    await page.evaluate(
        """(label) => {
            const buttons = Array.from(document.querySelectorAll('[data-testid="stSidebar"] button'));
            const match = buttons
                .filter(b => (b.innerText || '').trim() === label)
                .sort((a, z) => {
                    const ar = a.getBoundingClientRect();
                    const zr = z.getBoundingClientRect();
                    return (zr.width * zr.height) - (ar.width * ar.height);
                })[0];
            if (!match) throw new Error('nav button not found: ' + label);
            match.click();
        }""",
        label,
    )
    await page.get_by_text(wait_text).first.wait_for(timeout=20000)


async def hide_sidebar(page):
    """Inject the hide-sidebar style tag. Idempotent — if already
    injected the rule list just re-applies, no harm."""
    await page.add_style_tag(content=HIDE_SIDEBAR_CSS)


async def record(page):
    """Walkthrough script. Total target runtime around 60 seconds."""
    # Boot: load Overview, wait for the dashboard to populate, then hide
    # the sidebar before any frames matter. The viewer never sees the
    # nav rail in the recording.
    await page.goto(URL, wait_until="networkidle")
    await page.wait_for_selector('[data-testid="stSidebar"]', timeout=20000)
    await page.get_by_text("Mission control").first.wait_for(timeout=20000)
    await hide_sidebar(page)
    await page.wait_for_timeout(2500)

    # Overview: scroll to show the full dashboard (KPIs -> events chart
    # -> classifier confidence -> severity donut -> MITRE coverage ->
    # top classifications -> live tail).
    await page.evaluate("() => window.scrollTo({top: 800, behavior: 'smooth'})")
    await page.wait_for_timeout(2500)
    await page.evaluate("() => window.scrollTo({top: 1600, behavior: 'smooth'})")
    await page.wait_for_timeout(2500)
    await page.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")
    await page.wait_for_timeout(800)

    # Investigate: pick Phishing example, hit Triage, let the LLM
    # rationale render, then scroll through the full result card.
    await click_nav(page, "Investigate", "Run an incident through the AlertSage classifier")
    await hide_sidebar(page)  # re-assert in case Streamlit's rerun reset our injection
    await page.wait_for_timeout(1500)
    await page.get_by_role("button", name="Phishing").first.click()
    await page.wait_for_timeout(1200)
    await page.get_by_role("button", name="Triage").first.click()
    # Wait for the result card head to mount. Use the AS-NNNNNN id as
    # the signal because it's the first thing render_analysis_result
    # paints and it's unambiguous (no false matches earlier on the
    # page).
    try:
        await page.locator('text=/AS-\\d{6}/').first.wait_for(timeout=18000)
    except Exception:
        pass
    await page.wait_for_timeout(2500)
    # The result card is long: head + case stepper + kill chain + IOCs
    # + timeline + class probabilities + LLM rationale + playbook.
    # Scroll through it in three steps so each section gets ~3 seconds
    # of screen time.
    await page.evaluate("() => window.scrollTo({top: 600, behavior: 'smooth'})")
    await page.wait_for_timeout(3000)
    await page.evaluate("() => window.scrollTo({top: 1400, behavior: 'smooth'})")
    await page.wait_for_timeout(3000)
    await page.evaluate("() => window.scrollTo({top: 2200, behavior: 'smooth'})")
    await page.wait_for_timeout(3000)
    await page.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")
    await page.wait_for_timeout(800)

    # Hunt: populated interactive table with View / Bookmark per row.
    await click_nav(page, "Hunt", "Search past triage results")
    await hide_sidebar(page)
    await page.wait_for_timeout(2500)
    await page.evaluate("() => window.scrollTo({top: 600, behavior: 'smooth'})")
    await page.wait_for_timeout(2500)
    await page.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")
    await page.wait_for_timeout(500)

    # Batch: upload pane.
    await click_nav(page, "Batch", "incident_text")
    await hide_sidebar(page)
    await page.wait_for_timeout(3500)

    # Settings: BYOK fields, provider radio, demo generator panel.
    await click_nav(page, "Settings", "LLM provider, models, and Bring Your Own Key")
    await hide_sidebar(page)
    await page.wait_for_timeout(3000)
    await page.evaluate("() => window.scrollTo({top: 500, behavior: 'smooth'})")
    await page.wait_for_timeout(2500)
    await page.evaluate("() => window.scrollTo({top: 0, behavior: 'smooth'})")
    await page.wait_for_timeout(800)

    # Bookend on Overview.
    await click_nav(page, "Overview", "Mission control")
    await hide_sidebar(page)
    await page.wait_for_timeout(2500)


async def run_recording():
    print("recording walkthrough...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VP,
            color_scheme="dark",
            record_video_dir=str(VIDEO_DIR),
            record_video_size=VP,
        )
        page = await context.new_page()
        await record(page)
        await context.close()
        await browser.close()


def convert_webm_to_gif():
    webms = sorted(VIDEO_DIR.glob("*.webm"))
    if not webms:
        raise SystemExit(f"no webm produced in {VIDEO_DIR}")
    webm = webms[-1]
    print(f"recorded {webm.name}, converting to GIF...")
    palette = VIDEO_DIR / "palette.png"
    # Two-pass palette: pass 1 builds an optimized 256-color palette from
    # the dark UI; pass 2 quantizes against it. fps=12 keeps motion
    # legible while halving file size vs 24fps. Width 1100 matches
    # GitHub's inline render comfortably.
    common = "fps=12,scale=1100:-1:flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(webm), "-vf",
         f"{common},palettegen=stats_mode=diff",
         "-frames:v", "1", str(palette)],
        check=True, capture_output=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(webm), "-i", str(palette), "-lavfi",
         f"{common} [v]; [v][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
         str(OUT_GIF)],
        check=True, capture_output=True,
    )
    size_mb = OUT_GIF.stat().st_size / 1024 / 1024
    print(f"OK -> {OUT_GIF.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    if size_mb > 9.5:
        print("WARNING: GIF is over 9.5 MB. GitHub's inline image limit is 10 MB.")
        print("         Re-run with a shorter walkthrough or smaller fps/scale.")


def main():
    token_env_keys = (
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
        "HF_TOKEN", "TRIAGE_HF_TOKEN",
    )
    found = [k for k in token_env_keys if os.environ.get(k)]
    if not found:
        raise SystemExit(
            "No LLM provider token in this shell. Triage in the recording "
            "would fall back to the 'not configured' placeholder.\n"
            "Export one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN.\n"
            "  example: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    print(f"using provider token from env: {found[0]}")
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH. Install with: brew install ffmpeg")

    if VIDEO_DIR.exists():
        shutil.rmtree(VIDEO_DIR)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("killing any existing Streamlit on this port...")
    kill_existing_streamlit()
    print(f"launching Streamlit with {found[0]} propagated from this shell...")
    proc = launch_streamlit_with_env()
    try:
        if not wait_for_streamlit(timeout=60):
            stop_streamlit(proc)
            raise SystemExit(
                "Streamlit failed to come up on http://localhost:8765 within 60s. "
                "Try running it manually with the same env to see the error: "
                f"HF_TOKEN=$HF_TOKEN IS_HOSTED_DEMO=1 streamlit run app.py "
                f"--server.port {PORT} --server.headless true"
            )
        print("Streamlit is up. Starting recording.")
        # Tiny extra wait so the heavy first-paint settles.
        time.sleep(3)
        asyncio.run(run_recording())
        convert_webm_to_gif()
    finally:
        print("stopping Streamlit...")
        stop_streamlit(proc)


if __name__ == "__main__":
    sys.exit(main() or 0)
