"""Capture README screenshots of the AlertSage SOC console.

Outputs at 5120x2880 (Apple Studio Display native resolution): a 2560x1440
logical viewport at deviceScaleFactor=2. Dark mode forced. Sidebar hidden
via CSS injection AFTER navigation, so each page renders full-bleed in the
captured image.

Pages captured (one PNG per name in docs/images/):
  investigate Investigate page with the example chips and triage form
  hunt        Hunt page populated with synthetic triage history
  batch       Batch CSV upload page
  bookmarks   Bookmarks page with three pre-bookmarked rows so the
              section shows real saved investigations rather than the
              empty state. The script clears the bookmarks table at
              startup and re-populates it from the first three Hunt
              rows for a clean repeatable capture.
  settings    Settings page

Overview is intentionally NOT captured here. The README hero is the
demo.gif walkthrough recorded by scripts/record_demo_gif.py, and the
Overview section in the page tour relies on prose plus the GIF for
its visual.

Run: IS_HOSTED_DEMO=1 streamlit run app.py --server.port 8765
     python scripts/capture_readme_screenshots.py
"""
import asyncio
import sqlite3
import sys
from pathlib import Path
from playwright.async_api import async_playwright

URL = "http://localhost:8765/"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images"
DB_PATH = ROOT / "data" / "triage.db"

# Logical viewport. With deviceScaleFactor=2 the saved PNG is 5120x2880,
# which matches Apple Studio Display native resolution and gives plenty of
# pixel density for retina rendering in the README.
VP = {"width": 2560, "height": 1440}
SCALE = 2

# CSS injected AFTER navigating to a page, so the click on a sidebar nav
# item still works (we hide the sidebar AFTER it has done its job). Once
# sidebar is hidden we expand stMain to fill the whole viewport so the
# captured image has no empty left rail.
HIDE_SIDEBAR_CSS = """
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }
[data-testid="stMain"] {
    width: 100vw !important;
    max-width: 100vw !important;
    margin-left: 0 !important;
}
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 1900px !important;
    margin: 0 auto !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}
"""

# CSS used only on the Bookmarks shot: open every expander so the
# narrative + case stepper + timeline are visible in the screenshot
# without the user needing to click each one.
EXPAND_BOOKMARKS_CSS = """
[data-testid="stExpander"] details { open: true !important; }
"""

# (output_filename, sidebar nav button accessible name, wait-for-text on
# the destination page so we know the rerun completed)
PAGES = [
    ("investigate.png", "Investigate",   "Run an incident through the AlertSage classifier"),
    ("hunt.png",        "Hunt",          "Search past triage results"),
    ("batch.png",       "Batch",         "incident_text"),
    ("bookmarks.png",   "Bookmarks",     "Saved investigations and analyst notes"),
    ("settings.png",    "Settings",      "LLM provider, models, and Bring Your Own Key"),
]


def reset_and_seed_bookmarks(n: int = 3) -> None:
    """Wipe the bookmarks table and re-seed it from the n most recent
    analysis_history rows. Direct SQL so the capture is independent of
    the running Streamlit process and repeatable across script runs.
    """
    if not DB_PATH.exists():
        print(f"  WARN: {DB_PATH} missing, skipping bookmark reset")
        return
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM bookmarks")
        cur.execute(
            """
            SELECT id, incident_text, final_label
            FROM analysis_history
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (n,),
        )
        rows = cur.fetchall()
        for aid, incident_text, final_label in rows:
            cur.execute(
                """
                INSERT INTO bookmarks (analysis_id, incident_text, final_label, note)
                VALUES (?, ?, ?, ?)
                """,
                (aid, incident_text, final_label, ""),
            )
        conn.commit()
        print(f"  reset bookmarks: {len(rows)} seeded from analysis_history")
    finally:
        conn.close()


async def capture(page, filename: str, nav_label: str | None, wait_text: str):
    print(f"  capturing {filename} ...", end=" ", flush=True)

    # Always start fresh on the Overview landing so each shot is isolated.
    await page.goto(URL, wait_until="networkidle")
    await page.wait_for_selector('[data-testid="stSidebar"]', timeout=20000)
    await page.get_by_text("Mission control").first.wait_for(timeout=20000)
    await page.wait_for_timeout(2000)

    # Navigate to the target page if needed. Streamlit's sidebar buttons
    # have role=button + accessible name, so getByRole is the most stable
    # selector. .first() picks the visible nav button (Streamlit also
    # renders an aria-only duplicate that we don't want).
    if nav_label:
        await page.get_by_role("button", name=nav_label).first.click()
        await page.get_by_text(wait_text).first.wait_for(timeout=20000)
        await page.wait_for_timeout(1500)

    # Bookmarks page: open all expanders so the narrative + case stepper
    # + timeline render inside each card. Without this the screenshot
    # would just show three collapsed expander headers.
    if filename == "bookmarks.png":
        await page.evaluate(
            """() => {
                document.querySelectorAll('details').forEach(d => { d.open = true; });
            }"""
        )
        await page.wait_for_timeout(900)

    # Hide sidebar AFTER navigation completes.
    await page.add_style_tag(content=HIDE_SIDEBAR_CSS)
    await page.wait_for_timeout(700)

    # Measure actual content height (top of stMain to bottom of last
    # element inside the block container) and clip the screenshot to
    # that height. stMain itself has min-height: 100vh which would
    # leave huge empty regions on sparse pages like Investigate.
    rect = await page.evaluate(
        """() => {
            const main = document.querySelector('[data-testid="stMain"]');
            const block = document.querySelector('[data-testid="stMainBlockContainer"]')
                       || document.querySelector('.block-container');
            if (!main || !block) return null;
            const mr = main.getBoundingClientRect();
            const br = block.getBoundingClientRect();
            // Bottom edge: include block-container's full content with a
            // small bottom buffer so we don't slice off the last row.
            const bottom = Math.max(br.bottom + 32, mr.top + 320);
            return {x: 0, y: 0, width: window.innerWidth, height: bottom};
        }"""
    )
    if not rect:
        raise RuntimeError("could not measure content rect")

    out_path = OUT / filename
    await page.screenshot(
        path=str(out_path),
        clip={"x": 0, "y": 0, "width": rect["width"], "height": rect["height"]},
    )
    print(f"OK -> {out_path.name}  ({rect['width']:.0f}x{rect['height']:.0f} CSS)")


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("priming bookmarks ...")
    reset_and_seed_bookmarks(n=3)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport=VP,
            device_scale_factor=SCALE,
            color_scheme="dark",
        )
        page = await context.new_page()
        for filename, nav_label, wait_text in PAGES:
            try:
                await capture(page, filename, nav_label, wait_text)
            except Exception as exc:
                print(f"FAIL ({exc.__class__.__name__}: {exc})")
        await browser.close()


asyncio.run(main())
