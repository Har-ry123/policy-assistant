"""Records docs/demo.gif by driving the real website in a real browser.

    pip install -r requirements-demo.txt
    python -m demo.make_demo

It starts the server on a spare port, opens the page in Microsoft Edge (installed on Windows) or
Chromium, clicks through a short story, takes a screenshot at each step and joins them into a GIF.
It calls the real model once per question, so it spends a few thousand tokens of your Groq quota.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "demo.gif"
FRAMES = ROOT / "demo" / "_frames"
PORT = 8123
URL = f"http://127.0.0.1:{PORT}/"
WIDTH, HEIGHT = 1180, 760
BOT_ROWS = "document.querySelectorAll('#log .row.bot').length"


def start_server() -> subprocess.Popen:
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.api:app_factory", "--factory", "--port", str(PORT)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(90):
        try:
            if httpx.get(URL + "health", timeout=2).status_code == 200:
                return proc
        except Exception:
            time.sleep(1)
    proc.terminate()
    raise SystemExit("The server did not start.")


def and_wait_for_reply(page, action) -> None:
    """Run `action` (a click or key press), then wait for exactly one new assistant message and no typing dots."""
    before = page.evaluate(BOT_ROWS)
    action()
    page.wait_for_function(
        f"n => {BOT_ROWS} >= n && !document.querySelector('#log .typing')", arg=before + 1, timeout=90_000)
    page.wait_for_timeout(500)


def ask(page, role: str, question: str) -> None:
    page.click(f'#roleSeg [data-role="{role}"]')
    page.fill("#q", question)
    and_wait_for_reply(page, lambda: page.keyboard.press("Enter"))


def card(page, index: int) -> None:
    and_wait_for_reply(page, lambda: page.click(f"#cards .card >> nth={index}"))


def main() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    for old in FRAMES.glob("*.png"):
        old.unlink()
    proc = start_server()
    shots: list[tuple[Path, float]] = []

    def snap(page, name: str, seconds: float = 2.4) -> None:
        path = FRAMES / f"{len(shots):02d}_{name}.png"
        page.screenshot(path=str(path))
        shots.append((path, seconds))

    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="msedge")
            except Exception:
                browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, color_scheme="dark")
            page.goto(URL)
            page.wait_for_selector("#cards .card")
            page.wait_for_function("document.getElementById('statusText').innerText.startsWith('Online')")
            page.wait_for_timeout(600)
            snap(page, "welcome", 2.6)

            card(page, 0)                                                     # annual leave question
            page.click("#log .row.bot .cite >> nth=0")                        # open the cited source
            page.wait_for_timeout(400)
            snap(page, "answer_with_source", 3.0)

            page.click("#newChat")
            ask(page, "", "What is the salary band for level 3?")             # employee: refused
            snap(page, "employee_refused", 2.8)
            ask(page, "hr", "What is the salary band for level 3?")           # HR: allowed
            snap(page, "hr_allowed", 3.0)

            page.click("#newChat")
            ask(page, "engineering", "How fast must we respond to a SEV1 incident?")   # engineer: allowed
            snap(page, "engineer_allowed", 2.8)
            ask(page, "hr", "How fast must we respond to a SEV1 incident?")            # HR: refused
            snap(page, "hr_refused_engineering", 2.8)

            page.click("#newChat")
            page.click("#roleSeg [data-role='']")
            card(page, 6)                                                     # request leave: approval card
            snap(page, "approval_card", 3.0)
            page.click(".approval .btn.ok")
            page.wait_for_selector(".result")
            page.wait_for_timeout(500)
            snap(page, "approved", 2.8)

            page.click("#newChat")
            card(page, 7)                                                     # prompt-injection attack
            snap(page, "attack_blocked", 2.8)

            page.click("#newChat")
            card(page, 0)
            page.evaluate("document.querySelector('#log details.trace').open = true")
            page.wait_for_timeout(300)
            snap(page, "timeline", 3.2)
            browser.close()
    finally:
        proc.terminate()

    images = [Image.open(path).convert("RGB") for path, _ in shots]
    target_w = 960
    images = [im.resize((target_w, int(im.height * target_w / im.width)), Image.LANCZOS) for im in images]
    pal = [im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for im in images]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pal[0].save(OUT, save_all=True, append_images=pal[1:], duration=[int(s * 1000) for _, s in shots],
                loop=0, optimize=True, disposal=2)
    for path, _ in shots:
        path.unlink()
    FRAMES.rmdir()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, {len(shots)} frames)")


if __name__ == "__main__":
    main()
