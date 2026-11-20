"""Manual check: run the demo with a fake Claude and save screenshots of the panel (needs `pip install playwright`).

    python tests/shot.py      -> tests/ui.png, tests/ui-keys.png
"""
import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from conftest import C, FakeClaude, free_port  # noqa: E402

import uvicorn  # noqa: E402
from playwright.async_api import async_playwright  # noqa: E402


async def main():
    C._client = FakeClaude()
    C.S.demo, C.S.port = True, free_port()
    C.DEMO_SPEED, C.POINTS_EVERY = 3.0, 2
    C.TRANSCRIPT_DIR = HERE / "_shot_transcripts"
    server = uvicorn.Server(uvicorn.Config(C.app, host="127.0.0.1", port=C.S.port, log_level="warning"))
    st = asyncio.create_task(server.serve())
    await asyncio.sleep(1)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox"], executable_path=os.getenv("CHROME_PATH") or None)
        pg = await b.new_page(viewport={"width": 1280, "height": 760}, device_scale_factor=1)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto(f"http://127.0.0.1:{C.S.port}/")
        await asyncio.sleep(0.5)
        await pg.screenshot(path=str(HERE / "ui-keys.png"))  # keys dialog opens when keys are missing
        await pg.click("#closeKeys")
        await pg.click("#startBtn")
        await asyncio.sleep(9)
        pg.on("dialog", lambda d: asyncio.ensure_future(d.accept("Sarah, CFO")))
        await pg.click(".seg .who[data-label='Them 1']")
        await asyncio.sleep(0.3)
        assert "Sarah, CFO" in await pg.inner_text("#transcript"), "rename should show"
        await pg.click("details summary")
        await pg.fill("#askInput", "What should I charge for this?")
        await pg.press("#askInput", "Enter")
        await asyncio.sleep(1)
        await pg.screenshot(path=str(HERE / "ui.png"))
        print("page errors:", errs)
        await b.close()
    server.should_exit = True
    await st


asyncio.run(main())
