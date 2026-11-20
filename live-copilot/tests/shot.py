import asyncio, sys, os
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
os.environ["DISABLE_HOTKEY"] = "1"; os.environ["POINTS_EVERY_SEC"] = "4"
import copilot as C, uvicorn
ORIG = C.broadcast
from test_copilot import fake
C._client = fake; C.broadcast = ORIG; C.S.demo = True; C.S.segments.clear()
from playwright.async_api import async_playwright

async def main():
    server = uvicorn.Server(uvicorn.Config(C.app, host="127.0.0.1", port=8767, log_level="warning"))
    st = asyncio.create_task(server.serve()); await asyncio.sleep(1)
    async with async_playwright() as p:
        b = await p.chromium.launch(args=["--no-sandbox"])
        pg = await b.new_page(viewport={"width": 1280, "height": 760})
        errs = []; pg.on("pageerror", lambda e: errs.append(str(e)))
        await pg.goto("http://127.0.0.1:8767/"); await asyncio.sleep(0.5)
        await pg.click("#startBtn"); await asyncio.sleep(20)
        await pg.fill("#askInput", "What should I charge for this?"); await pg.press("#askInput", "Enter")
        await asyncio.sleep(1)
        await pg.screenshot(path=str(HERE / "ui.png"))
        print("page errors:", errs)
        await b.close()
    server.should_exit = True; await st

asyncio.run(main())
