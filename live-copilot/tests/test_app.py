import asyncio, json, sys, os, types, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
os.environ["DISABLE_HOTKEY"] = "1"
os.environ["POINTS_EVERY_SEC"] = "3"
import copilot as C, uvicorn, websockets, urllib.request
ORIG_BROADCAST = C.broadcast
from test_copilot import fake  # reuse fake Claude (runs parsing test on import too)
C._client = fake
C.broadcast = ORIG_BROADCAST
C.S.demo = True
C.S.segments.clear(); C.S.listening = False

async def main():
    server = uvicorn.Server(uvicorn.Config(C.app, host="127.0.0.1", port=8766, log_level="warning"))
    st = asyncio.create_task(server.serve())
    await asyncio.sleep(1)
    html = await asyncio.to_thread(lambda: urllib.request.urlopen("http://127.0.0.1:8766/").read().decode())
    assert "Live Copilot" in html
    msgs = []
    async with websockets.connect("ws://127.0.0.1:8766/ws") as ws:
        async def reader():
            async for m in ws: msgs.append(json.loads(m))
        r = asyncio.create_task(reader())
        await ws.send(json.dumps({"action": "context", "text": "Call with fintech CFO"}))
        await ws.send(json.dumps({"action": "start"}))
        await asyncio.sleep(24)
        await ws.send(json.dumps({"action": "answer", "question": "What should I charge?"}))
        await asyncio.sleep(1)
        await ws.send(json.dumps({"action": "stop"}))
        await asyncio.sleep(1)
        r.cancel()
    types_ = [m["type"] for m in msgs]
    print({t: types_.count(t) for t in set(types_)})
    assert types_[0] == "init"
    assert types_.count("final") == len(C.DEMO_SCRIPT)
    starts = [m for m in msgs if m["type"] == "answer_start"]
    print("answers:", [(m["trigger"], m["question"]) for m in starts])
    assert any(m["trigger"] == "auto" for m in starts)
    assert any(m["trigger"] == "typed" for m in starts)
    assert "points" in types_
    assert any("Transcript saved" in m.get("text", "") for m in msgs)
    last_stream = [c for c in fake.messages.calls if c[0] == "stream"][-1][1]
    assert "Call with fintech CFO" in last_stream["system"] and "What should I charge?" in last_stream["messages"][0]["content"]
    print("saved:", glob.glob(str(HERE.parent / "transcripts" / "*.md")))
    server.should_exit = True; await st
    print("OK  full app demo flow")

asyncio.run(main())
