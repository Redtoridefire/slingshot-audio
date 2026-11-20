"""End-to-end: the real FastAPI app in demo mode, driven over its websocket like the browser panel does."""
import asyncio
import json
import urllib.error
import urllib.request

import pytest
import uvicorn
import websockets
from conftest import C, content_text, free_port


@pytest.fixture
async def server(fake_claude, monkeypatch):
    port = free_port()
    C.S.demo, C.S.port = True, port
    monkeypatch.setattr(C, "DEMO_SPEED", 4.0)
    monkeypatch.setattr(C, "POINTS_EVERY", 1)
    srv = uvicorn.Server(uvicorn.Config(C.app, host="127.0.0.1", port=port, log_level="warning"))
    task = asyncio.create_task(srv.serve())
    while not srv.started:
        await asyncio.sleep(0.05)
    yield port
    srv.should_exit = True
    await asyncio.wait_for(task, 10)


async def collect(ws, into):
    async for m in ws:
        into.append(json.loads(m))


async def test_demo_flow(server, fake_claude):
    port = server
    html = await asyncio.to_thread(lambda: urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode())
    assert "Live Copilot" in html
    msgs = []
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws", origin=f"http://127.0.0.1:{port}") as ws:
        reader = asyncio.create_task(collect(ws, msgs))
        await ws.send(json.dumps({"action": "context", "text": "Call with fintech CFO"}))
        await ws.send(json.dumps({"action": "start"}))
        await asyncio.sleep(6.5)
        await ws.send(json.dumps({"action": "answer", "question": "What should I charge?"}))
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"action": "stop"}))
        await asyncio.sleep(0.5)
        reader.cancel()
    types_ = [m["type"] for m in msgs]
    assert types_[0] == "init" and "keys" in msgs[0]
    assert types_.count("final") == len(C.DEMO_SCRIPT)
    triggers = [m["trigger"] for m in msgs if m["type"] == "answer_start"]
    assert "auto" in triggers and "typed" in triggers
    assert "points" in types_ and "cost" in types_ and "answer_meta" in types_
    auto_meta = [m for m in msgs if m["type"] == "answer_meta" and m["from"] == "speech"]
    assert auto_meta, "auto answers report latency from end of speech"
    assert any("Transcript saved" in m.get("text", "") for m in msgs)
    last = fake_claude.messages.of("stream")[-1]
    assert "Call with fintech CFO" in last["system"][0]["text"]
    assert "What should I charge?" in content_text(last)
    assert list(C.TRANSCRIPT_DIR.glob("*.md"))
    await asyncio.wait_for(C.S.recap_task, 3)
    assert list(C.TRANSCRIPT_DIR.glob("*.summary.md"))


async def test_rejects_other_websites(server):
    port = server
    with pytest.raises(websockets.exceptions.InvalidStatus):
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws", origin="https://evil.example"):
            pass

    def get_with_bad_host():
        req = urllib.request.Request(f"http://127.0.0.1:{port}/", headers={"Host": f"evil.example:{port}"})
        return urllib.request.urlopen(req)

    with pytest.raises(urllib.error.HTTPError) as e:
        await asyncio.to_thread(get_with_bad_host)
    assert e.value.code == 403


async def test_keys_saved_but_never_sent_back(server, monkeypatch):
    port = server
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "")
    monkeypatch.setattr(C, "test_anthropic", lambda: _ok())
    monkeypatch.setattr(C, "test_deepgram", lambda: _ok())
    secret = "sk-ant-api03-SECRETVALUE1234"
    msgs = []
    async with websockets.connect(f"ws://127.0.0.1:{port}/ws", origin=f"http://localhost:{port}") as ws:
        reader = asyncio.create_task(collect(ws, msgs))
        await ws.send(json.dumps({"action": "save_keys", "keys": {"anthropic": secret}}))
        await asyncio.sleep(0.5)
        reader.cancel()
    assert secret in C.ENV_FILE.read_text()
    assert "SECRETVALUE" not in json.dumps(msgs)
    keys = [m for m in msgs if m["type"] == "keys"][-1]["keys"]
    assert keys["anthropic"] == {"set": True, "hint": "…1234"}
    assert any(m["type"] == "key_test" for m in msgs)


async def _ok():
    return {"ok": True, "text": "OK"}


async def test_overlay_page_served(server):
    html = await asyncio.to_thread(lambda: urllib.request.urlopen(f"http://127.0.0.1:{server}/overlay").read().decode())
    assert "Answer now" in html and "/ws" in html


def test_overlay_window_runs_server_in_thread_and_shuts_down(fake_claude, monkeypatch):
    """pywebview needs a desktop; stub it and check the server lifecycle around the window."""
    import sys
    import types

    seen = {}

    def create_window(title, url, **kw):
        seen["url"], seen["kw"] = url, kw
        return types.SimpleNamespace(destroy=lambda: None)

    def start():  # the window "runs": the server must be up and serving the overlay
        seen["page"] = urllib.request.urlopen(seen["url"]).read().decode()

    monkeypatch.setitem(sys.modules, "webview", types.SimpleNamespace(create_window=create_window, start=start))
    port = free_port()
    C.S.port = port
    srv = uvicorn.Server(uvicorn.Config(C.app, host="127.0.0.1", port=port, log_level="warning"))
    import signal
    old = signal.getsignal(signal.SIGINT)
    try:
        C.run_with_overlay(srv, f"http://127.0.0.1:{port}")
    finally:
        signal.signal(signal.SIGINT, old)
    assert seen["kw"]["on_top"] is True and "Answer now" in seen["page"]
    assert srv.should_exit
