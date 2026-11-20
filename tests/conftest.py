"""Shared fixtures: a fake Claude client, a fake Deepgram websocket server, and broadcast capture."""
import asyncio
import json
import os
import socket
import sys
import types
from pathlib import Path

import pytest
import websockets

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["DISABLE_HOTKEY"] = "1"

import copilot as C  # noqa: E402


# ---------------------------------------------------------------- fake Claude
def _usage(i=1000, o=100):
    return types.SimpleNamespace(input_tokens=i, output_tokens=o, cache_read_input_tokens=0, cache_creation_input_tokens=0)


class FakeStream:
    def __init__(self, chunks, delay, stop_reason):
        self.chunks, self.delay, self.stop_reason = chunks, delay, stop_reason

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    @property
    def text_stream(self):
        async def gen():
            for t in self.chunks:
                await asyncio.sleep(self.delay)
                yield t
        return gen()

    async def get_final_message(self):
        return types.SimpleNamespace(stop_reason=self.stop_reason, usage=_usage(), model=None,
                                     content=[types.SimpleNamespace(type="text", text="".join(self.chunks))])


class FakeMessages:
    def __init__(self):
        self.calls = []
        self.verdict = "YES"
        self.chunks = ["**Roughly 90 days** ", "to audit-ready.\n", "- Access reviews first"]
        self.delay = 0.01
        self.stop_reason = "end_turn"

    def stream(self, **kw):
        self.calls.append(("stream", kw))
        return FakeStream(self.chunks if self.stop_reason != "refusal" else [], self.delay, self.stop_reason)

    async def create(self, **kw):
        self.calls.append(("create", kw))
        system = kw["system"] if isinstance(kw["system"], str) else kw["system"][0]["text"]
        if system == C.SYSTEM_DETECT:
            txt = self.verdict
        elif system == C.SYSTEM_SUMMARY:
            txt = "- They are a 200-person fintech on AWS.\n- SOC 2 readiness went badly."
        else:
            txt = "- point one\n- point two\n- point three"
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=txt)], usage=_usage(500, 10))

    def of(self, kind):
        return [kw for k, kw in self.calls if k == kind]


class FakeClaude:
    def __init__(self):
        self.messages = FakeMessages()
        self.beta = types.SimpleNamespace(messages=self.messages)


def content_text(kw) -> str:
    return "\n".join(b["text"] for b in kw["messages"][0]["content"])


@pytest.fixture
def fake_claude(monkeypatch, tmp_path):
    C.reset_state()
    fake = FakeClaude()
    monkeypatch.setattr(C, "_client", fake)
    monkeypatch.setattr(C, "TRANSCRIPT_DIR", tmp_path / "transcripts")
    monkeypatch.setattr(C, "ENV_FILE", tmp_path / ".env")
    yield fake
    C.reset_state()


@pytest.fixture
def sent(monkeypatch):
    out = []

    async def capture(msg):
        out.append(msg)

    monkeypatch.setattr(C, "broadcast", capture)
    return out


# ---------------------------------------------------------------- fake Deepgram
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def dg_result(text, is_final, speech_final=False, spk=0) -> str:
    return json.dumps({"type": "Results", "is_final": is_final, "speech_final": speech_final,
                       "channel": {"alternatives": [{"transcript": text,
                                                     "words": [{"word": w, "speaker": spk} for w in text.split()]}]}})


@pytest.fixture
async def fake_dg(monkeypatch):
    """Usage: url = await fake_dg(handler). Points copilot at a local websocket server running `handler`."""
    servers = []
    monkeypatch.setenv("DEEPGRAM_API_KEY", "testkey")
    monkeypatch.setattr(C, "RECONNECT_MIN", 0.05)

    async def start(handler):
        port = free_port()
        srv = await websockets.serve(handler, "127.0.0.1", port)
        servers.append(srv)
        monkeypatch.setattr(C, "DG_URL", f"ws://127.0.0.1:{port}/v1/listen")
        return srv

    yield start
    for s in servers:
        s.close()
        await s.wait_closed()
