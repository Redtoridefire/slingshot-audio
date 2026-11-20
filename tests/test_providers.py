"""Provider switch: OpenAI for the LLM; AssemblyAI and local Whisper for speech-to-text (all faked)."""
import asyncio
import json
import types

import numpy as np
import pytest
from conftest import C


# ---------------------------------------------------------------- OpenAI
class FakeOpenAICompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kw):
        self.calls.append(kw)
        usage = types.SimpleNamespace(prompt_tokens=1200, completion_tokens=40,
                                      prompt_tokens_details=types.SimpleNamespace(cached_tokens=1000))
        if kw.get("stream"):
            async def gen():
                for t in ["About ", "90 days."]:
                    yield types.SimpleNamespace(usage=None, choices=[types.SimpleNamespace(
                        delta=types.SimpleNamespace(content=t), finish_reason=None)])
                yield types.SimpleNamespace(usage=None, choices=[types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=None), finish_reason="stop")])
                yield types.SimpleNamespace(usage=usage, choices=[])
            return gen()
        return types.SimpleNamespace(usage=usage, choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="YES"), finish_reason="stop")])


@pytest.fixture
def openai(fake_claude, monkeypatch):
    comp = FakeOpenAICompletions()
    monkeypatch.setattr(C, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(C, "ANSWER_MODEL", "gpt-5")
    monkeypatch.setattr(C, "FAST_MODEL", "gpt-5-mini")
    monkeypatch.setattr(C, "_openai", types.SimpleNamespace(chat=types.SimpleNamespace(completions=comp)))
    return comp


async def test_openai_auto_answer_and_stream(openai, fake_claude, sent):
    await C.commit("Them 1", "How long would it take?", "system")
    await asyncio.sleep(0.2)
    await asyncio.wait_for(C.S.answer_task, 3)
    detect, answer = openai.calls
    assert detect["model"] == "gpt-5-mini" and detect["max_completion_tokens"] >= 2000  # room for reasoning
    assert detect["messages"][0]["role"] == "system" and detect["messages"][0]["content"] == C.SYSTEM_DETECT
    assert answer["model"] == "gpt-5" and answer["stream"] is True
    assert "Them 1: How long would it take?" in answer["messages"][1]["content"]
    assert "".join(m["text"] for m in sent if m["type"] == "answer_delta") == "About 90 days."
    assert fake_claude.messages.calls == []  # Claude not used
    u = C.S.usage["gpt-5"]
    assert u == {"in": 200, "out": 40, "cache_read": 1000, "cache_write": 0}


def test_needed_keys_follow_providers(monkeypatch):
    monkeypatch.setattr(C, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(C, "STT_PROVIDER", "whisper")
    assert C.needed_keys() == ["openai"]
    monkeypatch.setattr(C, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(C, "STT_PROVIDER", "assemblyai")
    assert C.needed_keys() == ["anthropic", "assemblyai"]


# ---------------------------------------------------------------- AssemblyAI
def turn(text, order, end=False, formatted=False):
    return json.dumps({"type": "Turn", "turn_order": order, "transcript": text, "end_of_turn": end,
                       "turn_is_formatted": formatted, "words": []})


async def test_assemblyai_turns(fake_claude, sent, fake_dg, monkeypatch):
    seen = {}

    async def handler(ws):
        seen["auth"] = ws.request.headers["Authorization"]
        seen["path"] = ws.request.path
        await ws.send(json.dumps({"type": "Begin", "id": "abc"}))
        n = 0
        async for m in ws:
            if isinstance(m, bytes):
                n += 1
                if n == 1:
                    await ws.send(turn("how long would", 0))
                    await ws.send(turn("how long would it take", 0, end=True))
                    await ws.send(turn("How long would it take?", 0, end=True, formatted=True))
                    await ws.send(turn("thanks", 1, end=True))  # never formatted: committed when the stream ends
            elif json.loads(m) == {"type": "Terminate"}:
                await ws.send(json.dumps({"type": "Termination"}))
                await ws.close()
                return

    await fake_dg(handler)
    monkeypatch.setattr(C, "STT_PROVIDER", "assemblyai")
    monkeypatch.setattr(C, "AAI_URL", C.DG_URL)
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "aaikey")
    C.S.auto_answer = False
    q, stop = asyncio.Queue(), asyncio.Event()
    for _ in range(3):
        q.put_nowait(b"\x00\x00" * 1600)
    t = asyncio.create_task(C.stt_loop({"key": "system", "label": "Them", "diarize": True}, q, stop))
    await asyncio.sleep(0.6)
    stop.set()
    await asyncio.wait_for(t, 6)
    assert seen["auth"] == "aaikey"
    assert "sample_rate=16000" in seen["path"] and "encoding=pcm_s16le" in seen["path"]
    finals = [(m["speaker"], m["text"]) for m in sent if m["type"] == "final"]
    assert finals == [("Them", "How long would it take?"), ("Them", "thanks")]
    assert any(m["type"] == "interim" and m["text"] == "how long would" for m in sent)


# ---------------------------------------------------------------- local Whisper
def pcm(seconds, amp):
    t = np.arange(int(16000 * seconds)) / 16000
    return (np.sin(2 * np.pi * 220 * t) * amp * 32767).astype("<i2").tobytes()


async def test_whisper_endpointing(fake_claude, sent, monkeypatch):
    heard = []

    def fake_transcribe(audio):
        heard.append(len(audio) / 16000)
        return "hello from whisper"

    monkeypatch.setattr(C, "whisper_transcribe", fake_transcribe)
    monkeypatch.setattr(C, "STT_PROVIDER", "whisper")
    C.S.auto_answer = False
    audio = pcm(0.1, 0.0) * 3 + pcm(1.0, 0.2) + pcm(1.0, 0.0) + pcm(0.1, 0.2) + pcm(0.5, 0.0)  # speech, gap, a click
    q, stop = asyncio.Queue(), asyncio.Event()
    for i in range(0, len(audio), 3200):
        q.put_nowait(audio[i:i + 3200])
    stop.set()  # process what's queued, then finish
    await asyncio.wait_for(C.stt_loop({"key": "system", "label": "Them", "diarize": True}, q, stop), 5)
    finals = [(m["speaker"], m["text"]) for m in sent if m["type"] == "final"]
    assert finals == [("Them", "hello from whisper")]  # the 0.1 s click is ignored
    assert len(heard) == 1 and 1.0 <= heard[0] <= 1.9
    assert C.S.stt_seconds == pytest.approx(len(audio) / 32000)
