import asyncio, json, sys, types, os, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE))
os.environ["DISABLE_HOTKEY"] = "1"
os.environ["POINTS_EVERY_SEC"] = "2"
import copilot as C
import websockets, uvicorn

# ---------- fake Claude
class FakeStream:
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False
    @property
    def text_stream(self):
        async def gen():
            for t in ["**Roughly 90 days** ", "to audit-ready.\n", "- Access reviews first"]:
                await asyncio.sleep(0.01); yield t
        return gen()
class FakeMessages:
    def __init__(self): self.calls = []
    def stream(self, **kw): self.calls.append(("stream", kw)); return FakeStream()
    async def create(self, **kw):
        self.calls.append(("create", kw))
        txt = "YES" if kw["max_tokens"] == 5 else "- point one\n- point two\n- point three"
        return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=txt)])
fake = types.SimpleNamespace(messages=FakeMessages())
C._client = fake

def res(text, is_final, speech_final=False, spk=0):
    return json.dumps({"type": "Results", "is_final": is_final, "speech_final": speech_final,
        "channel": {"alternatives": [{"transcript": text, "words": [{"word": w, "speaker": spk} for w in text.split()]}]}})

async def fake_dg(ws):
    assert "Token testkey" in ws.request.headers["Authorization"], "auth header"
    assert "diarize=true" in ws.request.path
    got_audio = 0
    async for m in ws:
        if isinstance(m, bytes):
            got_audio += 1
            if got_audio == 1:
                await ws.send(res("How long", False, spk=1))
                await ws.send(res("How long would it take", True, spk=1))
                await ws.send(res("to get audit ready?", True, True, spk=1))
                await ws.send(res("Thanks.", True, spk=0))
                await ws.send(json.dumps({"type": "UtteranceEnd"}))
        else:
            if json.loads(m)["type"] == "CloseStream":
                await ws.close(); return

async def main():
    out = []
    async def capture(msg): out.append(msg)
    C.broadcast = capture  # capture broadcasts

    # ---- 1. Deepgram stream parsing
    C.DG_KEY = "testkey"
    srv = await websockets.serve(fake_dg, "127.0.0.1", 8799)
    C.DG_URL = "ws://127.0.0.1:8799/v1/listen"
    q, stop = asyncio.Queue(), asyncio.Event()
    for _ in range(5): q.put_nowait(b"\x00\x00" * 1600)
    C.S.auto_answer = True
    t = asyncio.create_task(C.stt_loop({"key": "system", "label": "Them", "diarize": True}, q, stop))
    await asyncio.sleep(1.0); stop.set(); await asyncio.wait_for(t, 5)
    srv.close()
    finals = [(m["speaker"], m["text"]) for m in out if m["type"] == "final"]
    print("finals:", finals)
    assert finals == [("Them 2", "How long would it take to get audit ready?"), ("Them 1", "Thanks.")], finals
    assert any(m["type"] == "interim" and m["text"] == "How long" for m in out)
    await asyncio.sleep(0.3)
    kinds = [m["type"] for m in out]
    print("after question:", kinds[-6:])
    assert "answer_start" in kinds and "answer_end" in kinds, "auto answer should fire"
    ans = "".join(m["text"] for m in out if m["type"] == "answer_delta")
    assert ans.startswith("**Roughly"), ans
    stream_kw = [c for c in fake.messages.calls if c[0] == "stream"][0][1]
    assert "Them 2: How long would it take" in stream_kw["messages"][0]["content"]
    assert "Mike Czarnecki" in stream_kw["system"]
    print("OK  deepgram parsing + auto-answer")

    # mic source must NOT auto-answer in call mode
    C.S.last_auto = 0; n = len(fake.messages.calls)
    await C.commit("You", "What do you think?", "mic"); await asyncio.sleep(0.1)
    assert len(fake.messages.calls) == n, "mic shouldn't trigger auto"
    # non-question shouldn't trigger
    await C.commit("Them 1", "We use AWS.", "system"); await asyncio.sleep(0.1)
    assert len(fake.messages.calls) == n
    print("OK  auto-answer filtering")

asyncio.run(main())
