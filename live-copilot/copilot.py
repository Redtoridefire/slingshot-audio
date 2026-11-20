#!/usr/bin/env python3
"""
Live Copilot — real-time call transcript -> Claude answers.

Captures your mic ("You") and your computer's audio output ("Them") separately,
streams both to Deepgram for live transcription, and uses Claude to:
  * answer on demand (button / global hotkey / typed question)
  * auto-answer when someone asks you a question
  * refresh rolling talking points

Everything is shown in a local browser panel at http://127.0.0.1:8765
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
import threading
import time
import uuid
import warnings
import webbrowser
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import uvicorn
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------- config
DG_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DG_URL = os.getenv("DEEPGRAM_URL", "wss://api.deepgram.com/v1/listen")
DG_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-5-5")
FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-5-5")
HOTKEY = os.getenv("HOTKEY", "<ctrl>+<alt>+a")
POINTS_EVERY = int(os.getenv("POINTS_EVERY_SEC", "30"))
AUTO_COOLDOWN = int(os.getenv("AUTO_COOLDOWN_SEC", "8"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "60000"))
CONTEXT_FILE = ROOT / "context.md"
TRANSCRIPT_DIR = ROOT / "transcripts"

SAMPLE_RATE = 16000
CHUNK = 1600  # 100 ms of audio per send

warnings.filterwarnings("ignore", message=".*discontinuity.*")  # soundcard noise

SYSTEM_ANSWER = """You are a real-time conversation copilot sitting beside the user during a live call or conversation.
The transcript comes from live speech recognition, so expect small errors and fragments. Lines labeled "You" are the user; "Them N" / "Speaker N" are other people.

Your job: give the user something they can use *right now*.
- Start with a direct answer of 1–3 sentences that the user could say out loud, in their voice.
- Then, only if it helps, add up to 3 short bullets: key facts, numbers, or a follow-up question.
- No preamble, no restating the question, no hedging filler. Be concrete and confident; flag real uncertainty in a few words.
"""

SYSTEM_POINTS = """You are a real-time conversation copilot. From the live transcript, give the user exactly 3 short bullets (each under 20 words): the most useful things to say next, a sharp question to ask, or a point they're missing. Plain "- " bullets only, no intro."""

SYSTEM_DETECT = """You classify live conversation lines. Answer only YES or NO."""

QUESTION_HINT = re.compile(
    r"\?\s*$|^(so,?\s+)?(what|how|why|when|where|who|which|can|could|would|will|do|does|did|is|are|should|have|has|tell me|walk me|explain|any thoughts)\b",
    re.I,
)


# ---------------------------------------------------------------- state
class State:
    def __init__(self) -> None:
        self.segments: list[dict] = []
        self.clients: set[WebSocket] = set()
        self.listening = False
        self.auto_answer = True
        self.talking_points = True
        self.meeting_context = ""
        self.mic_only = False
        self.demo = False
        self.answer_task: asyncio.Task | None = None
        self.last_auto = 0.0
        self.points_mark = 0
        self.tasks: list[asyncio.Task] = []
        self.stop_async: asyncio.Event | None = None
        self.stop_thread: threading.Event | None = None
        self.started_at: float | None = None


S = State()
_client = None


def claude():
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic()
    return _client


async def broadcast(msg: dict) -> None:
    data = json.dumps(msg)
    for ws in list(S.clients):
        try:
            await ws.send_text(data)
        except Exception:
            S.clients.discard(ws)


def status_msg() -> dict:
    return {
        "type": "status",
        "listening": S.listening,
        "auto_answer": S.auto_answer,
        "talking_points": S.talking_points,
        "mode": "demo" if S.demo else ("mic-only" if S.mic_only else "call"),
    }


# ---------------------------------------------------------------- transcript helpers
def transcript_text(max_chars: int = MAX_CONTEXT_CHARS) -> str:
    lines = [
        f"[{time.strftime('%H:%M:%S', time.localtime(s['ts']))}] {s['speaker']}: {s['text']}"
        for s in S.segments
    ]
    txt = "\n".join(lines)
    if len(txt) > max_chars:
        txt = "(…earlier conversation trimmed…)\n" + txt[-max_chars:]
    return txt or "(nothing transcribed yet)"


def background_context() -> str:
    parts = []
    try:
        file_ctx = CONTEXT_FILE.read_text(encoding="utf-8").strip()
        if file_ctx:
            parts.append(f"<about_me>\n{file_ctx}\n</about_me>")
    except FileNotFoundError:
        pass
    if S.meeting_context.strip():
        parts.append(f"<this_conversation>\n{S.meeting_context.strip()}\n</this_conversation>")
    return ("\n\n" + "\n\n".join(parts)) if parts else ""


async def commit(speaker: str, text: str, source: str) -> None:
    seg = {"speaker": speaker, "text": text, "ts": time.time(), "source": source}
    S.segments.append(seg)
    await broadcast({"type": "final", **seg})
    if S.auto_answer and (source != "mic" or S.mic_only):
        maybe_auto_answer(seg)


def save_transcript() -> Path | None:
    if not S.segments:
        return None
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    start = S.started_at or S.segments[0]["ts"]
    path = TRANSCRIPT_DIR / time.strftime("%Y-%m-%d_%H%M%S.md", time.localtime(start))
    body = "\n".join(
        f"**{s['speaker']}** ({time.strftime('%H:%M:%S', time.localtime(s['ts']))}): {s['text']}"
        for s in S.segments
    )
    path.write_text(f"# Conversation {time.strftime('%Y-%m-%d %H:%M', time.localtime(start))}\n\n{body}\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------- Claude: answers
def start_answer(trigger: str, question: str | None = None) -> None:
    if S.answer_task and not S.answer_task.done():
        S.answer_task.cancel()
    S.answer_task = asyncio.create_task(run_answer(trigger, question))


async def run_answer(trigger: str, question: str | None) -> None:
    aid = uuid.uuid4().hex[:8]
    await broadcast({"type": "answer_start", "id": aid, "trigger": trigger, "question": question, "ts": time.time()})
    ask = (
        f"My question: {question}"
        if question
        else "Help me respond to the most recent question or point directed at me."
    )
    try:
        async with claude().messages.stream(
            model=ANSWER_MODEL,
            max_tokens=700,
            system=SYSTEM_ANSWER + background_context(),
            messages=[{"role": "user", "content": f"<transcript>\n{transcript_text()}\n</transcript>\n\n{ask}"}],
        ) as stream:
            async for chunk in stream.text_stream:
                await broadcast({"type": "answer_delta", "id": aid, "text": chunk})
        await broadcast({"type": "answer_end", "id": aid})
    except asyncio.CancelledError:
        await broadcast({"type": "answer_end", "id": aid, "cancelled": True})
        raise
    except Exception as e:  # noqa: BLE001
        await broadcast({"type": "answer_end", "id": aid})
        await broadcast({"type": "error", "text": f"Claude error: {e}"})


# ---------------------------------------------------------------- Claude: auto-detect questions
def maybe_auto_answer(seg: dict) -> None:
    if not QUESTION_HINT.search(seg["text"]):
        return
    if time.time() - S.last_auto < AUTO_COOLDOWN:
        return
    if S.answer_task and not S.answer_task.done():
        return
    S.last_auto = time.time()
    asyncio.create_task(auto_check(seg))


async def auto_check(seg: dict) -> None:
    prompt = (
        f"Recent conversation:\n{transcript_text(3000)}\n\n"
        f'Last line — {seg["speaker"]}: "{seg["text"]}"\n\n'
        "Is the last line a real question or request that the user (labeled \"You\", or the listener if there is no \"You\") "
        "is expected to answer now? Ignore rhetorical questions, small talk like \"how are you\", and questions aimed at someone else. "
        "Reply YES or NO."
    )
    try:
        r = await claude().messages.create(
            model=FAST_MODEL, max_tokens=5, system=SYSTEM_DETECT,
            messages=[{"role": "user", "content": prompt}],
        )
        verdict = "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip().upper()
    except Exception as e:  # noqa: BLE001
        await broadcast({"type": "error", "text": f"Question detection error: {e}"})
        return
    if verdict.startswith("YES"):
        start_answer("auto")


# ---------------------------------------------------------------- Claude: rolling talking points
async def points_loop() -> None:
    while True:
        await asyncio.sleep(POINTS_EVERY)
        if not (S.talking_points and S.listening) or len(S.segments) == S.points_mark:
            continue
        S.points_mark = len(S.segments)
        await refresh_points()


async def refresh_points() -> None:
    try:
        r = await claude().messages.create(
            model=FAST_MODEL, max_tokens=300,
            system=SYSTEM_POINTS + background_context(),
            messages=[{"role": "user", "content": f"<transcript>\n{transcript_text(20000)}\n</transcript>"}],
        )
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text").strip()
        await broadcast({"type": "points", "text": text, "ts": time.time()})
    except Exception as e:  # noqa: BLE001
        await broadcast({"type": "error", "text": f"Talking points error: {e}"})


# ---------------------------------------------------------------- audio capture
def list_devices() -> None:
    import soundcard as sc

    print("\nInputs (use for MIC_DEVICE, or BlackHole on Mac for SYSTEM_DEVICE):")
    for m in sc.all_microphones():
        print("  -", m.name)
    loops = [m for m in sc.all_microphones(include_loopback=True) if getattr(m, "isloopback", False)]
    if loops:
        print("\nLoopback / system audio (use for SYSTEM_DEVICE):")
        for m in loops:
            print("  -", m.name)
    print(f"\nDefault mic:     {sc.default_microphone().name}")
    print(f"Default speaker: {sc.default_speaker().name}\n")


def resolve_devices():
    import soundcard as sc

    mic_name = os.getenv("MIC_DEVICE")
    mic = sc.get_microphone(mic_name) if mic_name else sc.default_microphone()
    if S.mic_only:
        return mic, None
    sys_name = os.getenv("SYSTEM_DEVICE")
    try:
        if sys_name:
            system = sc.get_microphone(sys_name, include_loopback=True)
        elif platform.system() == "Darwin":
            system = sc.get_microphone("BlackHole")
        else:  # Windows / Linux: loopback of the default speaker
            system = sc.get_microphone(str(sc.default_speaker().name), include_loopback=True)
    except Exception:  # noqa: BLE001
        system = None
    return mic, system


def capture_thread(device, source: str, loop, queue: asyncio.Queue, stop: threading.Event) -> None:
    if platform.system() == "Windows":  # soundcard needs COM initialised per thread
        import ctypes

        ctypes.windll.ole32.CoInitializeEx(None, 0)
    try:
        with device.recorder(samplerate=SAMPLE_RATE, blocksize=CHUNK) as rec:
            while not stop.is_set():
                data = rec.record(numframes=CHUNK)
                mono = data.mean(axis=1) if data.ndim > 1 else data
                pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                loop.call_soon_threadsafe(queue.put_nowait, pcm)
    except Exception as e:  # noqa: BLE001
        asyncio.run_coroutine_threadsafe(
            broadcast({"type": "error", "text": f"Audio capture failed for {source} ({device.name}): {e}"}), loop
        )


# ---------------------------------------------------------------- Deepgram streaming
def dg_query(diarize: bool) -> str:
    params = {
        "model": DG_MODEL, "encoding": "linear16", "sample_rate": SAMPLE_RATE, "channels": 1,
        "interim_results": "true", "smart_format": "true", "punctuate": "true",
        "endpointing": 300, "utterance_end_ms": 1000, "vad_events": "true",
        "diarize": "true" if diarize else "false",
    }
    return f"{DG_URL}?{urlencode(params)}"


def speaker_of(src: dict, alt: dict) -> str:
    words = alt.get("words") or []
    if not src["diarize"] or not words:
        return src["label"]
    n = Counter(w.get("speaker", 0) for w in words).most_common(1)[0][0]
    return f"{src['label']} {int(n) + 1}"


async def stt_loop(src: dict, queue: asyncio.Queue, stop: asyncio.Event) -> None:
    """Keep a Deepgram stream open for one audio source; reconnect on failure."""
    while not stop.is_set():
        try:
            async with websockets.connect(
                dg_query(src["diarize"]),
                additional_headers={"Authorization": f"Token {DG_KEY}"},
                max_size=None, ping_interval=20,
            ) as ws:
                await broadcast({"type": "info", "text": f"Transcribing {src['label']} audio"})
                sender = asyncio.create_task(_dg_send(ws, queue, stop))
                receiver = asyncio.create_task(_dg_recv(ws, src))
                done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
                if sender in done:  # stop requested: let Deepgram flush final results
                    try:
                        await asyncio.wait_for(receiver, 3)
                    except Exception:  # noqa: BLE001
                        pass
                for t in pending:
                    t.cancel()
                for t in done:
                    if t.exception():
                        raise t.exception()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if stop.is_set():
                break
            await broadcast({"type": "error", "text": f"Deepgram ({src['label']}): {e} — reconnecting"})
            await asyncio.sleep(2)


async def _dg_send(ws, queue: asyncio.Queue, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            chunk = await asyncio.wait_for(queue.get(), 1.0)
        except asyncio.TimeoutError:
            await ws.send(json.dumps({"type": "KeepAlive"}))
            continue
        await ws.send(chunk)
    await ws.send(json.dumps({"type": "CloseStream"}))


async def _dg_recv(ws, src: dict) -> None:
    pending: list[tuple[str, str]] = []  # finalized pieces of the current utterance

    async def flush() -> None:
        nonlocal pending
        # merge consecutive pieces from the same speaker
        merged: list[list[str]] = []
        for spk, txt in pending:
            if merged and merged[-1][0] == spk:
                merged[-1][1] += " " + txt
            else:
                merged.append([spk, txt])
        pending = []
        await broadcast({"type": "interim", "source": src["key"], "text": ""})
        for spk, txt in merged:
            await commit(spk, txt, src["key"])

    async for raw in ws:
        if isinstance(raw, bytes):
            continue
        msg = json.loads(raw)
        kind = msg.get("type")
        if kind == "Results":
            alt = (msg.get("channel", {}).get("alternatives") or [{}])[0]
            text = (alt.get("transcript") or "").strip()
            if msg.get("is_final"):
                if text:
                    pending.append((speaker_of(src, alt), text))
                if msg.get("speech_final") and pending:
                    await flush()
                elif pending:
                    await broadcast({"type": "interim", "source": src["key"], "text": " ".join(t for _, t in pending)})
            elif text:
                live = " ".join([t for _, t in pending] + [text])
                await broadcast({"type": "interim", "source": src["key"], "text": live, "speaker": speaker_of(src, alt)})
        elif kind == "UtteranceEnd" and pending:
            await flush()
    if pending:
        await flush()


# ---------------------------------------------------------------- demo feed (no audio / Deepgram needed)
DEMO_SCRIPT = [
    ("Them 1", "system", "Thanks for making time. Quick background: we're a two hundred person fintech, mostly on AWS."),
    ("You", "mic", "Great to meet you. What prompted the call?"),
    ("Them 1", "system", "We just went through a SOC 2 readiness assessment and it did not go well."),
    ("Them 1", "system", "The big gaps were access reviews, vendor risk management, and we basically have no incident response plan."),
    ("Them 2", "system", "And our board wants a security update next quarter."),
    ("Them 1", "system", "So realistically, how long would it take to get us audit ready if we brought you in?"),
    ("You", "mic", "It depends on scope, but let me give you a realistic picture."),
    ("Them 2", "system", "Also, what would you prioritize in the first thirty days?"),
]


async def demo_feed(stop: asyncio.Event) -> None:
    await asyncio.sleep(1)
    for speaker, source, text in DEMO_SCRIPT:
        words = text.split()
        for i in range(1, len(words) + 1, 2):
            if stop.is_set():
                return
            await broadcast({"type": "interim", "source": source, "text": " ".join(words[:i]), "speaker": speaker})
            await asyncio.sleep(0.12)
        await broadcast({"type": "interim", "source": source, "text": ""})
        await commit(speaker, text, source)
        try:
            await asyncio.wait_for(stop.wait(), 2.5 if "?" in text else 1.2)
            return
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------- start / stop
async def start_listening() -> None:
    if S.listening:
        return
    loop = asyncio.get_running_loop()
    S.stop_async, S.stop_thread = asyncio.Event(), threading.Event()
    S.tasks = []
    S.started_at = S.started_at or time.time()
    if S.demo:
        S.tasks.append(asyncio.create_task(demo_feed(S.stop_async)))
    else:
        if not DG_KEY:
            await broadcast({"type": "error", "text": "DEEPGRAM_API_KEY is not set in .env"})
            return
        try:
            mic, system = await asyncio.to_thread(resolve_devices)
        except Exception as e:  # noqa: BLE001
            await broadcast({"type": "error", "text": f"Could not open audio devices: {e}"})
            return
        sources = [("mic", mic, "Speaker" if S.mic_only else "You", S.mic_only)]
        if system is not None:
            sources.append(("system", system, "Them", True))
        elif not S.mic_only:
            await broadcast({"type": "error", "text": "No system-audio device found — capturing mic only. "
                             "On Mac install BlackHole (see README); or set SYSTEM_DEVICE in .env."})
        for key, dev, label, diarize in sources:
            q: asyncio.Queue = asyncio.Queue()
            threading.Thread(target=capture_thread, args=(dev, key, loop, q, S.stop_thread), daemon=True).start()
            S.tasks.append(asyncio.create_task(stt_loop({"key": key, "label": label, "diarize": diarize}, q, S.stop_async)))
            await broadcast({"type": "info", "text": f"{label}: {dev.name}"})
    S.listening = True
    await broadcast(status_msg())


async def stop_listening() -> None:
    if not S.listening:
        return
    S.stop_async.set()
    S.stop_thread.set()
    if S.tasks:
        _, pending = await asyncio.wait(S.tasks, timeout=4)
        for t in pending:
            t.cancel()
    S.listening = False
    path = save_transcript()
    await broadcast(status_msg())
    if path:
        await broadcast({"type": "info", "text": f"Transcript saved to {path.relative_to(ROOT)}"})


# ---------------------------------------------------------------- global hotkey
def start_hotkey(loop) -> None:
    try:
        from pynput import keyboard

        def fire() -> None:
            loop.call_soon_threadsafe(start_answer, "hotkey")

        hk = keyboard.GlobalHotKeys({HOTKEY: fire})
        hk.daemon = True
        hk.start()
        print(f"  Global hotkey: {HOTKEY}  (answer now)")
    except Exception as e:  # noqa: BLE001
        print(f"  Global hotkey unavailable ({e}). Use the button or Alt+A in the panel.")


# ---------------------------------------------------------------- web app
@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_running_loop()
    points = asyncio.create_task(points_loop())
    if os.getenv("DISABLE_HOTKEY") != "1":
        start_hotkey(loop)
    yield
    points.cancel()
    await stop_listening()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    S.clients.add(ws)
    await ws.send_text(json.dumps({**status_msg(), "type": "init", "segments": S.segments, "context": S.meeting_context}))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            await handle(msg)
    except WebSocketDisconnect:
        pass
    finally:
        S.clients.discard(ws)


async def handle(msg: dict) -> None:
    action = msg.get("action")
    if action == "start":
        await start_listening()
    elif action == "stop":
        await stop_listening()
    elif action == "answer":
        start_answer("typed" if msg.get("question") else "button", (msg.get("question") or "").strip() or None)
    elif action == "points_now":
        asyncio.create_task(refresh_points())
    elif action == "set":
        if "auto_answer" in msg:
            S.auto_answer = bool(msg["auto_answer"])
        if "talking_points" in msg:
            S.talking_points = bool(msg["talking_points"])
        await broadcast(status_msg())
    elif action == "context":
        S.meeting_context = msg.get("text", "")
    elif action == "clear":
        save_transcript()
        S.segments.clear()
        S.points_mark = 0
        S.started_at = None
        await broadcast({"type": "cleared"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Live Copilot: real-time transcript -> Claude")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--mic-only", action="store_true", help="in-person mode: one mic, speakers separated by diarization")
    ap.add_argument("--demo", action="store_true", help="play a scripted conversation (no audio or Deepgram needed)")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8765")))
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        list_devices()
        return
    S.mic_only, S.demo = args.mic_only, args.demo

    print("\nLive Copilot")
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("  ! ANTHROPIC_API_KEY not set — answers will fail until you add it to .env")
    if not DG_KEY and not S.demo:
        print("  ! DEEPGRAM_API_KEY not set — add it to .env (or run with --demo)")
    url = f"http://127.0.0.1:{args.port}"
    print(f"  Panel: {url}\n  Mode: {status_msg()['mode']}   Models: {ANSWER_MODEL} (answers), {FAST_MODEL} (detect/points)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
