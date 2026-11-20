#!/usr/bin/env python3
"""
Live Copilot: real-time call transcript -> Claude answers.

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
import difflib
import json
import logging
import os
import platform
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import warnings
import webbrowser
from collections import Counter
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import uvicorn
import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# ---------------------------------------------------------------- config
DG_URL = os.getenv("DEEPGRAM_URL", "wss://api.deepgram.com/v1/listen")
DG_API = os.getenv("DEEPGRAM_API_URL", "https://api.deepgram.com")
DG_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "claude-sonnet-5-5")
FAST_MODEL = os.getenv("FAST_MODEL", "claude-haiku-5-5")
ANSWER_EFFORT = os.getenv("ANSWER_EFFORT", "low")          # low = fastest first token; medium/high = more careful
ANSWER_THINKING = os.getenv("ANSWER_THINKING", "adaptive")  # "off" skips thinking entirely (where the model allows it)
FALLBACKS = os.getenv("CLAUDE_FALLBACKS", "1") != "0"       # server-side retry on another model if one declines
HOTKEY = os.getenv("HOTKEY", "<ctrl>+<alt>+a")
POINTS_EVERY = _env_int("POINTS_EVERY_SEC", 30)
AUTO_COOLDOWN = _env_int("AUTO_COOLDOWN_SEC", 8)
RECENT_CHARS = _env_int("RECENT_CHARS", 12000)          # transcript kept word for word; older parts get summarized
MAX_CONTEXT_CHARS = _env_int("MAX_CONTEXT_CHARS", 60000)  # hard cap, only hit if summarizing fails
FREEZE_STEP = 20                                          # segments per cache-friendly transcript chunk
ECHO_WINDOW = _env_float("ECHO_WINDOW_SEC", 2.0)
ECHO_MATCH = _env_float("ECHO_MATCH", 0.7)
AUDIO_QUEUE_MAX = _env_int("AUDIO_QUEUE_MAX", 50)         # 100 ms chunks -> 5 s of buffered audio per source
DG_PRICE_PER_MIN = _env_float("DEEPGRAM_PRICE_PER_MIN", 0.0077)  # check your Deepgram plan; this is an estimate
DEMO_SPEED = _env_float("DEMO_SPEED", 1.0)
POST_CALL_SUMMARY = os.getenv("POST_CALL_SUMMARY", "1") != "0"   # write a recap + action items when you stop
CONTEXT_FILE = ROOT / "context.md"
PREP_DIR = ROOT / "context"            # prep docs (prospect background, notes) you can switch on per meeting
PREP_MAX_CHARS = _env_int("PREP_MAX_CHARS", 60000)
TRANSCRIPT_DIR = ROOT / "transcripts"
LOG_DIR = ROOT / "logs"

SAMPLE_RATE = 16000
CHUNK = 1600  # 100 ms of audio per send
SILENCE_RMS = 0.003  # about -50 dBFS
RECONNECT_MIN, RECONNECT_MAX = 0.5, 10.0

# USD per million tokens: (input, output, cache read). 5-minute cache writes cost 1.25x input.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-opus-5": (5.00, 25.00, 0.50),
    "claude-sonnet-5": (2.00, 10.00, 0.20),
    "claude-haiku-4-5": (1.00, 5.00, 0.10),
}
EFFORT_MODELS = {
    "claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "claude-opus-5", "claude-sonnet-5",
    "claude-fable-5-2", "claude-fable-5-1", "claude-fable-5", "claude-opus-4-8", "claude-opus-4-7",
    "claude-opus-4-6", "claude-sonnet-4-6",
}
DISABLE_THINKING_MODELS = {"claude-haiku-5-5", "claude-opus-5", "claude-sonnet-5", "claude-opus-4-8", "claude-opus-4-7"}
FALLBACK_MODELS = {"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"

KEY_VARS = {"anthropic": "ANTHROPIC_API_KEY", "deepgram": "DEEPGRAM_API_KEY"}

warnings.filterwarnings("ignore", message=".*discontinuity.*")  # soundcard noise

log = logging.getLogger("copilot")

SYSTEM_ANSWER = """You are a real-time conversation copilot sitting beside the user during a live call or conversation.
The transcript comes from live speech recognition, so expect small errors and fragments. Lines labeled "You" are the user; "Them N" / "Speaker N" are other people.

Your job: give the user something they can use *right now*.
- Start with a direct answer of 1–3 sentences that the user could say out loud, in their voice.
- Then, only if it helps, add up to 3 short bullets: key facts, numbers, or a follow-up question.
- No preamble, no restating the question, no hedging filler. Be concrete and confident; flag real uncertainty in a few words.
"""

SYSTEM_POINTS = """You are a real-time conversation copilot. From the live transcript, give the user exactly 3 short bullets (each under 20 words): the most useful things to say next, a sharp question to ask, or a point they're missing. Plain "- " bullets only, no intro."""

SYSTEM_DETECT = """You classify live conversation lines. Answer only YES or NO."""

SYSTEM_SUMMARY = """You compress the earlier part of a live conversation so a real-time copilot can keep it in mind.
Write a dense summary (at most 250 words) in plain bullets. Keep: who is who, names, numbers, dates, prices, commitments,
questions asked (and whether they were answered), objections, decisions, and open action items. No intro."""

SYSTEM_RECAP = """You write the recap of a conversation the user just had, from its live transcript (speech recognition, so expect small errors).
Lines labeled "You" are the user. Use only what was said; never invent facts. Write "None" under a heading that has nothing.
Markdown, exactly these sections:
## Summary
3–6 bullets: who was there, what was discussed, where things landed.
## Action items
"- [ ] Owner — task (due date if one was mentioned)". Use "Me" for the user's own items.
## Decisions
## Open questions
## Follow-up email
A short draft the user could send, in their voice."""

QUESTION_HINT = re.compile(
    r"\?\s*$|^(so,?\s+)?(what|how|why|when|where|who|which|can|could|would|will|do|does|did|is|are|should|have|has|tell me|walk me|explain|any thoughts)\b",
    re.I,
)


# ---------------------------------------------------------------- logging
def setup_logging() -> None:
    if any(isinstance(h, RotatingFileHandler) for h in log.handlers):
        return
    LOG_DIR.mkdir(exist_ok=True)
    fh = RotatingFileHandler(LOG_DIR / "copilot.log", maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s"))
    log.addHandler(fh)
    log.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())


# ---------------------------------------------------------------- state
class State:
    def __init__(self) -> None:
        self.segments: list[dict] = []
        self.clients: set[WebSocket] = set()
        self.listening = False
        self.auto_answer = True
        self.talking_points = True
        self.echo_filter = os.getenv("ECHO_FILTER", "0") == "1"
        self.meeting_context = ""
        self.prep_files: list[str] = []
        self.names: dict[str, str] = {}  # "Them 1" -> "Sarah, CFO"
        self.mic_only = False
        self.demo = False
        self.port = _env_int("PORT", 8765)
        self.answer_task: asyncio.Task | None = None
        self.last_auto = 0.0
        self.points_mark = 0
        self.tasks: list[asyncio.Task] = []
        self.stop_async: asyncio.Event | None = None
        self.stop_thread: threading.Event | None = None
        self.started_at: float | None = None
        self.listen_since = 0.0
        self.dirty = False
        self.shutting_down = False
        # long-call memory
        self.summary = ""
        self.summary_upto = 0  # segments[:summary_upto] are covered by self.summary
        self.summary_task: asyncio.Task | None = None
        # audio health
        self.levels: dict[str, float] = {}
        self.heard: dict[str, bool] = {}
        self.silence_warned: set[str] = set()
        self.dropped: Counter = Counter()
        self.device_names: dict[str, str] = {}
        # cost
        self.usage: dict[str, dict[str, int]] = {}
        self.dg_seconds = 0.0
        self.hotkey_proc = None
        self.recap_task: asyncio.Task | None = None


S = State()
_client = None


def reset_state() -> None:
    S.__init__()


def get_key(which: str) -> str:
    """Key from the environment, treating the .env.example placeholders as unset."""
    v = os.getenv(KEY_VARS[which], "").strip()
    return "" if not v or v.endswith("...") else v


def claude():
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic(api_key=get_key("anthropic") or None)
    return _client


async def broadcast(msg: dict) -> None:
    data = json.dumps(msg)
    for ws in list(S.clients):
        try:
            await ws.send_text(data)
        except Exception:
            S.clients.discard(ws)


async def notify_error(text: str, exc: BaseException | None = None) -> None:
    if exc is not None:
        log.error("%s", text, exc_info=exc)
    else:
        log.error("%s", text)
    await broadcast({"type": "error", "text": text})


def status_msg() -> dict:
    return {
        "type": "status",
        "listening": S.listening,
        "auto_answer": S.auto_answer,
        "talking_points": S.talking_points,
        "echo_filter": S.echo_filter,
        "mode": "demo" if S.demo else ("mic-only" if S.mic_only else "call"),
    }


# ---------------------------------------------------------------- cost
def record_usage(model: str, usage) -> None:
    if usage is None:
        return
    u = S.usage.setdefault(model, {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0})
    u["in"] += getattr(usage, "input_tokens", 0) or 0
    u["out"] += getattr(usage, "output_tokens", 0) or 0
    u["cache_read"] += getattr(usage, "cache_read_input_tokens", 0) or 0
    u["cache_write"] += getattr(usage, "cache_creation_input_tokens", 0) or 0


def cost_msg() -> dict:
    claude_usd = 0.0
    for model, u in S.usage.items():
        p_in, p_out, p_read = PRICES.get(model, PRICES["claude-sonnet-5-5"])
        claude_usd += (u["in"] * p_in + u["out"] * p_out + u["cache_read"] * p_read + u["cache_write"] * p_in * 1.25) / 1e6
    dg_usd = S.dg_seconds / 60 * DG_PRICE_PER_MIN
    tokens = {k: sum(u[k] for u in S.usage.values()) for k in ("in", "out", "cache_read", "cache_write")}
    return {
        "type": "cost", "total": round(claude_usd + dg_usd, 4), "claude": round(claude_usd, 4),
        "deepgram": round(dg_usd, 4), "audio_min": round(S.dg_seconds / 60, 2), "tokens": tokens,
    }


# ---------------------------------------------------------------- transcript helpers
def display(label: str) -> str:
    return S.names.get(label, label)


def fmt_line(s: dict) -> str:
    return f"[{time.strftime('%H:%M:%S', time.localtime(s['ts']))}] {display(s['speaker'])}: {s['text']}"


SPEAKER_LABEL = re.compile(r"^(You|Them|Speaker)( \d{1,2})?$")


def rename_speaker(label: str, name: str) -> bool:
    label, name = str(label).strip(), " ".join(str(name).split())[:60]
    if not SPEAKER_LABEL.match(label):
        return False
    if name and name != label:
        S.names[label] = name
    else:
        S.names.pop(label, None)
    log.info("speaker %s renamed to %r", label, name or label)
    return True


def transcript_text(max_chars: int = 3000) -> str:
    """Plain tail of the transcript (for quick checks)."""
    lines: list[str] = []
    size = 0
    for s in reversed(S.segments):
        line = fmt_line(s)
        if size + len(line) > max_chars and lines:
            break
        lines.append(line)
        size += len(line) + 1
    return "\n".join(reversed(lines)) or "(nothing transcribed yet)"


def transcript_parts() -> tuple[str, str]:
    """(stable, recent). `stable` = rolling summary + older verbatim lines, frozen in chunks of
    FREEZE_STEP segments so it stays byte-identical between calls and can be prompt-cached.
    `recent` = the newest lines."""
    start = S.summary_upto
    live = S.segments[start:]
    cut = start + (max(0, len(live) - 4) // FREEZE_STEP) * FREEZE_STEP
    stable_lines = [fmt_line(s) for s in S.segments[start:cut]]
    recent_lines = [fmt_line(s) for s in S.segments[cut:]]
    stable, recent = "\n".join(stable_lines), "\n".join(recent_lines)
    if len(stable) + len(recent) > MAX_CONTEXT_CHARS:  # only if the summarizer is behind or failing
        budget = max(0, MAX_CONTEXT_CHARS - len(recent))
        stable = "(…earlier conversation trimmed…)\n" + (stable[len(stable) - budget:] if budget else "")
        recent = recent[-MAX_CONTEXT_CHARS:]
    if S.summary:
        stable = f"<summary_of_earlier_conversation>\n{S.summary}\n</summary_of_earlier_conversation>\n\n{stable}"
    return stable.strip(), recent


def transcript_blocks() -> list[dict]:
    stable, recent = transcript_parts()
    if not stable and not recent:
        return [{"type": "text", "text": "<transcript>\n(nothing transcribed yet)\n</transcript>"}]
    blocks = []
    if stable:
        blocks.append({"type": "text", "text": f"<transcript_part_1>\n{stable}\n</transcript_part_1>",
                       "cache_control": {"type": "ephemeral"}})
    blocks.append({"type": "text", "text": f"<transcript{'_part_2' if stable else ''}>\n{recent or '(no new lines)'}\n"
                                           f"</transcript{'_part_2' if stable else ''}>"})
    return blocks


def prep_docs() -> list[dict]:
    """Markdown / text files in context/ that can be switched on for a meeting."""
    if not PREP_DIR.is_dir():
        return []
    return [{"name": p.name, "kb": max(1, p.stat().st_size // 1024)}
            for p in sorted(PREP_DIR.iterdir(), key=lambda p: p.name.lower())
            if p.is_file() and p.suffix.lower() in (".md", ".txt") and not p.name.startswith(".")]


def set_prep_files(names) -> None:
    available = {d["name"] for d in prep_docs()}
    S.prep_files = [n for n in dict.fromkeys(names or []) if isinstance(n, str) and n in available]


def background_context() -> str:
    parts = []
    try:
        file_ctx = CONTEXT_FILE.read_text(encoding="utf-8").strip()
        if file_ctx:
            parts.append(f"<about_me>\n{file_ctx}\n</about_me>")
    except FileNotFoundError:
        pass
    for name in S.prep_files:  # re-read every call, so edits apply mid-call
        try:
            doc = (PREP_DIR / name).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        if len(doc) > PREP_MAX_CHARS:
            doc = doc[:PREP_MAX_CHARS] + "\n(…truncated…)"
        parts.append(f'<prep_doc name="{name}">\n{doc}\n</prep_doc>')
    if S.names:
        who = "\n".join(f"{label} is {name}" for label, name in sorted(S.names.items()))
        parts.append(f"<speakers>\nThe transcript uses these names (earlier summaries may use the labels):\n{who}\n</speakers>")
    if S.meeting_context.strip():
        parts.append(f"<this_conversation>\n{S.meeting_context.strip()}\n</this_conversation>")
    return ("\n\n" + "\n\n".join(parts)) if parts else ""


def system_blocks(base: str) -> list[dict]:
    """Stable prefix (instructions + context.md + meeting context), marked for prompt caching."""
    return [{"type": "text", "text": base + background_context(), "cache_control": {"type": "ephemeral"}}]


# ---------------------------------------------------------------- echo filter
def _words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", t.lower())


def is_echo(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    if min(len(wa), len(wb)) < 3:
        return False
    short, long_ = sorted((wa, wb), key=len)
    m = difflib.SequenceMatcher(None, short, long_, autojunk=False)
    matched = sum(bl.size for bl in m.get_matching_blocks())
    return matched / len(short) >= ECHO_MATCH


async def commit(speaker: str, text: str, source: str) -> None:
    now = time.time()
    seg = {"id": uuid.uuid4().hex[:8], "speaker": speaker, "text": text, "ts": now, "source": source,
           "mono": time.monotonic()}
    if S.echo_filter and not S.mic_only:
        recent = [s for s in reversed(S.segments) if now - s["ts"] <= ECHO_WINDOW]
        if source == "mic" and any(s["source"] == "system" and is_echo(text, s["text"]) for s in recent):
            log.info("echo dropped (mic): %s", text)
            await broadcast({"type": "interim", "source": "mic", "text": ""})
            return
        if source == "system":
            for s in recent:
                if s["source"] == "mic" and is_echo(text, s["text"]):
                    S.segments.remove(s)
                    log.info("echo retracted (mic): %s", s["text"])
                    await broadcast({"type": "retract", "id": s["id"]})
    S.segments.append(seg)
    S.dirty = True
    log.debug("segment %s: %s", speaker, text)
    await broadcast({"type": "final", **seg})
    maybe_summarize()
    if S.auto_answer and (source != "mic" or S.mic_only):
        maybe_auto_answer(seg)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def save_transcript() -> Path | None:
    if not S.segments:
        return None
    TRANSCRIPT_DIR.mkdir(exist_ok=True)
    start = S.started_at or S.segments[0]["ts"]
    path = TRANSCRIPT_DIR / time.strftime("%Y-%m-%d_%H%M%S.md", time.localtime(start))
    body = "\n".join(
        f"**{display(s['speaker'])}** ({time.strftime('%H:%M:%S', time.localtime(s['ts']))}): {s['text']}"
        for s in S.segments
    )
    c = cost_msg()
    footer = (f"\n\n---\n_Estimated cost: ${c['total']:.3f} (Claude ${c['claude']:.3f}, "
              f"Deepgram ${c['deepgram']:.3f} for {c['audio_min']} min of audio)_\n")
    path.write_text(f"# Conversation {time.strftime('%Y-%m-%d %H:%M', time.localtime(start))}\n\n{body}\n{footer}",
                    encoding="utf-8")
    S.dirty = False
    log.info("transcript saved: %s (%d segments)", path, len(S.segments))
    return path


# ---------------------------------------------------------------- Claude helpers
def model_opts(model: str, effort: str, thinking_off: bool) -> dict:
    """Effort / thinking settings that the given model accepts."""
    kw: dict = {}
    if model in EFFORT_MODELS:
        kw["output_config"] = {"effort": effort}
    if thinking_off:
        if model == "claude-sonnet-5-5":
            kw["thinking"] = {"type": "between_tools"}  # Sonnet 5.5's "no thinking" setting
        elif model in DISABLE_THINKING_MODELS:
            kw["thinking"] = {"type": "disabled"}
        # Opus 5.5 / Fable can't turn thinking off; low effort is the lever there.
    return kw


def text_of(resp) -> str:
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()


async def quick(model: str, system, prompt, max_tokens: int):
    """Short, fast, non-streamed call with thinking off (used for detection, points, summaries)."""
    content = prompt if isinstance(prompt, list) else [{"type": "text", "text": prompt}]
    r = await claude().messages.create(
        model=model, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": content}],
        **model_opts(model, "low", thinking_off=True),
    )
    record_usage(getattr(r, "model", model) or model, getattr(r, "usage", None))
    return r


# ---------------------------------------------------------------- Claude: answers
def start_answer(trigger: str, question: str | None = None, t_ref: float | None = None) -> None:
    if S.answer_task and not S.answer_task.done():
        S.answer_task.cancel()
    S.answer_task = asyncio.create_task(run_answer(trigger, question, t_ref or time.monotonic()))


async def run_answer(trigger: str, question: str | None, t_ref: float) -> None:
    aid = uuid.uuid4().hex[:8]
    await broadcast({"type": "answer_start", "id": aid, "trigger": trigger, "question": question, "ts": time.time()})
    ask = (f"My question: {question}" if question
           else "Help me respond to the most recent question or point directed at me.")
    kw = model_opts(ANSWER_MODEL, ANSWER_EFFORT, ANSWER_THINKING == "off")
    api = claude().messages
    if FALLBACKS and ANSWER_MODEL in FALLBACK_MODELS:
        api = claude().beta.messages
        kw.update(betas=[FALLBACK_BETA], fallbacks="default")
    got_text = False
    try:
        async with api.stream(
            model=ANSWER_MODEL,
            max_tokens=4000,
            system=system_blocks(SYSTEM_ANSWER),
            messages=[{"role": "user", "content": transcript_blocks() + [{"type": "text", "text": ask}]}],
            **kw,
        ) as stream:
            async for chunk in stream.text_stream:
                if not got_text:
                    got_text = True
                    ms = int((time.monotonic() - t_ref) * 1000)
                    log.info("answer %s (%s) first token after %d ms", aid, trigger, ms)
                    await broadcast({"type": "answer_meta", "id": aid, "latency_ms": ms,
                                     "from": "speech" if trigger == "auto" else "request"})
                await broadcast({"type": "answer_delta", "id": aid, "text": chunk})
            final = await stream.get_final_message()
        record_usage(getattr(final, "model", None) or ANSWER_MODEL, getattr(final, "usage", None))
        reason = getattr(final, "stop_reason", None)
        if reason == "refusal":
            await broadcast({"type": "answer_delta", "id": aid, "text": "\n(Claude declined to answer this one.)"})
        elif not got_text:
            await broadcast({"type": "answer_delta", "id": aid, "text": f"(No answer text — stop reason: {reason})"})
        await broadcast({"type": "answer_end", "id": aid})
        await broadcast(cost_msg())
    except asyncio.CancelledError:
        await broadcast({"type": "answer_end", "id": aid, "cancelled": True})
        raise
    except Exception as e:  # noqa: BLE001
        await broadcast({"type": "answer_end", "id": aid})
        await notify_error(f"Claude error: {e}", e)


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
        r = await quick(FAST_MODEL, SYSTEM_DETECT, prompt, max_tokens=16)
        verdict = text_of(r).upper()
    except Exception as e:  # noqa: BLE001
        await notify_error(f"Question detection error: {e}", e)
        return
    log.info("auto-check %r -> %s (%d ms after speech)", seg["text"][:60], verdict or "(empty)",
             int((time.monotonic() - seg["mono"]) * 1000))
    if verdict.startswith("YES"):
        start_answer("auto", None, seg["mono"])


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
        r = await quick(FAST_MODEL, system_blocks(SYSTEM_POINTS), transcript_blocks(), max_tokens=400)
        await broadcast({"type": "points", "text": text_of(r), "ts": time.time()})
        await broadcast(cost_msg())
    except Exception as e:  # noqa: BLE001
        await notify_error(f"Talking points error: {e}", e)


# ---------------------------------------------------------------- Claude: rolling summary for long calls
def maybe_summarize() -> None:
    if S.summary_task and not S.summary_task.done():
        return
    live = S.segments[S.summary_upto:]
    if sum(len(fmt_line(s)) + 1 for s in live) <= RECENT_CHARS:
        return
    # summarize the oldest lines until about half of RECENT_CHARS remains word for word
    keep, size, cut = RECENT_CHARS // 2, 0, len(S.segments)
    for i in range(len(S.segments) - 1, S.summary_upto - 1, -1):
        size += len(fmt_line(S.segments[i])) + 1
        if size > keep:
            cut = i + 1
            break
    if cut <= S.summary_upto:
        return
    S.summary_task = asyncio.create_task(summarize(S.summary_upto, cut))


async def summarize(start: int, cut: int) -> None:
    chunk = "\n".join(fmt_line(s) for s in S.segments[start:cut])
    prompt = (f"<previous_summary>\n{S.summary or '(none yet)'}\n</previous_summary>\n\n"
              f"<new_transcript>\n{chunk}\n</new_transcript>\n\n"
              "Write the updated summary covering both.")
    try:
        r = await quick(FAST_MODEL, SYSTEM_SUMMARY, prompt, max_tokens=1200)
        text = text_of(r)
        if text and S.summary_upto == start:  # transcript wasn't cleared meanwhile
            S.summary, S.summary_upto = text, cut
            log.info("summary updated: covers %d segments (%d chars)", cut, len(text))
    except Exception as e:  # noqa: BLE001
        log.warning("summary failed: %s", e)


# ---------------------------------------------------------------- post-call recap
def recap_path(transcript: Path) -> Path:
    return transcript.with_name(transcript.stem + ".summary.md")


def start_recap(transcript: Path | None) -> None:
    """Snapshot the conversation now (it may be cleared next) and write the recap in the background."""
    if not (POST_CALL_SUMMARY and transcript and len(S.segments) >= 3):
        return
    text = "\n".join(fmt_line(s) for s in S.segments)
    system = system_blocks(SYSTEM_RECAP)
    if S.recap_task and not S.recap_task.done():
        S.recap_task.cancel()
    S.recap_task = asyncio.create_task(write_recap(transcript, text, system))


async def write_recap(transcript: Path, text: str, system) -> Path | None:
    await broadcast({"type": "info", "text": "Writing the post-call summary…"})
    try:
        r = await claude().messages.create(
            model=ANSWER_MODEL, max_tokens=8000, system=system,
            messages=[{"role": "user", "content": f"<transcript>\n{text[-400_000:]}\n</transcript>\n\nWrite the recap."}],
            **model_opts(ANSWER_MODEL, "medium", thinking_off=False),
        )
        record_usage(getattr(r, "model", None) or ANSWER_MODEL, getattr(r, "usage", None))
        recap = text_of(r)
        if not recap:
            raise RuntimeError(f"empty response (stop reason: {getattr(r, 'stop_reason', None)})")
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        await notify_error(f"Post-call summary failed: {e}", e)
        return None
    path = recap_path(transcript)
    path.write_text(f"# Recap — {transcript.stem.replace('_', ' ')}\n\n{recap}\n", encoding="utf-8")
    log.info("recap saved: %s", path)
    await broadcast({"type": "recap", "text": recap, "path": rel(path), "ts": time.time()})
    await broadcast(cost_msg())
    return path


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
    except Exception as e:  # noqa: BLE001
        log.warning("no system-audio device: %s", e)
        system = None
    return mic, system


def enqueue(q: asyncio.Queue, item: bytes, source: str) -> None:
    """Runs on the event loop. If Deepgram stalls, drop the oldest chunk so memory stays bounded."""
    if q.full():
        try:
            q.get_nowait()
        except asyncio.QueueEmpty:
            pass
        S.dropped[source] += 1
        if S.dropped[source] % 50 == 1:
            log.warning("audio queue full for %s: dropped %d chunks so far", source, S.dropped[source])
    q.put_nowait(item)


def capture_thread(device, source: str, loop, queue: asyncio.Queue, stop: threading.Event) -> None:
    com = False
    if platform.system() == "Windows":  # soundcard needs COM initialised per thread
        import ctypes

        ctypes.windll.ole32.CoInitializeEx(None, 0)
        com = True
    last_notice = 0.0
    try:
        while not stop.is_set():
            try:
                with device.recorder(samplerate=SAMPLE_RATE, blocksize=CHUNK) as rec:
                    log.info("capture started: %s (%s)", source, device.name)
                    while not stop.is_set():
                        data = rec.record(numframes=CHUNK)
                        mono = data.mean(axis=1) if data.ndim > 1 else data
                        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
                        S.levels[source] = max(S.levels.get(source, 0.0), rms)
                        if rms > SILENCE_RMS:
                            S.heard[source] = True
                        pcm = (np.clip(mono, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                        loop.call_soon_threadsafe(enqueue, queue, pcm, source)
            except Exception as e:  # noqa: BLE001
                log.exception("audio capture failed for %s (%s)", source, device.name)
                if time.time() - last_notice > 60:
                    last_notice = time.time()
                    asyncio.run_coroutine_threadsafe(broadcast(
                        {"type": "error", "text": f"Audio capture failed for {source} ({device.name}): {e} — retrying"}), loop)
                if stop.wait(3):
                    break
    finally:
        log.info("capture stopped: %s", source)
        if com:
            ctypes.windll.ole32.CoUninitialize()


async def levels_loop() -> None:
    """Pushes audio levels to the UI and warns once if a source stays silent."""
    tick = 0
    while True:
        await asyncio.sleep(0.2)
        tick += 1
        if not S.listening:
            continue
        if S.levels:
            await broadcast({"type": "levels", "levels": {k: round(min(v * 4, 1.0), 3) for k, v in S.levels.items()}})
            for k in S.levels:
                S.levels[k] = 0.0
        if tick % 25 == 0:
            await broadcast(cost_msg())
        if S.demo or time.time() - S.listen_since < 15:
            continue
        for src, name in S.device_names.items():
            if not S.heard.get(src) and src not in S.silence_warned:
                S.silence_warned.add(src)
                if src == "mic":
                    hint = ("Check the mic, and on Mac allow Microphone access for your terminal "
                            "(System Settings → Privacy & Security → Microphone), then restart the app.")
                else:
                    hint = ("Is the call playing? On Mac, set the call's speaker to the Multi-Output Device "
                            "that includes BlackHole (see README).")
                await notify_error(f"No sound from {'You' if src == 'mic' else 'Them'} ({name}) for 15 s. {hint}")


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
    """Keep a Deepgram stream open for one audio source; reconnect with backoff on failure."""
    delay = RECONNECT_MIN
    while not stop.is_set():
        try:
            async with websockets.connect(
                dg_query(src["diarize"]),
                additional_headers={"Authorization": f"Token {get_key('deepgram')}"},
                max_size=None, ping_interval=20, open_timeout=10,
            ) as ws:
                log.info("deepgram connected: %s", src["label"])
                delay = RECONNECT_MIN
                await broadcast({"type": "source", "key": src["key"], "state": "live"})
                sender = asyncio.create_task(_dg_send(ws, queue, stop))
                receiver = asyncio.create_task(_dg_recv(ws, src))
                done, pending = await asyncio.wait({sender, receiver}, return_when=asyncio.FIRST_COMPLETED)
                if sender in done and not sender.exception():  # stop requested: let Deepgram flush final results
                    try:
                        await asyncio.wait_for(asyncio.shield(receiver), 3)
                    except Exception:  # noqa: BLE001
                        pass
                for t in pending:
                    t.cancel()
                for t in (sender, receiver):
                    if t.done() and not t.cancelled() and t.exception():
                        raise t.exception()
            if stop.is_set():
                break
            raise ConnectionError("stream closed by Deepgram")
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            if stop.is_set():
                break
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status in (401, 403):
                await notify_error(f"Deepgram rejected the API key (HTTP {status}). Open Settings to fix it.")
                await broadcast({"type": "source", "key": src["key"], "state": "error"})
                return
            log.warning("deepgram %s: %s; reconnecting in %.1fs", src["label"], e, delay)
            await broadcast({"type": "source", "key": src["key"], "state": "reconnecting"})
            if delay >= 2:  # don't toast every blip
                await broadcast({"type": "error", "text": f"Deepgram ({src['label']}): {e} — reconnecting"})
            try:
                await asyncio.wait_for(stop.wait(), delay)
                break
            except asyncio.TimeoutError:
                pass
            delay = min(delay * 2, RECONNECT_MAX)
    log.info("deepgram stream ended: %s", src["label"])


async def _dg_send(ws, queue: asyncio.Queue, stop: asyncio.Event) -> None:
    async def send(chunk: bytes) -> None:
        await ws.send(chunk)
        S.dg_seconds += len(chunk) / (2 * SAMPLE_RATE)

    while not stop.is_set():
        try:
            chunk = await asyncio.wait_for(queue.get(), 1.0)
        except asyncio.TimeoutError:
            await ws.send(json.dumps({"type": "KeepAlive"}))
            continue
        await send(chunk)
    while not queue.empty():  # send what was captured before stop
        await send(queue.get_nowait())
    await ws.send(json.dumps({"type": "CloseStream"}))


async def _dg_recv(ws, src: dict) -> None:
    pending: list[tuple[str, str]] = []  # finalized pieces of the current utterance

    async def flush() -> None:
        nonlocal pending
        merged: list[list[str]] = []  # merge consecutive pieces from the same speaker
        for spk, txt in pending:
            if merged and merged[-1][0] == spk:
                merged[-1][1] += " " + txt
            else:
                merged.append([spk, txt])
        pending = []
        await broadcast({"type": "interim", "source": src["key"], "text": ""})
        for spk, txt in merged:
            await commit(spk, txt, src["key"])

    try:
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
                        await broadcast({"type": "interim", "source": src["key"],
                                         "text": " ".join(t for _, t in pending)})
                elif text:
                    live = " ".join([t for _, t in pending] + [text])
                    await broadcast({"type": "interim", "source": src["key"], "text": live,
                                     "speaker": speaker_of(src, alt)})
            elif kind == "UtteranceEnd" and pending:
                await flush()
            elif kind == "Error" or "err_code" in msg:
                log.warning("deepgram %s error message: %s", src["label"], msg)
    finally:
        if pending:  # don't lose words on disconnect / stop
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
    await asyncio.sleep(1 / DEMO_SPEED)
    for speaker, source, text in DEMO_SCRIPT:
        words = text.split()
        for i in range(1, len(words) + 1, 2):
            if stop.is_set():
                return
            await broadcast({"type": "interim", "source": source, "text": " ".join(words[:i]), "speaker": speaker})
            await asyncio.sleep(0.12 / DEMO_SPEED)
        await broadcast({"type": "interim", "source": source, "text": ""})
        await commit(speaker, text, source)
        try:
            await asyncio.wait_for(stop.wait(), (2.5 if "?" in text else 1.2) / DEMO_SPEED)
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
    S.listen_since = time.time()
    S.heard, S.levels, S.silence_warned, S.device_names = {}, {}, set(), {}
    if S.demo:
        S.tasks.append(asyncio.create_task(demo_feed(S.stop_async)))
    else:
        if not get_key("deepgram"):
            await notify_error("DEEPGRAM_API_KEY is not set. Open Settings (⚙) to add it.")
            return
        try:
            mic, system = await asyncio.to_thread(resolve_devices)
        except Exception as e:  # noqa: BLE001
            await notify_error(f"Could not open audio devices: {e}", e)
            return
        sources = [("mic", mic, "Speaker" if S.mic_only else "You", S.mic_only)]
        if system is not None:
            sources.append(("system", system, "Them", True))
        elif not S.mic_only:
            await notify_error("No system-audio device found — capturing mic only. "
                               "On Mac install BlackHole (see README); or set SYSTEM_DEVICE in .env.")
        for key, dev, label, diarize in sources:
            q: asyncio.Queue = asyncio.Queue(maxsize=AUDIO_QUEUE_MAX)
            S.device_names[key] = dev.name
            threading.Thread(target=capture_thread, args=(dev, key, loop, q, S.stop_thread),
                             name=f"capture-{key}", daemon=True).start()
            S.tasks.append(asyncio.create_task(stt_loop({"key": key, "label": label, "diarize": diarize}, q, S.stop_async)))
            await broadcast({"type": "info", "text": f"{label}: {dev.name}"})
            log.info("source %s -> %s (diarize=%s)", key, dev.name, diarize)
    S.listening = True
    log.info("listening started (mode=%s)", status_msg()["mode"])
    await broadcast(status_msg())


async def stop_listening() -> Path | None:
    if not S.listening:
        return None
    S.stop_async.set()
    S.stop_thread.set()
    if S.tasks:
        _, pending = await asyncio.wait(S.tasks, timeout=5)
        for t in pending:
            t.cancel()
    S.listening = False
    path = save_transcript()
    log.info("listening stopped; dropped audio chunks: %s", dict(S.dropped) or "none")
    await broadcast(status_msg())
    if path:
        await broadcast({"type": "info", "text": f"Transcript saved to {rel(path)}"})
        start_recap(path)
    return path


async def autosave_loop() -> None:
    """Crash insurance: rewrite the transcript file every minute while it changes."""
    while True:
        await asyncio.sleep(60)
        if S.dirty and S.listening:
            try:
                save_transcript()
            except Exception as e:  # noqa: BLE001
                log.warning("autosave failed: %s", e)


# ---------------------------------------------------------------- global hotkey (separate process)
def hotkey_child() -> None:
    """Runs in its own process, so a pynput crash (seen on some macOS versions) can't take the app down."""
    from pynput import keyboard

    def fire() -> None:
        print("HOTKEY", flush=True)

    with keyboard.GlobalHotKeys({HOTKEY: fire}) as h:
        h.wait()
        print("UNTRUSTED" if getattr(h, "IS_TRUSTED", True) is False else "READY", flush=True)
        h.join()


async def hotkey_supervisor() -> None:
    LOG_DIR.mkdir(exist_ok=True)
    errlog = open(LOG_DIR / "hotkey.log", "ab")  # noqa: SIM115
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(ROOT / "copilot.py"), "--hotkey-child",
            stdout=asyncio.subprocess.PIPE, stderr=errlog, stdin=asyncio.subprocess.DEVNULL,
        )
    except Exception as e:  # noqa: BLE001
        errlog.close()
        log.warning("hotkey process failed to start: %s", e)
        return
    S.hotkey_proc = proc
    try:
        async for line in proc.stdout:
            word = line.strip()
            if word == b"HOTKEY":
                log.info("hotkey pressed")
                start_answer("hotkey")
            elif word == b"READY":
                print(f"  Global hotkey: {HOTKEY}  (answer now)")
                log.info("hotkey ready: %s", HOTKEY)
            elif word == b"UNTRUSTED":
                msg = ("Global hotkey needs permission: System Settings → Privacy & Security → Accessibility "
                       "(and Input Monitoring) → enable your terminal app, then restart. Alt+A in the panel still works.")
                print("  ! " + msg)
                await notify_error(msg)
        rc = await proc.wait()
        if not S.shutting_down:
            log.warning("hotkey process exited with code %s (see logs/hotkey.log)", rc)
            await broadcast({"type": "error", "text": "Global hotkey stopped (see logs/hotkey.log). "
                                                      "The button and Alt+A in the panel still work."})
    finally:
        errlog.close()


# ---------------------------------------------------------------- keys / settings
def key_status() -> dict:
    out = {}
    for which in KEY_VARS:
        v = get_key(which)
        out[which] = {"set": bool(v), "hint": f"…{v[-4:]}" if len(v) > 8 else ""}
    return out


def write_env(updates: dict[str, str], path: Path | None = None) -> None:
    """Update KEY=value lines in .env (keeping comments and other settings), readable only by you."""
    path = path or ENV_FILE
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    done = set()
    for i, line in enumerate(lines):
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
        if m and m.group(1) in updates and m.group(1) not in done:
            lines[i] = f"{m.group(1)}={updates[m.group(1)]}"
            done.add(m.group(1))
    lines += [f"{k}={v}" for k, v in updates.items() if k not in done]
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, path)
    os.environ.update(updates)


async def save_keys(values: dict) -> str | None:
    global _client
    updates = {}
    for which, var in KEY_VARS.items():
        v = (values.get(which) or "").strip()
        if not v:
            continue
        if len(v) > 400 or not re.fullmatch(r"[A-Za-z0-9_\-\.]+", v):
            return f"That {which} key doesn't look right (unexpected characters)."
        updates[var] = v
    if not updates:
        return "Nothing to save."
    write_env(updates)
    _client = None  # rebuild the Anthropic client with the new key
    log.info("keys updated: %s", ", ".join(sorted(updates)))  # names only, never values
    return None


async def test_anthropic() -> dict:
    if not get_key("anthropic"):
        return {"ok": False, "text": "not set"}
    import anthropic

    try:
        for m in dict.fromkeys([ANSWER_MODEL, FAST_MODEL]):
            await claude().models.retrieve(m)
        return {"ok": True, "text": f"OK — {ANSWER_MODEL} and {FAST_MODEL} available"}
    except anthropic.AuthenticationError:
        return {"ok": False, "text": "key rejected"}
    except anthropic.NotFoundError as e:
        return {"ok": False, "text": f"model not available to this key: {e}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "text": f"{type(e).__name__}: {e}"}


async def test_deepgram() -> dict:
    key = get_key("deepgram")
    if not key:
        return {"ok": False, "text": "not set"}

    def call() -> dict:
        req = urllib.request.Request(f"{DG_API}/v1/projects", headers={"Authorization": f"Token {key}"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return {"ok": r.status == 200, "text": "OK"}
        except urllib.error.HTTPError as e:
            return {"ok": False, "text": "key rejected" if e.code in (401, 403) else f"HTTP {e.code}"}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "text": f"{type(e).__name__}: {e}"}

    return await asyncio.to_thread(call)


# ---------------------------------------------------------------- web app
@asynccontextmanager
async def lifespan(_app: FastAPI):
    bg = [asyncio.create_task(points_loop()), asyncio.create_task(levels_loop()), asyncio.create_task(autosave_loop())]
    if os.getenv("DISABLE_HOTKEY") != "1":
        bg.append(asyncio.create_task(hotkey_supervisor()))
    yield
    S.shutting_down = True
    log.info("shutting down")
    await stop_listening()
    if S.dirty:
        save_transcript()
    if S.recap_task and not S.recap_task.done():
        print("  Writing the post-call summary… (Ctrl+C again to skip)")
        try:
            await asyncio.wait_for(S.recap_task, 90)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass
    if S.hotkey_proc and S.hotkey_proc.returncode is None:
        S.hotkey_proc.kill()
    for t in bg:
        t.cancel()


app = FastAPI(lifespan=lifespan)


def local_hosts() -> set[str]:
    return {f"127.0.0.1:{S.port}", f"localhost:{S.port}"}


def origin_ok(origin: str | None) -> bool:
    # Browsers always send Origin on websockets; other web pages must not be able to read the transcript.
    return origin is None or origin in {f"http://{h}" for h in local_hosts()}


@app.middleware("http")
async def only_local(request: Request, call_next):
    if request.headers.get("host") not in local_hosts():  # blocks DNS-rebinding tricks
        return PlainTextResponse("forbidden", status_code=403)
    return await call_next(request)


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    if ws.headers.get("host") not in local_hosts() or not origin_ok(ws.headers.get("origin")):
        log.warning("rejected websocket from origin=%s host=%s", ws.headers.get("origin"), ws.headers.get("host"))
        await ws.close(code=1008)
        return
    await ws.accept()
    S.clients.add(ws)
    await ws.send_text(json.dumps({**status_msg(), "type": "init", "segments": S.segments,
                                   "context": S.meeting_context, "keys": key_status(),
                                   "prep": {"docs": prep_docs(), "files": S.prep_files}, "names": S.names, **{"cost": cost_msg()}}))
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            await handle(msg, ws)
    except WebSocketDisconnect:
        pass
    finally:
        S.clients.discard(ws)


async def handle(msg: dict, ws: WebSocket | None = None) -> None:
    action = msg.get("action")

    async def reply(m: dict) -> None:
        if ws is not None:
            await ws.send_text(json.dumps(m))

    if action == "start":
        await start_listening()
    elif action == "stop":
        await stop_listening()
    elif action == "answer":
        start_answer("typed" if msg.get("question") else "button", (msg.get("question") or "").strip() or None)
    elif action == "points_now":
        asyncio.create_task(refresh_points())
    elif action == "set":
        for k in ("auto_answer", "talking_points", "echo_filter"):
            if k in msg:
                setattr(S, k, bool(msg[k]))
        await broadcast(status_msg())
    elif action == "context":
        S.meeting_context = str(msg.get("text", ""))[:20000]
    elif action == "clear":
        start_recap(save_transcript())
        S.segments.clear()
        S.points_mark = S.summary_upto = 0
        S.summary = ""
        S.started_at = None
        S.usage, S.dg_seconds = {}, 0.0
        S.names = {}
        await broadcast({"type": "names", "names": S.names})
        if S.listening:
            S.started_at = time.time()
        await broadcast({"type": "cleared"})
        await broadcast(cost_msg())
    elif action == "rename":
        if rename_speaker(msg.get("label", ""), msg.get("name", "")):
            await broadcast({"type": "names", "names": S.names})
    elif action == "prep":
        if "files" in msg:
            set_prep_files(msg["files"])
            log.info("prep docs for this conversation: %s", S.prep_files or "none")
        await broadcast({"type": "prep", "docs": prep_docs(), "files": S.prep_files})
    elif action == "get_keys":
        await reply({"type": "keys", "keys": key_status()})
    elif action == "save_keys":
        err = await save_keys(msg.get("keys") or {})
        await reply({"type": "keys", "keys": key_status(), "error": err})
        if not err:
            await reply({"type": "key_test", "anthropic": await test_anthropic(), "deepgram": await test_deepgram()})
    elif action == "test_keys":
        await reply({"type": "key_test", "anthropic": await test_anthropic(), "deepgram": await test_deepgram()})


# ---------------------------------------------------------------- self-check (`--check`)
async def self_check() -> int:
    """Checks keys, models, latency, devices and a live Deepgram stream; writes a report to logs/."""
    lines: list[str] = []
    problems = 0

    def out(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    out(f"Live Copilot self-check — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"OS: {platform.platform()}   Python {sys.version.split()[0]}")
    out()
    a = await test_anthropic()
    out(f"[{'ok' if a['ok'] else 'FAIL'}] Anthropic key: {a['text']}")
    problems += not a["ok"]
    if a["ok"]:
        t0 = time.monotonic()
        try:
            first = None
            async with claude().messages.stream(
                model=ANSWER_MODEL, max_tokens=200, system="Reply in five words or fewer.",
                messages=[{"role": "user", "content": "Say hello."}],
                **model_opts(ANSWER_MODEL, ANSWER_EFFORT, ANSWER_THINKING == "off"),
            ) as st:
                async for _ in st.text_stream:
                    first = first or time.monotonic() - t0
            out(f"[ok] {ANSWER_MODEL} first token: {int((first or 0) * 1000)} ms (effort={ANSWER_EFFORT})")
        except Exception as e:  # noqa: BLE001
            problems += 1
            out(f"[FAIL] answer model call: {e}")
        for line, want in (("So realistically, how long would it take to get us audit ready?", "YES"),
                           ("Hey, how's it going? Good weekend?", "NO")):
            try:
                t0 = time.monotonic()
                r = await quick(FAST_MODEL, SYSTEM_DETECT,
                                f'Last line — Them 1: "{line}"\n\nIs the last line a real question the user is expected '
                                "to answer now? Ignore small talk. Reply YES or NO.", 16)
                got = text_of(r).upper()[:3]
                good = got.startswith(want)
                problems += not good
                out(f"[{'ok' if good else 'FAIL'}] question detection ({FAST_MODEL}): {line!r} -> {got or '(empty)'} "
                    f"(want {want}, {int((time.monotonic() - t0) * 1000)} ms)")
            except Exception as e:  # noqa: BLE001
                problems += 1
                out(f"[FAIL] question detection: {e}")
    d = await test_deepgram()
    out(f"[{'ok' if d['ok'] else 'FAIL'}] Deepgram key: {d['text']}")
    problems += not d["ok"]
    out()
    try:
        import soundcard  # noqa: F401

        mic, system = await asyncio.to_thread(resolve_devices)
    except Exception as e:  # noqa: BLE001
        out(f"[FAIL] audio devices: {type(e).__name__}: {e}")
        return _finish_check(lines, problems + 1)
    out(f"Mic (You):            {mic.name}")
    out(f"System audio (Them):  {system.name if system else 'NOT FOUND'}")
    if system is None and not S.mic_only:
        problems += 1
        out("  -> Mac: install BlackHole 2ch and create a Multi-Output Device (README step 3).")
    out()
    out("Recording 4 s from each source. Talk, and play some audio (e.g. a YouTube video) now…")

    def measure(dev) -> tuple[float, bytes]:
        frames = []
        with dev.recorder(samplerate=SAMPLE_RATE, blocksize=CHUNK) as rec:
            for _ in range(40):
                d_ = rec.record(numframes=CHUNK)
                frames.append(d_.mean(axis=1) if d_.ndim > 1 else d_)
        x = np.concatenate(frames)
        return float(np.sqrt(np.mean(np.square(x)))), (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()

    pcm_mic = b""
    for label, dev in (("You", mic), ("Them", system)):
        if dev is None:
            continue
        try:
            if platform.system() == "Windows":
                import ctypes

                ctypes.windll.ole32.CoInitializeEx(None, 0)
            rms, pcm = await asyncio.to_thread(measure, dev)
            db = 20 * np.log10(max(rms, 1e-9))
            heard = rms > SILENCE_RMS
            problems += not heard
            out(f"[{'ok' if heard else 'SILENT'}] {label}: level {db:.0f} dBFS")
            if label == "You":
                pcm_mic = pcm
        except Exception as e:  # noqa: BLE001
            problems += 1
            out(f"[FAIL] {label} capture: {type(e).__name__}: {e}")
    if d["ok"] and pcm_mic:
        out()
        out("Sending the mic recording to Deepgram…")
        try:
            got: list[str] = []
            t0 = time.monotonic()
            async with websockets.connect(dg_query(False), additional_headers={"Authorization": f"Token {get_key('deepgram')}"},
                                          open_timeout=10) as ws:
                for i in range(0, len(pcm_mic), CHUNK * 2):
                    await ws.send(pcm_mic[i:i + CHUNK * 2])
                await ws.send(json.dumps({"type": "CloseStream"}))
                async for raw in ws:
                    if isinstance(raw, str):
                        m = json.loads(raw)
                        if m.get("type") == "Results" and m.get("is_final"):
                            got.append((m["channel"]["alternatives"][0].get("transcript") or "").strip())
            text = " ".join(t for t in got if t)
            out(f"[ok] Deepgram streaming works ({int((time.monotonic() - t0) * 1000)} ms). Heard: {text or '(silence)'}")
        except Exception as e:  # noqa: BLE001
            problems += 1
            out(f"[FAIL] Deepgram streaming: {e}")
    return _finish_check(lines, problems)


def _finish_check(lines: list[str], problems: int) -> int:
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / time.strftime("check-%Y%m%d-%H%M%S.txt")
    lines += ["", "All checks passed." if not problems else f"{problems} problem(s) found."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{lines[-1]}  Report saved to {rel(path)}")
    return 1 if problems else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Live Copilot: real-time transcript -> Claude")
    ap.add_argument("--list-devices", action="store_true", help="list audio devices and exit")
    ap.add_argument("--check", action="store_true", help="test keys, models, audio devices and Deepgram, then exit")
    ap.add_argument("--mic-only", action="store_true", help="in-person mode: one mic, speakers separated by diarization")
    ap.add_argument("--demo", action="store_true", help="play a scripted conversation (no audio or Deepgram needed)")
    ap.add_argument("--prep", default="", help="comma-separated files from context/ to load for this meeting")
    ap.add_argument("--port", type=int, default=_env_int("PORT", 8765))
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--hotkey-child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.hotkey_child:
        hotkey_child()
        return
    if args.list_devices:
        list_devices()
        return
    setup_logging()
    S.mic_only, S.demo, S.port = args.mic_only, args.demo, args.port
    if args.prep:
        set_prep_files([x.strip() for x in args.prep.split(",")])
        print(f"  Prep docs: {', '.join(S.prep_files) or 'none found in context/'}")
    if args.check:
        sys.exit(asyncio.run(self_check()))

    log.info("starting: %s, Python %s, mode=%s, answer=%s/%s, fast=%s",
             platform.platform(), sys.version.split()[0], status_msg()["mode"], ANSWER_MODEL, ANSWER_EFFORT, FAST_MODEL)
    print("\nLive Copilot")
    if not get_key("anthropic"):
        print("  ! ANTHROPIC_API_KEY not set — add it in the panel's Settings (⚙) or in .env")
    if not get_key("deepgram") and not S.demo:
        print("  ! DEEPGRAM_API_KEY not set — add it in Settings (⚙), or run with --demo")
    url = f"http://127.0.0.1:{args.port}"
    print(f"  Panel: {url}\n  Mode: {status_msg()['mode']}   Models: {ANSWER_MODEL} (answers), {FAST_MODEL} (detect/points)")
    print(f"  Log: {rel(LOG_DIR / 'copilot.log')}   Stop with Ctrl+C (the transcript is saved)")
    if not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning",
                                           timeout_graceful_shutdown=3))
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        if S.dirty:  # e.g. a second Ctrl+C skipped the normal shutdown
            save_transcript()
        log.info("exited")


if __name__ == "__main__":
    main()
