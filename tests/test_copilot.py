"""Unit tests: Deepgram handling, auto-answer, answers, long-call memory, echo, backpressure, keys."""
import asyncio
import json

import pytest
from conftest import C, content_text, dg_result


async def run_stt(src, chunks=5, run_for=1.0, queue=None):
    q = queue or asyncio.Queue()
    for _ in range(chunks):
        q.put_nowait(b"\x00\x00" * 1600)
    stop = asyncio.Event()
    t = asyncio.create_task(C.stt_loop(src, q, stop))
    await asyncio.sleep(run_for)
    stop.set()
    await asyncio.wait_for(t, 6)


THEM = {"key": "system", "label": "Them", "diarize": True}


# ---------------------------------------------------------------- Deepgram
async def test_deepgram_parsing_labels_and_auto_answer(fake_claude, sent, fake_dg):
    async def handler(ws):
        assert ws.request.headers["Authorization"] == "Token testkey"
        assert "diarize=true" in ws.request.path
        n = 0
        async for m in ws:
            if isinstance(m, bytes):
                n += 1
                if n == 1:
                    await ws.send(dg_result("How long", False, spk=1))
                    await ws.send(dg_result("How long would it take", True, spk=1))
                    await ws.send(dg_result("to get audit ready?", True, True, spk=1))
                    await ws.send(dg_result("Thanks.", True, spk=0))
                    await ws.send(json.dumps({"type": "UtteranceEnd"}))
            elif json.loads(m)["type"] == "CloseStream":
                await ws.close()
                return

    await fake_dg(handler)
    await run_stt(THEM)
    await asyncio.sleep(0.3)
    finals = [(m["speaker"], m["text"]) for m in sent if m["type"] == "final"]
    assert finals == [("Them 2", "How long would it take to get audit ready?"), ("Them 1", "Thanks.")]
    assert any(m["type"] == "interim" and m["text"] == "How long" for m in sent)
    kinds = [m["type"] for m in sent]
    assert "answer_start" in kinds and "answer_end" in kinds
    assert "".join(m["text"] for m in sent if m["type"] == "answer_delta").startswith("**Roughly")
    kw = fake_claude.messages.of("stream")[0]
    assert "Them 2: How long would it take" in content_text(kw)
    assert "Mike Czarnecki" in kw["system"][0]["text"]
    assert C.S.dg_seconds == pytest.approx(0.5)


async def test_reconnects_after_drop_and_keeps_pending_words(fake_claude, sent, fake_dg):
    conns = []

    async def handler(ws):
        conns.append(ws)
        if len(conns) > 1:
            await ws.send(dg_result("second connection words", True, True))
        async for m in ws:
            if isinstance(m, bytes):
                if len(conns) == 1:
                    # a finished piece with no speech_final, then the connection dies
                    await ws.send(dg_result("first connection words", True))
                    await ws.close(code=1011)
                    return
            elif json.loads(m)["type"] == "CloseStream":
                await ws.close()
                return

    await fake_dg(handler)
    C.S.auto_answer = False
    await run_stt(THEM, chunks=10, run_for=1.0)
    texts = [m["text"] for m in sent if m["type"] == "final"]
    assert len(conns) >= 2
    assert "first connection words" in texts and "second connection words" in texts


async def test_stop_flushes_unfinished_utterance(fake_claude, sent, fake_dg):
    async def handler(ws):
        async for m in ws:
            if isinstance(m, bytes):
                await ws.send(dg_result("words without an ending", True))
            elif json.loads(m)["type"] == "CloseStream":
                await ws.send(dg_result("and the last bit", True))
                await ws.close()
                return

    await fake_dg(handler)
    C.S.auto_answer = False
    await run_stt(THEM, chunks=1, run_for=0.4)
    finals = [m["text"] for m in sent if m["type"] == "final"]
    assert finals == ["words without an ending and the last bit"]


async def test_send_drains_queue_before_close():
    class WS:
        def __init__(self):
            self.sent = []

        async def send(self, x):
            self.sent.append(x)

    q, stop, ws = asyncio.Queue(), asyncio.Event(), WS()
    for i in range(5):
        q.put_nowait(bytes([i]) * 3200)
    stop.set()
    await C._dg_send(ws, q, stop)
    assert len([x for x in ws.sent if isinstance(x, bytes)]) == 5
    assert json.loads(ws.sent[-1]) == {"type": "CloseStream"}


async def test_rejected_key_stops_retrying(fake_claude, sent, monkeypatch):
    from conftest import free_port

    port = free_port()

    async def reject(conn, request):
        return conn.respond(401, "nope")

    import websockets
    srv = await websockets.serve(lambda ws: None, "127.0.0.1", port, process_request=reject)
    monkeypatch.setattr(C, "DG_URL", f"ws://127.0.0.1:{port}/v1/listen")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "badkey")
    stop = asyncio.Event()
    await asyncio.wait_for(C.stt_loop(THEM, asyncio.Queue(), stop), 3)  # returns on its own
    srv.close()
    assert any("rejected the API key" in m.get("text", "") for m in sent)


def test_backpressure_drops_oldest():
    C.reset_state()
    q = asyncio.Queue(maxsize=3)
    for i in range(5):
        C.enqueue(q, bytes([i]), "mic")
    assert [q.get_nowait() for _ in range(3)] == [b"\x02", b"\x03", b"\x04"]
    assert C.S.dropped["mic"] == 2


# ---------------------------------------------------------------- auto-answer
async def test_auto_answer_filtering(fake_claude, sent):
    await C.commit("You", "What do you think?", "mic")
    await C.commit("Them 1", "We use AWS.", "system")
    await asyncio.sleep(0.1)
    assert fake_claude.messages.calls == []


async def test_auto_answer_cooldown(fake_claude, sent):
    await C.commit("Them 1", "How long would it take?", "system")
    await asyncio.sleep(0.2)
    await C.commit("Them 1", "And what would it cost?", "system")
    await asyncio.sleep(0.2)
    assert len([kw for kw in fake_claude.messages.of("create") if kw["system"] == C.SYSTEM_DETECT]) == 1


async def test_detector_no_means_no_answer(fake_claude, sent):
    fake_claude.messages.verdict = "NO"
    await C.commit("Them 1", "How are you doing today?", "system")
    await asyncio.sleep(0.2)
    assert fake_claude.messages.of("stream") == []


async def test_detector_call_has_room_and_no_thinking(fake_claude, sent):
    await C.commit("Them 1", "How long would it take?", "system")
    await asyncio.sleep(0.2)
    kw = fake_claude.messages.of("create")[0]
    assert kw["model"] == "claude-haiku-5-5"
    assert kw["thinking"] == {"type": "disabled"} and kw["max_tokens"] >= 16
    assert kw["output_config"] == {"effort": "low"}


# ---------------------------------------------------------------- answers
async def test_new_answer_cancels_one_in_progress(fake_claude, sent):
    fake_claude.messages.delay = 0.3
    C.start_answer("button")
    await asyncio.sleep(0.1)
    first = C.S.answer_task
    C.start_answer("typed", "Second question?")
    await asyncio.sleep(0.05)
    assert first.cancelled()
    ends = [m for m in sent if m["type"] == "answer_end"]
    assert ends and ends[0].get("cancelled")
    await asyncio.wait_for(C.S.answer_task, 3)
    assert "Second question?" in content_text(fake_claude.messages.of("stream")[1])


async def test_answer_latency_cost_and_fallback_settings(fake_claude, sent):
    C.start_answer("hotkey")
    await asyncio.wait_for(C.S.answer_task, 3)
    meta = [m for m in sent if m["type"] == "answer_meta"]
    assert len(meta) == 1 and 0 <= meta[0]["latency_ms"] < 1000 and meta[0]["from"] == "request"
    kw = fake_claude.messages.of("stream")[0]
    assert kw["model"] == "claude-sonnet-5-5"
    assert kw["fallbacks"] == "default" and kw["betas"] == [C.FALLBACK_BETA]
    assert kw["output_config"] == {"effort": "low"} and "thinking" not in kw
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    cost = [m for m in sent if m["type"] == "cost"][-1]
    assert cost["claude"] == pytest.approx((1000 * 2.0 + 100 * 10.0) / 1e6)


async def test_refusal_is_shown(fake_claude, sent):
    fake_claude.messages.stop_reason = "refusal"
    C.start_answer("button")
    await asyncio.wait_for(C.S.answer_task, 3)
    assert "declined" in "".join(m["text"] for m in sent if m["type"] == "answer_delta")


def test_model_opts():
    assert C.model_opts("claude-sonnet-5-5", "low", True) == {"output_config": {"effort": "low"},
                                                              "thinking": {"type": "between_tools"}}
    assert C.model_opts("claude-opus-5-5", "low", True) == {"output_config": {"effort": "low"}}
    assert C.model_opts("claude-haiku-5-5", "low", True)["thinking"] == {"type": "disabled"}
    assert C.model_opts("some-old-model", "low", False) == {}


# ---------------------------------------------------------------- long calls
async def test_rolling_summary_and_cacheable_blocks(fake_claude, sent, monkeypatch):
    monkeypatch.setattr(C, "RECENT_CHARS", 3000)
    C.S.auto_answer = False
    for i in range(80):
        await C.commit("Them 1" if i % 2 else "You", f"line {i}: " + "we talked about access reviews " * 2,
                       "system" if i % 2 else "mic")
        if C.S.summary_task:
            await C.S.summary_task
    assert C.S.summary.startswith("- They are")
    assert 0 < C.S.summary_upto < 80
    blocks = C.transcript_blocks()
    assert blocks[0]["cache_control"] == {"type": "ephemeral"} and "summary_of_earlier_conversation" in blocks[0]["text"]
    assert "line 79:" in blocks[-1]["text"] and "line 0:" not in "".join(b["text"] for b in blocks)
    assert sum(len(b["text"]) for b in blocks) < 3000 + 1000
    assert C.transcript_blocks() == blocks  # stable between calls -> cache hits
    # the stable part changes only every FREEZE_STEP segments
    await C.commit("Them 1", "one more line", "system")
    assert C.transcript_blocks()[0] == blocks[0]


def test_hard_cap_when_summary_is_missing(monkeypatch):
    C.reset_state()
    monkeypatch.setattr(C, "MAX_CONTEXT_CHARS", 2000)
    C.S.segments = [{"speaker": "Them 1", "text": "x" * 100, "ts": 0, "source": "system"} for _ in range(100)]
    assert sum(len(b["text"]) for b in C.transcript_blocks()) < 2600


# ---------------------------------------------------------------- echo filter
async def test_echo_filter(fake_claude, sent):
    C.S.echo_filter, C.S.auto_answer = True, False
    await C.commit("Them 1", "We are mostly on AWS with some Azure", "system")
    await C.commit("You", "we're mostly on AWS with some Azure", "mic")  # mic heard the speakers
    await C.commit("You", "Got it, thanks for the background", "mic")   # really me
    assert [s["text"] for s in C.S.segments] == ["We are mostly on AWS with some Azure", "Got it, thanks for the background"]
    # mic line arrives first, then the matching call audio -> mic line is retracted
    await C.commit("You", "our board wants a security update next quarter", "mic")
    mic_id = C.S.segments[-1]["id"]
    await C.commit("Them 2", "And our board wants a security update next quarter.", "system")
    assert {"type": "retract", "id": mic_id} in sent
    assert all(s["id"] != mic_id for s in C.S.segments)


async def test_echo_filter_off_keeps_everything(fake_claude, sent):
    C.S.auto_answer = False
    await C.commit("Them 1", "We are mostly on AWS with some Azure", "system")
    await C.commit("You", "we're mostly on AWS with some Azure", "mic")
    assert len(C.S.segments) == 2


# ---------------------------------------------------------------- transcript + keys
async def test_transcript_saved_with_cost(fake_claude, sent):
    C.S.auto_answer = False
    await C.commit("Them 1", "Hello there", "system")
    path = C.save_transcript()
    body = path.read_text()
    assert "**Them 1**" in body and "Estimated cost" in body and not C.S.dirty


def test_write_env_keeps_other_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    env = tmp_path / ".env"
    env.write_text("# comment\nANTHROPIC_API_KEY=sk-ant-...\nPORT=8765\n")
    C.write_env({"ANTHROPIC_API_KEY": "sk-ant-abc123456789", "DEEPGRAM_API_KEY": "dg123456789"}, env)
    assert env.read_text() == "# comment\nANTHROPIC_API_KEY=sk-ant-abc123456789\nPORT=8765\nDEEPGRAM_API_KEY=dg123456789\n"
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg123456789")
    st = C.key_status()
    assert st["anthropic"] == {"set": True, "hint": "…6789"}
    assert "abc" not in json.dumps(st)


async def test_save_keys_validates_and_treats_placeholder_as_unset(fake_claude, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-...")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "")
    assert C.key_status()["anthropic"]["set"] is False
    assert "doesn't look right" in await C.save_keys({"anthropic": "bad key\nX=1"})
    assert await C.save_keys({"deepgram": "abcdef0123456789"}) is None
    assert C.get_key("deepgram") == "abcdef0123456789"
    assert C._client is None  # client rebuilt with the new key


# ---------------------------------------------------------------- post-call summary
async def test_recap_written_next_to_transcript(fake_claude, sent):
    C.S.auto_answer = False
    for t in ("We need SOC 2 by March.", "Can you send a proposal?", "Sure, I'll send it Friday."):
        await C.commit("Them 1", t, "system")
    path = C.save_transcript()
    C.start_recap(path)
    await asyncio.wait_for(C.S.recap_task, 3)
    recap = C.recap_path(path)
    assert recap.name.endswith(".summary.md") and "send proposal" in recap.read_text()
    kw = [k for k in fake_claude.messages.of("create") if k["system"][0]["text"].startswith(C.SYSTEM_RECAP)][0]
    assert "Sure, I'll send it Friday." in kw["messages"][0]["content"]
    assert any(m["type"] == "recap" for m in sent)


async def test_no_recap_for_tiny_or_disabled(fake_claude, sent, monkeypatch):
    await C.commit("Them 1", "Hi", "system")
    C.start_recap(C.save_transcript())
    assert C.S.recap_task is None
