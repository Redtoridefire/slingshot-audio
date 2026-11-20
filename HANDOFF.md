# Claude Code Handoff — Live Copilot

Paste everything below the line into Claude Code, run from inside the `live-copilot/` folder.

---

You're taking over **Live Copilot**, a small local app I (Mike) use for my own business and personal conversations, outside my day job. It listens to a live conversation (Zoom/Teams/Meet/phone call, or in person), transcribes it in real time with Deepgram, and uses Claude to put answers, auto-answers to questions aimed at me, and rolling talking points on screen in about 1–2 seconds. Read `README.md`, `copilot.py`, `static/index.html`, and `tests/` before changing anything.

## Goal
Get this from "works with mocks in a sandbox" to "reliable on my real machine during real calls". Then harden it and add the backlog items below, keeping it lightweight: one Python process, a local browser panel, no cloud backend, no meeting bot.

## Current architecture (as built)
- `copilot.py` (~600 lines, single file):
  - **Audio:** `soundcard` capture threads at 16 kHz mono int16, 100 ms chunks. Mic is labeled **You**. Computer audio is labeled **Them**: Windows/Linux use loopback of the default speaker, Mac uses the BlackHole virtual device. Overrides: `MIC_DEVICE` / `SYSTEM_DEVICE`. `--mic-only` is for in-person use (one mic, diarized into Speaker 1/2/…).
  - **STT:** one raw `websockets` (>=14, uses `additional_headers`) connection per source to Deepgram `nova-3`, with `interim_results`, `endpointing=300`, `utterance_end_ms=1000`, `vad_events`, `smart_format`, and diarize (on for Them and for mic-only). KeepAlive is sent when idle, CloseStream on stop, and it auto-reconnects. Final pieces are buffered until `speech_final` or `UtteranceEnd`, then committed as segments merged by speaker. Interim text is broadcast for live display.
  - **LLM:** Anthropic SDK `AsyncAnthropic`. `run_answer` streams answers with `ANSWER_MODEL` (default `claude-sonnet-5-5`). `auto_check` runs a regex pre-filter, then a YES/NO check with `FAST_MODEL` (`claude-haiku-5-5`); it only fires on non-mic speech (or any speech in mic-only), with a cooldown. `points_loop` refreshes 3 bullets every `POINTS_EVERY_SEC` when new speech has arrived. A new answer cancels the one in progress. The system prompt includes `context.md` (re-read on every call) and the per-meeting context from the UI.
  - **Server:** FastAPI + uvicorn bound to **127.0.0.1:8765**, with a single `/ws` websocket for the UI. Actions: start, stop, answer (optional question), points_now, set (toggles), context, clear.
  - Transcripts are saved as Markdown to `transcripts/` on stop and on "New conversation".
  - The global hotkey uses `pynput` (default `<ctrl>+<alt>+a`) and fails gracefully.
  - `--demo` plays a scripted call with no audio or Deepgram, so only the Anthropic key is needed. `--list-devices` lists audio devices.
- `static/index.html`: a single-file vanilla JS panel showing the transcript (with interim), answer cards (streaming, newest first), talking points, an ask box, toggles, and Alt+A. It reconnects automatically.
- `tests/`: `test_copilot.py` (fake Deepgram websocket server + fake Claude client: parsing, diarization labels, auto-answer and filtering), `test_app.py` (full app in demo mode over the websocket), and `shot.py` (Playwright screenshot of the UI). They're plain scripts, not pytest yet, and `test_app.py` imports `test_copilot.py`, which runs that file's checks on import.

## What has and hasn't been verified
- **Verified with mocks:** Deepgram message handling, speaker labeling, auto-answer filtering, typed and button answers, talking points, transcript save, and UI rendering with no JS errors.
- **Never run for real:** `soundcard` capture on Mac or Windows, a live Deepgram connection, live Claude calls, `pynput` on my OS, and long (60+ minute) sessions.

## Phase 1 — Make it work on my machine (do this first, with me)
1. Create a venv, `pip install -r requirements.txt`, and copy `.env.example` to `.env`. I'll paste my keys myself; never print or commit them.
2. Confirm the model IDs `claude-sonnet-5-5` / `claude-haiku-5-5` work with a one-line API call. If not, list the available models and fix the defaults.
3. Run `python copilot.py --demo` and confirm real Claude answers, auto-answers, and talking points stream into the panel.
4. Run `python copilot.py --list-devices` and set up system audio:
   - **Mac:** walk me through BlackHole 2ch and a Multi-Output Device, then the Microphone and Accessibility/Input Monitoring permissions.
   - **Windows:** confirm default-speaker loopback works. Watch for the COM init issue (`CoInitializeEx` is already called per thread) and numpy 2 / `fromstring` errors in soundcard; pin versions if needed.
5. Live test: play a YouTube interview through the speakers with headphones on, check that You and Them labels are correct, the delay feels fast enough, and diarization separates speakers. Then do a real Zoom test call.
6. Fix whatever breaks. Add a log file (`logs/copilot.log`, rotating) so failures during a call can be diagnosed afterwards.

## Phase 2 — Harden
- Convert the tests to **pytest** (with pytest-asyncio), remove the import side effect, and keep the fake Deepgram and fake Claude fixtures. Add tests for reconnect, stop/flush, cooldown, cancelling an answer in progress, and trimming long transcripts.
- **Long calls:** replace the hard tail-trim (`MAX_CONTEXT_CHARS`) with a rolling summary of older conversation plus the recent part word for word. Use Anthropic **prompt caching** on the stable prefix (system prompt + `context.md`) to cut delay and cost.
- **Backpressure:** cap the audio queues (drop the oldest chunk if Deepgram stalls) so memory can't grow without limit.
- **Echo handling:** optionally drop mic segments that closely match a system segment from the last 2 seconds, for when I'm not wearing headphones.
- Measure and show **latency** in the UI: time from end of speech to first answer token.
- Show a **running cost** for the session in the UI: Deepgram minutes plus Claude input/output tokens.
- Handle shutdown cleanly on Ctrl+C: stop the threads, flush Deepgram, save the transcript.

## Phase 3 — Backlog (ask me which to do; suggest an order)
- An **always-on-top compact overlay** (e.g. a pywebview window) showing just the latest answer, so I don't need a browser tab during calls.
- **Provider switch:** `LLM_PROVIDER=anthropic|openai` behind a small interface. Same idea for STT: Deepgram, AssemblyAI, or local faster-whisper.
- **Post-call summary:** on stop, generate a summary with action items and save it next to the transcript.
- **Prep docs:** load extra Markdown files (e.g. `context/*.md`, a prospect's background) into the context for a given meeting.
- **Speaker renaming** in the UI (Them 1 → "Sarah, CFO"), passed to Claude.
- **Packaging:** a one-command launcher for each OS (`run.command` / `run.bat`), and optionally a PyInstaller build.

## Constraints
- It's local only: keep the server bound to 127.0.0.1, no telemetry, and keys only in `.env` (already in `.gitignore`).
- Keep it lightweight and easy to read. Don't add a frontend framework or a database unless I ask.
- Keep the demo mode and the mock tests working after every change, and run the tests before saying something is done.
- Initialize a git repo and commit after each phase with clear messages.
- Recording laws differ by state, so keep the README note about telling people they're being transcribed.

## Definition of done for Phase 1
On my machine, during a real Zoom call with headphones on: You and Them are labeled correctly; transcript text appears in under 1 second; the hotkey answer starts showing in under 2.5 seconds; auto-answer fires on a direct question and not on small talk; and stopping saves the transcript. Tell me exactly what you verified and what you didn't.

Start by reading the code, then tell me in a few lines what you plan to do and which OS you're assuming, and ask me anything you need, before Phase 1 step 4.
