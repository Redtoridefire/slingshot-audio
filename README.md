# Live Copilot

Real-time conversation copilot: live transcript of a call → Claude → answers on screen in about 1–2 seconds.

```
 your mic ──────► Deepgram stream ─► "You: …"     ┐
 computer audio ► Deepgram stream ─► "Them 1: …"  ├─► transcript ─► Claude ─► browser panel
   (Zoom/Teams/Meet/anything)       "Them 2: …"   ┘
```

- **Answer now**: button, `Alt+A` in the panel, or a global hotkey (default `Ctrl+Alt+A`)
- **Auto-answer**: when someone else asks you a real question, Claude answers on its own (a fast model checks first, so small talk is skipped)
- **Talking points**: 3 bullets refreshed every ~30 s while the conversation moves
- **Ask box**: type any question about the conversation ("what number did they say?", "summarize their objections")
- **Long calls**: older conversation is summarized automatically, the last few minutes stay word for word
- **On screen**: audio level meters for You and Them, how fast each answer started, and a running cost estimate
- No bot joins the call. It works with any app because it listens to your computer's audio.
- Each transcript is saved to `transcripts/` as Markdown: every minute during a call, when you stop, on Ctrl+C, and on "New conversation".
- **Post-call summary**: when you stop (or start a new conversation), Claude writes a recap with action items, decisions, open questions and a follow-up email draft. It's saved next to the transcript as `….summary.md` and shown in the panel.
- Everything runs on your computer. The panel is only reachable from this machine (127.0.0.1), and no telemetry is sent.

## 1. Start it

Get the code (`git clone https://github.com/Redtoridefire/slingshot-audio.git`, or download the ZIP from GitHub), then:

- **Mac:** double-click `run.command` (or run `./run.command` in Terminal). Needs Python 3.10+: `brew install python@3.12` if you don't have it.
  If macOS says it can't verify the file, right-click it → **Open** once.
- **Windows:** double-click `run.bat`. Needs Python 3.10+ from python.org (tick *Add python.exe to PATH* when installing).

The first run sets everything up (a minute or two); later runs start in seconds. The panel opens at http://127.0.0.1:8765.

**API keys:** the panel asks for them on first start (⚙ button any time). They're saved to `.env` on your computer only, and the panel never shows them again.
- Anthropic key: console.anthropic.com
- Deepgram key: console.deepgram.com (new accounts get free credit)

Edit `context.md` with your background. Claude reads it before every answer.

**Prep docs for a meeting:** put Markdown or text files in the `context/` folder (a prospect's background, your notes, a proposal), then tick them in the panel under *Context for this conversation*, or start with `./run.command --prep acme.md,pricing.md`. Ticked files are added to what Claude knows for this conversation and re-read on every answer, so edits apply mid-call. See `context/example-prospect.md`. Files in `context/` are kept out of git.

The launchers pass options through: `./run.command --demo`, `run.bat --check`, and so on.

<details><summary>Manual setup instead of the launchers</summary>

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
python copilot.py
```
</details>

## 2. Try it without a call

```bash
./run.command --demo      # Windows: run.bat --demo
```

Plays a scripted sales call into the panel so you can watch auto-answer and talking points with real Claude responses. This needs only the Anthropic key.

## 3. Audio setup

### Windows
No setup needed. The app records your default speaker output through Windows' built-in loopback.

### Mac
macOS can't record system audio without a virtual device, so install BlackHole (free):

1. `brew install blackhole-2ch` (or download it from existential.audio/blackhole), then restart your Mac.
2. Open **Audio MIDI Setup**, click **+**, choose **Create Multi-Output Device**, and tick both **BlackHole 2ch** and your headphones/speakers. Put your headphones first in the list and tick **Drift Correction** for BlackHole.
3. Set your call's audio to that Multi-Output Device. You can do this in Zoom/Teams under Settings → Audio → Speaker, or system-wide in System Settings → Sound → Output. You'll still hear the call, and BlackHole gets a copy. (Volume keys don't work on a Multi-Output Device; set the volume in Zoom/Teams.)
4. The first run asks for **Microphone** permission for Terminal: allow it. If you missed it: System Settings → Privacy & Security → Microphone → enable Terminal, then restart the app.
5. The global hotkey needs **Accessibility** and **Input Monitoring** permission for Terminal (same settings page). If you skip that, the in-panel button and `Alt+A` still work, and the panel tells you the hotkey isn't allowed.

### Check everything in one go
```bash
./run.command --check     # Windows: run.bat --check
```
This tests both keys and models, measures how fast Claude starts answering, checks that question detection says YES to a real question and NO to small talk, records 4 seconds from your mic and from the call audio (talk and play a YouTube video while it runs), and sends the mic recording to Deepgram. It prints a report and saves it to `logs/check-….txt`.

```bash
./run.command --list-devices
```
If it picks the wrong device, set `MIC_DEVICE=` / `SYSTEM_DEVICE=` in `.env`. Part of the device name is enough.

## 4. Run

```bash
./run.command              # calls: You (mic) vs Them (computer audio)
./run.command --mic-only   # in-person: one mic, speakers separated as Speaker 1/2/3
```

Press **Start listening**. Click a speaker's name in the transcript (e.g. *Them 1*) to rename them ("Sarah, CFO"); Claude, the saved transcript and the recap all use the name. Speaker numbers come from Deepgram's diarization, so if a connection drops and reconnects mid-call, check the names still match. The You / Them meters in the top right should move when you talk and when the call plays; if one stays flat for 15 seconds, the panel tells you what to check.

**Wear headphones on calls.** Otherwise your mic also picks up the other side from your speakers. If you can't, tick **Echo filter**: it drops your mic lines that repeat what the call said in the last 2 seconds.

Stop with **Stop listening** or Ctrl+C in the terminal; both save the transcript.

### Overlay instead of a browser tab
```bash
./run.command --overlay    # Windows: run.bat --overlay
```
Opens a small always-on-top window with Start/Stop, **Answer now** and the latest answer, so it can sit next to Zoom/Teams. **Full panel ⤢** opens the full panel in your browser when you need the transcript or settings. The first time, it installs `pywebview` automatically. Closing the window quits the app (transcript and recap are saved).

## Tuning (`.env`)

| Setting | Default | What it does |
|---|---|---|
| `ANSWER_MODEL` | `claude-sonnet-5-5` | Model for answers. `claude-haiku-5-5` is faster and cheaper, `claude-opus-5-5` is stronger |
| `ANSWER_EFFORT` | `low` | `low` starts answering fastest; `medium` / `high` think more first |
| `FAST_MODEL` | `claude-haiku-5-5` | Question detection, talking points, summaries of long calls |
| `POINTS_EVERY_SEC` | 30 | How often talking points refresh (only if new speech arrived) |
| `AUTO_COOLDOWN_SEC` | 8 | Minimum gap between auto-answers |
| `HOTKEY` | `<ctrl>+<alt>+a` | Global hotkey; on Mac, `<cmd>+<shift>+a` also works |
| `ECHO_FILTER` | 0 | 1 = start with the echo filter on |
| `RECENT_CHARS` | 12000 | How much recent conversation stays word for word (older parts are summarized) |
| `DEEPGRAM_MODEL` | `nova-3` | Speech-to-text model |
| `DEEPGRAM_PRICE_PER_MIN` | 0.0077 | Used only for the cost estimate; set it to your plan's price |
| `POST_CALL_SUMMARY` | 1 | 0 = don't write the recap when you stop |
| `CLAUDE_FALLBACKS` | 1 | If Claude declines a request, retry it on another model server-side |
| `LOG_LEVEL` | INFO | `DEBUG` also logs each transcript line |

## Troubleshooting

Everything that goes wrong during a call is written to `logs/copilot.log` (rotating, max ~12 MB). The hotkey has its own `logs/hotkey.log`.

- **"Them" meter stays flat on Mac**: the call's audio isn't going to the Multi-Output Device (step 3), or the BlackHole name differs. Check with `--list-devices`.
- **"You" meter stays flat on Mac**: Terminal doesn't have Microphone permission (step 4).
- **Windows: error mentioning `fromstring`**: run `run.bat` again (it updates packages), or `pip install -U soundcard`.
- **Answers feel slow**: the card shows how long the first words took. Most of the delay is the model: try `ANSWER_MODEL=claude-haiku-5-5`.
- **Too many auto-answers**: untick *Auto-answer questions* and use the hotkey, or raise `AUTO_COOLDOWN_SEC`.
- **Hotkey does nothing on Mac**: give Terminal Accessibility and Input Monitoring permission, then restart.

## How it's built

`copilot.py` is one file: `soundcard` capture threads → one Deepgram websocket per source (interim results, endpointing, diarization, reconnect with backoff, bounded audio queues) → transcript list (+ rolling summary) → Claude via the Anthropic SDK (streamed answers, prompt caching on the stable part of the prompt) → FastAPI websocket → `static/index.html`. The global hotkey runs in a small child process, so a crash there can't take the app down.

Tests (fake Deepgram server, fake Claude, no keys needed):

```bash
pip install -r requirements-dev.txt
pytest
```

`python tests/shot.py` (needs Playwright) saves screenshots of the panel running the demo.

Recording laws differ by state. Some, like California, require everyone's consent, so tell people when you're transcribing.
