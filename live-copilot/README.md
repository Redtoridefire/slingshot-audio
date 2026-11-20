# Live Copilot

Real-time conversation copilot: live transcript of a call → Claude → answers on screen in ~1–2 s.

```
 your mic ──────► Deepgram stream ─► "You: …"     ┐
 computer audio ► Deepgram stream ─► "Them 1: …"  ├─► transcript ─► Claude ─► browser panel
   (Zoom/Teams/Meet/anything)       "Them 2: …"   ┘
```

- **Answer now**: button, `Alt+A` in the panel, or a global hotkey (default `Ctrl+Alt+A`)
- **Auto-answer**: when someone else asks you a real question, Claude answers on its own (a fast model checks first, so small talk is skipped)
- **Talking points**: 3 bullets refreshed every ~30 s while the conversation moves
- **Ask box**: type any question about the conversation ("what number did they say?", "summarize their objections")
- No bot joins the call. It works with any app because it listens to your computer's audio.
- Each transcript is saved to `transcripts/` as Markdown when you stop or start a new conversation.

## 1. Install (Mac or Windows, Python 3.10+)

```bash
cd live-copilot
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # Windows: copy .env.example .env
```

Put your keys in `.env`:
- `ANTHROPIC_API_KEY` from console.anthropic.com
- `DEEPGRAM_API_KEY` from console.deepgram.com (new accounts get free credit)

Edit `context.md` with your background. Claude reads it before every answer.

## 2. Test without a call

```bash
python copilot.py --demo
```

Plays a scripted sales call into the panel so you can watch auto-answer and talking points with real Claude responses. This needs only the Anthropic key.

## 3. Audio setup

### Windows
No setup needed. The app records your default speaker output through Windows' built-in loopback.

### Mac
macOS can't record system audio without a virtual device, so install BlackHole (free):

1. `brew install blackhole-2ch` (or download it from existential.audio/blackhole), then restart your Mac.
2. Open **Audio MIDI Setup**, click **+**, choose **Create Multi-Output Device**, and tick both **BlackHole 2ch** and your headphones/speakers.
3. Set your call's audio to that Multi-Output Device. You can do this in Zoom/Teams under Settings → Audio → Speaker, or system-wide in System Settings → Sound → Output. You'll still hear the call, and BlackHole gets a copy.
4. The first run asks for **Microphone** permission for your terminal. The global hotkey also needs **Accessibility / Input Monitoring** permission. If you skip that, the in-panel button and `Alt+A` still work.

### Check devices
```bash
python copilot.py --list-devices
```
If it picks the wrong device, set `MIC_DEVICE=` / `SYSTEM_DEVICE=` in `.env`. Part of the device name is enough.

## 4. Run

```bash
python copilot.py              # calls: You (mic) vs Them (computer audio)
python copilot.py --mic-only   # in-person: one mic, speakers separated as Speaker 1/2/3
```

The panel opens at http://127.0.0.1:8765. Press **Start listening**.

**Wear headphones on calls.** Otherwise your mic also picks up the other side from your speakers, and their words show up twice.

## Tuning (`.env`)

| Setting | Default | What it does |
|---|---|---|
| `ANSWER_MODEL` | `claude-sonnet-5-5` | Model for answers. Use `claude-haiku-5-5` for the fastest responses |
| `FAST_MODEL` | `claude-haiku-5-5` | Question detection and talking points |
| `POINTS_EVERY_SEC` | 30 | How often talking points refresh (only if new speech arrived) |
| `AUTO_COOLDOWN_SEC` | 8 | Minimum gap between auto-answers |
| `HOTKEY` | `<ctrl>+<alt>+a` | Global hotkey; on Mac, `<cmd>+<shift>+a` also works |
| `DEEPGRAM_MODEL` | `nova-3` | Speech-to-text model |

## Troubleshooting

- **No "Them" lines on Mac**: the call's audio isn't going to the Multi-Output Device (step 3), or the BlackHole name differs. Check with `--list-devices`.
- **Windows: error mentioning `fromstring`**: run `pip install -U soundcard`, or `pip install "numpy<2"`.
- **Answers feel slow**: set `ANSWER_MODEL=claude-haiku-5-5`. Most of the delay is the model, not the transcription.
- **Too many auto-answers**: untick *Auto-answer questions* and use the hotkey, or raise `AUTO_COOLDOWN_SEC`.

## How it's built

`copilot.py` is about 600 lines: `soundcard` capture threads → one Deepgram websocket per source (interim results, endpointing, diarization) → transcript list → Claude via the Anthropic SDK (streamed answers) → FastAPI websocket → `static/index.html`. To use ChatGPT instead, replace the three `claude()` calls (`run_answer`, `auto_check`, `refresh_points`) with OpenAI's equivalents.

Recording laws differ by state. Some, like California, require everyone's consent, so tell people when you're transcribing.
