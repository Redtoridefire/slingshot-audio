"""Build a standalone Live Copilot folder with PyInstaller. Run it on the OS you are building for:

    pip install pyinstaller
    python packaging/build.py            -> dist/LiveCopilot/LiveCopilot(.exe)

The build keeps your files (.env, context.md, context/, transcripts/, logs/) in ~/LiveCopilot, created on
first run. It's a terminal app: on a Mac, run it from Terminal (which holds the Microphone permission).
Optional features are included if their packages are installed when you build (pywebview, openai, faster-whisper).
"""
import importlib.util
import os
import sys
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent.parent
sep = os.pathsep

args = [
    str(ROOT / "copilot.py"), "--name", "LiveCopilot", "--onedir", "--console", "--noconfirm", "--clean",
    "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"), "--specpath", str(ROOT / "build"),
    "--add-data", f"{ROOT / 'static'}{sep}static",
    "--add-data", f"{ROOT / '.env.example'}{sep}.",
    "--add-data", f"{ROOT / 'context.md'}{sep}.",
    "--collect-data", "soundcard",        # soundcard loads C header files at runtime
    "--collect-submodules", "pynput",     # platform backends are imported dynamically
    "--collect-submodules", "uvicorn",
]
for module, how in (("webview", "--collect-all"), ("openai", "--collect-submodules"), ("faster_whisper", "--collect-all")):
    if importlib.util.find_spec(module):
        args += [how, module]
        print(f"including optional {module}")

PyInstaller.__main__.run(args)
exe = ROOT / "dist" / "LiveCopilot" / ("LiveCopilot.exe" if sys.platform == "win32" else "LiveCopilot")
print(f"\nBuilt {exe}\nTry: {exe} --demo")
