#!/bin/bash
# Live Copilot launcher for Mac: double-click this file (or run ./run.command --demo, --check, --mic-only).
# First run creates a private Python environment and installs everything; later runs start in seconds.
cd "$(dirname "$0")" || exit 1

pause_exit() { echo; read -r -p "Press Return to close." _; exit 1; }

PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 10))' 2>/dev/null; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "Live Copilot needs Python 3.10 or newer."
  echo "Install it with:  brew install python@3.12   (or from https://www.python.org/downloads/)"
  pause_exit
fi

if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up (takes a minute or two)…"
  "$PY" -m venv .venv || pause_exit
fi
if ! cmp -s requirements.txt .venv/installed-requirements.txt; then
  echo "Installing / updating packages…"
  .venv/bin/python -m pip install -q --upgrade pip && .venv/bin/python -m pip install -q -r requirements.txt || pause_exit
  cp requirements.txt .venv/installed-requirements.txt
fi
[ -f .env ] || { cp .env.example .env; chmod 600 .env; }

.venv/bin/python copilot.py "$@" || pause_exit
