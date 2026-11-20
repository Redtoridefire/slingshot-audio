@echo off
rem Live Copilot launcher for Windows: double-click this file (or run "run.bat --demo", "--check", "--mic-only").
rem First run creates a private Python environment and installs everything; later runs start in seconds.
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Live Copilot needs Python 3.10 or newer: https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^)
  goto fail
)

if not exist .venv\Scripts\python.exe (
  echo First run: setting up ^(takes a minute or two^)...
  %PY% -m venv .venv || goto fail
)
fc /b requirements.txt .venv\installed-requirements.txt >nul 2>nul
if errorlevel 1 (
  echo Installing / updating packages...
  .venv\Scripts\python.exe -m pip install -q --upgrade pip || goto fail
  .venv\Scripts\python.exe -m pip install -q -r requirements.txt || goto fail
  copy /y requirements.txt .venv\installed-requirements.txt >nul
)
if not exist .env copy .env.example .env >nul

.venv\Scripts\python.exe copilot.py %*
if errorlevel 1 goto fail
exit /b 0

:fail
echo.
pause
exit /b 1
