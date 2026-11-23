@echo off
setlocal
cd /d "%~dp0"
if exist "env\python.exe" (
    "env\python.exe" -u -m shared.desktop.web %*
    exit /b %errorlevel%
)
if exist "env\Scripts\python.exe" (
    "env\Scripts\python.exe" -u -m shared.desktop.web %*
    exit /b %errorlevel%
)
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -u -m shared.desktop.web %*
    exit /b %errorlevel%
)
python -u -m shared.desktop.web %*
exit /b %errorlevel%
