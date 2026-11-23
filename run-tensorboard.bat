@echo off
setlocal
cd /d "%~dp0"
title Tensorboard

if exist "env\python.exe" (
    set "PYTHON_EXE=env\python.exe"
) else if exist "env\Scripts\python.exe" (
    set "PYTHON_EXE=env\Scripts\python.exe"
) else (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

"%PYTHON_EXE%" core.py tensorboard
echo.
pause
