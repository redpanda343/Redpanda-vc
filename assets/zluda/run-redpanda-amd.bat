@echo off
setlocal
cd /d "%~dp0..\.."
title Redpanda VC (ZLUDA)

set HIP_VISIBLE_DEVICES="0"
set ZLUDA_COMGR_LOG_LEVEL=1
set DISABLE_ADDMM_CUDA_LT=1
set "REDPANDA_ZLUDA=%CD%\zluda\zluda.exe"

if exist "Redpanda.exe" (
    start "" "Redpanda.exe"
    exit /b
)
if exist "dist\Redpanda.exe" (
    start "" "dist\Redpanda.exe"
    exit /b
)
if exist "desktop\target\release\redpanda.exe" (
    start "" "desktop\target\release\redpanda.exe"
    exit /b
)
echo Redpanda.exe not found. Build with: cargo build --release --locked --manifest-path desktop\Cargo.toml
pause
exit /b 1
