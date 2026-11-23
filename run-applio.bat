@echo off
setlocal
cd /d "%~dp0"
if exist "Redpanda.exe" (
    start "" "Redpanda.exe"
    exit /b
)
if exist "desktop\target\release\redpanda.exe" (
    start "" "desktop\target\release\redpanda.exe"
    exit /b
)
echo Build with cargo build --release --locked --manifest-path desktop\Cargo.toml
exit /b 1
