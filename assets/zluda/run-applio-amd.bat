@echo off
setlocal
title Redpanda VC

if not exist env (
    echo Please run 'run-install.bat' first to set up the environment.
    pause
    exit /b 1
)

set HIP_VISIBLE_DEVICES="0"
set ZLUDA_COMGR_LOG_LEVEL=1
SET DISABLE_ADDMM_CUDA_LT=1
set "REDPANDA_ZLUDA=%CD%\zluda\zluda.exe"
call run-applio.bat
