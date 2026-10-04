@echo off
setlocal
cd /d "%~dp0"
set "FRESHLENS_PYTHON=%~dp0.venv\Scripts\python.exe"
set "PYTHONUTF8=1"
if not exist "%FRESHLENS_PYTHON%" (
    echo [LOI] Chua co .venv\Scripts\python.exe. Hay tao virtualenv cua du an.
    exit /b 1
)
if not "%~1"=="" set "FRESHLENS_DATASET_ROOT=%~1"
rem src.train validates the existing manifest and stops on errors.
rem Never regenerate a MongoDB-exported manifest here.
"%FRESHLENS_PYTHON%" -m src.train --ablation
exit /b %errorlevel%
