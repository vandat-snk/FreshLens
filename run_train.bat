@echo off
setlocal
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "FRESHLENS_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%FRESHLENS_PYTHON%" (
    echo [ERROR] Missing .venv\Scripts\python.exe
    echo Create the project virtual environment first.
    exit /b 1
)

rem Forward every command-line argument to the modular CNN V2 trainer.
rem Examples:
rem   run_train.bat --help
rem   run_train.bat --resume
rem   run_train.bat --device cuda
"%FRESHLENS_PYTHON%" TRAIN_CNN_V2.py %*
exit /b %errorlevel%
