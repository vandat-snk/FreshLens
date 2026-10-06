@echo off
setlocal
cd /d "%~dp0"

set "PYTHONUTF8=1"
set "FRESHLENS_PYTHON=%~dp0.venv\Scripts\python.exe"

if not exist "%FRESHLENS_PYTHON%" (
    echo [ERROR] Missing .venv\Scripts\python.exe
    echo Create the project virtual environment first.
    pause
    exit /b 1
)

"%FRESHLENS_PYTHON%" -m streamlit run APP_CNN_V2.py
exit /b %errorlevel%
