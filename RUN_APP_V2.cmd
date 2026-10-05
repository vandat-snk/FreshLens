@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Khong tim thay .venv\Scripts\python.exe
    echo Hay tao/cai virtual environment truoc.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m streamlit run APP_CNN_V2.py
