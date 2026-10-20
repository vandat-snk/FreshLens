@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0SETUP_FRESHLENS.ps1" -Device cpu
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
call "%~dp0run_app.bat"
exit /b %errorlevel%
