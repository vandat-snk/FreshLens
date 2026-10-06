@echo off
setlocal
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Khong tim thay .venv\Scripts\python.exe
  pause
  exit /b 1
)
if not exist "models\cnn_efficientnet_b0\open_set_gate.npz" (
  echo [ERROR] Chua co open-set gate. Hay chay SETUP_STEP5.cmd truoc.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run "FreshLens_Buoc5_AppThucTe\APP_CNN.py"
pause
