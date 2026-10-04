@echo off
setlocal
cd /d "%~dp0.."
echo ============================================================
echo FreshLens - Step 5 setup: open-set gate + final CNN app
echo ============================================================
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Khong tim thay .venv\Scripts\python.exe
  pause
  exit /b 1
)
if not exist "models\cnn_efficientnet_b0\best.pt" (
  echo [ERROR] Khong tim thay models\cnn_efficientnet_b0\best.pt
  pause
  exit /b 1
)
echo [1/2] Kiem tra Streamlit...
".venv\Scripts\python.exe" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
  echo Dang cai Streamlit...
  ".venv\Scripts\python.exe" -m pip install -r "FreshLens_Buoc5_AppThucTe\requirements-app-cnn.txt"
  if errorlevel 1 goto :fail
) else (
  echo [OK] Streamlit da san sang.
)
echo [2/2] Tao supported/unsupported gate. CNN best.pt KHONG bi train lai...
".venv\Scripts\python.exe" "FreshLens_Buoc5_AppThucTe\BUILD_OPENSET_GATE.py" --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" --data "data\cnn_dataset_v3" --checkpoint "models\cnn_efficientnet_b0\best.pt" --unknown "E:\VanDat_\XuLyAnh\FreshLens_external_test_v2\unknown" --output "models\cnn_efficientnet_b0\open_set_gate.npz" --meta "models\cnn_efficientnet_b0\open_set_gate.json" --device cuda --batch 8
if errorlevel 1 goto :fail
echo.
echo [OK] Step 5 hoan tat.
echo Bay gio chay FreshLens_Buoc5_AppThucTe\RUN_APP.cmd
pause
exit /b 0
:fail
echo.
echo [ERROR] Step 5 that bai. Gui anh man hinh nay cho ChatGPT.
pause
exit /b 1
