@echo off
setlocal
cd /d "%~dp0.."
if errorlevel 1 exit /b 1
set "FRESHLENS_PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%FRESHLENS_PYTHON%" (
  echo [ERROR] Khong thay .venv\Scripts\python.exe trong thu muc FreshLens.
  echo Dat folder FreshLens_Buoc3_CNN canh folder .venv, src va data.
  exit /b 1
)
if not exist ".cache\pip" mkdir ".cache\pip"
if not exist ".cache\tmp" mkdir ".cache\tmp"
set "PIP_CACHE_DIR=%CD%\.cache\pip"
set "TMP=%CD%\.cache\tmp"
set "TEMP=%CD%\.cache\tmp"
echo [1/3] Cai PyTorch CUDA 12.8 vao .venv cua FreshLens...
"%FRESHLENS_PYTHON%" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b 1
echo [2/3] Cai thu vien CNN...
"%FRESHLENS_PYTHON%" -m pip install -r "%~dp0requirements-cnn.txt"
if errorlevel 1 exit /b 1
echo [3/3] Kiem tra forward/backward tren GPU...
"%FRESHLENS_PYTHON%" "%~dp0CHECK_GPU.py"
if errorlevel 1 exit /b 1
echo [OK] Cai dat va kiem tra GPU thanh cong. Chay lenh TRAIN_CNN.py trong HUONG_DAN.md.
endlocal
