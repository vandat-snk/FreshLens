@echo off
setlocal
set "PROJECT=E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens"
set "ROOT=E:\VanDat_\XuLyAnh\FreshLens_external_test_v2"

cd /d "%PROJECT%"
if errorlevel 1 (
  echo [ERROR] Khong vao duoc project: %PROJECT%
  pause
  exit /b 1
)

if not exist ".\.venv\Scripts\python.exe" (
  echo [ERROR] Khong tim thay Python trong .venv
  pause
  exit /b 1
)

if not exist ".\FreshLens_Buoc4B_MoRongExternalTest\CHECK_COUNTS.py" (
  echo [ERROR] Thieu CHECK_COUNTS.py. Hay giai nen Buoc4B vao project.
  pause
  exit /b 1
)

if not exist ".\FreshLens_Buoc4_ExternalTest\EXTERNAL_TEST.py" (
  echo [ERROR] Thieu Buoc4 External Test cu: .\FreshLens_Buoc4_ExternalTest\EXTERNAL_TEST.py
  pause
  exit /b 1
)

.\.venv\Scripts\python.exe .\FreshLens_Buoc4B_MoRongExternalTest\CHECK_COUNTS.py --root "%ROOT%" --min-known 10 --min-unknown 20
if errorlevel 1 (
  echo.
  echo [STOP] Chua chay model vi bo V2 chua dat dieu kien.
  pause
  exit /b 1
)

echo.
echo [RUN] External Test V2...
.\.venv\Scripts\python.exe .\FreshLens_Buoc4_ExternalTest\EXTERNAL_TEST.py --root "%ROOT%" --checkpoint "models\cnn_efficientnet_b0\best.pt" --data "data\cnn_dataset_v3" --output "reports\external_test_v2" --device cuda
if errorlevel 1 (
  echo.
  echo [ERROR] External Test V2 that bai. Neu reports\external_test_v2 da ton tai, hay giu bao cao cu va doi output.
  pause
  exit /b 1
)

echo.
echo [OK] Hoan tat. Gui 3 file:
echo   reports\external_test_v2\metrics.json
echo   reports\external_test_v2\predictions.csv
echo   reports\external_test_v2\confusion_matrix.png
explorer "reports\external_test_v2"
pause
endlocal
