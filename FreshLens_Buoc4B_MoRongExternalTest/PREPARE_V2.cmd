@echo off
setlocal
set "SRC=E:\VanDat_\XuLyAnh\FreshLens_external_test"
set "DST=E:\VanDat_\XuLyAnh\FreshLens_external_test_v2"

if not exist "%SRC%" (
  echo [ERROR] Khong tim thay bo External Test V1: %SRC%
  pause
  exit /b 1
)

if exist "%DST%" (
  echo [ERROR] Thu muc V2 da ton tai: %DST%
  echo Hay doi ten/xoa thu muc V2 neu ban muon tao lai. KHONG xoa V1.
  pause
  exit /b 1
)

xcopy "%SRC%" "%DST%\" /E /I /H /K /Y >nul
if errorlevel 1 (
  echo [ERROR] Khong copy duoc V1 sang V2.
  pause
  exit /b 1
)

mkdir "%DST%\ambiguous_review" 2>nul

echo [OK] Da tao bo External Test V2 tu V1:
echo      %DST%
echo.
echo Them anh moi de moi lop known co it nhat 10 anh, unknown co it nhat 20 anh.
echo Anh kho/khong chac nhan thi bo vao ambiguous_review de KHONG tinh accuracy.
explorer "%DST%"
pause
endlocal
