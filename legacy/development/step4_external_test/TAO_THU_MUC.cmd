@echo off
setlocal
set "ROOT=E:\VanDat_\XuLyAnh\FreshLens_external_test"
mkdir "%ROOT%\apple_fresh" 2>nul
mkdir "%ROOT%\apple_rotten" 2>nul
mkdir "%ROOT%\banana_fresh" 2>nul
mkdir "%ROOT%\banana_rotten" 2>nul
mkdir "%ROOT%\orange_fresh" 2>nul
mkdir "%ROOT%\orange_rotten" 2>nul
mkdir "%ROOT%\tomato_fresh" 2>nul
mkdir "%ROOT%\tomato_rotten" 2>nul
mkdir "%ROOT%\unknown" 2>nul
echo [OK] Created: %ROOT%
explorer "%ROOT%"
endlocal
