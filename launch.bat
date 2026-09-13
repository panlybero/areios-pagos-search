@echo off
setlocal
cd /d "%~dp0"

echo ======================================================
echo           Areios Pagos - Case Law Search              
echo ======================================================

set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing runner environment (uv)...
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
)

echo Starting application and opening browser...
uv run apsearch launch %*
pause
