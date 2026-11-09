```bat
@echo off
title Audio Clustering - Setup

echo ====================================
echo   Audio Clustering - Installation
echo ====================================
echo.

cd /d "%~dp0"

echo Checking uv installation...
where uv >nul 2>&1

if errorlevel 1 (
    echo ERROR: uv is not installed or not in PATH.
    echo Please install uv first.
    pause
    exit /b 1
)

echo.
echo Initializing project...

if not exist pyproject.toml (
    uv init --bare
)

echo.
echo Installing dependencies...

uv add numpy librosa torch torchaudio soundfile matplotlib scikit-learn

if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed.
    pause
    exit /b 1
)

echo.
echo ====================================
echo   Installation completed!
echo ====================================
pause
```