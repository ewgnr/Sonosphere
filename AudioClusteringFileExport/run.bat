```bat
@echo off
title Audio Clustering

cd /d "%~dp0"

echo Starting audio clustering...

uv run audio_clustering.py

echo.
echo Process finished.
pause
```