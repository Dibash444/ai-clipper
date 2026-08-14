@echo off
title AI YouTube Shorts Generator
echo.
echo ======================================================
echo   AI YouTube Shorts Generator
echo   Starting web server...
echo ======================================================
echo.
cd /d "%~dp0"
start "" http://localhost:5000
python app.py
pause
