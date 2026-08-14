@echo off
title Share AI Shorts Generator Online
echo ======================================================
echo   Creating Public Access Link (Cloudflare Tunnel)...
echo ======================================================
echo.
npx untun@latest tunnel http://localhost:7860
pause
