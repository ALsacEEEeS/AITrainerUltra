@echo off
title AITrainerUltra
cd /d "%~dp0"

echo ========================================
echo    AITrainerUltra v2.0
echo    One-click launcher
echo ========================================
echo.

if "%1"=="--frontend" goto frontend
if "%1"=="--backend" goto backend

:: Default: start both in separate windows
start "AITrainerUltra Backend" cmd /c "%~f0 --backend"
start "AITrainerUltra Frontend" cmd /c "%~f0 --frontend"
echo Both services started.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5173
echo API Docs: http://127.0.0.1:8000/docs
echo.
echo Close the windows to stop.
goto end

:backend
echo [Backend] Starting uvicorn on http://127.0.0.1:8000
python -m uvicorn backend.api.server:app --host 127.0.0.1 --port 8000 --reload
pause
goto end

:frontend
echo [Frontend] Starting Vite on http://127.0.0.1:5173
cd frontend
call npm run dev 2>nul || (
    echo [Frontend] npm not found, try node...
    node node_modules/.bin/vite
)
pause
goto end

:end
