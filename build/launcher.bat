@echo off
title AITrainerUltra Dev Launcher

echo ========================================
echo    AITrainerUltra v2.0
echo    Development Launcher
echo ========================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    pause
    exit /b 1
)

echo [1/3] Installing backend deps...
pip install -r backend\requirements.txt --quiet

echo [2/3] Building frontend...
cd frontend
call npm install --silent
call npm run build
cd ..

echo [3/3] Starting server...
echo.
echo Web UI: http://127.0.0.1:8000
echo API:    http://127.0.0.1:8000/docs
echo.
start http://127.0.0.1:8000
python -m uvicorn backend.api.server:app --host 127.0.0.1 --port 8000

pause
