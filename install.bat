@echo off
title AITrainerUltra Installer

echo ========================================
echo    AITrainerUltra v2.0 Installer
echo    Multi-Model AI Training Framework
echo ========================================
echo.

REM Step 1: Check Python
echo [1/4] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo Please install Python 3.10+: https://www.python.org/downloads/
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set pyver=%%i
echo   [OK] Python %pyver%

REM Step 2: Install dependencies
echo [2/4] Installing Python dependencies...
pip install -r backend\requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARN] Some deps failed, installing core...
    pip install fastapi uvicorn pydantic websockets --quiet
)
echo   [OK] Dependencies installed

REM Step 3: Build frontend
echo [3/4] Building frontend...
if exist frontend\package.json (
    cd frontend
    call npm install --silent
    if %errorlevel% equ 0 (
        call npm run build --silent
        if %errorlevel% equ 0 (
            echo   [OK] Frontend built
        ) else (
            echo   [WARN] Frontend build failed, API-only mode
        )
    ) else (
        echo   [WARN] npm install failed, API-only mode
    )
    cd ..
) else (
    echo   [SKIP] No frontend directory found
)

REM Step 4: Create launcher
echo [4/4] Creating launcher...
echo @echo off > start_aitrainer.bat
echo title AITrainerUltra >> start_aitrainer.bat
echo echo. >> start_aitrainer.bat
echo echo ======================================== >> start_aitrainer.bat
echo echo    AITrainerUltra v2.0 >> start_aitrainer.bat
echo echo    Starting server... >> start_aitrainer.bat
echo echo ======================================== >> start_aitrainer.bat
echo echo. >> start_aitrainer.bat
echo echo Web UI: http://127.0.0.1:8000 >> start_aitrainer.bat
echo echo API:   http://127.0.0.1:8000/docs >> start_aitrainer.bat
echo echo. >> start_aitrainer.bat
echo start http://127.0.0.1:8000 >> start_aitrainer.bat
echo python -m uvicorn backend.api.server:app --host 127.0.0.1 --port 8000 >> start_aitrainer.bat
echo pause >> start_aitrainer.bat
echo   [OK] Launcher created: start_aitrainer.bat

echo.
echo ========================================
echo    Installation Complete!
echo ========================================
echo.
echo To start: double-click start_aitrainer.bat
echo Or run:   python -m uvicorn backend.api.server:app --host 127.0.0.1 --port 8000
echo.
echo Web UI: http://127.0.0.1:8000
echo API:    http://127.0.0.1:8000/docs
echo.
pause
