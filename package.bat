@echo off
chcp 65001 >nul
title AITrainerUltra 一键打包

echo ╔════════════════════════════════════════╗
echo ║   AITrainerUltra One-Click Package    ║
echo ╚════════════════════════════════════════╝
echo.

python package.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ 打包失败，请检查上面的错误信息
    echo.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ✨ 按任意键关闭此窗口...
pause >nul
