@echo off
title Deep Polyp Segmentation Studio Launcher
echo ====================================================================
echo   DEEP POLYP SEGMENTATION STUDIO - MULTI-MODEL RESEARCH PLATFORM
echo   Academic Modernization: PyTorch + FastAPI + Next.js
echo ====================================================================
echo.

cd /d "%~dp0"

echo [1/3] Activating Python Virtual Environment...
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe.
    echo Please create it with: python -m venv .venv
    pause
    exit /b 1
)

echo [2/3] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Polyp Backend (FastAPI)" cmd /k "title Polyp Backend && .venv\Scripts\uvicorn.exe backend.app.main:app --host 127.0.0.1 --port 8000"

echo [3/3] Starting Next.js Web Studio on http://localhost:3000 ...
start "Polyp Frontend (Next.js)" cmd /k "title Polyp Frontend && cd frontend && npm run start"

echo.
echo Waiting 3 seconds for services to initialize...
timeout /t 3 >nul

echo Launching browser...
start http://localhost:3000

echo.
echo ====================================================================
echo Both services are now running!
echo - Frontend Studio: http://localhost:3000
echo - Backend API Docs: http://127.0.0.1:8000/docs
echo ====================================================================
echo.
pause
