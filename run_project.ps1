# Deep Polyp Segmentation Studio Launcher (PowerShell)
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host "  DEEP POLYP SEGMENTATION STUDIO - MULTI-MODEL RESEARCH PLATFORM" -ForegroundColor Green
Write-Host "  Academic Modernization: PyTorch + FastAPI + Next.js" -ForegroundColor Yellow
Write-Host "====================================================================" -ForegroundColor Cyan

$WorkspaceRoot = $PSScriptRoot
Set-Location $WorkspaceRoot

if (-not (Test-Path "$WorkspaceRoot\.venv\Scripts\python.exe")) {
    Write-Host "[ERROR] Virtual environment not found at .venv\Scripts\python.exe." -ForegroundColor Red
    exit 1
}

Write-Host "`n[1/2] Starting FastAPI Backend on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "cd '$WorkspaceRoot'; .\.venv\Scripts\uvicorn.exe backend.app.main:app --host 127.0.0.1 --port 8000"

Write-Host "[2/2] Starting Next.js Web Studio on http://localhost:3000 ..." -ForegroundColor Cyan
Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "cd '$WorkspaceRoot\frontend'; npm run start"

Write-Host "`nWaiting for FastAPI backend to initialize (PyTorch models loading)..." -ForegroundColor Cyan
$BackendReady = $false
for ($i = 0; $i -lt 15; $i++) {
    Start-Sleep -Seconds 1
    try {
        $res = Invoke-WebRequest -Uri "http://127.0.0.1:8000/api/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($res.StatusCode -eq 200) {
            $BackendReady = $true
            break
        }
    } catch {
        # Waiting for Uvicorn and PyTorch startup
    }
}

if ($BackendReady) {
    Write-Host "FastAPI backend is online and healthy!" -ForegroundColor Green
} else {
    Write-Host "[NOTE] FastAPI backend is still initializing; frontend will automatically connect when ready." -ForegroundColor Yellow
}

Write-Host "`nOpening Web Application..." -ForegroundColor Green
Start-Process "http://localhost:3000"

Write-Host "`nServices are running in background processes." -ForegroundColor Green
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Yellow
Write-Host "API Docs: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
