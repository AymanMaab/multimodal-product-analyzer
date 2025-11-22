# Startup script for Multimodal Product Analyzer
# Ensures clean restart with updated code

Write-Host "`n🔄 Cleaning up old processes..." -ForegroundColor Cyan

# Kill all Python processes
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue

# Clear ports
Get-NetTCPConnection -LocalPort 8000,8501 -ErrorAction SilentlyContinue | ForEach-Object { 
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue 
}

Start-Sleep -Seconds 2

# Clear Python cache
Write-Host "🧹 Clearing Python cache..." -ForegroundColor Cyan
Remove-Item -Recurse -Force src\__pycache__,src\api\__pycache__,src\models\__pycache__,src\data\__pycache__ -ErrorAction SilentlyContinue

# Activate virtual environment
Write-Host "🐍 Activating virtual environment..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

# Start FastAPI server
Write-Host "Starting FastAPI server on port 8888..." -ForegroundColor Green
Start-Process python -ArgumentList "-m","uvicorn","src.api.main:app","--host","0.0.0.0","--port","8888" -NoNewWindow

# Wait for API to load
Write-Host "⏳ Waiting for models to load (30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Test API health
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8888/health"
    Write-Host "✅ API Status: $($health.status) - Models Loaded: $($health.models_loaded)" -ForegroundColor Green
} catch {
    Write-Host "⚠️  API health check failed - please verify manually" -ForegroundColor Red
}

# Start Streamlit
Write-Host "🎨 Starting Streamlit dashboard on port 8501..." -ForegroundColor Green
Start-Process python -ArgumentList "-m","streamlit","run","app/streamlit_app.py" -NoNewWindow

Write-Host "`n=== Servers started successfully ===" -ForegroundColor Green
Write-Host "API: http://localhost:8888" -ForegroundColor Cyan
Write-Host "Streamlit: http://localhost:8501" -ForegroundColor Cyan
Write-Host "`nTip: Neutral reviews now classify correctly!" -ForegroundColor Yellow
Write-Host "Words like okay, decent, acceptable trigger NEUTRAL (3 stars)" -ForegroundColor Yellow
