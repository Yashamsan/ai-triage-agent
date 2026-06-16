# AI Triage Agent — one-click startup
# Run from repo root: .\start.ps1

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

Write-Host ""
Write-Host "=== AI Triage Agent — Startup ===" -ForegroundColor Cyan
Write-Host ""

# ── 1. Docker stack ────────────────────────────────────────────────
Write-Host "[1/4] Starting LangFuse Docker stack..." -ForegroundColor Yellow
docker compose `
  --project-directory "$ROOT\docker" `
  --env-file "$ROOT\docker\.env.langfuse" `
  -f "$ROOT\docker\docker-compose.yml" `
  up -d 2>&1 | Where-Object { $_ -match "(Started|Running|healthy|Error)" }

# ── 2. Wait for LangFuse web to be reachable ───────────────────────
Write-Host "[2/4] Waiting for LangFuse UI (http://localhost:3100)..." -ForegroundColor Yellow
$tries = 0
do {
    Start-Sleep -Seconds 2
    $tries++
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:3100" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        $ok = $r.StatusCode -lt 500
    } catch { $ok = $false }
} while (-not $ok -and $tries -lt 30)

if ($ok) { Write-Host "  LangFuse ready." -ForegroundColor Green }
else      { Write-Host "  LangFuse not responding yet — check Docker logs." -ForegroundColor Red }

# ── 3. Verify/seed triage_agent database ──────────────────────────
Write-Host "[3/4] Checking triage_agent database..." -ForegroundColor Yellow
python -c @"
import psycopg2, sys
from pathlib import Path; from dotenv import load_dotenv; load_dotenv(Path(r'$ROOT\.env'))
try:
    conn = psycopg2.connect('postgresql://postgres:postgres@localhost/triage_agent')
    cur  = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM faq_articles')
    n = cur.fetchone()[0]
    conn.close()
    if n == 0:
        print('  No FAQ data found — running seed...')
        sys.exit(1)
    else:
        print(f'  Database OK ({n} FAQ articles)')
        sys.exit(0)
except Exception as e:
    print(f'  DB error: {e}')
    sys.exit(2)
"@

if ($LASTEXITCODE -eq 1) {
    Write-Host "  Seeding FAQ data..." -ForegroundColor Yellow
    python -m app.seed_data
}
if ($LASTEXITCODE -eq 2) {
    Write-Host "  Cannot reach database. Is PostgreSQL running?" -ForegroundColor Red
    exit 1
}

# ── 4. Start FastAPI server ────────────────────────────────────────
Write-Host "[4/4] Starting FastAPI server on http://localhost:8000 ..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
  "cd '$ROOT'; Write-Host 'FastAPI running — http://localhost:8000' -ForegroundColor Green; uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

Start-Sleep -Seconds 3

# ── Done ───────────────────────────────────────────────────────────
Write-Host ""
Write-Host "=== All services started ===" -ForegroundColor Green
Write-Host ""
Write-Host "  Chat UI    →  open ui\index.html in your browser" -ForegroundColor White
Write-Host "  API        →  http://localhost:8000/docs" -ForegroundColor White
Write-Host "  LangFuse   →  http://localhost:3100" -ForegroundColor White
Write-Host "              login: yashamsan@gmail.com / LangFuse2026!Login" -ForegroundColor Gray
Write-Host ""

# Open the UI automatically
Start-Process "$ROOT\ui\index.html"
