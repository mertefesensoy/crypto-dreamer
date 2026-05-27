# Native Windows dev orchestrator. Spawns redis + api + ui in parallel
# PowerShell jobs and waits on them. Ctrl+C cleans them all up.
#
# Usage:  pwsh -NoProfile -File scripts/dev.ps1
#         (or)  powershell -NoProfile -File scripts/dev.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$redis = Join-Path $PWD ".tools\redis\redis-server.exe"
if (-not (Test-Path $redis)) {
    Write-Error "Portable Redis not found at $redis. Run docker-compose up or fetch the zip into .tools/redis/."
    exit 1
}

$jobs = @()

Write-Host "[redis] starting" -ForegroundColor Magenta
$redisProc = Start-Process -FilePath $redis `
    -ArgumentList "--port", "6379", "--bind", "127.0.0.1", "--save", "", "--dir", ".tools\redis" `
    -PassThru -WindowStyle Hidden

Write-Host "[api]   starting" -ForegroundColor Cyan
$jobs += Start-Job -Name api -ScriptBlock {
    Set-Location $using:PWD
    uv run uvicorn serve.api:app --host 127.0.0.1 --port 8000 --reload
}

Write-Host "[ui]    starting" -ForegroundColor Green
$jobs += Start-Job -Name ui -ScriptBlock {
    Set-Location (Join-Path $using:PWD "dashboard")
    npm run dev
}

try {
    Write-Host "Press Ctrl+C to stop all services."
    while ($true) {
        foreach ($j in $jobs) {
            Receive-Job -Job $j -Keep | ForEach-Object { Write-Host "[$($j.Name)] $_" }
        }
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host "Stopping…"
    foreach ($j in $jobs) { Stop-Job -Job $j -ErrorAction SilentlyContinue; Remove-Job -Job $j -Force -ErrorAction SilentlyContinue }
    if ($redisProc -and -not $redisProc.HasExited) { Stop-Process -Id $redisProc.Id -Force -ErrorAction SilentlyContinue }
}
