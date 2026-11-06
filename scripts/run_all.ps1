$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $projectRoot
$apiProcess = Start-Process -FilePath "$projectRoot/.venv/Scripts/python.exe" -ArgumentList '-m','uvicorn','backend.app.main:app','--host','127.0.0.1','--port','8000' -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru
try {
    npm --prefix frontend run dev -- --host 127.0.0.1
} finally {
    if (!$apiProcess.HasExited) { Stop-Process -Id $apiProcess.Id }
}
