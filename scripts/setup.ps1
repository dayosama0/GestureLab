param([switch]$SkipInstall)
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
if (!(Test-Path .venv/Scripts/python.exe)) { python -m venv .venv }
if (!$SkipInstall) {
    & .venv/Scripts/python -m pip install --upgrade pip
    & .venv/Scripts/python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed' }
    npm --prefix frontend ci
    if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed' }
}
Write-Host 'Setup complete. See README.md for data audit and real training commands.'
