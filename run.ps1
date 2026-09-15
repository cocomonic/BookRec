param([switch]$Serve)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
function Check-Exit { if ($LASTEXITCODE -ne 0) { throw "Command failed: $LASTEXITCODE" } }
if (!(Test-Path .venv\Scripts\python.exe)) { python -m venv .venv; Check-Exit }
& .venv\Scripts\python -m pip install -r requirements.txt
Check-Exit
& .venv\Scripts\python -m bookrec.pipeline
Check-Exit
& .venv\Scripts\python -m pytest -q
Check-Exit
& .venv\Scripts\python scripts\build_report.py
Check-Exit
if ($Serve) { & .venv\Scripts\python -m uvicorn bookrec.api:app --host 127.0.0.1 --port 8000 }
