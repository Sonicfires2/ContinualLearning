#!/usr/bin/env pwsh
$ErrorActionPreference = 'Stop'
python -m venv .venv
Write-Output "Activating virtual environment..."
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Output "Activate script not found; to activate run: .\.venv\Scripts\Activate.ps1"
}
pip install --upgrade pip
pip install -r requirements.txt
