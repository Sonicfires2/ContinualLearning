#!/usr/bin/env pwsh
$ErrorActionPreference = 'Stop'

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    $pythonCmd = Get-Command py -ErrorAction SilentlyContinue
}

if (-not $pythonCmd) {
    throw "Python was not found. Install Python 3.14+ and make sure it is available as 'python' or 'py'."
}

& $pythonCmd.Source -m venv --without-pip .venv

$venvPython = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Virtual environment Python was not created at $venvPython."
}

& $pythonCmd.Source -m pip --python $venvPython install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Output "Virtual environment is ready."
Write-Output "Activate it with: .\.venv\Scripts\Activate.ps1"
