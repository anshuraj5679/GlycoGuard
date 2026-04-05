$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment not found at .venv\Scripts\python.exe"
}

& $python -m glycoguard.cli serve --separate-windows @args
