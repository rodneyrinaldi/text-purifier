$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python nao encontrado no PATH.'
}

Push-Location $ProjectRoot
try {
    python -m uvicorn main:app --host 0.0.0.0 --port 8000
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao iniciar a API local. Exit code: $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
