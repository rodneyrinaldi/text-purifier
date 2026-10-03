$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python nao encontrado no PATH.'
}

Write-Host 'Instalando dependencias no Python atual...'
Push-Location $ProjectRoot
try {
    python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw 'Falha ao atualizar pip.'
    }

    python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw 'Falha ao instalar dependencias.'
    }
}
finally {
    Pop-Location
}

Write-Host 'Dependencias instaladas.'
