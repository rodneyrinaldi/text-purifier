$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ImageName = 'text-purifier'
$ContainerName = 'text-purifier'
$EnvFile = Join-Path $ProjectRoot '.env'

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw 'Docker CLI nao encontrado no PATH.'
}

docker info | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw 'Docker daemon indisponivel. Inicie o Docker Desktop e tente novamente.'
}

if (-not (Test-Path $EnvFile)) {
    $EnvFile = Join-Path $ProjectRoot '.env.example'
}

Push-Location $ProjectRoot
try {
    docker build -t $ImageName .
    if ($LASTEXITCODE -ne 0) {
        throw 'Falha no build da imagem Docker.'
    }

    $Exists = docker ps -a --filter "name=^/${ContainerName}$" --format "{{.Names}}"
    if ($Exists) {
        docker rm -f $ContainerName | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw 'Falha ao remover container existente.'
        }
    }

    docker run -d --name $ContainerName -p 8000:8000 --env-file $EnvFile $ImageName | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw 'Falha ao subir container Docker.'
    }
    Write-Host "Container no ar: http://localhost:8000"
}
finally {
    Pop-Location
}
