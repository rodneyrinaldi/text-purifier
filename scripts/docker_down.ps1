param(
    [switch]$RemoveImage
)

$ErrorActionPreference = 'Stop'

$ImageName = 'text-purifier'
$ContainerName = 'text-purifier'

$Exists = docker ps -a --filter "name=^/${ContainerName}$" --format "{{.Names}}"
if ($Exists) {
    docker rm -f $ContainerName | Out-Null
    Write-Host 'Container removido.'
}
else {
    Write-Host 'Container nao encontrado.'
}

if ($RemoveImage) {
    $ImageExists = docker images --format "{{.Repository}}" | Select-String -SimpleMatch "^${ImageName}$"
    if ($ImageExists) {
        docker rmi $ImageName | Out-Null
        Write-Host 'Imagem removida.'
    }
}
