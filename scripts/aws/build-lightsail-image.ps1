param(
    [string]$ImageName = "futures-lab:aws"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

docker build `
    -f (Join-Path $RepoRoot "Dockerfile.aws") `
    -t $ImageName `
    $RepoRoot

Write-Host "Built $ImageName"
