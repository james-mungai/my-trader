param(
    [int]$Port = 8888,
    [string]$DataRoot = (Join-Path $HOME "projects\my-trader-research-data\20260711")
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = Join-Path $RepoRoot ".venv-research\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Research environment missing. Create .venv-research and install .[dev,research] first."
}

$env:FUTURES_LAB_RESEARCH_DATA = $DataRoot
& $Python -m jupyter lab `
    --notebook-dir $RepoRoot `
    --ip 127.0.0.1 `
    --port $Port `
    --no-browser `
    --IdentityProvider.token futures-lab-local
