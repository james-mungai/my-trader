param(
    [Parameter(Mandatory = $true)]
    [string]$Image,
    [string]$ServiceName = "futures-lab",
    [string]$Region = "ap-northeast-1",
    [string]$Power = "medium",
    [int]$Scale = 1,
    [string]$ApiToken = "change-me-before-deploy"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Template = Join-Path $RepoRoot "deployments\aws\lightsail-containers.template.json"
$PublicEndpoint = Join-Path $RepoRoot "deployments\aws\lightsail-public-endpoint.json"
$Rendered = Join-Path ([System.IO.Path]::GetTempPath()) "futures-lab-lightsail-containers.json"

$existing = aws lightsail get-container-services `
    --region $Region `
    --service-name $ServiceName `
    --query "containerServices[0].serviceName" `
    --output text 2>$null

if (-not $existing -or $existing -eq "None") {
    aws lightsail create-container-service `
        --region $Region `
        --service-name $ServiceName `
        --power $Power `
        --scale $Scale | Out-Null
    Write-Host "Created Lightsail container service $ServiceName ($Power x $Scale)."
}

(Get-Content $Template -Raw).
    Replace("__LIGHTSAIL_IMAGE__", $Image).
    Replace("__API_TOKEN__", $ApiToken) |
    Set-Content -Encoding utf8 $Rendered

aws lightsail create-container-service-deployment `
    --region $Region `
    --service-name $ServiceName `
    --containers "file://$Rendered" `
    --public-endpoint "file://$PublicEndpoint"

Write-Host ""
Write-Host "Deployment requested. Check status with:"
Write-Host "aws lightsail get-container-services --region $Region --service-name $ServiceName"
