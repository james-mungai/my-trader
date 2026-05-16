param(
    [string]$ServiceName = "futures-lab",
    [string]$Region = "ap-northeast-1",
    [string]$ImageName = "futures-lab:aws",
    [string]$Label = "futures-lab"
)

$ErrorActionPreference = "Stop"

aws lightsail push-container-image `
    --region $Region `
    --service-name $ServiceName `
    --label $Label `
    --image $ImageName

Write-Host ""
Write-Host "Copy the returned image identifier, for example :$ServiceName.$Label.1,"
Write-Host "then pass it to deploy-lightsail-container.ps1 -Image <identifier>."
