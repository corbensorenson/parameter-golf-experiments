param(
    [Parameter(Mandatory = $true)]
    [string] $HostName,

    [Parameter(Mandatory = $true)]
    [int] $Port,

    [string] $User = "root",
    [string] $KeyPath = "C:\Users\corbe\.ssh\runpod_codex_ed25519",
    [string] $RemoteDir = "/workspace",
    [string] $BundlePath = "",
    [switch] $NoStart
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($BundlePath)) {
    $bundleDir = Join-Path $repoRoot "tmp-runpod-bundles"
    $latest = Get-ChildItem -LiteralPath $bundleDir -Filter "*.tar.gz" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $latest) {
        throw "No bundle found under $bundleDir. Run scripts\make_runpod_novel_bundle.ps1 first."
    }
    $BundlePath = $latest.FullName
}

$BundlePath = (Resolve-Path -LiteralPath $BundlePath).Path
$KeyPath = (Resolve-Path -LiteralPath $KeyPath).Path
$archiveName = Split-Path -Leaf $BundlePath
$bundleName = $archiveName -replace '\.tar\.gz$', ''
$target = "${User}@${HostName}"
$mode = "final8x"
$checkMode = "final8x-check"
$sshCommon = @(
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=NUL",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
    "-p", "$Port",
    "-i", $KeyPath
)
$scpCommon = @(
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=NUL",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
    "-P", "$Port",
    "-i", $KeyPath
)

Write-Host "Uploading bundle:"
Write-Host "  bundle: $BundlePath"
Write-Host "  target: ${target}:$RemoteDir/"

& scp @scpCommon $BundlePath "${target}:$RemoteDir/"
if ($LASTEXITCODE -ne 0) {
    throw "scp failed with code $LASTEXITCODE"
}

if ($NoStart) {
    Write-Host "Upload complete. Start command:"
    Write-Host "  ssh -p $Port -i `"$KeyPath`" $target"
    Write-Host "  cd $RemoteDir && tar -xzf $archiveName && cd $bundleName && bash scripts/runpod_run_novel_no_fetch.sh final8x"
    exit 0
}

$remoteCommand = @"
set -euo pipefail
cd '$RemoteDir'
if [ -d '$bundleName' ]; then mv '$bundleName' '${bundleName}.bak-'`$(date +%s); fi
tar -xzf '$archiveName'
cd '$bundleName'
bash scripts/runpod_run_novel_no_fetch.sh '$checkMode'
nohup bash scripts/runpod_run_novel_no_fetch.sh '$mode' > '$RemoteDir/final8x.out' 2>&1 < /dev/null &
echo "`$!" > '$RemoteDir/final8x.pid'
echo "started final8x pid=`$!"
echo "log: $RemoteDir/final8x.out"
"@

Write-Host "Running remote check and starting final8x in background..."
$remoteCommand | & ssh @sshCommon $target bash -s
if ($LASTEXITCODE -ne 0) {
    throw "ssh start command failed with code $LASTEXITCODE"
}

Write-Host ""
Write-Host "Started. Follow logs with:"
Write-Host "  ssh -p $Port -i `"$KeyPath`" $target `"tail -f $RemoteDir/final8x.out`""
