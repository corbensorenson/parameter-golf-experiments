param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf",
    [int] $WaitPid = 0,
    [int] $Iterations = 3000,
    [switch] $RunDual
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $Root "records\mainline-dual-16mb-after-current-$stamp.queue.log"
$python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"

function Write-QueueLog {
    param([string] $Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append | Out-Null
}

if ($WaitPid -gt 0) {
    Write-QueueLog "waiting for pid=$WaitPid before launching 16MB mainline/dual scouts"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    Write-QueueLog "wait pid=$WaitPid finished"
}

$mainlineOut = Join-Path $Root "records\cap16-mainline-scout-${Iterations}-auto-$stamp"
$dualOut = Join-Path $Root "records\cap16-dualstream-scout-${Iterations}-auto-$stamp"

Write-QueueLog "starting cap16_mainline scout out=$mainlineOut"
$mainlineArgs = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "cap16_mainline",
    "--out", $mainlineOut,
    "--iterations", "$Iterations",
    "--warmdown-iters", "$Iterations",
    "--val-tokens", "131072",
    "--timeout", "9000",
    "--final-artifacts",
    "--train-quant-forward",
    "--wait-for-idle-gpu",
    "--idle-max-util", "90",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "5"
)
& $python @mainlineArgs *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$mainlineExit = $LASTEXITCODE
Write-QueueLog "cap16_mainline scout exited code=$mainlineExit out=$mainlineOut"
if ($mainlineExit -ne 0) {
    exit $mainlineExit
}

if (-not $RunDual) {
    Write-QueueLog "skipping cap16_dual_stream scout; pass -RunDual after cap-speed/mainline evidence justifies the spend"
    exit 0
}

Write-QueueLog "starting cap16_dual_stream scout out=$dualOut"
$dualArgs = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "cap16_dual_stream",
    "--out", $dualOut,
    "--iterations", "$Iterations",
    "--warmdown-iters", "$Iterations",
    "--val-tokens", "131072",
    "--timeout", "9000",
    "--final-artifacts",
    "--train-quant-forward",
    "--wait-for-idle-gpu",
    "--idle-max-util", "90",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "5"
)
& $python @dualArgs *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$dualExit = $LASTEXITCODE
Write-QueueLog "cap16_dual_stream scout exited code=$dualExit out=$dualOut"
exit $dualExit
