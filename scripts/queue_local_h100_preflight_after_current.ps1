param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf",
    [int] $WaitPid = 0,
    [int] $Iterations = 2000,
    [int] $ValTokens = 65536
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$out = Join-Path $Root "records\cap16-h100-preflight-${Iterations}-auto-$stamp"
$logPath = Join-Path $out "queue.launch.log"
$python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $out | Out-Null

function Write-QueueLog {
    param([string] $Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append | Out-Null
}

if ($WaitPid -gt 0) {
    Write-QueueLog "waiting for pid=$WaitPid before launching local H100 preflight"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    Write-QueueLog "wait pid=$WaitPid finished; giving any just-launched child trainer 60s to allocate GPU"
    Start-Sleep -Seconds 60
}

Write-QueueLog "starting cap16_h100_preflight out=$out iterations=$Iterations val_tokens=$ValTokens"
$args = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "cap16_h100_preflight",
    "--out", $out,
    "--iterations", "$Iterations",
    "--warmdown-iters", "$Iterations",
    "--val-tokens", "$ValTokens",
    "--timeout", "7200",
    "--final-artifacts",
    "--train-quant-forward",
    "--wait-for-idle-gpu",
    "--idle-max-util", "90",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "15"
)
& $python @args *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$exitCode = $LASTEXITCODE
Write-QueueLog "cap16_h100_preflight exited code=$exitCode out=$out"
exit $exitCode
