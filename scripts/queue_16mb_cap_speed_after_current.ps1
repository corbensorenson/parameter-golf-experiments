param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf",
    [int] $WaitPid = 0
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $Root "records\vocabmoe16-cap-speed-after-current-$stamp.queue.log"
$python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"
$out = Join-Path $Root "records\vocabmoe16-cap-speed-scout-3k-auto-$stamp"

function Write-QueueLog {
    param([string] $Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append | Out-Null
}

if ($WaitPid -gt 0) {
    Write-QueueLog "waiting for pid=$WaitPid before launching 16MB cap-speed scout"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    Write-QueueLog "wait pid=$WaitPid finished"
}

Write-QueueLog "starting 16MB cap-speed VocabMoE scout out=$out"
$args = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "cap16_speed",
    "--out", $out,
    "--iterations", "3000",
    "--warmdown-iters", "3000",
    "--val-tokens", "131072",
    "--timeout", "9000",
    "--final-artifacts",
    "--train-quant-forward",
    "--wait-for-idle-gpu",
    "--idle-max-util", "90",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "5"
)
& $python @args *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$exitCode = $LASTEXITCODE
Write-QueueLog "16MB cap-speed VocabMoE scout exited code=$exitCode out=$out"
exit $exitCode
