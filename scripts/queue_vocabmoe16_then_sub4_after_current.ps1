param(
    [Parameter(Mandatory = $true)]
    [int] $WaitPid,

    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$vocabOutDir = Join-Path $Root "records\vocabmoe16-5k-auto-$stamp"
$sub4OutDir = Join-Path $Root "records\sub4-leader-levers-i4-5k-auto-$stamp"
$logPath = Join-Path $Root "records\vocabmoe16-then-sub4-auto-$stamp.queue.log"
$python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"

function Write-QueueLog {
    param([string] $Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append | Out-Null
}

Write-QueueLog "queue started root=$Root wait_pid=$WaitPid"

try {
    $proc = Get-Process -Id $WaitPid -ErrorAction Stop
    Write-QueueLog "waiting for pid=$WaitPid process=$($proc.ProcessName) start=$($proc.StartTime)"
    Wait-Process -Id $WaitPid
    Write-QueueLog "wait pid exited"
} catch {
    Write-QueueLog "wait pid not found or already exited: $($_.Exception.Message)"
}

Write-QueueLog "starting 16MB vocab MoE matrix out=$vocabOutDir"
$vocabArgs = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--out", $vocabOutDir,
    "--iterations", "5000",
    "--warmdown-iters", "5000",
    "--val-tokens", "131072",
    "--timeout", "9000",
    "--wait-for-idle-gpu",
    "--idle-max-util", "25",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "30"
)
& $python @vocabArgs *>&1 | Tee-Object -FilePath $logPath -Append
$vocabExit = $LASTEXITCODE
Write-QueueLog "16MB vocab MoE matrix exited code=$vocabExit out=$vocabOutDir"

Write-QueueLog "starting queued sub4 leader-levers matrix out=$sub4OutDir"
$sub4Args = @(
    "scripts\run_sub4_iotail_quant_matrix.py",
    "--candidate-group", "sub4_leader_levers",
    "--out", $sub4OutDir,
    "--iterations", "5000",
    "--warmdown-iters", "5000",
    "--val-tokens", "65536",
    "--timeout", "7200",
    "--final-artifacts",
    "--train-quant-forward",
    "--quant-train-every", "100",
    "--allow-over-cap",
    "--wait-for-idle-gpu",
    "--idle-max-util", "25",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "30"
)
& $python @sub4Args *>&1 | Tee-Object -FilePath $logPath -Append
$sub4Exit = $LASTEXITCODE
Write-QueueLog "sub4 leader-levers matrix exited code=$sub4Exit out=$sub4OutDir"

if ($vocabExit -ne 0) {
    exit $vocabExit
}
exit $sub4Exit
