param(
    [Parameter(Mandatory = $true)]
    [int] $WaitPid,

    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir = Join-Path $Root "records\sub4-leader-levers-i4-5k-auto-$stamp"
$logPath = Join-Path $Root "records\sub4-leader-levers-i4-5k-auto-$stamp.queue.log"
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

Write-QueueLog "starting next matrix out=$outDir"

$args = @(
    "scripts\run_sub4_iotail_quant_matrix.py",
    "--candidate-group", "sub4_leader_levers",
    "--out", $outDir,
    "--iterations", "5000",
    "--warmdown-iters", "5000",
    "--val-tokens", "65536",
    "--timeout", "7200",
    "--final-artifacts",
    "--train-quant-forward",
    "--quant-train-every", "100",
    "--allow-over-cap",
    "--wait-for-idle-gpu",
    "--idle-max-util", "15",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "30"
)

& $python @args *>&1 | Tee-Object -FilePath $logPath -Append
$exitCode = $LASTEXITCODE
Write-QueueLog "next matrix exited code=$exitCode out=$outDir"
exit $exitCode
