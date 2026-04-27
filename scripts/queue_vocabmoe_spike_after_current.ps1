param(
    [Parameter(Mandatory = $true)]
    [int] $WaitPid,

    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir = Join-Path $Root "records\vocabmoe16-spike-5k-auto-$stamp"
$logPath = Join-Path $Root "records\vocabmoe16-spike-after-current-$stamp.queue.log"
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

Write-QueueLog "starting 16MB vocab MoE spike/self-election matrix out=$outDir"
$args = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "vocabmoe_spike",
    "--out", $outDir,
    "--iterations", "5000",
    "--warmdown-iters", "5000",
    "--val-tokens", "131072",
    "--timeout", "9000",
    "--wait-for-idle-gpu",
    "--idle-max-util", "25",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "30"
)
& $python @args *>&1 | Tee-Object -FilePath $logPath -Append
$exitCode = $LASTEXITCODE
Write-QueueLog "16MB vocab MoE spike/self-election matrix exited code=$exitCode out=$outDir"
exit $exitCode
