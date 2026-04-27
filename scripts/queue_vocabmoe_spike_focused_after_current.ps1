param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf",
    [int] $WaitPid = 39104
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $Root "records\vocabmoe-spike-focused-after-current-$stamp.queue.log"
$outDir = Join-Path $Root "records\vocabmoe16-spike-focused-5k-auto-$stamp"
$python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"
$idleMaxUtil = "90"

function Write-QueueLog {
    param([string] $Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append | Out-Null
}

Write-QueueLog "focused spike/self-election queue started root=$Root wait_pid=$WaitPid idle_max_util=$idleMaxUtil"

if ($WaitPid -gt 0) {
    $proc = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
    if ($null -ne $proc) {
        Write-QueueLog "waiting for pid=$WaitPid process=$($proc.ProcessName) start=$($proc.StartTime)"
        Wait-Process -Id $WaitPid
        Write-QueueLog "wait pid exited"
    } else {
        Write-QueueLog "wait pid=$WaitPid not found; starting immediately"
    }
}

$candidates = @(
    "i3l3r3_d640e256_q6_stable_control",
    "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_dense_anchor",
    "i3l3r3_d640e256_q6_vocabmoe_spikestatic_k16r2_input_top1",
    "i3l3r3_d640e256_q6_vocabmoe_spikestatic_k16r2_input_top2",
    "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopfirst_top2",
    "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_input_loopfirst_top2",
    "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopfirst_top2_nonorm"
) -join ","

Write-QueueLog "starting focused 16MB vocab MoE spike/self-election matrix out=$outDir"
$args = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "vocabmoe_spike",
    "--candidates", $candidates,
    "--out", $outDir,
    "--iterations", "5000",
    "--warmdown-iters", "5000",
    "--val-tokens", "131072",
    "--timeout", "9000",
    "--wait-for-idle-gpu",
    "--idle-max-util", $idleMaxUtil,
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "5"
)
& $python @args *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$exitCode = $LASTEXITCODE
Write-QueueLog "focused 16MB vocab MoE spike/self-election matrix exited code=$exitCode out=$outDir"
exit $exitCode
