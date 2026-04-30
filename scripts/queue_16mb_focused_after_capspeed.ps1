param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf",
    [int] $WaitPid = 0,
    [int] $Iterations = 3000
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $Root "records\focused-16mb-after-capspeed-$stamp.queue.log"
$python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"
$out = Join-Path $Root "records\focused-16mb-after-capspeed-${Iterations}-auto-$stamp"

$candidates = @(
    "mainline_i3l3r3_d768e384_q6all_vocabmoe_qk525_lqer16t32",
    "mainline_i3l3r3_d896e384_q6all_vocabmoe_qk525_lqer16t32",
    "mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32",
    "leader_i3l3r3_d768e320_q6all_polar_minlr_vocabmoe_qk525_lqer16t32",
    "leader_i3l3r3_d768e320_q6all_sparsegate_polar_minlr_vocabmoe_lqer16t32",
    "leader_i3l3r3_d768e320_q6all_depthlora4_polar_minlr_vocabmoe_lqer16t32",
    "leader_i3l5r1rev_d768e320_q6all_polar_minlr_vocabmoe_qk525_lqer16t32"
)
$candidateCsv = $candidates -join ","

function Write-QueueLog {
    param([string] $Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append | Out-Null
}

if ($WaitPid -gt 0) {
    Write-QueueLog "waiting for pid=$WaitPid before launching focused 16MB queue"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    Write-QueueLog "wait pid=$WaitPid finished"
}

Write-QueueLog "starting focused 16MB queue out=$out candidates=$candidateCsv"
$args = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "all",
    "--candidates", $candidateCsv,
    "--out", $out,
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
& $python @args *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$exitCode = $LASTEXITCODE
Write-QueueLog "focused 16MB queue exited code=$exitCode out=$out"
exit $exitCode
