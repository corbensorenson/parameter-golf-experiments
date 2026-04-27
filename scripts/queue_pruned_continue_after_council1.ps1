param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf"
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $Root "records\pruned-continue-after-council1-$stamp.queue.log"
$python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"
$idleMaxUtil = "90"

function Write-QueueLog {
    param([string] $Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append | Out-Null
}

$councilOut = Join-Path $Root "records\vocabmoe16-council-rlm-pruned-cont-5k-auto-$stamp"
$sub4Out = Join-Path $Root "records\sub4-leader-pruned-5k-auto-$stamp"
$spikeOut = Join-Path $Root "records\vocabmoe16-spike-pruned-5k-auto-$stamp"
$widthOut = Join-Path $Root "records\sub4-width-pruned-5k-auto-$stamp"

$councilCandidates = @(
    "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_hard_t60",
    "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_dynamic_council_t60",
    "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_rlm_input_d90_s002",
    "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_rlm_council_signperm"
) -join ","

$sub4Candidates = @(
    "i4l9r5_d640e256_q16q8q4t_qk525_lqer_lidx_r8t16",
    "i4l9r5_d640e256_q16q8q4t_qk525_attnout24_lqer_lidx_r8t16",
    "i4l9r5_d640e256_q16q8q4t_qk525_huberwd_lqer_lidx_r8t16",
    "i4l9r5_d640e256_q16q8q4t_publicsafe_lqer_lidx_r8t16",
    "i4l11r5_d640e256_q16q8q8t_qk525_lqer_lidx_r8t16"
) -join ","

$spikeCandidates = @(
    "i3l3r3_d640e256_q6_vocabmoe_spikestatic_k16r2_input_top2",
    "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_input_loopfirst_top2"
) -join ","

$widthCandidates = @(
    "i4l9r5_d640e256_q16q8q4t_wl400-480-560-640_attncore1_lqer_lidx_r8t16",
    "i4l9r5_d640e256_q16q8q8t_wl320-480-560-640_attncore1_lqer_lidx_r8t16"
) -join ","

Write-QueueLog "continuing pruned queue after completed council signperm row; idle_max_util=$idleMaxUtil"

Write-QueueLog "starting remaining council/RLM matrix out=$councilOut"
$councilArgs = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "council_rlm",
    "--candidates", $councilCandidates,
    "--out", $councilOut,
    "--iterations", "5000",
    "--warmdown-iters", "5000",
    "--val-tokens", "131072",
    "--timeout", "9000",
    "--wait-for-idle-gpu",
    "--idle-max-util", $idleMaxUtil,
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "5"
)
& $python @councilArgs *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$councilExit = $LASTEXITCODE
Write-QueueLog "remaining council/RLM matrix exited code=$councilExit out=$councilOut"

Write-QueueLog "starting pruned sub4 leader matrix out=$sub4Out"
$sub4Args = @(
    "scripts\run_sub4_iotail_quant_matrix.py",
    "--candidates", $sub4Candidates,
    "--out", $sub4Out,
    "--iterations", "5000",
    "--warmdown-iters", "5000",
    "--val-tokens", "65536",
    "--timeout", "7200",
    "--final-artifacts",
    "--train-quant-forward",
    "--quant-train-every", "100",
    "--allow-over-cap",
    "--wait-for-idle-gpu",
    "--idle-max-util", $idleMaxUtil,
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "5"
)
& $python @sub4Args *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$sub4Exit = $LASTEXITCODE
Write-QueueLog "pruned sub4 leader matrix exited code=$sub4Exit out=$sub4Out"

Write-QueueLog "starting pruned spike matrix out=$spikeOut"
$spikeArgs = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "vocabmoe_spike",
    "--candidates", $spikeCandidates,
    "--out", $spikeOut,
    "--iterations", "5000",
    "--warmdown-iters", "5000",
    "--val-tokens", "131072",
    "--timeout", "9000",
    "--wait-for-idle-gpu",
    "--idle-max-util", $idleMaxUtil,
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "5"
)
& $python @spikeArgs *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$spikeExit = $LASTEXITCODE
Write-QueueLog "pruned spike matrix exited code=$spikeExit out=$spikeOut"

Write-QueueLog "starting pruned width matrix out=$widthOut"
$widthArgs = @(
    "scripts\run_sub4_iotail_quant_matrix.py",
    "--candidates", $widthCandidates,
    "--out", $widthOut,
    "--iterations", "5000",
    "--warmdown-iters", "5000",
    "--val-tokens", "65536",
    "--timeout", "7200",
    "--final-artifacts",
    "--train-quant-forward",
    "--quant-train-every", "100",
    "--allow-over-cap",
    "--wait-for-idle-gpu",
    "--idle-max-util", $idleMaxUtil,
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "5"
)
& $python @widthArgs *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$widthExit = $LASTEXITCODE
Write-QueueLog "pruned width matrix exited code=$widthExit out=$widthOut"

foreach ($code in @($councilExit, $sub4Exit, $spikeExit, $widthExit)) {
    if ($code -ne 0) {
        exit $code
    }
}
exit 0
