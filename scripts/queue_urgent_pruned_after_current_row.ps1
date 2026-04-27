param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf",
    [string] $CurrentOutDir = "C:\Users\corbe\Documents\golf\workspace\parameter-golf\records\vocabmoe16-5k-auto-20260426-171524",
    [int] $TargetRows = 7,
    [int[]] $BroadQueuePids = @(31712, 31308, 5676, 38916)
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $Root "records\urgent-pruned-after-current-row-$stamp.queue.log"
$python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"

function Write-QueueLog {
    param([string] $Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append | Out-Null
}

function Get-CsvRowCount {
    param([string] $Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    $rows = Import-Csv -LiteralPath $Path
    if ($null -eq $rows) {
        return 0
    }
    return @($rows).Count
}

function Stop-ProcessTree {
    param([int] $ProcessId)
    $proc = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $proc) {
        return
    }
    $children = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $ProcessId }
    foreach ($child in $children) {
        Stop-ProcessTree -ProcessId ([int] $child.ProcessId)
    }
    Write-QueueLog "stopping pid=$ProcessId name=$($proc.ProcessName)"
    Stop-Process -Id $ProcessId -Force -ErrorAction SilentlyContinue
}

function Invoke-Matrix {
    param(
        [string] $Label,
        [string[]] $Args
    )
    Write-QueueLog "starting $Label"
    & $python @Args *>&1 | Tee-Object -FilePath $logPath -Append | Out-Host
    $exitCode = $LASTEXITCODE
    Write-QueueLog "$Label exited code=$exitCode"
    return $exitCode
}

$currentCsv = Join-Path $CurrentOutDir "train.csv"
Write-QueueLog "urgent prune watcher started current_csv=$currentCsv target_rows=$TargetRows broad_pids=$($BroadQueuePids -join ',')"
while ((Get-CsvRowCount -Path $currentCsv) -lt $TargetRows) {
    $count = Get-CsvRowCount -Path $currentCsv
    Write-QueueLog "waiting for active row to finish rows=$count/$TargetRows"
    Start-Sleep -Seconds 30
}
Write-QueueLog "active row recorded rows=$(Get-CsvRowCount -Path $currentCsv); stopping broad queues"
foreach ($pid in $BroadQueuePids) {
    Stop-ProcessTree -ProcessId $pid
}
Start-Sleep -Seconds 5

$councilOut = Join-Path $Root "records\vocabmoe16-council-rlm-urgent-5k-auto-$stamp"
$sub4Out = Join-Path $Root "records\sub4-leader-urgent-5k-auto-$stamp"
$spikeOut = Join-Path $Root "records\vocabmoe16-spike-urgent-5k-auto-$stamp"
$widthOut = Join-Path $Root "records\sub4-width-urgent-5k-auto-$stamp"

$councilCandidates = @(
    "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_signperm_o0m2",
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

$exitCodes = @()
$exitCodes += Invoke-Matrix -Label "urgent council/RLM matrix out=$councilOut" -Args @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "council_rlm",
    "--candidates", $councilCandidates,
    "--out", $councilOut,
    "--iterations", "5000",
    "--warmdown-iters", "5000",
    "--val-tokens", "131072",
    "--timeout", "9000",
    "--wait-for-idle-gpu",
    "--idle-max-util", "25",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "30"
)
$exitCodes += Invoke-Matrix -Label "urgent sub4 leader matrix out=$sub4Out" -Args @(
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
    "--idle-max-util", "25",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "30"
)
$exitCodes += Invoke-Matrix -Label "urgent spike matrix out=$spikeOut" -Args @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "vocabmoe_spike",
    "--candidates", $spikeCandidates,
    "--out", $spikeOut,
    "--iterations", "5000",
    "--warmdown-iters", "5000",
    "--val-tokens", "131072",
    "--timeout", "9000",
    "--wait-for-idle-gpu",
    "--idle-max-util", "25",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "30"
)
$exitCodes += Invoke-Matrix -Label "urgent width matrix out=$widthOut" -Args @(
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
    "--idle-max-util", "25",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "30"
)

$firstFail = @($exitCodes | Where-Object { $_ -ne 0 } | Select-Object -First 1)
if ($firstFail.Count -gt 0) {
    exit ([int] $firstFail[0])
}
exit 0
