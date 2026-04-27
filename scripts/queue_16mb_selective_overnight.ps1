param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf",
    [int] $WaitPid = 0,
    [int] $LongIterations = 5000,
    [int] $LeaderboardIterations = 5000,
    [int] $TopK = 6,
    [int] $DualIterations = 5000,
    [double] $DualThresholdBpb = 1.95,
    [switch] $SkipLeaderboard
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $Root "records\selective-16mb-overnight-$stamp.queue.log"
$python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"

function Write-QueueLog {
    param([string] $Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append | Out-Null
}

function Score-Row {
    param($Row)
    foreach ($field in @("final_quant_ttt_val_bpb", "final_export_val_bpb", "val_bpb")) {
        $raw = [string] $Row.$field
        if (-not [string]::IsNullOrWhiteSpace($raw)) {
            $value = 0.0
            if ([double]::TryParse($raw, [ref] $value)) {
                return $value
            }
        }
    }
    return [double]::PositiveInfinity
}

function Latest-TrainCsv {
    param([string] $Pattern)
    $dir = Get-ChildItem -LiteralPath (Join-Path $Root "records") -Directory -Filter $Pattern |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $dir) {
        return $null
    }
    $csv = Join-Path $dir.FullName "train.csv"
    if (Test-Path -LiteralPath $csv) {
        return $csv
    }
    return $null
}

if ($WaitPid -gt 0) {
    Write-QueueLog "waiting for pid=$WaitPid before launching selective 5k exploration queue"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    Write-QueueLog "wait pid=$WaitPid finished"
}

if (-not $SkipLeaderboard) {
    $leaderOut = Join-Path $Root "records\cap16-leaderboard-5k-${LeaderboardIterations}-auto-$stamp"
    Write-QueueLog "starting leaderboard-inspired 5k queue out=$leaderOut group=cap16_leaderboard"
    $leaderArgs = @(
        "scripts\run_16mb_vocab_moe_matrix.py",
        "--candidate-group", "cap16_leaderboard",
        "--out", $leaderOut,
        "--iterations", "$LeaderboardIterations",
        "--warmdown-iters", "$LeaderboardIterations",
        "--val-tokens", "131072",
        "--timeout", "18000",
        "--final-artifacts",
        "--train-quant-forward",
        "--wait-for-idle-gpu",
        "--idle-max-util", "90",
        "--idle-max-memory-mib", "2500",
        "--idle-seconds", "5"
    )
    & $python @leaderArgs *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
    $leaderExit = $LASTEXITCODE
    Write-QueueLog "leaderboard-inspired 5k queue exited code=$leaderExit out=$leaderOut"
    if ($leaderExit -ne 0) {
        exit $leaderExit
    }
}

$csvs = @()
foreach ($pattern in @("vocabmoe16-cap-speed-scout-3k-auto-*", "cap16-mainline-scout-*-auto-*")) {
    $csv = Latest-TrainCsv -Pattern $pattern
    if ($null -ne $csv) {
        $csvs += $csv
        Write-QueueLog "using scout csv=$csv"
    } else {
        Write-QueueLog "no scout csv found for pattern=$pattern"
    }
}

$rows = @()
foreach ($csv in $csvs) {
    foreach ($row in (Import-Csv -LiteralPath $csv)) {
        $score = Score-Row -Row $row
        $rc = [string] $row.returncode
        if ($rc -eq "0" -and [double]::IsFinite($score)) {
            $rows += [pscustomobject]@{
                candidate = [string] $row.candidate
                score = $score
                step_avg_ms = [string] $row.step_avg_ms
                bytes = [string] $row.artifact_total_bytes
                source = $csv
            }
        }
    }
}

if ($rows.Count -eq 0) {
    Write-QueueLog "no valid scout rows found; falling back to the three highest-confidence mainline rows"
    $selected = @(
        "mainline_i3l3r3_d768e384_q6all_vocabmoe_qk525_lqer16t32",
        "mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32",
        "mainline_i3l5r2_d768e320_q8q6q6_q4core_vocabmoe_qk525_lqer16t32"
    )
} else {
    $selected = $rows |
        Sort-Object score |
        Select-Object -First $TopK |
        ForEach-Object {
            Write-QueueLog (
                "selected long row candidate={0} bpb={1:n4} step_avg_ms={2} bytes={3} source={4}" -f
                $_.candidate, $_.score, $_.step_avg_ms, $_.bytes, $_.source
            )
            $_.candidate
        }
}

$selectedCsv = ($selected -join ",")
$longOut = Join-Path $Root "records\cap16-selected-5k-explore-${LongIterations}-auto-$stamp"
Write-QueueLog "starting selected 5k exploration queue out=$longOut candidates=$selectedCsv"
$longArgs = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "all",
    "--candidates", $selectedCsv,
    "--out", $longOut,
    "--iterations", "$LongIterations",
    "--warmdown-iters", "$LongIterations",
    "--val-tokens", "131072",
    "--timeout", "18000",
    "--final-artifacts",
    "--train-quant-forward",
    "--wait-for-idle-gpu",
    "--idle-max-util", "90",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "5"
)
& $python @longArgs *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$longExit = $LASTEXITCODE
Write-QueueLog "selected 5k exploration queue exited code=$longExit out=$longOut"
if ($longExit -ne 0) {
    exit $longExit
}

$bestMainline = $rows |
    Where-Object { $_.candidate -like "mainline_*" } |
    Sort-Object score |
    Select-Object -First 1
if ($null -eq $bestMainline) {
    Write-QueueLog "no valid mainline scout row; skipping dual-stream canaries"
    exit 0
}
if ($bestMainline.score -gt $DualThresholdBpb) {
    Write-QueueLog ("best mainline bpb={0:n4} is above dual threshold={1:n4}; skipping dual-stream canaries" -f $bestMainline.score, $DualThresholdBpb)
    exit 0
}

$dualCandidates = @(
    "dual_i3l3r3_d768e320_left320_q6all_vocabmoe_qk525_lqer12t24",
    "dual_i3l5r2_d768e320_left256_q6all_vocabmoe_qk525_lqer16t32"
)
if ($bestMainline.candidate -like "*d896*") {
    $dualCandidates += "dual_i3l5r2_d896e384_left320_q8q6q6_q4core_vocabmoe_qk525_lqer16t32"
}
$dualCsv = ($dualCandidates -join ",")
$dualOut = Join-Path $Root "records\cap16-dual-canary-${DualIterations}-auto-$stamp"
Write-QueueLog "starting dual canary queue out=$dualOut candidates=$dualCsv best_mainline=$($bestMainline.candidate) bpb=$($bestMainline.score)"
$dualArgs = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "cap16_dual_stream",
    "--candidates", $dualCsv,
    "--out", $dualOut,
    "--iterations", "$DualIterations",
    "--warmdown-iters", "$DualIterations",
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
Write-QueueLog "dual canary queue exited code=$dualExit out=$dualOut"
exit $dualExit
