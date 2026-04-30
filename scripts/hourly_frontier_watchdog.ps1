param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf",
    [string] $CurrentRun = "records\cap16-frontier-capfill-5k-auto-20260429-003802",
    [int] $ExpectedRows = 4
)

$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $Root

$currentDir = Join-Path $Root $CurrentRun
$watchdog = Join-Path $Root "records\hourly-watchdog.md"
$timestamp = Get-Date -Format o
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$marker = Join-Path $currentDir ".frontier-followup-launched"
$python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"

$capfillCsv = Join-Path $currentDir "train.csv"
if (Test-Path -LiteralPath $capfillCsv) {
    $capfillRows = @(Import-Csv -LiteralPath $capfillCsv)
    $latestFollowup = Get-ChildItem -LiteralPath (Join-Path $Root "records") -Directory -Filter "cap16-frontier-followup-5k-auto-*" -ErrorAction SilentlyContinue |
        Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "candidate_plan.md") } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($capfillRows.Count -ge $ExpectedRows -and $null -ne $latestFollowup) {
        $CurrentRun = "records\$($latestFollowup.Name)"
        $currentDir = $latestFollowup.FullName
    }
}
$isCapfillRun = (Split-Path -Leaf $currentDir) -like "cap16-frontier-capfill-*"

function Get-Score {
    param($Row)
    foreach ($field in @("final_quant_ttt_val_bpb", "final_export_val_bpb", "val_bpb")) {
        if ($Row.PSObject.Properties.Name -contains $field) {
            $raw = [string] $Row.$field
            if (-not [string]::IsNullOrWhiteSpace($raw)) {
                $value = 0.0
                if ([double]::TryParse($raw, [ref] $value)) {
                    return $value
                }
            }
        }
    }
    return [double]::PositiveInfinity
}

function Append-Watchdog {
    param([string[]] $Lines)
    Add-Content -LiteralPath $watchdog -Value $Lines -Encoding UTF8
}

$gpuLine = "nvidia-smi unavailable"
try {
    $gpuLine = nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
} catch {
    $gpuLine = "nvidia-smi unavailable: $($_.Exception.Message)"
}

$trainCsv = Join-Path $currentDir "train.csv"
$rows = @()
if (Test-Path -LiteralPath $trainCsv) {
    $rows = @(Import-Csv -LiteralPath $trainCsv)
}

$best = $null
if ($rows.Count -gt 0) {
    $best = $rows |
        Where-Object { [string]$_.returncode -eq "0" } |
        Sort-Object @{ Expression = { Get-Score $_ } } |
        Select-Object -First 1
}

$activePython = @(Get-Process python,py -ErrorAction SilentlyContinue)
$rowSummary = if ($rows.Count -gt 0) {
    ($rows | ForEach-Object {
        $score = Get-Score $_
        if ([double]::IsInfinity($score)) {
            "- ``{0}``: score=n/a rc={1}" -f $_.candidate, $_.returncode
        } else {
            "- ``{0}``: BPB={1:n8}, step={2}ms, bytes={3}, rc={4}" -f $_.candidate, $score, $_.step_avg_ms, $_.artifact_total_bytes, $_.returncode
        }
    }) -join "`n"
} else {
    "- no completed rows flushed yet"
}

$statusLines = @(
    "",
    "## $timestamp",
    "",
    "Automation status check for frontier cap-fill queue.",
    "",
    "- Run: ``$CurrentRun``",
    "- Completed rows: ``$($rows.Count)/$ExpectedRows``",
    "- GPU: ``$gpuLine``",
    "- Active Python processes: ``$($activePython.Count)``",
    "",
    "Rows:",
    $rowSummary
)

if ($best -ne $null) {
    $bestScore = Get-Score $best
    $statusLines += ""
    $statusLines += "Best so far: ``$($best.candidate)`` at ``$("{0:n8}" -f $bestScore)`` BPB."
}

$complete = ($rows.Count -ge $ExpectedRows)
if ($complete -and $isCapfillRun) {
    $markerExists = Test-Path -LiteralPath $marker
    if ($markerExists) {
        $statusLines += ""
        $statusLines += "Current queue is complete and follow-up marker already exists; no new launch to avoid duplicate queues."
    } else {
        $followOut = Join-Path $Root "records\cap16-frontier-followup-5k-auto-$stamp"
        $followStdout = Join-Path $followOut "queue.out.log"
        $followStderr = Join-Path $followOut "queue.err.log"
        New-Item -ItemType Directory -Force -Path $followOut | Out-Null
        $args = @(
            "-u", "scripts\run_16mb_vocab_moe_matrix.py",
            "--candidate-group", "cap16_frontier_followup",
            "--out", $followOut,
            "--iterations", "5000",
            "--warmdown-iters", "5000",
            "--val-tokens", "131072",
            "--timeout", "18000",
            "--final-artifacts",
            "--train-quant-forward",
            "--wait-for-idle-gpu",
            "--idle-max-util", "90",
            "--idle-max-memory-mib", "2500",
            "--idle-seconds", "5"
        )
        try {
            $proc = Start-Process -FilePath $python `
                -ArgumentList $args `
                -WorkingDirectory $Root `
                -WindowStyle Hidden `
                -RedirectStandardOutput $followStdout `
                -RedirectStandardError $followStderr `
                -PassThru

            Start-Sleep -Seconds 3
            if ($proc.HasExited) {
                $statusLines += ""
                $statusLines += "Current queue complete, but follow-up exited immediately with code ``$($proc.ExitCode)``. See ``$followStdout`` and ``$followStderr``."
            } else {
                "launched $timestamp pid=$($proc.Id) out=$followOut" | Set-Content -LiteralPath $marker -Encoding UTF8
                $statusLines += ""
                $statusLines += "Current queue complete. Launched ``cap16_frontier_followup`` as PID ``$($proc.Id)`` in ``$followOut``."
            }
        } catch {
            $statusLines += ""
            $statusLines += "Current queue complete, but follow-up launch failed: ``$($_.Exception.Message)``."
        }
    }
} elseif ($complete) {
    $statusLines += ""
    $statusLines += "Current monitored queue is complete; no automatic next queue is configured from this follow-up."
} else {
    $statusLines += ""
    $statusLines += "Current queue is still running or incomplete; no follow-up launch."
}

Append-Watchdog -Lines $statusLines
$statusLines -join "`n"
