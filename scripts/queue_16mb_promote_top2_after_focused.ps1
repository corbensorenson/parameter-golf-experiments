param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf",
    [int] $WaitPid = 0,
    [int] $Iterations = 5000,
    [int] $TopK = 2,
    [double] $MaxPromoteBpb = 2.2
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Root

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $Root "records\promote-top2-16mb-after-focused-$stamp.queue.log"
$python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"

function Write-QueueLog {
    param([string] $Message)
    $line = "$(Get-Date -Format o) $Message"
    $line | Tee-Object -FilePath $logPath -Append | Out-Null
}

function Get-Score {
    param($Row)
    foreach ($field in @("final_quant_ttt_val_bpb", "final_export_val_bpb", "val_bpb")) {
        if ($Row.PSObject.Properties.Name -contains $field) {
            $raw = [string]$Row.$field
            if (-not [string]::IsNullOrWhiteSpace($raw)) {
                $score = 0.0
                if ([double]::TryParse($raw, [ref]$score)) {
                    return $score
                }
            }
        }
    }
    return [double]::PositiveInfinity
}

if ($WaitPid -gt 0) {
    Write-QueueLog "waiting for pid=$WaitPid before promoting focused winners"
    while (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
    Write-QueueLog "wait pid=$WaitPid finished"
}

$trainCsvs = Get-ChildItem -LiteralPath (Join-Path $Root "records") -Directory |
    Where-Object {
        $_.Name -like "vocabmoe16-cap-speed-scout-3k-auto-*" -or
        $_.Name -like "focused-16mb-after-capspeed-3000-auto-*"
    } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 4 |
    ForEach-Object {
        $csv = Join-Path $_.FullName "train.csv"
        if (Test-Path -LiteralPath $csv) { Get-Item -LiteralPath $csv }
    }

$rows = @()
foreach ($csv in $trainCsvs) {
    foreach ($row in (Import-Csv -LiteralPath $csv.FullName)) {
        $rc = if ($row.PSObject.Properties.Name -contains "returncode") { [string]$row.returncode } else { [string]$row.rc }
        if ($rc -ne "0") { continue }
        $score = Get-Score $row
        if ([double]::IsInfinity($score)) { continue }
        if ($score -gt $MaxPromoteBpb) { continue }
        $rows += [pscustomobject]@{
            candidate = [string]$row.candidate
            score = $score
            source = $csv.FullName
        }
    }
}

$selected = $rows |
    Sort-Object score |
    Group-Object candidate |
    ForEach-Object { $_.Group | Select-Object -First 1 } |
    Sort-Object score |
    Select-Object -First $TopK

if (-not $selected -or $selected.Count -eq 0) {
    Write-QueueLog "no successful scout rows found; skipping promotion"
    exit 0
}

foreach ($row in $selected) {
    Write-QueueLog ("selected candidate={0} score={1:n4} source={2}" -f $row.candidate, $row.score, $row.source)
}

$candidateCsv = ($selected | ForEach-Object { $_.candidate }) -join ","
$out = Join-Path $Root "records\promote-top2-16mb-${Iterations}-auto-$stamp"
Write-QueueLog "starting promoted 16MB queue out=$out candidates=$candidateCsv"

$args = @(
    "scripts\run_16mb_vocab_moe_matrix.py",
    "--candidate-group", "all",
    "--candidates", $candidateCsv,
    "--out", $out,
    "--iterations", "$Iterations",
    "--warmdown-iters", "$Iterations",
    "--val-tokens", "131072",
    "--timeout", "12000",
    "--final-artifacts",
    "--train-quant-forward",
    "--wait-for-idle-gpu",
    "--idle-max-util", "90",
    "--idle-max-memory-mib", "2500",
    "--idle-seconds", "5"
)
& $python @args *>&1 | Tee-Object -FilePath $logPath -Append | Out-Null
$exitCode = $LASTEXITCODE
Write-QueueLog "promoted 16MB queue exited code=$exitCode out=$out"
exit $exitCode
