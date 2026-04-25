param(
    [string]$RunId = "caseops-overnight2060-loopmatrix",
    [int]$Iterations = 64,
    [int]$IntervalSeconds = 900
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$GolfRoot = Split-Path -Parent (Split-Path -Parent $Root)
$Python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"
$Summary = Join-Path $Root "scripts\summarize_caseops_ladder_results.py"
$LogDir = Join-Path $GolfRoot "logs"
$Output = Join-Path $LogDir "$RunId.leaderboard.md"
$Status = Join-Path $LogDir "$RunId.leaderboard.status.txt"

function Add-Status {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp $Message" | Tee-Object -FilePath $Status -Append
}

Add-Status "leaderboard_poller_start run_id=$RunId iterations=$Iterations interval_seconds=$IntervalSeconds output=$Output"

for ($i = 1; $i -le $Iterations; $i++) {
    & $Python $Summary --output $Output > $null
    Add-Status "leaderboard_refresh iteration=$i output=$Output"
    Start-Sleep -Seconds $IntervalSeconds
}

Add-Status "leaderboard_poller_done"
