$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$LogRoot = "C:\Users\corbe\Documents\golf\logs"
$Timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunId = "caseops-candidate2060-compare20k-$Timestamp"
$Stdout = Join-Path $LogRoot "$RunId.txt"
$Stderr = Join-Path $LogRoot "$RunId.err.txt"
$Status = Join-Path $LogRoot "caseops-candidate2060-compare20k-live.status.txt"
$Python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"
$Launcher = Join-Path $Root "scripts\run_caseops_candidate_2060_compare.py"

@"
run_id=$RunId
stdout=$Stdout
stderr=$Stderr
started=$(Get-Date -Format o)
"@ | Set-Content -LiteralPath $Status -Encoding ASCII

$env:CANDIDATE2060_PROFILE = "compare20k"
$env:RUN_ID = $RunId

& $Python -u $Launcher 1> $Stdout 2> $Stderr
$ExitCode = $LASTEXITCODE

Add-Content -LiteralPath $Status -Encoding ASCII -Value "completed=$(Get-Date -Format o)"
Add-Content -LiteralPath $Status -Encoding ASCII -Value "exit_code=$ExitCode"

exit $ExitCode
