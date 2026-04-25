param(
    [int[]]$WaitForPid = @(),
    [string]$RunId = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$GolfRoot = Split-Path -Parent (Split-Path -Parent $Root)
$LogDir = Join-Path $GolfRoot "logs"
$Python = Join-Path $Root ".venv-cuda313\Scripts\python.exe"
$Ladder = Join-Path $Root "scripts\run_caseops_loop_ladder_2060.py"

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Missing CUDA Python: $Python"
}
if (-not (Test-Path -LiteralPath $Ladder)) {
    throw "Missing loop ladder runner: $Ladder"
}
if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

if ([string]::IsNullOrWhiteSpace($RunId)) {
    $RunId = "caseops-overnight2060-loopmatrix-" + (Get-Date -Format "yyyyMMdd-HHmmss")
}

$StatusPath = Join-Path $LogDir "$RunId.status.txt"

function Add-Status {
    param([string]$Message)
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$stamp $Message" | Tee-Object -FilePath $StatusPath -Append
}

$Phases = @(
    @{
        Name = "geometry-expanded"
        Profiles = @(
            "loopplain5k_i2l3r3_q6proof",
            "loopplain5k_i3l4r3_q6proof",
            "loopplain5k_i3l5r2_q6proof",
            "loopplain5k_i3l6r2_q6proof",
            "loopplain5k_i4l3r2_q6proof",
            "loopplain5k_i2l6r1_q6proof"
        )
    },
    @{
        Name = "tail-repeat-sanity"
        Profiles = @(
            "loopplain5k_i1l3r3_q6proof",
            "loopplain5k_i2l3r2_q6proof",
            "loopplain5k_i3l3r1_q6proof",
            "loopplain5k_i4l3r1_q6proof",
            "loopplain5k_i2l5r2_q6proof"
        )
    },
    @{
        Name = "loop-feature-ttt"
        Profiles = @(
            "loopplain5k_i3l3r3_attngate_q6proof",
            "loopplain5k_i3l3r3_attngate_smear_q6proof",
            "loopplain5k_i3l3r3_ttt4_q6proof",
            "loopplain5k_i3l3r3_phasedttt_q6proof",
            "loopplain5k_i3l3r3_attngate_phasedttt_q6proof"
        )
    },
    @{
        Name = "loop-compression-rescue"
        Profiles = @(
            "loopplain5k_i3l3r3_q5mlp_q6proof",
            "loopplain5k_i3l3r3_q5blocks_q6proof",
            "loopplain5k_i3l3r3_q4mlp_q6proof",
            "loopplain5k_i3l3r3_ternarymlp_q6proof",
            "loopplain5k_i3l3r3_ternaryattn_q6proof",
            "loopplain5k_i3l3r3_ternaryblocks_g128_q6proof"
        )
    }
)

Add-Status "run_id=$RunId"
Add-Status "root=$Root"
Add-Status "wait_for_pid=$($WaitForPid -join ',')"
Add-Status "phase_count=$($Phases.Count)"

foreach ($pidToWait in $WaitForPid) {
    if ($pidToWait -gt 0) {
        $existing = Get-Process -Id $pidToWait -ErrorAction SilentlyContinue
        if ($null -ne $existing) {
            Add-Status "waiting pid=$pidToWait process=$($existing.ProcessName)"
            Wait-Process -Id $pidToWait -ErrorAction SilentlyContinue
            Add-Status "wait_done pid=$pidToWait"
        } else {
            Add-Status "wait_skipped_missing pid=$pidToWait"
        }
    }
}

foreach ($phase in $Phases) {
    $phaseRunId = "$RunId-$($phase.Name)"
    $phaseProfiles = $phase.Profiles -join ","
    $phaseLog = Join-Path $LogDir "$phaseRunId.host.txt"
    Add-Status "phase_start name=$($phase.Name) profiles=$phaseProfiles host_log=$phaseLog"

    $env:LOOP_LADDER_RUN_ID = $phaseRunId
    $env:LOOP_LADDER_PROFILES = $phaseProfiles
    $env:PYTHONUNBUFFERED = "1"

    & $Python -u $Ladder *>&1 | Tee-Object -FilePath $phaseLog
    $exitCode = $LASTEXITCODE

    Add-Status "phase_done name=$($phase.Name) exit_code=$exitCode suite_log=$(Join-Path $LogDir "$phaseRunId.suite.txt")"
    if ($exitCode -ne 0) {
        Add-Status "stopping_after_failed_phase name=$($phase.Name)"
        exit $exitCode
    }
}

Add-Status "overnight_done exit_code=0"
