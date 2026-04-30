param(
    [string]$HostName = "64.247.201.59",
    [int]$Port = 17996,
    [string]$KeyPath = "C:\Users\corbe\.ssh\runpod_codex_ed25519",
    [int]$Minutes = 120,
    [int]$PollSeconds = 600
)

$ErrorActionPreference = "Continue"

$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RemoteRoot = "/workspace/parameter-golf-novel-8x-ready"
$AuditDoc = Join-Path $Repo "records\h100_speed_audit_20260430.md"
$BreakDoc = Join-Path $Repo "records\h100_breakcliff_results_20260430.md"
$ArtifactRoot = Join-Path $Repo "artifacts\runpod-h100-primefollow-watch"
$StateDir = Join-Path $Repo "artifacts\watchdog-state"
$LogPath = Join-Path $StateDir "h100-primefollow-watchdog.log"
$EndAt = (Get-Date).AddMinutes($Minutes)

New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null
New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

function Write-WatchLog {
    param([string]$Message)
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-ddTHH:mm:ssK"), $Message
    Add-Content -Path $LogPath -Value $line
    Write-Output $line
}

function Invoke-Remote {
    param([string]$Command, [int]$TimeoutSeconds = 40)
    $args = @(
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=5",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-i", $KeyPath,
        "-p", "$Port",
        "root@$HostName",
        $Command
    )
    $p = Start-Process -FilePath "ssh" -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput "$StateDir\ssh.out" -RedirectStandardError "$StateDir\ssh.err"
    if (-not $p.WaitForExit($TimeoutSeconds * 1000)) {
        try { $p.Kill() } catch {}
        return @{ Ok = $false; Output = "ssh timeout" }
    }
    $out = ""
    if (Test-Path "$StateDir\ssh.out") { $out += Get-Content "$StateDir\ssh.out" -Raw }
    if (Test-Path "$StateDir\ssh.err") { $out += Get-Content "$StateDir\ssh.err" -Raw }
    return @{ Ok = ($p.ExitCode -eq 0); Output = $out }
}

function Copy-RemoteResults {
    param([string]$RemoteOutDir)
    if (-not $RemoteOutDir) { return $false }
    $safe = ($RemoteOutDir -replace "[/\\:]", "_")
    $dest = Join-Path $ArtifactRoot $safe
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $remotePath = "root@${HostName}:$RemoteRoot/$RemoteOutDir"
    $args = @(
        "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-i", $KeyPath,
        "-P", "$Port",
        "-r",
        $remotePath,
        $dest
    )
    $p = Start-Process -FilePath "scp" -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput "$StateDir\scp.out" -RedirectStandardError "$StateDir\scp.err"
    if (-not $p.WaitForExit(180000)) {
        try { $p.Kill() } catch {}
        Write-WatchLog "scp timeout for $RemoteOutDir"
        return $false
    }
    if ($p.ExitCode -ne 0) {
        $err = if (Test-Path "$StateDir\scp.err") { Get-Content "$StateDir\scp.err" -Raw } else { "" }
        Write-WatchLog "scp failed for ${RemoteOutDir}: $err"
        return $false
    }
    Write-WatchLog "copied $RemoteOutDir to $dest"
    return $true
}

function Get-LocalTrainRows {
    Get-ChildItem -Path $ArtifactRoot -Recurse -Filter train.csv -ErrorAction SilentlyContinue | ForEach-Object {
        try { Import-Csv $_.FullName } catch { @() }
    }
}

function Update-DocsFromRows {
    $rows = @(Get-LocalTrainRows | Where-Object { $_.final_export_val_bpb -and $_.artifact_total_bytes })
    if ($rows.Count -eq 0) { return $false }
    $sorted = $rows | Sort-Object {[double]$_.final_export_val_bpb}
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm zzz"
    $table = @()
    $table += ""
    $table += "### Prime-Follow Watchdog Snapshot - $stamp"
    $table += ""
    $table += "| Candidate | Final BPB | Steps | Step Speed | Bytes | Headroom |"
    $table += "| --- | ---: | ---: | ---: | ---: | ---: |"
    foreach ($row in ($sorted | Select-Object -First 12)) {
        $table += "| ``$($row.candidate)`` | ``$($row.final_export_val_bpb)`` | ``$($row.train_step)`` | ``$($row.step_avg_ms)ms`` | ``$($row.artifact_total_bytes)`` | ``$($row.artifact_headroom)`` |"
    }
    $block = ($table -join [Environment]::NewLine)
    Add-Content -Path $AuditDoc -Value $block
    Add-Content -Path $BreakDoc -Value $block
    Write-WatchLog "updated docs with $($rows.Count) exported rows"
    return $true
}

function Push-Results {
    Set-Location $Repo
    $status = git status --short
    if (-not $status) { return }
    git add README.md levers.md records/README.md records/h100_speed_audit_20260430.md records/h100_breakcliff_results_20260430.md scripts/run_h100_breakcliff_matrix.py scripts/watch_h100_primefollow.ps1 artifacts
    $staged = git diff --cached --name-only
    if (-not $staged) { return }
    $message = "Update H100 prime-skip results"
    git commit -m $message
    if ($LASTEXITCODE -ne 0) {
        Write-WatchLog "git commit skipped or failed"
        return
    }
    git push public HEAD
    Write-WatchLog "pushed to public remote with exit $LASTEXITCODE"
    git push fork HEAD
    Write-WatchLog "pushed to fork remote with exit $LASTEXITCODE"
}

function Ensure-NextQueue {
    $active = Invoke-Remote "pgrep -af 'run_h100_breakcliff|torchrun|train_gpt_arch' || true" 30
    if (-not $active.Ok) { return }
    if ($active.Output -match "run_h100_breakcliff|torchrun|train_gpt_arch") { return }
    $startedMarker = Join-Path $StateDir "prime-next-started.txt"
    if (Test-Path $startedMarker) { return }

    Write-WatchLog "remote appears idle; starting prime-next queue"
    $uploadArgs = @(
        "-o", "ConnectTimeout=15",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-i", $KeyPath,
        "-P", "$Port",
        (Join-Path $Repo "scripts\run_h100_breakcliff_matrix.py"),
        "root@${HostName}:$RemoteRoot/scripts/run_h100_breakcliff_matrix.py"
    )
    $p = Start-Process -FilePath "scp" -ArgumentList $uploadArgs -NoNewWindow -PassThru
    $p.WaitForExit(60000) | Out-Null

    $cmd = "cd $RemoteRoot && OUT=records/h100-prime-next-`$(date +%Y%m%d-%H%M%S) && echo `"`$OUT`" > /workspace/pg_h100_prime_next.outdir && nohup python scripts/run_h100_breakcliff_matrix.py --out `"`$OUT`" --skip-preflight --candidates prime_next_d640e960_lqer8t16,prime_next_qk525_d640e896,prime_next_wd02_d640e896,prime_next_d704e704_lqer8t16,prime_next_alt_skip02_d640e896 --nproc-per-node 1 --wallclock-seconds 600 --timeout 1500 > /workspace/pg_h100_prime_next.out 2>&1 < /dev/null & echo `$! > /workspace/pg_h100_prime_next.pid"
    $res = Invoke-Remote $cmd 30
    if ($res.Ok) {
        Set-Content -Path $startedMarker -Value (Get-Date).ToString("o")
        Write-WatchLog "started prime-next queue"
    } else {
        Write-WatchLog "failed to start prime-next queue: $($res.Output)"
    }
}

Write-WatchLog "watchdog starting for $Minutes minutes"

while ((Get-Date) -lt $EndAt) {
    $remote = Invoke-Remote "cd $RemoteRoot && echo reachable && date && (cat /workspace/pg_h100_primefollow_full.outdir 2>/dev/null || true) && (cat /workspace/pg_h100_prime_next.outdir 2>/dev/null || true) && (pgrep -af 'run_h100_breakcliff|torchrun|train_gpt_arch' || true) && nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader" 45
    if (-not $remote.Ok) {
        Write-WatchLog "remote unreachable: $($remote.Output)"
    } else {
        $snapshot = Join-Path $StateDir ("remote-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".txt")
        Set-Content -Path $snapshot -Value $remote.Output
        Write-WatchLog "remote reachable; wrote $snapshot"
        $outDirs = @()
        foreach ($line in ($remote.Output -split "`n")) {
            $trim = $line.Trim()
            if ($trim -like "records/h100-*") { $outDirs += $trim }
        }
        foreach ($dir in ($outDirs | Select-Object -Unique)) {
            Copy-RemoteResults $dir | Out-Null
        }
        $changed = Update-DocsFromRows
        if ($changed) { Push-Results }
        Ensure-NextQueue
    }
    Start-Sleep -Seconds $PollSeconds
}

Write-WatchLog "watchdog finished"
