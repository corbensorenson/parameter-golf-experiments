param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf"
)

$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $Root

Write-Output "timestamp=$(Get-Date -Format o)"
Write-Output "root=$Root"
Write-Output ""

Write-Output "== gpu =="
try {
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,clocks.mem --format=csv,noheader,nounits
} catch {
    Write-Output "nvidia-smi unavailable: $($_.Exception.Message)"
}
Write-Output ""

Write-Output "== active queue/train processes =="
$processes = Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -like "*queue_*16mb*.ps1*" -or
            $_.CommandLine -like "*queue_pruned_continue_after_council1.ps1*" -or
            $_.CommandLine -like "*run_16mb_vocab_moe_matrix.py*" -or
            $_.CommandLine -like "*train_gpt_cuda2060.py*"
        )
    } |
    Select-Object ProcessId,Name,CreationDate,CommandLine |
    Sort-Object ProcessId
if ($processes) {
    $processes | Format-Table -Wrap -AutoSize
} else {
    Write-Output "none"
}
Write-Output ""

Write-Output "== recent queue logs =="
Get-ChildItem -LiteralPath (Join-Path $Root "records") -Filter "*.queue.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 8 FullName,LastWriteTime |
    Format-Table -AutoSize
Write-Output ""

Write-Output "== recent queue log tails =="
Get-ChildItem -LiteralPath (Join-Path $Root "records") -Filter "*.queue.log" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 4 |
    ForEach-Object {
        Write-Output "-- $($_.FullName)"
        Get-Content -LiteralPath $_.FullName -Tail 8
        Write-Output ""
    }

Write-Output "== recent train.csv rows =="
Get-ChildItem -LiteralPath (Join-Path $Root "records") -Recurse -Filter "train.csv" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 8 |
    ForEach-Object {
        Write-Output "-- $($_.FullName)"
        try {
            Import-Csv -LiteralPath $_.FullName |
                Select-Object -Last 6 |
                Select-Object candidate,final_export_val_bpb,final_quant_ttt_val_bpb,val_bpb,step_avg_ms,elapsed_s,artifact_total_bytes,returncode,rc |
                Format-Table -AutoSize
        } catch {
            Write-Output "failed to read csv: $($_.Exception.Message)"
        }
        Write-Output ""
    }

Write-Output "== latest active train log tails =="
Get-ChildItem -LiteralPath (Join-Path $Root "logs") -Filter "*.txt" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 3 |
    ForEach-Object {
        Write-Output "-- $($_.FullName) lastwrite=$($_.LastWriteTime.ToString('o'))"
        Get-Content -LiteralPath $_.FullName -Tail 10
        Write-Output ""
    }
