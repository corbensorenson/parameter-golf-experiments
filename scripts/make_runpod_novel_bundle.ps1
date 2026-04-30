param(
    [string] $Root = "C:\Users\corbe\Documents\golf\workspace\parameter-golf",
    [string] $OutputDir = "C:\Users\corbe\Documents\golf\workspace\parameter-golf\tmp-runpod-bundles",
    [string] $BundleName = ""
)

$ErrorActionPreference = "Stop"

function Invoke-RobocopyChecked {
    param(
        [string] $Source,
        [string] $Destination,
        [string[]] $RobocopyArgs
    )
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    & robocopy $Source $Destination @RobocopyArgs | Out-Null
    $code = $LASTEXITCODE
    if ($code -ge 8) {
        throw "robocopy failed code=$code source=$Source destination=$Destination"
    }
}

function Require-StagedFile {
    param([string] $Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "staged bundle is missing required file: $Path"
    }
}

$Root = (Resolve-Path -LiteralPath $Root).Path
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
if ([string]::IsNullOrWhiteSpace($BundleName)) {
    $BundleName = "parameter-golf-novel-runpod-$stamp"
}
$OutputDir = (New-Item -ItemType Directory -Force -Path $OutputDir).FullName
$stageParent = Join-Path $OutputDir "stage-$stamp"
$stage = Join-Path $stageParent $BundleName
$archive = Join-Path $OutputDir "$BundleName.tar.gz"

$caseopsRel = "upstream_records\records\track_10min_16mb\2026-04-18_PR1626_CaseOps_Taper"
$caseopsSrc = Join-Path $Root $caseopsRel
if (-not (Test-Path -LiteralPath (Join-Path $caseopsSrc "datasets\fineweb10B_sp8192_lossless_caps_caseops_v1_reserved\fineweb_train_000000.bin"))) {
    throw "missing bundled CaseOps train shard under $caseopsSrc"
}
if (-not (Test-Path -LiteralPath (Join-Path $caseopsSrc "tokenizers\fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"))) {
    throw "missing bundled CaseOps tokenizer under $caseopsSrc"
}

if (Test-Path -LiteralPath $stageParent) {
    Remove-Item -LiteralPath $stageParent -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $stage | Out-Null

Write-Host "Copying repo to staging bundle..."
Invoke-RobocopyChecked -Source $Root -Destination $stage -RobocopyArgs @(
    "/E",
    "/XD", ".git", ".venv", ".venv-cuda313", ".venv-*", "tools", "logs", "tmp_cuda_extensions", ".mypy_cache", "__pycache__", "upstream_records", "data\datasets", "data\tokenizers", "tmp-runpod-bundles",
    "/XF", "*.pt", "*.ptz", "*.pth", "*.safetensors", "*.npy", "*.npz"
)

# Robocopy path-pattern exclusions can be permissive on Windows. Remove known
# heavyweight local-only trees explicitly so the pod upload stays focused.
foreach ($rel in @(
    "data\datasets",
    "data\tokenizers",
    "logs",
    "tools",
    "tmp_cuda_extensions",
    "records\track_10min_16mb",
    "records\track_non_record_16mb"
)) {
    $path = Join-Path $stage $rel
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
}
New-Item -ItemType Directory -Force -Path (Join-Path $stage "records") | Out-Null

Write-Host "Copying bundled CaseOps record/data..."
$caseopsDst = Join-Path $stage $caseopsRel
Invoke-RobocopyChecked -Source $caseopsSrc -Destination $caseopsDst -RobocopyArgs @(
    "/E",
    "/XD", "__pycache__", "logs",
    "/XF", "*.pt", "*.ptz", "*.pth", "*.safetensors", "*.npy", "*.npz"
)

# Be explicit about the files that make the no-fetch runner useful. If a future
# robocopy exclusion changes, this protects paid pod time by failing locally.
$caseopsDataRel = "datasets\fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
$caseopsTokRel = "tokenizers"
New-Item -ItemType Directory -Force -Path (Join-Path $caseopsDst $caseopsDataRel) | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $caseopsDst $caseopsTokRel) | Out-Null
foreach ($name in @("fineweb_train_000000.bin", "fineweb_val_000000.bin", "fineweb_val_bytes_000000.bin")) {
    Copy-Item -LiteralPath (Join-Path $caseopsSrc (Join-Path $caseopsDataRel $name)) -Destination (Join-Path $caseopsDst (Join-Path $caseopsDataRel $name)) -Force
}
foreach ($name in @("fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model", "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.vocab")) {
    Copy-Item -LiteralPath (Join-Path $caseopsSrc (Join-Path $caseopsTokRel $name)) -Destination (Join-Path $caseopsDst (Join-Path $caseopsTokRel $name)) -Force
}

# Remove stale export artifacts if they existed in the working tree. They are
# not needed on RunPod and only slow upload.
foreach ($name in @("final_model.pt", "final_model.int8.ptz")) {
    $stale = Join-Path $stage $name
    if (Test-Path -LiteralPath $stale) {
        Remove-Item -LiteralPath $stale -Force
    }
}

Require-StagedFile (Join-Path $caseopsDst "datasets\fineweb10B_sp8192_lossless_caps_caseops_v1_reserved\fineweb_train_000000.bin")
Require-StagedFile (Join-Path $caseopsDst "datasets\fineweb10B_sp8192_lossless_caps_caseops_v1_reserved\fineweb_val_000000.bin")
Require-StagedFile (Join-Path $caseopsDst "datasets\fineweb10B_sp8192_lossless_caps_caseops_v1_reserved\fineweb_val_bytes_000000.bin")
Require-StagedFile (Join-Path $caseopsDst "tokenizers\fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model")
Require-StagedFile (Join-Path $stage "scripts\run_h100_8x_final_matrix.py")
Require-StagedFile (Join-Path $stage "scripts\runpod_run_novel_no_fetch.sh")

$readme = @"
# RunPod No-Fetch Bundle

Created: $(Get-Date -Format o)

This archive contains the experiment repo plus the local CaseOps/SP8192 smoke
dataset shard and tokenizer needed by `scripts/runpod_run_novel_no_fetch.sh`.

On the pod:

```bash
cd /workspace
tar -xzf $BundleName.tar.gz
cd $BundleName
bash scripts/runpod_run_novel_no_fetch.sh check
bash scripts/runpod_run_novel_no_fetch.sh final8x-check
bash scripts/runpod_run_novel_no_fetch.sh smoke
```

For the five-row 1xH100 scout:

```bash
bash scripts/runpod_run_novel_no_fetch.sh round1-1xh100
```

For the five-row 8xH100 scout:

```bash
bash scripts/runpod_run_novel_no_fetch.sh round1-8xh100
```

For the focused paid 8xH100 final-hour slate:

```bash
bash scripts/runpod_run_novel_no_fetch.sh final8x
```

No git clone, PR fetch, or dataset download is performed by the no-fetch
runner. If a required file is missing, it exits.
"@
$readme | Set-Content -LiteralPath (Join-Path $stage "RUNPOD_BUNDLE_README.md") -Encoding UTF8

Write-Host "Creating archive $archive ..."
if (Test-Path -LiteralPath $archive) {
    Remove-Item -LiteralPath $archive -Force
}
tar -czf $archive -C $stageParent $BundleName

$item = Get-Item -LiteralPath $archive
Write-Host "Bundle ready:"
Write-Host "  $($item.FullName)"
Write-Host ("  size={0:n1} MB" -f ($item.Length / 1MB))
Write-Host ""
Write-Host "Upload example once RunPod gives SSH details:"
Write-Host "  scp `"$($item.FullName)`" root@<host>:/workspace/"
Write-Host "  ssh root@<host>"
Write-Host "  cd /workspace && tar -xzf $($item.Name) && cd $BundleName"
Write-Host "  bash scripts/runpod_run_novel_no_fetch.sh check"
