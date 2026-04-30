#!/usr/bin/env bash
set -euo pipefail

# No-fetch RunPod runner for the novel HRC/VocabMoE lane.
#
# This script intentionally does not git clone, pip install, wget, curl, or
# download data. It validates that the bundled repo, CaseOps data, tokenizer,
# and Python environment are present, then runs the requested matrix.
#
# Usage from an uploaded bundle on the pod:
#   bash scripts/runpod_run_novel_no_fetch.sh check
#   bash scripts/runpod_run_novel_no_fetch.sh smoke
#   bash scripts/runpod_run_novel_no_fetch.sh round1-1xh100
#   bash scripts/runpod_run_novel_no_fetch.sh round1-8xh100
#   bash scripts/runpod_run_novel_no_fetch.sh final8x

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CASEOPS_BASE="$ROOT/upstream_records/records/track_10min_16mb/2026-04-18_PR1626_CaseOps_Taper"
DATA_PATH="${DATA_PATH:-$CASEOPS_BASE/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$CASEOPS_BASE/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model}"

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

require_file() {
  [[ -f "$1" ]] || die "missing required file: $1"
}

require_dir() {
  [[ -d "$1" ]] || die "missing required directory: $1"
}

check_no_fetch_inputs() {
  cd "$ROOT"
  require_file "$ROOT/train_gpt.py"
  require_file "$ROOT/scripts/run_h100_novel_matrix.py"
  require_file "$ROOT/scripts/run_h100_8x_final_matrix.py"
  require_dir "$DATA_PATH"
  require_file "$TOKENIZER_PATH"
  require_file "$DATA_PATH/fineweb_train_000000.bin"
  require_file "$DATA_PATH/fineweb_val_000000.bin"
  require_file "$DATA_PATH/fineweb_val_bytes_000000.bin"
  python - <<'PY'
import importlib.util
import sys

missing = [name for name in ("torch", "numpy", "sentencepiece") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"missing Python modules: {', '.join(missing)}")
import torch
print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("cuda", torch.version.cuda, "available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(idx, torch.cuda.get_device_name(idx))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY
}

require_gpu_count() {
  local expected="$1"
  python - "$expected" <<'PY'
import sys
import torch

expected = int(sys.argv[1])
actual = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"required_gpus {expected} actual_gpus {actual}")
if actual < expected:
    raise SystemExit(f"expected at least {expected} CUDA devices, found {actual}")
if not torch.distributed.is_available() or not torch.distributed.is_nccl_available():
    raise SystemExit("torch.distributed NCCL backend is not available")
PY
}

list_candidates() {
  cd "$ROOT"
  local runner="${RUNPOD_LIST_RUNNER:-scripts/run_h100_novel_matrix.py}"
  local group="${RUNPOD_LIST_GROUP:-h100_novel_round1}"
  python "$runner" \
    --data-path "$DATA_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --candidate-group "$group" \
    --list
}

smoke() {
  cd "$ROOT"
  local out="records/runpod-no-fetch-smoke-$(date +%Y%m%d-%H%M%S)"
  log "starting no-fetch 1xH100 smoke out=$out"
  python scripts/run_h100_novel_matrix.py \
    --data-path "$DATA_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --candidate-group h100_novel_round1 \
    --candidates h100_capfill_i3l5r5_d640e896_q8_polar \
    --nproc-per-node 1 \
    --wallclock-seconds 180 \
    --val-tokens 65536 \
    --timeout 900 \
    --out "$out"
}

round1_1xh100() {
  cd "$ROOT"
  local out="records/runpod-no-fetch-round1-1x-$(date +%Y%m%d-%H%M%S)"
  log "starting no-fetch 1xH100 round1 out=$out"
  python scripts/run_h100_novel_matrix.py \
    --data-path "$DATA_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --candidate-group h100_novel_round1 \
    --nproc-per-node 1 \
    --wallclock-seconds 600 \
    --val-tokens 131072 \
    --timeout 1500 \
    --out "$out"
}

round1_8xh100() {
  cd "$ROOT"
  require_gpu_count 8
  local out="records/runpod-no-fetch-round1-8x-$(date +%Y%m%d-%H%M%S)"
  log "starting no-fetch 8xH100 round1 out=$out"
  python scripts/run_h100_novel_matrix.py \
    --data-path "$DATA_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --candidate-group h100_novel_round1 \
    --nproc-per-node 8 \
    --wallclock-seconds 600 \
    --val-tokens 131072 \
    --timeout 1500 \
    --out "$out"
}

final8x() {
  cd "$ROOT"
  require_gpu_count 8
  local out="records/runpod-final8x-$(date +%Y%m%d-%H%M%S)"
  log "starting focused no-fetch 8xH100 final matrix out=$out"
  python scripts/run_h100_8x_final_matrix.py \
    --data-path "$DATA_PATH" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --candidate-group h100_8x_final \
    --nproc-per-node 8 \
    --wallclock-seconds 600 \
    --val-tokens 131072 \
    --timeout 1800 \
    --out "$out"
}

mode="${1:-check}"
check_no_fetch_inputs
if [[ "$mode" == "final8x" || "$mode" == "final8x-check" ]]; then
  RUNPOD_LIST_RUNNER=scripts/run_h100_8x_final_matrix.py RUNPOD_LIST_GROUP=h100_8x_final list_candidates
else
  list_candidates
fi
nvidia-smi || true

case "$mode" in
  check)
    log "no-fetch check complete"
    ;;
  final8x-check)
    require_gpu_count 8
    log "focused 8x no-fetch check complete"
    ;;
  smoke)
    smoke
    ;;
  round1-1xh100)
    round1_1xh100
    ;;
  round1-8xh100)
    round1_8xh100
    ;;
  final8x)
    final8x
    ;;
  *)
    die "unknown mode '$mode'; expected check|final8x-check|smoke|round1-1xh100|round1-8xh100|final8x"
    ;;
esac
