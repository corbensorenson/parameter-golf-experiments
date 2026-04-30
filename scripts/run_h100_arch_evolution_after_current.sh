#!/usr/bin/env bash
set -euo pipefail

RUNNER_PID="${1:-}"
ROOT_DIR="/workspace/parameter-golf-novel-runpod-20260429-214806"
OUT_DIR="records/h100_arch_evolution_full10_after_beat135"

cd "$ROOT_DIR"
if [[ -n "$RUNNER_PID" ]]; then
  while kill -0 "$RUNNER_PID" 2>/dev/null; do
    sleep 30
  done
fi

python3 scripts/run_h100_arch_evolution_matrix.py \
  --out "$OUT_DIR" \
  --candidate-group h100_arch_evolution \
  --nproc-per-node 1 \
  --wallclock-seconds 600 \
  --val-tokens 131072 \
  --timeout 1500
