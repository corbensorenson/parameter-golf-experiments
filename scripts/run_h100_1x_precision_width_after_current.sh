#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WAIT_PID="${1:-}"

is_wait_pid_alive() {
  [ -n "$WAIT_PID" ] && [ -r "/proc/${WAIT_PID}/cmdline" ]
}

echo "[h100-precision-width] queue started $(date -Is), waiting on pid ${WAIT_PID:-none}"
while is_wait_pid_alive; do
  echo "[h100-precision-width] waiting $(date -Is)"
  sleep 30
done

OUT="records/h100_1x_precision_width_scout_proxy_live"
FINAL_OUT="records/h100_1x_precision_width_scout_top_final_live"
LOG="/workspace/pg_h100_1x_precision_width_scout.out"
SCOUT_SECONDS="${SCOUT_SECONDS:-360}"
SCOUT_TIMEOUT="${SCOUT_TIMEOUT:-900}"
FINAL_SECONDS="${FINAL_SECONDS:-600}"
FINAL_TIMEOUT="${FINAL_TIMEOUT:-1500}"

rm -rf "$OUT" "$FINAL_OUT"
mkdir -p "$OUT"

echo "[h100-precision-width] proxy launch $(date -Is) seconds=$SCOUT_SECONDS out=$OUT"
env H100_1X_TRAIN_BATCH_TOKENS=65536 H100_1X_VAL_BATCH_SIZE=65536 \
  python scripts/run_h100_novel_matrix.py \
    --out "$OUT" \
    --candidate-group h100_1x_precision_width_scout \
    --nproc-per-node 1 \
    --wallclock-seconds "$SCOUT_SECONDS" \
    --val-tokens 131072 \
    --timeout "$SCOUT_TIMEOUT" \
    --skip-final-artifacts \
    2>&1 | tee "$LOG"

BEST="$(
  python - <<'PY'
import csv
import math
from pathlib import Path

rows = []
with Path("records/h100_1x_precision_width_scout_proxy_live/train.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        if row.get("returncode") != "0":
            continue
        raw = row.get("val_bpb", "")
        if not raw:
            continue
        value = float(raw)
        if math.isfinite(value):
            rows.append((value, row["candidate"]))
if not rows:
    raise SystemExit("no successful finite proxy rows")
rows.sort()
print(rows[0][1])
PY
)"

echo "[h100-precision-width] top final launch $(date -Is) candidate=$BEST seconds=$FINAL_SECONDS out=$FINAL_OUT"
env H100_1X_TRAIN_BATCH_TOKENS=65536 H100_1X_VAL_BATCH_SIZE=65536 \
  python scripts/run_h100_novel_matrix.py \
    --out "$FINAL_OUT" \
    --candidate-group h100_1x_precision_width_scout \
    --candidates "$BEST" \
    --nproc-per-node 1 \
    --wallclock-seconds "$FINAL_SECONDS" \
    --val-tokens 131072 \
    --timeout "$FINAL_TIMEOUT" \
    2>&1 | tee -a "$LOG"

echo "[h100-precision-width] done $(date -Is)"
