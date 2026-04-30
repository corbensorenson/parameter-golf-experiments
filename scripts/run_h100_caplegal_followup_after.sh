#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WAIT_PID="${1:-}"

is_wait_pid_alive() {
  [ -n "$WAIT_PID" ] && [ -r "/proc/${WAIT_PID}/cmdline" ]
}

echo "[caplegal] queue started $(date -Is), waiting on pid ${WAIT_PID:-none}"
while is_wait_pid_alive; do
  echo "[caplegal] waiting $(date -Is)"
  sleep 30
done

OUT="records/h100_caplegal_followup_proxy_live"
FINAL_OUT="records/h100_caplegal_followup_top_final_live"
LOG="/workspace/pg_h100_caplegal_followup.out"
rm -rf "$OUT" "$FINAL_OUT"
mkdir -p "$OUT"

echo "[caplegal] proxy launch $(date -Is) out=$OUT"
env H100_1X_TRAIN_BATCH_TOKENS=65536 H100_1X_VAL_BATCH_SIZE=65536 \
  python scripts/run_h100_novel_matrix.py \
    --out "$OUT" \
    --candidate-group h100_caplegal_followup \
    --nproc-per-node 1 \
    --wallclock-seconds 600 \
    --val-tokens 131072 \
    --timeout 1200 \
    --skip-final-artifacts \
    2>&1 | tee "$LOG"

BEST="$(
  python - <<'PY'
import csv
from pathlib import Path

rows = []
with Path("records/h100_caplegal_followup_proxy_live/train.csv").open(newline="") as f:
    for row in csv.DictReader(f):
        if row.get("returncode") == "0" and row.get("val_bpb"):
            rows.append((float(row["val_bpb"]), row["candidate"]))
if not rows:
    raise SystemExit("no successful proxy rows")
rows.sort()
print(rows[0][1])
PY
)"

echo "[caplegal] top final launch $(date -Is) candidate=$BEST out=$FINAL_OUT"
env H100_1X_TRAIN_BATCH_TOKENS=65536 H100_1X_VAL_BATCH_SIZE=65536 \
  python scripts/run_h100_novel_matrix.py \
    --out "$FINAL_OUT" \
    --candidate-group h100_caplegal_followup \
    --candidates "$BEST" \
    --nproc-per-node 1 \
    --wallclock-seconds 600 \
    --val-tokens 131072 \
    --timeout 1200 \
    2>&1 | tee -a "$LOG"

echo "[caplegal] done $(date -Is)"
