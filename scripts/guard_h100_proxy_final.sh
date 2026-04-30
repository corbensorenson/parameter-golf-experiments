#!/usr/bin/env bash
set -euo pipefail

PROXY_PID="${1:?proxy pid required}"
WRAPPER_PID="${2:?wrapper pid required}"
TRAIN_CSV="${3:?train csv path required}"
THRESHOLD="${4:-1.60}"
FINAL_PATTERN="${5:-h100_1x_precision_width_scout_top_final_live}"

echo "[proxy-final-guard] started $(date -Is) proxy=$PROXY_PID wrapper=$WRAPPER_PID threshold=$THRESHOLD csv=$TRAIN_CSV"
while kill -0 "$PROXY_PID" 2>/dev/null; do
  sleep 5
done
sleep 1

BEST_INFO="$(
  python - "$TRAIN_CSV" <<'PY'
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = []
if path.is_file():
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("returncode") != "0" or not row.get("val_bpb"):
                continue
            value = float(row["val_bpb"])
            if math.isfinite(value):
                rows.append((value, row["candidate"]))
if not rows:
    print("inf none")
else:
    rows.sort()
    print(f"{rows[0][0]:.8f} {rows[0][1]}")
PY
)"
BEST_BPB="${BEST_INFO%% *}"
BEST_CANDIDATE="${BEST_INFO#* }"
echo "[proxy-final-guard] best=$BEST_BPB candidate=$BEST_CANDIDATE $(date -Is)"

if ! python - "$BEST_BPB" "$THRESHOLD" <<'PY'
import math
import sys

try:
    best = float(sys.argv[1])
except ValueError:
    best = math.inf
threshold = float(sys.argv[2])
raise SystemExit(0 if best <= threshold else 1)
PY
then
  echo "[proxy-final-guard] stopping final path because best $BEST_BPB > $THRESHOLD"
  pkill -TERM -P "$WRAPPER_PID" 2>/dev/null || true
  pkill -TERM -f "$FINAL_PATTERN" 2>/dev/null || true
  kill -TERM "$WRAPPER_PID" 2>/dev/null || true
fi

echo "[proxy-final-guard] done $(date -Is)"
