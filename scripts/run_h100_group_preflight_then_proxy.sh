#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WAIT_PID="${1:-}"
GROUP="${GROUP:-h100_1x_capfit_scout}"
CAP_BYTES="${CAP_BYTES:-16000000}"
PREFLIGHT_STEPS="${PREFLIGHT_STEPS:-1}"
PREFLIGHT_VAL_TOKENS="${PREFLIGHT_VAL_TOKENS:-8192}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600}"
SCOUT_SECONDS="${SCOUT_SECONDS:-360}"
SCOUT_TIMEOUT="${SCOUT_TIMEOUT:-900}"
SCOUT_VAL_TOKENS="${SCOUT_VAL_TOKENS:-131072}"
FINAL_SECONDS="${FINAL_SECONDS:-600}"
FINAL_TIMEOUT="${FINAL_TIMEOUT:-1500}"
FINAL_VAL_TOKENS="${FINAL_VAL_TOKENS:-131072}"
FINAL_MAX_PROXY_BPB="${FINAL_MAX_PROXY_BPB:-}"

PRE_OUT="${PRE_OUT:-records/${GROUP}_size_preflight_live}"
OUT="${OUT:-records/${GROUP}_proxy_live}"
FINAL_OUT="${FINAL_OUT:-records/${GROUP}_top_final_live}"
LOG="${LOG:-/workspace/${GROUP}_preflight_proxy.out}"

is_wait_pid_alive() {
  [ -n "$WAIT_PID" ] && [ -r "/proc/${WAIT_PID}/cmdline" ]
}

echo "[h100-preflight] group=$GROUP cap=$CAP_BYTES queue started $(date -Is), waiting on pid ${WAIT_PID:-none}"
while is_wait_pid_alive; do
  echo "[h100-preflight] waiting $(date -Is)"
  sleep 30
done

rm -rf "$PRE_OUT" "$OUT" "$FINAL_OUT"
mkdir -p "$PRE_OUT" "$OUT"
: > "$LOG"

if [ -n "${CANDIDATES:-}" ]; then
  IFS=',' read -r -a ALL_CANDIDATES <<< "$CANDIDATES"
else
  mapfile -t ALL_CANDIDATES < <(python scripts/run_h100_novel_matrix.py --list --candidate-group "$GROUP")
fi
ACCEPTED=()

for candidate in "${ALL_CANDIDATES[@]}"; do
  candidate_dir="$PRE_OUT/$candidate"
  echo "[h100-preflight] preflight launch $(date -Is) candidate=$candidate" | tee -a "$LOG"
  env H100_1X_TRAIN_BATCH_TOKENS=65536 H100_1X_VAL_BATCH_SIZE=65536 \
    python scripts/run_h100_novel_matrix.py \
      --out "$candidate_dir" \
      --candidate-group "$GROUP" \
      --candidates "$candidate" \
      --nproc-per-node 1 \
      --iterations "$PREFLIGHT_STEPS" \
      --warmdown-iters "$PREFLIGHT_STEPS" \
      --wallclock-seconds 0 \
      --val-tokens "$PREFLIGHT_VAL_TOKENS" \
      --timeout "$PREFLIGHT_TIMEOUT" \
      2>&1 | tee -a "$LOG"

  decision="$(
    python - "$candidate_dir/train.csv" "$CAP_BYTES" <<'PY'
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
cap = int(sys.argv[2])
if not path.is_file():
    print("reject missing_csv")
    raise SystemExit
rows = list(csv.DictReader(path.open(newline="")))
if not rows:
    print("reject empty_csv")
    raise SystemExit
row = rows[-1]
if row.get("returncode") != "0":
    print(f"reject rc={row.get('returncode')}")
    raise SystemExit
raw_bytes = row.get("artifact_total_bytes") or row.get("artifact_model_bytes") or ""
try:
    total = int(float(raw_bytes))
except ValueError:
    print(f"reject missing_bytes={raw_bytes!r}")
    raise SystemExit
headroom = cap - total
if total <= cap:
    print(f"accept bytes={total} headroom={headroom}")
else:
    print(f"reject bytes={total} headroom={headroom}")
PY
  )"
  echo "[h100-preflight] $candidate $decision" | tee -a "$LOG"
  if [[ "$decision" == accept* ]]; then
    ACCEPTED+=("$candidate")
  fi
done

if [ "${#ACCEPTED[@]}" -eq 0 ]; then
  echo "[h100-preflight] no candidates passed size preflight; exiting" | tee -a "$LOG"
  exit 2
fi

ACCEPTED_CSV="$(IFS=,; echo "${ACCEPTED[*]}")"
echo "[h100-preflight] accepted=$ACCEPTED_CSV" | tee -a "$LOG"

echo "[h100-preflight] proxy launch $(date -Is) seconds=$SCOUT_SECONDS out=$OUT" | tee -a "$LOG"
env H100_1X_TRAIN_BATCH_TOKENS=65536 H100_1X_VAL_BATCH_SIZE=65536 \
  python scripts/run_h100_novel_matrix.py \
    --out "$OUT" \
    --candidate-group "$GROUP" \
    --candidates "$ACCEPTED_CSV" \
    --nproc-per-node 1 \
    --wallclock-seconds "$SCOUT_SECONDS" \
    --val-tokens "$SCOUT_VAL_TOKENS" \
    --timeout "$SCOUT_TIMEOUT" \
    --skip-final-artifacts \
    2>&1 | tee -a "$LOG"

BEST_INFO="$(
  python - "$OUT/train.csv" <<'PY'
import csv
import math
import sys
from pathlib import Path

rows = []
with Path(sys.argv[1]).open(newline="") as f:
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
print(f"{rows[0][1]} {rows[0][0]:.8f}")
PY
)"
BEST="${BEST_INFO% *}"
BEST_BPB="${BEST_INFO##* }"
echo "[h100-preflight] best_proxy candidate=$BEST bpb=$BEST_BPB" | tee -a "$LOG"

if [ -n "$FINAL_MAX_PROXY_BPB" ]; then
  if ! python - "$BEST_BPB" "$FINAL_MAX_PROXY_BPB" <<'PY'
import sys
best = float(sys.argv[1])
threshold = float(sys.argv[2])
raise SystemExit(0 if best <= threshold else 1)
PY
  then
    echo "[h100-preflight] skip final: best proxy $BEST_BPB > threshold $FINAL_MAX_PROXY_BPB" | tee -a "$LOG"
    exit 0
  fi
fi

echo "[h100-preflight] top final launch $(date -Is) candidate=$BEST seconds=$FINAL_SECONDS out=$FINAL_OUT" | tee -a "$LOG"
env H100_1X_TRAIN_BATCH_TOKENS=65536 H100_1X_VAL_BATCH_SIZE=65536 \
  python scripts/run_h100_novel_matrix.py \
    --out "$FINAL_OUT" \
    --candidate-group "$GROUP" \
    --candidates "$BEST" \
    --nproc-per-node 1 \
    --wallclock-seconds "$FINAL_SECONDS" \
    --val-tokens "$FINAL_VAL_TOKENS" \
    --timeout "$FINAL_TIMEOUT" \
    2>&1 | tee -a "$LOG"

echo "[h100-preflight] done $(date -Is)" | tee -a "$LOG"
