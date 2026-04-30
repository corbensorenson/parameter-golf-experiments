#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

WAIT_PID="${1:-}"
GROUP="${GROUP:-h100_1x_loss_next}"
CAP_BYTES="${CAP_BYTES:-16000000}"
PREFLIGHT_STEPS="${PREFLIGHT_STEPS:-1}"
PREFLIGHT_VAL_TOKENS="${PREFLIGHT_VAL_TOKENS:-8192}"
PREFLIGHT_TIMEOUT="${PREFLIGHT_TIMEOUT:-600}"
FINAL_SECONDS="${FINAL_SECONDS:-600}"
FINAL_TIMEOUT="${FINAL_TIMEOUT:-1500}"
FINAL_VAL_TOKENS="${FINAL_VAL_TOKENS:-131072}"

PRE_OUT="${PRE_OUT:-records/${GROUP}_size_preflight_live}"
OUT="${OUT:-records/${GROUP}_full10_live}"
LOG="${LOG:-/workspace/${GROUP}_preflight_full.out}"

is_wait_pid_alive() {
  [ -n "$WAIT_PID" ] && [ -r "/proc/${WAIT_PID}/cmdline" ]
}

echo "[h100-full] group=$GROUP cap=$CAP_BYTES queue started $(date -Is), waiting on pid ${WAIT_PID:-none}"
while is_wait_pid_alive; do
  echo "[h100-full] waiting $(date -Is)"
  sleep 20
done

rm -rf "$PRE_OUT" "$OUT"
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
  echo "[h100-full] preflight launch $(date -Is) candidate=$candidate" | tee -a "$LOG"
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
  echo "[h100-full] $candidate $decision" | tee -a "$LOG"
  if [[ "$decision" == accept* ]]; then
    ACCEPTED+=("$candidate")
  fi
done

if [ "${#ACCEPTED[@]}" -eq 0 ]; then
  echo "[h100-full] no candidates passed size preflight; exiting" | tee -a "$LOG"
  exit 2
fi

ACCEPTED_CSV="$(IFS=,; echo "${ACCEPTED[*]}")"
echo "[h100-full] accepted=$ACCEPTED_CSV" | tee -a "$LOG"
echo "[h100-full] full matrix launch $(date -Is) seconds=$FINAL_SECONDS out=$OUT" | tee -a "$LOG"

env H100_1X_TRAIN_BATCH_TOKENS=65536 H100_1X_VAL_BATCH_SIZE=65536 \
  python scripts/run_h100_novel_matrix.py \
    --out "$OUT" \
    --candidate-group "$GROUP" \
    --candidates "$ACCEPTED_CSV" \
    --nproc-per-node 1 \
    --wallclock-seconds "$FINAL_SECONDS" \
    --val-tokens "$FINAL_VAL_TOKENS" \
    --timeout "$FINAL_TIMEOUT" \
    2>&1 | tee -a "$LOG"

echo "[h100-full] done $(date -Is)" | tee -a "$LOG"
