#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CURRENT_PID="${1:-}"

is_current_matrix() {
  if [ -z "$CURRENT_PID" ] || [ ! -r "/proc/${CURRENT_PID}/cmdline" ]; then
    return 1
  fi
  tr '\0' ' ' <"/proc/${CURRENT_PID}/cmdline" \
    | grep -Fq -- "run_h100_novel_matrix.py --out records/h100_hrc_nearcap_round_live"
}

echo "[sub4] queue started $(date -Is), waiting on matrix pid ${CURRENT_PID:-none}"
while is_current_matrix; do
  echo "[sub4] waiting for h100_hrc_nearcap_round_live $(date -Is)"
  sleep 30
done

COMMON_ARGS=(
  --iterations 1000000
  --warmdown-iters 12000
  --candidates i3l3r3_d768e256_q884_coret_lqer_r6t12
  --train-quant-forward
  --train-quant-embeddings
  --allow-over-cap
  --train-seq-len 64
  --train-batch-tokens 32768
  --val-batch-size 65536
  --train-log-every 250
  --run-env TRAIN_DEBUG_NONFINITE=0,TRAIN_ABORT_ON_NONFINITE=0,POST_STEP_ZERO_GRAD=0,SAVE_RAW_MODEL=0,LOG_NVIDIA_SMI=1,DISABLE_COMPILE=1,QK_GAIN_INIT=5.25,LR_MIN_SCALE=0.10,MUON_NS_VARIANT=polar_express
)

echo "[sub4] smoke start $(date -Is)"
python scripts/run_sub4_iotail_quant_matrix.py \
  --out records/h100_sub4_q884_smoke_live \
  --wallclock-seconds 60 \
  --val-tokens 65536 \
  --timeout 600 \
  "${COMMON_ARGS[@]}"

echo "[sub4] 10min start $(date -Is)"
python scripts/run_sub4_iotail_quant_matrix.py \
  --out records/h100_sub4_q884_10min_live \
  --wallclock-seconds 600 \
  --val-tokens 131072 \
  --timeout 1200 \
  --final-artifacts \
  "${COMMON_ARGS[@]}"

echo "[sub4] done $(date -Is)"
