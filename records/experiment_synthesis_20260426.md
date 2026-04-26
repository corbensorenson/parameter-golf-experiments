# Experiment Synthesis

Date: 2026-04-26

This is the short read of what we have explored so far. Raw run directories
remain the source of truth for exact logs, but this file records the current
engineering interpretation.

## Current State

- No local training/matrix process was active during this audit.
- The strongest clean sub-4MB local candidate is
  `i3l3r3_d768e256_q884_coret_lqer_r6t12`: final export `2.5749` BPB,
  `148.36ms/step`, `4046` wall-clock-stop steps, `3,967,875` bytes, and
  `32,125` bytes of decimal 4MB headroom.
- The best quality q884 row is slightly over the soft 4MB goal:
  `i3l3r3_d768e256_q884_coret_lqer_r6` reached `2.5505` BPB at
  `4,035,469` bytes, `35,469` bytes over cap.
- The i5/l5 q16/q8/q4/q2/ternary ladder is legal and fast, but too small:
  best row `i5l5r2_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12` reached
  `2.9888` BPB at `124.29ms/step`.
- The r9 loop-index test proved the loop index can help under high recurrence:
  r9 with loop index beat no-loop-index by `0.0455` BPB. But r9 was too slow
  locally, around `211-215ms/step`, and finished far behind q884 r3.
- The current local sub-16 q6 proof baseline completed with final export
  `1.7567` BPB, `679.52ms/step`, and `9,268,177` total bytes under the
  16MB cap.

## What We Built And Verified

- Train-time quantized forward path: `TRAIN_QUANT_FORWARD=1` makes selected
  linear layers use q8/q6/q4/q2/ternary/fp16 views from the first forward pass,
  rather than training dense and only quantizing at the end.
- HRC IO-tail routing: mirrored entry/exit tails with a tied looped middle,
  including `transition_recursive_cycle` routes such as `012|345|210` and
  deeper `012|(345)x9|210`.
- Factored tied embeddings: needed for SP8192/CaseOps experiments under the
  sub-4MB byte goal.
- LQER residual sidecars: asymmetric low-rank residuals now reload correctly,
  including TernaryLinear sidecars instead of being folded into weights that
  would be re-ternarized.
- Decimal cap and codec discipline: sub-4 runs use
  `SUBMISSION_SIZE_CAP_BYTES=4000000` and prefer `MODEL_CODEC=lzma`.
- Precision policy expansion: `train_gpt.py` now supports q2 and q16/fp16
  passthrough in the train-time/export policy.
- Local speed hygiene: fused QKV, fp16 Muon state where stable, no grad scaler
  in low-precision lanes, persistent/pinned loader buffers, idle-GPU guarded
  matrices, and no validation inside speed-critical sections unless requested.
- CUDA extension path: CUDA 12.6-compatible local environment checks were added
  after finding that PyTorch `2.11.0+cu126` and `nvcc` 11.7 were a bad build
  pairing on the Windows machine.
- Ternary kernel experiments: packed ternary helpers and benchmarks exist, but
  the practical next kernel direction is still a fused dense materialization
  path that feeds Tensor Cores rather than replacing mature GEMM kernels.

## Sub-4MB Lessons

1. Export-only quantization is misleading.
   Several longer fast-ternary runs looked acceptable before export, then had
   catastrophic final round-trip BPB. Train-time quantized forward is now the
   serious lane.

2. Width and byte spending beat extreme recurrence locally.
   The q884 r3 row is much better than r9, even though r9 reuses parameters
   more aggressively. More tied depth reduces step count too much on the 2060.

3. Loop index is conditional, not universally good.
   It hurt the q884 r3 legal row, helped every i5/l5 row, and helped r9. The
   signal is real, but it is only worth paying for when the loop repeats enough
   or when the route is otherwise ambiguous.

4. The i5/l5 precision ladder is correctly implemented but under-capacity.
   q16/q8/q4/q2/ternary from step one works, and r2+lidx was best, but d512/e192
   cannot match d768/e256 q884 quality.

5. LQER rank/top-K are real byte-quality knobs; factor bits were not.
   In the asymmetric LQER path, `LQER_FACTOR_BITS` does not save bytes. Rank and
   top-K are the knobs that mattered in the q884 byte-shave sweeps.

6. The old nano/micro family taught speed and schedule lessons, but it is no
   longer the quality lead.
   d96/d384 shapes were useful for smoke tests and schedule rescue, but the
   promoted sub-4 direction is wide shallow HRC plus q884 IO-tail/core policy.

## Sub-16MB Lessons

- The sub-16 lane should stay closer to the public HRC/q6 proof recipe:
  CaseOps/SP8192, q6 export proof, LQER, frozen carry, public-style gates, and
  eventual legal TTT.
- The completed local q6 proof row is well under the 16MB cap and much better
  than sub-4, but the local 2060 runtime is not representative of 8xH100.
- The next sub-16 cleanup is to resume the corrected frozen-carry/LQER ladder
  rather than trusting the suite that stopped after the baseline row.

## Tokenizer Lessons

- Custom tokenizers are allowed, but any tokenizer/data change must prove exact
  byte accounting and exact reconstruction.
- A naive whole-word tokenizer is not the first move for sub-4MB because it
  makes the embedding/output interface expensive and hurts rare words.
- The safer path is lossless CaseOps-v2 plus word-boundary-aware BPE/Unigram
  sweeps at 4096/6144/8192 vocab sizes, with exact validation byte sidecars.

## Current Promotion Order

1. Legal sub-4 default: `i3l3r3_d768e256_q884_coret_lqer_r6t12`.
2. Soft-cap quality reference: `i3l3r3_d768e256_q884_coret_lqer_r6`.
3. If we spend more local time: widen or improve q884 r3 before adding more
   repeats.
4. If we get H100 time: run d1536/e384 and d2048/e512 CaseOps/HRC capacity
   ladders with the same export-honest train-time quant policy.
5. For sub-16: resume q6 proof with LQER/frozen-carry/publicstack and then test
   speed levers only if the loss curve tracks the conservative baseline.

## Active Fixed-Step Follow-Up

Started 2026-04-26:
`records/sub4-lidx-fixed5k-depthcompare-20260426-010314`.

Purpose: compare high-recurrence loop-index rows at the same 5k train-step
budget instead of the same 10-minute wall-clock budget.

Rows:

- `i3l3r9_d768e256_q884_coret_lqer_lidx_r6t12`
- `i5l5r2_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`
- `i5l5r9_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`

Settings: `ITERATIONS=5000`, `MAX_WALLCLOCK_SECONDS=0`,
`WARMDOWN_ITERS=5000`, `TRAIN_QUANT_FORWARD=1`, final artifacts,
`--allow-over-cap`, idle GPU guard, and loop index enabled on all rows.
