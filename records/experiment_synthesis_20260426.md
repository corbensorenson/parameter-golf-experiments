# Experiment Synthesis

Date: 2026-04-26

This is the short read of what we have explored so far. Raw run directories
remain the source of truth for exact logs, but this file records the current
engineering interpretation.

## Current State

- No local training/matrix process was active during this audit.
- The strongest clean sub-4MB fixed-step local candidate is now
  `i5l5r9_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`: final export
  `2.5608` BPB, `188.33ms/step`, `3,882,787` bytes, and `117,213` bytes of
  decimal 4MB headroom at 5k steps.
- The best 10-minute local wall-clock row remains
  `i3l3r3_d768e256_q884_coret_lqer_r6t12`: final export `2.5749` BPB,
  `148.36ms/step`, `4046` wall-clock-stop steps, `3,967,875` bytes, and
  `32,125` bytes of decimal 4MB headroom.
- The best soft-target sub-4 quality reference is now
  `i4l9r5_d512e192_q16q8q4t_coret_lqer_lidx_r6t12`: final export `2.5009`
  BPB, `170.71ms/step`, `4,111,251` bytes, `111,251` bytes over the decimal
  4MB target. This beats `i5l9r5` by `0.0305` BPB while also being faster.
- The best quality q884 row is slightly over the 4MB target:
  `i3l3r3_d768e256_q884_coret_lqer_r6` reached `2.5505` BPB at
  `4,035,469` bytes, `35,469` bytes over cap.
- The q16/q8/q4/q2/ternary IO-ladder family is no longer just a small speed
  lane: `i5l5r9` is the clean legal quality leader at fixed 5k steps, while
  `i4l9r5` is the strongest soft-target quality row.
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
  `SUBMISSION_SIZE_CAP_BYTES=4000000` and prefer `MODEL_CODEC=lzma`. For this
  project, sub-4MB is a research target rather than the competition hard cap;
  slightly-over rows remain useful quality references.
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
   The q884 r3 row is much better than q884 r9, even though r9 reuses
   parameters more aggressively. However, the i5 precision-ladder family shows
   that deep virtual routes can work when the IO ladder and loop-index signal
   match the route.

3. Loop index is conditional, not universally good.
   It hurt the q884 r3 legal row, helped every i5/l5 row, and helped r9. The
   signal is real, but it is only worth paying for when the loop repeats enough
   or when the route is otherwise ambiguous.

4. The precision-ladder IO family is now a promoted sub-4 direction.
   q16/q8/q4/q2/ternary from step one works. At equal 55 virtual layers and 5k
   steps, `i5l9r5` beat `i5l5r9` by `0.0294` final BPB, but `i4l9r5` then
   beat `i5l9r5` by `0.0305` BPB while running faster. `i3l9r5` collapsed to
   `3.1201` BPB, so the q16/q8/t ladder is too abrupt; q16/q8/q4/t looks like
   the current sweet spot. Pushing `i5l9` from r5 to r9 worsened quality to
   `2.5731` BPB and slowed to `269.01ms/step`, so more repeats are not
   automatically better.

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
- 2026-04-27 update: VocabMoE is the strongest clean local 16MB lane. The best
  completed row is `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst`
  at `1.8710` final BPB, `832.11ms/step`, and `6,218,621` bytes. Council and
  RLM-lite variants did not beat it after export. The next selective scout is
  `cap16_speed`, which applies the sub-4 speed profile, QK 5.25, LQER r12/t24,
  and d768 width/embedding/unique-loop spends under the official 16MB cap.
- 2026-04-27 setup update: two next-stage groups are now runnable. `cap16_mainline`
  spends the cap on d768/d896, e384, VocabMoE input+loop-first, QK 5.25, LQER
  r16/t32 on blocks plus factored embedding projections, and q6/q4 taper probes.
  `cap16_dual_stream` adds a trained low-rank left/right advisor bridge at input,
  loop-entry, and pre-output sites so the dual-brain idea is trained end-to-end
  rather than added as eval-only council.
- 2026-04-27 queue update: `cap16_leaderboard` now ports public-leaderboard
  levers onto the HRC/VocabMoE spine at 5k steps: Polar/MIN_LR, QK 5.5, sparse
  attention gate, parallel residuals, moderate Huber Muon WD, BigramHash, and a
  local legal score-first TTT canary.
- 2026-04-27 prior-work pass: the closest established relatives are Subformer
  sandwich sharing, Universal-Transformer recurrence, ALBERT factorized
  embeddings, mixed-precision QAT/GPTQ, and sparse MoE/memory layers. The queue
  now also tests the main missing low-cost ideas: cycle-rev routes, loop-index
  recurrence signals, and rank-4 per-depth Q/V LoRA.

## Tokenizer Lessons

- Custom tokenizers are allowed, but any tokenizer/data change must prove exact
  byte accounting and exact reconstruction.
- A naive whole-word tokenizer is not the first move for sub-4MB because it
  makes the embedding/output interface expensive and hurts rare words.
- The safer path is lossless CaseOps-v2 plus word-boundary-aware BPE/Unigram
  sweeps at 4096/6144/8192 vocab sizes, with exact validation byte sidecars.

## Current Promotion Order

1. Fixed-step legal sub-4 default:
   `i5l5r9_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`.
2. Local 10-minute wall-clock default:
   `i3l3r3_d768e256_q884_coret_lqer_r6t12`.
3. Soft-target quality reference:
   `i4l9r5_d512e192_q16q8q4t_coret_lqer_lidx_r6t12`.
4. If we spend more local time: keep `i4l9r5` as the quality reference even if
   slightly over 4MB, then run a 10-minute wall-clock comparison against q884
   r3 and i5l5r9.
5. If we get H100 time: run d1536/e384 and d2048/e512 CaseOps/HRC capacity
   ladders with the same export-honest train-time quant policy.
6. For sub-16: resume q6 proof with LQER/frozen-carry/publicstack and then test
   speed levers only if the loss curve tracks the conservative baseline.

## Fixed-Step Follow-Ups

Started/completed 2026-04-26:
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

Results:

- `i5l5r9_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`: final export
  `2.5608` BPB, `188.33ms/step`, `3,882,787` bytes.
- `i5l5r2_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`: final export
  `3.2888` BPB, `126.33ms/step`, `3,871,503` bytes.
- `i3l3r9_d768e256_q884_coret_lqer_lidx_r6t12`: final export `3.6462` BPB,
  `224.62ms/step`, `3,738,987` bytes.

Follow-up:
`records/sub4-i5l9r5-fixed5k-20260426-015555`.

Purpose: compare `i5l5r9` against `i5l9r5` at the same 55 virtual layers and
5k train-step budget. The new row has more unique loop blocks but fewer loop
repeats.

Result:

- `i5l9r5_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`: final export `2.5314`
  BPB, `201.89ms/step`, `4,105,939` bytes, `105,939` bytes over the decimal
  4MB target.

Follow-up:
`records/sub4-i5l9r9-fixed5k-20260426-021633`.

Purpose: test whether the same `i5l9` physical shape improves with more loop
repeats. This raises effective depth from 55 to 91 without changing the unique
block count.

Result:

- `i5l9r9_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`: final export `2.5731`
  BPB, `269.01ms/step`, `4,083,767` bytes, `83,767` bytes over the decimal
  4MB target. This is worse and slower than r5, so r5 is the better tested
  repeat count for the i5/l9 physical shape.

Follow-up:
`records/sub4-i3i4l9r5-fixed5k-20260426-024640`.

Purpose: compare shorter IO tails against `i5l9r5` while keeping loop width 9,
route repeats 5, d512/e192, loop index, LQER r6t12, and train-time quantized
forward. The ladders were `i3: q16/q8/t` and `i4: q16/q8/q4/t`.

Results:

- `i4l9r5_d512e192_q16q8q4t_coret_lqer_lidx_r6t12`: final export `2.5009`
  BPB, `170.71ms/step`, `4,111,251` bytes, `111,251` bytes over the decimal
  4MB target.
- `i3l9r5_d512e192_q16q8t_coret_lqer_lidx_r6t12`: final export `3.1201` BPB,
  `164.37ms/step`, `4,068,247` bytes, `68,247` bytes over the decimal 4MB
  target.
