# Experiment Synthesis

Date: 2026-04-26

This is the short read of what we have explored so far. Raw run directories
remain the source of truth for exact logs, but this file records the current
engineering interpretation.

## Current State

- 2026-04-29 late update: local training has pivoted to the non-record/art
  lane. Active matrix:
  `records/cap16-art-showcase-2k-auto-20260429-173955`, six 2k smoke rows
  testing prime skip recurrence, mirrored IO-tail/width ladders, spike
  VocabMoE, RLM-lite memory, council distributions, and trained dual streams.
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
- 2026-04-29 update: the current best local 16MB row is now
  `frontier_polarminlr10_i3l5r5_d640e512_q8`: final export `1.53362322` BPB,
  `1560.22ms/step`, and `14,091,166` bytes. The winning shift was not simply
  spending more bytes; it was q8 train/export, e512 factored embeddings, more
  unique loop blocks with r5 recurrence, and Polar Express Muon with a 10%
  minimum LR floor.

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
- 2026-04-27 overnight close-out: the d640/e256 input+loop-first hybrid
  VocabMoE row remains the local 16MB frontier at `1.87104756` exported BPB,
  `832.11ms/step`, and `6,218,621` bytes. The best new promoted row,
  `mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32`, reached
  `1.88161553` BPB at `1085.46ms/step` and `8,289,170` bytes: close, but
  slower, larger, and still worse. The corrected spikehybrid row reached
  `1.87915642` BPB, which makes spiking/self-election a narrow near-miss branch
  rather than the mainline. Sparsegate, depth-LoRA, cycle-rev, d768/d896 width,
  and e384 embeddings did not transfer into a win. The aggressive fp16-param
  speed profile was rejected because it collapsed exported BPB to about `4.15`.
  See `records/overnight_synthesis_20260427.md`.
- 2026-04-29 structure close-out: the q8/e512 i3/l5/r5 spine is the 16MB
  control to beat. Dual-stream on that spine was a strong near miss
  (`1.54399822` BPB) but did not beat the single-stream baseline. Hourglass
  block widths were faster and smaller (`1423.07ms/step`, `13,102,734` bytes)
  but worse in quality (`1.55335515` BPB). The combined row
  q16/q8/q4 IO ladder + q8 core + hourglass + dual-stream reached a good
  train-time validation (`1.5411` BPB) but exported poorly (`1.55647474` BPB),
  exposing a train/export gap. Do not promote dual-stream, hourglass adapters,
  or the IO-tail q4 combo until a specific export-gap fix exists.
- 2026-04-29 frontier cap-fill setup: launched
  `records/cap16-frontier-capfill-5k-auto-20260429-003802` with four focused
  rows on the current best i3/l5/r5 q8/e512 spine. The rows test e640
  embedding spend, stronger LQER including `embed_proj`, Polar/MIN_LR schedule
  polish, and QK 5.25 plus a wider parallel-residual tail. This replaces broad
  cap-fill sweeps with the smallest set of public-leaderboard transfers that
  still addresses the unused `2.18MB` headroom.
- 2026-04-29 cap-fill result: the first frontier row,
  `frontier_capfill_i3l5r5_d640e640_q8`, is the new local 16MB best at
  `1.53679911` final export BPB, `1604.10ms/step`, and `14,574,862` bytes.
  It beats the e512 control by about `0.00469` BPB and confirms that some
  remaining cap should go into token-interface rank on the i3/l5/r5 q8 spine.
- 2026-04-29 LQER repair result: `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`
  landed at `1.54073632` final export BPB, `1576.04ms/step`, and `13,956,518`
  bytes. That is a tiny improvement over the old e512 control, but still worse
  than the e640 token-interface row, so LQER is a secondary repair lever here.
- 2026-04-29 completed cap-fill result: `frontier_polarminlr10_i3l5r5_d640e512_q8`
  is the new local 16MB best at `1.53362322` final export BPB,
  `1560.22ms/step`, and `14,091,166` bytes. QK 5.25 plus parres4 was negative
  at `1.54894140` BPB. The next useful branch is e640/e768 plus Polar/MIN_LR,
  not more QK/parallel-residual routing.
- 2026-04-30 H100 precision/width update: the active one-H100 architecture
  scout showed e1536/e1856 are close, second core-attention is worse, and
  i3/l7/r2 is worse in the 360-second proxy. A focused follow-up group,
  `h100_1x_precision_width_scout`, is queued behind the current job. It tests
  only six rows on the proven i3/l5/r5 spine: q8 e512, q8 e640, q8 d704,
  hourglass width ladder, q16/q8/q8 IO-tail ladder with q8 core, and
  q16/q8/q4 IO-tail ladder with q4 core. The earlier q16/q8/q4 IO + q8 core
  idea was removed before launch because the core should not exceed the
  narrowest precision in the IO ladder.
- 2026-04-30 train-time precision audit: promoted rows train at their assigned
  precision from step zero. `TRAIN_QUANT_FORWARD=1` configures quantized STE
  forward views before the training loop, `TRAIN_QUANT_EMBEDDINGS=1` applies
  the same idea to embedding lookup, and VocabMoE weights use
  `VOCAB_MOE_TRAIN_QUANT_BITS`. `QUANT_TRAIN_MODE=none` is not a dense-training
  fallback here; it just avoids the old QAT mode while the step-zero quantized
  forward path is active.
- 2026-04-30 cap-preflight policy: after the d768/e1536 H100 row exported at
  `17,866,783` bytes and e1728 exported at `18,787,023` bytes, new near-cap H100
  groups now run a one-step artifact export before proxy training. Rows above
  the decimal `16,000,000` byte cap are rejected before the 6-10 minute run.
  The first group using this gate is `h100_1x_capfit_scout`, testing
  d768/e1024, e1088, e1152, and e1216 on the one-attention q8/QK5.5 spine.
- 2026-04-29 automation setup: hourly automation
  `parameter-golf-frontier-watchdog` now runs
  `scripts/hourly_frontier_watchdog.ps1`, appends status to
  `records/hourly-watchdog.md`, updates docs when completed rows change
  conclusions, and launches the selective `cap16_frontier_followup` group once
  the current four-row queue finishes.

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
6. For the 16MB lane: use `frontier_capfill_i3l5r5_d640e640_q8` as the current
   quality control. It beats the older q6/e256 VocabMoE anchor by a large
   margin, although it is much slower locally.
7. Do not promote the latest structure experiments. Dual-stream was a near
   miss, hourglass was smaller/faster but worse, and the combined IO ladder +
   hourglass + dual-stream row widened the export gap.
8. If we run more local 16MB probes, keep them narrow: one export-gap repair
   around the q8/e512 i3/l5/r5 spine, one schedule/optimizer tweak, or one
   matched unique-loop variant. Avoid broad d768/d896 sweeps and avoid fp16
   trainable params/no GradScaler for quality rows.
9. Active next probe: `records/cap16-frontier-followup-5k-auto-20260429-164230`.
   Promote by final export BPB against the new `1.53362322` Polar/MIN_LR
   control, not by train-time validation.
10. The follow-up group is now focused on the completed signal: e640+Polar,
    e768+Polar, e640+Polar+LQER, and e640+Polar with a 5% LR floor. Keep it at
    5k steps and do not broaden unless one of these rows improves final export
    BPB.

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
