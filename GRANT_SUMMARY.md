# Parameter Golf Compute Grant Summary

This repository contains my Parameter Golf experiments, focused on making
sub-4MB and sub-16MB language-model submissions competitive under the
10-minute wall-clock constraint.

## Research Direction

I am exploring two connected ideas:

1. MirrorLoop Recurrent Core, called `HRC` in code. The model has higher
   capacity entry and exit blocks around a tied looped core, for example
   `012|345|345|210`. The goal is to reuse a small number of parameters several
   times while giving the recurrent middle enough loop/depth signal to know
   where it is in the computation.
2. Train-time mixed precision by layer. Instead of training a dense model and
   only quantizing at export, the current CUDA trainer can make selected
   CastedLinear modules use q8/q6/q4 or ternary STE forward views from the
   first step. Outer IO blocks can stay higher precision while the repeated
   middle is trained as ternary or lower precision.

Supporting pieces include factored tied embeddings for 8192-token vocabularies,
Lexical Low-Rank Experts (LexLoRE, still called `VocabMoE` in code), LQER
low-rank residual sidecars for quantization error recovery, lossless CaseOps
tokenization experiments, and local 2060 SUPER wall-clock matrices before
scaling to larger GPU runs.

## What Has Been Tried

- Ported train-time ternary layers into the MirrorLoop / IO-tail model family.
- Added factored tied embeddings to make sub-4MB artifacts realistic with larger
  vocabularies.
- Added LexLoRE token-conditioned low-rank experts at input and loop-entry
  sites for the 16MB family.
- Added mixed q8/q6/q4/ternary export and lzma artifact compression under a
  decimal 4,000,000 byte cap.
- Added LQER residual sidecars and fixed strict final reload validation.
- Added `TRAIN_QUANT_FORWARD=1`, which makes q8/q6/q4/ternary training happen
  in the forward pass from step one rather than only at final export.
- Ran local proxy matrices on an RTX 2060 SUPER to compare shallow wide models,
  looped IO-tail models, LQER variants, and wall-clock speed/quality tradeoffs.

## Current Local Signals

These are local proxy measurements, not official leaderboard submissions.

- Best clean legal sub-4MB row:
  `i3l3r3_d768e256_q884_coret_lqer_r6t12`, `2.5749` final export BPB,
  `148.36 ms/step`, `3,967,875` total bytes.
- Best soft-cap q884 quality row:
  `i3l3r3_d768e256_q884_coret_lqer_r6`, `2.5505` final export BPB,
  but `4,035,469` bytes, about `35KB` over the 4MB goal.
- Per-layer precision ladder tested from step one:
  q16/q8/q4/q2/ternary IO tail rows were legal and fast, with best result
  `2.9888` BPB, but d512/e192 did not have enough capacity.
- Loop index finding:
  it helped high-repeat r9 and i5/l5 rows, but hurt the current q884 r3 legal
  row. It should be treated as a route-specific knob, not a default.
- Sub-16 transfer baseline:
  the local q6 proof row reached `1.7567` final export BPB with a `9.27MB`
  artifact, leaving substantial 16MB headroom.

## Why More Compute Helps

The RTX 2060 SUPER is useful for iteration, but it cannot answer the main
competition question: which candidates are genuinely best under the official
10-minute 8xH100 regime. The current experiments need larger-scale runs to:

- test whether the slower IO-tail/LQER models win once wall-clock and batch size
  are tuned on modern GPUs;
- sweep q8/q6/q4/ternary schedules, loop counts, embedding ranks, and LQER ranks
  without waiting hours per local matrix;
- validate sub-16MB variants using the same speed levers;
- run fair candidate promotion instead of relying on tiny local proxy splits.

The development grant tier is the right fit: the approach is concrete and
implemented locally, but needs more GPU time to turn promising proxy signals
into official-quality submissions.
