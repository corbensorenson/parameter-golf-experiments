# Parameter Golf Compute Grant Summary

This repository contains my Parameter Golf experiments, focused on making
sub-4MB and sub-16MB language-model submissions competitive under the
10-minute wall-clock constraint.

## Research Direction

I am exploring two connected ideas:

1. A mirrored IO-tail / recurrent-middle architecture. The model has higher
   capacity entry and exit blocks around a tied looped core, for example
   `012|345|210`. The goal is to reuse a small number of parameters several
   times while giving the recurrent middle enough loop/depth signal to know
   where it is in the computation.
2. Train-time mixed precision by layer. Instead of training a dense model and
   only quantizing at export, the current CUDA trainer can make selected
   CastedLinear modules use q8/q6/q4 or ternary STE forward views from the
   first step. Outer IO blocks can stay higher precision while the repeated
   middle is trained as ternary or lower precision.

Supporting pieces include factored tied embeddings for 8192-token vocabularies,
LQER low-rank residual sidecars for quantization error recovery, lossless
CaseOps tokenization experiments, and local 2060 SUPER wall-clock matrices
before scaling to larger GPU runs.

## What Has Been Tried

- Ported train-time ternary layers into HRC / IO-tail model families.
- Added factored tied embeddings to make sub-4MB artifacts realistic with larger
  vocabularies.
- Added mixed q8/q6/q4/ternary export and lzma artifact compression under a
  decimal 4,000,000 byte cap.
- Added LQER residual sidecars and fixed strict final reload validation.
- Added `TRAIN_QUANT_FORWARD=1`, which makes q8/q6/q4/ternary training happen
  in the forward pass from step one rather than only at final export.
- Ran local proxy matrices on an RTX 2060 SUPER to compare shallow wide models,
  looped IO-tail models, LQER variants, and wall-clock speed/quality tradeoffs.

## Current Local Signals

These are local proxy measurements, not official leaderboard submissions.

- Best clean shallow guarded sub-4MB proxy row:
  `i1l2r2_d768/e256 cooltaper5k_cold_tokens8k`, about `2.7573` export BPB,
  `70.6 ms/step`, `2.86MB` artifact.
- Best fixed IO-tail LQER 3k proxy row:
  `i3l3r3_d768e256_q864_coret_lqer`, about `2.7550` export BPB,
  `162.5 ms/step`, `3.47MB` artifact, strict final reload clean.
- Active next matrix:
  six 10-minute wall-clock sub-4MB candidates using `TRAIN_QUANT_FORWARD=1`,
  mixed q8/q6/q4 IO tails, ternary recurrent cores, LQER, final artifacts, and
  the decimal 4MB cap.

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

