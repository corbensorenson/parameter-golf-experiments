# Project Levers

Date: 2026-04-27

This is the working lever catalog for the Parameter Golf experiments in this
repo. It is meant to answer: "What knobs can we pull, what do they buy us, what
do they cost, and what have we already learned?"

The short current read:

- Best completed 16MB VocabMoE row:
  `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst`, final export
  `1.8710` BPB, `832.11ms/step`, `6,218,621` bytes. The winning placement is
  "input plus first recurrent-loop block", not loop-only or hidden-only.
- Council/RLM-lite revival on that VocabMoE anchor did not beat the dense
  anchor after export. The best completed council/RLM row is hard-gated council
  at `1.8846` BPB; dynamic council, RLM-lite, and RLM+council all exported
  worse.
- Spiking/self-election VocabMoE has not yet produced a valid matrix result.
  The first spike queue only wrote a plan and exited at the old idle threshold.
  The spike candidate definitions now use a small token-prior tie-breaker
  (`VOCAB_MOE_PRIOR_INIT_STD=0.01`) so hard top-k does not send every token to
  the same expert bucket at step zero.
- Best fixed-step clean sub-4MB local row:
  `i5l5r9_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`, final export
  `2.5608` BPB, `188.33ms/step`, `3,882,787` bytes.
- Best local 10-minute wall-clock sub-4MB row:
  `i3l3r3_d768e256_q884_coret_lqer_r6t12`, final export `2.5749` BPB,
  `148.36ms/step`, `3,967,875` bytes.
- Best soft-target sub-4MB quality row:
  `i4l9r5_d512e192_q16q8q4t_coret_lqer_lidx_r6t12`, final export `2.5009`
  BPB, `170.71ms/step`, `4,111,251` bytes, about `111KB` over the decimal 4MB
  target.
- Best i5/l9 soft-target row:
  `i5l9r5_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`, final export
  `2.5314` BPB, `201.89ms/step`, `4,105,939` bytes.
- Latest deeper-repeat i5/l9 row:
  `i5l9r9_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`, final export
  `2.5731` BPB, `269.01ms/step`, `4,083,767` bytes.
- Local sub-16 q6 proof baseline:
  `loopplain5k_i3l3r3_q6proof`, final export `1.7567` BPB,
  `9,268,177` bytes.
- Completed quality-first i4/l9/r5 update:
  `records/sub4-quality-first-i4-5k-20260426-034039`. The best row was
  `i4l9r5_d640e256_q16q8q4t_attncore1_lqer_lidx_r8t16`, final export
  `2.4884` BPB, `267.67ms/step`, `6,050,853` bytes. This made "one full
  attention block at the recurrent-core entry" the current best soft-size
  quality lever.
- Current pruned sub4 follow-up is running under
  `records/sub4-leader-pruned-5k-auto-20260427-012503`. Early completed rows:
  QK 5.25 baseline exported at `2.4792` BPB and the attention-output-gate row
  exported worse at `2.4997` BPB. The public-style stacked row went nonfinite
  and was stopped; do not keep stacking public levers until each one has a
  stable local anchor.
- Next 16MB cap-speed scout:
  `cap16_speed` in `scripts/run_16mb_vocab_moe_matrix.py`. It reruns the best
  dense VocabMoE placement with sub-4 speed levers, QK 5.25, LQER r12/t24, and
  selective d768 width/embedding/unique-loop spends under the official 16MB
  cap.
- New 16MB setup groups:
  `cap16_mainline` spends the cap on d768/d896 width, e384 embeddings, q6/q4
  taper variants, QK 5.25, stronger LQER, and VocabMoE input+loop-first.
  `cap16_dual_stream` adds an opt-in trained left/right advisor bridge on top
  of that spine.
- Next prepared matrix group:
  `sub4_leader_levers` in `scripts/run_sub4_iotail_quant_matrix.py`, covering
  QK gain, scalar SmearGate, attention-output gates, sparse attention gates,
  Huber Muon decay, parallel residuals, frozen recurrent carry, score-first
  TTT, a conservative stacked public-style row, and prime loop-width l11/l13
  probes.
- Newly prepared width-density group:
  `sub4_width_ladder` in `scripts/run_sub4_iotail_quant_matrix.py`, covering
  block-internal width ladders that keep the residual stream at the core width
  while making high-precision IO-tail blocks internally narrower.

## How To Read This

Each lever has:

- Goal: mainly quality, speed, artifact size, legality, or measurement.
- Knobs: env vars, candidate names, scripts, or profile fields.
- Upside: why we would pull it.
- Cost/risk: what it can break or slow down.
- Current read: what the local experiments suggest so far.

Use the current promoted rows as anchors. A lever is only promoted if it wins
under final artifact round-trip, not only train-time loss.

## GPU Time Selection Policy

Goal: spend local 2060 time only on rows that can change the next decision.

Rules:

- Every matrix needs a named anchor and a decision it can change. If a row
  cannot beat, disprove, or calibrate against that anchor, it should not run.
- Prefer small, high-information batches: one anchor if the code path changed,
  two or three lever probes, then stop and read the final export rows.
- Do not rerun controls whose code path and training profile have not changed.
  Reuse the existing final-export row instead.
- Treat final artifact round-trip as the score. Mid-run BPB only decides whether
  to continue or prune a lane.
- Freeze weak lanes until a new mechanism addresses their failure mode. Current
  examples: council/RLM-lite is frozen until the export gap is fixed; broad
  spike sweeps are frozen until the two corrected spike probes show promise.
- Use staged budgets for speculative ideas: first 1-2 corrected probes, then a
  focused follow-up only if they come close to the anchor.
- Stop broad "because it exists" matrices. Candidate count should stay small
  enough that every row has a sentence-level reason to exist.

Current queue policy:

- Keep the active pruned queue because its remaining rows are still high-signal:
  `i4l11r5` softer ladder, two corrected spike probes, and two width-density
  probes. The public-style sub4 stack was pruned after going nonfinite.
- Cancelled the extra focused spike wait-queue that would have added seven more
  rows. It should only be relaunched if the two corrected spike probes beat or
  nearly match the dense VocabMoE anchor.
- Queue the 16MB cap-speed scout after the active pruned queue, not in parallel,
  because the d768 rows need clean GPU telemetry and should not compete with
  local games or other CUDA work.

## 2026-04-26 Leaderboard-Inspired Addendum

Goal: fold current public competition ideas into our HRC/IO-tail search without
copying a whole late-stage transformer stack blindly.

Sources:

- Official README leaderboard:
  <https://github.com/openai/parameter-golf>
- merged PR #1493:
  <https://github.com/openai/parameter-golf/pull/1493>
- open PR #1790:
  <https://github.com/openai/parameter-golf/pull/1790>
- open PR #1791:
  <https://github.com/openai/parameter-golf/pull/1791>
- open PR #1797:
  <https://github.com/openai/parameter-golf/pull/1797>
- open tokenizer-normalization policy issue #1604:
  <https://github.com/openai/parameter-golf/issues/1604>

Current read:

- The accepted leaderboard is still an SP8192 transformer family: recurrence,
  QK gain, parallel residuals, GPTQ/SDClip, and legal score-first TTT.
- Public open PRs suggest the frontier moved further with SmearGate,
  attention-output gating, phased/LoRA TTT, LQER-style quant repair, and larger
  FLA/GDN pivots.
- For our sub-4/soft-8 lane, the safe immediate move is not a full architecture
  rewrite. It is to test cheap control levers on the best local IO-tail shape.
- CaseOps remains useful empirically, but normalization policy is still an
  explicit review surface. Keep exact byte sidecars and treat tokenizer changes
  as audit-heavy.

Implemented for the next local matrix:

- `QK_GAIN_INIT` sweep: `5.0`, `5.25`, `5.5`.
- `SMEAR_GATE_ENABLED=1`, `SMEAR_GATE_MODE=scalar`.
- `ATTN_OUT_GATE_ENABLED=1`, width `24`.
- `SPARSE_ATTN_GATE_ENABLED=1`, width `12`, mutually exclusive with
  attention-output gate.
- `MUON_WEIGHT_DECAY=0.095`, `MUON_WEIGHT_DECAY_MODE=huber`.
- `PARALLEL_RESIDUAL_LAST_N=4` and `8` with the residual mixer enabled.
- `HRC_FROZEN_CARRY_ENABLED=1` on repeated core blocks `4,5,6` so the existing
  3x3 carry coefficients are valid for i4/l9.
- `TTT_SCORE_FIRST_ENABLED=1` control-only legal eval probe at LR `0.005`,
  max `24` updates.
- Prime loop-width expansion:
  `i4l11r5` and `i4l13r5` at d640/e256, with both `q16/q8/q4/t` and the newly
  promising `q16/q8/q8/t` IO ladder. The q8 ladder variants also get QK 5.25
  probes.

Not yet implemented as drop-in local levers:

- Full Hessian-aware SDClip/GPTQ calibration for this ternary export path.
- Progressive recurrence activation during training. We can approximate it with
  route/repeat sweeps today, but a true schedule would need trainer work.
- Full LoRA/phased TTT from the public transformer lane.
- FLA/GatedDeltaNet and byte-level PPM mixture. These should be separate
  branches because they change the predictor class and review profile.

## Layer Width / Data-Density Ladder

Goal: keep information density from collapsing as the precision ladder moves
from q16/q8/q4 into ternary, without forcing every high-precision outer block
to pay the full core width.

Knobs:

- `LAYER_WIDTH_SCHEDULE`, one physical block width per HRC unique block, or one
  per normal layer in baseline mode.
- `MODEL_DIM` remains the residual-stream width and should be set to the
  largest/core width.
- Existing precision knobs still apply by block name:
  `QUANT_BITS_OVERRIDES`, `QUANT_TERNARY_PATTERNS`, and `TRAIN_QUANT_FORWARD`.
- runner group: `sub4_width_ladder` in
  `scripts/run_sub4_iotail_quant_matrix.py`.

Implementation:

- The residual stream, skip path, tied output projection, loop-index controls,
  pass embeddings, and recurrent controls stay at `MODEL_DIM`.
- Blocks whose scheduled width is smaller than `MODEL_DIM` run through a
  down-projection, an internal transformer block at the smaller width, then an
  up-projected residual delta back into the full residual stream.
- This means a q16 outer block can be narrower, while the q4/ternary inner
  blocks can use the full width where each stored bit carries less precision.
- The first version deliberately rejects VE, depth LoRA, and shared-basis XSA
  with non-default widths. Those can be adapted later, but testing the core
  width-density idea first keeps the path auditable.

Why it might help:

- It matches the intuition that effective capacity is roughly width times
  precision, not width alone.
- It may reduce wasted bytes in the precise IO tail while preserving or
  improving low-precision core capacity.
- It preserves HRC recurrence, mirrored IO tails, loop index, and existing
  per-block quantization patterns.

Cost/risk:

- Each narrowed block pays two extra adapter matmuls, so the speed win is not
  guaranteed. The win has to come from cheaper inner attention/MLP and smaller
  quantized exported tensors.
- Adapter projections add parameters; for very mild narrowing, they may eat the
  savings.
- Mirrored HRC blocks mirror the adapter weights too. That is consistent, but it
  is a new behavior and should be compared against the same shape without a
  width ladder.

First queued candidates:

- `i4l9r5_d640e256_q16q8q4t_wl320-480-560-640_attncore1_lqer_lidx_r8t16`
- `i4l9r5_d640e256_q16q8q4t_wl400-480-560-640_attncore1_lqer_lidx_r8t16`
- `i4l9r5_d640e256_q16q8q4t_wl480-560-640_attncore1_lqer_lidx_r8t16`
- `i4l9r5_d640e256_q16q8q8t_wl320-480-560-640_attncore1_lqer_lidx_r8t16`
- `i4l9r5_d768e320_q16q8q4t_wl384-576-672-768_attncore1_lqer_lidx_r8t16`
- `i4l11r5_d640e256_q16q8q4t_wl320-480-560-640_attncore1_lqer_lidx_r8t16`
- `i5l5r5_d512e192_q16q8q4q2t_wl256-320-384-448-512_attncore1_lqer_lidx_r6t12`

Current read:

- Implemented and smoke-tested on CPU forward/backward.
- Not yet locally scored. The key comparison will be against the non-ladder
  `attncore1` row at `2.4884` BPB.

## 16MB Vocabulary-MoE Lever

Goal: spend some of the official 16MB budget on token-conditioned computation
without paying for one literal micro-network per token.

Knobs:

- `VOCAB_MOE_ENABLED=1`
- `VOCAB_MOE_EXPERTS`, usually `16` or `32`
- `VOCAB_MOE_RANK`, usually `1` or `2` locally
- `VOCAB_MOE_MODE=static|hybrid|hidden`
- `VOCAB_MOE_LAYERS=input|loop_first|loop_every3|...`
- `VOCAB_MOE_PRIOR_INIT_STD`, usually `0.0` for dense routing
- `VOCAB_MOE_TRAIN_QUANT_BITS=6`
- `VOCAB_MOE_SITE_BIAS_ENABLED=1`
- `VOCAB_MOE_SITE_SCALE_ENABLED=1`
- `QUANT_FORCE_PATTERNS=vocab_moe.token_prior.weight,vocab_moe.down,vocab_moe.up`
- runner: `scripts/run_16mb_vocab_moe_matrix.py`

Design:

- Each token gets a tiny learned router prior (`vocab_size x experts`).
- The actual experts are shared low-rank bases (`experts x dim x rank` and
  `experts x rank x dim`), mixed with a softmax router.
- `static` is token-prior only. `hybrid` adds a hidden-state router. `hidden`
  is a pure hidden-state router control.
- The adapter can run at the embedding, selected virtual layers, or HRC loop
  aliases such as `loop_first` and `loop_every3`.
- When multiple sites are active, small learned site biases/scales let the same
  token-conditioned expert bank behave differently at embedding, loop-entry, and
  repeated-loop positions.

Why this is in the 16MB lane:

- It is too expensive and too speculative for the sub-4MB family right now.
- In the 16MB lane, it is a clean way to buy token-specialized behavior while
  keeping matmuls batched and GPU-friendly.
- The matrix trains the VocabMoE expert weights and token priors with fake q6
  from the start (`VOCAB_MOE_TRAIN_QUANT_BITS=6`) and force-quantizes those
  same tensors at export, so the score is the exported model rather than a
  full-precision training-only proxy.

Queued local probes:

- `i3l3r3_d640e256_q6_publicstack_control`
- `i3l3r3_d640e256_q6_vocabmoe_static_k16r2_input`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopevery3`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopall`
- `i3l3r3_d640e256_q6_vocabmoe_hidden_k16r2_loopfirst`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k32r1_loopfirst`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k8r4_loopfirst`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r4_loopfirst`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k32r2_loopfirst`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst_t07`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst_t15`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst_nosite`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopevery3`
- `i4l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst`
- `i3l5r2_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst`
- `i4l5r2_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst`
- `i3l3r3_d768e256_q6_vocabmoe_hybrid_k16r2_loopfirst`
- `i3l3r3_d768e320_q6_vocabmoe_hybrid_k16r2_loopfirst`

Current read:

- The first queued run used the public-style q6 stack too hot for
  train-quant-forward on the 2060; rows reached about `1.83-1.96` BPB by
  `1k-1.5k` steps and then diverged to `nan`. That run is not a valid quality
  comparison.
- The corrected 5k run cooled the train-quant-forward stack:
  `TIED_EMBED_LR=0.002`, `MATRIX_LR=0.0016`, `SCALAR_LR=0.0016`,
  `WARMUP_STEPS=20`, no frozen carry, no Muon WD, and
  `TRAIN_ABORT_ON_NONFINITE=1`.
- Valid completed rows from `records/vocabmoe16-5k-auto-20260426-171524`:
  control `1.9377`, static input `1.9201`, hybrid loop-first `1.9098`, and
  hybrid input+loop-first `1.8710` final export BPB.
- The win is therefore not just the cooled q6 scaffold. Token-conditioned
  shared low-rank experts helped, and the best placement is both before the
  stack and at the first repeated block.
- Dense loop-every3 and loop-all variants looked strong mid-run but crashed
  before final export. Treat those as unstable hints, not results.
- Hidden-only routing exported worse (`1.9319`), so the token prior matters.

## 16MB Cap-Speed Scout

Goal: stop optimizing only around the 4MB research target and start spending
the official 16MB cap on the strongest clean lane we have.

Anchor:

- Best completed dense VocabMoE:
  `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst`,
  `1.8710` BPB, `832.11ms/step`, `6,218,621` bytes.

Applied levers:

- `PARAM_DTYPE=fp16`, `TRAIN_CASTED_LINEAR_PARAM_DTYPE=model`, and
  `MUON_DTYPE=fp16` so the heavy trainable linears are lower precision from the
  start instead of only cast during matmul.
- `USE_GRAD_SCALER=0`, `LOSS_FP32=0`, `POST_STEP_ZERO_GRAD=0`, and
  `TRAIN_FUSED_QKV=1` from the sub-4 speed lane.
- `TRAIN_QUANT_FORWARD=1`, `QUANT_WEIGHT_BITS=6`, and
  `VOCAB_MOE_TRAIN_QUANT_BITS=6` so the q6 path is trained rather than only
  exported.
- `QK_GAIN_INIT=5.25`, because the local sub4 follow-up improved from
  `2.4983` to `2.4792` BPB with that change.
- `LQER_RANK=12`, `LQER_TOP_K=24`, because 16MB has enough room to buy more
  quantization-error repair than the sub-4 rows.

Candidate group:

- runner group: `--candidate-group cap16_speed` in
  `scripts/run_16mb_vocab_moe_matrix.py`.
- queue script: `scripts/queue_16mb_cap_speed_after_current.ps1`.

Rows:

- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24`
- `i3l3r3_d768e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24`
- `i3l3r3_d768e320_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24`
- `i3l5r2_d768e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24`

Decision:

- If the d640 fast anchor beats or closely matches the old d640 anchor while
  stepping faster, promote the speed profile for all 16MB work.
- If d768/e256 wins, spend the cap on residual width first.
- If d768/e320 wins over d768/e256, embeddings are still the bottleneck.
- If i3/l5/r2 wins, unique loop diversity is more valuable than more recurrence
  at this cap.

## 16MB Mainline Cap-Spend

Goal: make the highest-confidence 16MB family runnable without hand-editing
env vars.

Runner group: `--candidate-group cap16_mainline` in
`scripts/run_16mb_vocab_moe_matrix.py`.

Ingredients:

- Lossless CaseOps/SP8192 data and byte sidecars.
- HRC mirrored IO tail plus looped middle.
- Train-time quantized forward, not export-only quantization.
- q6 default weights, with explicit q8/q6/q6 IO-tail and q4 recurrent-core
  taper rows.
- Factored tied embeddings at e320/e384.
- VocabMoE at `input,loop_first`.
- QK gain `5.25`.
- LQER r16/t32 for richer quantization-error repair on block matrices and the
  factored embedding projections.
- The sub-4 speed profile: fp16 params, fp16 Muon, fused QKV, no GradScaler,
  no post-step grad zeroing, and no avoidable fp32 cast path.

Rows:

- `mainline_i3l3r3_d768e384_q6all_vocabmoe_qk525_lqer16t32`
- `mainline_i3l3r3_d896e384_q6all_vocabmoe_qk525_lqer16t32`
- `mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32`
- `mainline_i3l5r2_d768e320_q8q6q6_q4core_vocabmoe_qk525_lqer16t32`
- `mainline_i3l5r2_d896e384_q8q6q6_q4core_vocabmoe_qk525_lqer16t32`

Decision:

- If q6-all width rows win, quality is still width/embedding bound.
- If q4-core taper rows win, the IO-tail precision ladder is transferable to
  the 16MB family.
- If d896 is unstable or too slow locally, keep d768 as the local iteration
  default and reserve d896 for H100s.

## 16MB Dual-Stream Advisor

Goal: test the left/right-brain idea as a trained architecture, not an
eval-only council.

Implementation:

- `DUAL_STREAM_ENABLED=1` creates a low-rank `DualStreamBridge`.
- The residual feature vector is split into a token-facing left lane and a
  recurrent/semantic right lane.
- At selected sites, the bridge sends low-rank messages in both directions:
  left-to-right and right-to-left. The output remains one residual stream and
  one legal normalized distribution.
- Sites are resolved by `DUAL_STREAM_SITES`, currently supporting
  `input`, `loop_first`, `loop_exit`, and `pre_output`.

Knobs:

- `DUAL_STREAM_LEFT_DIM`
- `DUAL_STREAM_RANK`
- `DUAL_STREAM_SITES`
- `DUAL_STREAM_SCALE_INIT`
- `DUAL_STREAM_ACTIVATION`

Runner group: `--candidate-group cap16_dual_stream`.

Rows:

- `dual_i3l3r3_d768e320_left256_q6all_vocabmoe_qk525_lqer12t24`
- `dual_i3l3r3_d768e320_left320_q6all_vocabmoe_qk525_lqer12t24`
- `dual_i3l5r2_d768e320_left256_q6all_vocabmoe_qk525_lqer16t32`
- `dual_i3l5r2_d896e384_left320_q8q6q6_q4core_vocabmoe_qk525_lqer16t32`

Decision:

- Run dual rows only after cap-speed/mainline evidence says the extra matmuls
  are worth the local GPU time.
- Compare dual rows only against matching single-stream mainline rows. If the
  bridge does not beat the same spine, freeze it.
- If left320 beats left256, surface/token precision needs more width.
- If i3/l5/r2 dual wins, the bridge helps most when the loop has more unique
  physical blocks.

## 16MB Spiking / Self-Election Vocab-MoE

Goal: test the user's "each token expert elects itself" idea without launching
one literal Python/CUDA thread or one literal micro-network per vocabulary item.

References:

- Switch Transformer top-1 sparse routing:
  <https://arxiv.org/abs/2101.03961>
- Expert Choice routing:
  <https://arxiv.org/abs/2202.09368>
- Product-key memory as a large sparse token-conditioned memory:
  <https://arxiv.org/abs/1907.05242>

Knobs:

- `VOCAB_MOE_MODE=spike_static|spike_hybrid|spike_hidden`
- `VOCAB_MOE_SPIKE_TOP_K`, usually `1` or `2`
- `VOCAB_MOE_SPIKE_STE=1`
- `VOCAB_MOE_SPIKE_NORMALIZE=1`
- `VOCAB_MOE_PRIOR_INIT_STD=0.01` for spike rows, used as a tiny
  reproducible tie-breaker so hard top-k starts with token-specific expert
  elections instead of a single global tied expert.
- same expert count/rank/layer-site knobs as the dense VocabMoE family.
- runner group: `--candidate-group vocabmoe_spike` in
  `scripts/run_16mb_vocab_moe_matrix.py`.

Design:

- `spike_static` is the pure self-election version: token id -> expert prior ->
  hard top-k expert mask.
- `spike_hybrid` adds the hidden-state router to the token prior before the hard
  mask, so the token gets a vote and the current context gets a vote.
- `spike_hidden` is a router-only sparse control.
- The forward pass uses hard top-k expert selection, but keeps a straight-through
  gradient through the original softmax when `VOCAB_MOE_SPIKE_STE=1`.
- The selected experts are renormalized by default. A non-normalized candidate
  keeps only the selected softmax mass, which makes the adapter damped and may
  be more stable.

Why it might help:

- Dense VocabMoE mixes every expert for every token, which is expressive but
  may blur specialization.
- Hard top-k gates can force token/expert clusters to specialize, closer to the
  "spiking" intuition.
- Top-1/top-2 gates reduce multiply work after a future sparse/fused
  implementation. The first implementation is correctness-first and still
  computes the shared expert bank densely, so the near-term signal is quality,
  stability, and export behavior rather than speed.

Cost/risk:

- Hard routing can starve experts, especially with zero-initialized token
  priors. The STE path is meant to reduce that risk.
- On the current dense batched implementation, top-k does not yet save FLOPs
  because the expert basis matmuls are still dense. A custom gather/fused path is
  a later systems lever if the scores justify it.
- Router logits should stay small and stable; this remains train-quant-forward
  q6 from the start for the VocabMoE tensors.

Queued local probes:

- `i3l3r3_d640e256_q6_vocabmoe_spikestatic_k16r2_input_top1`
- `i3l3r3_d640e256_q6_vocabmoe_spikestatic_k16r2_input_top2`
- `i3l3r3_d640e256_q6_vocabmoe_spikestatic_k32r1_input_top2`
- `i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopfirst_top1`
- `i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopfirst_top2`
- `i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_input_loopfirst_top2`
- `i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopevery3_top2`
- `i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopall_top2`
- `i3l3r3_d640e256_q6_vocabmoe_spikehidden_k16r2_loopfirst_top2`
- `i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k32r1_loopfirst_top2`
- `i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r4_loopfirst_top2`
- `i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopfirst_top2_nonorm`

Current read:

- Implemented as a new optional path, but not yet validly scored.
- The first full spike queue
  `records/vocabmoe16-spike-5k-auto-20260426-195631` did not actually train:
  it wrote `candidate_plan.md`, waited on the old `25%` idle-GPU threshold, and
  exited with code `-1`.
- The original zero-prior spike design was also a poor test: hard top-k with
  all-zero token priors can tie every token into the same expert bucket at step
  zero. The candidate definitions now use `VOCAB_MOE_PRIOR_INIT_STD=0.01` for
  spike rows to make self-election real from the first forward pass.
- The active pruned queue includes two corrected spike probes after the sub4
  leader rows:
  `spikestatic_k16r2_input_top2` and
  `spikehybrid_k16r2_input_loopfirst_top2`.
- A focused post-current queue is prepared in
  `scripts/queue_vocabmoe_spike_focused_after_current.ps1`, but it is not
  currently scheduled. Run it only if the two corrected spike probes in the
  active pruned queue beat or nearly match the dense VocabMoE anchor.

## 16MB Council, Dynamic Depth, And RLM-Lite

Goal: revive the council idea on the current clean VocabMoE lane, and test a
legal recursive-language-model style memory without leaking validation targets.

References:

- Recursive Language Models, for the broad idea of recursively compressing
  context into a persistent state:
  <https://arxiv.org/abs/2512.24601>
- Mixture-of-Depths, for conditional extra compute using a routing decision:
  <https://arxiv.org/abs/2404.02258>
- Self-consistency, translated here as mixing full predictive distributions
  before the realized token is known:
  <https://arxiv.org/abs/2203.11171>
- Parameter Golf legality discussion on strict causal dependence and
  score-before-update:
  <https://github.com/openai/parameter-golf/issues/1017>

Knobs:

- `HRC_COUNCIL_MODE=base_mirror|base_mirror_hybrid`
- `HRC_COUNCIL_TRAIN_MODE=eval_only`
- `HRC_COUNCIL_DEPTH_OFFSETS`, one value per peer
- `HRC_COUNCIL_HARD_GATE=1`
- `HRC_DYNAMIC_COUNCIL_ENABLED=1`
- `HRC_DYNAMIC_COUNCIL_THRESHOLD`
- `HRC_DYNAMIC_COUNCIL_MIN_GATE`
- `RLM_MEMORY_ENABLED=1`
- `RLM_MEMORY_TRAIN_ENABLED=1`
- `RLM_MEMORY_DECAY`
- `RLM_MEMORY_SCALE_INIT`
- `RLM_MEMORY_INJECT=input|loop_first|input_loop_first`
- runner group: `--candidate-group council_rlm` in
  `scripts/run_16mb_vocab_moe_matrix.py`

Implementation:

- Council remains a score-first distribution mixture: base and peer logits are
  projected into full vocabulary distributions, entropy/confidence weights are
  computed without the target, and the mixed logits are scored once.
- Dynamic council runs the base peer first, computes base entropy, and only runs
  the mirror peer for evaluation chunks whose base entropy crosses the chosen
  gate. This is a batch/chunk-level Mixture-of-Depths analogue; it is not a
  target-conditioned vote.
- RLM-lite adds a non-persistent prefix memory buffer initialized to zero at
  validation start. The model injects that memory into the next chunk, scores
  the current chunk, then updates memory from the just-scored hidden states.
  Current-chunk tokens never get memory derived from their own targets.
- The memory buffer is not serialized into the artifact. Only the small learned
  injection scale is saved, so every evaluation replay starts from the same
  deterministic zero memory.
- Validation uses smaller `VAL_BATCH_SIZE=4096` on RLM rows so the memory can
  update more often while staying causal.

Queued local probes:

- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_anchor`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_signperm_o0m2`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_house_o0m2`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_signperm_o00`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_hybrid_o00m1`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_hard_t60`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_dynamic_council_t60`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_dynamic_council_t55`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_rlm_input_d90_s002`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_rlm_loopfirst_d90_s002`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_rlm_inputloop_d95_s001`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_rlm_council_signperm`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopevery3_s002_council_signperm`
- `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopevery3_s002_rlm_council`

Current read:

- Implemented as fully opt-in code paths. Existing running matrices are
  unaffected unless their env explicitly enables these knobs.
- The strongest completed VocabMoE anchor is
  `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst` at final export
  `1.8710` BPB, so the council/RLM matrix was pinned to that row first.
- The `base_mirror_hybrid` candidate uses offsets `0,0,-1`, because the
  implementation correctly requires one depth offset per peer.
- Completed pruned results did not beat the dense anchor: hard-gated council
  `1.8846`, dynamic council `1.9551`, RLM-lite input memory `1.9363`, and
  RLM-lite plus signperm council `2.0098` final export BPB.
- Current conclusion: council and RLM-lite are implemented and legal enough for
  local probes, but they are not promoted. They add complexity and/or eval
  compute without beating the dense VocabMoE anchor yet.

## Measurement And Legality Levers

### Final Artifact Round-Trip

Goal: measurement correctness.

Knobs:

- `--final-artifacts`
- `SKIP_FINAL_ARTIFACTS=0`
- final export log lines: `final_export_roundtrip`, `final_int8_zlib_roundtrip`
- strict reload checks for missing/unexpected tensors

Upside:

- Prevents us from optimizing a model that trains well but exports badly.
- Catches missing LQER sidecars, codec bugs, dtype mistakes, and cap failures.

Cost/risk:

- Adds time at the end of each row.
- Makes local sweeps slower and noisier if we validate too often.

Current read:

- This is mandatory for serious comparisons.
- The early 10k CaseOps run showed train-time BPB around `3.9750` but final
  export BPB around `15.7419`; that entire class of result was misleading.
- Any result without final export BPB is a scout signal only.

### Decimal Size Cap

Goal: artifact legality.

Knobs:

- `SUBMISSION_SIZE_CAP_BYTES=4000000` for the sub-4 lane.
- `SUBMISSION_SIZE_CAP_BYTES=16000000` for official 16MB submission thinking.
- `FAIL_ON_ARTIFACT_CAP=1` for strict runs.
- `--allow-over-cap` for soft-cap research rows.

Upside:

- Keeps byte accounting honest.
- Soft-cap mode lets near-miss rows complete final validation so we can see
  whether they are worth shaving.

Cost/risk:

- The official competition cap is still the real hard cap for submissions.
- In this repo, the sub-4MB lane is a research target. Slightly-over rows are
  still important if they improve quality or identify a better architecture.

Current read:

- `i4l9r5` is the best current soft-target quality reference at `2.5009` BPB,
  about `111KB` over the sub-4 target.
- `i5l5r9` is the best clean fixed-step row at `2.5608` BPB.
- q884 `r6t12` remains the best tested 10-minute local wall-clock row at
  `2.5749` BPB.

### Fixed-Step Versus Wall-Clock Comparison

Goal: fair experiment design.

Knobs:

- `ITERATIONS`
- `MAX_WALLCLOCK_SECONDS`
- `WARMDOWN_ITERS`
- `--wait-for-idle-gpu`
- `--idle-max-util`
- `--idle-max-memory-mib`
- `--idle-seconds`

Upside:

- Fixed-step runs compare optimization quality per step.
- Wall-clock runs compare competition-relevant throughput and score.
- Idle guards reduce contamination from other GPU use.

Cost/risk:

- Fixed-step runs can favor slow models that would lose the 10-minute contest.
- Wall-clock runs can punish models that need fewer but more expensive steps.

Current read:

- r9 recurrence looked worse under wall-clock because it got only about
  `2791-2850` steps in 10 minutes.
- The active fixed-5k follow-up compares the high-repeat loop-index rows at the
  same step budget.

### Exact Byte-Sidecar BPB

Goal: tokenizer/evaluation correctness.

Knobs:

- CaseOps/SP8192 validation byte sidecars.
- `VAL_TOKENS_LIMIT` for local proxy speed.
- exact `val_bpb` calculation from original bytes.
- tokenizer fingerprint/manifest tools in `data/`.

Upside:

- Keeps custom tokenizer experiments legal and auditable.
- Avoids comparing token loss across incompatible tokenizers.

Cost/risk:

- More complicated than simple token loss.
- Tokenizer changes receive extra review burden.

Current read:

- Do not compare old SP1024 proxy loss directly to CaseOps/SP8192 public or
  16MB results.
- CaseOps/SP8192 with byte sidecars is the serious lane.

## Architecture And Capacity Levers

### Width And Body Capacity

Goal: quality.

Knobs:

- `SUB4_PROFILE`
- profile fields: model width `d`, factored embedding rank `e`, number of
  heads, KV heads, MLP multiplier.
- H100 candidate profiles:
  `i1l2r2_d1536_e384_h24kv1_mlpinner_mlp050`,
  `i1l2r2_d2048_e512_h32kv1_mlpinner_mlp025`.

Upside:

- More capacity is the cleanest way to spend unused bytes.
- Wider shallow HRC was much more promising than the old nano family.

Cost/risk:

- Slower per step.
- More VRAM pressure on the 2060.
- Some wide shapes became unstable under sprint learning rates.

Current read:

- d96/d192/d384 taught schedule and speed lessons, but the quality lead moved
  to d768/e256 q884.
- d512/e192 i5/l5 was legal and fast but under-capacity.
- Local 2060 should screen bugs; d1536/e384 and d2048/e512 need H100 testing.

### Factored Tied Embeddings

Goal: size savings, quality under larger vocab.

Knobs:

- `FACTORED_EMBED_DIM`
- tied embedding settings in profiles.
- candidate names with `e128`, `e192`, `e256`, `e384`, `e512`.

Upside:

- Makes SP8192/CaseOps viable under small artifacts.
- Lets us keep richer tokenization without full `8192 x dim` embedding cost.

Cost/risk:

- Too small an embedding rank can bottleneck quality.
- Larger rank spends bytes quickly.

Current read:

- Full embeddings are brutal under sub-4MB.
- d768/e256 is currently the strongest local sub-4 width/rank region.
- d512/e192 was too small for the precision-ladder lane.

### Mirrored IO Tail

Goal: quality per byte.

Knobs:

- `HRC_RECURSIVE_CORE_START`
- `HRC_ROUTE_REPEATS`
- `HRC_DEPTH_SCHEDULE_MODE=transition_recursive_cycle`
- `NUM_UNIQUE_BLOCKS`
- `EFFECTIVE_DEPTH`
- candidate shapes such as `i3l3r3`, `i5l5r2`, `i6l9r3`

Upside:

- Gives separate entry/exit transforms while reusing a small core.
- Lets outer blocks carry higher precision and inner blocks be ternary.

Cost/risk:

- Per unique block, not per virtual occurrence, unless the trainer is changed.
- More virtual depth increases step time.

Current read:

- q884 `i3l3r3` is the current best sub-4 direction.
- Extremely deep recurrence such as r9 helps test loop-index behavior, but
  under wall-clock it loses too many steps locally.

### Repeated Core Depth

Goal: quality via test-time/train-time compute reuse.

Knobs:

- `HRC_ROUTE_REPEATS`
- candidate suffixes like `r1`, `r2`, `r3`, `r9`
- `EFFECTIVE_DEPTH`

Upside:

- Adds effective depth without adding many physical parameters.
- Can improve quality if the tied core learns useful iterative refinement.

Cost/risk:

- Directly slows each step.
- High repeats can hurt if the core lacks position/depth signal.
- More repeats were often worse in local wall-clock runs.

Current read:

- q884 r3 is strong under wall-clock.
- q884 r9 with loop index improved over q884 r9 without loop index, but both
  were far behind q884 r3 under 10-minute local wall-clock.
- i5/l5 r9 is much better than r2 at fixed 5k steps.
- i5/l9 r5 beat i5/l5 r9 at the same 55 virtual layers, but i5/l9 r9 was worse
  and slower than r5.
- i4/l9 r5 beat i5/l9 r5 and is now the best soft-target fixed-step row.

### Loop Index

Goal: quality for repeated cores.

Knobs:

- `HRC_LOOP_INDEX_ENABLED=1`
- `HRC_LOOP_INDEX_DIM`
- `HRC_LOOP_INDEX_SCALE_INIT`

Upside:

- Gives the looped middle a signal for virtual pass position.
- Helps when repeated applications are otherwise ambiguous.

Cost/risk:

- Adds small control parameters and complexity.
- Can hurt if the route does not need it.

Current read:

- Hurt q884 r3 legal row: `2.5749` without loop index versus `2.6935` with
  loop index.
- Helped r9 by about `0.0455` BPB.
- Helped every tested i5/l5 precision-ladder repeat count.
- Treat as route-specific, not a default.

### MLP-Only Repeated Core

Goal: speed.

Knobs:

- `HRC_MLP_ONLY_BLOCKS`
- profile choices that make only outer blocks full attention.

Upside:

- Cuts repeated-core compute.
- Useful for high-repeat HRC routes.

Cost/risk:

- May remove too much capacity from the core.
- If the repeated middle is too weak, more repeats only burn time.

Current read:

- The i5/l5 ladder uses MLP-only blocks `5-9` to keep the repeated core cheap.
- d384 early lanes used attention only on the first block and MLP-only looped
  middle blocks.

### Frozen Carry

Goal: quality for recurrence with low byte cost.

Knobs:

- `HRC_FROZEN_CARRY_ENABLED=1`
- `HRC_FROZEN_CARRY_BLOCKS`
- frozen carry detach controls

Upside:

- Targets repeated middle states directly.
- Costs little artifact size compared with widening.

Cost/risk:

- Wrong block selection can break shape assumptions.
- `HRC_FROZEN_CARRY_BLOCKS=all` pulled in mirrored IO-tail repeats and caused a
  sub-16 suite stop with the smaller core carry matrix.

Current read:

- Implemented and worth retesting.
- Use default repeated-core selector first.

### Recurrent Injection And Pass Roles

Goal: quality and route conditioning.

Knobs:

- recurrent injection settings logged as `hrc_recur_inject_*`.
- pass embeddings and pass-role schedule settings.
- guarded presets that restore pass embeddings, pass roles, loop index, and
  recurrent injection.

Upside:

- Gives HRC blocks more context about their phase and recurrence.
- Stabilized or improved some proxy runs.

Cost/risk:

- More controls can slow or overfit small shapes.
- Guarded d384 improved SP1024 proxy but did not help CaseOps at 1k.

Current read:

- Useful as a guarded quality lane, not a universal default.

## Precision And Quantization Levers

### Train-Time Quantized Forward

Goal: quality honesty and artifact alignment.

Knobs:

- `TRAIN_QUANT_FORWARD=1`
- `QUANT_BITS_OVERRIDES`
- `QUANT_TERNARY_PATTERNS`
- `QUANT_TRAIN_MODE=none`

Upside:

- Trains under the same low-precision views used by export.
- Avoids the train-dense/export-low cliff.
- Keeps the training loop free of export/reload conversions.

Cost/risk:

- Fake-quant paths can be slower or unstable for some bit widths on Windows.
- q6-containing rows showed CUDA illegal-memory failures locally.

Current read:

- This is the serious sub-4 lane.
- Use `--train-quant-forward` without `--roundtrip-guard` for clean wall-clock
  training.

### Periodic Quant Roundtrip Guard

Goal: export-honest training guardrail.

Knobs:

- `QUANT_TRAIN_MODE=roundtrip`
- `QUANT_TRAIN_EVERY`
- `QUANT_TRAIN_START_FRACTION`
- runner flag `--roundtrip-guard`

Upside:

- Forces stored weights through the export codec during training.
- Catches export mismatch early.

Cost/risk:

- Too slow and invasive for serious wall-clock sweeps.
- Combining it with `TRAIN_QUANT_FORWARD=1` contaminated speed conclusions and
  caused deeper IO-tail rows to crash or time out.

Current read:

- Keep it as a debugging guardrail, not a default training mode.

### Mixed Precision By Block

Goal: quality and size balance.

Knobs:

- `QUANT_BITS_OVERRIDES=blocks.0.:8,blocks.1.:8,blocks.2.:4`
- supported bits include q2, q4, q5, q6, q8, and q16/fp16 passthrough.
- `QUANT_TERNARY_PATTERNS=blocks.3.,blocks.4.,...`

Upside:

- Lets IO blocks keep more precision while repeated core blocks go ternary.
- q16/q8/q4/q2/ternary ladders can test smooth precision tapering.

Cost/risk:

- Per unique HRC block, not per virtual occurrence.
- q2 uses fake-quant/export codes, not a packed q2 Tensor Core kernel.
- q6 path was locally unstable in some Windows/CUDA runs.

Current read:

- q884 IO tail with ternary core is still the strongest local 10-minute
  wall-clock family.
- q16/q8/q4/ternary i4/l9/r5 is the strongest fixed-step/soft-target quality
  row so far.
- q16/q8/ternary i3/l9/r5 lost too much IO precision and fell to `3.1201` BPB.

### Ternary Core

Goal: size savings and parameter reuse.

Knobs:

- `QUANT_TERNARY_PATTERNS`
- `TRAIN_TERNARY_BLOCKS`
- `TRAIN_TERNARY_GROUP_SIZE`
- `QUANT_TERNARY_GROUP_SIZE`
- `TRAIN_TERNARY_SCALE_STAT`
- `QUANT_TERNARY_SCALE_STAT`
- `QUANT_TERNARY_SHRINKAGE_FIX=1`

Upside:

- Major artifact compression.
- Lets the repeated middle be cheap in bytes.

Cost/risk:

- Quality loss if too much of the model is ternary.
- Requires train-time low precision to avoid export cliff.

Current read:

- Ternary is real and active; we are not merely quantizing at the end.
- Ternary core plus higher-precision IO tail is better than tiny all-ternary
  nano shapes.

### Ternary Group Size And Scale Statistic

Goal: quality/size tradeoff.

Knobs:

- `TRAIN_TERNARY_GROUP_SIZE`
- `QUANT_TERNARY_GROUP_SIZE`
- `TRAIN_TERNARY_SCALE_STAT`
- `QUANT_TERNARY_SCALE_STAT`

Upside:

- Smaller groups can improve reconstruction quality.
- Scale statistic can affect stability and export quality.

Cost/risk:

- Smaller groups add scale overhead.
- Median/mean choices can interact with shrinkage and optimizer behavior.

Current read:

- Group sizes `64`, `128`, and `256` have been used in different lanes.
- The q884 lane currently matters more than further group-size tuning, but this
  remains a real knob.

### q16 Passthrough

Goal: high-precision IO entry/exit.

Knobs:

- `QUANT_BITS_OVERRIDES=blocks.0.:16`

Upside:

- Keeps the first/last physical tail block as fp16 passthrough.
- Useful for precision-ladder experiments.

Cost/risk:

- Spends more artifact bytes.
- Did not rescue the d512/e192 i5/l5 lane enough by itself.

Current read:

- q16 passthrough is implemented in both train-time forward and export.

## LQER And Artifact Byte-Spend Levers

### LQER Enablement

Goal: quality recovery after low-bit export.

Knobs:

- `LQER_ENABLED=1`
- `LQER_RANK`
- `LQER_TOP_K`
- `LQER_INCLUDE_PATTERNS`
- `LQER_EXCLUDE_PATTERNS`
- `LQER_ASYM_ENABLED=1`
- `LQER_ASYM_GROUP`

Upside:

- Spends bytes on the tensors with biggest quantization residuals.
- Restores quality while keeping the main tensor low-bit/ternary.

Cost/risk:

- Can push near-cap rows over the byte goal.
- Must reload correctly. Ternary layers need sidecars instead of folding into
  latent weights that would be ternarized away.

Current read:

- Reload-safe LQER is implemented and critical.
- q884 quality is strongly tied to LQER rank/top-K choices.

### LQER Rank And Top-K

Goal: quality versus size.

Knobs:

- `LQER_RANK=6`, `8`, `16`
- `LQER_TOP_K=11`, `12`, `14`, `16`, `24`, `32`

Upside:

- Rank controls correction capacity per selected tensor.
- Top-K controls how many residual tensors receive sidecars.

Cost/risk:

- Higher rank/top-K can exceed the cap.
- Too low gives up BPB.

Current read:

- q884 `r6t12` is the best clean legal row.
- q884 `r6` is better quality but slightly over cap.
- `t12` alone was closest to legal in one sweep but gave up quality.

### LQER Factor Bits

Goal: size savings in symmetric fallback.

Knobs:

- `LQER_FACTOR_BITS`

Upside:

- Could reduce sidecar bytes in symmetric LQER mode.

Cost/risk:

- Does not help when asymmetric LQER is enabled.

Current read:

- With `LQER_ASYM_ENABLED=1`, factor bits are not a useful byte lever. Do not
  spend more matrix time on `fb3` unless testing the symmetric fallback.

### IO-Aware LQER

Goal: quality recovery where it matters most.

Knobs:

- `LQER_INCLUDE_PATTERNS=tok_emb.weight,embed_proj,blocks.`
- `LQER_EXCLUDE_PATTERNS=lm_head.weight,token_smear,attn_gate_w,attn_out_gate`

Upside:

- Directs residual bytes to embeddings/projection/blocks rather than tiny
  controls or tied output.

Cost/risk:

- Include/exclude mistakes can waste bytes or miss important tensors.

Current read:

- IO-aware LQER helped in 1k export-aware screens.
- For the q884 lane, rank/top-K tuning is the practical byte/quality control.

### Codec Choice

Goal: artifact size.

Knobs:

- `MODEL_CODEC=lzma`
- `MODEL_CODEC_LEVEL=9`
- zlib fallback in older logs

Upside:

- lzma is preferred for tiny artifacts.
- Can be the difference between soft-cap and legal.

Cost/risk:

- Compression time and compatibility must remain acceptable.

Current read:

- Use lzma for sub-4.
- Sub-16 q6 proof logs used zlib in some paths; sub-16 has more headroom.

### Float Keep/Control Tensor Policy

Goal: stability and size.

Knobs:

- `INT8_KEEP_FLOAT_MAX_NUMEL`
- `KEEP_CONTROL_PARAMS_FP32`
- ternary exclude patterns: `tok_emb.weight,lm_head.weight,embed_proj`
- int8 promote patterns

Upside:

- Keeps tiny control tensors stable and avoids quantizing things that are not
  worth compressing.

Cost/risk:

- Too many keep-float tensors waste bytes.
- Too few can destabilize training or export.

Current read:

- Keep tiny controls precise in serious lanes.
- Exclude/promote patterns need to be checked when adding new modules.

## Optimizer And Schedule Levers

### Hybrid Muon Versus AdamW

Goal: quality and speed tradeoff.

Knobs:

- `OPTIMIZER_PRESET=hybrid`
- AdamW-only speed probes
- Muon trunk with Adam for scalar/vector/embedding params

Upside:

- Muon improved quality substantially in early sub-4 quality rescue runs.

Cost/risk:

- Slower than AdamW.
- Needs dtype and backend tuning.

Current read:

- AdamW was worse for quality in the sub-4 rescue phase.
- Keep hybrid Muon as the default serious lane unless a speed probe preserves
  the loss curve.

### Learning Rates By Parameter Family

Goal: quality and stability.

Knobs:

- `TIED_EMBED_LR`
- `MATRIX_LR`
- `SCALAR_LR`
- profile presets such as `cooltaper5k_cold_tokens8k`

Upside:

- Separate LR for embeddings, matrices, and scalar/control params is a strong
  stabilizer.

Cost/risk:

- Larger shapes can explode with sprint LR.
- Too-low LR underlearns.

Current read:

- Cooler/cold LR rescued d768/e256 on CaseOps.
- Early d96 wins came more from LR/warmdown fixes than from deeper shapes.

### Warmup And Warmdown

Goal: stability and final quality.

Knobs:

- `LR_WARMUP_ITERS`
- `WARMDOWN_ITERS`
- `LR_WARMDOWN_STYLE=cosine`
- `LR_MIN_SCALE`
- `WARMUP_STEPS` only as runtime reset/warmup, not LR warmup

Upside:

- Real LR warmup stabilized larger models.
- Long cosine warmdown prevented late non-finite failures.
- min-LR can preserve movement late.

Cost/risk:

- Wrong warmdown can waste steps or hurt late quality.
- In fixed-iteration runs, warmdown should match actual step budget.

Current read:

- d384 needed long wall-clock warmdown for finite 600s runs.
- d768/e256 cold 8k fixed 5k was better than simply running to 600s in one
  tested lane.
- `LR_MIN_SCALE` is implemented but not promoted by itself.

### Muon Variants

Goal: quality and stability.

Knobs:

- `MUON_NS_VARIANT=polar_express`
- `MUON_NS_VARIANT=gram_polar`
- `MUON_ROW_NORMALIZE=1`
- `MUON_WEIGHT_DECAY`
- `MUON_WD`
- `MUON_WEIGHT_DECAY_MODE=huber`
- `MUON_WEIGHT_DECAY_HUBER_DELTA_SCALE`
- `MUON_BACKEND_STEPS`
- `MUON_DTYPE=fp16`

Upside:

- Matches public leader-inspired optimizer tricks.
- Huber/decoupled decay may suppress outlier tails before low-bit export.
- fp16 Muon state can speed local training.

Cost/risk:

- Several public-knob screens did not beat the cold baseline.
- Row-normalized Muon and gates were slower in some screens.

Current read:

- Implemented and available.
- Use as ablation knobs, not default replacements, until they win at 5k or
  wall-clock.

### QK Gain

Goal: attention quality/stability.

Knobs:

- `QK_GAIN_INIT=5.25`

Upside:

- Public-leader-derived ablation.

Cost/risk:

- Did not beat current local lead in the sub-4 screens.

Current read:

- Keep as a controlled ablation.

### Logit Softcap

Goal: stability and loss behavior.

Knobs:

- `LOGIT_SOFTCAP`

Upside:

- Helped guarded stability paths.

Cost/risk:

- Some sub-4 lead/probe lanes run without it for speed or because the public
  fused CE path differs.

Current read:

- Useful for guarded stability and d384/d512 rescue.
- Not a universal default for q884.

### Loss Precision

Goal: stability versus speed.

Knobs:

- `LOSS_FP32=1` or `0`
- `LOSS_TOKEN_STRIDE`
- `LOSS_TOKEN_RANDOM_OFFSET`

Upside:

- fp32 loss improves stability.
- Lower precision/probe loss can speed experiments.

Cost/risk:

- Turning off fp32 loss can damage quality/stability.
- Striding/sampling loss can bias training.

Current read:

- Keep `LOSS_FP32=1` for serious quality runs.
- Use speed shortcuts only as probes.

## Neural Control And Side-Channel Levers

### Scalar Smear Gate

Goal: quality.

Knobs:

- `SMEAR_GATE_MODE=scalar`
- publicstack presets

Upside:

- Public leader-inspired residual control.

Cost/risk:

- Did not beat the cold baseline in the local 1k/5k screens.
- Stacking gates blindly can hurt.

Current read:

- Implemented and smoke-tested.
- Retune before promotion.

### Sparse Attention Gate

Goal: quality and attention control.

Knobs:

- `SPARSE_ATTN_GATE_ENABLED=1`
- older transparent path: `ATTN_OUT_GATE_ENABLED=1`, `ATTN_OUT_GATE_WIDTH`

Upside:

- Public-style attention-output gating.

Cost/risk:

- Slight overhead.
- Stacking with scalar smear hurt in the 1k screen.

Current read:

- Works, not promoted.

### BigramHash Side Channel

Goal: quality via cheap lexical side information.

Knobs:

- Bigram vocab and dimension model knobs.
- competitor records mention BigramHash sizes such as 10240/3072.

Upside:

- Can add lexical/context signal with a compact interface.

Cost/risk:

- Local sub-4 BigramHash 16k x 128 did not win.
- Larger variants caused Windows CUDA teardown failures in some runs.

Current read:

- Not a promoted sub-4 lever right now.

### VE / Extra Byte-Spend Side Channels

Goal: quality by spending unused bytes.

Knobs:

- VE settings in model profiles.

Upside:

- Another way to spend byte headroom.

Cost/risk:

- Earlier VE/bigram byte-spend lanes did not beat plain e80 in the nano
  family.

Current read:

- Low priority compared with q884/LQER/capacity.

### XSA / Extra Attention

Goal: quality.

Knobs:

- XSA settings in sub-16/public branch families.

Upside:

- Strong public records use XSA/partial XSA.

Cost/risk:

- More code and compute.
- Not yet the main sub-4 branch.

Current read:

- More relevant to sub-16 than current sub-4 q884 work.

### Test-Time Training

Goal: evaluation quality.

Knobs:

- score-first TTT control presets.
- future warm-A/phased LoRA TTT.
- TTT prefix/update controls in the trainer.

Upside:

- Public leaders use legal TTT variants.
- Local control TTT helped itself slightly.

Cost/risk:

- Legality rules are strict: can only train on already-scored validation tokens.
- Full warm-A/phased LoRA is larger branch work.

Current read:

- Control-only TTT improved one run from `2.6976` to `2.6834` BPB, but did not
  beat the lead.
- Full public-style TTT remains one of the biggest missing sub-16/sub-4 quality
  levers.

## Tokenizer And Data Levers

### CaseOps/SP8192

Goal: quality and fair comparison.

Knobs:

- CaseOps dataset path:
  `fineweb10B_sp8192_lossless_caps_caseops_v1_reserved`
- tokenizer:
  `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- `DATA_PATH`
- `TOKENIZER_PATH`
- `VOCAB_SIZE=8192`

Upside:

- Aligns with current strong public transformer/HRC records.
- Uses exact byte sidecars for BPB.

Cost/risk:

- Larger vocab interface; needs factored tied embeddings under sub-4.

Current read:

- Serious experiments should use CaseOps/SP8192 unless specifically testing a
  tokenizer alternative.

### Vocab Size Sweep

Goal: tokenizer quality/size/speed tradeoff.

Knobs:

- `VOCAB_SIZE=4096`, `6144`, `8192`
- tokenizer sweep specs in `data/tokenizer_specs_lossless_caseops_sweep.json`

Upside:

- Smaller vocab can reduce embedding/output bytes and maybe speed.
- Larger vocab can reduce token fertility and improve sequence modeling.

Cost/risk:

- Smaller vocab increases token count.
- Larger vocab is expensive under sub-4.

Current read:

- Sub-4 should test 4096/6144 CaseOps or word-boundary variants.
- Sub-16 can afford 8192 as the main lane.

### Word-Boundary-Aware BPE/Unigram

Goal: quality by smarter tokenization without lossy whole-word vocab.

Knobs:

- SentencePiece BPE/Unigram config.
- word-boundary behavior.
- byte fallback and exact reconstruction.

Upside:

- Tests the "whole word" instinct safely.
- Could reduce harmful fragmentation without a huge pure word vocab.

Cost/risk:

- Must prove exact byte accounting.
- Needs artifact and step-speed measurement, not only token fertility.

Current read:

- Worth exploring after the current model-shape/quant levers.

### Lossless CaseOps-v2 / Reserved Operators

Goal: quality and token fertility.

Knobs:

- reversible capitalization controls.
- reversible symbols for repeated whitespace/newline runs, URLs/emails, numeric
  formatting.
- exact original-byte sidecars.

Upside:

- Can clean the modeling task while staying legal.

Cost/risk:

- Any many-to-one normalization without a sidecar is risky.
- More tokenizer controls add review burden.

Current read:

- Safe if reversible and byte-accounted.
- Avoid lossy casefold/lowercase/accent stripping/NFKC lanes for default
  submission strategy.

### Training Shards And Local Proxy Data

Goal: speed of iteration and fair data scaling.

Knobs:

- `--train-shards`
- local proxy split tools.
- `MATCHED_FINEWEB_*` export env vars.

Upside:

- Faster local iteration.
- Reproducible shard prefixes.

Cost/risk:

- Small local shards do not represent official 8xH100 token exposure.
- Token budget per wall-clock can dominate model comparison.

Current read:

- Use local proxies for screening only.
- H100 runs are needed to judge official competitiveness.

## Training Throughput Levers

### Batch Tokens And Sequence Length

Goal: speed/quality balance.

Knobs:

- `TRAIN_BATCH_TOKENS`
- `TRAIN_SEQ_LEN`
- `VAL_BATCH_SIZE`
- microbatch settings
- `GRAD_ACCUM_STEPS`

Upside:

- Larger batches can improve token exposure and stability.
- Shorter sequences speed local runs.

Cost/risk:

- Larger batches can hurt early local validation or exceed VRAM.
- Short sequences may under-train long-context behavior.

Current read:

- d768/e256 exact 8k-token batches improved fixed 5k quality but slowed steps.
- Larger batch tokens were worse in some d384 1k screens.
- 16MB public lanes use vastly larger token budgets than the local 2060.

### Fused QKV

Goal: speed.

Knobs:

- `TRAIN_FUSED_QKV=1`

Upside:

- Reduces projection overhead.
- Ported from sub-4 speed work into sub-16 probes.

Cost/risk:

- Needs correctness checks when attention shapes change.

Current read:

- Keep enabled in speed-focused lanes where verified.

### Lower Precision Params And Optimizer State

Goal: speed and memory savings.

Knobs:

- `PARAM_DTYPE=fp16`
- `MUON_DTYPE=fp16`
- `TRAIN_CASTED_LINEAR_PARAM_DTYPE=model`
- `TRAIN_TERNARY_PARAM_DTYPE=model`
- `USE_GRAD_SCALER=0`

Upside:

- Faster and lower memory on local CUDA.
- Avoids extra scaler work when the lane is already low precision.

Cost/risk:

- Can reduce stability for some larger or conservative sub-16 rows.

Current read:

- Good for sub-4 lanes.
- Sub-16 conservative q6 proof remains fp32/GradScaler until speed probes prove
  stable.

### Zero-Grad And Validation Frequency

Goal: speed.

Knobs:

- `POST_STEP_ZERO_GRAD=0`
- `VAL_LOSS_EVERY=0` for speed probes.
- `SKIP_INITIAL_VAL=1`
- `TRAIN_LOG_EVERY`

Upside:

- Avoids extra work inside the timed loop.
- More steps in 10 minutes.

Cost/risk:

- Less monitoring during runs.
- Need final validation to avoid fooling ourselves.

Current read:

- Good for speed probes.
- Serious matrix rows should still produce final artifacts and final validation.

### Loss Shortcuts

Goal: speed.

Knobs:

- `LOSS_FP32=0`
- `LOSS_VOCAB_SAMPLE_SIZE`
- sampled vocab correction.
- loss token stride/random offset.

Upside:

- Can reduce compute on local 2060.

Cost/risk:

- Sampled vocab loss was much worse in CaseOps tests.
- Lower precision loss can destabilize quality comparisons.

Current read:

- Use only as risk probes.
- Do not promote sampled vocab for the current sub-4 lane.

### CUDA Environment

Goal: build/runtime stability.

Knobs:

- `scripts/check_cuda126_env.py`
- `scripts/use_cuda126.ps1`
- local CUDA 12.6-compatible tooling.
- `CUDA_LAUNCH_BLOCKING=1` for crash repro.

Upside:

- Avoids PyTorch `2.11.0+cu126` versus `nvcc` 11.7 extension mismatch.
- Makes CUDA extension experiments less risky.

Cost/risk:

- Windows extension builds remain higher friction than Linux/H100.

Current read:

- Local extension work should use the CUDA 12.6 environment checks.
- q6 illegal-memory rows need Linux/H100 repro before drawing ML conclusions.

### Custom Ternary Kernels

Goal: speed.

Knobs:

- `TRAIN_TERNARY_DENSE_KERNEL=1`
- `TRAIN_TERNARY_PACKED_KERNEL=1`
- `scripts/bench_packed_ternary_linear.py`
- `ternary_golf/packed_cuda.py`

Upside:

- Potentially reduces ternary materialization/matmul overhead.
- Long-term path for "train ternary from the start" without waste.

Cost/risk:

- Current packed matmul replacement is much slower locally.
- JIT/load overhead matters inside timed competition settings.
- Replacing mature GEMM is hard, especially on RTX 2060.

Current read:

- Packed kernel is not competitive yet.
- Next serious kernel work should target fused dense ternary materialization
  feeding Tensor Cores rather than replacing Tensor Core GEMM.

### Compile / Triton / Fused CE

Goal: speed and quality.

Knobs:

- torch compile controls if added.
- fused softcapped CE branch.
- Triton kernels.

Upside:

- Public leaders use fused paths for speed.

Cost/risk:

- High risk on this Windows/2060 setup.
- Compile overhead may not pay back in short local runs.

Current read:

- Avoid compile locally unless a profile proves payoff.
- Fused softcapped CE is a future branch, not current sub-4 default.

## Export And Size-Saving Levers

### Bit Width

Goal: size savings.

Knobs:

- q2, q4, q5, q6, q8, q16 policy.
- `QUANT_WEIGHT_BITS`
- `QUANT_BITS_OVERRIDES`

Upside:

- Direct control over artifact size.
- Mixed precision lets us spend bits where needed.

Cost/risk:

- Too low a bit width damages quality.
- q6 local path showed instability.

Current read:

- q884 is the best tested sub-4 compromise.
- q16/q8/q4/q2/ternary is correct but needs more capacity to matter.

### Reduce LQER Payload

Goal: fit under cap.

Knobs:

- lower `LQER_TOP_K`
- lower `LQER_RANK`
- exclude tensors from `LQER_INCLUDE_PATTERNS`
- compress with lzma

Upside:

- Can turn a strong soft-cap row into a legal row.

Cost/risk:

- The best quality row may lose BPB when shaving residuals.

Current read:

- q884 `r6` needs about `35KB` shaved.
- q884 `r6t12` is the current legal compromise.

### Code Size And Logging

Goal: artifact size and run cleanliness.

Knobs:

- `LOG_CODE_SNAPSHOT=0`
- `LOG_NVIDIA_SMI=0`
- keep generated logs/checkpoints out of the submission artifact.

Upside:

- Saves counted bytes where code size matters.
- Keeps logs cleaner.

Cost/risk:

- Less debugging context unless logs are stored separately.

Current read:

- Disable snapshots in sub-4 matrix runs.

### Smaller Vocab Or Embedding Rank

Goal: size savings.

Knobs:

- `VOCAB_SIZE`
- `FACTORED_EMBED_DIM`
- tokenizer choice.

Upside:

- Large byte savings at the lexical interface.

Cost/risk:

- More tokens per byte or weaker lexical representation can hurt BPB.

Current read:

- Test 4096/6144 CaseOps variants for sub-4.
- Keep SP8192 as main sub-16 lane.

## Public-Leader-Inspired Levers

### CaseOps And Byte Sidecars

Goal: quality and legality.

Status:

- Already part of the serious lane.

### Scalar Smear + Sparse Gate + LQER

Goal: match public neural controls.

Status:

- Implemented.
- Not automatically promoted; publicstack helped 1k export-aware screens but
  not enough later.

### Polar/Gram Muon, Row Norm, Muon WD

Goal: optimizer quality.

Status:

- Implemented.
- Treat as ablation knobs.

### Full Warm-A / Phased LoRA TTT

Goal: eval-time quality.

Status:

- Not fully implemented in the current sub-4 lane.
- High-value future work if legality and byte accounting stay clean.

### FLA / GatedDeltaNet

Goal: architecture leap.

Status:

- Public branch pressure for sub-16.
- Large architecture pivot, not a safe incremental HRC lever.

### Byte-Level PPM Mixture

Goal: score.

Status:

- Potentially very strong but legality-gated.
- Keep separate from default neural lane until organizer guidance is clear.

## Candidate Family Levers

### Nano/Micro Family

Examples:

- `i1l2r2_d96_e48_h3mha_mlpinner_mlp2`
- `i1l2r2_d192_e80_h3mha_mlpinner_mlp15`
- `i1l2r2_d384_e128_h8kv1_mlpinner_mlp10`

Use when:

- Fast smoke tests.
- Debugging train-time ternary or schedule behavior.

Current read:

- No longer the quality lead.
- Useful for fast iteration and stability lessons.

### Wide Shallow HRC

Examples:

- `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075`
- `i1l2r2_d1536_e384_h24kv1_mlpinner_mlp050`
- `i1l2r2_d2048_e512_h32kv1_mlpinner_mlp025`

Use when:

- Spending byte headroom on capacity.
- H100 scaling runs.

Current read:

- d768/e256 is the local anchor.
- H100 capacity ladder is a high-value next step.

### q884 IO-Tail Family

Examples:

- `i3l3r3_d768e256_q884_coret_lqer_r6t12`
- `i3l3r3_d768e256_q884_coret_lqer_r6`

Use when:

- Chasing current best sub-4 quality.

Current read:

- Current promoted legal family.
- Improve by shaving soft-cap row, widening carefully, or retuning LQER.

### Precision Ladder i3/i4/i5 Family

Examples:

- `i3l9r5_d512e192_q16q8t_coret_lqer_lidx_r6t12`
- `i4l9r5_d512e192_q16q8q4t_coret_lqer_lidx_r6t12`
- `i5l5r2_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`
- `i5l5r9_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`
- `i5l9r5_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`
- `i5l9r9_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`

Use when:

- Testing the hypothesis that smooth precision tapering into a ternary core
  helps.

Current read:

- Correctly implemented from step one.
- Loop index helps this family.
- `i5l5r9` is the best clean fixed-step sub-4 row so far.
- `i4l9r5` is the best soft-target quality row so far.
- `i5l9r5` is still strong, but slower and worse than i4/l9/r5.
- `i5l9r9` shows that more repeats can overdo the route: it was slower and
  worse than r5.
- `i3l9r5` shows that dropping the q4 tail layer is too aggressive.

### i6/l9 Deep IO Tail

Examples:

- `i6l9r3_d256e96_q886644_coret`
- `i6l9r3_d320e128_q886644_coret_lqer`

Use when:

- Testing long IO ladder shapes.

Current read:

- Did not beat q884 r3.
- Often slower or too narrow.

## Negative Or Caution Levers

Do not promote these without new evidence:

- Export-only quantization.
- `QUANT_TRAIN_MODE=roundtrip` inside serious wall-clock runs.
- `LQER_FACTOR_BITS` as a size lever with asymmetric LQER enabled.
- Naive whole-word tokenizer.
- Lossy lowercasing, casefolding, accent stripping, or NFKC normalization.
- Sampled vocab loss for current CaseOps sub-4.
- AdamW-only for quality.
- Very high recurrence locally without a step-budget reason.
- Packed ternary matmul replacement as currently implemented.
- q6 train-time fake-quant on this Windows/CUDA path without crash repro.
- Comparing SP1024 proxy loss to CaseOps/SP8192 BPB.
- Wall-clock comparisons made while the GPU is busy.

## Practical Pull Lists

### If The Goal Is Better Sub-4 Quality

1. Start from `i4l9r5_d512e192_q16q8q4t_coret_lqer_lidx_r6t12` for
   fixed-step quality exploration.
2. Start from `i3l3r3_d768e256_q884_coret_lqer_r6t12` for local 10-minute
   wall-clock comparison.
3. Treat 4MB as a target for this experimental lane; keep slightly-over rows if
   they reveal a better architecture.
4. Spend bytes on capacity or LQER, not unused headroom.
5. Keep `TRAIN_QUANT_FORWARD=1`.
6. Use CaseOps/SP8192 and final export BPB.
7. Test loop index only if route depth is high enough to justify it.

### If The Goal Is More Sub-4 Speed

1. Use shallower/wider rows before deep r9 recurrence.
2. Keep fused QKV and low-precision params where stable.
3. Avoid roundtrip projection during training.
4. Skip periodic validation inside timed runs.
5. Use idle GPU guards for fair wall-clock matrices.
6. Consider kernel work only after the model lane is worth optimizing.

### If The Goal Is Smaller Artifact Size

1. Lower LQER top-K or rank.
2. Tighten LQER include/exclude patterns.
3. Use lzma level 9.
4. Reduce factored embedding rank or vocab size.
5. Push more core blocks to ternary.
6. Keep code/log artifacts out of counted payload.

### If The Goal Is Better Sub-16

1. Stay close to q6 proof plus CaseOps/SP8192.
2. Resume LQER/frozen-carry/publicstack ladder.
3. Add speed probes only if loss movement tracks baseline.
4. Implement or isolate legal full TTT.
5. Consider larger architecture branches separately from HRC incremental work.

### If The Goal Is A Safe Tokenizer Improvement

1. Keep every transform reversible.
2. Preserve exact original byte sidecars.
3. Sweep CaseOps/word-boundary BPE vocab sizes 4096, 6144, 8192.
4. Measure token fertility, final artifact bytes, BPB, and step speed.
5. Avoid lossy normalization unless explicitly approved.

## Current Best Next Bets

1. Run a 10-minute wall-clock comparison between `i4l9r5`, `i5l9r5`,
   `i5l5r9`, and the q884 r3 wall-clock anchor.
2. Explore i4/l9 size/quality tweaks around r5 rather than adding more repeats.
3. Improve the q884 r3 legal row or shave the q884 `r6` soft-cap row.
4. Run the d1536/e384 and d2048/e512 H100 capacity ladder with export-honest
   train-time quant.
5. Resume sub-16 LQER/frozen-carry/publicstack after the corrected carry block
   selector.
6. Build the safe tokenizer sweep only with exact byte accounting.
