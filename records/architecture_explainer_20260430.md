# Architecture Explainer: MirrorLoop, LexLoRE, And The Novel Lane

Date: 2026-04-30

This note explains the project-specific model ideas in submission language.
The code still uses the historical names `HRC_*` and `VOCAB_MOE_*` because the
run matrix, logs, and active H100 jobs depend on them. For public writeups, the
clearer names are:

| Public name | Code name | Short meaning |
| --- | --- | --- |
| MirrorLoop Recurrent Core | `MODEL_FAMILY=hrc`, `HRC_*` | A mirrored token-facing IO shell wrapped around a reused recurrent middle. |
| Lexical Low-Rank Experts, or LexLoRE | `VocabMoELite`, `VOCAB_MOE_*` | Token-conditioned shared low-rank adapters, placed at input and selected virtual layers. |
| Data-Density Ladder | `LAYER_WIDTH_SCHEDULE` plus quant overrides | Higher-precision token-facing blocks can be narrower, while lower-precision core blocks can spend width. |
| Export-Honest Quantization | `TRAIN_QUANT_FORWARD=1` and final export roundtrip | Train through the same low-precision forward view that will be scored after export. |
| Quant Error Repair | `LQER_*` | Low-rank sidecars that repair the largest quantization residuals inside the artifact. |

## Short Pitch

The main non-record architecture is a mirrored recurrent token specialist. A
small token-facing shell reads the sequence, a cheap recurrent core is reused
several times, and the shell is traversed backward on the way out. Token-aware
low-rank experts steer the stream at the embedding and loop-entry sites. The
model is trained through quantized weights from step 0, then the final exported
artifact is reloaded and rescored.

This is not just "a smaller transformer." It is a transformer block reuse
scheme plus token-conditioned micro-specialization plus export-honest
quantization.

## MirrorLoop Recurrent Core

MirrorLoop is what the code calls `HRC`. It is not a new attention primitive.
It is a route over a smaller set of physical transformer blocks.

The normal transformer path has one physical block per depth position:

```text
0 1 2 3 4 5 6 7 ...
```

MirrorLoop separates physical blocks into two regions:

- IO shell: the first `i` blocks. These are closest to the token embedding and
  output distribution.
- Recurrent core: the next `l` blocks. These are reused `r` times.

Candidate names like `i3l5r2` mean:

- `i=3`: three token-facing shell blocks;
- `l=5`: five unique core blocks;
- `r=2`: two passes through that core.

The implementation uses:

- `NUM_UNIQUE_BLOCKS=i+l`
- `HRC_RECURSIVE_CORE_START=i`
- `HRC_ROUTE_REPEATS=r`
- `HRC_DEPTH_SCHEDULE_MODE=transition_recursive_cycle`
- `EFFECTIVE_DEPTH=i + l*r + i`

For `i3l5r2`, the virtual route is:

```text
0 1 2 | 3 4 5 6 7 | 3 4 5 6 7 | 2 1 0
```

The final `2 1 0` is the mirrored exit tail. It reuses the same shell blocks,
but the code can apply a mirror transform to the feature dimension. The current
mirror transforms are `signperm` and `householder`; `signperm` reverses feature
order with alternating signs, while `householder` reflects features across a
fixed vector. The point is to let the same physical block play a related but
not identical "exit" role.

## Why MirrorLoop Exists

The competition rewards artifact efficiency. MirrorLoop buys extra virtual
depth without storing extra full blocks:

- More compute depth per stored parameter.
- A natural place for precision tapering: higher precision near the token IO,
  lower precision in the reused middle.
- A natural place for role information: loop index, pass embeddings, route
  phase embeddings, and mirrored role transforms tell a reused block where it is
  in the virtual route.
- A clean art/non-record story: the model literally has a read shell, a looped
  reasoning middle, and a write shell.

In code, `build_hrc_route_package()` creates `block_schedule` and
`mirror_schedule`. The forward pass walks `block_schedule`, calls
`self.blocks[block_idx]`, applies optional pass/loop controls, then applies
LexLoRE and dual-stream bridges at selected virtual positions.

## Attention In The Core

MirrorLoop still uses attention. Each physical `Block` normally has attention
and MLP branches. For speed and size, many candidates make most repeated core
blocks MLP-only through `HRC_MLP_ONLY_BLOCKS`, then keep attention at the shell
and sometimes one core-entry block. The best H100 branch so far used one
attention-capable recurrent-core entry rather than an attentionless core.

That is important for the writeup: the model is not "no attention." It is
"attention concentrated at token-facing and loop-entry positions, with cheap
recurrent MLP passes doing most of the reused work."

## LexLoRE: Lexical Low-Rank Experts

`VocabMoE` was the prototype name, but it is more precise to call it LexLoRE:
Lexical Low-Rank Experts.

It is not a huge MoE where every token owns a separate micro-network. That
would be slow and artifact-expensive. Instead:

- Each token has a tiny learned router prior:

```text
token_prior: vocab_size x num_experts
```

- The expert transforms are shared low-rank bases:

```text
down: num_experts x rank x dim
up:   num_experts x dim x rank
```

- At a selected site, the routing logits are:

```text
token_prior[token_id] + optional hidden_router(x) + optional site_bias
```

- A softmax, or optional sparse top-k "spike" router, mixes the shared expert
  bases and adds a residual delta back into the stream.

The forward path is deliberately batched and regular:

```text
x_norm -> expert down projection -> activation -> expert mixing -> up projection -> x + scaled delta
```

This gives token-specialized behavior without launching one CUDA path per
vocabulary item.

## LexLoRE Placement

`VOCAB_MOE_LAYERS` accepts route-aware aliases. The important ones are:

- `input`: after token embedding, before the route begins;
- `loop_first`: the first virtual layer whose physical block is in the
  recurrent core;
- `loop`: every recurrent-core virtual position;
- `loop_everyN`: every Nth recurrent-core virtual position;
- literal virtual layer indices and ranges.

The strongest clean pattern has been input plus loop-first. That gives the
model token-specific steering before the shell and again as it enters the
reused core. Loop-all and spike variants are more exotic and sometimes
unstable.

## Spike / Self-Election Variant

The spike version is the user's "experts elect themselves" idea. In code it is
`VOCAB_MOE_MODE=spike_static`, `spike_hybrid`, or `spike_hidden`, with
`VOCAB_MOE_SPIKE_TOP_K`.

Instead of using the full softmax mixture, it keeps only the top-k experts for
each token/site. With straight-through estimation enabled, training still gets
soft gradients while the forward path behaves like sparse self-election.

The signal so far: it is implemented and real, but the dense LexLoRE branch has
been stronger than spike in the best H100-quality runs. It remains a good art
exhibit because it is visibly different from ordinary transformer routing.

## Export-Honest Quantization

The project has been careful about training at the intended precision instead
of only compressing after training.

The key flags are:

- `TRAIN_QUANT_FORWARD=1`: selected `CastedLinear` weights use STE fake
  quantization during forward passes.
- `TRAIN_QUANT_EMBEDDINGS=1`: the tied token embedding is fake-quantized during
  training too. This fixed a real train/export mismatch on H100.
- `QUANT_WEIGHT_BITS`: default export/train bit width for matrix weights.
- `QUANT_BITS_OVERRIDES`: per-layer or per-tensor bit overrides.
- `QUANT_TERNARY_PATTERNS`: tensors trained and exported as ternary.
- `VOCAB_MOE_TRAIN_QUANT_BITS`: the LexLoRE token prior and expert bases can
  also be trained through q8/q6/etc. from the start.

For IO-tail precision ladders, the design rule is: the recurrent core should
not have higher precision than the smallest meaningful precision in the IO
ladder. For example, `q16/q8/q4` IO implies a q4 or ternary core; if the core is
q8, the IO ladder should be closer to `q16/q8/q8`.

## Factored Tied Embeddings

Full `vocab_size x model_dim` embeddings are expensive, especially with
SP8192/CaseOps. The code supports factored tied embeddings:

```text
tok_emb:        vocab_size x factored_embed_dim
embed_proj:     factored_embed_dim -> model_dim
embed_proj_rev: model_dim -> factored_embed_dim
```

The same token table is used for input and output logits. The factor rank is a
major byte-spending lever: too small hurts the token interface, too large can
push the artifact over the 16MB cap.

## LQER: Low-Rank Quantization Error Repair

LQER is a final-artifact repair mechanism. During export, the code quantizes a
matrix, measures its residual error, and stores low-rank sidecars for the top
residual tensors:

```text
full_weight ~= quantized_weight + A @ B
```

The sidecars are compressed and counted in the artifact. On reload, they are
applied before final scoring. This made the quantized roundtrip more honest and
allowed q8/q6 candidates to recover quality without pretending the full-float
training model is what will be submitted.

## Data-Density Ladder

The width ladder is the user's "data density should not collapse as precision
drops" idea.

`LAYER_WIDTH_SCHEDULE` assigns each physical block an internal width. The
residual stream stays at `MODEL_DIM`, while a `WidthAdaptedBlock` projects down
to the block width, runs the transformer block, and projects the residual delta
back up.

This lets us test shapes like:

```text
high precision IO block: narrower internal width
lower precision core:    wider internal width
```

The tested adapter-style width ladder has not beaten the full-width spine yet,
but it is implemented and useful as an art-lane expression of the precision
ladder idea.

## Dual-Stream Advisor

The dual-stream idea splits the residual feature dimension into two lanes:

- left/token-facing stream;
- right/recurrent-semantic stream.

At configured sites, a low-rank bridge sends small messages both ways:

```text
left  -> low-rank -> right delta
right -> low-rank -> left delta
```

This is trained end-to-end. It is different from the older council idea because
it is not an eval-only mirrored peer vote. It is a real internal communication
bridge inside one normalized predictive distribution.

Result so far: promising but not the current best. Dual-only was a near miss in
local structure probes, while the H100 best branch stayed simpler.

## RLM-Lite Memory And Council

RLM-lite is a legal recursive memory sketch: after a chunk has already been
scored, the model can summarize its hidden states into a small recurrent memory
vector and inject that memory into later chunks. It must never condition the
current token distribution on the current target.

Council mixes multiple predictive distributions before seeing the target. It
was implemented as mirrored peer distributions and dynamic/hard-gated variants.
It is useful for art submissions, but it has not beaten the simpler
MirrorLoop+LexLoRE spine.

## Current Best Read

The current H100 evidence favors this spine:

```text
CaseOps/SP8192
+ MirrorLoop i3/l5/r2
+ d704, factored embedding around e832
+ q8 train/export, train-quant embeddings enabled
+ one attention-capable core-entry block
+ LexLoRE input + loop-first
+ QK gain 5.5
+ LQER around r10/t20
+ small single-H100 batch around 24k-32k tokens
```

Best legal H100 row so far:

```text
h100_batch32k_d704e832_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55
BPB: 1.35692129
steps: 5018
step speed: 119.57 ms
artifact: 15,658,145 bytes
```

A nearby 24k batch row reached `1.35552525` BPB but exported over the 16MB cap,
so the active queue is trying to legalize that branch.

## What Is Actually Novel

Most individual ingredients have relatives in prior work: recurrence,
quantization-aware training, low-rank adapters, MoE routing, tied embeddings,
and low-rank quant repair all exist somewhere.

The novel project identity is the combination:

- mirrored IO shell plus looped recurrent middle as the organizing shape;
- token-conditioned low-rank expert steering at the input and loop-entry sites;
- train-time quantization aligned with the final artifact;
- precision and width treated as a coupled data-density budget;
- H100 batch/update-rate tuning as part of the architecture, not only systems
  polish.

For a non-record/art submission, that is the story to tell: a strange but
auditable model family that is not trying to be a direct clone of the accepted
leaderboard transformer stack.

## Architecture Evolution Queue

Added 2026-04-30 while the H100 export-fix queue was running:

- `train_gpt_arch_evolution.py` is a cloned trainer for route experiments, so
  the current best-family `train_gpt.py` stays stable.
- `scripts/run_h100_arch_evolution_matrix.py` runs six full 10-minute H100
  probes after the current queue exits.
- New cloned-only HRC route mode: `transition_tailN_cycle`. For the active
  probe, `N=3`, `io=3`, `loop=5`, and `repeats=2`, giving:

```text
0 1 2 | 3 4 5 6 7 | 3 4 5 6 7 | 5 6 7 | 2 1 0
```

This is the partial-recurrence version of MirrorLoop: repeat the full semantic
core twice, then refine only the deepest semantic tail before the mirrored exit.

The queued probes test:

- LexLoRE at `input,loop_first,loop_last`.
- Warmer LexLoRE with rank 4, prior noise, and larger initial scale.
- 32-expert LexLoRE at the same rank 2.
- HRC CycleFuse summaries from first-pass states into later repeated phases.
- ValueEmbedding at virtual layer 3 as token-conditioned value-side attention
  information.
- Partial tail recurrence with loop index enabled.

## What Not To Claim

- Do not claim official SOTA. We have not verified on 8xH100 final hardware.
- Do not claim every component is individually new.
- Do not call LexLoRE a full per-token micro-network MoE. It is a shared
  low-rank expert bank with token-conditioned routing.
- Do not report train-time validation without the final exported reload score.
  The export roundtrip is the truth.
