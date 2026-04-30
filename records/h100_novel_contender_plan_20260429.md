# H100 Novel Contender Plan

Date: 2026-04-29

Public naming note: the code and candidate names still say `HRC` and
`VocabMoE`. In the submission story, those are MirrorLoop Recurrent Core and
Lexical Low-Rank Experts (LexLoRE). See
`records/architecture_explainer_20260430.md`.

This plan is for paid H100 time. It deliberately does not chase a vanilla
leaderboard clone. The point is to give the project-specific HRC/VocabMoE lane
a fair official-style scaling test: 1024-token context, large token batches,
10-minute wall clock, final artifact roundtrip, and 8-process launch when an
8xH100 box is available.

Before renting hardware, run the local gate in
`records/local_h100_preflight_plan_20260429.md`. A row that crashes, exports
poorly, or is clearly dominated locally should not consume H100 time.

For RunPod execution, use the no-fetch bundle workflow in
`records/runpod_no_fetch_plan_20260429.md`. The paid runner should validate a
locally uploaded bundle and should not clone repos, fetch PRs, or download data
during the experiment.

## Thesis

The strongest local signal is not "ordinary small transformer." It is:

- lossless CaseOps/SP8192 token-facing interface;
- factored tied embeddings;
- mirrored HRC IO shell with a looped middle;
- q8 train/export on the 16MB lane to avoid the q6 export gap;
- dense VocabMoE at input plus loop-first;
- more unique loop blocks, especially the i3/l5/r5 family;
- Polar Express Muon plus a warmdown LR floor;
- final export roundtrip as the only score that counts.

Local 2060 numbers are undertrained proxies. The current best local row is
`frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.5336` BPB and `14,091,166`
bytes after 5k local steps, while its e640 sibling used `14,574,862` bytes.
Those are useful, but they still leave too much room under the 16MB cap. The
paid matrix now pushes the same spine into the e768/e896/e1024 token-rank
region and spends extra bytes on dual/spike/BigramHash rows instead of testing
short artifacts. The H100 path uses much longer context and far more tokens per
optimizer step, so final artifact size can drift; slightly over-cap rows are
kept only as scout signal and must be filtered before any record-style claim.

## Round A: Five Paid 8xH100 Rows

Run group: `h100_novel_round1` in
`scripts/run_h100_novel_matrix.py`.

| Candidate | Why It Exists |
| --- | --- |
| `h100_capfill_i3l5r5_d640e896_q8_polar` | Main near-cap HRC/VocabMoE row; spends the proven e640 gain further on token rank. |
| `h100_capedge_i3l5r5_d640e1024_q8_polar` | Cap-edge probe; expected to land closest to 16MB and tells us whether token rank keeps buying BPB. |
| `h100_capfill_i3l7r4_d640e640_q8_polar_bigram` | More unique loop blocks plus BigramHash, with enough embed rank to avoid another short artifact. |
| `h100_capfill_dual_i3l5r5_d640e896_q8_polar_left320` | Trained left/right advisor bridge with more left-stream width and bridge rank. |
| `h100_capfill_spike_loopall_i3l5r5_d640e768_q8_polar` | Hard token/expert self-election at every recurrent-loop position, but sized closer to cap. |

These five are all still "ours": HRC route, looped middle, VocabMoE or
self-electing experts, and train-time quantized q8 export policy. The public
leaderboard influence is limited to stable schedule/system levers, not the
model identity.

## 1xH100 Execution Profile

The first RunPod checks exposed one important launch issue: the common profile
was 8xH100-sized, so the single-H100 smoke needed an explicit 1x batch override.
After that, three no-math-change speed fixes were applied:

- fused QKV enabled (`TRAIN_FUSED_QKV=1`);
- reset-only warmup removed while compile remains disabled;
- every-step nonfinite scans and redundant post-step `zero_grad` disabled for
  paid scouts.

Observed step speed on `h100_capfill_i3l5r5_d640e896_q8_polar`:

| Profile | Step Speed |
| --- | ---: |
| patched 1x batch, unfused QKV, safety scans on | ~314 ms/step |
| fused QKV | ~298 ms/step |
| fused QKV + lean safety/zeroing profile | ~271 ms/step |

The active 1xH100 round now uses the lean profile. Re-enable nonfinite checks
only for debugging unstable rows, not for paid ranking runs.

## Cloud Commands

First smoke one row on a single H100 to check data, launch, export, and logs:

```bash
python scripts/run_h100_novel_matrix.py \
  --candidate-group h100_novel_round1 \
  --candidates h100_capfill_i3l5r5_d640e896_q8_polar \
  --nproc-per-node 1 \
  --wallclock-seconds 180 \
  --val-tokens 65536 \
  --timeout 600 \
  --out records/h100-smoke-novel-$(date +%Y%m%d-%H%M%S)
```

Then run the paid 8xH100 scout:

```bash
python scripts/run_h100_novel_matrix.py \
  --candidate-group h100_novel_round1 \
  --nproc-per-node 8 \
  --wallclock-seconds 600 \
  --val-tokens 131072 \
  --timeout 1200 \
  --out records/h100-roundA-novel-$(date +%Y%m%d-%H%M%S)
```

For final packaging or a higher-confidence score, rerun the top rows with full
validation by setting `--val-tokens 0`:

```bash
python scripts/run_h100_novel_matrix.py \
  --candidate-group h100_novel_round1 \
  --candidates <comma-separated-top-candidates> \
  --nproc-per-node 8 \
  --wallclock-seconds 600 \
  --val-tokens 0 \
  --timeout 1800 \
  --out records/h100-final-novel-$(date +%Y%m%d-%H%M%S)
```

## Data Setup

The H100 runner expects the CaseOps SP8192 data exported under the same path
used by the local matrix:

`upstream_records/records/track_10min_16mb/2026-04-18_PR1626_CaseOps_Taper/`

Smoke data command:

```bash
MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1 \
MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets \
python upstream_records/records/track_10min_16mb/2026-04-18_PR1626_CaseOps_Taper/cached_challenge_fineweb.py \
  --variant sp8192_lossless_caps_caseops_v1_reserved \
  --train-shards 1
```

For serious runs, use the full available train-shard set if the pod has time
and disk. The script scores with `VAL_TOKENS_LIMIT` so Round A can stay fast,
then the final rerun can use full validation.

## Promotion Rules

- Promote by final exported BPB, not train-time validation.
- If the anchor is already above roughly `1.35` BPB on 8xH100, do not spend the
  final paid slot on narrow polish; use the result as a non-record/art lane.
- If any row reaches the low `1.2x` band, rerun that row and the closest sibling
  with full validation and at least one extra seed.
- If dual or spike wins, the submission story is novel. If only the anchor wins,
  the story is still HRC plus VocabMoE plus train-time q8 export, but less wild.
- Do not expand to broad dense-transformer controls unless the user explicitly
  pivots back to official-record cloning.

## Why This Is Worth Paid Time

The local 2060 runs are bottlenecked by short context, small token batches, and
slow wall-clock exposure. A `1.53` local export BPB on a novel architecture is
not proof of official competitiveness, but it is good enough to justify one
clean H100 scaling read. If the long-context, high-token version does not move,
the field learned something useful. If it does move, this becomes a genuinely
interesting non-record submission and possibly a contender-shaped architecture.

## 2026-04-30 Architecture Evolution Queue

While the `h100_1x_beat135_exportfix` queue is running, the next isolated lane
has been staged on RunPod to start automatically afterward:

```bash
python3 scripts/run_h100_arch_evolution_matrix.py \
  --out records/h100_arch_evolution_full10_after_beat135 \
  --candidate-group h100_arch_evolution \
  --nproc-per-node 1 \
  --wallclock-seconds 600 \
  --val-tokens 131072 \
  --timeout 1500
```

This lane uses `train_gpt_arch_evolution.py`, a cloned trainer, so the current
best-family trainer stays stable. The six probes are one-lever evolutions of
the best legal H100 spine: exit-side LexLoRE, warmer rank-4 LexLoRE, 32-expert
LexLoRE, CycleFuse, ValueEmbedding in the first core-attention block, and a new
partial-tail recurrence route `012|34567|34567|567|210`.

## 2026-04-30 Export-Fix Queue Result

`records/h100_1x_beat135_exportfix_full10_direct` completed and did not
promote:

- `h100_beat135_32k_d704e864_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55`:
  final export `1.35934317` BPB, `119.73ms/step`, `16,176,261` bytes
  (`176,261` over cap).
- `h100_beat135_24k_d704e800_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55`:
  final export `1.35945832` BPB, `95.68ms/step`, `16,238,469` bytes
  (`238,469` over cap).
- `h100_beat135_20k_d704e768_w3200_q8_coreattn1_lqer11t22_vocabmoe_qk55`:
  final export `1.36275698` BPB, `84.84ms/step`, `15,992,737` bytes
  (`7,263` headroom).

Conclusion: pushing e864/e800 or stronger LQER around this exact r2 spine does
not beat the known legal best and can silently consume the remaining code-byte
headroom. Keep the `1.35692129` legal row as the reference until an evolution
probe beats it after export.

## 2026-04-30 Architecture Evolution Interim Result

`records/h100_arch_evolution_full10_after_beat135` is running.

Completed:

- `h100_evo_32k_e832_lexlore_exit_looplast`: final export `1.35868072` BPB,
  `120.83ms/step`, `15,743,042` bytes, `256,958` bytes headroom.
- `h100_evo_32k_e832_lexlore_rank4_warm_exit`: final export `1.36211877`
  BPB, `120.90ms/step`, `15,760,710` bytes, `239,290` bytes headroom.
- `h100_evo_32k_e832_lexlore_32x2_warm_exit`: final export `1.36056721`
  BPB, `120.80ms/step`, `16,109,898` bytes, `109,898` bytes over cap.
- `h100_evo_32k_e832_cyclefuse_auto`: final export `1.36939320` BPB,
  `133.07ms/step`, `15,473,918` bytes, `526,082` bytes headroom.

Interim read: exit-side LexLoRE is legal and slightly cleaner than the failed
e864/e800 cap-spend rows, but it does not beat the current legal best
`1.35692129`. Warm rank-4 LexLoRE is a negative: it likely wakes the adapter too
aggressively and worsens export BPB. More expert bins at rank 2 is also a
negative and goes over cap. CycleFuse is the clearest no so far: it slows steps
and hurts export BPB. The remaining useful questions in this queue are VE and
partial-tail recurrence.
