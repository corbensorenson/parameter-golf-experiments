# Parameter Golf Competitor Research

Date: 2026-04-25

## Sources Checked

- OpenAI repo README / current leaderboard notes: <https://github.com/openai/parameter-golf>
- PR #1797: <https://github.com/openai/parameter-golf/pull/1797>
- PR #1801: <https://github.com/openai/parameter-golf/pull/1801>
- PR #1809: <https://github.com/openai/parameter-golf/pull/1809>
- PR #1787: <https://github.com/openai/parameter-golf/pull/1787>
- PR #1795: <https://github.com/openai/parameter-golf/pull/1795>
- PR #1791: <https://github.com/openai/parameter-golf/pull/1791>
- Submission rules / legality guide: <https://github.com/openai/parameter-golf/issues/1017>

## Current Public Target

As of the 2026-04-25 audit, the strongest clearly neural public lane is PR
#1797: PR #1787 base plus scalar SmearGate and asymmetric rank-4 LQER,
reporting a 3-seed mean `val_bpb=1.06157` under the 16,000,000 byte decimal
cap. PR #1801 is close behind at `val_bpb=1.06287` with updated frozen
recurrent carry. PR #1795 reports a much lower byte-level PPM mixture score,
but its own README calls out organizer-ruling risk, so it should remain a
separate legality-gated lane rather than a default neural submission path.

Our sub-4 lane is therefore not one optimizer knob away from leaderboard
parity. To become competitive, it needs three fronts at once:

- exact leader-side neural controls where they are cheap: scalar smear, sparse
  attention-output gate, min-LR, Polar/Gram Muon, Huber/decoupled Muon decay,
  CaseOps, legal TTT;
- a better export codec than plain ternary tensors: LQER-style residual
  recovery or another low-rank correction that spends the remaining 2MB+ of
  headroom on score, not unused cap;
- H100-scaled capacity and runtime tuning: the local 2060 is a screening rig,
  while real 10-minute competitiveness needs the d1536/e384 and d2048/e512
  profiles under 8xH100 step budgets.

## Implementation Audit

The public transformer tricks that are cheap enough for the sub-4MB HRC lane are
now implemented as explicit knobs in `train_gpt.py` and repeatable presets in
`train_gpt_ternary.py`.

## Public Target Update

As of 2026-04-25, the clean transformer/HRC comparison point remains PR #1797
at `val_bpb=1.06157`, with PR #1801 close at `val_bpb=1.06287`. PR #1791 reports
`val_bpb=1.0339` from a K/KV-share FLA/GatedDeltaNet-family branch, but it is a
larger architecture pivot and the thread contains byte-accounting skepticism; do
not treat it as a drop-in HRC trick. PR #1795 reports `val_bpb=1.01252` from a
byte-level PPM mixture with a strict-legal gate; keep that lane separate until
legality is settled for our submission strategy.

Sources:

- <https://github.com/openai/parameter-golf/pull/1797>
- <https://github.com/openai/parameter-golf/pull/1801>
- <https://github.com/openai/parameter-golf/pull/1791>
- <https://github.com/openai/parameter-golf/pull/1795>
- <https://github.com/openai/parameter-golf/issues/1017>

Implemented:

- `LR_MIN_SCALE`: public min-LR warmdown floor, applied only inside warmdown.
- `MUON_NS_VARIANT=polar_express`: exact PR #1787 Polar Express coefficient
  sequence, using the first `steps` tuples.
- `MUON_NS_VARIANT=gram_polar`: exact PR #1809 Turbo/Gram Polar sequence, using
  the last `steps` tuples and Gram-NS dispatch for aspect ratios >= 1.5.
- `MUON_ROW_NORMALIZE=1`: public row-normalized Muon gradient option.
- `MUON_WEIGHT_DECAY` / `MUON_WD`: public decoupled Muon weight decay option.
- `MUON_WEIGHT_DECAY_MODE=huber`: public Huber-style Muon decay option for
  suppressing outlier tails before low-bit export.
- `QK_GAIN_INIT=5.25`: public high-QK-gain ablation preset.
- `SMEAR_GATE_MODE=scalar`: exact scalar residual SmearGate mode, separate
  from the older wider vector gate.
- `SPARSE_ATTN_GATE_ENABLED=1`: exact sparse attention-output gate mode, kept
  mutually exclusive with the older transparent `ATTN_OUT_GATE_ENABLED` path.
- `LQER_ENABLED=1`: asymmetric low-rank export residuals. For train-time
  ternary layers these now load as real `TernaryLinear` sidecar adapters instead
  of being folded into a latent weight that would be ternarized away. Quant-train
  projections temporarily disable LQER so this remains final-export-only.
- `HRC_FROZEN_CARRY_ENABLED=1`: frozen alpha/beta carry over repeated HRC
  middle states, matching the shape of the public PR #1801 idea.
- score-first TTT control preset copied into the sub-4 runner family.
- BigramHash side-channel ablation through existing model knobs.
- tiny attention-output gate ablation through the existing transparent
  `ATTN_OUT_GATE_ENABLED=1`, `ATTN_OUT_GATE_WIDTH=12` path.

Already part of the serious sub-4 lane:

- CaseOps/SP8192 data path.
- train-time ternary blocks, not post-hoc quantization only.
- factored tied embeddings.
- ternary shrinkage fix at export.
- decimal `SUBMISSION_SIZE_CAP_BYTES=4000000`.
- `MODEL_CODEC=lzma`.
- low-precision params and matmuls from the start for the promoted local lane.

Not promoted into sub-4:

- Fused softcapped CE from PR #1787: useful for the softcap transformer lane,
  but the current sub-4 lead runs `LOGIT_SOFTCAP=0` and the Triton path is
  high-risk on this Windows/2060 setup.
- byte-level PPM from PR #1795: very strong public number, but legality/ruling
  risk remains too high to make it a default lane.
- full FLA/GatedDeltaNet from PR #1791: promising for sub-16, but a large branch
  rather than a safe incremental sub-4 change.
- full warm-A/phased LoRA TTT from PR #1787/#1797: we only have the smaller
  score-first control/phase runner, not the full public adaptation stack.

## Local Ablations

Main profile:

- `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075`

Current promoted local preset:

- `2060sprint_micro_muon_cooltaper5k_cold_tokens8k`

### Current-code 5k audit

Record: `records/sub4_micro_matrix_20260425_020306`

| Preset | Steps | Val loss | Val BPB | Step avg |
|---|---:|---:|---:|---:|
| `2060sprint_micro_muon_cooltaper5k_cold_tokens8k` | 5k | 5.7789 | 2.6522 | 82.45ms |

The older best record remains slightly lower at 2.6397 BPB
(`records/sub4_micro_matrix_20260425_001851`), but the current-code audit is
faster and includes the full logging for the new optimizer knobs.

### Public-knob 1k screens

Record: `records/sub4_micro_matrix_20260425_012313`

| Preset | Val loss | Val BPB | Step avg | Result |
|---|---:|---:|---:|---|
| `2060sprint_micro_muon_minlr5k` | 8.1125 | 3.7232 | 65.41ms | Better at 1k, worse at 5k. |
| `2060sprint_micro_muon_polar5k` | 8.2744 | 3.7975 | 65.41ms | Not promoted. |
| `2060sprint_micro_muon_turbogram5k` | 8.3552 | 3.8346 | 64.79ms | Not promoted. |
| `2060sprint_micro_muon_qk525_5k` | 8.3206 | 3.8187 | 65.64ms | Not promoted. |

Record: `records/sub4_micro_matrix_20260425_015448`

| Preset | Val loss | Val BPB | Step avg | Result |
|---|---:|---:|---:|---|
| `2060sprint_micro_muon_cooltaper5k_cold_tokens8k` | 8.1409 | 3.7362 | 102.46ms | Best in this screen. |
| `2060sprint_micro_muon_rownorm_wd5k` | 8.2361 | 3.7799 | 116.45ms | Implemented, not promoted. |
| `2060sprint_micro_muon_attngate5k` | 8.2875 | 3.8035 | 113.74ms | Implemented, not promoted. |
| `2060sprint_micro_muon_rownorm5k` | 8.2948 | 3.8069 | 115.98ms | Implemented, not promoted. |

Record: `records/sub4_micro_matrix_20260425_022316`

| Preset | Val loss | Val BPB | Step avg | Result |
|---|---:|---:|---:|---|
| `2060sprint_micro_muon_cooltaper5k_cold_tokens8k` | 8.2201 | 3.6897 | 64.04ms | Best in the new 1k current-code screen. |
| `2060sprint_micro_muon_sparsegate5k` | 8.2759 | 3.7147 | 65.67ms | Exact public sparse gate works, but is not promoted yet. |
| `2060sprint_micro_muon_huberwd5k` | 8.3150 | 3.7323 | 58.82ms | Faster in this run but lower quality; retune WD before promotion. |
| `2060sprint_micro_muon_smear_scalar5k` | 8.3223 | 3.7356 | 65.25ms | Exact scalar smear works, but is not promoted yet. |
| `2060sprint_micro_muon_smear_sparse5k` | 8.3740 | 3.7588 | 62.44ms | Stack hurt at 1k; do not stack by default. |

### LQER and frozen carry wiring

The new export/carry candidates are now code-valid and included in the repeatable
matrix:

- `2060sprint_micro_muon_lqer5k`
- `2060sprint_micro_muon_fcarry5k`
- `2060sprint_micro_muon_fcarry_lqer5k`
- `2060sprint_micro_muon_publicstack5k`

Smoke checks:

| Check | Result |
|---|---|
| d768/e256 fcarry+LQER probe | `total_submission_bytes=1441930`, `headroom=2558070` |
| strict dequant/load | `lqer_tensors=8`, `sidecar_modules=8`, no missing/unexpected keys |
| one-step CaseOps artifact round trip | `total_submission_bytes=1823342`, `headroom=2176658`, `final_val_bpb=4.0472` |

These are not quality promotions yet. They only prove the code path is real and
under cap; the next required evidence is a 5k or 600s comparison.

Record: `records/sub4_micro_matrix_20260425_014426`

| Ablation | Val loss | Val BPB | Step avg | Result |
|---|---:|---:|---:|---|
| BigramHash 16k x 128 side channel | 8.2947 | 3.8068 | 65.43ms | Not promoted. |

Record: `records/sub4_micro_matrix_20260425_013633`

| Preset | Normal Val BPB | Final TTT Val BPB | Step avg | Result |
|---|---:|---:|---:|---|
| `2060sprint_micro_muon_ttt_control5k` | 2.6976 | 2.6834 | 58.82ms | TTT helps itself but does not beat the lead. |

## Current Recommendation

For local 2060 iteration, keep the cold lane as the conservative baseline:

- `SUB4_PROFILE=i1l2r2_d768_e256_h12kv1_mlpinner_mlp075`
- `SUB4_SPEED_PRESET=2060sprint_micro_muon_cooltaper5k_cold_tokens8k`

The new export-aware 1k matrix has made
`2060sprint_micro_muon_publicstack_lqerio5k` the next promotion candidate:
final export BPB `3.6378` versus `3.7331` for the cold baseline, with both under
3MB total artifact+code size. Do not call it the default until it wins a 5k or
600-second wall-clock run, but it is no longer just a theoretical public-knob
stack.

For H100/high-throughput experiments, keep the same cold taper and move up the
capacity ladder while staying under the decimal cap:

- `i1l2r2_d1280_e320_h20kv1_mlpinner_mlp050`
- `i1l2r2_d1536_e384_h24kv1_mlpinner_mlp050`
- `i1l2r2_d2048_e512_h32kv1_mlpinner_mlp025`
