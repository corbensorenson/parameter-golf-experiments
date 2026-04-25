# Sub-4MB vs 16MB Gap Findings

Date: 2026-04-24

## Bottom Line

The sub-4MB family is training in the intended low-precision ternary path, but
the prior comparison against the 16MB CaseOps lane was not apples-to-apples.
The 16MB lane has a much stronger data/eval path, a much larger model, much
larger token batches, longer context, and a final TTT phase.

The sub-4MB lane should not be judged by the 1024-token proxy loss against the
16MB CaseOps loss. When moved to the same 8192 CaseOps data path, the current
d384 lane is still far behind at 1k, which points to real capacity/schedule
gaps after the metric mismatch is removed.

## What Is Correct

- Train-time ternary is active: `TRAIN_TERNARY_BLOCKS=1`, `ternary_linear_count=8`.
- The promoted fast trunk trains from low precision: `params=float16`,
  `train_ternary_param_dtype=model`, `train_casted_linear_param_dtype=model`.
- Factored tied embeddings are active: `tie_embeddings=True`,
  `factored_embed_dim=128` on d384 and larger ranks on new near-cap profiles.
- Export settings are on the sub-4 path: `MODEL_CODEC=lzma`,
  `SUBMISSION_SIZE_CAP_BYTES=4000000`, ternary shrinkage fix enabled.
- Host-side training path is already avoiding obvious transfer overhead:
  pinned host memory, prefetching, and persistent batch buffers are enabled.

## Why 16MB Is Doing Better

### 1. Different tokenizer/data/eval

Archived 16MB CaseOps uses:

- `fineweb10B_sp8192_lossless_caps_caseops_v1_reserved`
- tokenizer `fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model`
- `val_bpb:byte_sidecar:enabled`
- eval sequence length 2048

Most sub-4 matrix results were on:

- `fineweb10B_sp1024`
- tokenizer `fineweb_1024_bpe.model`
- no byte sidecar
- sequence length 64

Raw loss numbers across those are not directly comparable.

### 2. Token budget per wall-clock is not close

The 16MB archive trains with:

- `world_size=8`
- `train_batch_tokens=786432`
- about 4.9k train steps before the wall-clock cap

That is roughly 3.8B tokens seen during the run.

The current sub-4 d384 2060 wall-clock run used:

- `train_batch_tokens=4096`
- about 27.8k train steps

That is about 114M tokens. More steps do not make up for a 30x+ token exposure
gap on the local 2060.

### 3. The 16MB model has far more capacity and late eval adaptation

The 16MB CaseOps model:

- `model_params:35944602`
- 11 layers, width 512, MLP mult 4
- XSA across all 11 layers
- GQA with 4 KV heads
- skip gates and looping
- GPTQ/int6-int7 export
- final quantized phased TTT, improving final score

The promoted d384 sub-4 model on CaseOps:

- `model_params:2366088`
- 3 unique blocks, effective depth 6
- only block 0 has attention, looped middle blocks are MLP-only
- no TTT
- only e128 factored tied IO

## Fixes Added In This Pass

### Guarded quality preset

Added `2060sprint_micro_muon_guarded`, plus seq128 and wall-clock variants.
This restores cheap HRC/quality controls that sprint presets had disabled:

- `LOGIT_SOFTCAP=30`
- `LOSS_FP32=1`
- pass embeddings
- pass roles
- loop index
- recurrent injection
- fp32 tiny control params

On the 1024 proxy at 1k, guarded improved d384:

| Profile | Preset | Val loss | Step avg |
|---|---|---:|---:|
| d384/e128 | damped full | 4.8018 | 26.91ms |
| d384/e128 | guarded | 4.5543 | 35.91ms |
| d384/e128 | guarded seq128 | 4.7483 | 33.06ms |

On 8192 CaseOps at 1k, guarded did not help:

| Profile | Preset | Val loss | Val BPB | Step avg |
|---|---|---:|---:|---:|
| d384/e128 | damped full | 8.0221 | 3.6817 | 37.61ms |
| d384/e128 | guarded | 8.1241 | 3.7285 | 49.86ms |

Keep guarded as a proxy-quality candidate, but do not promote it for CaseOps yet.

### Sampled vocab presets

Added `2060sprint_micro_muon_damped_full_vsample2k` and
`2060sprint_micro_muon_damped_full_tokens8k_vsample2k`.

They reduced step time on CaseOps, but quality was much worse:

| Profile | Preset | Val loss | Val BPB | Step avg |
|---|---|---:|---:|---:|
| d384/e128 | exact full vocab | 8.0221 | 3.6817 | 37.61ms |
| d384/e128 | vsample2k | 8.9609 | 4.1125 | 34.62ms |
| d384/e128 | 8k tokens + vsample2k | 9.1879 | 4.2167 | 43.17ms |

Do not promote sampled vocab for this lane.

### Near-cap profiles

Added wider shallow-HRC candidates that finally spend meaningful byte budget:

| Profile | Params | Estimated total bytes | Headroom |
|---|---:|---:|---:|
| `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` | 6.43M | 1.39MB | 2.61MB |
| `i1l2r2_d1280_e320_h20kv1_mlpinner_mlp050` | 11.81M | 2.14MB | 1.86MB |
| `i1l2r2_d1536_e384_h24kv1_mlpinner_mlp050` | 16.33M | 2.76MB | 1.24MB |
| `i1l2r2_d2048_e384_h32kv1_mlpinner_mlp025` | 19.68M | 3.23MB | 0.77MB |
| `i1l2r2_d1792_e448_h28kv1_mlpinner_mlp050` | 21.57M | 3.49MB | 0.51MB |
| `i1l2r2_d2048_e512_h32kv1_mlpinner_mlp025` | 21.25M | 3.65MB | 0.35MB |

The d2048/e512 profile is the first serious sub-4 candidate by capacity: about
21M params under the decimal cap.

### Cooler long-taper presets

Added `2060sprint_micro_muon_cooltaper5k`, cold variants, and wall-clock
variants. The best local quality run so far uses the cold LR levels with exact
full-vocab 8k token batches:

- `TIED_EMBED_LR=0.0003`
- `MATRIX_LR=0.0004`
- `SCALAR_LR=0.0004`
- `TRAIN_BATCH_TOKENS=8192`
- `VAL_BATCH_SIZE=8192`
- `WARMDOWN_ITERS=5000` for the 5k screen, `5200` for the 600s wall-clock preset

The default d768 5k CaseOps run went non-finite. Cool taper fixed stability:

| Profile | Preset | Val loss | Val BPB | Step avg |
|---|---|---:|---:|---:|
| d768/e256 | damped full, 1k | 8.4388 | 3.8729 | 52.37ms |
| d768/e256 | damped full, 5k | NaN | NaN | 51.72ms |
| d768/e256 | cooltaper5k, 5k | 6.7525 | 3.0990 | 51.32ms |

The colder taper was the real rescue on the 8192 CaseOps path:

| Profile | Preset | Steps / cap | Val loss | Val BPB | Step avg |
|---|---|---:|---:|---:|---:|
| d768/e192 | cooltaper5k, cold LR | 5k | 5.9143 | 2.7143 | 51.35ms |
| d768/e192 | cold wall-clock, 12k taper | 600s / 12,401 | 5.8398 | 2.6801 | 48.39ms |
| d768/e256 | cooltaper5k, cold LR | 5k | 6.0603 | 2.7814 | 39.11ms |
| d768/e256 | cold wall-clock, 16k taper | 600s / 11,485 | 5.8030 | 2.6632 | 52.26ms |
| d768/e256 | cold 8k tokens, exact vocab | 5k / 574s train | 5.7516 | 2.6397 | 114.87ms |
| d768/e256 | cold 8k tokens, wall-clock 5.2k taper | 600s / 6,111 | 5.7986 | 2.6612 | 98.22ms |
| d768/e256 | competitor-meta: QK 5.25 + Polar NS + minLR 0.1 | 5k | 5.9444 | 2.7281 | 66.12ms |
| d896/e256 | cooltaper5k, cold LR | 5k | 6.1573 | 2.8259 | 64.54ms |

Equal-step d768/e192 is better than d768/e256 at 4k tokens, but d768/e256 wins
in the real 10-minute local lane once it uses the colder schedule and then
wins again with exact 8k-token batches at a fixed 5k steps. The 8k-token run
has lower token/sec, but better loss/BPB inside the wall-clock budget. Letting
that 8k-token preset continue to the full 600s cap was worse than stopping at
5k, so the promoted 8k lane is fixed-iteration, not "run until cap".

The public-leaderboard-inspired stacked preset improved the 1k screen but hurt
the 5k score. Keep `LR_MIN_SCALE`, `MUON_NS_VARIANT=polar_express`, and QK 5.25
as ablation knobs, but do not promote the stacked `competitor_meta5k` preset.

d896/e256 is not a local 2060 promotion candidate: it is slower and worse than
d768/e256 at the same 5k cold schedule.

The attempted d384/e128 5k cool-taper CaseOps baseline timed out externally
without usable stdout after the earlier CUDA invalid-argument failure. Treat
that run as infrastructure noise until rerun in a fresh CUDA process/session.

### Repeatable CaseOps runner

Added `scripts/run_sub4_caseops_wide_matrix.py` so the current lane can be
rerun without hand-typing the long CaseOps dataset/tokenizer paths:

- `--mode local-5k` runs the d768/e256 8k-token lead plus the d768/e192, d768/e256, and d896/e256 cold 5k controls.
- `--mode local-wallclock` runs the current 600s d768/e192 and d768/e256 lanes.
- `--mode h100-probe` writes the size probe for d1280/e320, d1536/e384, and d2048/e512.

Also fixed `scripts/run_sub4_micro_matrix.py` so profile probes pass the
selected profile list into `probe_sub4_profiles.py` instead of probing the
entire registry and filtering afterward.

Added generic competitor-derived knobs:

- `LR_MIN_SCALE`
- `MUON_NS_VARIANT=polar_express`

They are implemented and logged, but remain experimental for sub-4.

## 2026-04-25 Online-Trick Audit Update

The public PR audit is now reflected in code rather than just notes.

Implemented in `train_gpt.py`:

- exact PR #1787 Polar Express Muon coefficients as `MUON_NS_VARIANT=polar_express`
- exact PR #1809 Turbo/Gram Polar Muon coefficients as `MUON_NS_VARIANT=gram_polar`
- Gram-NS dispatch for rectangular Muon matrices when aspect ratio is >= 1.5
- public min-LR warmdown floor as `LR_MIN_SCALE`
- public row-normalized Muon option as `MUON_ROW_NORMALIZE`
- public decoupled Muon weight decay option as `MUON_WEIGHT_DECAY` / `MUON_WD`

Implemented in the sub-4 preset registry:

- one-at-a-time ablations for min-LR, Polar, Gram-Polar, QK 5.25,
  row-normalized Muon, row-normalized Muon plus weight decay, tiny
  attention-output gate, score-first TTT, and a stacked competitor-meta preset

Current-code 5k audit:

| Profile | Preset | Val loss | Val BPB | Step avg |
|---|---|---:|---:|---:|
| d768/e256 | cold 8k tokens | 5.7789 | 2.6522 | 82.45ms |

Current-code 1k public-knob screen:

| Preset | Val loss | Val BPB | Step avg |
|---|---:|---:|---:|
| cold 8k tokens | 8.1409 | 3.7362 | 102.46ms |
| rownorm + Muon WD | 8.2361 | 3.7799 | 116.45ms |
| tiny attn gate | 8.2875 | 3.8035 | 113.74ms |
| rownorm only | 8.2948 | 3.8069 | 115.98ms |

The public knobs are useful to have, but they do not replace the current lead
for this sub-4 HRC family. The promoted local lane remains d768/e256 cold 8k
tokens. The best older 5k record is still 2.6397 BPB, while the cleaner
current-code audit record is 2.6522 BPB with fuller optimizer logging.

### 2026-04-25 leader-derived gate/decay screen

Added exact public-style scalar SmearGate, exact public-style sparse attention
gate, and Huber Muon decay as opt-in presets. All train-path smoke tests passed,
but the 1k CaseOps screen still favors the current lead.

Record: `records/sub4_micro_matrix_20260425_022316`

| Profile | Preset | Val loss | Val BPB | Step avg |
|---|---|---:|---:|---:|
| d768/e256 | cold 8k tokens | 8.2201 | 3.6897 | 64.04ms |
| d768/e256 | exact sparse attention gate | 8.2759 | 3.7147 | 65.67ms |
| d768/e256 | Huber Muon WD | 8.3150 | 3.7323 | 58.82ms |
| d768/e256 | exact scalar smear | 8.3223 | 3.7356 | 65.25ms |
| d768/e256 | scalar smear + sparse gate | 8.3740 | 3.7588 | 62.44ms |

These are available for retuning, but they are not promoted defaults. The big
remaining gap versus the public 16MB leaders is now export/capacity/TTT, not
whether these small neural controls exist in the code.

## Current Interpretation

The sub-4 family is not "cheating" or forgetting ternary training. The big
miss was evaluating and tuning it on a much easier/faster proxy while comparing
against a full CaseOps/TTT 16MB lane. Once we run CaseOps directly, the current
d384 lane is too small and too token-starved.

The next serious lane is not the old d96/d192 nano family. It is a wide shallow
HRC profile with a cooler long taper on the 8192 CaseOps data. For local 2060
experiments, the current lead is fixed 5k d768/e256 with exact 8k-token batches
via `2060sprint_micro_muon_cooltaper5k_cold_tokens8k`. Keep d768/e192 as the
smaller control because it is better per step and very close at 4k-token
batches.

For H100 experiments, test d1536/e384 and d2048/e512 under the same cold taper,
then retune warmdown to match the number of train steps actually reached in the
10-minute cap.

## Recommended Next Matrix

Local 2060:

- fixed-iteration lead: `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` + `2060sprint_micro_muon_cooltaper5k_cold_tokens8k`
- `i1l2r2_d768_e192_h12kv1_mlpinner_mlp050` + `2060sprint_micro_muon_cooltaper_cold_wallclock`
- 4k-token control: `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` + `2060sprint_micro_muon_cooltaper_cold_wallclock16k`
- optional discard/control only: `i1l2r2_d896_e256_h14kv1_mlpinner_mlp050` + `2060sprint_micro_muon_cooltaper5k_cold`

H100 / high-throughput:

- `i1l2r2_d1280_e320_h20kv1_mlpinner_mlp050`
- `i1l2r2_d1536_e384_h24kv1_mlpinner_mlp050`
- `i1l2r2_d2048_e512_h32kv1_mlpinner_mlp025`

Use the 8192 CaseOps data path for these comparisons.

## Key Records

- `records/sub4_micro_matrix_20260424_221717`
- `records/sub4_micro_matrix_20260424_221931`
- `records/sub4_micro_matrix_20260424_222222`
- `records/sub4_micro_matrix_20260424_222703`
- `records/sub4_micro_matrix_20260424_223009`
- `records/sub4_micro_matrix_20260424_223639`
- `records/sub4_micro_matrix_20260424_233206`
- `records/sub4_micro_matrix_20260424_233739`
- `records/sub4_micro_matrix_20260424_235003`
- `records/sub4_micro_matrix_20260424_235411`
- `records/sub4_micro_matrix_20260425_001116`
- `records/sub4_micro_matrix_20260425_001851`
- `records/sub4_micro_matrix_20260425_003032`
- `records/sub4_micro_matrix_20260425_010844`
- `records/sub4_micro_matrix_20260425_011015`
- `records/sub4_micro_matrix_20260425_011144`
- `records/sub4_micro_matrix_20260425_012313`
- `records/sub4_micro_matrix_20260425_012815`
- `records/sub4_micro_matrix_20260425_013633`
- `records/sub4_micro_matrix_20260425_014426`
- `records/sub4_micro_matrix_20260425_015448`
- `records/sub4_micro_matrix_20260425_020306`
- `records/sub4_micro_matrix_20260425_022316`
- `records/sub4_caseops_wide_h100-probe_20260425_001742`
- `records/parameter_golf_competitor_research_20260425.md`
- `records/sub4_competitive_plan_20260425.md`
