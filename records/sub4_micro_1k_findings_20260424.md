# Sub-4MB Micro/Nano 1k Findings

Date: 2026-04-24

## Runner Changes

- `scripts/run_sub4_micro_matrix.py` can now skip probe/bench, run explicit `profile:preset` train pairs, parse `nan`/`inf`, record timeouts as rows, and provide CUDA 12.6 plus MSVC/ninja paths to child processes.
- `train_gpt_ternary.py` now has lower-LR `*_stable` presets and disables code snapshots by default for the ternary wrapper.

## 1k Winners

All rows used `ITERATIONS=1000`, `VAL_TOKENS_LIMIT=16384`, local 1024-token proxy data, `SKIP_FINAL_ARTIFACTS=1`.

| Rank | Profile | Preset | Val loss | Val bpb | Step avg | Peak MiB | Notes |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | `i1l2r2_d96_e48_h3mha_mlpinner_mlp2` | `2060sprint_micro_tokens_stable` | 5.5618 | 3.2538 | 15.40 ms | 334 | Best quality, still fast. |
| 2 | `i1l2r2_d96_e48_h3mha_mlpinner_mlp2` | `2060sprint_micro_stable` | 5.5785 | 3.2636 | 13.21 ms | 107 | Best speed/quality trade. |
| 3 | `i1l2r2_d128_e64_h4mha_mlpinner_mlp2` | `2060sprint_micro_tokens_stable` | 5.9798 | 3.4984 | 18.92 ms | 431 | Best larger candidate so far. |
| 4 | `i1l2r2_d128_e64_h4mha_mlpinner_mlp2` | `2060sprint_micro_stable` | 6.5629 | 3.8395 | 12.87 ms | 131 | Faster but clearly worse than token-stable. |
| 5 | `i1l2r2_d256_e96_h4mha_mlpinner` | `2060sprint_micro_stable` | 6.7220 | 3.9326 | 12.56 ms | 298 | Stable at lower LR, but not worth the size yet. |

## Rejected For Now

- `*_dense`: CUDA dense materialization works after fixing MSVC/ninja PATH, but JIT/load overhead is too high for the timed lane, and larger dense variants still timeout or crash.
- `*_throughput_stable`: unstable on the tested 2060 Super path. The d96 throughput run hit CUDA invalid argument; d128 throughput timed out.
- fp16 gradient clipping: caused `nan` runs across the board. The stable preset now uses lower LR without clipping.
- Larger original-LR profiles: d192/d224/d256 frequently explode to `nan` or CUDA errors in the first 10 steps.

## Current Recommendation

Use the d96 profile as the primary sub-4MB lane:

- Fast lane: `SUB4_PROFILE=i1l2r2_d96_e48_h3mha_mlpinner_mlp2`, `SUB4_SPEED_PRESET=2060sprint_micro_stable`
- Quality lane: `SUB4_PROFILE=i1l2r2_d96_e48_h3mha_mlpinner_mlp2`, `SUB4_SPEED_PRESET=2060sprint_micro_tokens_stable`
- Challenger lane: `SUB4_PROFILE=i1l2r2_d128_e64_h4mha_mlpinner_mlp2`, `SUB4_SPEED_PRESET=2060sprint_micro_tokens_stable`

Probe from `records/sub4_micro_matrix_20260424_164053/probe_selected.md` estimates all these profiles comfortably under the decimal 4,000,000 byte cap with 8192 vocab. The d96 profile had an estimated compressed artifact around 429,615 bytes with about 3,570,385 bytes of headroom.

## Data Sources

- `records/sub4_micro_matrix_20260424_165128/train.csv`
- `records/sub4_micro_matrix_20260424_170712/train.csv`
- `records/sub4_micro_matrix_20260424_172630/train.csv`
- `records/sub4_micro_matrix_20260424_173529/train.csv`
- `records/sub4_micro_matrix_20260424_164053/probe_selected.md`
