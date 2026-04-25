# Sub-4MB Quality/Speed Findings

Date: 2026-04-24

## New Profiles Added

Added middle-depth d96/d128 profiles to test whether the looped-middle idea improves quality without jumping to the unstable wide family:

- `i1l2r3_d96_e48_h3mha_mlpinner_mlp2`
- `i1l3r2_d96_e48_h3mha_mlpinner_mlp2`
- `i2l3r2_d96_e48_h3mha_mlpinner_mlp2`
- `i1l2r3_d128_e64_h4mha_mlpinner_mlp2`
- `i1l3r2_d128_e64_h4mha_mlpinner_mlp2`
- `i2l3r2_d128_e64_h4mha_mlpinner_mlp2`

Probe estimates remained comfortably under the decimal 4,000,000 byte cap. The new compressed-model estimates were roughly 436KB to 508KB before code bytes.

## Main Result

More loop depth did not beat the original d96 winner at 1k. The bigger gain came from correcting LR/warmdown:

- Previous best 1k: d96 `2060sprint_micro_tokens_stable`, val `5.5618`, `15.40ms/step`
- New best 1k: d96 fast lane with flat low LR, val `5.1192`, `13.18ms/step`
- New 10k check: d96 fast lane with 1k final warmdown, val `4.3464`, `13.00ms/step`

## Promoted Preset

Added `2060sprint_micro_cool1k` in `train_gpt_ternary.py`:

- `TRAIN_BATCH_TOKENS=4096`
- `TIED_EMBED_LR=0.0015`
- `MATRIX_LR=0.001`
- `SCALAR_LR=0.001`
- `WARMDOWN_ITERS=1000`

This is now the primary speed/quality lane:

```powershell
$env:SUB4_PROFILE = "i1l2r2_d96_e48_h3mha_mlpinner_mlp2"
$env:SUB4_SPEED_PRESET = "2060sprint_micro_cool1k"
```

## 1k Shape Sweep

| Profile | Preset | Val loss | Step avg | Notes |
|---|---|---:|---:|---|
| `i1l3r2_d96_e48_h3mha_mlpinner_mlp2` | `2060sprint_micro_tokens_stable` | 5.7455 | 18.63ms | Best deeper shape, still worse than original d96. |
| `i1l2r3_d96_e48_h3mha_mlpinner_mlp2` | `2060sprint_micro_tokens_stable` | 5.9973 | 18.51ms | Extra repeats hurt speed/quality. |
| `i2l3r2_d96_e48_h3mha_mlpinner_mlp2` | `2060sprint_micro_tokens_stable` | 6.0106 | 27.23ms | Too slow for the quality. |
| `i1l2r3_d128_e64_h4mha_mlpinner_mlp2` | `2060sprint_micro_tokens_stable` | 6.0353 | 22.60ms | d128 still not winning. |
| `i2l3r2_d128_e64_h4mha_mlpinner_mlp2` | `2060sprint_micro_tokens_stable` | 6.0355 | 33.60ms | Too slow. |

## LR/Warmdown Sweep

| Candidate | Overrides | Val loss | Step avg |
|---|---|---:|---:|
| d96 fast | `WARMDOWN_ITERS=0,TIED_EMBED_LR=0.001,MATRIX_LR=0.00075,SCALAR_LR=0.00075` | 5.1192 | 13.18ms |
| d96 fast | `WARMDOWN_ITERS=0,TIED_EMBED_LR=0.0015,MATRIX_LR=0.001,SCALAR_LR=0.001` | 5.1647 | 13.18ms |
| d96 tokens | `WARMDOWN_ITERS=0,TIED_EMBED_LR=0.0015,MATRIX_LR=0.001,SCALAR_LR=0.001` | 5.2666 | 15.67ms |
| d96 tokens | `WARMDOWN_ITERS=0,TIED_EMBED_LR=0.002,MATRIX_LR=0.0015,SCALAR_LR=0.0015` | 5.4485 | 15.62ms |

The 5k follow-up reversed the top two LR settings:

| Candidate | Val loss | Step avg |
|---|---:|---:|
| d96 fast, LR `0.0015/0.001/0.001`, no warmdown | 4.6231 | 12.95ms |
| d96 fast, LR `0.001/0.00075/0.00075`, no warmdown | 4.6729 | 13.01ms |

The 10k follow-up with `WARMDOWN_ITERS=1000` landed at val `4.3464`, bpb `2.5428`, `13.00ms/step`.

## Current Recommendation

Do not spend the next run on wider or deeper shapes yet. Use `i1l2r2_d96_e48_h3mha_mlpinner_mlp2` plus `2060sprint_micro_cool1k`, then tune the long-run schedule around it:

- test 20k-40k steps with `MAX_WALLCLOCK_SECONDS=600`
- compare `WARMDOWN_ITERS=1000`, `2000`, and `4000`
- only revisit deeper d96 if long-run d96 plateaus early

## Data Sources

- `records/sub4_micro_matrix_20260424_174617/train.csv`
- `records/sub4_micro_matrix_20260424_175634/train.csv`
- `records/sub4_micro_matrix_20260424_175736/train.csv`
- `records/sub4_micro_matrix_20260424_175840/train.csv`
- `records/sub4_micro_matrix_20260424_175941/train.csv`
- `records/sub4_micro_matrix_20260424_180125/train.csv`
- `records/sub4_micro_matrix_20260424_180206/train.csv`
- `records/sub4_micro_matrix_20260424_180245/train.csv`
- `records/sub4_micro_matrix_20260424_180355/train.csv`
- `records/sub4_micro_matrix_20260424_180506/train.csv`
- `records/sub4_micro_matrix_20260424_180636/train.csv`
