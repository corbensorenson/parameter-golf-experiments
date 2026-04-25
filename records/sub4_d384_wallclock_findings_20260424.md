# Sub-4MB D384 Wall-Clock Findings

Date: 2026-04-24

## Promoted Lane

Profile:

```powershell
$env:SUB4_PROFILE = "i1l2r2_d384_e128_h8kv1_mlpinner_mlp10"
```

Wall-clock preset:

```powershell
$env:SUB4_SPEED_PRESET = "2060sprint_micro_muon_wallclock"
```

This lane uses:

- d384 model width, e128 factored tied IO, 8 query heads, 1 KV head
- 3 unique HRC blocks, effective route depth 6: `0,1,2,1,2,0`
- full attention only on block 0, MLP-only looped middle blocks `1,2`
- train-time ternary trunk from the start, fp16 latent params, group size 256
- Muon trunk optimizer with Adam for scalar/vector and tied IO params
- LR warmup 200 steps, depth scale init 0.25 to 0.75, QK gain 1.0
- full loss stride 1
- long wall-clock warmdown: `WARMDOWN_ITERS=10000`

Probe estimate with vocab 8192:

| Profile | Estimated total bytes | Headroom |
|---|---:|---:|
| `i1l2r2_d384_e128_h8kv1_mlpinner_mlp10` | 745,721 | 3,254,279 |

## Why This Replaces D192

The earlier best stable lane was:

| Candidate | Horizon | Val loss | Val bpb | Step avg |
|---|---:|---:|---:|---:|
| d192/e80/mlp1.5 Muon | 10k | 3.8508 | 2.2529 | 21.25ms |
| d192/e80/mlp1.5 Muon | 600s | 3.7322 | 2.1835 | 21.22ms |

The new d384/e128 damped lane improves both fixed-step and wall-clock quality:

| Candidate | Horizon | Val loss | Val bpb | Step avg |
|---|---:|---:|---:|---:|
| d384/e128 damped full-loss | 10k | 3.6514 | 2.1362 | 22.44ms |
| d384/e128 wall-clock warmdown | 600s | 3.5042 | 2.0501 | 21.59ms |

The 600-second run reached step 27,793 and stayed finite.

## Important Negative Results

- The same 600-second d384 full-loss lane with only `WARMDOWN_ITERS=1000` went non-finite by final validation at step 27,119.
- The d384 lane is finite at 15k with the shorter warmdown, so the failure is a long-run schedule problem, not an immediate architecture failure.
- `i1l2r2_d512_e128_h8kv1_mlpinner_mlp075` is a strong fixed-step challenger: 5k val 3.7827 and 10k val 3.6167. It is not promoted for the 10-minute lane yet because the 600-second run with `WARMDOWN_ITERS=10000` went non-finite at final validation, and the stricter 15k warmdown attempt did not produce a valid completed row before the external timeout.
- `i1l2r2_d512_e160_h8kv1_mlpinner_mlp10` hit a CUDA invalid-argument failure during the 5k follow-up and should not be used until that shape is debugged.
- `TRAIN_TERNARY_DENSE_KERNEL=1` was slower on this 2060 Super path: 1k step avg 23.58ms versus about 21.8ms for the plain cached STE path.
- `TRAIN_TERNARY_PACKED_KERNEL=1` is not competitive yet: 1k step avg 110.21ms.
- Larger batch tokens were worse in the 1k screen. `TRAIN_BATCH_TOKENS=8192` increased memory and hurt early validation.

## Data Sources

- `records/sub4_micro_matrix_20260424_202756/train.csv`
- `records/sub4_micro_matrix_20260424_203803/train.csv`
- `records/sub4_micro_matrix_20260424_204946/train.csv`
- `records/sub4_micro_matrix_20260424_205358/train.csv`
- `records/sub4_micro_matrix_20260424_211548/train.csv`
- `records/sub4_micro_matrix_20260424_212148/train.csv`
- `records/sub4_micro_matrix_20260424_213500/train.csv`
- `records/sub4_micro_matrix_20260424_213801/train.csv`
- `records/sub4_micro_matrix_20260424_214134/train.csv`
- `records/sub4_micro_matrix_20260424_214540/train.csv`
- `records/sub4_micro_matrix_20260424_215608/train.csv`

## Next Experiments

- Tune the wall-clock warmdown around 8k, 10k, and 12k steps.
- Revisit d512/e128 mlp0.75 with lower matrix/scalar LR, not just longer warmdown.
- Try a slightly lower d384 matrix/scalar LR for long-run stability without needing as much warmdown.
- Test d448/e128 mlp1.0 with the damped full-loss preset; the size cap still has room, but d448 must justify the extra compute.
- If revisiting custom kernels, prioritize a faster dense materialization path that feeds Tensor Cores; the current packed matmul replacement is too slow.
