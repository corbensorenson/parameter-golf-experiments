# Sub-4MB Quality Rescue Findings

Date: 2026-04-24

## Why The Prior Nano Lane Was Not Enough

The d96 nano lane was fast, but it was much too small. On the local 1024-vocab proxy, `i1l2r2_d96_e80_h3mha_mlpinner_mlp2` has only 245k trainable parameters, so the 10-minute run was never likely to approach the 16MB lane.

Three misses showed up:

- We were under-spending the 4MB cap. The d96/e80 model probes at only about 503KB total estimated submission size.
- AdamW was worse than the hybrid Muon optimizer for quality. Muon is slower, but it moves the loss curve meaningfully.
- `WARMUP_STEPS` was only a reset-after-warmup runtime warmup, not learning-rate warmup. Added `LR_WARMUP_ITERS` for real LR ramping so larger models can be stabilized in future tests.

## Best Current Sub-4 Candidate

Promoted quality lane:

```powershell
$env:SUB4_PROFILE = "i1l2r2_d192_e80_h3mha_mlpinner_mlp15"
$env:SUB4_SPEED_PRESET = "2060sprint_micro_muon_quality"
```

This preset uses:

- `OPTIMIZER_PRESET=hybrid`
- `TIED_EMBED_LR=0.002`
- `MATRIX_LR=0.003`
- `SCALAR_LR=0.003`
- `MUON_BACKEND_STEPS=5`

## Key Results

| Candidate | Horizon | Val loss | Val bpb | Step avg |
|---|---:|---:|---:|---:|
| d96/e80 AdamW | 10k | 4.2539 | 2.4887 | 13.13ms |
| d96/e80 Muon | 10k | 4.0653 | 2.3783 | 21.35ms |
| d192/e80/mlp1.5 Muon | 10k | 3.8508 | 2.2529 | 21.25ms |
| d192/e80/mlp1.5 Muon | 600s wall-clock | 3.7322 | 2.1835 | 21.22ms |

The 600s wall-clock run reached step 28,270.

## Negative Results

Restoring attention to the d96 looped middle did not solve quality:

- d96/e80 full inner, AdamW, 1k: 5.2443 at 21.04ms
- d96/e80 attention-only inner, AdamW, 1k: 5.3876 at 16.92ms
- d96/e80 attention-only inner, Muon, 1k: 5.1792 at 25.51ms

Bigram and VE byte-spend lanes did not beat plain e80, and some larger bigram variants triggered Windows CUDA teardown failures.

Large d384/d512 HRC bodies fit the 4MB cap by probe, but are not yet stable/useful in the sprint preset:

- `i2l3r2_d384_e128` probes at about 1.26MB total estimated submission size.
- `i2l3r2_d512_e128_mlp2` probes at about 1.50MB total estimated submission size.
- With sprint LR and no softcap/fp32-loss restoration, they go non-finite.
- With fp32 latent params and very low LR, d384 is stable but underlearns: 1k val around 6.0 at about 58ms/step.

## Next Experiment

The current best sub-4 path is not the tiny d96 lane. It is a medium-width ternary HRC with Muon. To push toward the 16MB loss range, the next round should build a stable "large sub-4" preset:

- keep `LOGIT_SOFTCAP=30` and `LOSS_FP32=1` for d384/d512
- use `LR_WARMUP_ITERS=200-1000`
- use fp32 latent ternary params only if needed for stability
- search LR below the explosive sprint range but above the underlearning safe range
- target d384 first, not d512, because d384 is less memory-bound on the 2060 Super

## Data Sources

- `records/sub4_micro_matrix_20260424_184400/train.csv`
- `records/sub4_micro_matrix_20260424_191732/train.csv`
- `records/sub4_micro_matrix_20260424_192749/train.csv`
- `records/sub4_micro_matrix_20260424_194425/train.csv`
- `records/sub4_micro_matrix_20260424_195539/train.csv`
- `records/sub4_micro_matrix_20260424_201319/train.csv`
