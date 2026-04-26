# Leader-Inspired Sub-4 / Soft-8 Lever Matrix

Date: 2026-04-26

## Why This Matrix Exists

The active quality-first i4/l9/r5 run is testing capacity, precision ladders,
LQER spend, and loop attention placement. While that runs, this matrix prepares
the next sweep: cheap public-leader control ideas applied to the best local
IO-tail shape instead of the older nano probes.

The accepted leaderboard is dominated by SP8192 transformer stacks with depth
recurrence, QK gain, parallel residuals, GPTQ/SDClip, and legal score-first TTT.
Recent open PR titles point at SmearGate, attention-output gating, LQER-style
quant repair, FLA/GDN pivots, and byte-level PPM mixtures. For this repo, the
lowest-risk next step is to test the cheap, already-implemented neural control
levers on our HRC/IO-tail family.

Sources:

- <https://github.com/openai/parameter-golf>
- <https://github.com/openai/parameter-golf/pull/1493>
- <https://github.com/openai/parameter-golf/pull/1790>
- <https://github.com/openai/parameter-golf/pull/1791>
- <https://github.com/openai/parameter-golf/pull/1797>
- <https://github.com/openai/parameter-golf/pull/1795>
- <https://github.com/openai/parameter-golf/issues/1604>

## Current Anchor

The current active run is `records/sub4-quality-first-i4-5k-20260426-034039`.
At the time the prime-loop candidates were added, five rows had completed:

| Candidate | Final BPB | Step Avg | Total Bytes |
|---|---:|---:|---:|
| `i4l9r5_d640e256_q16q8q8t_lqer_lidx_r8t16` | 2.4949 | 236.55ms | 6,259,569 |
| `i4l9r5_d768e320_q16q8q4t_lqer_lidx_r8t16` | 2.4962 | 273.78ms | 7,868,221 |
| `i4l9r5_d640e256_q16q8q4t_lqer_lidx_r8t16` | 2.4983 | 227.55ms | 5,863,841 |
| `i4l9r5_d640e224_q16q8q4t_lqer_lidx_r8t16` | 2.5074 | 228.77ms | 5,651,929 |
| `i4l9r5_d768e256_q16q8q4t_lqer_lidx_r8t16` | 2.5096 | 271.03ms | 7,455,817 |

The d640/e256 `q16/q8/q8/t` row is now the interim quality leader. The
d768/e320 `q16/q8/q4/t` row is close but slower and much larger. The next matrix
therefore tests most levers on d640/e256, keeps two d768/e320 confirmation
probes, and adds prime loop-width probes at l11 and l13.

## Candidate Group

Run with:

```powershell
python scripts/run_sub4_iotail_quant_matrix.py --candidate-group sub4_leader_levers --list
```

Candidates:

| Candidate | Lever |
|---|---|
| `i4l9r5_d640e256_q16q8q4t_qk500_lqer_lidx_r8t16` | QK gain 5.0 |
| `i4l9r5_d640e256_q16q8q4t_qk525_lqer_lidx_r8t16` | QK gain 5.25 |
| `i4l9r5_d640e256_q16q8q4t_qk550_lqer_lidx_r8t16` | QK gain 5.5 |
| `i4l9r5_d640e256_q16q8q4t_smear_lqer_lidx_r8t16` | scalar SmearGate |
| `i4l9r5_d640e256_q16q8q4t_attnout24_lqer_lidx_r8t16` | attention-output gate width 24 |
| `i4l9r5_d640e256_q16q8q4t_sparsegate_lqer_lidx_r8t16` | sparse attention gate |
| `i4l9r5_d640e256_q16q8q4t_huberwd095_lqer_lidx_r8t16` | Huber Muon WD 0.095 |
| `i4l9r5_d640e256_q16q8q4t_parres4_lqer_lidx_r8t16` | parallel residual last 4 virtual layers |
| `i4l9r5_d640e256_q16q8q4t_parres8_lqer_lidx_r8t16` | parallel residual last 8 virtual layers |
| `i4l9r5_d640e256_q16q8q4t_fcarry_lqer_lidx_r8t16` | frozen carry on repeated blocks 4,5,6 |
| `i4l9r5_d640e256_q16q8q4t_tttctrl005u24_lqer_lidx_r8t16` | score-first control TTT, LR 0.005, 24 updates |
| `i4l9r5_d640e256_q16q8q4t_qk525_smear_lqer_lidx_r8t16` | QK 5.25 plus scalar SmearGate |
| `i4l9r5_d640e256_q16q8q4t_qk525_attnout24_lqer_lidx_r8t16` | QK 5.25 plus attention-output gate |
| `i4l9r5_d640e256_q16q8q4t_qk525_huberwd_lqer_lidx_r8t16` | QK 5.25 plus Huber Muon WD |
| `i4l9r5_d640e256_q16q8q4t_publicsafe_lqer_lidx_r8t16` | conservative stacked row: QK, SmearGate, Huber WD, parres4, frozen carry |
| `i4l9r5_d768e320_q16q8q4t_qk525_lqer_lidx_r8t16` | wider-shape QK 5.25 confirmation |
| `i4l9r5_d768e320_q16q8q4t_smear_lqer_lidx_r8t16` | wider-shape SmearGate confirmation |
| `i4l11r5_d640e256_q16q8q4t_lqer_lidx_r8t16` | prime loop-width baseline: l11/r5, original q4 third IO rung |
| `i4l13r5_d640e256_q16q8q4t_lqer_lidx_r8t16` | prime loop-width baseline: l13/r5, original q4 third IO rung |
| `i4l11r5_d640e256_q16q8q8t_lqer_lidx_r8t16` | prime loop-width baseline: l11/r5, softer q8 third IO rung |
| `i4l13r5_d640e256_q16q8q8t_lqer_lidx_r8t16` | prime loop-width baseline: l13/r5, softer q8 third IO rung |
| `i4l11r5_d640e256_q16q8q8t_qk525_lqer_lidx_r8t16` | l11/r5 plus QK gain 5.25 |
| `i4l13r5_d640e256_q16q8q8t_qk525_lqer_lidx_r8t16` | l13/r5 plus QK gain 5.25 |

## Suggested Command After The Current Matrix Finishes

```powershell
$out = "records/sub4-leader-levers-i4-5k-$(Get-Date -Format yyyyMMdd-HHmmss)"
python scripts/run_sub4_iotail_quant_matrix.py `
  --candidate-group sub4_leader_levers `
  --out $out `
  --iterations 5000 `
  --warmdown-iters 5000 `
  --val-tokens 65536 `
  --timeout 7200 `
  --final-artifacts `
  --train-quant-forward `
  --quant-train-every 100 `
  --allow-over-cap `
  --wait-for-idle-gpu `
  --idle-max-util 15 `
  --idle-max-memory-mib 2500 `
  --idle-seconds 30
```

## Interpretation Rules

- Promote only final artifact round-trip BPB.
- For the TTT row, compare both `final_export_val_bpb` and
  `final_quant_ttt_val_bpb`; the latter is the legal score-first adaptation
  signal.
- If a single control wins on d640/e256, rerun it on d768/e320 and the final
  best shape from the active matrix.
- If l11/l13 beat l9, the next structural sweep should stay in prime-ish
  territory: l11/r7, l13/r5, l13/r7, and maybe i5/l11/r5 if bytes are still
  acceptable.
- If the stacked `publicsafe` row loses, do not discard its components. These
  controls are often non-additive in small models.
- Keep CaseOps/SP8192 exact-byte sidecars for all tokenizer experiments.

## Deferred Branches

- True progressive recurrence scheduling: promising, but it needs trainer work
  instead of an env-only candidate.
- Hessian-aware SDClip/GPTQ for the ternary export lane: likely useful, but it
  is a quant/export project rather than a one-line matrix lever.
- FLA/GatedDeltaNet: high-upside sub-16 branch.
- Byte-level PPM mixture: high-upside but policy-sensitive branch; keep it
  separate until legality is clearer.
