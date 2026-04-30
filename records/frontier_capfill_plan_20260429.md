# 16MB Frontier Cap-Fill Plan

Date: 2026-04-29

This plan follows the structure close-out in
`records/structure_synthesis_20260429.md`. The current local control is:

| Candidate | Export BPB | Step | Bytes | Headroom |
| --- | ---: | ---: | ---: | ---: |
| `loopfollow_i3l5r5_d640e512_q8` | `1.54148844` | `1559.74ms` | `13,814,802` | `2,185,198` |

## Public Frontier Signal

The accepted upstream leaderboard still centers on SP8192, recurrence,
parallel residuals, QK gain 5.25, and legal score-first TTT:
<https://github.com/openai/parameter-golf>

Useful open PR signals as of this pass:

- PR #1790 reports a no-CaseOps SP8192 transformer stack at `1.06991`, using
  SmearGate, attention-output gate width 24, QK gain 5.25, and improved phased
  LoRA TTT: <https://github.com/openai/parameter-golf/pull/1790>
- PR #1874 reports `1.06766` by adding Polar Express Newton-Schulz,
  `MIN_LR=0.10`, and small asymmetric LQER on top of PR #1790:
  <https://github.com/openai/parameter-golf/pull/1874>
- PR #1797 reports a stronger CaseOps stack using SmearGate plus asymmetric
  LQER: <https://github.com/openai/parameter-golf/pull/1797>
- PR #1791 reports a separate GDN/FLA line at `1.0339`, but that is a larger
  architecture branch, not a quick port into the current HRC/VocabMoE trainer:
  <https://github.com/openai/parameter-golf/pull/1791>

## Active Matrix

Run directory:
`records/cap16-frontier-capfill-5k-auto-20260429-003802`

Command:

```powershell
python -u scripts/run_16mb_vocab_moe_matrix.py `
  --candidate-group cap16_frontier_capfill `
  --iterations 5000 `
  --warmdown-iters 5000 `
  --val-tokens 131072 `
  --timeout 18000 `
  --out records/cap16-frontier-capfill-5k-auto-20260429-003802
```

Rows:

| Candidate | Lever | Why It Is Worth GPU Time |
| --- | --- | --- |
| `frontier_capfill_i3l5r5_d640e640_q8` | Spend bytes on factored embedding rank | Directly tests whether the remaining 2.18MB headroom should go into the token interface on the winning i3/l5/r5 spine. |
| `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed` | Stronger LQER including `embed_proj` | Tries to repair the observed train/export gap and ports the public frontier's quant-error-repair lesson. |
| `frontier_polarminlr10_i3l5r5_d640e512_q8` | Polar Express NS plus 10% LR floor | Ports a public optimizer/schedule gain that costs no artifact bytes and may improve late training. |
| `frontier_qk525_parres4_i3l5r5_d640e512_q8` | QK 5.25 plus wider parallel residual tail | Ports the accepted-leader recurrence/routing/gain pattern onto the HRC/VocabMoE spine. |

All rows keep the current best full-width single-stream HRC/VocabMoE shape,
train with q8 forward views from the start, and skip periodic validation so
local time is spent on training updates. Promotion is based only on the final
export roundtrip.

## Results So Far

| Candidate | Export BPB | Train-Time Val BPB | Step | Bytes | Headroom | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `frontier_capfill_i3l5r5_d640e640_q8` | `1.53679911` | `1.5261` | `1604.10ms` | `14,574,862` | `1,425,138` | new local best; e640 token-interface spend pays |
| `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed` | `1.54073632` | `1.5292` | `1576.04ms` | `13,956,518` | `2,043,482` | slight repair over e512 control, but loses to e640 |
| `frontier_polarminlr10_i3l5r5_d640e512_q8` | `1.53362322` | `1.5336` | `1560.22ms` | `14,091,166` | `1,908,834` | new best; Polar/MIN_LR transfers cleanly |
| `frontier_qk525_parres4_i3l5r5_d640e512_q8` | `1.54894140` | `1.5489` | `1585.39ms` | `13,900,982` | `2,099,018` | negative; do not promote QK+parres here |

The e640 row beat the previous `i3/l5/r5 d640/e512 q8` control
(`1.54148844` BPB, `13,814,802` bytes) by about `0.00469` BPB while spending
about `760KB` more artifact bytes. Then the Polar/MIN_LR row beat both at
`1.53362322` BPB while keeping the e512 embedding rank. The two live levers are
therefore token-interface rank and Polar/MIN_LR schedule polish.

The LQER r12/t24 plus `embed_proj` row improved slightly over the old e512
control (`1.54073632` vs `1.54148844` BPB), but it did not catch the e640 row.
This keeps the e640+LQER follow-up useful, but LQER alone is not the main
quality lever.

QK 5.25 plus parres4 was a clear negative on this spine. Do not spend another
row on that bundle unless a later result changes the architecture context.

## Watchdog And Follow-Up

Automation: `parameter-golf-frontier-watchdog`

Local script:
`scripts/hourly_frontier_watchdog.ps1`

The hourly watchdog appends status to `records/hourly-watchdog.md`, checks the
current queue for new final export rows, and avoids duplicate queue launches.
When all four current rows are complete, it launches the next selective 5k group
exactly once using a marker file in the current run directory.

Recovered follow-up run:
`records/cap16-frontier-followup-5k-auto-20260429-164230`

Follow-up group:
`cap16_frontier_followup`

Rows:

| Candidate | Why |
| --- | --- |
| `frontier_follow_i3l5r5_d640e640_q8_polarminlr10` | Combines the two winning levers: e640 token-interface rank and Polar/MIN_LR. |
| `frontier_follow_i3l5r5_d640e768_q8_polarminlr10` | Tests whether cap spend continues closer to 16MB when paired with Polar/MIN_LR. |
| `frontier_follow_i3l5r5_d640e640_q8_polarminlr10_lqer12t24_embed` | Tests whether the small LQER repair adds to e640+Polar. |
| `frontier_follow_i3l5r5_d640e640_q8_polarminlr05` | Schedule sensitivity row: 5% LR floor versus the winning 10% floor. |

## What Was Deliberately Not Queued

- No broad d768/d896 sweep: local evidence says width spend lost after export.
- No more hourglass or dual-stream combinations: both were near-misses or
  negatives on the current spine.
- No PPM/GDN reproduction in this queue: those are larger branches and would
  deserve their own setup rather than being mixed into this HRC cap-fill pass.
- No 20k runs: 5k probes are still the right selection unit until one row
  clearly beats `1.54148844` after export.
