# H100 Break-Cliff Results

Date: 2026-04-30

This was a two-hour 1xH100 RunPod scout launched after the 8xH100 run showed
only a tiny legal improvement over the best 1xH100 candidate. The goal was not
more polish on the same r2 spine; it was to test architectural changes that
could plausibly break the `~1.35` exported-BPB plateau.

Remote evidence was copied back to:

`artifacts/runpod-h100-breakcliff-20260430-145928/`

Runner:

`scripts/run_h100_breakcliff_matrix.py`

The runner first did a one-step export preflight to catch broken rows, then ran
full 600-second candidates. The preflight correctly caught the first
precision/width ladder as invalid because the width ladder entries were not
compatible with `d704 / 11 heads`; a fixed version was run afterward.

## Result Table

Sorted by final exported BPB:

| Candidate | Final Export BPB | Steps | Step Avg | Bytes | Headroom | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `break_prime_skip_superloop_d640e768` | `1.35504224` | `5563` | `107.87ms` | `14,051,162` | `1,948,838` | Best row; fastest and below the prior legal 1x/8x plateau. |
| `break_coreattn2_d704e768` | `1.35740546` | `4731` | `126.83ms` | `15,780,710` | `219,290` | Tiny improvement over local control, but slower. |
| `break_best32k_d704e832_control` | `1.35752021` | `5057` | `118.65ms` | `15,903,150` | `96,850` | Reproduced known legal r2 spine on the new pod. |
| `break_lexlore_exit_rank4_warm_d704e768` | `1.35835216` | `5020` | `119.53ms` | `15,509,538` | `490,462` | Write-side/warm LexLoRE did not help. |
| `break_bigram_sidefeat_d704e704` | `1.36000511` | `5042` | `119.02ms` | `15,202,482` | `797,518` | Bigram side feature was negative on this spine. |
| `break_lexlore_spike32_top4_exit_d704e768` | `1.36355161` | `4993` | `120.17ms` | `15,539,318` | `460,682` | Sparse/self-election LexLoRE was negative. |
| `break_i3l7r2_unique_loop_d640e768` | `1.36477677` | `4918` | `122.02ms` | `15,421,822` | `578,178` | More physical loop blocks lost in this 10-minute setting. |
| `break_partial_tail_cycle_lidx_d704e768` | `1.36736347` | `4585` | `130.87ms` | `15,319,058` | `680,942` | Partial-tail recurrence was slower and worse. |
| `break_coreattn_all_d640e768` | `1.36810324` | `4258` | `140.94ms` | `15,373,986` | `626,014` | All-core attention overpaid in speed and quality. |
| `break_precision_width_q16q8q8_d704e704_fixed` | `1.36314982` | `4984` | `120.39ms` | `18,245,526` | `-2,245,526` | Invalid for 16MB; train-time q16/q8/q8 plus width ladder was worse and over cap. |

## What Changed Our Beliefs

### Prime-Skip Superloop Is The New Best 1x Architecture Signal

`break_prime_skip_superloop_d640e768` uses the same 16 virtual HRC steps as the
i3/l5/r2 family, but instead of repeating the core in the same order it walks
the five-block core with two coprime skip programs:

`012 | 34567 | 35746 | 210`

In code this is `HRC_DEPTH_SCHEDULE_MODE=prime_skip_superloop`,
`HRC_SUPERLOOP_SKIP_SCHEDULE=0,1`, with route phase and loop-index conditioning
enabled.

Why it matters:

- It beat the best legal 1xH100 control from this pod:
  `1.35504224` vs `1.35752021`.
- It beat the earlier best legal 1xH100 record:
  `1.35504224` vs `1.35692129`.
- It is close to the best legal 8xH100 one-hour result:
  `1.35504224` vs `1.35496419`.
- It is much faster than the control:
  `107.87ms/step` vs `118.65ms/step`.
- It has almost `1.95MB` of headroom, so it is not byte-saturated.

The practical next step is not another broad HRC matrix. It is a cap-spend
matrix on the prime route: larger body or embed rank, modest LQER repair, maybe
one extra core-attention block, and a careful 24k/32k batch comparison.

### More Attention Has A Narrow Sweet Spot

Adding one extra attention-enabled core block (`coreattn2`) barely improved the
control, but all-core attention was much worse:

- `coreattn2`: `1.35740546`, `126.83ms/step`.
- all-core attention: `1.36810324`, `140.94ms/step`.

This says the recurrent middle probably benefits from occasional token mixing,
but turning the entire loop into attention destroys the update-count advantage.

### LexLoRE Variants Did Not Wake Up Fast Enough

Exit-side/warm rank-4 LexLoRE and sparse top-k self-election both lost. That
does not invalidate LexLoRE as the input/loop-entry adapter, but it suggests
that, in 10-minute H100 runs, spending more bytes on the token expert bank is
less useful than changing the HRC route itself.

### Width/Precision Ladder Needs A Different Legal Form

The fixed q16/q8/q8 IO-tail plus width ladder trained correctly from step zero,
but exported to `18,245,526` bytes and scored `1.36314982`. This is not a
promoted 16MB direction. If we keep testing train-time precision ladders, the
legal version should use a smaller `d640` or fewer promoted q16/q8 blocks.

## Next Candidate Recommendation

The next paid candidate family should start from:

`prime_skip_superloop_d640e768`

Then spend the `~1.95MB` headroom deliberately:

1. `d704/e768` prime route if preflight/trial export suggests it can stay under
   cap after training.
2. `d640/e896` or `d640/e1024` prime route to spend more on the token
   interface while preserving speed.
3. prime route plus `coreattn2` if the extra attention can be added without
   losing too many steps.
4. prime route plus LQER `r12/t24` or `r8/t16` A/B, because the current row
   used `r10/t20` and has room.
5. 24k-per-rank style batch on the prime route, only if it remains legal after
   export; the older plain-HRC 24k raw score was promising but hard to legalize.

This is now the cleanest novel architecture story: MirrorLoop with a prime
skip-program recurrent core, LexLoRE at input/loop entry, train-time q8
forward/export, LQER repair, and CaseOps/SP8192.
