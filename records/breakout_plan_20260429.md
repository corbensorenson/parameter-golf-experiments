# 16MB Breakout Plan

Date: 2026-04-29

The current best local HRC/VocabMoE line is good but still far from the public
leaderboard regime. The best completed row is
`frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53362322` final export BPB.
That is a strong local improvement, but it is not a plausible path to `<1.2`
by only tuning embedding rank, LQER, or the LR floor.

The next run therefore pivots from single-spine exploitation to six distinct
5k-step breakout tests:

| Candidate | Question |
| --- | --- |
| `breakout_dense11_d512e512_q8_polar_vocabmoe_bigram` | Is HRC recurrence the quality ceiling versus a dense SP8192-like transformer? |
| `breakout_dense13_d512e384_q8_polar_vocabmoe` | Does spending bytes on physical depth beat recurrent reuse? |
| `breakout_dense11_d640e384_q6_fullkv_memattn_vocabmoe_bigram` | Does a wider dense model with full K/V heads and memory-efficient attention improve quality/speed? |
| `breakout_hrc_i3l5r5_d640e512_q8_polar_ttt_control24` | Does legal score-first TTT help the current best HRC spine after export? |
| `breakout_hrc_i3l7r4_d640e512_q8_polar_bigram` | Does the more-unique-loop-block signal keep improving when paired with BigramHash? |
| `breakout_dual_i3l5r2_d768e320_left256_q6_polar_vocabmoe_lqer16t32` | Does trained dual-stream help as a real architecture branch rather than an eval-only council? |

All rows use `TRAIN_LOG_EVERY=250` and the matrix runner now tees live trainer
stdout to both `queue.out.log` and the active `train_*.txt`, so progress is
visible while a candidate is running instead of only after it exits.

Interpretation rule: promote only rows that improve final exported BPB, not
train-time validation alone. If none of these rows moves substantially below
the `1.53` local band, the honest conclusion is that we need to reproduce a
stronger public stack branch, such as a cleaner SP8192 transformer/TTT baseline
or a GDN/FLA-style branch, rather than keep polishing this HRC family.
