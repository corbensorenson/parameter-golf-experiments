# 16MB Structure Synthesis

Date: 2026-04-29

This note closes out the dual-stream / hourglass / IO-tail-ladder structure
probe on the current best local 16MB spine.

## Control

Current control:

| Candidate | Export BPB | Step | Bytes | Read |
| --- | ---: | ---: | ---: | --- |
| `loopfollow_i3l5r5_d640e512_q8` | `1.54148844` | `1559.74ms` | `13,814,802` | best local 16MB row |

The strongest proven pattern is q8 train/export, e512 factored embeddings,
dense VocabMoE at input plus loop-first, and more unique loop blocks with r5
recurrence.

## Structure Results

Run: `records/cap16-structure-followup-5k-auto-20260428-174616`

| Candidate | Export BPB | Step | Bytes | Read |
| --- | ---: | ---: | ---: | --- |
| `structure_dual_i3l5r5_d640e512_q8_left256_r16_loopx` | `1.54399822` | `1591.48ms` | `13,770,082` | near miss, not a win |
| `structure_hourglass_i3l5r5_d640e512_q8_w400-480-560-640` | `1.55335515` | `1423.07ms` | `13,102,734` | faster/smaller, worse quality |

Run: `records/cap16-structure-combo-5k-auto-20260428-180151`

| Candidate | Export BPB | Train-Time Val BPB | Step | Bytes | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| `structure_combo_i3l5r5_d640e512_q16q8q4io_q8core_w400-480-560-640_dual` | `1.55647474` | `1.5411` | `1500.47ms` | `13,816,482` | export gap, not a win |

## Lessons

1. The simple full-width single-stream i3/l5/r5 q8/e512 spine remains the best
   local 16MB candidate.
2. Dual-stream is real but not yet worth promoting. It nearly matched the best
   row, but added cost and missed by about `0.0025` BPB.
3. The current hourglass implementation buys speed and bytes at the cost of
   quality. It also disables depth LoRA, basis XSA, and VE, so it removes some
   useful specialization capacity while testing width adapters.
4. The combined IO-tail ladder + hourglass + dual row looked better before
   export than after export. The final train-time validation was close to the
   best control, but quantized reload widened the BPB by about `0.0154`.
5. Do not trust train-time validation alone on these quantized structure rows.
   Final export roundtrip is the only promotion metric.

## Next Read

The next useful work is not more broad structure stacking. It is either:

- an export-gap repair around the q8/e512 i3/l5/r5 spine,
- a matched schedule/optimizer test on the same spine,
- or a very small unique-loop shape probe that keeps the proven q8/e512 policy.

Do not queue more dual/hourglass/IO-ladder combinations unless the next change
directly addresses the export gap.
