# Overnight Synthesis

Date: 2026-04-27

Scope: late 2026-04-26 through the promoted 16MB close-out on 2026-04-27.
Raw `train.csv` files remain the source of truth; this note records the current
engineering read.

## Headline

The overnight queue did its job: it pruned weak lanes, promoted only two 16MB
rows, and found a near miss. It did not beat the existing local 16MB VocabMoE
frontier.

Current local 16MB frontier remains:

| Candidate | Export BPB | Step | Bytes |
| --- | ---: | ---: | ---: |
| `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst` | `1.87104756` | `832.11ms` | `6,218,621` |

The best overnight/promoted row was close but worse:

| Candidate | Export BPB | Step | Bytes |
| --- | ---: | ---: | ---: |
| `mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32` | `1.88161553` | `1085.46ms` | `8,289,170` |

## Completed Result Groups

### Council / RLM-Lite

Run: `records/vocabmoe16-council-rlm-pruned-cont-5k-auto-20260427-012503`

| Candidate | Export BPB | Step | Bytes | Read |
| --- | ---: | ---: | ---: | --- |
| `..._council_hard_t60` | `1.88455034` | `1047.79ms` | `6,240,963` | best council row, near miss |
| `..._rlm_input_d90_s002` | `1.93625839` | `874.50ms` | `6,226,187` | worse than anchor |
| `..._dynamic_council_t60` | `1.95507240` | `887.70ms` | `6,228,023` | worse than anchor |
| `..._rlm_council_signperm` | `2.00975743` | `832.23ms` | `6,228,491` | clear loser |

Lesson: eval-only council and RLM-lite memory are not enough on this spine.
Hard council is close, but it is slower and still worse than the dense
VocabMoE anchor. Freeze broad council/RLM sweeps unless we train a genuinely
joint bridge or fix a specific export gap.

### Corrected Spike / Self-Election VocabMoE

Run: `records/vocabmoe16-spike-pruned-5k-auto-20260427-012503`

| Candidate | Export BPB | Step | Bytes | Read |
| --- | ---: | ---: | ---: | --- |
| `..._spikehybrid_k16r2_input_loopfirst_top2` | `1.87915642` | `866.19ms` | `6,254,694` | closest new row |
| `..._spikestatic_k16r2_input_top2` | `1.89526450` | `838.47ms` | `6,231,939` | viable, weaker |

Lesson: the corrected spiking/self-election implementation is real, not a
dead branch. Hybrid top-2 self-election is close to the anchor but still loses
by about `0.0081` BPB. This is the best "new idea" from the overnight run, but
it should be probed narrowly rather than swept broadly.

### Sub-4 / Soft-Size Public Lever Rows

Run: `records/sub4-leader-pruned-5k-auto-20260427-012503`

| Candidate | Export BPB | Step | Bytes | Read |
| --- | ---: | ---: | ---: | --- |
| `i4l9r5_d640e256_q16q8q4t_qk525_lqer_lidx_r8t16` | `2.47924839` | `227.90ms` | `5,908,181` | best soft-size row |
| `..._qk525_huberwd...` | `2.47924839` | `232.19ms` | `5,908,181` | no extra gain over qk |
| `..._qk525_attnout24...` | `2.49970160` | `232.56ms` | `5,897,461` | attention-output gate hurts |
| `..._publicsafe...` | `nan` | `248.13ms` | n/a | nonfinite, stopped |
| `i4l11r5_d640e256_q16q8q8t_qk525...` | `2.5458` at prune | `273.53ms` | n/a | slower and worse, stopped |

Lesson: QK 5.25 transfers to the soft-size i4/l9/r5 lane, improving the
previous quality-first soft-size best, but this is now a roughly 5.9MB row,
not a serious decimal sub-4MB candidate. Do not stack public levers blindly:
attention-output gating hurt, Huber WD was neutral here, and the stacked public
row went nonfinite.

### Width-Density / Internal Width Ladder

Run: `records/sub4-width-pruned-5k-auto-20260427-012503`

| Candidate | Export BPB | Step | Bytes | Read |
| --- | ---: | ---: | ---: | --- |
| `..._wl400-480-560-640_attncore1...` | `2.50753244` | `269.37ms` | `5,730,884` | worse/slower |
| `..._wl320-480-560-640_attncore1...` | `2.50851753` | `265.61ms` | `5,870,552` | worse/slower |

Lesson: making outer high-precision blocks internally narrower while widening
toward the low-precision core did not beat the plain i4/l9/r5 soft-size rows.
The intuition is still interesting, but the current implementation adds cost
without improving final export BPB.

### Cap-Speed Scout

Run: `records/vocabmoe16-cap-speed-scout-3k-auto-20260427-075723`

| Candidate | Export BPB | Step | Bytes | Read |
| --- | ---: | ---: | ---: | --- |
| `...d768e256...cap16fast...` | `4.14716446` | `934.79ms` | `6,612,446` | broken quality |
| `...d640e256...cap16fast...` | `4.15886638` | `740.83ms` | `5,543,162` | broken quality |

Lesson: the full fp16-param/no-GradScaler/no-fp32-loss speed profile is not a
valid quality path here. It ran faster but collapsed the final export. Future
16MB runs should keep fp32 params, fp32 Muon, GradScaler, fp32 loss, and
`GRAD_ACCUM_STEPS=4`; retain only safe speed levers like fused QKV,
train-time q6 forward, host prefetch, persistent buffers, and final-only
validation.

### Focused 16MB Scout

Run: `records/focused-16mb-after-capspeed-3000-auto-20260427-100310`

| Candidate | Export BPB | Step | Bytes | Read |
| --- | ---: | ---: | ---: | --- |
| `mainline_i3l5r2_d768e320_q6all...` | `1.98432387` | `1085.55ms` | `7,343,282` | best 3k focused row |
| `leader_i3l3r3_d768e320_q6all_sparsegate...` | `1.99466786` | `991.63ms` | `6,473,538` | second, promoted |
| `leader_i3l5r1rev...` | `2.02782949` | `1032.95ms` | `7,656,578` | loser |
| `leader_i3l3r3_depthlora4...` | `2.03060640` | `1035.03ms` | `6,511,626` | loser |
| `leader_i3l3r3_polar_minlr_qk525...` | `2.03322719` | `990.91ms` | `6,460,014` | loser |
| `mainline_i3l3r3_d768e384...` | `2.08196718` | `994.79ms` | `6,670,982` | wider embedding hurt |
| `mainline_i3l3r3_d896e384...` | `2.17243617` | `1196.48ms` | `7,536,806` | wider model hurt |

Lesson: spending bytes on d768/d896 width and e384 embeddings did not pay off
under the local 3k scout. The unique-loop i3/l5/r2 row earned promotion, but
the plain d640/e256 anchor remains the better local shape so far. Sparsegate
looked okay at 3k, but not enough to be the mainline.

### Promoted Top-2 5k

Run: `records/promote-top2-16mb-5000-auto-20260427-101142`

| Candidate | Export BPB | Step | Bytes | Read |
| --- | ---: | ---: | ---: | --- |
| `mainline_i3l5r2_d768e320_q6all...` | `1.88161553` | `1085.46ms` | `8,289,170` | near miss |
| `leader_i3l3r3_d768e320_q6all_sparsegate...` | `1.95062561` | `993.82ms` | `7,299,974` | did not transfer |

Lesson: the top-2 promotion rule worked. It deepened the two best scout rows
without wasting the whole night. The i3/l5/r2 row improved a lot from 3k to 5k
but still did not beat the d640/e256 anchor. Sparsegate improved from 3k but
fell far behind the mainline row, so it should not be promoted again without a
new mechanism.

## Consolidated Lessons

1. The d640/e256 input+loop-first hybrid VocabMoE anchor is still the local
   16MB spine to beat.
2. More width, bigger embeddings, and more artifact bytes were not automatic
   quality wins on the 2060 proxy.
3. The corrected spiking/self-election VocabMoE path is the most interesting
   new branch, but it is a near miss rather than a win.
4. Eval-only council and RLM-lite memory are useful negative results. Hard
   council is close, but still slower and worse than the anchor.
5. The fastest dtype path was a false economy. Speed levers must be judged by
   final export BPB, not only ms/step.
6. Public-leaderboard micro-levers are non-additive in our HRC/VocabMoE spine:
   sparsegate, depth LoRA, cycle-rev routing, and extra width all lost locally.
7. For sub-4/soft-size, QK 5.25 helps the i4/l9/r5 soft-size lane, but the best
   result is far over 4MB. Internal width ladders did not help.

## Recommended Next Moves

- Keep the current queue empty until we choose a small, high-signal next
  matrix.
- Treat `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst` as the
  16MB control row.
- If we run more local 16MB probes, prefer two or three rows around that
  control: one corrected spikehybrid variant, one conservative 10k/longer
  anchor replay, and maybe one i3/l5/r2 schedule-tuned replay.
- Do not run more broad d768/d896 width sweeps locally unless a schedule or
  optimization change specifically targets the failure.
- Keep stable dtype settings for 16MB quality rows. Use fused QKV and final-only
  validation, but do not use fp16 trainable params/no GradScaler as a quality
  path.
