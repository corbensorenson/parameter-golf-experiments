# Final Submission Notes

This repository is being submitted as a **non-record / art-lane** Parameter
Golf entry.

It is not a SOTA claim. It is a documented architecture exploration around a
compact recurrent transformer family called **MirrorLoop Recurrent Core** with
**Lexical Low-Rank Experts (LexLoRE)**.

## Reviewer Entry Points

- Submission folder:
  `records/track_non_record_16mb/2026-04-30_Corben_MirrorLoop_LexLoRE_HRC/`
- Submission README:
  `records/track_non_record_16mb/2026-04-30_Corben_MirrorLoop_LexLoRE_HRC/README.md`
- PR body text:
  `records/track_non_record_16mb/2026-04-30_Corben_MirrorLoop_LexLoRE_HRC/PR_BODY.md`
- Architecture explainer:
  `records/architecture_explainer_20260430.md`
- Full experiment repo:
  https://github.com/corbensorenson/parameter-golf-experiments
- Submission fork branch:
  https://github.com/corbensorenson/parameter-golf/tree/codex/agent-b-harness

## Main Preserved Result

Best preserved under-cap 8xH100 evidence:

| Candidate | Final export BPB | Train-time val BPB | Steps | Step speed | Bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer6t12_vocabmoe_qk55` | `1.35496419` | `1.3191` | `6658` | `90.13ms` | `15,989,749` |

This was a narrow, self-funded RunPod 8xH100 run late in the challenge window.
It is submitted as evidence for the architecture, not as an official
statistically significant record attempt.

## Strongest Architecture Signal

The strongest late architectural finding was a prime-skip MirrorLoop route:

```text
012 | 34567 | 35746 | 210
```

Preserved 1xH100 result:

| Candidate | Final export BPB | Steps | Step speed | Bytes | Headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| `break_prime_skip_superloop_d640e768` | `1.35504224` | `5563` | `107.87ms` | `14,051,162` | `1,948,838` |

This is not the main submitted 8x score. It is the clearest signal for where
the model family should go next: changing the recurrent route helped more than
adding more repeated depth.

## Architecture In One Paragraph

MirrorLoop stores a small set of physical transformer blocks and walks them as
a route: token-facing entry shell, reused recurrent middle, mirrored exit
shell. LexLoRE adds token-conditioned low-rank expert residuals at the input
and first recurrent-core entry. The model trains through q8 quantized forward
views from step 0, including tied embeddings, then reports final exported reload
BPB after LQER quantization-error repair.

## Honest Negative Findings

The submission intentionally includes what did not work:

- More recurrence did not automatically help.
- Smaller batches gave more updates but did not beat the best legal export
  score.
- Exit-side/warm LexLoRE and sparse/self-election LexLoRE did not improve the
  best H100 spine.
- Bigram side features did not improve the best H100 spine.
- All-core attention was slower and worse than keeping attention concentrated
  at token-facing and loop-entry positions.
- The tested q16/q8/q8 train-time precision-width ladder was implemented
  honestly, but the H100 row was worse and over the cap.
- 8xH100 improved final BPB only slightly over the best 1xH100 evidence, so the
  remaining bottleneck is architecture/export gap, not simply hardware scale.

## Claim Boundary

This submission claims:

- A coherent, auditable non-record architecture family.
- Preserved under-cap 8xH100 evidence at `1.35496419` BPB.
- A promising prime-skip recurrent-route direction at `1.35504224` BPB on
  1xH100 with substantial byte headroom.

This submission does not claim:

- SOTA.
- Statistical significance versus the leaderboard.
- That every component is individually novel.
- That train-time validation is the score. The final exported reload BPB is the
  score used throughout the submission docs.
