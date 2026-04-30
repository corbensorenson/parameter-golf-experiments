# Non-record: MirrorLoop Recurrent Core + LexLoRE

## Summary

This is a non-record / art-lane submission for a novel compact architecture
family, not a SOTA claim.

The submitted model family combines:

- **MirrorLoop Recurrent Core:** a mirrored token-facing IO shell around a reused
  recurrent middle.
- **LexLoRE:** token-conditioned shared low-rank expert adapters at input and
  loop-entry sites.
- **Export-honest q8 training:** train-time quantized forward from step 0,
  including tied embeddings.
- **Factored tied SP8192/CaseOps embeddings** and **LQER sidecars** counted in
  the artifact.

## Best Preserved Evidence

Best preserved under-cap 8xH100 result:

| Candidate | Final export BPB | Steps | Step speed | Bytes |
| --- | ---: | ---: | ---: | ---: |
| `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer6t12_vocabmoe_qk55` | `1.35496419` | `6658` | `90.13ms` | `15,989,749` |

The strongest late architecture signal was a 1xH100 prime-skip MirrorLoop
route:

| Candidate | Final export BPB | Steps | Step speed | Bytes |
| --- | ---: | ---: | ---: | ---: |
| `break_prime_skip_superloop_d640e768` | `1.35504224` | `5563` | `107.87ms` | `14,051,162` |

Prime-skip is documented as follow-up architecture evidence, not the main
submitted 8x score.

## Full Experiment Log

The full working repo with run ledgers, negative findings, and broader search
history is here:

https://github.com/corbensorenson/parameter-golf-experiments

## Validity Notes

- This is under `track_non_record_16mb`.
- It is not claiming SOTA or statistical significance.
- It reports final exported reload BPB as the main score.
- The README includes negative findings: recurrence depth, spike LexLoRE,
  Bigram side features, all-core attention, and precision-width ladder variants
  that did not transfer.

## Files

Submission folder:

`records/track_non_record_16mb/2026-04-30_Corben_MirrorLoop_LexLoRE_HRC/`

Includes the trainer, launch scripts, `submission.json`, compact train log, and
raw preserved stdout for the best 8x row plus the prime-skip architecture
signal.
