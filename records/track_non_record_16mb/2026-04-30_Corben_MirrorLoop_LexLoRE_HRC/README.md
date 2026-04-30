# MirrorLoop HRC + LexLoRE

**Track:** non-record / art submission  
**Author:** Corben Sorenson ([@corbensorenson](https://github.com/corbensorenson))  
**Status:** 1xH100 evidence now, intended 8xH100 follow-up if capacity becomes available before review.

## Summary

This submission explores a deliberately nonstandard small language model shape:

- **MirrorLoop HRC:** a mirrored input/output shell around a recurrent middle:
  `012 | 34567 | 34567 | 210`.
- **LexLoRE:** token-conditioned low-rank lexical expert residual adapters at
  the input and loop-entry sites.
- **Train-time quantization from step 0:** the model is trained through the same
  q8 quantized forward path used by the final artifact, including embeddings.
- **Factored tied embeddings:** the token interface is widened without paying
  the full dense `vocab x dim` cost.
- **LQER:** low-rank quantization error repair is applied at export.
- **One attention-capable core-entry block:** the recurrent core is otherwise
  MLP-only for speed.

The goal is not to clone the accepted leaderboard transformer stack. It is to
test whether a mirrored recurrent core plus lexical low-rank steering can form a
compact, auditable architecture family under the 16MB artifact constraint.

## Best Current Result

The best legal H100 result available before this PR was a 1xH100, 10-minute
wall-clock run:

| Candidate | BPB | Steps | Step speed | Total bytes |
|---|---:|---:|---:|---:|
| `h100_batch32k_d704e832_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55` | `1.35692129` | `5018` | `119.57 ms` | `15,658,145` |

This is below the decimal 16MB cap. It is **not** claimed as a record
submission, and it has **not** yet been reproduced on the official 8xH100
configuration. It is submitted here as a non-record/art lane result so the
architecture and negative/positive findings are visible before the challenge
deadline.

The raw RunPod pod used for the strongest 1xH100 queue became unavailable after
the wallet ran out of funds, so `train.log` contains the preserved result notes
from the project audit rather than the full raw stdout. If 8xH100 capacity
becomes available, this PR should be updated with the raw 8x logs.

## Reproduction

This record folder includes `train_gpt.py` and the small `ternary_golf` helper
package imported by the trainer. The exact 1xH100 command is in
`run_1xh100_best.sh`.

Expected data/tokenizer:

- `DATA_PATH`: CaseOps/SP8192 lossless dataset directory.
- `TOKENIZER_PATH`: matching CaseOps/SP8192 SentencePiece model.

Run:

```bash
bash run_1xh100_best.sh
```

For an 8xH100 follow-up, keep the architecture constants and change only the
distributed launch/batch schedule. The project repository contains a prepared
`final8x` runner for that paid test.

## Notes On Validity

This submission is intentionally conservative about claims:

- It is under `track_non_record_16mb`.
- It does not claim an 8xH100 official score.
- It does not claim SOTA.
- It reports the current best legal 1xH100 evidence and the intended 8x path.
- It keeps the architecture self-contained and auditable.

## Why This Is Interesting

Most ingredients have relatives in prior work: recurrence, adapters,
quantization-aware training, factored embeddings, and low-rank repair. The art
piece is the combination:

1. A mirrored IO shell that returns through the same semantic ladder it entered.
2. A looped middle that spends compute without spending many new parameters.
3. Lexical low-rank experts that steer both token read-in and recurrent entry.
4. Training on the quantized forward path from the first step rather than
   quantizing only after training.
5. Treating batch/update rate as part of the architecture search rather than a
   separate systems detail.
