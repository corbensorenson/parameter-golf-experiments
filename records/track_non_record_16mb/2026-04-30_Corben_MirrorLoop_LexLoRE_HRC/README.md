# MirrorLoop Recurrent Core + LexLoRE

**Track:** non-record / art submission  
**Author:** Corben Sorenson ([@corbensorenson](https://github.com/corbensorenson))  
**Public experiment log:** [corbensorenson/parameter-golf-experiments](https://github.com/corbensorenson/parameter-golf-experiments)  
**Submission branch:** [corbensorenson/parameter-golf, `codex/agent-b-harness`](https://github.com/corbensorenson/parameter-golf/tree/codex/agent-b-harness)

## Summary

This is an art-lane submission for a deliberately unusual compact language
model family. The goal was not to clone the current accepted leaderboard stack.
The goal was to test whether a transformer can spend parameters more
aggressively by routing through a mirrored IO shell and a reused recurrent
middle, while token-conditioned low-rank experts steer the lexical interface.

The submitted family combines:

- **MirrorLoop Recurrent Core:** an entry shell, reused recurrent middle, and
  mirrored exit shell.
- **LexLoRE:** Lexical Low-Rank Experts, a shared low-rank expert bank with
  token-conditioned routing at `input` and `loop_first`.
- **Export-honest quantization:** q8 train-time forward from step 0, including
  tied embeddings, followed by final exported reload scoring.
- **Factored tied embeddings:** a lower-rank SP8192 token interface so the
  artifact can stay near the decimal 16MB cap.
- **LQER:** low-rank quantization-error repair sidecars counted inside the
  artifact.
- **Attention concentrated at the core entry:** the shell and first recurrent
  core block keep token mixing, while deeper recurrent core blocks are MLP-only
  for speed.

## Best Preserved Evidence

The best preserved under-cap result is from a self-funded 8xH100 RunPod run.
The run used the official-style 10-minute wall-clock budget, but it should
still be treated as non-record evidence because the sweep was narrow and
self-funded late in the challenge window.

| Candidate | Final export BPB | Train-time val BPB | Steps | Step speed | Total bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer6t12_vocabmoe_qk55` | `1.35496419` | `1.3191` | `6658` | `90.13ms` | `15,989,749` |

This is not presented as SOTA. It is presented as the cleanest preserved score
for the MirrorLoop/LexLoRE family.

The strongest preserved 1xH100 architecture signal came later from a
prime-skip MirrorLoop route:

| Candidate | Final export BPB | Steps | Step speed | Total bytes | Headroom |
| --- | ---: | ---: | ---: | ---: | ---: |
| `break_prime_skip_superloop_d640e768` | `1.35504224` | `5563` | `107.87ms` | `14,051,162` | `1,948,838` |

That route was:

```text
012 | 34567 | 35746 | 210
```

It is not used as the main submitted score because it was only preserved on
1xH100, but it is the most interesting late architecture finding: it improved
quality and speed while leaving almost 2MB of artifact headroom.

## Architecture

The ordinary HRC route used in the 8x preserved score is:

```text
entry      loop pass 1       loop pass 2       mirrored exit
0 1 2  ->  3 4 5 6 7   ->    3 4 5 6 7   ->   2 1 0
```

The implementation stores only eight physical transformer blocks:

- blocks `0,1,2`: token-facing IO shell;
- blocks `3,4,5,6,7`: recurrent middle;
- the exit tail reuses `2,1,0` with role conditioning.

The recurrent middle is not simply repeated blindly. The route carries pass
embeddings, loop-index information, and recurrent injection parameters so the
same physical block can learn different virtual-depth roles.

LexLoRE is placed at:

```text
input, loop_first
```

The adapter is not a full per-token network. Each token owns a small learned
router prior over shared low-rank expert bases:

```text
token_prior: vocab_size x num_experts
down:        num_experts x rank x dim
up:          num_experts x dim x rank
```

This gives token-specific steering while keeping the CUDA work batched and the
artifact auditable.

## Quantization And Scoring

The training path uses the low-precision forward view from the first step:

```text
TRAIN_QUANT_FORWARD=1
TRAIN_QUANT_EMBEDDINGS=1
QUANT_WEIGHT_BITS=8
VOCAB_MOE_TRAIN_QUANT_BITS=8
```

The score reported above is the final exported reload score, not the pre-export
training model. This matters because several candidate families looked much
better before export and then lost quality after compression.

## What Worked

- MirrorLoop/HRC produced a coherent and reproducible family of compact models.
- LexLoRE at `input,loop_first` was the best lexical expert placement tested.
- q8 train/export with train-time embedding quantization fixed an important
  train/export mismatch.
- Factored embeddings and LQER were necessary to get useful quality near 16MB.
- 8xH100 used the hardware properly: the preserved legal row ran at about
  `90ms/step` with all GPUs active.
- Prime-skip recurrence was the strongest late architecture signal and is the
  best next branch if this line is continued.

## Negative Findings

The project is intentionally including the failed ideas because they are useful
for review and future work:

- More recurrence was not automatically better. A legal r3 probe was worse
  than the r2 spine.
- Smaller batches gave more optimizer updates but did not beat the best legal
  export score.
- Exit-side/warm LexLoRE, sparse/self-election LexLoRE, and Bigram side
  features did not improve the best H100 spine.
- All-core attention was slower and worse; occasional token mixing helped more
  than making every recurrent block attentional.
- A train-time q16/q8/q8 precision-width ladder was implemented honestly, but
  the tested H100 row was worse and over the cap.
- The 8x run improved final BPB by only about `0.002` over the best 1x result,
  so this family is not simply waiting for more GPUs; the architecture/export
  gap is the bottleneck.

## Reproduction

This record folder includes:

- `train_gpt.py`: self-contained trainer used by the submitted family;
- `run_8xh100_best.sh`: preserved 8xH100 command for the best under-cap score;
- `run_1xh100_best.sh`: preserved 1xH100 command for the earlier scout row;
- `train.log`: compact score log and audit notes;
- `train_8xh100_legal_best.log`: raw preserved stdout for the best 8x row;
- `train_1xh100_prime_skip.log`: raw preserved stdout for the prime-skip
  architecture signal;
- `ternary_golf/`: small helper package imported by the trainer.

Expected data/tokenizer:

- `DATA_PATH`: CaseOps/SP8192 lossless dataset directory.
- `TOKENIZER_PATH`: matching CaseOps/SP8192 SentencePiece model.

Run the preserved 8x command:

```bash
bash run_8xh100_best.sh
```

Run the earlier 1x command:

```bash
bash run_1xh100_best.sh
```

## Validity Notes

- This is a `track_non_record_16mb` submission.
- It is not claiming SOTA.
- It is not claiming that every ingredient is individually new.
- It reports final exported reload BPB as the main metric.
- It links the full experiment repository so reviewers can inspect the broader
  search, including negative results and failed lanes.

## Why This Is Interesting

Most individual pieces have relatives in prior work: recurrence, low-rank
adapters, MoE routing, train-time quantization, tied embeddings, and low-rank
quant repair. The contribution here is the particular architecture package:

1. Mirror the token-facing shell instead of treating depth as a flat stack.
2. Spend compute by looping the semantic middle without storing a full deeper
   model.
3. Use lexical low-rank expert steering at the read-in and loop-entry points.
4. Train through the intended quantized forward path from step 0.
5. Treat routing, precision, embedding rank, and export repair as one coupled
   artifact budget.

The result is not a leaderboard-winning model. It is a strange, auditable model
family with enough evidence to show what parts helped, what parts failed, and
where the next branch should go.
