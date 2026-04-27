# Parameter Golf Experiments

This is Corben's working repository for Parameter Golf experiments. It starts
from `openai/parameter-golf`, but the purpose of this fork is narrower: test
very small language-model candidates, especially sub-4MB artifacts, and figure
out which ideas are worth scaling to the official 10-minute 8xH100 setting.

For the original challenge rules, leaderboard, and submission process, see the
upstream repository:
<https://github.com/openai/parameter-golf>

## Main Idea

The current bet is that a tiny model can buy back quality with parameter reuse,
train-time quantization, and careful byte spending:

- **Mirrored IO-tail / looped middle.** Candidate names like `i3l3r3` mean
  three entry blocks, three recurrent middle blocks, three middle repeats, and
  a mirrored three-block exit tail. A route like `012|345|210` lets the model
  spend more precision on the way in and out while reusing a small core.
- **Train-time mixed precision.** The model is not just quantized at export.
  With `TRAIN_QUANT_FORWARD=1`, selected linear layers use q8/q6/q4/ternary
  STE forward views from the first training step.
- **Ternary recurrent cores.** The repeated middle can be trained as ternary
  while IO blocks remain higher precision. This is the main sub-4MB compression
  lane.
- **Factored tied embeddings.** Full `8192 x dim` embeddings are too expensive
  under 4MB, so the sub-4 family uses lower-rank embedding factors.
- **LQER sidecars.** Low-rank quantization-error residuals recover some quality
  after aggressive quantization while staying inside the artifact cap.
- **16MB vocabulary MoE probes.** The official-size lane can spend bytes on a
  token-conditioned low-rank expert adapter: each token learns a tiny router
  prior, while shared expert bases keep the CUDA work batched.
- **Dual-stream advisor probes.** A trained left/right bridge can split the
  residual features into a token-facing lane and a recurrent lane, then exchange
  tiny low-rank messages at input, loop-entry, and pre-output sites.
- **Lossless tokenizer probes.** CaseOps and word-boundary-aware tokenizer
  experiments are tracked separately from architecture changes so byte
  accounting stays auditable.

## Current Experiment Lanes

| Lane | What it tests | Current local read |
| --- | --- | --- |
| Promoted sub-4MB q884 IO-tail | `i3l3r3`, q8/q8/q4 IO blocks, ternary core, LQER, train-time quantized forward | Best clean legal row: `2.5749` BPB, `148.36ms/step`, `3,967,875` bytes |
| Soft-cap q884 quality reference | Same family with slightly larger residual spend | `2.5505` BPB, but `4,035,469` bytes, about `35KB` over the decimal 4MB goal |
| Precision-ladder IO tail | q16/q8/q4/q2/ternary entry and mirrored exit from the first training step | Legal and fast; best row `2.9888` BPB, so d512/e192 is under-capacity |
| Loop-index recurrence | Whether the looped middle benefits from virtual-position information | Helps r9 and i5/l5, hurts q884 r3; do not enable blindly |
| Sub-16MB transfer lane | Ports useful sub-4 speed and quality levers into a less byte-starved model | Local q6 proof baseline: `1.7567` final BPB, `9.27MB` artifact |
| 16MB Vocab-MoE lane | Token-conditioned shared low-rank experts on top of the q6 HRC/CaseOps stack | Best completed dense row: `1.8710` BPB with input+loop-first hybrid Vocab-MoE; next scout spends cap on d768 width, richer LQER, QK 5.25, and fp16/fused-QKV speed levers |
| 16MB leaderboard-blend lane | Ports current public-leaderboard tricks onto our HRC/VocabMoE spine | Queued 5k probes cover Polar/MIN_LR, QK 5.5, sparse attention gate, parallel residuals, moderate Muon WD, BigramHash, and legal score-first TTT |
| 16MB spike/self-election Vocab-MoE | Hard top-k token/expert election variants of Vocab-MoE | Old queue aborted before training; corrected spike rows now use nonzero token-prior tie-breaks and are queued for final-export testing |
| Tokenizer lane | Lossless CaseOps, word-boundary BPE/Unigram, vocab sweeps | Legal path is exact byte sidecars and reversible transforms, not lossy whole-word shortcuts |

These are local proxy numbers, not official leaderboard submissions. The
experiment records are meant to explain candidate selection and grant-compute
priorities, not claim final scores.

## Repository Map

- `train_gpt.py` - main CUDA trainer with HRC routing, quantized export, LQER,
  mixed-quant forward training, and sub-16 speed probes.
- `train_gpt_ternary.py` - sub-4MB wrapper with profile presets and ternary
  experiment defaults.
- `ternary_golf/` - train-time ternary layers plus packed/dense ternary CUDA
  helper experiments.
- `scripts/run_sub4_iotail_quant_matrix.py` - mirrored IO-tail mixed-precision
  matrix, including wall-clock and final-artifact modes.
- `scripts/run_sub4_caseops_wide_matrix.py` - wide/shallow CaseOps candidate
  matrix.
- `scripts/run_16mb_vocab_moe_matrix.py` - 16MB Vocab-MoE control/probe
  matrix with final artifact round-trip enabled, including mainline cap-spend
  and dual-stream advisor candidate groups.
- `scripts/bench_packed_ternary_linear.py` - benchmark harness for packed
  ternary linear experiments.
- `scripts/check_cuda126_env.py` and `scripts/use_cuda126.ps1` - local CUDA
  12.6 environment checks for Windows/RTX 2060 SUPER iteration.
- `data/README.md` - dataset, tokenizer, fingerprint, and local proxy split
  workflows.
- `records/*.md` - human-written experiment ledgers and current conclusions.
- `records/experiment_synthesis_20260426.md` - compact read of the current
  state across sub-4, sub-16, tokenizer, and systems work.
- `levers.md` - catalog of the quality, speed, size, tokenizer, systems, and
  legality levers explored so far.
- `GRANT_SUMMARY.md` - short grant-application summary.

Generated run directories, checkpoints, datasets, tokenizer downloads, virtual
environments, local logs, and bundled runtimes are intentionally ignored.

## Useful Commands

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Download a small local FineWeb slice:

```bash
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

List sub-4 IO-tail quant candidates:

```bash
python scripts/run_sub4_iotail_quant_matrix.py --list
```

Run a 10-minute sub-4 IO-tail wall-clock matrix with train-time quantization:

```bash
python scripts/run_sub4_iotail_quant_matrix.py \
  --wallclock-seconds 600 \
  --final-artifacts \
  --train-quant-forward
```

Run the wide/shallow CaseOps matrix:

```bash
python scripts/run_sub4_caseops_wide_matrix.py \
  --mode local-wallclock \
  --wallclock-seconds 600 \
  --final-artifacts
```

Run the 16MB Vocab-MoE matrix:

```bash
python scripts/run_16mb_vocab_moe_matrix.py --wait-for-idle-gpu
```

Run the selective 16MB cap-speed scout after the active queue:

```powershell
.\scripts\queue_16mb_cap_speed_after_current.ps1 -WaitPid <queue_pid>
```

List the two next 16MB candidate families:

```bash
python scripts/run_16mb_vocab_moe_matrix.py --candidate-group cap16_mainline --list
python scripts/run_16mb_vocab_moe_matrix.py --candidate-group cap16_leaderboard --list
python scripts/run_16mb_vocab_moe_matrix.py --candidate-group cap16_dual_stream --list
```

Queue the mainline scout after an existing queue process. Add `-RunDual` only
after the cap-speed/mainline evidence says the extra dual-stream matmuls are
worth spending GPU time on:

```powershell
.\scripts\queue_16mb_mainline_dual_after_current.ps1 -WaitPid <queue_pid>
.\scripts\queue_16mb_mainline_dual_after_current.ps1 -WaitPid <queue_pid> -RunDual
```

For unattended queue fill, queue the selective 5k exploration tail after the
mainline scout. It first runs the leaderboard-blend HRC/VocabMoE rows at 5k
steps, then reads the finished cap-speed/mainline CSVs, reruns only the best
export-roundtrip candidates at 5k steps, and runs dual-stream canaries only if
a mainline scout clears the configured BPB threshold:

```powershell
.\scripts\queue_16mb_selective_overnight.ps1 -WaitPid <mainline_queue_pid>
```

Run the focused spike/self-election Vocab-MoE matrix only if the two corrected
spike probes in the active pruned queue look promising:

```powershell
.\scripts\queue_vocabmoe_spike_focused_after_current.ps1
```

Probe packed ternary linear speed:

```bash
python scripts/bench_packed_ternary_linear.py
```

## Important Environment Switches

- `SUBMISSION_SIZE_CAP_BYTES=4000000` - decimal 4MB cap for the sub-4 lane.
- `MODEL_CODEC=lzma` - preferred model codec for tiny artifacts.
- `TRAIN_QUANT_FORWARD=1` - train with quantized forward views from step one.
- `LQER_ENABLED=1` - export low-rank quantization-error residual sidecars.
- `HRC_ROUTE_REPEATS`, `HRC_RECURSIVE_CORE_START` - define the looped middle.
- `HRC_LOOP_INDEX_ENABLED=1` - tells the recurrent middle where it is in the
  loop.
- `HRC_FROZEN_CARRY_ENABLED=1` - tests a cheap recurrent carry route.
- `VOCAB_MOE_ENABLED=1` - enables token-conditioned shared low-rank experts in
  the 16MB lane.
- `VOCAB_MOE_LAYERS=input,loop_first` - places the adapter at embeddings and/or
  selected virtual HRC layers.
- `VOCAB_MOE_TRAIN_QUANT_BITS=6` - trains the Vocab-MoE adapter against a q6
  forward view from the first step.
- `VOCAB_MOE_MODE=spike_static|spike_hybrid|spike_hidden` and
  `VOCAB_MOE_SPIKE_TOP_K=1|2` - hard top-k self-election variants.
- `VOCAB_MOE_PRIOR_INIT_STD=0.01` - used by spike rows as a tiny per-token
  tie-breaker so hard routing does not start with every token selecting the
  same expert.
- `QUANT_FORCE_PATTERNS=vocab_moe.token_prior.weight,vocab_moe.down,vocab_moe.up`
  keeps Vocab-MoE train/export precision aligned for truthful matrix scores.
- `TRAIN_FUSED_QKV=1`, `USE_GRAD_SCALER=0`, `MUON_DTYPE=fp16` - speed probes
  carried from sub-4 into sub-16.
- `DUAL_STREAM_ENABLED=1` plus `DUAL_STREAM_LEFT_DIM`, `DUAL_STREAM_RANK`, and
  `DUAL_STREAM_SITES=input,loop_first,pre_output` - trained left/right advisor
  bridge for 16MB candidates.
- `LR_MIN_SCALE=0.026`, `MUON_NS_VARIANT=polar_express`,
  `SPARSE_ATTN_GATE_ENABLED=1`, `PARALLEL_RESIDUAL_LAST_N`, and
  `TTT_SCORE_FIRST_ENABLED=1` - leaderboard-inspired 16MB probe levers.

## What Is Not Included

This repo does not check in large datasets, checkpoints, generated run
directories, tokenizer downloads, or local CUDA/Python runtime bundles. Those
are recreated through the scripts above.

## License And Attribution

MIT licensed. This fork preserves the upstream Parameter Golf license and
third-party notices, with additional local experiment code and documentation.
See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
