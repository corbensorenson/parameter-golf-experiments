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
- **Lossless tokenizer probes.** CaseOps and word-boundary-aware tokenizer
  experiments are tracked separately from architecture changes so byte
  accounting stays auditable.

## Current Experiment Lanes

| Lane | What it tests | Current local read |
| --- | --- | --- |
| Sub-4MB shallow HRC | Wide, shallow, fast `i1l2r2` models with cool taper schedules | Best local proxy around `2.7573` BPB, `70.6ms/step`, `2.86MB` artifact |
| Sub-4MB IO-tail + LQER | Mirrored route, q8/q6/q4 IO layers, ternary core, LQER sidecars | Best 3k proxy around `2.7550` BPB, `162.5ms/step`, `3.47MB` artifact |
| Train-time quant matrix | Checks whether q8/q6/q4/ternary training from step one beats export-only quantization | Active candidate family includes `i3l3r3`, `i6l9r3`, q864/q884/q886 ladders |
| Sub-16MB transfer lane | Ports useful sub-4 speed and quality levers into a less byte-starved model | q6 proof profiles plus LQER, frozen carry, fused QKV, and fp16 Muon probes |
| Tokenizer lane | Lossless CaseOps, word-boundary BPE/Unigram, vocab sweeps | Tracked in tokenizer research docs; no lossy tokenizer shortcuts |

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
- `scripts/bench_packed_ternary_linear.py` - benchmark harness for packed
  ternary linear experiments.
- `scripts/check_cuda126_env.py` and `scripts/use_cuda126.ps1` - local CUDA
  12.6 environment checks for Windows/RTX 2060 SUPER iteration.
- `data/README.md` - dataset, tokenizer, fingerprint, and local proxy split
  workflows.
- `records/*.md` - human-written experiment ledgers and current conclusions.
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
- `TRAIN_FUSED_QKV=1`, `USE_GRAD_SCALER=0`, `MUON_DTYPE=fp16` - speed probes
  carried from sub-4 into sub-16.

## What Is Not Included

This repo does not check in large datasets, checkpoints, generated run
directories, tokenizer downloads, or local CUDA/Python runtime bundles. Those
are recreated through the scripts above.

## License And Attribution

MIT licensed. This fork preserves the upstream Parameter Golf license and
third-party notices, with additional local experiment code and documentation.
See `LICENSE` and `THIRD_PARTY_NOTICES.md`.
