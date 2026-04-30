# Parameter Golf Experiments

This is Corben's working repository for Parameter Golf experiments. It starts
from `openai/parameter-golf`, but the purpose of this fork is narrower: test
very small language-model candidates, especially sub-4MB artifacts, and figure
out which ideas are worth scaling to the official 10-minute 8xH100 setting.

For the original challenge rules, leaderboard, and submission process, see the
upstream repository:
<https://github.com/openai/parameter-golf>

## Submission Snapshot

This project is now being presented as a **non-record / art-lane** submission,
not an official SOTA claim.

- Submission folder:
  `records/track_non_record_16mb/2026-04-30_Corben_MirrorLoop_LexLoRE_HRC/`
- Public experiment repository:
  <https://github.com/corbensorenson/parameter-golf-experiments>
- Submission fork branch:
  <https://github.com/corbensorenson/parameter-golf/tree/nonrecord-mirrorloop-lexlore>

Best preserved under-cap 8xH100 evidence:

| Candidate | Final export BPB | Steps | Step speed | Bytes |
| --- | ---: | ---: | ---: | ---: |
| `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer6t12_vocabmoe_qk55` | `1.35496419` | `6658` | `90.13ms` | `15,989,749` |

Best late architecture signal:

| Candidate | Final export BPB | Steps | Step speed | Bytes |
| --- | ---: | ---: | ---: | ---: |
| `break_prime_skip_superloop_d640e768` | `1.35504224` | `5563` | `107.87ms` | `14,051,162` |

The prime-skip row is documented as follow-up architecture evidence, not the
main submitted 8x score.

## Documentation First

The experiment notes are intentionally detailed, but they are no longer all
equal source-of-truth. Start with `SUBMISSION.md` for the final reviewer-facing
summary, then `records/README.md` for the documentation map and
`records/h100_speed_audit_20260430.md` for the RunPod/H100 ledger. For the
architecture vocabulary used in grant/non-record writeups, read
`records/architecture_explainer_20260430.md`.
Generated run directories under `records/` are raw evidence; the human-written
Markdown files are the interpretation layer.

## Main Idea

Public-facing names:

- **MirrorLoop Recurrent Core** is the code path called `HRC`.
- **Lexical Low-Rank Experts (LexLoRE)** is the code path called `VocabMoE`.

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
- **Hourglass width schedules.** Selected HRC blocks can run internally at a
  narrower width and project residual deltas back to the full stream, testing
  the idea that high-precision IO blocks can be narrower while the low-precision
  recurrent core gets the width.
- **Lossless tokenizer probes.** CaseOps and word-boundary-aware tokenizer
  experiments are tracked separately from architecture changes so byte
  accounting stays auditable.

## Current Experiment Lanes

This table is an orientation layer. For the newest paid-cloud results and any
candidate currently running, defer to `records/h100_speed_audit_20260430.md`.

| Lane | What it tests | Current local read |
| --- | --- | --- |
| Promoted sub-4MB q884 IO-tail | `i3l3r3`, q8/q8/q4 IO blocks, ternary core, LQER, train-time quantized forward | Best clean legal row: `2.5749` BPB, `148.36ms/step`, `3,967,875` bytes |
| Soft-cap q884 quality reference | Same family with slightly larger residual spend | `2.5505` BPB, but `4,035,469` bytes, about `35KB` over the decimal 4MB goal |
| Precision-ladder IO tail | q16/q8/q4/q2/ternary entry and mirrored exit from the first training step | Legal and fast; best row `2.9888` BPB, so d512/e192 is under-capacity |
| Loop-index recurrence | Whether the looped middle benefits from virtual-position information | Helps r9 and i5/l5, hurts q884 r3; do not enable blindly |
| Sub-16MB transfer lane | Ports useful sub-4 speed and quality levers into a less byte-starved model | Local q6 proof baseline: `1.7567` final BPB, `9.27MB` artifact |
| 16MB LexLoRE lane | Token-conditioned shared low-rank experts on top of the MirrorLoop/CaseOps stack | Current best export-honest row is `i3/l5/r5 d640/e512/q8 Polar/MIN_LR` at `1.5336` BPB and `14,091,166` bytes; more unique loop blocks, e640 token-interface spend, and Polar/MIN_LR all improved quality |
| 16MB structure follow-up | Dual-stream, hourglass, and IO-tail quant-ladder variants on the current best `i3/l5/r5 d640/e512/q8` spine | Scored: dual-only was a near miss at `1.5440`, hourglass-only was worse at `1.5534`, and the combo row exported worse at `1.5565`; keep the simpler full-width single-stream spine |
| 16MB frontier cap-fill | Selective public-leaderboard transfers on the current best `i3/l5/r5` q8 spine | Completed: Polar/MIN_LR is the new best at `1.5336` BPB; e640 rank spend is second at `1.5368`; LQER was a small repair at `1.5407`; QK+parres was negative at `1.5489` |
| 16MB breakout matrix | Broad 5k scouts for getting out of the `1.5x` local band: dense SP8192-like controls, physical-depth spend, full-K/V attention, legal TTT, BigramHash, more unique loop blocks, and one dual-stream row | Stopped after the first dense rows failed/crashed and the project pivoted away from official-record chasing on local hardware |
| 16MB non-record art showcase | Weird-but-auditable architecture exhibits for `records/track_non_record_16mb`: prime skip routes, mirrored IO tails, width/precision ladders, spike VocabMoE, RLM-lite memory, council distributions, and trained dual streams | Active in `records/cap16-art-showcase-2k-auto-20260429-173955`; six 2k smoke rows with final-export scoring and live 250-step logs. See `records/art_showcase_plan_20260429.md` |
| H100 novel contender lane | Paid-cloud scaling read for our own MirrorLoop/LexLoRE family, not a vanilla leaderboard clone | Best legal 8x result: `1.35496419` BPB at `15,989,749` bytes. Best new 1x architecture signal: `break_prime_skip_superloop_d640e768`, `1.35504224` BPB, `107.87ms/step`, `14,051,162` bytes, with almost `1.95MB` headroom. |
| 16MB leaderboard-blend lane | Ports public-leaderboard and prior-work tricks onto our MirrorLoop/LexLoRE spine | Focused scout and best-2 promotion completed; sparsegate, depth-LoRA, cycle-rev, d768/d896 width, and the fp16-param speed profile did not beat the d640/e256 anchor |
| 16MB spike/self-election LexLoRE | Hard top-k token/expert election variants of LexLoRE/VocabMoE | Corrected spikehybrid is real and close: `1.8792` BPB, but still behind the dense input+loop-first anchor |
| Tokenizer lane | Lossless CaseOps, word-boundary BPE/Unigram, vocab sweeps | Legal path is exact byte sidecars and reversible transforms, not lossy whole-word shortcuts |

These are local proxy numbers, not official leaderboard submissions. The
experiment records are meant to explain candidate selection and grant-compute
priorities, not claim final scores.

## Repository Map

- `train_gpt.py` - main CUDA trainer with HRC routing, quantized export, LQER,
  mixed-quant forward training, and sub-16 speed probes.
- `SUBMISSION.md` - final reviewer-facing summary: claim boundary, best
  preserved evidence, architecture pitch, and negative findings.
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
- `scripts/run_h100_novel_matrix.py` - H100/8xH100 launcher for the novel
  MirrorLoop/LexLoRE contenders using the real trainer instead of the RTX 2060
  wrapper.
- `scripts/run_h100_breakcliff_matrix.py` - 1xH100 architecture scout for
  escaping the `~1.35` exported-BPB plateau; includes prime-skip HRC, LexLoRE
  exit/spike variants, core-attention probes, width/precision ladders, and
  preflight export guards.
- `scripts/bench_packed_ternary_linear.py` - benchmark harness for packed
  ternary linear experiments.
- `scripts/check_cuda126_env.py` and `scripts/use_cuda126.ps1` - local CUDA
  12.6 environment checks for Windows/RTX 2060 SUPER iteration.
- `data/README.md` - dataset, tokenizer, fingerprint, and local proxy split
  workflows.
- `records/README.md` - documentation index, current source-of-truth map, and
  notes about superseded claims.
- `records/architecture_explainer_20260430.md` - public/writeup explanation of
  MirrorLoop/HRC, LexLoRE/VocabMoE, train-time quantization, LQER, width
  ladders, dual-stream bridges, and what is actually novel.
- `records/*.md` - human-written experiment ledgers and current conclusions.
- `records/h100_speed_audit_20260430.md` - live H100/RunPod speed, result, and
  queue ledger.
- `records/h100_breakcliff_results_20260430.md` - latest H100 architecture
  scout; prime-skip MirrorLoop/HRC is the current best 1xH100 route signal.
- `records/experiment_synthesis_20260426.md` - compact read of the current
  state across sub-4, sub-16, tokenizer, and systems work.
- `records/overnight_synthesis_20260427.md` - close-out of the overnight
  council/RLM, spike, sub4 soft-size, cap-speed, focused 16MB, and promoted
  top-2 runs.
- `records/structure_synthesis_20260429.md` - close-out of the 16MB
  dual-stream, hourglass, and IO-tail-ladder interaction probes.
- `records/frontier_capfill_plan_20260429.md` - current selective 16MB
  cap-fill plan based on the best local spine and public leaderboard signals.
- `records/breakout_plan_20260429.md` - current broad search plan for
  escaping the single-HRC-spine local optimum and testing `<1.2`-relevant
  branches.
- `records/art_showcase_plan_20260429.md` - current non-record/art-lane plan:
  a small matrix of unusual but auditable HRC, VocabMoE, RLM, council, dual
  stream, and width/precision arrangements.
- `records/h100_novel_contender_plan_20260429.md` - paid-cloud runbook for
  testing the best novel MirrorLoop/LexLoRE rows under official-style H100
  batch/context settings.
- `records/runpod_no_fetch_plan_20260429.md` - RunPod workflow for uploading a
  local self-contained bundle and running without git/data downloads.
- `records/local_h100_preflight_plan_20260429.md` - local RTX 2060 gate for
  the same paid-candidate ideas before spending cloud money.
- `records/prior_work_model_insights_20260427.md` - prior-work map for the
  MirrorLoop/LexLoRE architecture and the concrete candidate changes it
  suggests.
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
python scripts/run_16mb_vocab_moe_matrix.py --candidate-group cap16_frontier_capfill --list
python scripts/run_16mb_vocab_moe_matrix.py --candidate-group cap16_frontier_followup --list
python scripts/run_16mb_vocab_moe_matrix.py --candidate-group cap16_art_showcase --list
python scripts/run_16mb_vocab_moe_matrix.py --candidate-group cap16_leaderboard --list
python scripts/run_16mb_vocab_moe_matrix.py --candidate-group cap16_dual_stream --list
```

Run the hourly local frontier watchdog once:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\hourly_frontier_watchdog.ps1
```

List the H100 novel contender rows:

```bash
python scripts/run_h100_novel_matrix.py --candidate-group h100_novel_round1 --list
```

List the local H100-spend preflight rows:

```bash
python scripts/run_16mb_vocab_moe_matrix.py --candidate-group cap16_h100_preflight --list
```

Queue local H100-spend preflight after an active queue:

```powershell
.\scripts\queue_local_h100_preflight_after_current.ps1 -WaitPid <queue_pid> -Iterations 2000 -ValTokens 65536
```

Build a no-fetch RunPod bundle from this machine:

```powershell
.\scripts\make_runpod_novel_bundle.ps1
```

On the pod after uploading/extracting the archive:

```bash
bash scripts/runpod_run_novel_no_fetch.sh check
bash scripts/runpod_run_novel_no_fetch.sh smoke
```

Smoke the best HRC/VocabMoE row on a single H100:

```bash
python scripts/run_h100_novel_matrix.py \
  --candidate-group h100_novel_round1 \
  --candidates h100_novel_i3l5r5_d640e512_q8_polar \
  --nproc-per-node 1 \
  --wallclock-seconds 180 \
  --val-tokens 65536
```

Run the five-row 8xH100 novel scout:

```bash
python scripts/run_h100_novel_matrix.py \
  --candidate-group h100_novel_round1 \
  --nproc-per-node 8 \
  --wallclock-seconds 600 \
  --val-tokens 131072
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

The corrected spike/self-election Vocab-MoE matrix has now been scored. The
best spikehybrid row reached `1.8792` BPB, close to but still behind the
`1.8710` dense VocabMoE anchor, so future spike work should stay tightly
targeted:

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
- `VOCAB_MOE_LAYERS=input,loop_first,loop_last` - LexLoRE read/write symmetry
  probe: token experts advise both the entry side and the mirrored exit side.
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
- `HRC_CYCLE_FUSE_ENABLED=1` - lets later recurrent/mirrored phases receive
  compact learned summaries from first-pass HRC states.
- `VE_ENABLED=1`, `VE_DIM`, and `VE_LAYERS` - ValueEmbedding probes that inject
  token-conditioned value features directly into selected attention blocks.
- `HRC_DEPTH_SCHEDULE_MODE=transition_tailN_cycle` in
  `train_gpt_arch_evolution.py` - cloned-trainer partial recurrence, e.g.
  `012|34567|34567|567|210`.
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
