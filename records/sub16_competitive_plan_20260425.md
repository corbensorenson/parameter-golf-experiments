# Sub-16MB Competitive Plan

Date: 2026-04-25

## Target

The strongest transformer/HRC public target is PR #1797 at 3-seed mean
`val_bpb=1.06157`, with PR #1801 close at `val_bpb=1.06287`. PR #1791 reports
`val_bpb=1.0339` for an FLA/GatedDeltaNet-family branch, but it is a larger
architecture branch and the public discussion includes byte-accounting concerns,
so treat it as research pressure rather than a direct HRC baseline. PR #1795
reports `val_bpb=1.01252` for a byte-level PPM mixture, but that remains
legality-gated for our neural default lane.

Sources:

- <https://github.com/openai/parameter-golf/pull/1797>
- <https://github.com/openai/parameter-golf/pull/1801>
- <https://github.com/openai/parameter-golf/pull/1791>
- <https://github.com/openai/parameter-golf/pull/1795>
- <https://github.com/openai/parameter-golf/issues/1017>

## What Changed

The sub-16 CUDA runner can now directly test the same two public-leader export
and routing ideas that were added for sub-4:

- `loopplain5k_i3l3r3_lqer_fcarry_q6proof`
- `loopplain5k_i3l3r3_lqerio_fcarry_q6proof`
- `loopplain5k_i3l3r3_lqerio_r16t24_fcarry_q6proof`
- `loopplain5k_i3l3r3_publicstack_q6proof`
- `compare5k_publicstack_q6`

These add:

- `LQER_ENABLED=1`, rank 8, top-K 12, asymmetric group-64 factors.
- IO-aware LQER variants with rank 8, top-K 16, and include patterns for
  `tok_emb.weight`, `embed_proj`, and `blocks.` while still excluding the tied
  `lm_head.weight` and tiny control tensors.
- A more aggressive rank-16/top-24 IO-aware LQER q6 proof profile, so sub-16
  can test whether the 16MB lane should spend more bytes on low-rank recovery
  instead of only widening the base graph.
- `HRC_FROZEN_CARRY_ENABLED=1`, with default repeated-core block selection.
- publicstack variant: attention output gate, smear gate, and Huber Muon decay.

## Current Read

Sub-16 should stay closer to the public leader recipe than sub-4:

- keep CaseOps/SP8192 data and byte sidecars;
- use the looped HRC middle route for the q6 proof lane;
- spend artifact bytes on q6 plus LQER instead of leaving quantization error on
  the table;
- keep frozen carry as a route-side candidate because it costs almost no
  artifact bytes and targets the repeated middle directly. The profile uses the
  default repeated-core selector; `HRC_FROZEN_CARRY_BLOCKS=all` also grabs
  mirrored IO-tail repeats and requires a larger carry matrix.

The largest remaining gaps versus the public top neural branch are full
warm-A/phased LoRA TTT and fused softcapped CE. Those are bigger branches than
today's safe changes and should be isolated before promotion.

## Sub-4 Speed Levers Ported To Sub-16

The sub-4 family taught us which training-loop switches are real wall-clock
levers on the local 2060 Super:

- fused QKV projection (`TRAIN_FUSED_QKV=1`);
- no grad scaler when model params are already lower precision
  (`USE_GRAD_SCALER=0`);
- lower precision Muon state where stable (`MUON_DTYPE=fp16`);
- skip fp32 CE promotion on probe runs (`LOSS_FP32=0`);
- zero grads before the next backward instead of doing extra post-step work
  (`POST_STEP_ZERO_GRAD=0`);
- no periodic validation during speed probes (`VAL_LOSS_EVERY=0`);
- avoid compile on the 2060 path unless a profile proves it pays back.
- targeted risk probes for sub-4-only tricks: no logit softcap, AdamW-only
  optimizer, sampled vocab loss, and seq_len=64 sprint geometry.

These are now available as opt-in sub-16 speed probes rather than replacing the
conservative q6 proof lane:

- `loopplain1k_i3l3r3_q6proof_speedprobe`
- `loopplain1k_i3l3r3_q6proof_fusedqkv_speedprobe`
- `loopplain1k_i3l3r3_q6proof_fp16_speedprobe`
- `loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_speedprobe`
- `loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_mb2_speedprobe`
- `loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_lvs4096_speedprobe`
- `loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_nosoftcap_speedprobe`
- `loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_adamw_speedprobe`
- `loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_seq64_speedprobe`

The quality default intentionally remains fp32-param / GradScaler / q6 proof
until the speed probes show stable loss movement. If a fast probe keeps loss
tracking the conservative lane, promote the matching 5k profile:
`loopplain5k_i3l3r3_q6proof_fastfp16` or
`loopplain5k_i3l3r3_publicstack_q6proof_fastfp16`.

Early baseline from the conservative row:
`loopplain5k_i3l3r3_q6proof` reached step 1000 at about `712.73ms/step` with
validation `val_bpb=1.9781` on the local proxy. The speed ladder should be
judged against that wall-clock number first, then against final validation
movement.

## How To Run

```powershell
$env:CANDIDATE2060_PROFILE='loopplain5k_i3l3r3_lqer_fcarry_q6proof'
.\\.venv-cuda313\\Scripts\\python.exe scripts\\run_caseops_candidate_2060_compare.py
```

For the wider near-cap lane:

```powershell
$env:CANDIDATE2060_PROFILE='compare5k_publicstack_q6'
.\\.venv-cuda313\\Scripts\\python.exe scripts\\run_caseops_candidate_2060_compare.py
```

For the serial competitive matrix that runs sub-4 export-aware rows first and
then the sub-16 q6 loop-proof ladder:

```powershell
.\\.venv-cuda313\\Scripts\\python.exe scripts\\run_competitive_matrix_2060.py
```

For the sub-16 speed-lever ladder:

```powershell
$env:LOOP_LADDER_MODE='speed'
$env:LOOP_LADDER_RUN_ID='sub16-speed-2060-YYYYMMDD-HHMMSS'
.\\.venv-cuda313\\Scripts\\python.exe scripts\\run_caseops_loop_ladder_2060.py
```

Or through the serial matrix harness:

```powershell
$env:COMP_MATRIX_STAGES='sub16_loopproof'
$env:COMP_MATRIX_SUB16_MODE='speed'
.\\.venv-cuda313\\Scripts\\python.exe scripts\\run_competitive_matrix_2060.py
```

Completed local suite:
`logs/competitive-matrix-2060-20260425-031727.suite.txt`.

- The sub-4 export-aware stage completed.
- The sub-16 baseline `loopplain5k_i3l3r3_q6proof` completed: final export
  `1.7567` BPB, `679.52ms/step` at 5k, `9,268,177` total bytes, and
  `7,509,039` bytes of headroom under the 16MB cap.
- The suite then stopped before the frozen-carry/LQER row produced a useful
  result. The first frozen-carry profile had used `HRC_FROZEN_CARRY_BLOCKS=all`
  with only the 3x3 core carry matrix. The profile has since been corrected to
  use the default repeated-core selector before the next resume run.

Current read: the conservative q6 proof baseline is healthy and well under cap.
The next sub-16 job should resume the corrected LQER/frozen-carry/publicstack
ladder, then compare speed probes only against rows whose loss movement tracks
this baseline.
