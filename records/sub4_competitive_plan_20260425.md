# Sub-4MB Competitive Plan

Date: 2026-04-25

## Public Baseline To Beat

Current online audit:

- Neural leader path: PR #1797, 3-seed mean `val_bpb=1.06157`, using PR #1787
  CaseOps/SparseAttnGate/PolarNS/min-LR/FusedCE/warm-A TTT plus scalar
  SmearGate and asymmetric rank-4 LQER.
- Close neural branch: PR #1801, `val_bpb=1.06287`, using updated frozen
  recurrent carry stacked with the PR #1787 lineage.
- Legality-gated byte-model branch: PR #1795 reports a much lower PPM mixture
  score, but explicitly asks for organizer guidance. Keep this separate from
  the default neural lane until ruled safe.

Sources:

- <https://github.com/openai/parameter-golf/pull/1797>
- <https://github.com/openai/parameter-golf/pull/1801>
- <https://github.com/openai/parameter-golf/pull/1795>
- <https://github.com/openai/parameter-golf/issues/1017>

## Diagnosis

Our current promoted sub-4 local lane is technically correct but not yet
competitive:

- profile: `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075`
- preset: `2060sprint_micro_muon_cooltaper5k_cold_tokens8k`
- current-code 5k audit: `val_bpb=2.6522`
- compressed probe size: about 1.4MB total, leaving about 2.6MB unused under
  the decimal `4000000` byte cap

The gap to the public neural leader is about 1.59 BPB. That is too large for
schedule tuning alone. The sub-4 lane needs to spend its byte headroom on
quality and move the serious candidates to 8xH100-scale throughput.

## Implemented This Pass

New opt-in switches:

- `SMEAR_GATE_MODE=scalar`: public-style scalar residual smear gate.
- `SPARSE_ATTN_GATE_ENABLED=1`: public-style attention-output sparse gate.
- `MUON_WEIGHT_DECAY_MODE=huber`: Huber Muon decay with
  `MUON_WEIGHT_DECAY_HUBER_DELTA_SCALE`.

New sub-4 presets:

- `2060sprint_micro_muon_smear_scalar5k`
- `2060sprint_micro_muon_sparsegate5k`
- `2060sprint_micro_muon_smear_sparse5k`
- `2060sprint_micro_muon_huberwd5k`
- `2060sprint_micro_muon_lqer5k`
- `2060sprint_micro_muon_fcarry5k`
- `2060sprint_micro_muon_fcarry_lqer5k`
- `2060sprint_micro_muon_publicstack5k`
- `2060sprint_micro_muon_lqer_r8t16_5k`
- `2060sprint_micro_muon_lqerio_r8t16_5k`
- `2060sprint_micro_muon_lqerio_r16t24_5k`
- `2060sprint_micro_muon_lqerio_r16t32_5k`
- `2060sprint_micro_muon_fcarry_lqerio_r8t16_5k`
- `2060sprint_micro_muon_fcarry_lqerio_r16t24_5k`
- `2060sprint_micro_muon_fcarry_lqerio_r16t32_5k`
- `2060sprint_micro_muon_fcarry_lqerio_nodetach5k`
- `2060sprint_micro_muon_publicstack_smear5k`
- `2060sprint_micro_muon_publicstack_lqerio5k`

Continuation update:

- `TernaryLinear` now supports real LQER sidecars (`lqer_A`, `lqer_B`) on
  round-trip load. This matters because simply adding a dense residual into a
  ternary layer's latent weight gets ternarized away on forward.
- `LQER_ENABLED=1` exports asymmetric low-rank residual factors for the top-K
  residual tensors and reconstructs them as sidecar adapters for ternary
  weights, or dense corrections for non-ternary weights.
- Quant-train round trips explicitly disable LQER so export sidecars stay a
  final-artifact trick and do not mutate the training graph mid-run.
- `HRC_FROZEN_CARRY_ENABLED=1` adds the frozen alpha/beta carry route over
  repeated HRC middle states.
- The CaseOps wide matrix now includes LQER, frozen carry, combined
  frozen-carry+LQER, and publicstack candidates.
- A new export-aware 1k matrix mode writes final artifact round-trip BPB and
  byte headroom for each candidate, so quality comparisons use the actual
  submitted model path rather than only the train-time graph.

Validation:

- `py_compile` passed for `train_gpt.py`, `train_gpt_ternary.py`,
  `scripts/probe_sub4_profiles.py`, and
  `scripts/run_sub4_caseops_wide_matrix.py`.
- One-step real train-path smoke passed for the combined scalar-smear +
  sparse-gate preset.
- Probe for the combined preset estimated `1412499` total bytes, leaving
  `2587501` bytes of decimal-cap headroom.
- Probe for `2060sprint_micro_muon_fcarry_lqer5k` on
  `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` estimated `1441930` total bytes,
  leaving `2558070` bytes of decimal-cap headroom.
- Strict round-trip load check: `lqer_tensors=8`, `sidecar_modules=8`,
  `missing=[]`, `unexpected=[]`.
- One-step CaseOps artifact smoke passed with `int8/ternary8/lqer8+lzma`:
  total submission bytes `1823342`, headroom `2176658`, final round-trip
  `val_loss=9.0165`, `val_bpb=4.0472`.

## New 1k Screen

Record: `records/sub4_micro_matrix_20260425_022316`

Profile: `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075`

| Preset | Val loss | Val BPB | Step avg | Read |
|---|---:|---:|---:|---|
| `2060sprint_micro_muon_cooltaper5k_cold_tokens8k` | 8.2201 | 3.6897 | 64.04ms | Still best. |
| `2060sprint_micro_muon_sparsegate5k` | 8.2759 | 3.7147 | 65.67ms | Worth a retune, not default. |
| `2060sprint_micro_muon_huberwd5k` | 8.3150 | 3.7323 | 58.82ms | Faster, quality hit; retune WD. |
| `2060sprint_micro_muon_smear_scalar5k` | 8.3223 | 3.7356 | 65.25ms | Works, not default. |
| `2060sprint_micro_muon_smear_sparse5k` | 8.3740 | 3.7588 | 62.44ms | Stack hurts at 1k. |

Conclusion: do not promote the new neural switches blindly. They are now
implemented correctly enough to test. The cold d768/e256 preset was the prior
local default, but the newer export-aware matrix is already showing that
publicstack+IO-LQER deserves the next 5k/wall-clock promotion attempt.

## Export-Aware Matrix Results

Suite: `competitive-matrix-2060-20260425-031727`

Record: `records/competitive-matrix-2060-20260425-031727-sub4-export`

This completed sub-4 stage uses `--mode local-export1k --final-artifacts`,
decimal
`SUBMISSION_SIZE_CAP_BYTES=4000000`, CaseOps/SP8192 data, 32k validation tokens,
and 1k steps per candidate.

Completed rows:

| Preset | Train Val BPB | Final Export BPB | Step avg | Total bytes | Headroom |
|---|---:|---:|---:|---:|---:|
| `2060sprint_micro_muon_publicstack_lqerio5k` | 3.6354 | 3.6378 | 68.36ms | 2,840,775 | 1,159,225 |
| `2060sprint_micro_muon_publicstack_smear5k` | 3.6477 | 3.6746 | 69.39ms | 2,832,291 | 1,167,709 |
| `2060sprint_micro_muon_lqerio_r8t16_5k` | 3.7053 | 3.7092 | 65.55ms | 2,824,098 | 1,175,902 |
| `2060sprint_micro_muon_lqer_r8t16_5k` | 3.7020 | 3.7097 | 65.53ms | 2,819,482 | 1,180,518 |
| `2060sprint_micro_muon_fcarry5k` | 3.7222 | 3.7272 | 66.67ms | 2,794,518 | 1,205,482 |
| `2060sprint_micro_muon_cooltaper5k_cold_tokens8k` | 3.7318 | 3.7331 | 65.48ms | 2,781,902 | 1,218,098 |
| `2060sprint_micro_muon_fcarry_lqerio_r8t16_5k` | 3.7422 | 3.7441 | 66.83ms | 2,824,626 | 1,175,374 |
| `2060sprint_micro_muon_fcarry_lqerio_nodetach5k` | 3.7619 | 3.7658 | 67.53ms | 2,828,394 | 1,171,606 |

Provisional read: `publicstack_lqerio5k` is the clear winner in this matrix:
about 0.095 BPB better than the cold baseline at only a ~3ms/step cost, while
still leaving 1.16MB under cap. The scalar-smear publicstack variant trains
okay but loses too much on export round-trip, so do not promote it. Plain LQER
is a smaller win. Frozen carry helps by itself but does not stack with IO-LQER
at 1k, and no-detach carry is a clear reject in this screen.

Follow-up byte-spend candidates were added for the next matrix pass:
`lqerio_r16t24`, `lqerio_r16t32`, and frozen-carry variants of both. The r8/t16
IO-LQER row still leaves about 1.18MB, so these deliberately spend more of the
sub-4 cap on export recovery instead of capacity-free headroom.

Probe check for the most aggressive new candidate
`2060sprint_micro_muon_fcarry_lqerio_r16t32_5k` on d768/e256 estimated
`total_submission_bytes=1498763` and `headroom=2501237` on the untrained probe
path. Trained artifacts compress less favorably than the probe, but this is
enough headroom to justify a real matrix row.

## Highest-Leverage Next Work

1. H100 capacity ladder.
   The local 2060 should screen bugs and obvious losers only. Serious sub-4
   candidates are:

   - `i1l2r2_d1536_e384_h24kv1_mlpinner_mlp050`
   - `i1l2r2_d2048_e512_h32kv1_mlpinner_mlp025`

2. LQER/frozen-carry quality screen.
   Run the new LQER and frozen-carry presets at 5k minimum before promotion.
   The code path is now correct; the remaining question is quality, not wiring.

3. Legal TTT retune.
   Current control-only TTT helps itself but not the lead. The public lane uses
   phased LoRA/warm-A style adaptation, so the sub-4 version needs a tiny
   export-aware LoRA/control hybrid rather than only scalar/control updates.

4. Legality-gated byte mixture.
   If PR #1795-style PPM is ruled legal, build a minimal byte-level mixture
   branch. Until then, do not let it contaminate the default neural results.

## Current Default

Keep running local 2060 screens with:

```powershell
$env:SUB4_PROFILE='i1l2r2_d768_e256_h12kv1_mlpinner_mlp075'
$env:SUB4_SPEED_PRESET='2060sprint_micro_muon_cooltaper5k_cold_tokens8k'
$env:MODEL_CODEC='lzma'
$env:SUBMISSION_SIZE_CAP_BYTES='4000000'
```

Near-term promotion candidate:

```powershell
$env:SUB4_PROFILE='i1l2r2_d768_e256_h12kv1_mlpinner_mlp075'
$env:SUB4_SPEED_PRESET='2060sprint_micro_muon_publicstack_lqerio5k'
$env:MODEL_CODEC='lzma'
$env:SUBMISSION_SIZE_CAP_BYTES='4000000'
```

Promote it over the cold baseline only if it also wins at 5k or in the
600-second local wall-clock lane, not just because it wins the 1k screen.

## 10k Matrix Launched

The next sub-4 screen is a 10k export-aware matrix. It keeps decimal
`SUBMISSION_SIZE_CAP_BYTES=4000000`, CaseOps/SP8192 data, final artifact export,
and one final proxy validation per row.

Rows:

| Profile | Preset | Purpose |
|---|---|---|
| `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` | `2060sprint_micro_muon_cooltaper5k_cold_tokens8k` | Baseline control from earlier screens. |
| `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` | `2060sprint_micro_muon_publicstack_lqerio5k` | Prior 1k export winner. |
| `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` | `2060sprint_micro_muon_lqerio_r16t24_5k` | Spend headroom on larger IO-aware LQER. |
| `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` | `2060sprint_micro_muon_lqerio_r16t32_5k` | More aggressive LQER top-K spend. |
| `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` | `2060sprint_micro_muon_fcarry5k` | Retest carry-only signal at longer run length. |
| `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` | `2060sprint_micro_muon_fcarry_lqerio_r16t24_5k` | Carry plus large IO-aware LQER. |
| `i1l2r2_d768_e192_h12kv1_mlpinner_mlp050` | `2060sprint_micro_muon_publicstack_lqerio5k` | Thinner tied IO rank with winning lever stack. |
| `i1l2r2_d896_e256_h14kv1_mlpinner_mlp050` | `2060sprint_micro_muon_publicstack_lqerio5k` | Wider body at same tied IO rank. |
| `i1l2r2_d896_e256_h14kv1_mlpinner_mlp050` | `2060sprint_micro_muon_lqerio_r16t24_5k` | Wider body plus larger LQER sidecars. |
| `i1l2r2_d1024_e256_h16kv1_mlpinner_mlp050` | `2060sprint_micro_muon_publicstack_lqerio5k` | Larger body stress test. |
| `i2l3r2_d384_e128_h8kv1_mlpinner_mlp10` | `2060sprint_micro_muon_publicstack_lqerio5k` | Deeper recursive middle at lower width. |
| `i2l3r2_d384_e128_h8kv1_mlpinner_mlp15` | `2060sprint_micro_muon_publicstack_lqerio5k` | Deeper middle with more MLP capacity. |

Run command:

```powershell
.\\.venv-cuda313\\Scripts\\python.exe -u scripts\\run_sub4_caseops_wide_matrix.py --mode local-10k --iterations 10000 --val-tokens 65536 --timeout 2400 --out records\\sub4-caseops-10k-<timestamp> --final-artifacts
```

Active run:

- record dir: `records/sub4-caseops-10k-20260425-054606`
- runner stdout: `logs/sub4-caseops-10k-20260425-054606.out.txt`
- runner stderr: `logs/sub4-caseops-10k-20260425-054606.err.txt`
- first row log:
  `logs/matrix_i1l2r2_d768_e256_h12kv1_mlpinner_mlp075__2060sprint_micro_muon_cooltaper5k_cold_tokens8k__10000.txt`

Startup check: the first row reached training and logged the expected sub-4
settings: fp16 params/Muon, train-time ternary blocks, fused QKV, lzma codec,
and decimal 4,000,000 byte submission cap.

Status update:

The first 10k run was stopped after five full rows plus a partial sixth row.
All five completed rows stayed comfortably under 4MB, but final export
round-trip BPB was catastrophically worse than train-time BPB. Example:
`cooltaper5k_cold_tokens8k` ended at train-time `val_bpb=3.9750` but exported
round-trip `val_bpb=15.7419`. This is a useful finding: long fast-ternary runs
need export-honest training, not just final export.

Guarded relaunch:

- record dir: `records/sub4-caseops-10k-guarded-20260425-064355`
- runner stdout: `logs/sub4-caseops-10k-guarded-20260425-064355.out.txt`
- runner stderr: `logs/sub4-caseops-10k-guarded-20260425-064355.err.txt`

The guarded mode uses six focused rows and adds `QUANT_TRAIN_MODE=roundtrip`
from 30% of training onward, 128-token ternary groups, `LOGIT_SOFTCAP=12`,
fp32 loss, fp32 control params, full-run cosine warmdown, lower LR, and
validation every 1000 steps. First row startup is healthy:
`step:1000/10000 val_bpb=2.9054`, `step_avg=71.44ms`.

## IO-Tail Quantization Ladder

Implemented `scripts/run_sub4_iotail_quant_matrix.py` to test the mirrored
IO-tail precision idea without editing counted trainer files during active
runs. It uses existing export mechanics:

- `QUANT_BITS_OVERRIDES` targets unique block names such as `blocks.0.:8`.
- `QUANT_TERNARY_PATTERNS` targets the repeated middle block names.
- `QUANT_TRAIN_MODE=roundtrip` keeps training export-honest by periodically
  projecting weights through the mixed q8/q6/q4/ternary export path.

Important scope note: this is per unique HRC block, not per virtual occurrence.
For a route like `012|345|210`, block `2` is shared on the way in and way out,
so it receives one export precision. True per-virtual-layer precision would
require a trainer change after the current counted-code run finishes.

Candidate rows:

| Candidate | Route idea |
|---|---|
| `i3l3r2_d768e256_q864_coret` | IO blocks `0/1/2` as q8/q6/q4, repeated core `3/4/5` ternary. |
| `i3l3r3_d768e256_q864_coret` | Same IO ladder, one more middle repeat. |
| `i3l3r3_d768e256_q864_coret_lqer` | Same plus IO/core LQER sidecars. |
| `i6l9r3_d256e96_q886644_coret` | Wide IO tail `0..5` as q8/q8/q6/q6/q4/q4, core `6..14` ternary. |
| `i6l9r3_d256e96_q888666_coret` | Wider-precision IO tail q8/q8/q8/q6/q6/q6, core ternary. |
| `i6l9r3_d320e96_q886644_coret` | Wider d320 body with q886644 IO tail. |
| `i6l9r3_d320e128_q886644_coret_lqer` | d320/e128 plus LQER byte-spend. |

Scout run:

- record dir: `records/sub4-iotail-quant-3k-20260425-065158`
- combined log: `logs/sub4-iotail-quant-3k-20260425-065158.out.txt`
- summary: `records/sub4-iotail-quant-3k-20260425-065158/summary.md`
- all five non-LQER rows completed and validated cleanly under the decimal
  4,000,000 byte cap.
- both LQER rows trained and exported under cap, but failed final reload
  validation because exported `lqer_A`/`lqer_B` tensors are unexpected when
  loading the dequantized state into the base model. Treat their 3k BPB as a
  scout signal only until the reload path is fixed.

Best clean IO-tail quant rows at 3k:

| Candidate | Export BPB | Step avg | Bytes | Headroom |
|---|---:|---:|---:|---:|
| `i3l3r3_d768e256_q864_coret` | 2.7853 | 145.88 ms | 3,430,799 | 569,201 |
| `i3l3r2_d768e256_q864_coret` | 2.7905 | 134.87 ms | 3,431,443 | 568,557 |
| `i6l9r3_d256e96_q886644_coret` | 2.9952 | 165.60 ms | 2,224,495 | 1,775,505 |
| `i6l9r3_d256e96_q888666_coret` | 3.0821 | 166.02 ms | 2,459,971 | 1,540,029 |
| `i6l9r3_d320e96_q886644_coret` | 3.1086 | 216.38 ms | 2,875,403 | 1,124,597 |

The `i3l3r3` IO-tail ladder is the only IO-tail shape that beat the guarded
10k baseline at an early 3k checkpoint, but it is roughly 2.1x slower per step
than the best shallow guarded row on the 2060 SUPER. Promote it only if a
longer run keeps improving enough to justify the step cost.

LQER reload fix:

`train_gpt.py` now restores ternary LQER in a module-aware way. If the target
module supports `lqer_A`/`lqer_B` sidecars, the dequantizer emits them. If the
target is an ordinary dense/CastedLinear module, it folds the low-rank
correction directly into the restored weight so strict `load_state_dict`
validation stays clean.

Post-fix validation:

- smoke run: `records/sub4-iotail-lqer-reload-smoke-20260425`
- 3k rerun: `records/sub4-iotail-lqer-fixed-3k-20260425-171432`
- `i3l3r3_d768e256_q864_coret_lqer`: final export BPB `2.7550`, train proxy
  BPB `2.7550`, step avg `162.51 ms`, artifact `3,473,483` bytes, headroom
  `526,517` bytes, `returncode=0`.

This makes the IO-tail LQER lane submission-clean. It now reaches the same
quality neighborhood as the best guarded 10k shallow run by 3k steps, but it is
still much slower per local 2060 step.

## Guarded 10k Completion

The guarded 10k matrix completed all six rows cleanly:

- record dir: `records/sub4-caseops-10k-guarded-20260425-064355`
- summary: `records/sub4-caseops-10k-guarded-20260425-064355/summary.md`
- key result: export-honest roundtrip training fixed the earlier export BPB
  blow-up. Train-time proxy BPB and final exported BPB now match.

Best guarded rows:

| Profile / Preset | Export BPB | Step avg | Bytes | Headroom |
|---|---:|---:|---:|---:|
| `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` / `cooltaper5k_cold_tokens8k` | 2.7573 | 70.59 ms | 2,857,223 | 1,142,777 |
| `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` / `lqerio_r16t32_5k` | 2.7602 | 66.61 ms | 2,904,347 | 1,095,653 |
| `i1l2r2_d768_e256_h12kv1_mlpinner_mlp075` / `publicstack_lqerio5k` | 2.8880 | 67.78 ms | 2,840,115 | 1,159,885 |
| `i1l2r2_d896_e256_h14kv1_mlpinner_mlp050` / `publicstack_lqerio5k` | 2.9392 | 71.47 ms | 2,917,703 | 1,082,297 |
| `i2l3r2_d384_e128_h8kv1_mlpinner_mlp15` / `publicstack_lqerio5k` | 2.9540 | 68.39 ms | 1,824,759 | 2,175,241 |
| `i2l3r2_d384_e128_h8kv1_mlpinner_mlp10` / `publicstack_lqerio5k` | 3.0737 | 61.68 ms | 1,677,787 | 2,322,213 |

Current recommendation: use `i1l2r2_d768/e256 cooltaper5k_cold_tokens8k` as
the clean quality baseline, and keep `lqerio_r16t32_5k` as the speed-tie
candidate because it is slightly faster with only +0.0029 BPB. The next
quality experiment should spend the remaining 1.09-1.14MB on either clean
capacity or a reload-safe LQER/export fix, not on deeper narrow recursion.

## Train-Time Mixed Quantization

Lesson learned: the first IO-tail quant ladder was export-honest, but not the
full version of the idea. It used `QUANT_TRAIN_MODE=roundtrip` from step 1,
which periodically projected the stored weights through the export codec. That
prevented the final-export BPB cliff, but q8/q6/q4 IO blocks and export-ternary
core blocks still used ordinary dense CastedLinear matmuls between projections.

`train_gpt.py` now supports `TRAIN_QUANT_FORWARD=1`. When enabled, CastedLinear
modules are assigned train-time forward quantization from the same export
patterns:

- `QUANT_BITS_OVERRIDES` controls q8/q6/q4 linear STE views.
- `QUANT_TERNARY_PATTERNS` controls ternary STE views.
- small/control tensors still follow the existing keep-float policy.
- the HRC repeated middle is cached within one forward pass, so repeated block
  calls reuse the same quantized view instead of rematerializing it.

Scope note: this is still per unique HRC block, not per virtual occurrence. A
route like `012|345|210` gives block `2` one precision wherever that tied block
appears. True per-virtual-occurrence precision would require passing the virtual
layer index into shared modules or duplicating the tail blocks.

Smoke checks:

- `records/sub4-trainquant-smoke-20260425`: q8/q6/q4 + ternary + LQER, strict
  final reload clean.
- `records/sub4-trainquant-q886-smoke-20260425`: more aggressive q8/q8/q6 IO
  tail, strict final reload clean and under 4MB.

Superseded 10-minute wall-clock matrix:

- record dir: `records/sub4-trainquant-wallclock10m-20260425-182043`
- runner stdout: `logs/sub4-trainquant-wallclock10m-20260425-182043.out.txt`
- settings: `TRAIN_QUANT_FORWARD=1`, `MAX_WALLCLOCK_SECONDS=600`,
  `QUANT_TRAIN_MODE=roundtrip`, `QUANT_TRAIN_EVERY=100`, final artifacts,
  decimal `SUBMISSION_SIZE_CAP_BYTES=4000000`.
- result: invalid for clean train-time quant speed conclusions. It combined the
  new STE forward path with the old periodic export/reload projection. The
  shallow rows completed, but the deeper IO-tail rows crashed or timed out near
  the projection path.
- useful completed rows:
  - `i1l2r2_d768e256_q8_coret_lqer`: `2.7552` final BPB,
    `69.87ms/step`, `3,030,597` bytes, `969,403` headroom.
  - `i1l2r2_d896e256_q8_coret_lqer`: `2.7436` final BPB,
    `76.34ms/step`, `3,093,805` bytes, `906,195` headroom.
- failed rows:
  - `i3l3r2_d768e256_q864_coret_lqer`: Windows `0xC0000409` crash.
  - `i3l3r3_d768e256_q864_coret_lqer`: runner timeout after 900s.

Fix:

- `scripts/run_sub4_iotail_quant_matrix.py` now defaults
  `QUANT_TRAIN_MODE=none`.
- The old projection guardrail is opt-in via `--roundtrip-guard`.
- Clean wall-clock sweeps should use `--train-quant-forward` without
  `--roundtrip-guard`, so the training loop avoids export/reload conversions.

Active corrected 10-minute wall-clock matrix:

- record dir: `records/sub4-trainquant-clean-wallclock10m-20260425-190647`
- settings: `TRAIN_QUANT_FORWARD=1`, `QUANT_TRAIN_MODE=none`,
  `MAX_WALLCLOCK_SECONDS=600`, final artifacts, decimal
  `SUBMISSION_SIZE_CAP_BYTES=4000000`.
- first row startup confirms the intended semantics:
  `train_quant_forward:1`, `train_quant_linear_count:6`,
  `train_quant_ternary_count:4`, `quant_train_mode:none`,
  `max_wallclock_seconds:600`.

Completion update:

- The machine was also used for games during this sweep. Treat wall-clock
  ranking and step counts as contaminated; quality/export success is still
  useful, but fair speed claims need an idle rerun.
- Completed rows:
  - `i3l3r2_d768e256_q864_coret_lqer`: best local quality in this sweep,
    `2.5572` final BPB, `3,527,629` bytes, `472,371` headroom,
    `3347` wallclock-stop steps.
  - `i3l3r3_d768e256_q884_coret_lqer`: `2.6543` final BPB,
    `3,993,505` bytes, only `6,495` headroom, `2902` wallclock-stop steps.
  - `i1l2r2_d768e256_q8_coret_lqer`: `3.8918` final BPB,
    `2,955,769` bytes, `1,044,231` headroom.
  - `i1l2r2_d896e256_q8_coret_lqer`: `3.9234` final BPB,
    `3,033,977` bytes, `966,023` headroom.
- Failed rows:
  - `i3l3r3_d768e256_q864_coret_lqer`: timeout after 900s.
  - `i3l3r3_d768e256_q886_coret_lqer`: CUDA illegal memory access during the
    q8/q8/q6 train-time fake-quant forward path.
- Read: the serious follow-up is an idle rerun of
  `i3l3r2_d768e256_q864_coret_lqer` and `i3l3r3_d768e256_q884_coret_lqer`,
  plus a shorter debug repro of q886 with `CUDA_LAUNCH_BLOCKING=1`.
- Runner support: `scripts/run_sub4_iotail_quant_matrix.py` now has
  `--wait-for-idle-gpu`, `--idle-max-util`, `--idle-max-memory-mib`,
  `--idle-seconds`, and `--idle-poll-seconds` so future wall-clock sweeps can
  avoid starting candidates while the GPU is busy.

Active fair rerun:

- record dir: `records/sub4-trainquant-fair-wallclock10m-20260425-202436`
- candidates:
  - `i1l2r2_d768e256_q8_coret_lqer`
  - `i1l2r2_d896e256_q8_coret_lqer`
  - `i3l3r2_d768e256_q864_coret_lqer`
  - `i3l3r3_d768e256_q884_coret_lqer`
- settings: `TRAIN_QUANT_FORWARD=1`, `QUANT_TRAIN_MODE=none`,
  `MAX_WALLCLOCK_SECONDS=600`, final artifacts, idle guard enabled with
  `--idle-max-util 25`, `--idle-max-memory-mib 2500`, `--idle-seconds 30`.
- first candidate waited `30.311s` and started at `18%` GPU utilization and
  `994MiB` VRAM, so this should be a much cleaner local 2060 wall-clock
  comparison.

Fair rerun completion:

- record dir: `records/sub4-trainquant-fair-wallclock10m-20260425-202436`
- successful under-cap rows:
  - `i1l2r2_d896e256_q8_coret_lqer`: `3.8455` final BPB,
    `8813` wallclock-stop steps, `68.10ms/step`, `2,994,525` bytes,
    `1,005,475` headroom.
  - `i1l2r2_d768e256_q8_coret_lqer`: `3.8943` final BPB,
    `9160` wallclock-stop steps, `65.53ms/step`, `2,955,513` bytes,
    `1,044,487` headroom.
- failed/near-miss rows:
  - `i3l3r2_d768e256_q864_coret_lqer`: CUDA illegal memory access during
    train-time fake-quant forward, after about `10` logged steps.
  - `i3l3r3_d768e256_q884_coret_lqer`: strong quality but over cap:
    `2.5252` train/export proxy BPB, `4114` wallclock-stop steps,
    `145.88ms/step`, `4,060,837` bytes, `60,837` bytes over cap.
- Read: the clean fair run says q884-style IO-tail quality is the best signal,
  but the candidate needs a byte shave before it is legal. The q864 r2 row is
  not trustworthy until the illegal-memory issue is debugged with
  `CUDA_LAUNCH_BLOCKING=1`.

## q884 IO-Tail Finding

The fair rerun promoted `i3l3r3_d768e256_q884_coret_lqer` as the best current
sub-4 direction, even though it landed just over the strict decimal cap:

- score signal: `2.5252` BPB at the 600-second wallclock stop.
- throughput: `4114` steps at `145.88ms/step` on the local 2060 Super.
- artifact: `4,060,837` bytes, only `60,837` bytes over the decimal 4MB goal.
- route: `0,1,2,3,4,5,3,4,5,3,4,5,2,1,0`.
- train-time precision: q8/q8/q4 IO blocks plus ternary repeated core blocks.

Why it likely beat the other fair rows:

- It is the first train-time-quantized IO-tail row that spends the byte budget.
  The shallow q8 rows leave about 1MB unused and stop around `3.85-3.89` BPB.
- The mirrored IO tail gives a real entry/exit transform instead of asking one
  shallow tied block stack to do everything.
- The ternary middle is reused three times, buying effective depth without
  paying for three separate dense cores.
- q8/q8/q4 avoids the q6 train-time fake-quant path. Both q6-containing fair
  candidates hit CUDA illegal memory access on this Windows/PyTorch/CUDA stack,
  so q6 should be treated as suspect until separately debugged.
- LQER is probably carrying meaningful export recovery. The over-cap artifact
  reported `235,664` bytes of raw LQER payload, which is exactly the part we can
  squeeze first.

The curve is also informative: `3.0393` BPB at 1k, `2.5695` at 2k, `2.5237` at
3k, and `2.5252` at 4k. Most of the local 10-minute quality arrives by 3k
steps; further gains are more likely to come from byte allocation, the LR
tail/cooldown, and better sidecar choices than from only pushing longer local
runs.

Runner changes for the next pass:

- `scripts/run_sub4_iotail_quant_matrix.py` now exposes
  `LQER_FACTOR_BITS` through `lqer_env(...)`.
- It adds `--allow-over-cap`, which sets `FAIL_ON_ARTIFACT_CAP=0` so near-cap
  candidates still complete final export validation.
- New q884 byte/quality rows:
  - `i3l3r3_d768e256_q884_coret_lqer_t12`
  - `i3l3r3_d768e256_q884_coret_lqer_fb3`
  - `i3l3r3_d768e256_q884_coret_lqer_r6`
  - `i3l3r3_d768e256_q884_coret_lqer_t12fb3`
- Follow-up q884 loop-index rows:
  - `i3l3r3_d768e256_q884_coret_lqer_lidx`
  - `i3l3r3_d768e256_q884_coret_lqer_lidx_t12fb3`

Next targeted sweep: rerun q884 as a soft-cap baseline with exact final
roundtrip, then compare smaller LQER top-K/factor/rank variants. If the smaller
sidecars hold BPB while dropping below 4MB, promote the best legal row. If the
soft-cap baseline is clearly better, keep it as the 4MB-plus research lane and
try to recover the missing `60-100KB` from code size, LQER exclusions, or q4
payload compression.

Active targeted sweep:

- record dir: `records/sub4-q884-byte-quality-10m-20260425-211131`
- settings: 10-minute local wallclock per row, `TRAIN_QUANT_FORWARD=1`,
  `QUANT_TRAIN_MODE=none`, final artifacts, idle guard, and
  `--allow-over-cap`.
- rows:
  - `i3l3r3_d768e256_q884_coret_lqer`
  - `i3l3r3_d768e256_q884_coret_lqer_t12`
  - `i3l3r3_d768e256_q884_coret_lqer_fb3`
  - `i3l3r3_d768e256_q884_coret_lqer_r6`
  - `i3l3r3_d768e256_q884_coret_lqer_t12fb3`
- startup check: the first row waited `30.284s` for the idle guard and then
  began training with the GPU at full load.
- first-row update: the exact q884 soft-cap rerun crashed after the first few
  logged steps with Windows `0xC0000409` / CUDA illegal memory access reported
  at RoPE application. This is different from the previous fair q884 completion
  and should be treated as a local stability/repro issue, not a quality result.
  The runner continued to the `t12` sidecar row after a fresh idle wait.
- second-row checkpoint: `i3l3r3_d768e256_q884_coret_lqer_t12` reached 1k
  steps cleanly with `2.9395` BPB at `159.60ms/step`, so the q884 family is not
  globally broken. Let this row complete before making a byte/quality call.

Queued follow-up: the best q884 run did not enable
`HRC_LOOP_INDEX_ENABLED`, even though the looped-middle idea wants virtual-pass
position information. The trainer already supports a tiny sinusoidal loop-index
control path, so the next pass should compare the baseline and best byte-shaved
q884 rows with `HRC_LOOP_INDEX_ENABLED=1`, `HRC_LOOP_INDEX_DIM=32`.

Completion:

| Candidate | Final Export BPB | Step avg | Stop step | Bytes | Headroom |
|---|---:|---:|---:|---:|---:|
| `i3l3r3_d768e256_q884_coret_lqer_r6` | 2.5505 | 145.05 ms | 4138 | 4,035,469 | -35,469 |
| `i3l3r3_d768e256_q884_coret_lqer_fb3` | 2.5634 | 145.06 ms | 4138 | 4,061,901 | -61,901 |
| `i3l3r3_d768e256_q884_coret_lqer_t12` | 2.5779 | 150.76 ms | 3981 | 4,006,973 | -6,973 |
| `i3l3r3_d768e256_q884_coret_lqer_t12fb3` | 2.6524 | 145.07 ms | 4138 | 4,033,585 | -33,585 |

Read:

- `r6` is the best quality row in this sweep and is only 35KB over the 4MB
  goal. It is the new quality favorite for the q884 branch.
- `t12` is the closest-to-legal row, only 6,973 bytes over, but gives up about
  0.027 BPB versus `r6`.
- `factor_bits=3` is not a real byte lever while asymmetric LQER is enabled.
  The export path uses the asym int2/int4 factor layout when
  `LQER_ASYM_ENABLED=1`, so `LQER_FACTOR_BITS` is only meaningful for the
  symmetric fallback. Do not spend more matrix time on `fb3` in this lane.
- The exact q884 baseline rerun crashed early with CUDA illegal memory access,
  while the derived rows completed. Keep treating this as a local Windows/CUDA
  stability issue unless it reproduces on Linux/H100.

New follow-up candidates:

- `i3l3r3_d768e256_q884_coret_lqer_t11`: shave the near-legal `t12` row below
  cap with minimal expected quality loss.
- `i3l3r3_d768e256_q884_coret_lqer_r6t14`: see whether most of the `r6`
  quality can be kept while shaving bytes.
- `i3l3r3_d768e256_q884_coret_lqer_r6t12`: likely first truly under-4MB row in
  the higher-quality `r6` branch.
- `i3l3r3_d768e256_q884_coret_lqer_lidx_t11` and
  `i3l3r3_d768e256_q884_coret_lqer_lidx_r6t12`: test whether tiny loop-index
  conditioning helps the looped middle once the byte budget is legal.

Active follow-up:

- record dir: `records/sub4-q884-legal-lidx-10m-20260425-223322`
- settings: 10-minute local wallclock per row, final artifacts,
  `TRAIN_QUANT_FORWARD=1`, `QUANT_TRAIN_MODE=none`, `--allow-over-cap`, idle
  guard.
- rows:
  - `i3l3r3_d768e256_q884_coret_lqer_t11`
  - `i3l3r3_d768e256_q884_coret_lqer_r6t14`
  - `i3l3r3_d768e256_q884_coret_lqer_r6t12`
  - `i3l3r3_d768e256_q884_coret_lqer_lidx_t11`
  - `i3l3r3_d768e256_q884_coret_lqer_lidx_r6t12`
- startup check: the first row waited `30.287s` for the idle guard and then
  began training with the GPU at full load.

## i5/l5 Precision-Ladder Tail

New idea to test: make the outer IO path progressively lower precision from
the first training forward, then let the repeated middle stay ternary:

- entry: block `0` q16/fp16 passthrough, block `1` q8, block `2` q4,
  block `3` q2, block `4` ternary.
- middle: blocks `5-9` ternary.
- exit: mirrored reuse of blocks `4,3,2,1,0`, so the exit precision ladder is
  ternary/q2/q4/q8/q16.

Implementation audit:

- `transition_recursive_cycle` can represent the exact route. For r1 it is
  `0,1,2,3,4,5,6,7,8,9,4,3,2,1,0`.
- `TRAIN_QUANT_FORWARD=1` applies the q16/q8/q4/q2/ternary views from the first
  forward pass; this is not a final-only quantization experiment.
- `train_gpt.py` now accepts `2` and `16` in `QUANT_BITS_OVERRIDES`.
- q16 means fp16 passthrough in the export codec and unquantized fp16 matmul in
  the train-time forward path.
- q2 uses the existing linear fake-quant/export path with a per-row scale and
  low-cardinality int8 codes. This is train-time low precision, but not yet a
  custom packed q2 storage kernel; if q2 wins, packing q2 is the next systems
  cleanup.
- `route_env(...)` now separates `io_width` from `ternary_start`, so block `4`
  can be part of the mirrored IO shell while still being ternary.
- `HRC_MLP_ONLY_BLOCKS` is set to `5,6,7,8,9`, so the repeated core is cheap,
  while block `4` remains a full IO-tail block.

Candidates added:

- `i5l5r1_d512e192_q16q8q4q2t_coret_lqer_r6t12`
- `i5l5r1_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`
- `i5l5r2_d512e192_q16q8q4q2t_coret_lqer_r6t12`
- `i5l5r2_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`
- `i5l5r3_d512e192_q16q8q4q2t_coret_lqer_r6t12`
- `i5l5r3_d512e192_q16q8q4q2t_coret_lqer_lidx_r6t12`
- `i5l5r2_d448e160_q16q8q4q2t_coret_lqer_lidx_r6t12` as a smaller fallback if
  the d512 r2/r3 rows are too slow locally.

Route verification:

- r1 route: `0,1,2,3,4,5,6,7,8,9,4,3,2,1,0`
- r2 route: `0,1,2,3,4,5,6,7,8,9,5,6,7,8,9,4,3,2,1,0`
- r3 route:
  `0,1,2,3,4,5,6,7,8,9,5,6,7,8,9,5,6,7,8,9,4,3,2,1,0`

Validation:

- `py_compile` passed for `train_gpt.py`, `train_gpt_ternary.py`, and
  `scripts/run_sub4_iotail_quant_matrix.py`.
- `--list` shows all r1/r2/r3 i5/l5 precision-ladder candidates.
