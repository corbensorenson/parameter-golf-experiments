# Records And Documentation Map

Last organized: 2026-04-30

This folder contains two different kinds of material:

- Human-written research notes, plans, and synthesis docs.
- Generated run directories with `train.csv`, `candidate_plan.md`, logs, and
  artifact summaries.

Treat the generated run directories as raw evidence. Treat the docs below as
the interpretation layer.

## Read First

Start here when coming back to the project:

1. `../README.md`
   - Project overview, major model ideas, repo layout, and common commands.
2. `./architecture_explainer_20260430.md`
   - Submission-facing explanation of MirrorLoop/HRC, LexLoRE/VocabMoE,
     train-time quantization, LQER, width ladders, dual-stream bridges, and
     what is actually novel.
3. `./h100_speed_audit_20260430.md`
   - Current paid RunPod/H100 ledger, including live/queued jobs, corrected
     H100 speed settings, cap problems, and the latest cloud results.
4. `./h100_breakcliff_results_20260430.md`
   - Latest 1xH100 architecture scout. Prime-skip HRC is the new best
     architecture signal and has the cleanest next-step headroom.
5. `../levers.md`
   - Full lever catalog: quality, speed, size, tokenizer, legality, and systems
     knobs explored so far.
6. `./experiment_synthesis_20260426.md`
   - Older but still useful synthesis across sub-4, sub-16, tokenizer, and
     systems work.
7. `./overnight_synthesis_20260427.md`
   - Close-out of council/RLM, spike VocabMoE, sub-4 soft-size, cap-speed, and
     top-2 promotion runs.

## Current Truth Snapshot

As of 2026-04-30, the project has pivoted from "official-record contender on
local hardware" toward "novel, auditable art/non-record candidate with paid
H100 scaling evidence."

The strongest current ideas are:

- MirrorLoop/HRC mirrored IO tail plus looped middle.
- Train-time quantized forward, not export-only quantization.
- LexLoRE/VocabMoE at input and loop-entry for the 16MB family.
- Factored tied embeddings so bytes can be spent deliberately.
- LQER sidecars for export-roundtrip repair.
- QK gain, Polar/MIN_LR, fused QKV, no grad scaler, reduced debug scans, and
  other speed/settings transfers from public leaderboard work.

Important current results:

- Around-4MB H100 result:
  `i3l3r3_d768e256_q884_coret_lqer_r6t12` reached `2.1052` BPB at
  `84.14ms/step`, but exported to `4,238,717` bytes, so it is an around-4MB
  art signal rather than a strict decimal sub-4MB result.
- Over-cap H100 result:
  `h100_hrc_dual_i3l5r2_d768e2560_q8_coremlp_left320_embedq` reached `1.4894`
  BPB at `224.03ms/step`, but exported to `22,584,399` bytes.
- Best legal 8xH100 evidence:
  `final8x_legal_196k_r2_d704e768_w2200_wd02_lqer6t12_vocabmoe_qk55`
  reached `1.35496419` BPB at `90.13ms/step`, with `15,989,749` bytes. This is
  the best official-shaped self-funded evidence, but the 8x run bought only a
  tiny final-BPB improvement over the 1x rows.
- New best 1xH100 architecture signal:
  `break_prime_skip_superloop_d640e768` reached `1.35504224` BPB at
  `107.87ms/step`, with `14,051,162` bytes. It beats the pod control, has
  almost `1.95MB` headroom, and should be the next H100 cap-spend anchor.
- Best under/around-8MB local evidence:
  q6 HRC/VocabMoE rows around `6.2MB` reached about `1.87` BPB locally. That
  lane is better than the ternary IO-tail lane when quality matters more than
  strict 4MB size.

## H100 And RunPod Docs

- `h100_speed_audit_20260430.md`
  - Live source of truth for H100 speed, status, and results.
  - Includes the corrected 1xH100 batch policy, cap-legal follow-up, sub-4 H100
    probe, and known cap/export issues.
- `h100_breakcliff_results_20260430.md`
  - Two-hour 1xH100 architecture scout after the 8x plateau.
  - Promotes prime-skip HRC as the next clean novel route and demotes heavier
    LexLoRE, bigram side features, all-core attention, and the tested
    train-time q16/q8/q8 width ladder.
- `h100_novel_contender_plan_20260429.md`
  - Paid-cloud candidate plan from before the first H100 runs.
  - Useful for intent, but superseded by the speed audit for actual results.
- `runpod_no_fetch_plan_20260429.md`
  - Self-contained bundle/runbook for RunPod use without pulling code/data from
    GitHub at runtime.
- `local_h100_preflight_plan_20260429.md`
  - Local RTX 2060 gate for paid H100 candidates.
  - Historical now that H100 runs are live.

## Model-Lane Docs

- `architecture_explainer_20260430.md`
  - Best single document for describing what HRC and VocabMoE actually are.
  - Uses public names: MirrorLoop Recurrent Core and Lexical Low-Rank Experts.
- `sub4_competitive_plan_20260425.md`
  - Historical sub-4 plan and matrix reasoning.
  - Read with the later H100 sub-4 result in mind: strict ternary sub-4 is very
    fast but quality-limited.
- `sub16_competitive_plan_20260425.md`
  - Early 16MB transfer plan.
- `frontier_capfill_plan_20260429.md`
  - Selective 16MB cap-fill plan based on public-leaderboard signals and our
    best local HRC/VocabMoE spine.
  - Useful, but some "near-cap" assumptions were invalidated by full H100
    exports landing above 16MB.
- `structure_synthesis_20260429.md`
  - Summary of dual-stream, hourglass, and IO-tail-ladder interaction probes.
  - Current read: dual stream remains interesting, hourglass was not a clean
    win in the tested form.
- `art_showcase_plan_20260429.md`
  - Non-record/art-lane matrix: prime routes, spike VocabMoE, RLM-lite,
    council distributions, dual streams, and width/precision experiments.
- `breakout_plan_20260429.md`
  - Broad search plan for escaping the local `1.5x` BPB band.
  - Some rows were intentionally abandoned when dense/control rows looked less
    useful than the novel HRC/VocabMoE path.

## Research And Rule Docs

- `parameter_golf_competitor_research_20260425.md`
  - Public leaderboard and PR research snapshot.
- `prior_work_model_insights_20260427.md`
  - Prior-work map for recurrence, MoE, recursive/context memory, dynamic
    depth, and quantization ideas related to this project.
- `leader_lever_matrix_20260426.md`
  - Public-leaderboard trick matrix and how those tricks map onto our code.
- `tokenizer_rules_research_20260425.md`
  - Tokenizer/rules research, especially lossless tokenizer constraints.

## Older Sub-4 Result Docs

These are useful historical records but not current source-of-truth:

- `sub4_vs_16mb_gap_findings_20260424.md`
- `sub4_d384_wallclock_findings_20260424.md`
- `sub4_quality_rescue_findings_20260424.md`
- `sub4_quality_speed_findings_20260424.md`
- `sub4_micro_1k_findings_20260424.md`

They explain why the project moved from tiny ternary-only models toward wider
q6/q8 VocabMoE and HRC variants for quality.

## Queue And Watchdog Logs

- `hourly-watchdog.md`
  - Operational log from local unattended runs.
- `queue-prune-notes-20260427.md`
  - Queue triage, pruned candidate rationale, and status notes.

These are working logs, not polished conclusions.

## Generated Run Directories

Directories such as `cap16-frontier-capfill-5k-auto-*`,
`vocabmoe16-5k-auto-*`, `sub4-*`, and `promote-top2-*` are run artifacts.

The normal pattern is:

- `candidate_plan.md` says what was intended.
- `train.csv` is the machine-readable result table.
- `summary.md` is the generated or hand-summarized outcome.
- Logs/checkpoints/artifacts may be present depending on the run.

Do not hand-edit `train.csv`. If a result interpretation changes, update a
human-written synthesis doc instead.

## Known Superseded Claims

When reading older docs, watch for these traps:

- Any statement that e2688/e2560 H100 rows are close to 16MB is superseded by
  final H100 exports around `22.6-23.7MB`.
- Any "strict sub-4" claim before train-time embedding quantization should be
  checked against the later H100 around-4 result.
- Any claim that the best 1xH100 route is the plain batch32k r2 spine is
  superseded by the prime-skip break-cliff row at `1.35504224` BPB and
  `14,051,162` bytes.
- Early H100 smoke results accidentally used baseline model routing before
  `MODEL_FAMILY=hrc` was forced in HRC helpers.
- Local wall-clock comparisons from periods where the desktop was used for
  games are weaker timing evidence than the later controlled matrices.
- Large old queues are not necessarily endorsed candidates. The current policy
  is selective GPU use: run only candidates that test a high-value hypothesis.

## Maintenance Rule

When a new run finishes:

1. Put raw evidence in the generated run directory.
2. Update the one current synthesis file that owns the topic.
3. If the result changes candidate selection, update this index and
   `h100_speed_audit_20260430.md` if H100/RunPod is involved.
