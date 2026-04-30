# 16MB Non-Record Art Showcase Plan

Date: 2026-04-29

Architecture vocabulary: this plan uses the historical code names `HRC` and
`VocabMoE` in candidate names. In submission prose, call them MirrorLoop
Recurrent Core and Lexical Low-Rank Experts (LexLoRE). See
`records/architecture_explainer_20260430.md` for the writeup-ready definitions.

This run pivots from "can we contend on the official 8xH100 leaderboard?" to
"can we show a coherent weird lane that works?" The upstream challenge has an
official `records/track_non_record_16mb` path for interesting approaches that
are experimental, not intended as a SOTA record, or not verified under the
10-minute 8xH100 record setting. That is the right home for this project unless
we later get official hardware verification.

The local art-showcase matrix is deliberately small: six 2k-step smoke runs,
final-export scored, with live logs every 250 steps. The point is not to beat
the current local best immediately. The point is to identify which unusual
mechanisms train, export, and produce honest signal, then promote one or two
rows to 5k and build the non-record submission narrative around them.

Active run directory:
`records/cap16-art-showcase-2k-auto-20260429-173955`.

## Candidates

| Candidate | What It Shows |
| --- | --- |
| `art_prime_skip_spike_i3p5s1r2_d640e512_q8` | Prime-width recurrent core with a nontrivial skip-walk route, hard token self-election, and q8 export. This replaces the first smoke row that wedged after step 10 when dual-stream was also enabled. |
| `art_pal_ladder_hourglass_dual_spike_i3l5r1_d640e512_q8` | Mirrored palindrome IO tail, q16/q8/q4 train-time IO ladder, head-compatible hourglass block widths `400,440,520,640,640,640,640,640`, spike VocabMoE, and dual stream in one compact exhibit. |
| `art_rlm_council_hybrid_i3l5r5_d640e512_q8` | Legal recursive prefix memory plus a mirrored three-peer council distribution on the strongest HRC/VocabMoE spine. |
| `art_loopall_self_electing_rlm_i3l5r5_d640e512_q8` | Every recurrent-loop position gets sparse token expert self-election, with causal RLM-lite memory injected at loop entry. |
| `art_firstattn_mlp_core_i3l7r3_d640e512_q8` | Token mixing at the IO shell and first core block, then a mostly MLP recurrent semantic engine with legal memory. |
| `art_dual_hourglass_q2core_i4l5r3_d768e512` | The full data-density sketch: higher-precision narrow IO blocks, wider q2 recurrent core, head-compatible width ladder `384,432,504,576,648,696,768,768,768`, spike VocabMoE, and trained dual-stream bridges. |

## Submission Read

If a row trains and exports cleanly, it gives us a defensible non-record story:

- lossless token-facing interface with factored tied embeddings;
- mirrored IO-tail plus looped middle route instead of ordinary physical depth;
- train-time quantized forward, not post-hoc-only quantization;
- token-wise micro-expert self-election or dense VocabMoE;
- optional causal document memory and legal self-consistency council;
- final normalized distribution scored before seeing each target.

Promotion rule: after the 2k smoke matrix, promote only the most coherent
working rows to 5k. The submission should be framed as an experimental
architecture/art lane, not as an official record claim.

## Live Read

Latest checked status:

- `art_prime_skip_spike_dual_i3p5s1r2_d640e512_q8` wedged after step 10 and
  was manually pruned. The corrected follow-up removes the dual-stream bridge
  from the prime-skip/spike row.
- `art_pal_ladder_hourglass_dual_spike_i3l5r1_d640e512_q8` failed before
  training because the first width schedule used a block width that was not
  divisible by the configured attention head count. The corrected follow-up
  uses head-compatible widths.
- `art_rlm_council_hybrid_i3l5r5_d640e512_q8` finished cleanly:
  `1.72014328` export BPB, `1605.57ms/step`, and `12,545,134` bytes after
  2k local steps. This is not close to the best 5k local quality row, but it is
  a real, export-honest signal for legal recursive memory plus a council
  distribution.
- `art_loopall_self_electing_rlm_i3l5r5_d640e512_q8` crashed after step 10.
  Treat loop-all spike plus RLM as too unstable until isolated into a smaller
  spike-only or memory-only canary.
- `art_firstattn_mlp_core_i3l7r3_d640e512_q8` finished cleanly and is the best
  completed art row so far: `1.71144547` export BPB, `953.32ms/step`, and
  `11,993,894` bytes after 2k local steps. This beats the RLM+council art row
  while being much faster locally. The useful signal is "one attention-capable
  recurrent-core entry plus mostly MLP recurrent core" rather than council.
- `art_dual_hourglass_q2core_i4l5r3_d768e512` failed before training because
  `layer_widths[1]=448` was not divisible by `NUM_HEADS=12`. This is a shape
  definition bug, not an ML result.
