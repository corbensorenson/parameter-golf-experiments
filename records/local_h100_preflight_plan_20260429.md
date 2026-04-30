# Local H100 Preflight Plan

Date: 2026-04-29

Before spending H100 money, run the paid-candidate shapes on the local RTX 2060
as a crash/export/ranking gate. This is not expected to produce official-grade
scores: local context and token exposure are much smaller. The goal is to avoid
paying for rows that are unstable, export poorly, or already look dominated by
a simpler sibling.

## Candidate Group

Runner group: `cap16_h100_preflight` in
`scripts/run_16mb_vocab_moe_matrix.py`.

| Candidate | Local Gate |
| --- | --- |
| `preflight_h100_anchor_i3l5r5_d640e512_q8_polar` | Rechecks the current best novel spine under the current code. |
| `preflight_h100_e640_i3l5r5_d640e640_q8_polar` | Tests whether token-interface byte spend still helps when paired with Polar/MIN_LR. |
| `preflight_h100_i3l7r4_d640e512_q8_polar_bigram` | Tests more unique loop blocks plus BigramHash. |
| `preflight_h100_dual_i3l5r5_d640e512_q8_polar_left256` | Tests the trained advisor bridge on the best route. |
| `preflight_h100_spike_loopall_i3l5r5_d640e512_q8_polar` | Tests spike/self-election without the RLM bundle that crashed locally. |

## Queue Command

The helper waits for an existing queue PID, lets any child trainer allocate the
GPU, then waits for the GPU to become idle before starting each preflight row.

```powershell
.\scripts\queue_local_h100_preflight_after_current.ps1 -WaitPid <queue_or_waiter_pid> -Iterations 2000 -ValTokens 65536
```

For a manual run:

```bash
python scripts/run_16mb_vocab_moe_matrix.py \
  --candidate-group cap16_h100_preflight \
  --iterations 2000 \
  --warmdown-iters 2000 \
  --val-tokens 65536 \
  --timeout 7200 \
  --final-artifacts \
  --train-quant-forward \
  --wait-for-idle-gpu \
  --idle-max-util 90 \
  --idle-max-memory-mib 2500 \
  --idle-seconds 15
```

## Promotion Rules

- A row that crashes locally does not go to H100.
- A row with a large export gap does not go to H100 unless it is the only row
  testing a uniquely important mechanism.
- If spike-loopall crashes without RLM, replace the H100 spike row with a
  safer spike-loop-first or drop spike from the paid matrix.
- If dual-stream remains a near miss but stable, keep it as a novelty row only
  if we are explicitly targeting non-record/art.
- If i3/l7/r4 beats or nearly matches the anchor at 2k, promote it to a 5k
  local run before paid H100 time.

The expected outcome is fewer paid candidates, not more. This preflight should
shrink the cloud matrix to the two or three local rows that are both novel and
actually learning.

## Current Queue

Queued directory:
`records/cap16-h100-preflight-2000-auto-20260429-191804`.

Watcher PID `16540` is waiting for the active GPU work to clear. The repaired
art follow-up was stopped before launch so this preflight can run sooner.

Latest status:

- The watcher waited correctly for the art matrix to release the GPU, then
  started the preflight at `2026-04-29 19:20:34`.
- Current active row:
  `preflight_h100_anchor_i3l5r5_d640e512_q8_polar`.
- Row 1 finished cleanly:
  `preflight_h100_anchor_i3l5r5_d640e512_q8_polar` exported at `1.69313786`
  BPB, `1578.09ms/step`, and `12,550,430` bytes after 2k local steps. This is
  consistent with an early-stop gate on the old 5k winner, not a full quality
  score.
- Row 2 is active:
  `preflight_h100_e640_i3l5r5_d640e640_q8_polar`. It reached step 10/2000 at
  `1573.15ms/step`; early train loss is lower than the e512 anchor at the same
  step, but export BPB will decide.
- Latest row-2 checkpoint: step 1250/2000, train loss `3.7174`,
  `1573.84ms/step`. This is slightly ahead of row 1 at the same checkpoint
  (`3.7346`), with nearly identical speed. Export BPB still decides.
- Row 2 finished cleanly and won the 2k gate:
  `preflight_h100_e640_i3l5r5_d640e640_q8_polar` exported at `1.68806706`
  BPB, `1593.56ms/step`, and `13,264,106` bytes. It is about `0.0051` BPB
  better than the e512 anchor at 2k with roughly `15.5ms/step` extra local
  cost, so e640 remains worth considering for the H100 smoke/round.
