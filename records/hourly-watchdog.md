# Hourly Watchdog

Purpose: hourly status snapshots for the local Parameter Golf queue while long
matrix runs are active.

Each entry should record:

- active queue/train process state
- current candidate and latest step/BPB if available
- newly completed candidates and final exported BPB
- failures, suspected cause, and whether anything was restarted
- short interpretation of what the results imply for the next matrix

Restart policy:

- Restart only when the failure looks transient or infrastructure-related:
  timeout, idle-GPU launch race, process interruption, CUDA OOM after an
  obvious batch/accumulation fix, or a queue script dying before launching the
  next candidate.
- Do not restart deterministic model-code failures, repeated nonfinite rows, or
  candidates that are clearly losing badly unless there is a specific reason to
  recover that row.
- Prefer keeping the queue selective over filling idle time with low-signal
  experiments.

## 2026-04-27 10:16 CDT

Automation `parameter-golf-hourly-watchdog` created as an hourly local workspace
job. It will run `scripts/hourly_queue_status.ps1`, inspect queue/train logs and
recent `train.csv` files, append status here, update broader docs only when a
result changes conclusions, and restart only transient/infrastructure failures
without duplicating active processes.

## 2026-04-27 11:17 CDT

Active state: one live VocabMoE cap-speed scout is running behind
`vocabmoe16-cap-speed-after-current-20260427-075723.queue.log`; the queue
launcher started `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24`
at 11:13 CDT after the prior width-pruned queue finished. `Get-Process` shows
the expected Python workers from 11:13 still active, while the downstream
focused/promote PowerShell wrappers remain in wait mode on predecessor PIDs, so
the queue chain is intact and not duplicated. The current train log has only
flushed through step 10/3000 so far, with `step_avg` settling around 742.5 ms;
no BPB is available yet.

Newly completed candidates: `sub4-width-pruned-5k-auto-20260427-012503`
finished cleanly at 11:12:59 CDT. Best row by exported BPB was
`i4l9r5_d640e256_q16q8q4t_wl400-480-560-640_attncore1_lqer_lidx_r8t16` at
`2.50753244` final export BPB, `269.37 ms/step`, `5730884` compressed artifact
bytes. The sibling
`i4l9r5_d640e256_q16q8q8t_wl320-480-560-640_attncore1_lqer_lidx_r8t16` was a bit
faster at `265.61 ms/step` but worse on quality and size at `2.50851753` BPB
and `5870552` bytes. Both remain well over the 4 MB cap and far behind the
recent 16 MB VocabMoE frontier (`1.8846` BPB class), so they do not change the
project conclusion.

Failures/restarts: no new candidate failure in this hour. The status script's
`Get-CimInstance Win32_Process` probe hit `Access denied`, but queue and train
inspection succeeded via logs and `Get-Process`, so no restart was needed or
taken. Interpretation: keep the queue selective and let the current 16 MB
cap-speed scout finish; the newly completed sub4 width variants are low-signal
losers and do not justify broadening search or updating `README.md`/`levers.md`.

## 2026-04-27 12:19 CDT

Active state: the same `vocabmoe16-cap-speed-after-current-20260427-075723`
queue chain is still alive. `Get-Process` shows the long-lived queue
PowerShell wrapper from 07:57 CDT plus one GPU-attached Python process
(`python.exe` PID 22076 started 11:50 CDT), and the active train log for
`i3l3r3_d768e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24`
last flushed at 12:13 CDT with progress through step `1500/3000` at
`890.85 ms/step`. The downstream focused/promote wrappers are still waiting on
predecessor PIDs, so the queue remains serialized rather than duplicated.

Newly completed candidates: one new cap-speed scout row finished cleanly since
the last run,
`i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24`,
with `4.15886638` final export BPB, `740.83 ms/step`, and `5543162` total
artifact bytes (`10456838` bytes of 16 MB cap headroom). That is the best and
only newly exported result this hour. Relative to the current 16 MB VocabMoE
quality frontier (`1.8846` BPB class), it is far worse on quality even though
it fits the cap comfortably.

Failures/restarts: no candidate failed and no restart was taken. The only fresh
error signal was again the status script's `Get-CimInstance Win32_Process`
probe returning `Access denied`; cross-checks via queue logs, log mtimes,
`Get-Process`, and `nvidia-smi` show live work rather than a dead queue.
Interpretation: let the in-flight `d768` cap-speed row finish, keep the queue
selective, and do not update `README.md` or `levers.md` because the new
completed row weakens the cap-speed branch rather than changing project
conclusions.

## 2026-04-27 12:44 CDT

Manual intervention after user status check. The cap-speed fp16-param lane
proved qualitatively broken, not merely noisy: the first two completed rows were
`i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24`
at `4.1589` exported BPB, `740.83 ms/step`, `5,543,162` bytes, and
`i3l3r3_d768e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24`
at `4.1472` exported BPB, `934.79 ms/step`, `6,612,446` bytes. This is far
worse than the stable spike/VocabMoE rows around `1.88` BPB, so speed was
buying a false signal.

Action taken: stopped the third cap-speed row while it was still early, killed
the cap-speed wrapper, patched future 16MB candidates back to fp32 params,
fp32 Muon, GradScaler on, fp32 loss, `GRAD_ACCUM_STEPS=4`, and kept only safe
speed levers: fused QKV, train-time q6 forward, host prefetch/persistent
buffers, and final-only validation. Also gated the top-2 promoter with
`MaxPromoteBpb=2.2` so the bad fp16 rows cannot be promoted if focused rows
fail. The focused queue then launched
`mainline_i3l3r3_d768e384_q6all_vocabmoe_qk525_lqer16t32` under the corrected
stable dtype path; its log confirms `params=float32`, `muon=float32`,
`grad_scaler:1`, and early loss is normal rather than the fp16 blow-up.

## 2026-04-27 13:22 CDT

Active state: `scripts/hourly_queue_status.ps1` ran successfully from the nested
workspace, but its `Get-CimInstance Win32_Process` section still reports
`Access denied`, so process state was cross-checked with `Get-Process` and
`nvidia-smi`. The queue is active, not duplicated: PowerShell PID `38176`
owns the focused 16MB queue, PowerShell PID `33196` is waiting to promote
focused winners, and GPU-bound Python PID `40376` is consuming about `92%` GPU
with `4820/8192 MiB` allocated. The active candidate is
`mainline_i3l3r3_d768e384_q6all_vocabmoe_qk525_lqer16t32` in
`focused-16mb-after-capspeed-3000-auto-20260427-100310`; its train log has
flushed through step `2000/3000` at `996.42 ms/step`. No focused `train.csv`
exists yet.

Newly completed candidates: since the 12:19 automation memory, the second
cap-speed scout row completed cleanly:
`i3l3r3_d768e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24`
at `4.14716446` final export BPB, `934.79 ms/step`, and `6612446` total
artifact bytes. That is the best completed cap-speed row by exported BPB, but
it is far worse than the stable 16MB VocabMoE anchor
`i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst` at `1.87104756`
BPB, `832.11 ms/step`, and `6218621` bytes. The third fp16 cap-speed row
started, logged only ten poor early steps ending around `17.4071` train loss,
and was intentionally stopped during the 12:44 intervention already recorded
above.

Failures/restarts: no new infrastructure failure and no restart was taken. The
only fresh script-level failure signal is the recurring CIM permission denial;
the queue itself is healthy by log mtimes, process state, and GPU load. The
fp16-param cap-speed branch remains rejected as a deterministic low-signal
loser rather than a transient failure. Interpretation: let the corrected
focused queue continue, do not broaden the candidate family, and do not update
`README.md` or `levers.md` again until a focused or promoted row finishes with
a materially competitive exported BPB.

## 2026-04-27 14:22 CDT

Active state: `scripts/hourly_queue_status.ps1` ran successfully from
`C:\Users\corbe\Documents\golf\workspace\parameter-golf`; its CIM process probe
still reports `Access denied`, so process state was cross-checked with
`Get-Process`, `nvidia-smi`, queue logs, and train log mtimes. The queue is
active and not duplicated: focused queue wrapper PID `38176` is still alive,
the top-2 promoter PID `33196` is waiting for PID `38176`, and GPU-bound Python
PID `39584` is active with the GPU around `94%` util and `5788/8192 MiB` used.
Current candidate is
`mainline_i3l3r3_d896e384_q6all_vocabmoe_qk525_lqer16t32`; its live log last
flushed at 14:14 CDT through step `2000/3000`, `3.9037` train loss, and
`1196.31 ms/step`.

Newly completed candidates: one focused 16MB row completed since the previous
watchdog run:
`mainline_i3l3r3_d768e384_q6all_vocabmoe_qk525_lqer16t32` at `2.08196718`
final exported BPB, `994.79 ms/step`, and `6670982` total artifact bytes
(`9329018` bytes headroom under the 16MB cap). It trained to `1.7910` BPB before
export but lost quality on roundtrip, so it is not a new frontier result.

Best final exported result: best newly completed row is the same focused
`d768e384` row (`2.08196718` BPB, `994.79 ms/step`, `6670982` bytes). The
overall exported frontier remains unchanged at
`i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst` with `1.87104756`
BPB, `832.11 ms/step`, and `6218621` bytes.

Failures/restarts: no new queue or train failure was found, and no restart was
taken. The recurring CIM `Access denied` status-script error is a monitoring
permission issue, not a queue failure. Interpretation: the wider e384 focused
row fits the cap but is materially worse after export, so leave `README.md` and
`levers.md` unchanged, keep the queue selective, and let the active `d896e384`
row finish before promoting or drawing conclusions.

## 2026-04-27 15:24 CDT

Active state: the requested status script is not present at the top-level
container path, but it reran successfully from the nested project root
`C:\Users\corbe\Documents\golf\workspace\parameter-golf`. Its
`Get-CimInstance Win32_Process` probe still reports `Access denied`, so process
state was cross-checked with `Get-Process`, `nvidia-smi`, queue logs, and train
log mtimes. The focused queue remains active and not duplicated: wrapper PID
`38176` is alive, promoter PID `33196` is waiting for PID `38176`, and
GPU-bound Python PID `27428` is active with the GPU around `92%` util and
`5068/8192 MiB` used. The current candidate is
`mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32`; its train log last
flushed at 15:20 CDT through step `2500/3000`, `3.7239` train loss, and
`1085.49 ms/step`.

Newly completed candidates: since the 14:22 watchdog run, one focused 16MB row
completed cleanly:
`mainline_i3l3r3_d896e384_q6all_vocabmoe_qk525_lqer16t32` at `2.17243617`
final exported BPB, `1196.48 ms/step`, `7536806` total artifact bytes, and
`8463194` bytes cap headroom. It trained to `1.7903` validation BPB before
export but lost substantial quality on roundtrip, so it is worse than the
previous focused row and not promotion-worthy by project standards.

Best final exported result: best newly completed row is the `d896e384` focused
row (`2.17243617` BPB, `1196.48 ms/step`, `7536806` bytes). Best focused row in
the current queue remains
`mainline_i3l3r3_d768e384_q6all_vocabmoe_qk525_lqer16t32` (`2.08196718` BPB,
`994.79 ms/step`, `6670982` bytes). The overall exported frontier remains
unchanged at `i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst`
(`1.87104756` exact BPB, `832.11 ms/step`, `6218621` bytes).

Failures/restarts: no queue failure, CUDA OOM, nonfinite abort, or train
return-code failure was found, and no restart was taken. The only monitoring
issues were the top-level missing script path and the recurring CIM permission
denial; neither indicates a failed candidate. Interpretation: wider/deeper 16MB
mainline rows continue to fit the artifact cap but degrade exported BPB, so
keep the queue selective, do not add new families, and leave `README.md` and
`levers.md` unchanged until the active `i3l5r2` row or later promoter output
materially improves the exported frontier.

## 2026-04-27 16:24 CDT

Active state: the requested top-level script path is still absent, so the
repo-local status script was run from
`C:\Users\corbe\Documents\golf\workspace\parameter-golf`. Its CIM process
probe still reports `Access denied`, so process state was cross-checked with
`Get-Process`, `nvidia-smi`, queue logs, train logs, and train CSVs. The
focused queue remains active and not duplicated: wrapper PID `38176` is alive,
top-2 promoter PID `33196` is waiting for PID `38176`, and GPU-bound Python
PID `1472` plus helper PID `21732` are active. GPU state at the check was
about `92%` util, `4863/8192 MiB`, and `171W`.

Current candidate:
`leader_i3l3r3_d768e320_q6all_sparsegate_polar_minlr_vocabmoe_lqer16t32`.
The focused queue log shows it launched after the polar row completed. Its
train log last flushed at 16:21 CDT through step `10/3000`, train loss
`7.2838`, and `983.50 ms/step`; with GPU load still high, it appears to be
continuing normally between sparse log flushes.

Newly completed candidates: since the 15:24 watchdog run, two focused 16MB
rows completed with return code `0`:
`mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32` at `1.98432387`
final exported BPB, `1085.55 ms/step`, and `7343282` artifact bytes; and
`leader_i3l3r3_d768e320_q6all_polar_minlr_vocabmoe_qk525_lqer16t32` at
`2.03322719` final exported BPB, `990.91 ms/step`, and `6460014` artifact
bytes.

Best final exported result: best newly completed row is the focused `i3l5r2`
row (`1.98432387` BPB, `1085.55 ms/step`, `7343282` bytes). The overall
exported frontier remains unchanged at
`i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst` with
`1.87104756` exact BPB, `832.11 ms/step`, and `6218621` bytes.

Failures/restarts: no queue failure, CUDA OOM, nonfinite abort, or train
return-code failure was found, and no restart was taken. The polar row emitted
a single `train_nonfinite_debug` gradient sample near step 2355 but completed
and exported cleanly, so it is not a restart condition. Interpretation: the
focused 16MB variants continue to fit the cap but remain materially behind the
VocabMoE 5k exported frontier; keep the queue selective, let sparsegate finish,
and leave `README.md` and `levers.md` unchanged.

## 2026-04-27 17:25 CDT

Active state: `scripts/hourly_queue_status.ps1` ran successfully from
`C:\Users\corbe\Documents\golf\workspace\parameter-golf`; its CIM process probe
still reports `Access denied`, so process state was cross-checked with
`Get-Process`, `nvidia-smi`, queue logs, train logs, and train CSVs. The
focused queue is still active and not duplicated: wrapper PID `38176` is alive,
top-2 promoter PID `33196` is waiting for PID `38176`, and GPU-bound Python PID
`38632` is active. GPU state at the check was about `88%` util,
`4895/8192 MiB`, and `99W`.

Current candidate:
`leader_i3l3r3_d768e320_q6all_depthlora4_polar_minlr_vocabmoe_lqer16t32`.
The focused queue log shows it launched after sparsegate completed. Its train
log last flushed at 17:20 CDT through step `500/3000`, train loss `5.1669`,
and `1034.58 ms/step`; with GPU load still high, it appears to be continuing
normally between sparse log flushes.

Newly completed candidates: since the 16:24 watchdog run, one focused 16MB row
completed with return code `0`:
`leader_i3l3r3_d768e320_q6all_sparsegate_polar_minlr_vocabmoe_lqer16t32` at
`1.99466786` final exported BPB, `991.63 ms/step`, and `6473538` artifact
bytes. It reached `1.7925` validation BPB before export but lost enough quality
on roundtrip to remain behind the better focused `i3l5r2` row.

Best final exported result: best newly completed row is the sparsegate row
(`1.99466786` BPB, `991.63 ms/step`, `6473538` bytes). Best focused row in the
current queue remains `mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32`
(`1.98432387` BPB, `1085.55 ms/step`, `7343282` bytes). The overall exported
frontier remains unchanged at
`i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst` with `1.87104756`
exact BPB, `832.11 ms/step`, and `6218621` bytes.

Failures/restarts: no queue failure, CUDA OOM, nonfinite abort, or train
return-code failure was found, and no restart was taken. The only monitoring
issue remains the recurring CIM permission denial in the status script, which
does not match a failed candidate. Interpretation: sparsegate is a clean but
low-signal loser against the focused best and the project frontier; keep the
queue selective, let depth-lora and the remaining focused/promoter work finish,
and leave `README.md` and `levers.md` unchanged.

## 2026-04-27 18:26 CDT

Active state: `scripts\hourly_queue_status.ps1` ran from
`C:\Users\corbe\Documents\golf\workspace\parameter-golf` after the requested
top-level path was still absent. The script completed, but its CIM process
probe again reported `Access denied`, so process state was cross-checked with
`Get-Process`, `nvidia-smi`, queue logs, train logs, and train CSVs. The
focused queue remains active and not duplicated: wrapper PID `38176` is alive,
top-2 promoter PID `33196` is waiting for PID `38176`, and GPU-bound Python PID
`39580` plus helper PID `40508` are active. GPU state at the check was about
`90%` util, `4892/8192 MiB`, `173W`, and `70C`.

Current candidate:
`leader_i3l5r1rev_d768e320_q6all_polar_minlr_vocabmoe_qk525_lqer16t32`.
The focused queue log shows it launched after the depth-LoRA row completed.
Its train log last flushed at 18:21 CDT through step `1000/3000`, train loss
`5.0688`, and `1032.60 ms/step`; with GPU load still high, it appears to be
continuing normally between sparse log flushes.

Newly completed candidates: since the 17:25 watchdog run, one focused 16MB row
completed with return code `0`:
`leader_i3l3r3_d768e320_q6all_depthlora4_polar_minlr_vocabmoe_lqer16t32` at
`2.03060640` exact final exported BPB, `1035.03 ms/step`, and `6511626`
artifact bytes (`6181184` model bytes, `9488374` bytes headroom). It is worse
than both sparsegate and the focused best, so it is not promotion-worthy.

Best final exported result: best newly completed row is the depth-LoRA row
(`2.03060640` BPB, `1035.03 ms/step`, `6511626` bytes). Best focused row in
the current queue remains `mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32`
(`1.98432387` BPB, `1085.55 ms/step`, `7343282` bytes). The overall exported
frontier remains unchanged at
`i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst` with `1.87104756`
exact BPB, `832.11 ms/step`, and `6218621` bytes.

Failures/restarts: no new queue failure, CUDA OOM, nonfinite abort, or train
return-code failure was found since the prior watchdog run, and no restart was
taken. The recurring status-script CIM denial is a monitoring limitation, not
a failed candidate. Interpretation: depth-LoRA relaxation is another clean
low-signal loser for the focused 16MB lane; keep the queue selective, let the
remaining cycle-rev row and top-2 promoter finish, and leave `README.md` and
`levers.md` unchanged.

## 2026-04-27 19:26 CDT

Active state: the requested top-level `scripts\hourly_queue_status.ps1` path is
still absent, so the watchdog script was run from
`C:\Users\corbe\Documents\golf\workspace\parameter-golf`. It completed, but the
script's CIM process probe again reported `Access denied`; process state was
cross-checked with `Get-Process`, `nvidia-smi`, queue logs, train logs, and
recent train CSVs. The focused 16MB queue wrapper PID `38176` finished cleanly
at 18:56 CDT, and promoter wrapper PID `33196` is now active with one GPU-bound
Python process, PID `10416`. GPU state at the check was about `90%` util,
`5116/8192 MiB`, `179W`, and `70C`.

Current candidate:
`mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32` in the promoted
top-2 16MB 5k queue. The promoter log selected it plus the sparsegate row at
18:57 CDT and started the 5k queue. No promoted `train.csv` exists yet; the
active train log last flushed through step `1500/5000`, train loss `5.1351`,
and `1085.21 ms/step`, with high GPU load indicating normal continuation
between sparse log flushes.

Newly completed candidates: since the 18:26 watchdog run, the focused queue
completed its final row with return code `0`:
`leader_i3l5r1rev_d768e320_q6all_polar_minlr_vocabmoe_qk525_lqer16t32` at
`2.02782949` exact final exported BPB, `1032.95 ms/step`, and `7656578`
artifact bytes (`7326136` model bytes, `8343422` bytes headroom). The focused
queue then exited code `0`, and the promoter selected the two best focused
exports: `mainline_i3l5r2...` (`1.9843`) and sparsegate (`1.9947`).

Best final exported result: best newly completed row is the cycle-rev row
(`2.02782949` BPB, `1032.95 ms/step`, `7656578` bytes), but best focused row
remains `mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32`
(`1.98432387` BPB, `1085.55 ms/step`, `7343282` bytes). The overall exported
frontier remains unchanged at
`i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst` with `1.87104756`
exact BPB, `832.11 ms/step`, and `6218621` bytes.

Failures/restarts: no queue failure, CUDA OOM, nonfinite abort, or train
return-code failure was found in the recent queue or train logs, and no restart
was taken. The recurring CIM access denial is a monitoring limitation, not a
candidate failure. Interpretation: the cycle-rev focused row is another clean
low-signal loser, but the automatic promotion path is doing the right selective
thing by rerunning only the two focused winners at 5k. Leave `README.md` and
`levers.md` unchanged until a promoted 5k export materially moves the frontier.

## 2026-04-27 20:29 CDT

Active state: the requested top-level `scripts\hourly_queue_status.ps1` path is
still absent, so the watchdog script was run from
`C:\Users\corbe\Documents\golf\workspace\parameter-golf`. It completed, but the
script's CIM process probe again reported `Access denied`; process state was
cross-checked with `Get-Process`, `nvidia-smi`, queue logs, train logs, and the
promoted `train.csv`. The promoted queue wrapper PID `33196` remains active,
with GPU-bound Python PID `35276` and helper PID `36080` training the second
promoted row. GPU state at the check was about `91%` util, `4902/8192 MiB`,
`174W`, and `69C`. No duplicate active queue was found.

Current candidate:
`leader_i3l3r3_d768e320_q6all_sparsegate_polar_minlr_vocabmoe_lqer16t32` in
the promoted top-2 16MB 5k queue. It started at 20:28 CDT after the mainline
row completed, and its train log has flushed through step `10/5000`, train
loss `7.1980`, and `985.10 ms/step`; high GPU load indicates normal
continuation between sparse log flushes.

Newly completed candidates: since the 19:26 watchdog run, the first promoted
5k row completed with return code `0`:
`mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32` at `1.88161553`
exact final exported BPB, `1085.46 ms/step`, and `8289170` artifact bytes
(`7958728` model bytes, `7710830` bytes headroom). The train log had one
post-unscale nonfinite debug sample at step `4283`, but it did not abort and
the final export roundtrip was valid.

Best final exported result: best newly completed/promoted row is
`mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32` (`1.88161553` BPB,
`1085.46 ms/step`, `8289170` bytes). It improves its 3k focused export
(`1.98432387`) but does not beat the overall exported frontier:
`i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst` with `1.87104756`
exact BPB, `832.11 ms/step`, and `6218621` bytes.

Failures/restarts: no queue failure, CUDA OOM, train return-code failure, or
nonfinite abort was found in the recent queue/train logs, and no restart was
taken. The recurring CIM access denial is still a monitoring limitation, not a
candidate failure. Interpretation: the 5k mainline promotion is a useful near
miss but still a clean loser against the project frontier; keep the queue
selective, let the sparsegate 5k row finish, and leave `README.md` and
`levers.md` unchanged.

## 2026-04-27 21:25 CDT

Manual user-requested status check. The promoted top-2 queue is still active
and serialized: promoter wrapper PID `33196` is alive, with promoted matrix
PIDs `29888`/`30648` and GPU train PIDs `35276`/`36080`. GPU state at the
check was `92%` util, `4993/8192 MiB`, about `174W`.

Current candidate:
`leader_i3l3r3_d768e320_q6all_sparsegate_polar_minlr_vocabmoe_lqer16t32`.
Its promoted 5k log has flushed through step `3000/5000`, train loss `4.6720`,
and `989.94 ms/step`. It had one post-unscale nonfinite debug sample at step
`2198`, but no abort and training continues normally.

Completed since the last manual check: the first promoted row
`mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32` finished at
`1.88161553` exported BPB, `1085.46 ms/step`, and `8,289,170` bytes. This
improves its 3k result but remains behind the project frontier
`i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst` at `1.87104756`
BPB. No restart was taken.

## 2026-04-27 22:23 CDT

Manual user-requested status check. The promoted top-2 queue finished cleanly
at 21:52 CDT and no queue/train processes remain active. GPU is idle-ish at
`16%` util and `470/8192 MiB`.

Final promoted results:

- `mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32`: `1.88161553`
  exact exported BPB, `1085.46 ms/step`, `8,289,170` bytes.
- `leader_i3l3r3_d768e320_q6all_sparsegate_polar_minlr_vocabmoe_lqer16t32`:
  `1.95062561` exact exported BPB, `993.82 ms/step`, `7,299,974` bytes.

Interpretation: the promoted 5k reruns validated the queue's ranking but did
not beat the existing project frontier. Sparsegate does not transfer well to a
longer run here; the mainline i3/l5/r2 is close but still worse than the older
d640/e256 input+loop-first VocabMoE anchor. No restart or new queue was
launched.

## 2026-04-29T04:47:29.9511409-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `1/4`
- GPU: `94, 60, 7445, 8192, 72, 167.38`
- Active Python processes: `3`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0

Best so far: `frontier_capfill_i3l5r5_d640e640_q8` at `1.53680000` BPB.

Current queue is still running or incomplete; no follow-up launch.

## 2026-04-29T05:19:25.7744846-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `2/4`
- GPU: `89, 56, 7578, 8192, 72, 136.86`
- Active Python processes: `3`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0

Best so far: `frontier_capfill_i3l5r5_d640e640_q8` at `1.53680000` BPB.

Current queue is still running or incomplete; no follow-up launch.

## 2026-04-29T05:49:41.6829856-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `2/4`
- GPU: `89, 59, 7215, 8192, 72, 172.43`
- Active Python processes: `3`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0

Best so far: `frontier_capfill_i3l5r5_d640e640_q8` at `1.53680000` BPB.

Current queue is still running or incomplete; no follow-up launch.

## 2026-04-29T06:51:35.8340647-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `2/4`
- GPU: `97, 69, 7220, 8192, 73, 167.14`
- Active Python processes: `3`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0

Best so far: `frontier_capfill_i3l5r5_d640e640_q8` at `1.53680000` BPB.

Current queue is still running or incomplete; no follow-up launch.

## 2026-04-29T07:53:38.4856922-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `3/4`
- GPU: `97, 69, 7420, 8192, 72, 170.62`
- Active Python processes: `3`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue is still running or incomplete; no follow-up launch.

## 2026-04-29T08:54:44.0156009-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `3/4`
- GPU: `97, 69, 7420, 8192, 72, 176.21`
- Active Python processes: `3`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue is still running or incomplete; no follow-up launch.

## 2026-04-29T09:55:10.3106212-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 43, 18.50`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue complete. Launched `cap16_frontier_followup` as PID `` in `C:\Users\corbe\Documents\golf\workspace\parameter-golf\records\cap16-frontier-followup-5k-auto-20260429-095510`.

## 2026-04-29T09:55:52.4616003-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 43, 18.50`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue is complete and follow-up marker already exists; no new launch.

## 2026-04-29T09:56:08.7861672-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 43, 18.50`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue complete, but follow-up launch failed: `Item has already been added. Key in dictionary: 'Path'  Key being added: 'PATH'`.

## 2026-04-29T09:56:39.8668806-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 43, 18.50`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue complete, but follow-up launch failed: `You cannot call a method on a null-valued expression.`.

## 2026-04-29T09:56:47.3962105-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 43, 18.49`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue complete, but follow-up launch failed: `You cannot call a method on a null-valued expression.`.

## 2026-04-29T09:56:57.8137643-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 43, 18.49`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue complete. Launched `cap16_frontier_followup` as PID `20628` in `C:\Users\corbe\Documents\golf\workspace\parameter-golf\records\cap16-frontier-followup-5k-auto-20260429-095657`.

## 2026-04-29T10:56:41.2445740-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 45, 18.94`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue complete. Launched `cap16_frontier_followup` as PID `4000` in `C:\Users\corbe\Documents\golf\workspace\parameter-golf\records\cap16-frontier-followup-5k-auto-20260429-105641`.

## 2026-04-29T10:57:46.6802477-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 45, 19.03`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue is complete and follow-up marker already exists; no new launch to avoid duplicate queues.

## 2026-04-29T11:58:17.3844512-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 45, 18.96`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue is complete and follow-up marker already exists; no new launch to avoid duplicate queues.

## 2026-04-29T11:59:58.0646236-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 45, 18.80`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue is complete and follow-up marker already exists; no new launch to avoid duplicate queues.

## 2026-04-29T12:59:48.2947216-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 45, 18.86`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue is complete and follow-up marker already exists; no new launch to avoid duplicate queues.

## 2026-04-29T14:01:13.9119976-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 45, 18.96`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue is complete and follow-up marker already exists; no new launch to avoid duplicate queues.

## 2026-04-29T15:02:19.2358739-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `0, 0, 780, 8192, 43, 18.75`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue is complete and follow-up marker already exists; no new launch to avoid duplicate queues.

## 2026-04-29T16:04:25.4986704-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `1, 2, 841, 8192, 43, 19.88`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue is complete and follow-up marker already exists; no new launch to avoid duplicate queues.

## 2026-04-29T16:40:32.1299032-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-capfill-5k-auto-20260429-003802`
- Completed rows: `4/4`
- GPU: `16, 13, 918, 8192, 44, 23.96`
- Active Python processes: `0`

Rows:
- `frontier_capfill_i3l5r5_d640e640_q8`: BPB=1.53680000, step=1604.1ms, bytes=14574862, rc=0
- `frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed`: BPB=1.54070000, step=1576.04ms, bytes=13956518, rc=0
- `frontier_polarminlr10_i3l5r5_d640e512_q8`: BPB=1.53360000, step=1560.22ms, bytes=14091166, rc=0
- `frontier_qk525_parres4_i3l5r5_d640e512_q8`: BPB=1.54890000, step=1585.39ms, bytes=13900982, rc=0

Best so far: `frontier_polarminlr10_i3l5r5_d640e512_q8` at `1.53360000` BPB.

Current queue is complete and follow-up marker already exists; no new launch to avoid duplicate queues.

## 2026-04-29T16:45:10.0891498-05:00

Automation status check for frontier cap-fill queue.

- Run: `records\cap16-frontier-followup-5k-auto-20260429-164230`
- Completed rows: `0/4`
- GPU: `89, 56, 7375, 8192, 68, 175.52`
- Active Python processes: `4`

Rows:
- no completed rows flushed yet

Current queue is still running or incomplete; no follow-up launch.
