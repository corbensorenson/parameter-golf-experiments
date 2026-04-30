# RunPod No-Fetch Plan

Date: 2026-04-29

The paid RunPod path should not fetch repos, PR folders, or datasets during the
experiment. We will build a local archive that already contains:

- this experiment repo and current runner code;
- the CaseOps/SP8192 smoke train shard;
- the CaseOps validation tokens and byte sidecar;
- the CaseOps tokenizer model/vocab;
- `scripts/runpod_run_novel_no_fetch.sh`, which validates inputs and runs the
  H100 matrix without network setup.

## RunPod UI

Use the Parameter Golf template, but enable SSH:

- GPU count: `1` for the first smoke/sanity run. Use `8` only after local and
  1xH100 gates look worth it.
- Instance pricing: On-Demand is fine.
- SSH terminal access: enable this. This is required for Codex to upload the
  bundle and automate the pod.
- Jupyter: optional. SSH is the important one.
- Container disk: `50GB` is fine for the no-fetch smoke bundle.
- Volume disk: `50GB` is fine for the current bundled one-shard data. Use more
  only if we later upload many train shards.

After deploy, send Codex the SSH command/details RunPod shows.

## Local Bundle

Build the upload archive from this machine:

```powershell
.\scripts\make_runpod_novel_bundle.ps1
```

The script writes a `.tar.gz` under `tmp-runpod-bundles/` and prints upload
commands. It includes the local CaseOps data shard, so the pod does not need to
download data.

Latest verified archive:

```text
C:\Users\corbe\Documents\golf\workspace\parameter-golf\tmp-runpod-bundles\parameter-golf-novel-runpod-20260429-214806.tar.gz
```

Verification performed locally:

- extracted archive lists all five `h100_novel_round1` candidates;
- `scripts/run_h100_novel_matrix.py` and its matrix dependency compile from the
  extracted archive;
- required CaseOps files are present:
  `fineweb_train_000000.bin`, `fineweb_val_000000.bin`,
  `fineweb_val_bytes_000000.bin`, and the SP8192 CaseOps tokenizer;
- the no-fetch runner has no active `git clone`, `pip install`, `wget`, `curl`,
  dataset download, or Hugging Face fetch path.

The paid `h100_novel_round1` set is now cap-spend focused. The previous
e512/e640 preflight rows used only about 12.55MB and 13.26MB at 2k local steps;
even the 5k local cap-fill rows still left about 1.4-1.9MB. The current paid
set therefore starts at e768/e896/e1024-style token-rank spend and keeps the
novel levers:

- `h100_capfill_i3l5r5_d640e896_q8_polar`
- `h100_capedge_i3l5r5_d640e1024_q8_polar`
- `h100_capfill_i3l7r4_d640e640_q8_polar_bigram`
- `h100_capfill_dual_i3l5r5_d640e896_q8_polar_left320`
- `h100_capfill_spike_loopall_i3l5r5_d640e768_q8_polar`

## H100 Speed Patch

After the first RunPod smoke, the original 1xH100 profile was corrected before
the paid matrix:

- `TRAIN_FUSED_QKV=1`; same Q/K/V capacity, fewer projection launches and fewer
  train-time quant materializations.
- `WARMUP_STEPS=0`; the reset-only warmup was useful for compile debugging but
  wasted real paid wallclock with `DISABLE_COMPILE=1`.
- `TRAIN_ABORT_ON_NONFINITE=0` and `TRAIN_DEBUG_NONFINITE=0`; stability was
  already smoked, and the safety path scans gradients/parameters every step.
- `POST_STEP_ZERO_GRAD=0`; gradients are still cleared at the start of each
  step, so the post-step clear was redundant work for scout runs.
- `TRAIN_LOG_EVERY=250`; enough progress signal without frequent loss `.item()`
  synchronizations.

Measured on the first H100 row at `TRAIN_BATCH_TOKENS=131072`:

- original 8x-style profile fixed for 1x: about `314 ms/step`;
- fused QKV profile: about `298 ms/step`;
- lean fused profile with safety scans/post-step zero disabled: about
  `271 ms/step`.

That is roughly a 14% step-speed improvement without changing the modeled
candidate, only the execution profile.

## Pod Commands

Once the archive is uploaded to `/workspace` and extracted:

```bash
cd /workspace/<bundle-dir>
bash scripts/runpod_run_novel_no_fetch.sh check
bash scripts/runpod_run_novel_no_fetch.sh smoke
```

If smoke passes, run the five-row 1xH100 scout:

```bash
bash scripts/runpod_run_novel_no_fetch.sh round1-1xh100
```

For an 8xH100 pod:

```bash
bash scripts/runpod_run_novel_no_fetch.sh round1-8xh100
```

The no-fetch runner exits if required files are missing. It does not call git,
pip, wget, curl, or the dataset downloader.

## Important Limitation

The current bundle includes the one local train shard plus validation sidecar.
That is enough for smoke and first scaling signal. If we want a larger
no-fetch training set, we must first prepare/upload a bigger local bundle with
more shards. The no-fetch runner will not silently download them on the pod.
