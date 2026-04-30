# 8xH100 Launch Plan - 2026-04-30

Goal: use one paid 8xH100 hour on the strongest novel HRC/LexLoRE candidates, not a broad random matrix.

## Architecture Held Constant

- CaseOps/SP8192 lossless tokenizer path.
- HRC mirrored IO tail with looped middle: `012|34567|34567|210` for the r2 rows.
- q8 train-time quantized forward from step 0, including embeddings.
- Factored tied token interface.
- LexLoRE/VocabMoE at `input,loop_first`.
- One attention-capable core block, with the rest of the recurrent core MLP-only for speed.
- QK gain 5.5, Polar/MIN_LR schedule, final LZMA export roundtrip.
- LQER export repair on large block/embed projection tensors.

## First-Hour Candidates

| Candidate | Why it is in the first hour |
|---|---|
| `final8x_196k_r2_d704e832_w2200_wd02_lqer8t16_vocabmoe_qk55` | Best-known legal 1x package; preserves the 24k tokens/rank rhythm on 8 GPUs. |
| `final8x_196k_r2_d704e832_w2200_wd02_lqer9t18_vocabmoe_qk55` | Near-cap repair row between known-safe LQER8/T16 and slightly-over LQER10/T20. |
| `final8x_262k_r2_d704e832_w2200_lqer10t20_vocabmoe_qk55` | 32k tokens/rank middle ground, likely legal, better GPU occupancy than 24k/rank. |
| `final8x_524k_r2_d704e832_w3500_lqer10t20_vocabmoe_qk55` | Official-style global batch; tests whether 8x's main benefit is more tokens per update. |
| `final8x_196k_r3_d704e768_w3000_wd02_lqer8t16_lidx_vocabmoe_qk55` | Only deeper recurrent probe; r3 plus loop index, legalized through e768/LQER8. |

This is intentionally five rows. Each gets a full 600-second training wall clock, final artifact export, and a 131k-token validation probe for quick ranking.

## Launch

Build the bundle locally:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make_runpod_novel_bundle.ps1 -BundleName parameter-golf-novel-8x-ready
```

After the pod exists and SSH is available:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\upload_and_start_8xh100.ps1 -HostName <ip> -Port <port> -KeyPath C:\Users\corbe\.ssh\runpod_codex_ed25519
```

The remote log is `/workspace/final8x.out`.

The launcher now runs `final8x-check` before starting paid training. That check verifies the bundled files, CUDA availability, at least 8 GPUs, and NCCL support, then lists the exact focused candidates.

## Local Preflight Status

Completed immediately before launch readiness:

- `python -m py_compile` on the focused 8x runner.
- `bash -n` on the no-fetch RunPod shell launcher using Git Bash.
- PowerShell parser check on the upload/start helper.
- Extracted the exact tarball and verified candidate listing from inside the extracted bundle.
- Verified required bundled CaseOps train/val/val-bytes files and tokenizer are present.
- Verified all five candidates use 600-second wall clock, train-time quantized forward, train-time quantized embeddings, HRC family, VocabMoE/LexLoRE enabled, q8 export, and batch sizes divisible across 8 ranks at sequence length 1024.

The current launch archive is:

```text
C:\Users\corbe\Documents\golf\workspace\parameter-golf\tmp-runpod-bundles\parameter-golf-novel-8x-ready.tar.gz
```

## Capacity Decision

Do not rent partial H100 pods for this launch. The only paid run path worth using now is an actual 8xH100 pod, because the question we need answered is whether the architecture clears the 10-minute 8x wall-clock bar. If 8 H100s are unavailable, wait rather than spending credits on 3x.

## Funding Pause

The prior 1xH100 endpoint stopped responding after the RunPod wallet ran out of funds. If that pod is merely stopped/migrating, the `/workspace` files may reappear after restart; if it was terminated, only logs already copied back locally are safe.

The 8x-ready no-fetch bundle remains local at:

```text
C:\Users\corbe\Documents\golf\workspace\parameter-golf\tmp-runpod-bundles\parameter-golf-novel-8x-ready.tar.gz
```

When a fresh 8x pod is available, upload that bundle and start `final8x`; no repo clone or dataset download is needed.
