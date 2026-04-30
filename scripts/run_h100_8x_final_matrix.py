"""Focused 8xH100 launch matrix for the novel HRC/LexLoRE lane.

This runner is intentionally tiny. The paid 8x window should not spend time on
broad architecture fishing; it should test the strongest 1xH100 legal signal
under the real distributed budget plus a few high-value batch/depth variants.

The current best 1x signal is the d704/e832 i3l5r2 q8-train/export HRC spine:
CaseOps/SP8192, mirrored IO tail, looped middle, one attention-capable core
block, LexLoRE/VocabMoE at input+loop entry, QK gain 5.5, and LQER export
repair. The main uncertainty for 8x is whether to preserve the winning
per-GPU batch size or use the official large global batch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import run_h100_novel_matrix as base


TRAINER = base.ROOT / "train_gpt.py"
base.TRAINER = TRAINER


def eightx_tail(
    *,
    name: str,
    batch_tokens: int,
    embed_dim: int,
    warmdown_iters: int,
    repeats: int = 2,
    lqer_rank: int = 10,
    lqer_top_k: int = 20,
    weight_decay: float = 0.0,
    loop_index: bool = False,
    extra: dict[str, str] | None = None,
    notes: str,
) -> dict[str, Any]:
    env = base.baseline_chase_tail(
        model_dim=704,
        embed_dim=embed_dim,
        repeats=repeats,
        keep_attn_blocks=(3,),
        lqer_rank=lqer_rank,
        lqer_top_k=lqer_top_k,
        weight_decay=weight_decay,
    )
    if loop_index:
        env.update(base.loop_index_env(dim=64, scale=0.02))
    if extra:
        env.update(extra)
    return {
        "name": name,
        "env": base.h100_env_with_overrides(
            env,
            {
                "TRAIN_BATCH_TOKENS": str(batch_tokens),
                "VAL_BATCH_SIZE": str(batch_tokens),
                "GRAD_ACCUM_STEPS": "1",
                "WARMDOWN_ITERS": str(warmdown_iters),
                "TRAIN_LOG_EVERY": "250",
            },
        ),
        "notes": notes,
    }


H100_8X_FINAL: list[dict[str, Any]] = [
    eightx_tail(
        name="final8x_196k_r2_d704e832_w2200_wd02_lqer8t16_vocabmoe_qk55",
        batch_tokens=196_608,
        embed_dim=832,
        warmdown_iters=2200,
        lqer_rank=8,
        lqer_top_k=16,
        weight_decay=0.02,
        notes=(
            "Best-known legal 1x package moved to 8x while preserving the "
            "winning 24k tokens/rank optimizer rhythm."
        ),
    ),
    eightx_tail(
        name="final8x_196k_r2_d704e832_w2200_wd02_lqer9t18_vocabmoe_qk55",
        batch_tokens=196_608,
        embed_dim=832,
        warmdown_iters=2200,
        lqer_rank=9,
        lqer_top_k=18,
        weight_decay=0.02,
        notes=(
            "Near-cap legalizer between the known-safe LQER8/T16 row and the "
            "slightly-over 1x LQER10/T20 row."
        ),
    ),
    eightx_tail(
        name="final8x_262k_r2_d704e832_w2200_lqer10t20_vocabmoe_qk55",
        batch_tokens=262_144,
        embed_dim=832,
        warmdown_iters=2200,
        lqer_rank=10,
        lqer_top_k=20,
        notes=(
            "32k tokens/rank row. This is the strongest legal middle point "
            "between local more-step training and the official 524k batch."
        ),
    ),
    eightx_tail(
        name="final8x_524k_r2_d704e832_w3500_lqer10t20_vocabmoe_qk55",
        batch_tokens=524_288,
        embed_dim=832,
        warmdown_iters=3500,
        lqer_rank=10,
        lqer_top_k=20,
        notes=(
            "Official-style 8x global batch. Tests whether the cluster's main "
            "gift is cleaner gradients and far more training tokens per update."
        ),
    ),
    eightx_tail(
        name="final8x_196k_r3_d704e768_w3000_wd02_lqer8t16_lidx_vocabmoe_qk55",
        batch_tokens=196_608,
        embed_dim=768,
        warmdown_iters=3000,
        repeats=3,
        lqer_rank=8,
        lqer_top_k=16,
        weight_decay=0.02,
        loop_index=True,
        notes=(
            "Only deeper-loop probe in the first paid hour: r3 plus loop index, "
            "legalized through e768/LQER8 while preserving 24k tokens/rank."
        ),
    ),
]


GROUPS: dict[str, list[dict[str, Any]]] = {
    "h100_8x_final": H100_8X_FINAL,
    "all": H100_8X_FINAL,
}


def selected_candidates(raw: str, group: str) -> list[dict[str, Any]]:
    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; choices: {', '.join(sorted(GROUPS))}")
    pool = GROUPS[group]
    if not raw.strip():
        return pool
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    out = [candidate for candidate in pool if candidate["name"] in wanted]
    missing = wanted - {candidate["name"] for candidate in out}
    if missing:
        raise ValueError(f"unknown candidates for {group}: {', '.join(sorted(missing))}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--candidate-group", default="h100_8x_final", choices=sorted(GROUPS))
    parser.add_argument("--candidates", default="")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--wallclock-seconds", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--warmdown-iters", type=int, default=-1)
    parser.add_argument("--val-tokens", type=int, default=131072)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--data-path", default=str(base.DATASET_DIR))
    parser.add_argument("--tokenizer-path", default=str(base.TOKENIZER_PATH))
    parser.add_argument("--final-artifacts", action="store_true", default=True)
    parser.add_argument("--skip-final-artifacts", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    candidates = selected_candidates(args.candidates, args.candidate_group)
    if args.list:
        for candidate in candidates:
            print(candidate["name"])
        return 0
    if not TRAINER.is_file():
        print(f"Missing trainer: {TRAINER}", file=sys.stderr)
        return 1
    data_path = Path(args.data_path)
    tokenizer_path = Path(args.tokenizer_path)
    if not data_path.is_dir() or not tokenizer_path.is_file():
        print(f"Missing CaseOps SP8192 data/tokenizer:\n  data={data_path}\n  tokenizer={tokenizer_path}")
        return 1
    if args.nproc_per_node != 8:
        print(f"WARNING: final 8x runner launched with nproc_per_node={args.nproc_per_node}", flush=True)
    out_dir = Path(args.out) if args.out else base.ROOT / "records" / f"h100-8x-final-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    iterations = args.iterations if args.iterations > 0 else None
    warmdown_iters = args.warmdown_iters if args.warmdown_iters >= 0 else None
    base.write_candidate_plan(out_dir, candidates, iterations or 1_000_000, args.val_tokens)
    base.run_matrix(
        out_dir=out_dir,
        candidates=candidates,
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        nproc_per_node=args.nproc_per_node,
        wallclock_seconds=args.wallclock_seconds,
        val_tokens=args.val_tokens,
        timeout=args.timeout,
        final_artifacts=bool(args.final_artifacts and not args.skip_final_artifacts),
        iterations=iterations,
        warmdown_iters=warmdown_iters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
