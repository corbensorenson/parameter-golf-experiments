"""Legal-size fallback 8xH100 matrix for MirrorLoop HRC + LexLoRE.

The first official-shaped 8x row preserved the 24k/rank rhythm but exported at
16.41MB, so this runner trims the factored embedding rank from e832 to e768 and
keeps the rest of the architecture as close as possible.
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


def legal_tail(
    *,
    name: str,
    batch_tokens: int,
    embed_dim: int = 768,
    warmdown_iters: int = 2200,
    repeats: int = 2,
    lqer_rank: int = 8,
    lqer_top_k: int = 16,
    weight_decay: float = 0.02,
    loop_index: bool = False,
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


LEGAL_FALLBACK: list[dict[str, Any]] = [
    legal_tail(
        name="final8x_legal_196k_r2_d704e768_w2200_wd02_lqer8t16_vocabmoe_qk55",
        batch_tokens=196_608,
        notes=(
            "Direct legalizer for the over-cap e832/LQER8 row: keep the "
            "24k/rank rhythm and trim only the factored embedding rank."
        ),
    ),
    legal_tail(
        name="final8x_legal_196k_r2_d704e768_w2200_wd02_lqer6t12_vocabmoe_qk55",
        batch_tokens=196_608,
        lqer_rank=6,
        lqer_top_k=12,
        notes=(
            "Extra-safe size variant. Tests whether a smaller LQER payload "
            "closes the export/artifact risk without a large quality hit."
        ),
    ),
    legal_tail(
        name="final8x_legal_262k_r2_d704e768_w2500_wd02_lqer8t16_vocabmoe_qk55",
        batch_tokens=262_144,
        warmdown_iters=2500,
        notes=(
            "32k/rank middle point with e768 legalization; tests whether a "
            "larger global batch improves export BPB on 8x."
        ),
    ),
    legal_tail(
        name="final8x_legal_196k_r3_d704e768_w3000_wd02_lqer8t16_lidx_vocabmoe_qk55",
        batch_tokens=196_608,
        warmdown_iters=3000,
        repeats=3,
        loop_index=True,
        notes=(
            "Single deeper recurrent sanity row, legalized through e768 and "
            "loop-indexed so repeated passes can specialize."
        ),
    ),
]


GROUPS: dict[str, list[dict[str, Any]]] = {
    "h100_8x_legal_fallback": LEGAL_FALLBACK,
    "all": LEGAL_FALLBACK,
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
    parser.add_argument("--candidate-group", default="h100_8x_legal_fallback", choices=sorted(GROUPS))
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
    out_dir = Path(args.out) if args.out else base.ROOT / "records" / f"h100-8x-legal-fallback-{time.strftime('%Y%m%d-%H%M%S')}"
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
