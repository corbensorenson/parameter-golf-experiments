"""Last-window H100 candidates around the strongest legal novel HRC row.

These rows are intentionally narrow. The late H100 data says the best signal is
the d704/e832, r2, q8, core-attention, LexLoRE/VocabMoE spine with smaller
batches and LQER10/T20. The only broad architecture probes left in the previous
matrix were negative, so this runner spends the final paid window on the
highest-probability ways to beat 1.3569 BPB:

- legalize the faster/better but over-cap 24k row with mild entropy pressure
- tune warmdown around the 32k legal winner
- avoid already-negative levers such as BPB-weighted loss, large e1088, LQER12,
  CycleFuse, and aggressive LexLoRE warm-start
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


def best_tail(
    *,
    name: str,
    batch_tokens: int,
    embed_dim: int,
    warmdown_iters: int,
    lqer_rank: int = 10,
    lqer_top_k: int = 20,
    weight_decay: float = 0.0,
    extra: dict[str, str] | None = None,
    notes: str,
) -> dict[str, Any]:
    env = base.baseline_chase_tail(
        model_dim=704,
        embed_dim=embed_dim,
        repeats=2,
        lqer_rank=lqer_rank,
        lqer_top_k=lqer_top_k,
        weight_decay=weight_decay,
    )
    env.update(
        {
            "H100_1X_TRAIN_BATCH_TOKENS": str(batch_tokens),
            "H100_1X_VAL_BATCH_SIZE": "32768",
        }
    )
    if extra:
        env.update(extra)
    return {
        "name": name,
        "env": base.h100_env_with_overrides(env, {"WARMDOWN_ITERS": str(warmdown_iters)}),
        "notes": notes,
    }


H100_LAST90: list[dict[str, Any]] = [
    best_tail(
        name="h100_last90_24k_e832_w2200_wd02_lqer10t20",
        batch_tokens=24576,
        embed_dim=832,
        warmdown_iters=2200,
        lqer_rank=10,
        lqer_top_k=20,
        weight_decay=0.02,
        notes=(
            "Most direct legalization of the 1.3555 BPB over-cap row: keep the "
            "fast 24k/more-step recipe and add mild entropy-aware Muon WD."
        ),
    ),
    best_tail(
        name="h100_last90_24k_e832_w2200_wd02_lqer8t16",
        batch_tokens=24576,
        embed_dim=832,
        warmdown_iters=2200,
        lqer_rank=8,
        lqer_top_k=16,
        weight_decay=0.02,
        notes=(
            "Cap-safe version of the already-legal 24k/e832/LQER8 row, but with "
            "the stronger 2200 warmdown instead of the weaker 3000 schedule."
        ),
    ),
    best_tail(
        name="h100_last90_24k_e800_w2200_wd02_lqer10t20",
        batch_tokens=24576,
        embed_dim=800,
        warmdown_iters=2200,
        lqer_rank=10,
        lqer_top_k=20,
        weight_decay=0.02,
        notes=(
            "Safer cap version of the 24k row: modestly trims token rank while "
            "keeping LQER10/T20 and the high-update batch schedule."
        ),
    ),
    best_tail(
        name="h100_last90_32k_e832_w2000_lqer10t20",
        batch_tokens=32768,
        embed_dim=832,
        warmdown_iters=2000,
        lqer_rank=10,
        lqer_top_k=20,
        notes=(
            "Schedule-only A/B on the best legal row; slightly later high-LR "
            "training before the cosine taper."
        ),
    ),
    best_tail(
        name="h100_last90_32k_e832_w2400_lqer10t20",
        batch_tokens=32768,
        embed_dim=832,
        warmdown_iters=2400,
        lqer_rank=10,
        lqer_top_k=20,
        notes=(
            "Schedule-only A/B on the best legal row; slightly longer taper "
            "than the 2200-step winner."
        ),
    ),
    best_tail(
        name="h100_last90_32k_e832_w2200_wd02_lqer10t20",
        batch_tokens=32768,
        embed_dim=832,
        warmdown_iters=2200,
        lqer_rank=10,
        lqer_top_k=20,
        weight_decay=0.02,
        notes=(
            "Exact best legal training shape plus mild entropy pressure; tests "
            "whether WD improves final export BPB without paying the 24k noise cost."
        ),
    ),
    best_tail(
        name="h100_last90_24k_e768_w2200_lqer10t20",
        batch_tokens=24576,
        embed_dim=768,
        warmdown_iters=2200,
        lqer_rank=10,
        lqer_top_k=20,
        notes=(
            "Conservative legal fallback for the 24k/more-update idea: smaller "
            "token rank, no WD perturbation, same LQER10/T20 repair."
        ),
    ),
]


GROUPS: dict[str, list[dict[str, Any]]] = {
    "h100_last90": H100_LAST90,
    "all": H100_LAST90,
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
    parser.add_argument("--candidate-group", default="h100_last90", choices=sorted(GROUPS))
    parser.add_argument("--candidates", default="")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--wallclock-seconds", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--warmdown-iters", type=int, default=-1)
    parser.add_argument("--val-tokens", type=int, default=131072)
    parser.add_argument("--timeout", type=int, default=1500)
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
    out_dir = Path(args.out) if args.out else base.ROOT / "records" / f"h100-last90-{time.strftime('%Y%m%d-%H%M%S')}"
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
