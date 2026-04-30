"""Run isolated H100 architecture-evolution probes.

This runner intentionally uses train_gpt_arch_evolution.py so we can test new
HRC route code without mutating the current best-family trainer.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import run_h100_novel_matrix as base


TRAINER = base.ROOT / "train_gpt_arch_evolution.py"
base.TRAINER = TRAINER


def best_spine_env(
    *,
    embed_dim: int = 832,
    batch_tokens: int = 32768,
    warmdown_iters: int = 2200,
    lqer_rank: int = 10,
    lqer_top_k: int = 20,
) -> dict[str, str]:
    """Current best legal H100 spine, kept as the control for one-lever probes."""

    return base.h100_env_with_overrides(
        {
            **base.baseline_chase_tail(
                model_dim=704,
                embed_dim=embed_dim,
                repeats=2,
                lqer_rank=lqer_rank,
                lqer_top_k=lqer_top_k,
            ),
            "H100_1X_TRAIN_BATCH_TOKENS": str(batch_tokens),
            "H100_1X_VAL_BATCH_SIZE": "32768",
        },
        {"WARMDOWN_ITERS": str(warmdown_iters)},
    )


def partial_tail_route(*, io_width: int = 3, loop_width: int = 5, repeats: int = 2, tail_width: int = 3) -> dict[str, str]:
    """Route 012|34567 repeated, then only the semantic tail, then mirrored exit."""

    return {
        "MODEL_FAMILY": "hrc",
        "NUM_UNIQUE_BLOCKS": str(io_width + loop_width),
        "EFFECTIVE_DEPTH": str(io_width + loop_width * repeats + tail_width + io_width),
        "HRC_RECURSIVE_CORE_START": str(io_width),
        "HRC_ROUTE_REPEATS": str(repeats),
        "HRC_DEPTH_SCHEDULE_MODE": f"transition_tail{tail_width}_cycle",
        "HRC_ROUTE_PHASE_ENABLED": "0",
    }


H100_ARCH_EVOLUTION: list[dict[str, Any]] = [
    {
        "name": "h100_evo_32k_e832_lexlore_exit_looplast",
        "env": {
            **best_spine_env(embed_dim=832),
            "VOCAB_MOE_LAYERS": "input,loop_first,loop_last",
        },
        "notes": (
            "LexLoRE read/write symmetry: reuse the same token expert bank at input, "
            "first loop entry, and final loop/mirror-side recurrent site"
        ),
    },
    {
        "name": "h100_evo_32k_e832_lexlore_rank4_warm_exit",
        "env": {
            **best_spine_env(embed_dim=832),
            "VOCAB_MOE_LAYERS": "input,loop_first,loop_last",
            "VOCAB_MOE_RANK": "4",
            "VOCAB_MOE_PRIOR_INIT_STD": "0.01",
            "VOCAB_MOE_SCALE_INIT": "0.08",
        },
        "notes": (
            "LexLoRE stronger wake-up: higher rank and non-uniform token priors so the "
            "adapter can specialize inside a 10-minute run"
        ),
    },
    {
        "name": "h100_evo_32k_e832_lexlore_32x2_warm_exit",
        "env": {
            **best_spine_env(embed_dim=832),
            "VOCAB_MOE_LAYERS": "input,loop_first,loop_last",
            "VOCAB_MOE_EXPERTS": "32",
            "VOCAB_MOE_PRIOR_INIT_STD": "0.01",
            "VOCAB_MOE_SCALE_INIT": "0.08",
        },
        "notes": (
            "LexLoRE more active bins: keep rank 2 but double the lexical expert bank "
            "with the same warm-start prior"
        ),
    },
    {
        "name": "h100_evo_32k_e832_cyclefuse_auto",
        "env": {
            **best_spine_env(embed_dim=832),
            "HRC_CYCLE_FUSE_ENABLED": "1",
            "HRC_CYCLE_FUSE_TAPS": "auto",
            "HRC_CYCLE_FUSE_INJECT_MODE": "repeat_phase",
            "HRC_CYCLE_FUSE_INIT_STD": "0.02",
        },
        "notes": (
            "CycleFuse HRC communication: later reused/mirrored phases see learned "
            "summaries of first-pass states instead of only repeating weights"
        ),
    },
    {
        "name": "h100_evo_32k_e768_ve32_layer3",
        "env": {
            **best_spine_env(embed_dim=768),
            "VE_ENABLED": "1",
            "VE_DIM": "32",
            "VE_LAYERS": "3",
        },
        "notes": (
            "ValueEmbedding as LexLoRE-inside-attention: inject token-conditioned value "
            "features into the first core attention block with a cap-safe VE32 spend"
        ),
    },
    {
        "name": "h100_evo_32k_e800_tail3after2_lidx",
        "env": {
            **best_spine_env(embed_dim=800),
            **partial_tail_route(io_width=3, loop_width=5, repeats=2, tail_width=3),
            **base.loop_index_env(dim=64, scale=0.02),
        },
        "notes": (
            "Partial recurrence HRC route: 012|34567|34567|567|210, using loop "
            "indexing so the extra semantic-tail refinement knows its pass position"
        ),
    },
]


GROUPS: dict[str, list[dict[str, Any]]] = {
    "h100_arch_evolution": H100_ARCH_EVOLUTION,
    "all": H100_ARCH_EVOLUTION,
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
    parser.add_argument("--candidate-group", default="h100_arch_evolution", choices=sorted(GROUPS))
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
    out_dir = Path(args.out) if args.out else base.ROOT / "records" / f"h100-arch-evolution-{time.strftime('%Y%m%d-%H%M%S')}"
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
