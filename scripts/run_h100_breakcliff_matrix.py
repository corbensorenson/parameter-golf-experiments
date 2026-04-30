"""Run 1xH100 break-cliff probes for the novel HRC/LexLoRE lane.

This is a paid-time runner, so it intentionally does three things:

1. Keep the candidate list short and architecture-distinguishing.
2. Preflight each row with a one-step export so over-cap/broken rows are skipped.
3. Run only full 10-minute candidates after preflight passes.

The rows below try to answer the post-8xH100 question: is the ~1.35 BPB wall
caused by schedule/step count, or by the architecture lacking token mixing,
write-side lexical specialization, or cross-pass communication?
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import run_h100_novel_matrix as base
from run_16mb_vocab_moe_matrix import (
    bigram_hash_env,
    dynamic_council_env,
    prime_superloop_route,
    rlm_memory_env,
    spike_vocab_moe_env,
)


TRAINER = base.ROOT / "train_gpt_arch_evolution.py"
base.TRAINER = TRAINER

CAP_BYTES = 16_000_000


def best32k(
    *,
    model_dim: int = 704,
    embed_dim: int = 832,
    repeats: int = 2,
    loop_width: int = 5,
    keep_attn_blocks: tuple[int, ...] = (3,),
    lqer_rank: int = 10,
    lqer_top_k: int = 20,
    batch_tokens: int = 32_768,
    warmdown_iters: int = 2200,
    qk_gain: float = 5.5,
) -> dict[str, str]:
    """Best legal 1xH100 spine with batch/step-count settings preserved."""

    return base.h100_env_with_overrides(
        {
            **base.baseline_chase_tail(
                model_dim=model_dim,
                embed_dim=embed_dim,
                loop_width=loop_width,
                repeats=repeats,
                keep_attn_blocks=keep_attn_blocks,
                lqer_rank=lqer_rank,
                lqer_top_k=lqer_top_k,
                qk_gain=qk_gain,
            ),
            "H100_1X_TRAIN_BATCH_TOKENS": str(batch_tokens),
            "H100_1X_VAL_BATCH_SIZE": "32768",
        },
        {"WARMDOWN_ITERS": str(warmdown_iters)},
    )


def partial_tail_route(
    *,
    io_width: int = 3,
    loop_width: int = 5,
    repeats: int = 2,
    tail_width: int = 3,
) -> dict[str, str]:
    """Route like 012|34567|34567|567|210: refine only the semantic tail."""

    return {
        "MODEL_FAMILY": "hrc",
        "NUM_UNIQUE_BLOCKS": str(io_width + loop_width),
        "EFFECTIVE_DEPTH": str(io_width + loop_width * repeats + tail_width + io_width),
        "HRC_RECURSIVE_CORE_START": str(io_width),
        "HRC_ROUTE_REPEATS": str(repeats),
        "HRC_DEPTH_SCHEDULE_MODE": f"transition_tail{tail_width}_cycle",
        "HRC_ROUTE_PHASE_ENABLED": "0",
    }


def prime_skip_env(
    *,
    model_dim: int = 640,
    embed_dim: int = 768,
    keep_attn_blocks: tuple[int, ...] = (3,),
    lqer_rank: int = 10,
    lqer_top_k: int = 20,
    batch_tokens: int = 32_768,
    warmdown_iters: int = 2200,
    qk_gain: float = 5.5,
    skip_ids: tuple[int, ...] = (0, 1),
) -> dict[str, str]:
    """Prime-skip HRC route around the best break-cliff signal."""

    return {
        **best32k(
            model_dim=model_dim,
            embed_dim=embed_dim,
            keep_attn_blocks=keep_attn_blocks,
            lqer_rank=lqer_rank,
            lqer_top_k=lqer_top_k,
            batch_tokens=batch_tokens,
            warmdown_iters=warmdown_iters,
            qk_gain=qk_gain,
        ),
        **prime_superloop_route(shell_width=3, prime_width=5, laps_per_skip=1, skip_ids=skip_ids),
        "HRC_LOOP_INDEX_ENABLED": "1",
        "HRC_LOOP_INDEX_DIM": "64",
        "HRC_LOOP_INDEX_SCALE_INIT": "0.02",
    }


def set_width_ladder(widths: str) -> dict[str, str]:
    return {
        "LAYER_WIDTH_SCHEDULE": widths,
        # Width adapters and depth-LoRA/basis sharing are intentionally not
        # combined in this trainer path.
        "DEPTH_LORA_RANK": "0",
        "BASIS_XSA_ENABLED": "0",
        "VE_ENABLED": "0",
    }


BREAKCLIFF_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "break_best32k_d704e832_control",
        "env": best32k(),
        "notes": (
            "Control for the new pod: known best legal 1xH100 family, batch32k, "
            "q8 train/export, one core-attention block, LexLoRE, LQER r10/t20."
        ),
    },
    {
        "name": "break_lexlore_exit_rank4_warm_d704e768",
        "env": {
            **best32k(embed_dim=768),
            "VOCAB_MOE_LAYERS": "input,loop_first,loop_last",
            "VOCAB_MOE_RANK": "4",
            "VOCAB_MOE_PRIOR_INIT_STD": "0.01",
            "VOCAB_MOE_SCALE_INIT": "0.08",
        },
        "notes": (
            "Write-side LexLoRE: lexical low-rank experts advise input, loop "
            "entry, and mirrored/exit-side state with a warmer rank-4 adapter."
        ),
    },
    {
        "name": "break_lexlore_spike32_top4_exit_d704e768",
        "env": {
            **best32k(embed_dim=768),
            **spike_vocab_moe_env(
                experts=32,
                rank=2,
                mode="hybrid",
                layers="input,loop_first,loop_last",
                train_quant_bits=8,
                spike_top_k=4,
            ),
            "VOCAB_MOE_PRIOR_INIT_STD": "0.01",
            "VOCAB_MOE_SCALE_INIT": "0.08",
        },
        "notes": (
            "Self-election LexLoRE: token experts make a sparse top-k choice "
            "instead of only dense prior mixing, including the write-side site."
        ),
    },
    {
        "name": "break_coreattn2_d704e768",
        "env": best32k(embed_dim=768, keep_attn_blocks=(3, 4)),
        "notes": (
            "Token-mixing test: keep attention alive in the first two recurrent "
            "core blocks instead of a mostly MLP-only middle."
        ),
    },
    {
        "name": "break_coreattn_all_d640e768",
        "env": best32k(model_dim=640, embed_dim=768, keep_attn_blocks=(3, 4, 5, 6, 7)),
        "notes": (
            "Strong token-mixing test: every recurrent core block keeps attention; "
            "body width is trimmed to keep the preflight likely cap-safe."
        ),
    },
    {
        "name": "break_i3l7r2_unique_loop_d640e768",
        "env": best32k(model_dim=640, embed_dim=768, loop_width=7, keep_attn_blocks=(3,)),
        "notes": (
            "More unique loop blocks rather than more repeats: tests the user's "
            "hypothesis that physical block diversity is the missing capacity."
        ),
    },
    {
        "name": "break_partial_tail_cycle_lidx_d704e768",
        "env": {
            **best32k(embed_dim=768),
            **partial_tail_route(io_width=3, loop_width=5, repeats=2, tail_width=3),
            **base.loop_index_env(dim=64, scale=0.02),
        },
        "notes": (
            "HRC route evolution: full core twice, then only the semantic tail "
            "before mirror exit, with loop-index conditioning."
        ),
    },
    {
        "name": "break_prime_skip_superloop_d640e768",
        "env": prime_skip_env(model_dim=640, embed_dim=768),
        "notes": (
            "Prime-route HRC: same 16 virtual steps as i3l5r2, but traverses "
            "the five-block core by two coprime skip programs."
        ),
    },
    {
        "name": "prime_follow_d704e768",
        "env": prime_skip_env(model_dim=704, embed_dim=768),
        "notes": (
            "Prime-skip cap spend on body width: tests whether the new route "
            "can use d704 without giving back the speed/quality win."
        ),
    },
    {
        "name": "prime_follow_d640e896",
        "env": prime_skip_env(model_dim=640, embed_dim=896),
        "notes": (
            "Prime-skip cap spend on token interface: uses the large headroom "
            "for a larger factored tied embedding while preserving d640 speed."
        ),
    },
    {
        "name": "prime_follow_d640e1024_lqer6t12",
        "env": prime_skip_env(model_dim=640, embed_dim=1024, lqer_rank=6, lqer_top_k=12),
        "notes": (
            "Aggressive token-interface spend on prime-skip, with smaller "
            "LQER to keep the exported artifact inside the decimal cap."
        ),
    },
    {
        "name": "prime_follow_coreattn2_d640e768",
        "env": prime_skip_env(model_dim=640, embed_dim=768, keep_attn_blocks=(3, 4)),
        "notes": (
            "Prime-skip plus the only attention result that looked mildly "
            "positive: keep attention in two core blocks, not all of them."
        ),
    },
    {
        "name": "prime_follow_lqer12t24_d640e768",
        "env": prime_skip_env(model_dim=640, embed_dim=768, lqer_rank=12, lqer_top_k=24),
        "notes": (
            "Prime-skip export-repair spend: use remaining bytes on stronger "
            "LQER instead of width or extra attention."
        ),
    },
    {
        "name": "prime_follow_batch24k_d640e896",
        "env": prime_skip_env(model_dim=640, embed_dim=896, batch_tokens=24_576),
        "notes": (
            "Prime-skip update-count test: smaller single-H100 batch should "
            "increase optimizer steps while still spending headroom on e896."
        ),
    },
    {
        "name": "prime_next_d640e960_lqer8t16",
        "env": prime_skip_env(model_dim=640, embed_dim=960, lqer_rank=8, lqer_top_k=16),
        "notes": (
            "Prime-skip middle token-interface spend: between e896 and e1024, "
            "with lighter LQER to preserve export headroom."
        ),
    },
    {
        "name": "prime_next_qk525_d640e896",
        "env": prime_skip_env(model_dim=640, embed_dim=896, qk_gain=5.25),
        "notes": (
            "Prime-skip QK gain sanity row: tests whether the public-leaderboard "
            "5.25 gain transfers better than the local H100 5.5 setting."
        ),
    },
    {
        "name": "prime_next_wd02_d640e896",
        "env": {
            **prime_skip_env(model_dim=640, embed_dim=896),
            "MUON_WEIGHT_DECAY": "0.02",
        },
        "notes": (
            "Prime-skip compressibility row: mild Muon WD may reduce the final "
            "export gap and spend bytes more cleanly than extra rank."
        ),
    },
    {
        "name": "prime_next_d704e704_lqer8t16",
        "env": prime_skip_env(model_dim=704, embed_dim=704, lqer_rank=8, lqer_top_k=16),
        "notes": (
            "Body-width spend with a safer token interface: d704 capacity, e704 "
            "embedding rank, and lighter LQER."
        ),
    },
    {
        "name": "prime_next_alt_skip02_d640e896",
        "env": prime_skip_env(model_dim=640, embed_dim=896, skip_ids=(0, 2)),
        "notes": (
            "Alternative prime-skip program with the same virtual depth: tests "
            "whether a more separated second lap is better than skip 1."
        ),
    },
    {
        "name": "break_hourglass_width_q888_d704e768",
        "env": {
            **best32k(embed_dim=768),
            **set_width_ladder("512,576,640,704,704,704,704,704"),
        },
        "notes": (
            "Hourglass width ladder: narrower high-precision IO blocks, full "
            "width recurrent core, q8 from the first training forward pass."
        ),
    },
    {
        "name": "break_precision_width_q16q8q8_d704e704",
        "env": {
            **best32k(embed_dim=704),
            **base.q8_core_io_ladder_env((16, 8, 8)),
            **set_width_ladder("512,576,640,704,704,704,704,704"),
        },
        "notes": (
            "True train-time precision ladder plus width ladder: q16/q8/q8 "
            "mirrored IO tail and q8 core, not post-hoc quantization."
        ),
    },
    {
        "name": "break_precision_width_q16q8q8_d704e704_fixed",
        "env": {
            **best32k(embed_dim=704),
            **base.q8_core_io_ladder_env((16, 8, 8)),
            **set_width_ladder("528,616,704,704,704,704,704,704"),
        },
        "notes": (
            "Fixed train-time precision plus width ladder: same q16/q8/q8 IO "
            "tail and q8 core, but widths are valid for d704/11 heads."
        ),
    },
    {
        "name": "break_rlm_memory_loopfirst_d704e768",
        "env": {
            **best32k(embed_dim=768),
            **rlm_memory_env(inject="loop_first", decay=0.90, scale=0.02),
        },
        "notes": (
            "Legal recursive memory: already-scored prefix state is compressed "
            "into a small vector and injected at loop entry."
        ),
    },
    {
        "name": "break_bigram_sidefeat_d704e704",
        "env": {
            **best32k(embed_dim=704),
            **bigram_hash_env(vocab_size=10240, dim=32),
        },
        "notes": (
            "Cheap token side-channel: BigramHash feature injection on the "
            "HRC/LexLoRE spine, borrowed from leaderboard-compatible tricks."
        ),
    },
    {
        "name": "break_dynamic_council_d704e704",
        "env": {
            **best32k(embed_dim=704),
            **dynamic_council_env(threshold=6.0, min_gate=0.01),
        },
        "notes": (
            "Conditional council: ask a mirrored peer only on uncertain contexts, "
            "mixing full distributions without target-conditioned routing."
        ),
    },
]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _candidate_subset(raw: str) -> list[dict[str, Any]]:
    if not raw.strip():
        return BREAKCLIFF_CANDIDATES
    wanted = [item.strip() for item in raw.split(",") if item.strip()]
    by_name = {str(candidate["name"]): candidate for candidate in BREAKCLIFF_CANDIDATES}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise ValueError(f"unknown candidate(s): {', '.join(missing)}")
    return [by_name[name] for name in wanted]


def _preflight(
    *,
    out_dir: Path,
    candidates: list[dict[str, Any]],
    data_path: Path,
    tokenizer_path: Path,
    nproc_per_node: int,
    cap_bytes: int,
    val_tokens: int,
    timeout: int,
) -> list[dict[str, Any]]:
    preflight_dir = out_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    rows = base.run_matrix(
        out_dir=preflight_dir,
        candidates=candidates,
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        nproc_per_node=nproc_per_node,
        wallclock_seconds=45,
        val_tokens=val_tokens,
        timeout=timeout,
        final_artifacts=True,
        iterations=1,
        warmdown_iters=1,
    )
    survivors: list[dict[str, Any]] = []
    decisions: list[dict[str, object]] = []
    by_name = {str(candidate["name"]): candidate for candidate in candidates}
    for row in rows:
        name = str(row.get("candidate", ""))
        total_bytes = row.get("artifact_total_bytes")
        headroom = row.get("artifact_headroom")
        returncode = int(row.get("returncode", 1))
        ok = returncode == 0 and isinstance(total_bytes, int) and total_bytes <= cap_bytes
        if ok:
            survivors.append(by_name[name])
        decisions.append(
            {
                "candidate": name,
                "returncode": returncode,
                "artifact_total_bytes": total_bytes if total_bytes is not None else "",
                "artifact_headroom": headroom if headroom is not None else "",
                "decision": "run_full" if ok else "skip",
                "reason": "cap_safe" if ok else "failed_or_over_cap_or_missing_size",
            }
        )
    _write_csv(out_dir / "preflight_decisions.csv", decisions)
    print(
        f"preflight survivors: {len(survivors)}/{len(candidates)} under {cap_bytes:,} bytes",
        flush=True,
    )
    for candidate in survivors:
        print(f"  run_full {candidate['name']}", flush=True)
    return survivors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--candidates", default="")
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--wallclock-seconds", type=int, default=600)
    parser.add_argument("--val-tokens", type=int, default=131072)
    parser.add_argument("--timeout", type=int, default=1500)
    parser.add_argument("--data-path", default=str(base.DATASET_DIR))
    parser.add_argument("--tokenizer-path", default=str(base.TOKENIZER_PATH))
    parser.add_argument("--cap-bytes", type=int, default=CAP_BYTES)
    parser.add_argument("--preflight", action="store_true", default=True)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--preflight-val-tokens", type=int, default=4096)
    parser.add_argument("--preflight-timeout", type=int, default=420)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    candidates = _candidate_subset(args.candidates)
    if args.list:
        for candidate in candidates:
            print(candidate["name"])
        return 0
    if not TRAINER.is_file():
        print(f"Missing trainer: {TRAINER}")
        return 1

    data_path = Path(args.data_path)
    tokenizer_path = Path(args.tokenizer_path)
    if not data_path.is_dir() or not tokenizer_path.is_file():
        print(f"Missing CaseOps SP8192 data/tokenizer:\n  data={data_path}\n  tokenizer={tokenizer_path}")
        return 1

    out_dir = Path(args.out) if args.out else base.ROOT / "records" / f"h100-breakcliff-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base.write_candidate_plan(out_dir, candidates, 1_000_000, args.val_tokens)

    full_candidates = candidates
    if bool(args.preflight and not args.skip_preflight):
        full_candidates = _preflight(
            out_dir=out_dir,
            candidates=candidates,
            data_path=data_path,
            tokenizer_path=tokenizer_path,
            nproc_per_node=args.nproc_per_node,
            cap_bytes=args.cap_bytes,
            val_tokens=args.preflight_val_tokens,
            timeout=args.preflight_timeout,
        )
    if not full_candidates:
        print("No candidates survived preflight; nothing to run.", flush=True)
        return 2

    full_dir = out_dir / "full"
    full_dir.mkdir(parents=True, exist_ok=True)
    base.write_candidate_plan(full_dir, full_candidates, 1_000_000, args.val_tokens)
    base.run_matrix(
        out_dir=full_dir,
        candidates=full_candidates,
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        nproc_per_node=args.nproc_per_node,
        wallclock_seconds=args.wallclock_seconds,
        val_tokens=args.val_tokens,
        timeout=args.timeout,
        final_artifacts=True,
        iterations=None,
        warmdown_iters=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
