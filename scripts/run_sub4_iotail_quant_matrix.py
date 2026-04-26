"""Run sub-4MB IO-tail precision ladder experiments.

This matrix explores HRC routes where the mirrored IO tail keeps more export
precision while the repeated middle is exported ternary. The core trainer
already supports pattern-based export quantization; this script wires those
patterns per candidate without editing counted trainer files during active runs.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Any

from run_sub4_micro_matrix import (
    ROOT,
    PYTHON,
    configure_env,
    merged_train_output,
    parse_train,
    run_command,
    write_csv,
)


CASEOPS_DATA_PATH = (
    "upstream_records/records/track_10min_16mb/2026-04-18_PR1626_CaseOps_Taper/"
    "datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
)
CASEOPS_TOKENIZER_PATH = (
    "upstream_records/records/track_10min_16mb/2026-04-18_PR1626_CaseOps_Taper/"
    "tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
)


def block_patterns(start: int, stop: int) -> str:
    return ",".join(f"blocks.{idx}." for idx in range(start, stop))


def quant_bits(*pairs: tuple[int, int]) -> str:
    return ",".join(f"blocks.{idx}.:{bits}" for idx, bits in pairs)


def mlp_only(start: int, stop: int) -> str:
    return ",".join(str(idx) for idx in range(start, stop))


COMMON_ENV = {
    "DATA_PATH": CASEOPS_DATA_PATH,
    "TOKENIZER_PATH": CASEOPS_TOKENIZER_PATH,
    "VOCAB_SIZE": "8192",
    "MODEL_CODEC": "lzma",
    "MODEL_CODEC_LEVEL": "9",
    "SUBMISSION_SIZE_CAP_BYTES": "4000000",
    "FAIL_ON_ARTIFACT_CAP": "1",
    "LOG_CODE_SNAPSHOT": "0",
    "LOG_NVIDIA_SMI": "0",
    "SKIP_INITIAL_VAL": "1",
    "VAL_LOSS_EVERY": "1000",
    "TRAIN_LOG_EVERY": "1000",
    "TRAIN_SEQ_LEN": "64",
    "TRAIN_BATCH_TOKENS": "8192",
    "VAL_BATCH_SIZE": "8192",
    "GRAD_ACCUM_STEPS": "1",
    "MAX_WALLCLOCK_SECONDS": "0",
    "OPTIMIZER_PRESET": "hybrid",
    "USE_GRAD_SCALER": "0",
    "PARAM_DTYPE": "fp16",
    "MUON_DTYPE": "fp16",
    "POST_STEP_ZERO_GRAD": "0",
    "TRAIN_FUSED_QKV": "1",
    "TRAIN_CASTED_LINEAR_PARAM_DTYPE": "model",
    # The runner can additionally enable TRAIN_QUANT_FORWARD=1, which makes
    # CastedLinear use q8/q6/q4/ternary STE views from the first forward pass.
    # Keep the old export/reload roundtrip projection opt-in only: it is useful
    # as a guardrail, but it is too slow and invasive for wall-clock sweeps.
    "TRAIN_TERNARY_BLOCKS": "0",
    "QUANT_TRAIN_MODE": "none",
    "QUANT_TRAIN_START_FRACTION": "0.0",
    "QUANT_TRAIN_EVERY": "1",
    "QUANT_TERNARY_GROUP_SIZE": "128",
    "QUANT_TERNARY_SCALE_STAT": "mean",
    "QUANT_TERNARY_SHRINKAGE_FIX": "1",
    "QUANT_TERNARY_EXCLUDE_PATTERNS": "tok_emb.weight,lm_head.weight,embed_proj",
    # Keep the cheap control path precise while we test aggressive block export.
    "INT8_KEEP_FLOAT_MAX_NUMEL": "4096",
    "KEEP_CONTROL_PARAMS_FP32": "1",
    "LOSS_FP32": "1",
    "LOSS_TOKEN_STRIDE": "1",
    "LOGIT_SOFTCAP": "12",
    "LR_WARMUP_ITERS": "200",
    "LR_WARMDOWN_STYLE": "cosine",
    "LR_MIN_SCALE": "0",
    "TIED_EMBED_LR": "0.00015",
    "MATRIX_LR": "0.0002",
    "SCALAR_LR": "0.0002",
}


def route_env(
    *,
    io_width: int,
    loop_width: int,
    repeats: int,
    model_dim: int,
    embed_dim: int,
    heads: int,
    mlp_mult: float,
    io_quant: tuple[int, ...],
) -> dict[str, str]:
    unique_blocks = io_width + loop_width
    effective_depth = io_width + loop_width * repeats + io_width
    if len(io_quant) != io_width:
        raise ValueError("io_quant must provide one bit width per IO block")
    return {
        "MODEL_DIM": str(model_dim),
        "FACTORED_EMBED_DIM": str(embed_dim),
        "NUM_HEADS": str(heads),
        "NUM_KV_HEADS": "1",
        "MLP_MULT": str(float(mlp_mult)),
        "NUM_UNIQUE_BLOCKS": str(unique_blocks),
        "EFFECTIVE_DEPTH": str(effective_depth),
        "HRC_RECURSIVE_CORE_START": str(io_width),
        "HRC_ROUTE_REPEATS": str(repeats),
        "HRC_DEPTH_SCHEDULE_MODE": "transition_recursive_cycle",
        "HRC_MLP_ONLY_BLOCKS": mlp_only(io_width, unique_blocks),
        "QUANT_WEIGHT_BITS": "4",
        "QUANT_BITS_OVERRIDES": quant_bits(*[(idx, bits) for idx, bits in enumerate(io_quant)]),
        "QUANT_TERNARY_PATTERNS": block_patterns(io_width, unique_blocks),
    }


def lqer_env(rank: int = 8, top_k: int = 16) -> dict[str, str]:
    return {
        "LQER_ENABLED": "1",
        "LQER_RANK": str(rank),
        "LQER_TOP_K": str(top_k),
        "LQER_ASYM_ENABLED": "1",
        "LQER_ASYM_GROUP": "64",
        "LQER_INCLUDE_PATTERNS": "tok_emb.weight,embed_proj,blocks.",
        "LQER_EXCLUDE_PATTERNS": "lm_head.weight,token_smear,attn_gate_w,attn_out_gate",
    }


CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "i1l2r2_d768e256_q8_coret_lqer",
        "base_profile": "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": {
            **route_env(
                io_width=1,
                loop_width=2,
                repeats=2,
                model_dim=768,
                embed_dim=256,
                heads=12,
                mlp_mult=0.75,
                io_quant=(8,),
            ),
            **lqer_env(rank=8, top_k=16),
        },
    },
    {
        "name": "i1l2r2_d896e256_q8_coret_lqer",
        "base_profile": "i1l2r2_d896_e256_h14kv1_mlpinner_mlp050",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": {
            **route_env(
                io_width=1,
                loop_width=2,
                repeats=2,
                model_dim=896,
                embed_dim=256,
                heads=14,
                mlp_mult=0.5,
                io_quant=(8,),
            ),
            **lqer_env(rank=8, top_k=16),
        },
    },
    {
        "name": "i3l3r2_d768e256_q864_coret",
        "base_profile": "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": route_env(
            io_width=3,
            loop_width=3,
            repeats=2,
            model_dim=768,
            embed_dim=256,
            heads=12,
            mlp_mult=0.75,
            io_quant=(8, 6, 4),
        ),
    },
    {
        "name": "i3l3r2_d768e256_q864_coret_lqer",
        "base_profile": "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": {
            **route_env(
                io_width=3,
                loop_width=3,
                repeats=2,
                model_dim=768,
                embed_dim=256,
                heads=12,
                mlp_mult=0.75,
                io_quant=(8, 6, 4),
            ),
            **lqer_env(rank=8, top_k=16),
        },
    },
    {
        "name": "i3l3r3_d768e256_q864_coret",
        "base_profile": "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": route_env(
            io_width=3,
            loop_width=3,
            repeats=3,
            model_dim=768,
            embed_dim=256,
            heads=12,
            mlp_mult=0.75,
            io_quant=(8, 6, 4),
        ),
    },
    {
        "name": "i3l3r3_d768e256_q864_coret_lqer",
        "base_profile": "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": {
            **route_env(
                io_width=3,
                loop_width=3,
                repeats=3,
                model_dim=768,
                embed_dim=256,
                heads=12,
                mlp_mult=0.75,
                io_quant=(8, 6, 4),
            ),
            **lqer_env(rank=8, top_k=16),
        },
    },
    {
        "name": "i3l3r3_d768e256_q884_coret_lqer",
        "base_profile": "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": {
            **route_env(
                io_width=3,
                loop_width=3,
                repeats=3,
                model_dim=768,
                embed_dim=256,
                heads=12,
                mlp_mult=0.75,
                io_quant=(8, 8, 4),
            ),
            **lqer_env(rank=8, top_k=16),
        },
    },
    {
        "name": "i3l3r3_d768e256_q886_coret_lqer",
        "base_profile": "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": {
            **route_env(
                io_width=3,
                loop_width=3,
                repeats=3,
                model_dim=768,
                embed_dim=256,
                heads=12,
                mlp_mult=0.75,
                io_quant=(8, 8, 6),
            ),
            **lqer_env(rank=8, top_k=16),
        },
    },
    {
        "name": "i6l9r3_d256e96_q886644_coret",
        "base_profile": "i6l9r2_d256_e96",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": route_env(
            io_width=6,
            loop_width=9,
            repeats=3,
            model_dim=256,
            embed_dim=96,
            heads=8,
            mlp_mult=3.0,
            io_quant=(8, 8, 6, 6, 4, 4),
        ),
    },
    {
        "name": "i6l9r3_d256e96_q888666_coret",
        "base_profile": "i6l9r2_d256_e96",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": route_env(
            io_width=6,
            loop_width=9,
            repeats=3,
            model_dim=256,
            embed_dim=96,
            heads=8,
            mlp_mult=3.0,
            io_quant=(8, 8, 8, 6, 6, 6),
        ),
    },
    {
        "name": "i6l9r3_d320e96_q886644_coret",
        "base_profile": "i6l10r2_d320_e96",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": route_env(
            io_width=6,
            loop_width=9,
            repeats=3,
            model_dim=320,
            embed_dim=96,
            heads=8,
            mlp_mult=3.0,
            io_quant=(8, 8, 6, 6, 4, 4),
        ),
    },
    {
        "name": "i6l9r3_d320e128_q886644_coret_lqer",
        "base_profile": "i6l10r2_d320_e96",
        "preset": "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
        "env": {
            **route_env(
                io_width=6,
                loop_width=9,
                repeats=3,
                model_dim=320,
                embed_dim=128,
                heads=8,
                mlp_mult=3.0,
                io_quant=(8, 8, 6, 6, 4, 4),
            ),
            **lqer_env(rank=8, top_k=16),
        },
    },
]


def selected_candidates(raw: str) -> list[dict[str, Any]]:
    if not raw:
        return CANDIDATES
    wanted = {item.strip() for item in raw.replace(";", ",").split(",") if item.strip()}
    out = [candidate for candidate in CANDIDATES if candidate["name"] in wanted]
    missing = wanted - {candidate["name"] for candidate in out}
    if missing:
        raise ValueError(f"unknown candidate(s): {', '.join(sorted(missing))}")
    return out


def run_matrix(
    *,
    out_dir: Path,
    candidates: list[dict[str, Any]],
    iterations: int,
    wallclock_seconds: int,
    warmdown_iters: int,
    val_tokens: int,
    timeout: int,
    final_artifacts: bool,
    train_quant_forward: bool,
    quant_train_every: int,
    roundtrip_guard: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        run_id = f"iotail_quant_{name}_{iterations}"
        env = configure_env()
        env.update(COMMON_ENV)
        env.update(candidate["env"])
        env.update(
            {
                "SUB4_PROFILE": str(candidate["base_profile"]),
                "SUB4_SPEED_PRESET": str(candidate["preset"]),
                "ITERATIONS": str(iterations),
                "WARMDOWN_ITERS": str(warmdown_iters if warmdown_iters > 0 else iterations),
                "VAL_TOKENS_LIMIT": str(val_tokens),
                "RUN_ID": run_id,
                "MAX_WALLCLOCK_SECONDS": str(wallclock_seconds),
                "TRAIN_QUANT_FORWARD": "1" if train_quant_forward else "0",
                "QUANT_TRAIN_MODE": "roundtrip" if roundtrip_guard else "none",
                "QUANT_TRAIN_EVERY": str(quant_train_every),
                "SKIP_FINAL_ARTIFACTS": "0" if final_artifacts else "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        started = time.perf_counter()
        proc = run_command([str(PYTHON), "train_gpt_ternary.py"], env, timeout=timeout)
        stdout = merged_train_output(proc.stdout, run_id)
        raw_path = out_dir / f"train_{name}.txt"
        raw_path.write_text(stdout, encoding="utf-8")
        row: dict[str, object] = {
            "candidate": name,
            "base_profile": candidate["base_profile"],
            "preset": candidate["preset"],
            "iterations": iterations,
            "wallclock_seconds": wallclock_seconds,
            "warmdown_iters": warmdown_iters if warmdown_iters > 0 else iterations,
            "final_artifacts": int(final_artifacts),
            "train_quant_forward": int(train_quant_forward),
            "quant_train_every": quant_train_every,
            "roundtrip_guard": int(roundtrip_guard),
            "returncode": proc.returncode,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "raw_log": raw_path.name,
            "run_id": run_id,
            "env_overrides": ",".join(f"{key}={value}" for key, value in sorted(candidate["env"].items())),
        }
        parsed = parse_train(stdout)
        if parsed:
            row.update(parsed)
        if proc.returncode != 0:
            row["error_tail"] = "\n".join(stdout.splitlines()[-12:])
        rows.append(row)
        write_csv(out_dir / "train.csv", rows)
        print(f"train {name} rc={proc.returncode}", flush=True)
    return rows


def write_candidate_plan(out_dir: Path, candidates: list[dict[str, Any]], iterations: int) -> None:
    lines = [
        "# Sub-4MB IO-Tail Quant Matrix",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Iterations: {iterations}",
        "",
        "| Candidate | Route | IO Quant | Core Quant | Base |",
        "|---|---|---|---|---|",
    ]
    for candidate in candidates:
        env = candidate["env"]
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{candidate['name']}`",
                    f"unique={env['NUM_UNIQUE_BLOCKS']} depth={env['EFFECTIVE_DEPTH']} "
                    f"start={env['HRC_RECURSIVE_CORE_START']} repeats={env['HRC_ROUTE_REPEATS']}",
                    f"`{env['QUANT_BITS_OVERRIDES']}`",
                    f"`{env['QUANT_TERNARY_PATTERNS']}`",
                    f"`{candidate['base_profile']}`",
                ]
            )
            + " |"
        )
    (out_dir / "candidate_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(out_dir: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Sub-4MB IO-Tail Quant Matrix Summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    good = [row for row in rows if row.get("returncode") == 0 and "val_bpb" in row]

    def key(row: dict[str, object]) -> float:
        return float(row.get("final_export_val_bpb", row.get("val_bpb", 999.0)))

    for row in sorted(good, key=key):
        score = (
            f"final_bpb={row['final_export_val_bpb']}"
            if "final_export_val_bpb" in row
            else f"val_bpb={row['val_bpb']}"
        )
        lines.append(
            f"- `{row['candidate']}`: {score}, train_val={row.get('val_bpb')}, "
            f"step_avg={row.get('step_avg_ms')}ms, bytes={row.get('artifact_total_bytes', 'n/a')}, "
            f"headroom={row.get('artifact_headroom', 'n/a')}"
        )
    if not good:
        lines.append("No completed rows yet.")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--wallclock-seconds", type=int, default=0)
    parser.add_argument("--warmdown-iters", type=int, default=0)
    parser.add_argument("--val-tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--candidates", default="")
    parser.add_argument("--final-artifacts", action="store_true")
    parser.add_argument("--train-quant-forward", action="store_true")
    parser.add_argument("--roundtrip-guard", action="store_true")
    parser.add_argument("--quant-train-every", type=int, default=100)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    candidates = selected_candidates(args.candidates)
    if args.list:
        for candidate in candidates:
            print(candidate["name"])
        return 0
    out_dir = Path(args.out) if args.out else ROOT / "records" / f"sub4_iotail_quant_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_plan(out_dir, candidates, args.iterations)
    rows = run_matrix(
        out_dir=out_dir,
        candidates=candidates,
        iterations=args.iterations,
        wallclock_seconds=args.wallclock_seconds,
        warmdown_iters=args.warmdown_iters,
        val_tokens=args.val_tokens,
        timeout=args.timeout,
        final_artifacts=args.final_artifacts,
        train_quant_forward=args.train_quant_forward,
        quant_train_every=args.quant_train_every,
        roundtrip_guard=args.roundtrip_guard,
    )
    write_summary(out_dir, rows)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
