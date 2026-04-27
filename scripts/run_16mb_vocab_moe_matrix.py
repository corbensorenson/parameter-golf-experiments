"""Run 16MB vocabulary-MoE probes on the local 2060.

The feature under test is VOCAB_MOE_* in train_gpt.py: token-conditioned
low-rank shared experts. These rows keep final artifact round-trip enabled so
the signal is the exported model, not only train-time loss.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from run_caseops_candidate_2060_compare import (
    DATASET_DIR,
    LOOP_EXACT_5K_Q6_PROOF_BASE,
    TOKENIZER_PATH,
    TRAINER,
    plain_loop_route,
)
from run_sub4_iotail_quant_matrix import wait_for_idle_gpu
from run_sub4_micro_matrix import (
    PYTHON,
    ROOT,
    configure_env,
    merged_train_output,
    parse_train,
    run_command,
    write_csv,
)


PUBLIC_STACK_16MB: dict[str, str] = {
    "MODEL_CODEC": "lzma",
    "MODEL_CODEC_LEVEL": "9",
    "SUBMISSION_SIZE_CAP_BYTES": "16000000",
    "FAIL_ON_ARTIFACT_CAP": "0",
    "TRAIN_ABORT_ON_NONFINITE": "1",
    "QUANT_WEIGHT_BITS": "6",
    "QUANT_INT8_PROMOTE_PATTERNS": "tok_emb.weight,lm_head.weight,embed_proj",
    "MODEL_DIM": "640",
    "NUM_HEADS": "10",
    "NUM_KV_HEADS": "1",
    "FACTORED_EMBED_DIM": "256",
    "MLP_MULT": "2.0",
    # Train-quant-forward is touchier than the old post-training q6 proof
    # runs. These cooler values keep the control row finite on the 2060 so the
    # VocabMoE comparisons produce usable signal instead of late-run NaNs.
    "WARMUP_STEPS": "20",
    "TIED_EMBED_LR": "0.002",
    "MATRIX_LR": "0.0016",
    "SCALAR_LR": "0.0016",
    "QK_GAIN_INIT": "5.0",
    "LOGIT_SOFTCAP": "12",
    "ATTN_OUT_GATE_ENABLED": "1",
    "ATTN_OUT_GATE_WIDTH": "24",
    "SMEAR_GATE_ENABLED": "1",
    "SMEAR_GATE_WIDTH": "12",
    "HRC_FROZEN_CARRY_ENABLED": "0",
    "HRC_FROZEN_CARRY_BLOCKS": "",
    "MUON_WEIGHT_DECAY": "0.0",
    "MUON_WEIGHT_DECAY_MODE": "huber",
    "MUON_WEIGHT_DECAY_HUBER_DELTA_SCALE": "3.0",
    "LQER_ENABLED": "1",
    "LQER_RANK": "8",
    "LQER_TOP_K": "12",
    "LQER_ASYM_ENABLED": "1",
    "LQER_ASYM_GROUP": "64",
    "LQER_INCLUDE_PATTERNS": "blocks.",
    "LQER_EXCLUDE_PATTERNS": "tok_emb.weight,lm_head.weight,embed_proj,token_smear,attn_gate_w,attn_out_gate,vocab_moe",
}


BASE_16MB: dict[str, str] = {
    **LOOP_EXACT_5K_Q6_PROOF_BASE,
    **PUBLIC_STACK_16MB,
    **plain_loop_route(3, 3, 3),
}


def vocab_moe_env(
    *,
    experts: int,
    rank: int,
    mode: str,
    layers: str,
    scale: float = 0.05,
    up_init: float = 0.001,
    temperature: float = 1.0,
    activation: str = "relu2",
    site_bias: bool = True,
    site_scale: bool = True,
    train_quant_bits: int = 6,
    spike_top_k: int = 0,
    spike_ste: bool = True,
    spike_normalize: bool = True,
) -> dict[str, str]:
    return {
        "VOCAB_MOE_ENABLED": "1",
        "VOCAB_MOE_EXPERTS": str(experts),
        "VOCAB_MOE_RANK": str(rank),
        "VOCAB_MOE_MODE": mode,
        "VOCAB_MOE_LAYERS": layers,
        "VOCAB_MOE_SCALE_INIT": f"{scale:g}",
        "VOCAB_MOE_PRIOR_INIT_STD": "0.0",
        "VOCAB_MOE_DOWN_INIT_STD": "0.02",
        "VOCAB_MOE_UP_INIT_STD": f"{up_init:g}",
        "VOCAB_MOE_TEMPERATURE": f"{temperature:g}",
        "VOCAB_MOE_ACTIVATION": activation,
        "VOCAB_MOE_TRAIN_QUANT_BITS": str(train_quant_bits),
        "VOCAB_MOE_SITE_BIAS_ENABLED": "1" if site_bias else "0",
        "VOCAB_MOE_SITE_SCALE_ENABLED": "1" if site_scale else "0",
        "VOCAB_MOE_SITE_SCALE_INIT": "1.0",
        "VOCAB_MOE_SPIKE_TOP_K": str(spike_top_k),
        "VOCAB_MOE_SPIKE_STE": "1" if spike_ste else "0",
        "VOCAB_MOE_SPIKE_NORMALIZE": "1" if spike_normalize else "0",
        "QUANT_FORCE_PATTERNS": "vocab_moe.token_prior.weight,vocab_moe.down,vocab_moe.up",
    }


CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "i3l3r3_d640e256_q6_stable_control",
        "env": dict(BASE_16MB),
        "notes": "same cooled q6 route/stack, no vocab MoE",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_static_k16r2_input",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="static", layers="input"),
        },
        "notes": "token prior only, cheapest vocabulary-conditioned adapter",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="loop_first"),
        },
        "notes": "hidden router plus token prior at first repeated block",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="input,loop_first"),
        },
        "notes": "adapter before the stack and at loop entry",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopevery3",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="loop_every3"),
        },
        "notes": "regular token-conditioned repair through the recurrent middle",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopall",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="loop"),
        },
        "notes": "maximal repeated-core token conditioning; slower but high-signal",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hidden_k16r2_loopfirst",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="hidden", layers="loop_first"),
        },
        "notes": "hidden-state router only, checks whether token priors matter",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k32r1_loopfirst",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=32, rank=1, mode="hybrid", layers="loop_first"),
        },
        "notes": "more token clusters at similar expert parameter cost",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k8r4_loopfirst",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=8, rank=4, mode="hybrid", layers="loop_first"),
        },
        "notes": "fewer token clusters with richer per-expert rank",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r4_loopfirst",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=4, mode="hybrid", layers="loop_first"),
        },
        "notes": "more rank at the same expert count",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k32r2_loopfirst",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=32, rank=2, mode="hybrid", layers="loop_first"),
        },
        "notes": "larger adapter for the 16MB lane, tests byte-for-quality spend",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst_t07",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="loop_first", temperature=0.7),
        },
        "notes": "sharper routing distribution",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst_t15",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="loop_first", temperature=1.5),
        },
        "notes": "softer routing distribution",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst_nosite",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(
                experts=16,
                rank=2,
                mode="hybrid",
                layers="loop_first",
                site_bias=False,
                site_scale=False,
            ),
        },
        "notes": "ablation for site bias/scale controls",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopevery3",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="input,loop_every3"),
        },
        "notes": "input feature plus recurrent repair sites",
    },
    {
        "name": "i4l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst",
        "env": {
            **BASE_16MB,
            **plain_loop_route(4, 3, 3),
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="loop_first"),
        },
        "notes": "wider mirrored IO tail with the same first-loop adapter",
    },
    {
        "name": "i3l5r2_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst",
        "env": {
            **BASE_16MB,
            **plain_loop_route(3, 5, 2),
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="loop_first"),
        },
        "notes": "more unique loop blocks, fewer repeats, same adapter",
    },
    {
        "name": "i4l5r2_d640e256_q6_vocabmoe_hybrid_k16r2_loopfirst",
        "env": {
            **BASE_16MB,
            **plain_loop_route(4, 5, 2),
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="loop_first"),
        },
        "notes": "more IO and more unique loop capacity",
    },
    {
        "name": "i3l3r3_d768e256_q6_vocabmoe_hybrid_k16r2_loopfirst",
        "env": {
            **BASE_16MB,
            "MODEL_DIM": "768",
            "NUM_HEADS": "12",
            "FACTORED_EMBED_DIM": "256",
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="loop_first"),
        },
        "notes": "width spend probe; may be slower on the 2060 but valuable for H100 planning",
    },
    {
        "name": "i3l3r3_d768e320_q6_vocabmoe_hybrid_k16r2_loopfirst",
        "env": {
            **BASE_16MB,
            "MODEL_DIM": "768",
            "NUM_HEADS": "12",
            "FACTORED_EMBED_DIM": "320",
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="loop_first"),
        },
        "notes": "wider embedding factor with Vocab-MoE",
    },
]


SPIKE_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "i3l3r3_d640e256_q6_stable_control",
        "env": dict(BASE_16MB),
        "notes": "same cooled q6 route/stack, no vocab MoE; anchor for sparse routing",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_static_k16r2_input_dense_anchor",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="static", layers="input"),
        },
        "notes": "dense static VocabMoE anchor from the current family",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikestatic_k16r2_input_top1",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="spike_static", layers="input", spike_top_k=1),
        },
        "notes": "pure token self-election: each token fires one expert",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikestatic_k16r2_input_top2",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="spike_static", layers="input", spike_top_k=2),
        },
        "notes": "token self-election with two active experts",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikestatic_k32r1_input_top2",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=32, rank=1, mode="spike_static", layers="input", spike_top_k=2),
        },
        "notes": "more token clusters at similar rank cost",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopfirst_top1",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="spike_hybrid", layers="loop_first", spike_top_k=1),
        },
        "notes": "hidden-state router nudges token self-election at loop entry",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopfirst_top2",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="spike_hybrid", layers="loop_first", spike_top_k=2),
        },
        "notes": "hybrid self-election with two loop-entry experts",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_input_loopfirst_top2",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(
                experts=16,
                rank=2,
                mode="spike_hybrid",
                layers="input,loop_first",
                spike_top_k=2,
            ),
        },
        "notes": "self-election before the stack and at the first repeated block",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopevery3_top2",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="spike_hybrid", layers="loop_every3", spike_top_k=2),
        },
        "notes": "sparse token repair through the recurrent middle",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopall_top2",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="spike_hybrid", layers="loop", spike_top_k=2),
        },
        "notes": "max sparse recurrent repair; slower but high-signal",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehidden_k16r2_loopfirst_top2",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="spike_hidden", layers="loop_first", spike_top_k=2),
        },
        "notes": "router-only sparse control, no token prior",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k32r1_loopfirst_top2",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=32, rank=1, mode="spike_hybrid", layers="loop_first", spike_top_k=2),
        },
        "notes": "more expert buckets with cheap rank-1 bases",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r4_loopfirst_top2",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=4, mode="spike_hybrid", layers="loop_first", spike_top_k=2),
        },
        "notes": "more rank behind the sparse gate",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopfirst_top2_nonorm",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(
                experts=16,
                rank=2,
                mode="spike_hybrid",
                layers="loop_first",
                spike_top_k=2,
                spike_normalize=False,
            ),
        },
        "notes": "damped sparse gate: selected experts keep original softmax mass",
    },
]


CANDIDATE_GROUPS: dict[str, list[dict[str, Any]]] = {
    "default": CANDIDATES,
    "vocabmoe": CANDIDATES,
    "vocabmoe_spike": SPIKE_CANDIDATES,
    "all": CANDIDATES + SPIKE_CANDIDATES,
}


def selected_candidates(raw: str, group: str = "default") -> list[dict[str, Any]]:
    if group not in CANDIDATE_GROUPS:
        raise ValueError(f"unknown candidate group {group!r}; choices: {', '.join(sorted(CANDIDATE_GROUPS))}")
    pool = CANDIDATE_GROUPS[group]
    if not raw.strip():
        return pool
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    out = [candidate for candidate in pool if candidate["name"] in wanted]
    missing = wanted - {candidate["name"] for candidate in out}
    if missing:
        raise ValueError(f"unknown candidate(s): {', '.join(sorted(missing))}")
    return out


def write_candidate_plan(out_dir: Path, candidates: list[dict[str, Any]], iterations: int, val_tokens: int) -> None:
    lines = [
        "# 16MB Vocab-MoE Matrix",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Iterations: {iterations}",
        f"Validation tokens: {val_tokens}",
        "",
        "| Candidate | Route | Vocab MoE | Notes |",
        "|---|---|---|---|",
    ]
    for candidate in candidates:
        env = candidate["env"]
        moe = "disabled"
        if env.get("VOCAB_MOE_ENABLED") == "1":
            moe = (
                f"k={env['VOCAB_MOE_EXPERTS']} r={env['VOCAB_MOE_RANK']} "
                f"mode={env['VOCAB_MOE_MODE']} layers={env['VOCAB_MOE_LAYERS']} "
                f"temp={env['VOCAB_MOE_TEMPERATURE']} "
                f"site={env['VOCAB_MOE_SITE_BIAS_ENABLED']}/{env['VOCAB_MOE_SITE_SCALE_ENABLED']} "
                f"train_q={env['VOCAB_MOE_TRAIN_QUANT_BITS']} "
                f"spike_k={env.get('VOCAB_MOE_SPIKE_TOP_K', '0')} "
                f"ste={env.get('VOCAB_MOE_SPIKE_STE', '1')} "
                f"norm={env.get('VOCAB_MOE_SPIKE_NORMALIZE', '1')}"
            )
        route = (
            f"unique={env['NUM_UNIQUE_BLOCKS']} depth={env['EFFECTIVE_DEPTH']} "
            f"start={env['HRC_RECURSIVE_CORE_START']} repeats={env['HRC_ROUTE_REPEATS']}"
        )
        lines.append(f"| `{candidate['name']}` | `{route}` | `{moe}` | {candidate['notes']} |")
    (out_dir / "candidate_plan.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(out_dir: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# 16MB Vocab-MoE Matrix Summary",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    good = [row for row in rows if row.get("returncode") == 0 and ("val_bpb" in row or "final_export_val_bpb" in row)]

    def score(row: dict[str, object]) -> float:
        return float(row.get("final_quant_ttt_val_bpb", row.get("final_export_val_bpb", row.get("val_bpb", 999.0))))

    if not good:
        lines.append("No completed rows yet.")
    for row in sorted(good, key=score):
        if "final_quant_ttt_val_bpb" in row:
            score_text = f"final_ttt_bpb={row['final_quant_ttt_val_bpb']}"
        elif "final_export_val_bpb" in row:
            score_text = f"final_bpb={row['final_export_val_bpb']}"
        else:
            score_text = f"val_bpb={row['val_bpb']}"
        lines.append(
            f"- `{row['candidate']}`: {score_text}, train_val={row.get('val_bpb', 'n/a')}, "
            f"step_avg={row.get('step_avg_ms', 'n/a')}ms, "
            f"bytes={row.get('artifact_total_bytes', 'n/a')}, "
            f"headroom={row.get('artifact_headroom', 'n/a')}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_matrix(
    *,
    out_dir: Path,
    candidates: list[dict[str, Any]],
    iterations: int,
    warmdown_iters: int,
    val_tokens: int,
    timeout: int,
    wallclock_seconds: int,
    final_artifacts: bool,
    train_quant_forward: bool,
    wait_for_idle: bool,
    idle_max_util: int,
    idle_max_memory_mib: int,
    idle_seconds: int,
    idle_poll_seconds: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        idle_wait_s = wait_for_idle_gpu(
            enabled=wait_for_idle,
            max_util=idle_max_util,
            max_memory_mib=idle_max_memory_mib,
            quiet_seconds=idle_seconds,
            poll_seconds=idle_poll_seconds,
            label=name,
        )
        run_id = f"vocabmoe16_{name}_{iterations}"
        env = configure_env()
        env.update(candidate["env"])
        env.update(
            {
                "RUN_ID": run_id,
                "DATA_PATH": str(DATASET_DIR),
                "TOKENIZER_PATH": str(TOKENIZER_PATH),
                "VOCAB_SIZE": "8192",
                "ITERATIONS": str(iterations),
                "WARMDOWN_ITERS": str(warmdown_iters if warmdown_iters > 0 else iterations),
                "VAL_TOKENS_LIMIT": str(val_tokens),
                "MAX_WALLCLOCK_SECONDS": str(wallclock_seconds),
                "SKIP_FINAL_ARTIFACTS": "0" if final_artifacts else "1",
                "TRAIN_QUANT_FORWARD": "1" if train_quant_forward else "0",
                "QUANT_TRAIN_MODE": "none",
                "LOG_CODE_SNAPSHOT": "0",
                "LOG_NVIDIA_SMI": "0",
                "PYTHONUNBUFFERED": "1",
            }
        )
        started = time.perf_counter()
        proc = run_command([str(PYTHON), "-u", str(TRAINER)], env, timeout=timeout)
        stdout = merged_train_output(proc.stdout, run_id)
        raw_path = out_dir / f"train_{name}.txt"
        raw_path.write_text(stdout, encoding="utf-8")
        row: dict[str, object] = {
            "candidate": name,
            "iterations": iterations,
            "warmdown_iters": warmdown_iters if warmdown_iters > 0 else iterations,
            "wallclock_seconds": wallclock_seconds,
            "val_tokens": val_tokens,
            "final_artifacts": int(final_artifacts),
            "train_quant_forward": int(train_quant_forward),
            "idle_wait_s": idle_wait_s,
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
        write_summary(out_dir, rows)
        print(f"train {name} rc={proc.returncode}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--warmdown-iters", type=int, default=0)
    parser.add_argument("--val-tokens", type=int, default=131072)
    parser.add_argument("--timeout", type=int, default=9000)
    parser.add_argument("--wallclock-seconds", type=int, default=0)
    parser.add_argument("--candidate-group", default="default", choices=sorted(CANDIDATE_GROUPS))
    parser.add_argument("--candidates", default="")
    parser.add_argument("--final-artifacts", action="store_true", default=True)
    parser.add_argument("--skip-final-artifacts", action="store_true")
    parser.add_argument("--train-quant-forward", action="store_true", default=True)
    parser.add_argument("--no-train-quant-forward", action="store_true")
    parser.add_argument("--wait-for-idle-gpu", action="store_true")
    parser.add_argument("--idle-max-util", type=int, default=15)
    parser.add_argument("--idle-max-memory-mib", type=int, default=2500)
    parser.add_argument("--idle-seconds", type=int, default=30)
    parser.add_argument("--idle-poll-seconds", type=int, default=5)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    candidates = selected_candidates(args.candidates, args.candidate_group)
    if args.list:
        for candidate in candidates:
            print(candidate["name"])
        return 0
    if not DATASET_DIR.is_dir() or not TOKENIZER_PATH.is_file():
        print(
            f"Missing CaseOps SP8192 data/tokenizer:\n  data={DATASET_DIR}\n  tokenizer={TOKENIZER_PATH}",
            flush=True,
        )
        return 1
    out_dir = Path(args.out) if args.out else ROOT / "records" / f"vocabmoe16_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_artifacts = bool(args.final_artifacts and not args.skip_final_artifacts)
    train_quant_forward = bool(args.train_quant_forward and not args.no_train_quant_forward)
    write_candidate_plan(out_dir, candidates, args.iterations, args.val_tokens)
    rows = run_matrix(
        out_dir=out_dir,
        candidates=candidates,
        iterations=args.iterations,
        warmdown_iters=args.warmdown_iters,
        val_tokens=args.val_tokens,
        timeout=args.timeout,
        wallclock_seconds=args.wallclock_seconds,
        final_artifacts=final_artifacts,
        train_quant_forward=train_quant_forward,
        wait_for_idle=args.wait_for_idle_gpu,
        idle_max_util=args.idle_max_util,
        idle_max_memory_mib=args.idle_max_memory_mib,
        idle_seconds=args.idle_seconds,
        idle_poll_seconds=args.idle_poll_seconds,
    )
    write_summary(out_dir, rows)
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
