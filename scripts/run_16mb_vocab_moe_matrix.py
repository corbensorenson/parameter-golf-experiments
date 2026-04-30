"""Run 16MB vocabulary-MoE probes on the local 2060.

The feature under test is VOCAB_MOE_* in train_gpt.py: token-conditioned
low-rank shared experts. These rows keep final artifact round-trip enabled so
the signal is the exported model, not only train-time loss.
"""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from run_caseops_candidate_2060_compare import (
    DATASET_DIR,
    LOOP_EXACT_5K_Q6_PROOF_BASE,
    SUB16_PORTED_FAST_LOOP,
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


def block_patterns(start: int, stop: int) -> str:
    return ",".join(f"blocks.{idx}." for idx in range(start, stop))


def quant_bits(*pairs: tuple[int, int]) -> str:
    return ",".join(f"blocks.{idx}.:{bits}" for idx, bits in pairs)


CAP16_STABLE_SPEED_COMMON: dict[str, str] = {
    "PARAM_DTYPE": "fp32",
    "USE_GRAD_SCALER": "1",
    "MUON_DTYPE": "fp32",
    "LOSS_FP32": "1",
    "POST_STEP_ZERO_GRAD": "1",
    "TRAIN_CASTED_LINEAR_PARAM_DTYPE": "model",
    "TRAIN_TERNARY_PARAM_DTYPE": "model",
    "KEEP_CONTROL_PARAMS_FP32": "1",
    "TRAIN_FUSED_QKV": "1",
    "GRAD_ACCUM_STEPS": "4",
    # Final export roundtrip is the score that matters for these triage rows.
    # Periodic validation was useful while debugging, but it now costs walltime.
    "VAL_LOSS_EVERY": "0",
    "TRAIN_LOG_EVERY": "250",
}


CAP16_SPEED_COMMON: dict[str, str] = {
    # Make the low-precision path truthful from the first forward pass while
    # keeping the optimizer numerics on the stability-safe 2060 path. The
    # fully fp16-param/no-scaler experiment was faster, but collapsed exported
    # BPB to ~4.15, so future rows use the stable dtype path and only retain
    # speed levers that have not broken quality.
    **CAP16_STABLE_SPEED_COMMON,
    "QK_GAIN_INIT": "5.25",
    "LQER_RANK": "12",
    "LQER_TOP_K": "24",
}


def cap16_clean_base(
    *,
    model_dim: int = 640,
    embed_dim: int = 256,
    io_width: int = 3,
    loop_width: int = 3,
    repeats: int = 3,
) -> dict[str, str]:
    if model_dim % 64 != 0:
        raise ValueError(f"model_dim must keep 64-wide heads, got {model_dim}")
    return {
        **BASE_16MB,
        **CAP16_STABLE_SPEED_COMMON,
        **plain_loop_route(io_width, loop_width, repeats),
        "MODEL_DIM": str(model_dim),
        "NUM_HEADS": str(model_dim // 64),
        "FACTORED_EMBED_DIM": str(embed_dim),
    }


def cap16_nospeed_base(
    *,
    model_dim: int = 640,
    embed_dim: int = 256,
    io_width: int = 3,
    loop_width: int = 3,
    repeats: int = 3,
) -> dict[str, str]:
    if model_dim % 64 != 0:
        raise ValueError(f"model_dim must keep 64-wide heads, got {model_dim}")
    return {
        **BASE_16MB,
        **plain_loop_route(io_width, loop_width, repeats),
        "MODEL_DIM": str(model_dim),
        "NUM_HEADS": str(model_dim // 64),
        "FACTORED_EMBED_DIM": str(embed_dim),
    }


def cap16_speed_base(
    *,
    model_dim: int = 640,
    embed_dim: int = 256,
    io_width: int = 3,
    loop_width: int = 3,
    repeats: int = 3,
) -> dict[str, str]:
    if model_dim % 64 != 0:
        raise ValueError(f"model_dim must keep 64-wide heads, got {model_dim}")
    return {
        **BASE_16MB,
        **CAP16_SPEED_COMMON,
        **plain_loop_route(io_width, loop_width, repeats),
        "MODEL_DIM": str(model_dim),
        "NUM_HEADS": str(model_dim // 64),
        "FACTORED_EMBED_DIM": str(embed_dim),
    }


def cap16_baseline_base(
    *,
    model_dim: int = 512,
    embed_dim: int = 384,
    num_layers: int = 11,
    num_kv_heads: int = 1,
) -> dict[str, str]:
    """Dense transformer branch for testing whether HRC is the bottleneck."""
    if model_dim % 64 != 0:
        raise ValueError(f"model_dim must keep 64-wide heads, got {model_dim}")
    num_heads = model_dim // 64
    if num_heads % max(num_kv_heads, 1) != 0:
        raise ValueError(f"num_kv_heads={num_kv_heads} must divide num_heads={num_heads}")
    return {
        **BASE_16MB,
        **CAP16_STABLE_SPEED_COMMON,
        "MODEL_FAMILY": "baseline",
        "NUM_LAYERS": str(num_layers),
        "MODEL_DIM": str(model_dim),
        "NUM_HEADS": str(num_heads),
        "NUM_KV_HEADS": str(num_kv_heads),
        "FACTORED_EMBED_DIM": str(embed_dim),
        # HRC defaults leak in through the local 2060 launcher; keep the dense
        # branch clean unless a row explicitly asks for a routing trick.
        "PARALLEL_RESIDUAL_LAST_N": "0",
        "RESIDUAL_MIXER_ENABLED": "0",
        "HRC_LOOP_INDEX_ENABLED": "0",
        "HRC_PASS_EMBED_ENABLED": "0",
        "HRC_RECUR_INJECT_ENABLED": "0",
        "HRC_ROUTE_PHASE_ENABLED": "0",
        "DEPTH_LORA_RANK": "0",
    }


def prime_superloop_route(
    *,
    shell_width: int,
    prime_width: int,
    laps_per_skip: int,
    skip_ids: tuple[int, ...] = (1,),
) -> dict[str, str]:
    """Prime-width HRC route that walks the recurrent ring by skip programs."""
    if prime_width < 3:
        raise ValueError("prime_width must be >= 3")
    unique_blocks = int(shell_width) + int(prime_width)
    effective_depth = 2 * int(shell_width) + int(prime_width) * max(int(laps_per_skip), 1) * len(skip_ids)
    return {
        "MODEL_FAMILY": "hrc",
        "NUM_UNIQUE_BLOCKS": str(unique_blocks),
        "EFFECTIVE_DEPTH": str(effective_depth),
        "HRC_RECURSIVE_CORE_START": str(shell_width),
        "HRC_ROUTE_REPEATS": str(max(int(laps_per_skip), 1)),
        "HRC_DEPTH_SCHEDULE_MODE": "prime_skip_superloop",
        "HRC_SUPERLOOP_SKIP_SCHEDULE": ",".join(str(skip_id) for skip_id in skip_ids),
        "HRC_ROUTE_PHASE_ENABLED": "1",
    }


def palindrome_loop_route(io_width: int, loop_width: int, loop_repeats: int) -> dict[str, str]:
    """Build entry-tail + cycle-rev recurrent middle + mirrored-exit route settings."""
    core_depth = (2 * loop_width - 1) + max(loop_repeats - 1, 0) * (2 * loop_width - 2)
    effective_depth = io_width + core_depth + io_width
    return {
        "MODEL_FAMILY": "hrc",
        "NUM_UNIQUE_BLOCKS": str(io_width + loop_width),
        "EFFECTIVE_DEPTH": str(effective_depth),
        "HRC_RECURSIVE_CORE_START": str(io_width),
        "HRC_ROUTE_REPEATS": str(loop_repeats),
        "HRC_DEPTH_SCHEDULE_MODE": "transition_recursive_palindrome",
        "HRC_ROUTE_PHASE_ENABLED": "0",
    }


def cap16_palindrome_base(
    *,
    model_dim: int = 768,
    embed_dim: int = 320,
    io_width: int = 3,
    loop_width: int = 3,
    repeats: int = 2,
) -> dict[str, str]:
    if model_dim % 64 != 0:
        raise ValueError(f"model_dim must keep 64-wide heads, got {model_dim}")
    return {
        **BASE_16MB,
        **CAP16_SPEED_COMMON,
        **palindrome_loop_route(io_width, loop_width, repeats),
        "MODEL_DIM": str(model_dim),
        "NUM_HEADS": str(model_dim // 64),
        "FACTORED_EMBED_DIM": str(embed_dim),
    }


def cap16_taper_env(
    *,
    io_width: int,
    loop_width: int,
    io_bits: tuple[int, ...],
    core_bits: int = 4,
) -> dict[str, str]:
    if len(io_bits) != io_width:
        raise ValueError("io_bits must provide one entry per IO-tail unique block")
    unique_blocks = int(io_width) + int(loop_width)
    return {
        "QUANT_WEIGHT_BITS": "6",
        "QUANT_BITS_OVERRIDES": quant_bits(
            *[(idx, bits) for idx, bits in enumerate(io_bits)],
            *[(idx, core_bits) for idx in range(io_width, unique_blocks)],
        ),
        "LQER_INCLUDE_PATTERNS": "blocks.,embed_proj",
        "LQER_EXCLUDE_PATTERNS": (
            "tok_emb.weight,lm_head.weight,token_smear,attn_gate_w,attn_out_gate,"
            "vocab_moe,dual_stream"
        ),
    }


def cap16_lqer_env(rank: int, top_k: int) -> dict[str, str]:
    return {
        "LQER_RANK": str(rank),
        "LQER_TOP_K": str(top_k),
        "LQER_INCLUDE_PATTERNS": "blocks.,embed_proj",
        "LQER_EXCLUDE_PATTERNS": (
            "tok_emb.weight,lm_head.weight,token_smear,attn_gate_w,attn_out_gate,"
            "vocab_moe,dual_stream"
        ),
    }


def q8_train_export_env() -> dict[str, str]:
    return {
        "QUANT_WEIGHT_BITS": "8",
        "VOCAB_MOE_TRAIN_QUANT_BITS": "8",
        "QUANT_INT8_PROMOTE_PATTERNS": "tok_emb.weight,lm_head.weight,embed_proj",
    }


FINAL_ONLY_EVAL_ENV: dict[str, str] = {
    # Promotion is based on the exported artifact roundtrip. Skipping periodic
    # validation saves local wall time without changing training updates.
    "VAL_LOSS_EVERY": "0",
    "TRAIN_LOG_EVERY": "250",
}


def q8_core_io_ladder_env(io_bits: tuple[int, ...] = (16, 8, 4)) -> dict[str, str]:
    return {
        **q8_train_export_env(),
        # Keep the proven q8 recurrent core and token interface, but train the
        # mirrored IO-tail ladder from the first forward pass.
        "QUANT_BITS_OVERRIDES": quant_bits(*[(idx, bits) for idx, bits in enumerate(io_bits)]),
    }


def leaderboard_schedule_env(*, min_lr: float = 0.026, ns_variant: str = "polar_express") -> dict[str, str]:
    return {
        "LR_MIN_SCALE": f"{min_lr:g}",
        "MUON_NS_VARIANT": ns_variant,
        "MUON_BACKEND_STEPS": "5",
    }


def sparse_gate_env(*, width: int = 24, scale: float = 1.0, init_std: float = 0.0) -> dict[str, str]:
    return {
        "ATTN_OUT_GATE_ENABLED": "0",
        "SPARSE_ATTN_GATE_ENABLED": "1",
        "ATTN_OUT_GATE_WIDTH": str(width),
        "SPARSE_ATTN_GATE_INIT_STD": f"{init_std:g}",
        "SPARSE_ATTN_GATE_SCALE": f"{scale:g}",
    }


def parallel_residual_env(*, last_n: int = 3) -> dict[str, str]:
    return {
        "PARALLEL_RESIDUAL_LAST_N": str(last_n),
        "RESIDUAL_MIXER_ENABLED": "1",
    }


def legal_ttt_env(*, lr: float = 0.005, updates: int = 24) -> dict[str, str]:
    return {
        "TTT_SCORE_FIRST_ENABLED": "1",
        "TTT_SCORE_FIRST_PARAM_MODE": "control",
        "TTT_SCORE_FIRST_OPTIMIZER": "sgd",
        "TTT_SCORE_FIRST_LR": f"{lr:g}",
        "TTT_SCORE_FIRST_WEIGHT_DECAY": "0.0",
        "TTT_SCORE_FIRST_GRAD_CLIP": "1.0",
        "TTT_SCORE_FIRST_MAX_UPDATES": str(updates),
    }


def bigram_hash_env(*, vocab_size: int = 10240, dim: int = 32) -> dict[str, str]:
    return {
        "BIGRAM_VOCAB_SIZE": str(vocab_size),
        "BIGRAM_DIM": str(dim),
        "BIGRAM_INIT_STD": "0.02",
        "BIGRAM_SCALE_INIT": "0.05",
        "LQER_EXCLUDE_PATTERNS": (
            "tok_emb.weight,lm_head.weight,token_smear,attn_gate_w,attn_out_gate,"
            "vocab_moe,dual_stream,bigram"
        ),
    }


def dual_stream_env(
    *,
    left_dim: int,
    rank: int = 16,
    sites: str = "input,loop_first,pre_output",
    scale: float = 0.02,
) -> dict[str, str]:
    return {
        "DUAL_STREAM_ENABLED": "1",
        "DUAL_STREAM_LEFT_DIM": str(left_dim),
        "DUAL_STREAM_RANK": str(rank),
        "DUAL_STREAM_SITES": sites,
        "DUAL_STREAM_SCALE_INIT": f"{scale:g}",
        "DUAL_STREAM_DOWN_INIT_STD": "0.02",
        "DUAL_STREAM_UP_INIT_STD": "0.001",
        "DUAL_STREAM_ACTIVATION": "silu",
    }


def vocab_moe_env(
    *,
    experts: int,
    rank: int,
    mode: str,
    layers: str,
    scale: float = 0.05,
    prior_init_std: float = 0.0,
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
        "VOCAB_MOE_PRIOR_INIT_STD": f"{prior_init_std:g}",
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


def spike_vocab_moe_env(**kwargs: Any) -> dict[str, str]:
    # Hard top-k routing needs a tiny deterministic tie-breaker; otherwise
    # zero token priors make every token initially elect the same expert bucket.
    kwargs.setdefault("prior_init_std", 0.01)
    return vocab_moe_env(**kwargs)


BEST_CLEAN_VOCABMOE = vocab_moe_env(
    experts=16,
    rank=2,
    mode="hybrid",
    layers="input,loop_first",
)


def council_env(
    *,
    mode: str = "base_mirror",
    mirror: str = "signperm",
    offsets: str = "0,-2",
    conf_scale: float = 1.0,
    train_mode: str = "eval_only",
    hard_gate: bool = False,
    threshold: float = 6.0,
) -> dict[str, str]:
    return {
        "HRC_COUNCIL_MODE": mode,
        "HRC_MIRROR_MODE": mirror,
        "HRC_COUNCIL_TRAIN_MODE": train_mode,
        "HRC_COUNCIL_DEPTH_OFFSETS": offsets,
        "HRC_COUNCIL_CONF_SCALE_INIT": f"{conf_scale:g}",
        "HRC_COUNCIL_HARD_GATE": "1" if hard_gate else "0",
        "HRC_COUNCIL_ENTROPY_THRESHOLD": f"{threshold:g}",
        "HRC_COUNCIL_ENTROPY_SHARPNESS": "8.0",
        "HRC_COUNCIL_SANITIZE": "1",
        "HRC_COUNCIL_LOGIT_CLAMP": "60",
    }


def dynamic_council_env(*, threshold: float = 6.0, min_gate: float = 0.01) -> dict[str, str]:
    return {
        **council_env(hard_gate=True, threshold=threshold),
        "HRC_DYNAMIC_COUNCIL_ENABLED": "1",
        "HRC_DYNAMIC_COUNCIL_THRESHOLD": f"{threshold:g}",
        "HRC_DYNAMIC_COUNCIL_SHARPNESS": "8.0",
        "HRC_DYNAMIC_COUNCIL_MIN_GATE": f"{min_gate:g}",
    }


def rlm_memory_env(*, inject: str = "input", decay: float = 0.90, scale: float = 0.02) -> dict[str, str]:
    return {
        "RLM_MEMORY_ENABLED": "1",
        "RLM_MEMORY_TRAIN_ENABLED": "1",
        "RLM_MEMORY_DECAY": f"{decay:g}",
        "RLM_MEMORY_SCALE_INIT": f"{scale:g}",
        "RLM_MEMORY_INJECT": inject,
        "RLM_MEMORY_UPDATE": "hidden_mean",
        "RLM_MEMORY_RESET_EACH_EVAL": "1",
        # Smaller validation batches let the legal prefix memory update more
        # often without seeing targets from the current chunk.
        "VAL_BATCH_SIZE": "4096",
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
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_dense_anchor",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "best dense VocabMoE anchor for fair sparse/self-election comparison",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikestatic_k16r2_input_top1",
        "env": {
            **BASE_16MB,
            **spike_vocab_moe_env(experts=16, rank=2, mode="spike_static", layers="input", spike_top_k=1),
        },
        "notes": "pure token self-election: each token fires one expert",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikestatic_k16r2_input_top2",
        "env": {
            **BASE_16MB,
            **spike_vocab_moe_env(experts=16, rank=2, mode="spike_static", layers="input", spike_top_k=2),
        },
        "notes": "token self-election with two active experts",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikestatic_k32r1_input_top2",
        "env": {
            **BASE_16MB,
            **spike_vocab_moe_env(experts=32, rank=1, mode="spike_static", layers="input", spike_top_k=2),
        },
        "notes": "more token clusters at similar rank cost",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopfirst_top1",
        "env": {
            **BASE_16MB,
            **spike_vocab_moe_env(experts=16, rank=2, mode="spike_hybrid", layers="loop_first", spike_top_k=1),
        },
        "notes": "hidden-state router nudges token self-election at loop entry",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopfirst_top2",
        "env": {
            **BASE_16MB,
            **spike_vocab_moe_env(experts=16, rank=2, mode="spike_hybrid", layers="loop_first", spike_top_k=2),
        },
        "notes": "hybrid self-election with two loop-entry experts",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_input_loopfirst_top2",
        "env": {
            **BASE_16MB,
            **spike_vocab_moe_env(
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
            **spike_vocab_moe_env(experts=16, rank=2, mode="spike_hybrid", layers="loop_every3", spike_top_k=2),
        },
        "notes": "sparse token repair through the recurrent middle",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopall_top2",
        "env": {
            **BASE_16MB,
            **spike_vocab_moe_env(experts=16, rank=2, mode="spike_hybrid", layers="loop", spike_top_k=2),
        },
        "notes": "max sparse recurrent repair; slower but high-signal",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehidden_k16r2_loopfirst_top2",
        "env": {
            **BASE_16MB,
            **spike_vocab_moe_env(experts=16, rank=2, mode="spike_hidden", layers="loop_first", spike_top_k=2),
        },
        "notes": "router-only sparse control, no token prior",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k32r1_loopfirst_top2",
        "env": {
            **BASE_16MB,
            **spike_vocab_moe_env(experts=32, rank=1, mode="spike_hybrid", layers="loop_first", spike_top_k=2),
        },
        "notes": "more expert buckets with cheap rank-1 bases",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r4_loopfirst_top2",
        "env": {
            **BASE_16MB,
            **spike_vocab_moe_env(experts=16, rank=4, mode="spike_hybrid", layers="loop_first", spike_top_k=2),
        },
        "notes": "more rank behind the sparse gate",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_spikehybrid_k16r2_loopfirst_top2_nonorm",
        "env": {
            **BASE_16MB,
            **spike_vocab_moe_env(
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


COUNCIL_RLM_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_anchor",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "best completed clean VocabMoE lane anchor for council/RLM comparisons",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_signperm_o0m2",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
            **council_env(mirror="signperm", offsets="0,-2"),
        },
        "notes": "eval-only self-consistency council: base plus sign-permutation peer, shallower mirror",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_house_o0m2",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
            **council_env(mirror="householder", offsets="0,-2"),
        },
        "notes": "eval-only council with householder peer transform",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_signperm_o00",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
            **council_env(mirror="signperm", offsets="0,0"),
        },
        "notes": "same-depth base/mirror council; checks whether the -2 depth offset is too conservative",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_hybrid_o00m1",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
            **council_env(mode="base_mirror_hybrid", mirror="signperm", offsets="0,0,-1"),
        },
        "notes": "three-peer base/mirror/hybrid council; adapts the requested 0,-1 offsets to one value per peer",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_council_hard_t60",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
            **council_env(mirror="signperm", offsets="0,-2", hard_gate=True, threshold=6.0),
        },
        "notes": "entropy-gated self-consistency: mix the peer only where base uncertainty is high",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_dynamic_council_t60",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
            **dynamic_council_env(threshold=6.0, min_gate=0.01),
        },
        "notes": "dynamic-depth analogue: run the mirror peer only for eval chunks with hard-token entropy",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_dynamic_council_t55",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
            **dynamic_council_env(threshold=5.5, min_gate=0.01),
        },
        "notes": "more aggressive dynamic council threshold",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_rlm_input_d90_s002",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
            **rlm_memory_env(inject="input", decay=0.90, scale=0.02),
        },
        "notes": "legal RLM-lite prefix memory, injected at the next chunk's input",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_rlm_loopfirst_d90_s002",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
            **rlm_memory_env(inject="loop_first", decay=0.90, scale=0.02),
        },
        "notes": "prefix memory injected only at recurrent-core entry",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_rlm_inputloop_d95_s001",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
            **rlm_memory_env(inject="input_loop_first", decay=0.95, scale=0.01),
        },
        "notes": "slower-decay prefix memory at input and loop entry",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_rlm_council_signperm",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
            **rlm_memory_env(inject="input", decay=0.90, scale=0.02),
            **council_env(mirror="signperm", offsets="0,-2"),
        },
        "notes": "RLM-lite persistent prefix plus eval-only self-consistency council",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_loopevery3_s002_council_signperm",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="loop_every3", scale=0.02),
            **council_env(mirror="signperm", offsets="0,-2"),
        },
        "notes": "stabilized loop-every-3 adapter plus eval-only base/mirror council",
    },
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopevery3_s002_rlm_council",
        "env": {
            **BASE_16MB,
            **vocab_moe_env(experts=16, rank=2, mode="hybrid", layers="input,loop_every3", scale=0.02),
            **rlm_memory_env(inject="input", decay=0.90, scale=0.02),
            **council_env(mirror="signperm", offsets="0,-2"),
        },
        "notes": "high-upside combined row: stabilized recurrent VocabMoE, RLM-lite memory, and council",
    },
]


CAP16_SPEED_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "i3l3r3_d640e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24",
        "env": {
            **cap16_speed_base(model_dim=640, embed_dim=256),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": (
            "changed-code-path anchor: best dense VocabMoE placement plus fp16 "
            "linear params, fp16 Muon, fused QKV, q6 forward from step zero, QK 5.25, and richer LQER"
        ),
    },
    {
        "name": "i3l3r3_d768e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=256),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "spend 16MB headroom on residual width while keeping the proven input+loop-first adapter",
    },
    {
        "name": "i3l3r3_d768e320_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "same residual width as the d768 probe, but spends extra bytes on the factored tied embedding",
    },
    {
        "name": "i3l5r2_d768e256_q6_vocabmoe_hybrid_k16r2_input_loopfirst_cap16fast_qk525_lqer12t24",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=256, io_width=3, loop_width=5, repeats=2),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "tests the sub-4 lesson that more unique loop blocks can beat more repeats at similar virtual depth",
    },
]


CAP16_MAINLINE_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "mainline_i3l3r3_d768e384_q6all_vocabmoe_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=384),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "mainline width/embedding spend: q6 throughout, e384, QK 5.25, VocabMoE input+loop-first, LQER r16/t32",
    },
    {
        "name": "mainline_i3l3r3_d896e384_q6all_vocabmoe_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=896, embed_dim=384),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "larger residual-width spend under the 16MB cap; useful if d768 is not memory-bound locally",
    },
    {
        "name": "mainline_i3l5r2_d768e320_q6all_vocabmoe_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320, io_width=3, loop_width=5, repeats=2),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "tests more unique loop blocks with the full mainline stack but no q4 pressure",
    },
    {
        "name": "mainline_i3l5r2_d768e320_q8q6q6_q4core_vocabmoe_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320, io_width=3, loop_width=5, repeats=2),
            **cap16_taper_env(io_width=3, loop_width=5, io_bits=(8, 6, 6), core_bits=4),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "explicit q8/q6/q6 IO-tail and q4 recurrent-core taper from the first forward pass",
    },
    {
        "name": "mainline_i3l5r2_d896e384_q8q6q6_q4core_vocabmoe_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=896, embed_dim=384, io_width=3, loop_width=5, repeats=2),
            **cap16_taper_env(io_width=3, loop_width=5, io_bits=(8, 6, 6), core_bits=4),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "quality-first d896 version of the precision-tapered unique-loop mainline",
    },
]


CAP16_ANCHOR_SPEND_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "anchor_i3l3r3_d640e256_q6_control",
        "env": {
            **BASE_16MB,
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "fresh same-night control for the current best dense VocabMoE anchor",
    },
    {
        "name": "anchor_i3l3r3_d640e256_q6_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=640, embed_dim=256),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "lowest-risk upgrade: current anchor plus QK 5.25 and stronger LQER",
    },
    {
        "name": "anchor_i3l3r3_d640e384_q6_qk525_lqer24t48",
        "env": {
            **cap16_speed_base(model_dim=640, embed_dim=384),
            **cap16_lqer_env(24, 48),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "spends bytes on the token interface and more quant-error repair without widening residual blocks",
    },
    {
        "name": "anchor_i3l3r3_d640e512_q8_qk525_lqer24t48",
        "env": {
            **cap16_speed_base(model_dim=640, embed_dim=512),
            **cap16_lqer_env(24, 48),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "quality-first cap-fill row: richer factored embeddings plus train-time q8 instead of q6",
    },
    {
        "name": "anchor_i3l3r3_d704e320_q6_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=704, embed_dim=320),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "moderate residual-width spend between d640 and the d768 rows that overpaid locally",
    },
    {
        "name": "anchor_i3l3r3_d704e384_q8_qk525_lqer24t48",
        "env": {
            **cap16_speed_base(model_dim=704, embed_dim=384),
            **cap16_lqer_env(24, 48),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "closer-to-16MB quality row: moderate width, richer embedding rank, q8 train/export, and stronger LQER",
    },
    {
        "name": "anchor_i3l3r3_d640e256_q6_vocabmoe_k16r4_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=640, embed_dim=256),
            **cap16_lqer_env(16, 32),
            **vocab_moe_env(experts=16, rank=4, mode="hybrid", layers="input,loop_first"),
        },
        "notes": "spends bytes inside the winning VocabMoE placement by doubling expert rank",
    },
    {
        "name": "anchor_i3l3r3_d640e256_q6_vocabmoe_k32r2_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=640, embed_dim=256),
            **cap16_lqer_env(16, 32),
            **vocab_moe_env(experts=32, rank=2, mode="hybrid", layers="input,loop_first"),
        },
        "notes": "spends bytes on more token/expert buckets while preserving the proven rank-2 basis size",
    },
    {
        "name": "anchor_i3l5r2_d640e256_q6_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=640, embed_dim=256, io_width=3, loop_width=5, repeats=2),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "tests whether the i3/l5/r2 near-miss was loop diversity rather than d768 width",
    },
    {
        "name": "anchor_i3l3r3_d640e256_q6_spikehybrid_top2_t15_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=640, embed_dim=256),
            **cap16_lqer_env(16, 32),
            **spike_vocab_moe_env(
                experts=16,
                rank=2,
                mode="spike_hybrid",
                layers="input,loop_first",
                spike_top_k=2,
                temperature=1.5,
            ),
        },
        "notes": "targeted follow-up to the 1.8792 spikehybrid near-miss with softer self-election logits",
    },
]


CAP16_ANCHOR_CLEAN_SPEND_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "clean_i3l3r3_d640e256_q6_control_speed",
        "env": {
            **cap16_clean_base(model_dim=640, embed_dim=256),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "same anchor recipe with only stable speed levers added; no QK 5.25 or heavier LQER",
    },
    {
        "name": "clean_i3l3r3_d640e384_q6",
        "env": {
            **cap16_clean_base(model_dim=640, embed_dim=384),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "spends bytes on factored embedding rank while keeping q6 and the proven QK/LQER settings",
    },
    {
        "name": "clean_i3l3r3_d640e512_q8",
        "env": {
            **cap16_clean_base(model_dim=640, embed_dim=512),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "cap-fill token-interface row: larger factored embedding and q8 train/export without QK/LQER confound",
    },
    {
        "name": "clean_i3l3r3_d704e320_q6",
        "env": {
            **cap16_clean_base(model_dim=704, embed_dim=320),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "moderate residual-width spend between d640 and the d768 rows that overpaid locally",
    },
    {
        "name": "clean_i3l3r3_d704e384_q8",
        "env": {
            **cap16_clean_base(model_dim=704, embed_dim=384),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "closer-to-cap quality row with moderate width, richer embedding rank, and q8 train/export",
    },
    {
        "name": "clean_i3l3r3_d640e256_q6_vocabmoe_k16r4",
        "env": {
            **cap16_clean_base(model_dim=640, embed_dim=256),
            **vocab_moe_env(experts=16, rank=4, mode="hybrid", layers="input,loop_first"),
        },
        "notes": "spends bytes inside the winning VocabMoE placement by doubling expert rank",
    },
    {
        "name": "clean_i3l3r3_d640e256_q6_vocabmoe_k32r2",
        "env": {
            **cap16_clean_base(model_dim=640, embed_dim=256),
            **vocab_moe_env(experts=32, rank=2, mode="hybrid", layers="input,loop_first"),
        },
        "notes": "spends bytes on more token/expert buckets while preserving rank-2 experts",
    },
    {
        "name": "clean_i3l5r2_d640e256_q6",
        "env": {
            **cap16_clean_base(model_dim=640, embed_dim=256, io_width=3, loop_width=5, repeats=2),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "tests whether i3/l5/r2 loop diversity helps without the d768/QK/LQER confounds",
    },
]


CAP16_ANCHOR_NOSPEED_SPEND_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "nospeed_i3l3r3_d640e256_q6_control",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=256),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "fresh export-honest anchor control without fused-QKV/casted-linear speed shortcuts",
    },
    {
        "name": "nospeed_i3l3r3_d640e384_q6",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=384),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "spend bytes on factored embedding rank while keeping the old export-honest training path",
    },
    {
        "name": "nospeed_i3l3r3_d640e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "token-interface cap-fill row: larger factored embedding and q8 train/export, no speed confound",
    },
    {
        "name": "nospeed_i3l3r3_d704e320_q6",
        "env": {
            **cap16_nospeed_base(model_dim=704, embed_dim=320),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "moderate residual-width spend between d640 and the previously weak d768 rows",
    },
    {
        "name": "nospeed_i3l3r3_d704e384_q8",
        "env": {
            **cap16_nospeed_base(model_dim=704, embed_dim=384),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "closer-to-cap quality row: d704 width, richer embedding rank, and q8 train/export",
    },
    {
        "name": "nospeed_i3l3r3_d640e256_q6_vocabmoe_k16r4",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=256),
            **vocab_moe_env(experts=16, rank=4, mode="hybrid", layers="input,loop_first"),
        },
        "notes": "spend bytes inside the winning VocabMoE placement by doubling expert rank",
    },
    {
        "name": "nospeed_i3l3r3_d640e256_q6_vocabmoe_k32r2",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=256),
            **vocab_moe_env(experts=32, rank=2, mode="hybrid", layers="input,loop_first"),
        },
        "notes": "spend bytes on more token/expert buckets while preserving rank-2 experts",
    },
    {
        "name": "nospeed_i3l5r2_d640e256_q6",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=256, io_width=3, loop_width=5, repeats=2),
            **BEST_CLEAN_VOCABMOE,
        },
        "notes": "tests whether i3/l5/r2 loop diversity helps without d768, QK, LQER, or speed confounds",
    },
]


CAP16_Q8_CAPFILL_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "capfill_i3l3r3_d640e640_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=640),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "extends the d640/e512/q8 win by spending more bytes on the token interface",
    },
    {
        "name": "capfill_i3l3r3_d640e768_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=768),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "more aggressive token-interface spend, expected to move closer to the 16MB cap",
    },
    {
        "name": "capfill_i3l3r3_d640e1024_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=1024),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "near-cap/over-cap probe: tests whether very large q8 factored embeddings keep buying BPB",
    },
    {
        "name": "capfill_i3l3r3_d704e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=704, embed_dim=512),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "spends bytes on moderate residual width while keeping the proven e512/q8 token interface",
    },
    {
        "name": "capfill_i3l3r3_d704e640_q8",
        "env": {
            **cap16_nospeed_base(model_dim=704, embed_dim=640),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "higher cap-fill row: d704 width plus larger q8 token interface",
    },
    {
        "name": "capfill_i3l3r3_d768e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=768, embed_dim=512),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "checks whether d768 width becomes worthwhile once q8/e512 fixes the token interface",
    },
    {
        "name": "capfill_i3l3r3_d640e512_q8_vocabmoe_k16r4",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512),
            **vocab_moe_env(experts=16, rank=4, mode="hybrid", layers="input,loop_first", train_quant_bits=8),
            **q8_train_export_env(),
        },
        "notes": "spends remaining bytes inside the winning VocabMoE placement with q8 expert rank",
    },
    {
        "name": "capfill_i3l3r3_d640e512_q8_vocabmoe_k32r2",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512),
            **vocab_moe_env(experts=32, rank=2, mode="hybrid", layers="input,loop_first", train_quant_bits=8),
            **q8_train_export_env(),
        },
        "notes": "spends remaining bytes on more q8 token/expert buckets at the proven rank",
    },
    {
        "name": "capfill_i3l5r2_d640e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=2),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "combines the q8/e512 token-interface win with the more-unique-loop i3/l5/r2 shape",
    },
]


CAP16_PRIORITY_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "priority_capfill_i3l3r3_d640e640_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=640),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": "direct cap-fill continuation of the d640/e512/q8 win; isolates token-interface spend",
    },
    {
        "name": "priority_vocabmoe_i3l3r3_d640e512_q8_k16r4",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512),
            **vocab_moe_env(experts=16, rank=4, mode="hybrid", layers="input,loop_first", train_quant_bits=8),
            **q8_train_export_env(),
        },
        "notes": "spends bytes on richer VocabMoE experts on the proven q8/e512 spine",
    },
    {
        "name": "priority_loop_i3l5r2_d640e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=2),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": (
            "tests whether the q8/e512 spine benefits from more unique loop "
            "blocks; repeat count differs from i3/l3/r3 and needs a follow-up"
        ),
    },
    {
        "name": "priority_dual_i3l3r3_d640e512_q8_left256_r16",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **dual_stream_env(left_dim=256, rank=16, sites="input,loop_first,pre_output", scale=0.02),
        },
        "notes": "trained dual-stream advisor bridge on the proven q8/e512 spine; the actual dual-stream canary",
    },
]


CAP16_LOOP_FOLLOWUP_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "loopfollow_i3l5r5_d640e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
        },
        "notes": (
            "tests the user's sharper read from the priority win: keep the "
            "i3/l5 unique-loop shape and increase repeats to isolate whether "
            "more recurrence on the wider unique core buys quality"
        ),
    },
]


def layer_width_schedule_env(widths: tuple[int, ...]) -> dict[str, str]:
    return {
        "LAYER_WIDTH_SCHEDULE": ",".join(str(width) for width in widths),
        # Width ladders currently cannot share the depth-LoRA/basis/VE paths.
        "DEPTH_LORA_RANK": "0",
        "BASIS_XSA_ENABLED": "0",
        "VE_ENABLED": "0",
    }


CAP16_STRUCTURE_FOLLOWUP_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "structure_dual_i3l5r5_d640e512_q8_left256_r16_loopx",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **dual_stream_env(
                left_dim=256,
                rank=16,
                sites="input,loop_first,loop_exit,pre_output",
                scale=0.02,
            ),
        },
        "notes": (
            "ports the trained left/right advisor bridge onto the current "
            "best i3/l5/r5 spine and adds a loop-exit bridge site"
        ),
    },
    {
        "name": "structure_hourglass_i3l5r5_d640e512_q8_w400-480-560-640",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **layer_width_schedule_env((400, 480, 560, 640, 640, 640, 640, 640)),
        },
        "notes": (
            "hourglass-width probe: narrower token-facing IO blocks, full-width "
            "recurrent core, same q8/e512 i3/l5/r5 spine"
        ),
    },
]


CAP16_STRUCTURE_COMBO_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "structure_combo_i3l5r5_d640e512_q16q8q4io_q8core_w400-480-560-640_dual",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_core_io_ladder_env((16, 8, 4)),
            **layer_width_schedule_env((400, 480, 560, 640, 640, 640, 640, 640)),
            **dual_stream_env(
                left_dim=256,
                rank=16,
                sites="input,loop_first,loop_exit,pre_output",
                scale=0.02,
            ),
        },
        "notes": (
            "single high-signal interaction row: train-time IO-tail precision "
            "ladder q16/q8/q4, hourglass block widths, q8 recurrent core, and "
            "the trained dual-stream advisor on the current best i3/l5/r5 spine"
        ),
    },
]


CAP16_FRONTIER_CAPFILL_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "frontier_capfill_i3l5r5_d640e640_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=640, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "directly spends the remaining 16MB headroom on factored tied "
            "embedding rank while preserving the current best full-width "
            "single-stream i3/l5/r5 q8 spine"
        ),
    },
    {
        "name": "frontier_lqer_i3l5r5_d640e512_q8_r12t24_embed",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **cap16_lqer_env(12, 24),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "export-gap repair probe: stronger asymmetric LQER and embed_proj "
            "coverage on the exact best q8/e512 i3/l5/r5 spine"
        ),
    },
    {
        "name": "frontier_polarminlr10_i3l5r5_d640e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "ports the current public transformer schedule polish: Polar "
            "Express Newton-Schulz plus a 10% warmdown LR floor"
        ),
    },
    {
        "name": "frontier_qk525_parres4_i3l5r5_d640e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **parallel_residual_env(last_n=4),
            "QK_GAIN_INIT": "5.25",
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "ports the accepted-leader routing/gain pair: wider parallel "
            "residual tail plus QK gain 5.25 on the current best HRC/VocabMoE spine"
        ),
    },
]


CAP16_FRONTIER_FOLLOWUP_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "frontier_follow_i3l5r5_d640e640_q8_polarminlr10",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=640, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "combines the two winning levers from the completed cap-fill "
            "matrix: e640 token-interface spend and Polar/MIN_LR schedule"
        ),
    },
    {
        "name": "frontier_follow_i3l5r5_d640e768_q8_polarminlr10",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=768, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "tests whether the e640 cap-spend win keeps scaling closer to 16MB "
            "when paired with the now-winning Polar/MIN_LR schedule"
        ),
    },
    {
        "name": "frontier_follow_i3l5r5_d640e640_q8_polarminlr10_lqer12t24_embed",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=640, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **cap16_lqer_env(12, 24),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "combines the winning e640+Polar idea with the small positive LQER "
            "repair row; useful because artifact headroom remains"
        ),
    },
    {
        "name": "frontier_follow_i3l5r5_d640e640_q8_polarminlr05",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=640, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.05),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "schedule sensitivity row: tests whether a gentler 5% LR floor "
            "beats the public-inspired 10% floor on the e640 spine"
        ),
    },
]


CAP16_BREAKOUT_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "breakout_dense11_d512e512_q8_polar_vocabmoe_bigram",
        "env": {
            **cap16_baseline_base(model_dim=512, embed_dim=512, num_layers=11, num_kv_heads=1),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **bigram_hash_env(vocab_size=10240, dim=32),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "dense SP8192-like transformer branch: tests whether the HRC "
            "recurrence itself is the quality ceiling, with BigramHash as the "
            "cheap public-side-feature canary"
        ),
    },
    {
        "name": "breakout_dense13_d512e384_q8_polar_vocabmoe",
        "env": {
            **cap16_baseline_base(model_dim=512, embed_dim=384, num_layers=13, num_kv_heads=1),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "physical-depth control: spends bytes on unique transformer blocks "
            "instead of recurrent reuse or larger embeddings"
        ),
    },
    {
        "name": "breakout_dense11_d640e384_q6_fullkv_memattn_vocabmoe_bigram",
        "env": {
            **cap16_baseline_base(model_dim=640, embed_dim=384, num_layers=11, num_kv_heads=10),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(min_lr=0.10),
            **bigram_hash_env(vocab_size=10240, dim=32),
            "SDP_BACKEND": "mem_efficient",
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "quality/systems branch: wider dense model with full K/V heads so "
            "the 2060 can use memory-efficient attention instead of the GQA "
            "math path"
        ),
    },
    {
        "name": "breakout_hrc_i3l5r5_d640e512_q8_polar_ttt_control24",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **legal_ttt_env(lr=0.005, updates=24),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "legal score-first TTT canary on the current best HRC/VocabMoE "
            "spine; if this fails, polishing eval-only adaptation is not the "
            "right next local spend"
        ),
    },
    {
        "name": "breakout_hrc_i3l7r4_d640e512_q8_polar_bigram",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=7, repeats=4),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **bigram_hash_env(vocab_size=10240, dim=32),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "tests the user's unique-loop-block signal while also porting the "
            "cheap BigramHash lever from the public transformer lane"
        ),
    },
    {
        "name": "breakout_dual_i3l5r2_d768e320_left256_q6_polar_vocabmoe_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320, io_width=3, loop_width=5, repeats=2),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
            **dual_stream_env(left_dim=256, rank=16),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "one true dual-stream probe, trained end-to-end rather than eval "
            "council; included as a broad architecture branch, not as a whole "
            "night of variants"
        ),
    },
]


CAP16_ART_SHOWCASE_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "art_prime_skip_spike_i3p5s1r2_d640e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=2),
            **prime_superloop_route(shell_width=3, prime_width=5, laps_per_skip=2, skip_ids=(1,)),
            **spike_vocab_moe_env(
                experts=16,
                rank=2,
                mode="spike_hybrid",
                layers="input,loop_every3",
                spike_top_k=2,
                train_quant_bits=8,
            ),
            **q8_train_export_env(),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "art lane: prime-width recurrent core with a nontrivial skip walk, "
            "and hard token self-election without the extra dual-stream bridge "
            "that wedged the first smoke after step 10"
        ),
    },
    {
        "name": "art_pal_ladder_hourglass_dual_spike_i3l5r1_d640e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=1),
            **palindrome_loop_route(io_width=3, loop_width=5, loop_repeats=1),
            **q8_core_io_ladder_env((16, 8, 4)),
            **layer_width_schedule_env((400, 440, 520, 640, 640, 640, 640, 640)),
            **spike_vocab_moe_env(
                experts=16,
                rank=2,
                mode="spike_hybrid",
                layers="input,loop_first",
                spike_top_k=2,
                train_quant_bits=8,
            ),
            **dual_stream_env(left_dim=256, rank=16, sites="input,loop_first,loop_exit,pre_output"),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "art lane: mirrored palindrome route plus q16/q8/q4 IO-tail ladder, "
            "hourglass block widths, spike VocabMoE, and dual-stream bridges"
        ),
    },
    {
        "name": "art_rlm_council_hybrid_i3l5r5_d640e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **rlm_memory_env(inject="input_loop_first", decay=0.95, scale=0.01),
            **council_env(mode="base_mirror_hybrid", mirror="householder", offsets="0,0,-1"),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "art lane: legal recursive prefix memory plus a three-peer mirrored "
            "council distribution on the strongest HRC/VocabMoE spine"
        ),
    },
    {
        "name": "art_loopall_self_electing_rlm_i3l5r5_d640e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **q8_train_export_env(),
            **spike_vocab_moe_env(
                experts=32,
                rank=1,
                mode="spike_hybrid",
                layers="input,loop",
                spike_top_k=2,
                train_quant_bits=8,
            ),
            **rlm_memory_env(inject="loop_first", decay=0.90, scale=0.02),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "art lane: every recurrent-loop position gets sparse token expert "
            "self-election, with a causal RLM-lite memory injected at loop entry"
        ),
    },
    {
        "name": "art_firstattn_mlp_core_i3l7r3_d640e512_q8",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=7, repeats=3),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            "HRC_MLP_ONLY_BLOCKS": "4,5,6,7,8,9",
            **rlm_memory_env(inject="input", decay=0.90, scale=0.02),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "art lane: one attention-capable block at recurrent-core entry, "
            "then a mostly MLP recurrent semantic engine with legal memory"
        ),
    },
    {
        "name": "art_dual_hourglass_q2core_i4l5r3_d768e512",
        "env": {
            **cap16_nospeed_base(model_dim=768, embed_dim=512, io_width=4, loop_width=5, repeats=3),
            **cap16_taper_env(io_width=4, loop_width=5, io_bits=(16, 8, 6, 4), core_bits=2),
            **layer_width_schedule_env((384, 432, 504, 576, 648, 696, 768, 768, 768)),
            **spike_vocab_moe_env(
                experts=16,
                rank=2,
                mode="spike_hybrid",
                layers="input,loop_first",
                spike_top_k=2,
                train_quant_bits=6,
            ),
            **dual_stream_env(left_dim=320, rank=24, sites="input,loop_first,loop_exit,pre_output"),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "art lane: the full data-density sketch: higher-precision narrow "
            "IO blocks, wider low-precision q2 recurrent core, spike VocabMoE, "
            "and trained dual-stream advisor bridges"
        ),
    },
]


CAP16_H100_PREFLIGHT_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "preflight_h100_anchor_i3l5r5_d640e512_q8_polar",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "local H100-spend gate: exact clean anchor from the paid matrix, "
            "run locally first to recheck export behavior and current code"
        ),
    },
    {
        "name": "preflight_h100_e640_i3l5r5_d640e640_q8_polar",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=640, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "local H100-spend gate: spends more bytes on the token interface "
            "without changing the proven i3/l5/r5 route"
        ),
    },
    {
        "name": "preflight_h100_i3l7r4_d640e512_q8_polar_bigram",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=7, repeats=4),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **bigram_hash_env(vocab_size=10240, dim=32),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "local H100-spend gate: more unique recurrent blocks plus the "
            "cheap BigramHash side-feature lever"
        ),
    },
    {
        "name": "preflight_h100_dual_i3l5r5_d640e512_q8_polar_left256",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **BEST_CLEAN_VOCABMOE,
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **dual_stream_env(left_dim=256, rank=16, sites="input,loop_first,loop_exit,pre_output"),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "local H100-spend gate: trained left/right advisor bridge on the "
            "current best route, not eval-only council"
        ),
    },
    {
        "name": "preflight_h100_spike_loopall_i3l5r5_d640e512_q8_polar",
        "env": {
            **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
            **q8_train_export_env(),
            **leaderboard_schedule_env(min_lr=0.10),
            **spike_vocab_moe_env(
                experts=32,
                rank=1,
                mode="spike_hybrid",
                layers="input,loop",
                spike_top_k=2,
                train_quant_bits=8,
            ),
            **FINAL_ONLY_EVAL_ENV,
        },
        "notes": (
            "local H100-spend gate: exact spike/self-election paid-row shape, "
            "without the RLM bundle that crashed locally"
        ),
    },
]


CAP16_LEADERBOARD_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "leader_i3l3r3_d768e320_q6all_polar_minlr_vocabmoe_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
        },
        "notes": "leaderboard-blend anchor: HRC/VocabMoE spine plus Polar Express Muon and a small MIN_LR floor",
    },
    {
        "name": "leader_i3l3r3_d768e320_q6all_polar_minlr_vocabmoe_qk550_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
            "QK_GAIN_INIT": "5.5",
        },
        "notes": "tests whether the public QK-gain push past 5.25 transfers to the HRC/VocabMoE spine",
    },
    {
        "name": "leader_i3l3r3_d768e320_q6all_sparsegate_polar_minlr_vocabmoe_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
            **sparse_gate_env(width=24),
        },
        "notes": "replaces the attention-output gate with the public sparse attention gate on the same spine",
    },
    {
        "name": "leader_i3l3r3_d768e320_q6all_parres3_polar_minlr_vocabmoe_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
            **parallel_residual_env(last_n=3),
        },
        "notes": "public parallel-residual analogue on the last three HRC tail layers",
    },
    {
        "name": "leader_i3l3r2rev_d768e320_q6all_polar_minlr_vocabmoe_lqer16t32",
        "env": {
            **cap16_palindrome_base(model_dim=768, embed_dim=320, io_width=3, loop_width=3, repeats=2),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
        },
        "notes": "cycle-rev route at the same virtual depth as i3/l3/r3 cycle; inspired by parameter-sharing studies",
    },
    {
        "name": "leader_i3l3r3_d768e320_q6all_loopidx_polar_minlr_vocabmoe_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
            "HRC_LOOP_INDEX_ENABLED": "1",
            "HRC_LOOP_INDEX_DIM": "64",
            "HRC_LOOP_INDEX_SCALE_INIT": "0.02",
        },
        "notes": "adds Universal-Transformer-style recurrence step signal to the 16MB HRC/VocabMoE spine",
    },
    {
        "name": "leader_i3l3r3_d768e320_q6all_depthlora4_polar_minlr_vocabmoe_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
            "DEPTH_LORA_RANK": "4",
            "HRC_DEPTH_ADAPTER_TIE_MODE": "none",
        },
        "notes": "light per-virtual-depth Q/V LoRA relaxation so shared blocks can specialize by pass",
    },
    {
        "name": "leader_i3l3r3_d768e320_q6all_wd04_polar_minlr_vocabmoe_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
            "MUON_WEIGHT_DECAY": "0.04",
            "MUON_WEIGHT_DECAY_MODE": "huber",
        },
        "notes": "moderate compressibility-aware Muon WD; lower than the public 0.09-ish frontier for local stability",
    },
    {
        "name": "leader_i3l5r2_d768e320_q6all_polar_minlr_vocabmoe_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320, io_width=3, loop_width=5, repeats=2),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
        },
        "notes": "leaderboard schedule on the more-unique-loop HRC row that local sub-4 sweeps favored",
    },
    {
        "name": "leader_i3l5r1rev_d768e320_q6all_polar_minlr_vocabmoe_qk525_lqer16t32",
        "env": {
            **cap16_palindrome_base(model_dim=768, embed_dim=320, io_width=3, loop_width=5, repeats=1),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
        },
        "notes": "cycle-rev unique-loop row with roughly the same virtual depth as i3/l5/r2 cycle",
    },
    {
        "name": "leader_i3l5r2_d768e320_q8q6q6_q4core_sparsegate_polar_minlr_vocabmoe_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320, io_width=3, loop_width=5, repeats=2),
            **cap16_taper_env(io_width=3, loop_width=5, io_bits=(8, 6, 6), core_bits=4),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
            **sparse_gate_env(width=24),
        },
        "notes": "strong combined row: precision taper, sparse attention gate, Polar/MIN_LR, and VocabMoE",
    },
    {
        "name": "leader_i3l3r3_d768e320_q6all_bigram_polar_minlr_vocabmoe_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **leaderboard_schedule_env(),
            **bigram_hash_env(vocab_size=10240, dim=32),
        },
        "notes": "tests the older but repeatedly useful BigramHash side channel alongside VocabMoE",
    },
    {
        "name": "leader_i3l3r3_d768e320_q6all_ttt_control24_vocabmoe_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **legal_ttt_env(lr=0.005, updates=24),
        },
        "notes": "local legal score-first TTT canary; compare final_quant_ttt_val_bpb against final_export_val_bpb",
    },
]


CAP16_DUAL_STREAM_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "dual_i3l3r3_d768e320_left256_q6all_vocabmoe_qk525_lqer12t24",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **BEST_CLEAN_VOCABMOE,
            **dual_stream_env(left_dim=256, rank=16),
        },
        "notes": "trained left/right advisor bridge on the d768/e320 VocabMoE spine; q6 throughout",
    },
    {
        "name": "dual_i3l3r3_d768e320_left320_q6all_vocabmoe_qk525_lqer12t24",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320),
            **BEST_CLEAN_VOCABMOE,
            **dual_stream_env(left_dim=320, rank=16),
        },
        "notes": "same spine, more token-facing left lane; tests whether surface precision needs more width",
    },
    {
        "name": "dual_i3l5r2_d768e320_left256_q6all_vocabmoe_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=768, embed_dim=320, io_width=3, loop_width=5, repeats=2),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **dual_stream_env(left_dim=256, rank=16),
        },
        "notes": "dual-stream advisor on the more-unique-loop mainline; no q4 taper yet",
    },
    {
        "name": "dual_i3l5r2_d896e384_left320_q8q6q6_q4core_vocabmoe_qk525_lqer16t32",
        "env": {
            **cap16_speed_base(model_dim=896, embed_dim=384, io_width=3, loop_width=5, repeats=2),
            **cap16_taper_env(io_width=3, loop_width=5, io_bits=(8, 6, 6), core_bits=4),
            **cap16_lqer_env(16, 32),
            **BEST_CLEAN_VOCABMOE,
            **dual_stream_env(left_dim=320, rank=24),
        },
        "notes": "quality-first dual-stream row: larger width, q4 recurrent taper, stronger bridge rank",
    },
]


CANDIDATE_GROUPS: dict[str, list[dict[str, Any]]] = {
    "default": CANDIDATES,
    "vocabmoe": CANDIDATES,
    "vocabmoe_spike": SPIKE_CANDIDATES,
    "council_rlm": COUNCIL_RLM_CANDIDATES,
    "cap16_speed": CAP16_SPEED_CANDIDATES,
    "cap16_mainline": CAP16_MAINLINE_CANDIDATES,
    "cap16_anchor_spend": CAP16_ANCHOR_SPEND_CANDIDATES,
    "cap16_anchor_clean_spend": CAP16_ANCHOR_CLEAN_SPEND_CANDIDATES,
    "cap16_anchor_nospeed_spend": CAP16_ANCHOR_NOSPEED_SPEND_CANDIDATES,
    "cap16_q8_capfill": CAP16_Q8_CAPFILL_CANDIDATES,
    "cap16_priority": CAP16_PRIORITY_CANDIDATES,
    "cap16_loop_followup": CAP16_LOOP_FOLLOWUP_CANDIDATES,
    "cap16_structure_followup": CAP16_STRUCTURE_FOLLOWUP_CANDIDATES,
    "cap16_structure_combo": CAP16_STRUCTURE_COMBO_CANDIDATES,
    "cap16_frontier_capfill": CAP16_FRONTIER_CAPFILL_CANDIDATES,
    "cap16_frontier_followup": CAP16_FRONTIER_FOLLOWUP_CANDIDATES,
    "cap16_breakout": CAP16_BREAKOUT_CANDIDATES,
    "cap16_art_showcase": CAP16_ART_SHOWCASE_CANDIDATES,
    "cap16_h100_preflight": CAP16_H100_PREFLIGHT_CANDIDATES,
    "cap16_leaderboard": CAP16_LEADERBOARD_CANDIDATES,
    "cap16_dual_stream": CAP16_DUAL_STREAM_CANDIDATES,
    "all": (
        CANDIDATES
        + SPIKE_CANDIDATES
        + COUNCIL_RLM_CANDIDATES
        + CAP16_SPEED_CANDIDATES
        + CAP16_MAINLINE_CANDIDATES
        + CAP16_ANCHOR_SPEND_CANDIDATES
        + CAP16_ANCHOR_CLEAN_SPEND_CANDIDATES
        + CAP16_ANCHOR_NOSPEED_SPEND_CANDIDATES
        + CAP16_Q8_CAPFILL_CANDIDATES
        + CAP16_PRIORITY_CANDIDATES
        + CAP16_LOOP_FOLLOWUP_CANDIDATES
        + CAP16_STRUCTURE_FOLLOWUP_CANDIDATES
        + CAP16_STRUCTURE_COMBO_CANDIDATES
        + CAP16_FRONTIER_CAPFILL_CANDIDATES
        + CAP16_FRONTIER_FOLLOWUP_CANDIDATES
        + CAP16_BREAKOUT_CANDIDATES
        + CAP16_ART_SHOWCASE_CANDIDATES
        + CAP16_H100_PREFLIGHT_CANDIDATES
        + CAP16_LEADERBOARD_CANDIDATES
        + CAP16_DUAL_STREAM_CANDIDATES
    ),
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
                f"prior_std={env.get('VOCAB_MOE_PRIOR_INIT_STD', '0')} "
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
        extras: list[str] = []
        if env.get("HRC_COUNCIL_MODE", "none") != "none":
            extras.append(
                "council="
                f"{env.get('HRC_COUNCIL_MODE')} mirror={env.get('HRC_MIRROR_MODE', 'default')} "
                f"offsets={env.get('HRC_COUNCIL_DEPTH_OFFSETS', 'default')} "
                f"hard={env.get('HRC_COUNCIL_HARD_GATE', '0')}"
            )
        if env.get("HRC_DYNAMIC_COUNCIL_ENABLED") == "1":
            extras.append(
                "dynamic="
                f"t{env.get('HRC_DYNAMIC_COUNCIL_THRESHOLD')} "
                f"min_gate={env.get('HRC_DYNAMIC_COUNCIL_MIN_GATE')}"
            )
        if env.get("RLM_MEMORY_ENABLED") == "1":
            extras.append(
                "rlm="
                f"{env.get('RLM_MEMORY_INJECT')} decay={env.get('RLM_MEMORY_DECAY')} "
                f"scale={env.get('RLM_MEMORY_SCALE_INIT')}"
            )
        if env.get("DUAL_STREAM_ENABLED") == "1":
            extras.append(
                "dual="
                f"left={env.get('DUAL_STREAM_LEFT_DIM')} rank={env.get('DUAL_STREAM_RANK')} "
                f"sites={env.get('DUAL_STREAM_SITES')} scale={env.get('DUAL_STREAM_SCALE_INIT')}"
            )
        if env.get("QUANT_BITS_OVERRIDES"):
            extras.append(f"bits={env.get('QUANT_BITS_OVERRIDES')}")
        if env.get("LAYER_WIDTH_SCHEDULE"):
            extras.append(f"widths={env.get('LAYER_WIDTH_SCHEDULE')}")
        if extras:
            moe = f"{moe}; " + "; ".join(extras)
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


def run_command_live(
    args: list[str],
    env: dict[str, str],
    timeout: int,
    live_path: Path,
    label: str,
) -> subprocess.CompletedProcess[str]:
    """Run a candidate while teeing progress to disk and queue stdout."""
    sentinel = object()
    output_queue: queue.Queue[str | object] = queue.Queue()
    lines: list[str] = []

    def reader(pipe: Any) -> None:
        try:
            for line in pipe:
                output_queue.put(line)
        finally:
            output_queue.put(sentinel)

    live_path.parent.mkdir(parents=True, exist_ok=True)
    with live_path.open("w", encoding="utf-8", errors="replace") as live_file:
        proc = subprocess.Popen(
            args,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if proc.stdout is None:
            raise RuntimeError("subprocess stdout pipe was not created")
        thread = threading.Thread(target=reader, args=(proc.stdout,), daemon=True)
        thread.start()
        started = time.perf_counter()
        timed_out = False
        reader_done = False
        while True:
            try:
                item = output_queue.get(timeout=0.5)
            except queue.Empty:
                if timeout > 0 and time.perf_counter() - started > timeout:
                    timed_out = True
                    proc.kill()
                    break
                if proc.poll() is not None and reader_done:
                    break
                continue

            if item is sentinel:
                reader_done = True
            else:
                line = str(item)
                lines.append(line)
                live_file.write(line)
                live_file.flush()
                text = line.strip()
                if text.startswith("step:") or text.startswith("final_") or text.startswith("TIMEOUT"):
                    print(f"{label}: {text}", flush=True)

            if proc.poll() is not None and reader_done:
                break

        if timed_out:
            timeout_line = f"\nTIMEOUT after {timeout}s\n"
            lines.append(timeout_line)
            live_file.write(timeout_line)
            live_file.flush()
            print(f"{label}: TIMEOUT after {timeout}s", flush=True)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            return subprocess.CompletedProcess(args, 124, "".join(lines))

        return subprocess.CompletedProcess(args, proc.wait(), "".join(lines))


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
                "TRAIN_LOG_EVERY": "250",
                "PYTHONUNBUFFERED": "1",
            }
        )
        started = time.perf_counter()
        raw_path = out_dir / f"train_{name}.txt"
        proc = run_command_live([str(PYTHON), "-u", str(TRAINER)], env, timeout=timeout, live_path=raw_path, label=name)
        stdout = merged_train_output(proc.stdout, run_id)
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
