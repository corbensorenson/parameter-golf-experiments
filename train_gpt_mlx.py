#!/usr/bin/env python3
"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""
from __future__ import annotations

import glob
import json
import lzma
import math
import os
import pickle
import sys
import time
import traceback
import uuid
import zlib
from concurrent.futures import Future, ThreadPoolExecutor
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sentencepiece as spm

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from mlx_frontier_utils import BigramHashEmbedding, accumulate_flat_grads, apply_xsa, build_hemisphere_layer_flags, build_layer_mlp_mults, build_parallel_residual_flags, build_tail_flags, build_xsa_gate_flags, build_xsa_layer_flags, hemisphere_transform, leaky_relu, microbatch_plan, mirror_weight, parse_csv_int_list, round_hidden_dim, silu

# ==============================================================================
# SHARD FORMAT + COMPUTE DTYPE
# ==============================================================================

COMPUTE_DTYPE = mx.bfloat16

# ==============================================================================
# HYPERPARAMETERS
# ==============================================================================
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap
class Hyperparameters:
    # Data / tokenizer.
    data_path: str = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    tokenizer_path: str = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id: str = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed: int = int(os.environ.get("SEED", 1337))

    # Training loop. These defaults now mirror train_gpt.py on a single process.
    iterations: int = int(os.environ.get("ITERATIONS", 20_000))
    val_loss_every: int = int(os.environ.get("VAL_LOSS_EVERY", 0))
    # Validation always uses the full fineweb_val split.
    val_batch_size: int = int(os.environ.get("VAL_BATCH_SIZE", 524_288))
    # Optional local-proxy eval cap. Keep 0 to evaluate the full val split.
    val_tokens_limit: int = int(os.environ.get("VAL_TOKENS_LIMIT", 0))
    train_log_every: int = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    train_batch_tokens: int = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    grad_accum_steps: int = int(os.environ.get("GRAD_ACCUM_STEPS", 8))
    train_seq_len: int = int(os.environ.get("TRAIN_SEQ_LEN", os.environ.get("TRAIN_MAX_SEQ_LEN", 1024)))
    # Chunk each logical MLX microbatch into smaller sub-batches to reduce peak
    # memory pressure without changing the effective optimizer batch.
    mlx_max_microbatch_tokens: int = int(os.environ.get("MLX_MAX_MICROBATCH_TOKENS", 8_192))
    # Force MLX to materialize the graph after every sub-batch, preventing lazy
    # graph buildup across accumulation steps. Keeps peak memory low on 16GB machines.
    # Disable on 32GB+ unified memory for better throughput (MLX_EAGER_EVAL=0).
    mlx_eager_eval: bool = bool(int(os.environ.get("MLX_EAGER_EVAL", "1")))
    # When eager materialization is enabled, dispatch it asynchronously so host-side
    # work like the next batch fetch can overlap with MLX execution.
    mlx_async_eval: bool = bool(int(os.environ.get("MLX_ASYNC_EVAL", "0")))
    mlx_memory_telemetry: bool = bool(int(os.environ.get("MLX_MEMORY_TELEMETRY", "0")))
    mlx_memory_limit_gb: float = float(os.environ.get("MLX_MEMORY_LIMIT_GB", "0"))
    mlx_cache_limit_gb: float = float(os.environ.get("MLX_CACHE_LIMIT_GB", "0"))
    mlx_wired_limit_gb: float = float(os.environ.get("MLX_WIRED_LIMIT_GB", "0"))
    disable_compile: bool = bool(int(os.environ.get("DISABLE_COMPILE", "0")))
    warmup_steps: int = int(os.environ.get("WARMUP_STEPS", 20))
    warmdown_iters: int = int(os.environ.get("WARMDOWN_ITERS", 1200))
    max_wallclock_seconds: float = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    scout_mode: str = os.environ.get("SCOUT_MODE", "none").strip().lower()
    skip_post_average_eval: bool = bool(int(os.environ.get("SKIP_POST_AVERAGE_EVAL", "0")))
    skip_probe_metrics: bool = bool(int(os.environ.get("SKIP_PROBE_METRICS", "0")))
    skip_final_artifacts: bool = bool(int(os.environ.get("SKIP_FINAL_ARTIFACTS", "0")))
    skip_final_quant_eval: bool = bool(int(os.environ.get("SKIP_FINAL_QUANT_EVAL", "0")))
    trainable_mode: str = os.environ.get("TRAINABLE_MODE", "all").strip().lower()
    trainable_include_patterns: str = os.environ.get("TRAINABLE_INCLUDE_PATTERNS", "").strip()
    trainable_exclude_patterns: str = os.environ.get("TRAINABLE_EXCLUDE_PATTERNS", "").strip()
    train_shard_prefetch: int = int(os.environ.get("TRAIN_SHARD_PREFETCH", "1"))
    host_token_dtype: str = os.environ.get("HOST_TOKEN_DTYPE", "uint16").strip().lower()

    # Model (defaults match the current baseline setup).
    model_family: str = os.environ.get("MODEL_FAMILY", "baseline").strip().lower()
    vocab_size: int = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers: int = int(os.environ.get("NUM_LAYERS", 9))
    num_unique_blocks: int = int(os.environ.get("NUM_UNIQUE_BLOCKS", 3))
    effective_depth: int = int(os.environ.get("EFFECTIVE_DEPTH", 9))
    model_dim: int = int(os.environ.get("MODEL_DIM", 512))
    num_heads: int = int(os.environ.get("NUM_HEADS", 8))
    num_kv_heads: int = int(os.environ.get("NUM_KV_HEADS", 4))
    mlp_mult: int = int(os.environ.get("MLP_MULT", 2))
    tie_embeddings: bool = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    tied_embed_init_std: float = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    logit_chunk_tokens: int = int(os.environ.get("LOGIT_CHUNK_TOKENS", 0))
    bpb_weighted_loss: bool = bool(int(os.environ.get("BPB_WEIGHTED_LOSS", "0")))
    logit_softcap: float = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    rope_base: float = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims: int = int(os.environ.get("ROPE_DIMS", 0))
    qk_gain_init: float = float(os.environ.get("QK_GAIN_INIT", 1.5))
    rbf_attn_last_n: int = int(os.environ.get("RBF_ATTN_LAST_N", 0))
    rbf_attn_qk_norm: bool = bool(int(os.environ.get("RBF_ATTN_QK_NORM", "0")))
    rbf_attn_disable_rope: bool = bool(int(os.environ.get("RBF_ATTN_DISABLE_ROPE", "1")))
    ln_scale: bool = bool(int(os.environ.get("LN_SCALE", "0")))
    smear_enabled: bool = bool(int(os.environ.get("SMEAR_ENABLED", "0")))
    activation_kind: str = os.environ.get("ACTIVATION_KIND", "relu2").strip().lower()
    activation_negative_slope: float = float(os.environ.get("ACTIVATION_NEGATIVE_SLOPE", 0.5))
    qsparse_enabled: bool = bool(int(os.environ.get("QSPARSE_ENABLED", "0")))
    qsparse_topk: int = int(os.environ.get("QSPARSE_TOPK", 0))
    qsparse_last_n: int = int(os.environ.get("QSPARSE_LAST_N", 0))
    qsparse_gate_init: float = float(os.environ.get("QSPARSE_GATE_INIT", -1.5))
    monarch_enabled: bool = bool(int(os.environ.get("MONARCH_ENABLED", "0")))
    monarch_last_n: int = int(os.environ.get("MONARCH_LAST_N", 0))
    monarch_preferred_blocks: int = int(os.environ.get("MONARCH_PREFERRED_BLOCKS", 0))
    monarch_gate_init: float = float(os.environ.get("MONARCH_GATE_INIT", -1.5))
    bigram_vocab_size: int = int(os.environ.get("BIGRAM_VOCAB_SIZE", 0))
    bigram_dim: int = int(os.environ.get("BIGRAM_DIM", 128))
    coil_mode: str = os.environ.get("COIL_MODE", "none").strip().lower()
    coil_prime_window: int = int(os.environ.get("COIL_PRIME_WINDOW", 17))
    coil_tap_count: int = int(os.environ.get("COIL_TAP_COUNT", 6))
    coil_sparse_topk: int = int(os.environ.get("COIL_SPARSE_TOPK", 0))
    coil_pctm_mode: str = os.environ.get("COIL_PCTM_MODE", "weighted").strip().lower()
    coil_pctm_basis_count: int = int(os.environ.get("COIL_PCTM_BASIS_COUNT", 4))
    coil_anti_enabled: bool = bool(int(os.environ.get("COIL_ANTI_ENABLED", "0")))
    coil_probe_mode: str = os.environ.get("COIL_PROBE_MODE", "none").strip().lower()
    coil_probe_bias_init: float = float(os.environ.get("COIL_PROBE_BIAS_INIT", -2.0))
    coil_probe_scale_init: float = float(os.environ.get("COIL_PROBE_SCALE_INIT", 4.0))
    coil_act_enabled: bool = bool(int(os.environ.get("COIL_ACT_ENABLED", "0")))
    coil_act_pair_count: int = int(os.environ.get("COIL_ACT_PAIR_COUNT", 2))
    coil_act_scale_init: float = float(os.environ.get("COIL_ACT_SCALE_INIT", 0.03))
    coil_residual_gate_init: float = float(os.environ.get("COIL_RESIDUAL_GATE_INIT", -2.0))
    ve_enabled: bool = bool(int(os.environ.get("VE_ENABLED", "0")))
    ve_dim: int = int(os.environ.get("VE_DIM", 128))
    ve_layers: str = os.environ.get("VE_LAYERS", "9,10").strip()
    layer_mlp_mult_schedule: str = os.environ.get("LAYER_MLP_MULT_SCHEDULE", "").strip()
    nonuniform_mlp_schedule: bool = bool(int(os.environ.get("NONUNIFORM_MLP_SCHEDULE", "0")))
    nonuniform_mlp_mult_min: float = float(os.environ.get("NONUNIFORM_MLP_MULT_MIN", 2.0))
    nonuniform_mlp_mult_max: float = float(os.environ.get("NONUNIFORM_MLP_MULT_MAX", 4.0))
    parallel_residual_last_n: int = int(os.environ.get("PARALLEL_RESIDUAL_LAST_N", 0))
    hemisphere_last_n: int = int(os.environ.get("HEMISPHERE_LAST_N", 0))
    hemisphere_mix_init: float = float(os.environ.get("HEMISPHERE_MIX_INIT", -2.0))
    xsa_last_n: int = int(os.environ.get("XSA_LAST_N", 0))
    xsa_gate_mode: str = os.environ.get("XSA_GATE_MODE", "none").strip().lower()
    xsa_gate_init: float = float(os.environ.get("XSA_GATE_INIT", 2.0))
    depth_lora_rank: int = int(os.environ.get("DEPTH_LORA_RANK", 0))
    hrc_mirror_mode: str = os.environ.get("HRC_MIRROR_MODE", "signperm").strip().lower()
    hrc_depth_schedule_mode: str = os.environ.get("HRC_DEPTH_SCHEDULE_MODE", "cycle").strip().lower()
    hrc_route_repeats: int = int(os.environ.get("HRC_ROUTE_REPEATS", 1))
    hrc_recursive_core_start: int = int(os.environ.get("HRC_RECURSIVE_CORE_START", 2))
    hrc_superloop_skip_schedule: str = os.environ.get("HRC_SUPERLOOP_SKIP_SCHEDULE", "").strip()
    hrc_council_mode: str = os.environ.get("HRC_COUNCIL_MODE", "none").strip().lower()
    hrc_council_train_mode: str = os.environ.get("HRC_COUNCIL_TRAIN_MODE", "always").strip().lower()
    hrc_council_depth_offsets: str = os.environ.get("HRC_COUNCIL_DEPTH_OFFSETS", "").strip()
    hrc_council_conf_scale_init: float = float(os.environ.get("HRC_COUNCIL_CONF_SCALE_INIT", 1.0))
    hrc_base_peer_mode: str = os.environ.get("HRC_BASE_PEER_MODE", "none").strip().lower()
    hrc_pass_embed_enabled: bool = bool(int(os.environ.get("HRC_PASS_EMBED_ENABLED", "0")))
    hrc_pass_embed_init_std: float = float(os.environ.get("HRC_PASS_EMBED_INIT_STD", 0.01))
    hrc_pass_embed_mode: str = os.environ.get("HRC_PASS_EMBED_MODE", "shared").strip().lower()
    hrc_pass_role_mode: str = os.environ.get("HRC_PASS_ROLE_MODE", "none").strip().lower()
    hrc_pass_role_init_std: float = float(os.environ.get("HRC_PASS_ROLE_INIT_STD", 0.003))
    hrc_depth_adapter_tie_mode: str = os.environ.get("HRC_DEPTH_ADAPTER_TIE_MODE", "none").strip().lower()
    hrc_route_phase_enabled: bool = bool(int(os.environ.get("HRC_ROUTE_PHASE_ENABLED", "0")))
    hrc_route_phase_init_std: float = float(os.environ.get("HRC_ROUTE_PHASE_INIT_STD", 0.003))
    hrc_loop_index_enabled: bool = bool(int(os.environ.get("HRC_LOOP_INDEX_ENABLED", "0")))
    hrc_loop_index_dim: int = int(os.environ.get("HRC_LOOP_INDEX_DIM", 0))
    hrc_loop_index_scale_init: float = float(os.environ.get("HRC_LOOP_INDEX_SCALE_INIT", 0.03))
    hrc_recur_inject_enabled: bool = bool(int(os.environ.get("HRC_RECUR_INJECT_ENABLED", "0")))
    hrc_recur_inject_log_a_init: float = float(os.environ.get("HRC_RECUR_INJECT_LOG_A_INIT", 0.0))
    hrc_recur_inject_log_b_init: float = float(os.environ.get("HRC_RECUR_INJECT_LOG_B_INIT", -2.0))
    hrc_council_hard_gate: bool = bool(int(os.environ.get("HRC_COUNCIL_HARD_GATE", "0")))
    hrc_council_entropy_threshold: float = float(os.environ.get("HRC_COUNCIL_ENTROPY_THRESHOLD", 6.0))
    hrc_council_entropy_sharpness: float = float(os.environ.get("HRC_COUNCIL_ENTROPY_SHARPNESS", 8.0))
    hrc_attn_only_blocks: str = os.environ.get("HRC_ATTN_ONLY_BLOCKS", "").strip()
    hrc_mlp_only_blocks: str = os.environ.get("HRC_MLP_ONLY_BLOCKS", "").strip()
    refiner_enabled: bool = bool(int(os.environ.get("REFINER_ENABLED", "0")))
    refiner_rank: int = int(os.environ.get("REFINER_RANK", 32))
    refiner_steps: int = int(os.environ.get("REFINER_STEPS", 2))
    refiner_context_window: int = int(os.environ.get("REFINER_CONTEXT_WINDOW", 4))
    refiner_gate_mode: str = os.environ.get("REFINER_GATE_MODE", "none").strip().lower()
    refiner_gate_init: float = float(os.environ.get("REFINER_GATE_INIT", -1.5))
    refiner_entropy_threshold: float = float(os.environ.get("REFINER_ENTROPY_THRESHOLD", 0.60))
    refiner_entropy_sharpness: float = float(os.environ.get("REFINER_ENTROPY_SHARPNESS", 12.0))
    refiner_aux_base_loss: float = float(os.environ.get("REFINER_AUX_BASE_LOSS", 0.10))
    delta_sidecar_enabled: bool = bool(int(os.environ.get("DELTA_SIDECAR_ENABLED", "0")))
    delta_sidecar_rank: int = int(os.environ.get("DELTA_SIDECAR_RANK", 24))
    delta_sidecar_context_window: int = int(os.environ.get("DELTA_SIDECAR_CONTEXT_WINDOW", 4))
    delta_sidecar_gate_mode: str = os.environ.get("DELTA_SIDECAR_GATE_MODE", "none").strip().lower()
    delta_sidecar_gate_init: float = float(os.environ.get("DELTA_SIDECAR_GATE_INIT", -1.5))
    delta_sidecar_entropy_threshold: float = float(os.environ.get("DELTA_SIDECAR_ENTROPY_THRESHOLD", 0.60))
    delta_sidecar_entropy_sharpness: float = float(os.environ.get("DELTA_SIDECAR_ENTROPY_SHARPNESS", 12.0))
    delta_sidecar_scale_init: float = float(os.environ.get("DELTA_SIDECAR_SCALE_INIT", 0.10))
    council_mode: str = os.environ.get("COUNCIL_MODE", "none").strip().lower()
    micro_recur_last_n: int = int(os.environ.get("MICRO_RECUR_LAST_N", 0))
    micro_recur_steps: int = int(os.environ.get("MICRO_RECUR_STEPS", 1))
    micro_recur_mirror: bool = bool(int(os.environ.get("MICRO_RECUR_MIRROR", "0")))
    micro_recur_gate_init: float = float(os.environ.get("MICRO_RECUR_GATE_INIT", -2.0))
    quant_train_mode: str = os.environ.get("QUANT_TRAIN_MODE", "none").strip().lower()
    quant_train_start_fraction: float = float(os.environ.get("QUANT_TRAIN_START_FRACTION", 0.8))
    quant_train_every: int = int(os.environ.get("QUANT_TRAIN_EVERY", 1))
    probe_tokens_limit: int = int(os.environ.get("PROBE_TOKENS_LIMIT", 8192))
    probe_entropy_threshold: float = float(os.environ.get("PROBE_ENTROPY_THRESHOLD", 6.0))
    ema_enabled: bool = bool(int(os.environ.get("EMA_ENABLED", "0")))
    ema_decay: float = float(os.environ.get("EMA_DECAY", 0.997))
    swa_enabled: bool = bool(int(os.environ.get("SWA_ENABLED", "0")))
    swa_start_fraction: float = float(os.environ.get("SWA_START_FRACTION", 0.60))
    swa_every: int = int(os.environ.get("SWA_EVERY", 1))
    ema_swa_blend: float = float(os.environ.get("EMA_SWA_BLEND", 0.50))

    # Optimizer. We keep the same per-group defaults as train_gpt.py.
    beta1: float = float(os.environ.get("BETA1", 0.9))
    beta2: float = float(os.environ.get("BETA2", 0.95))
    adam_eps: float = float(os.environ.get("ADAM_EPS", 1e-8))
    tied_embed_lr: float = float(os.environ.get("TIED_EMBED_LR", 0.05))
    matrix_lr: float = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr: float = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum: float = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps: int = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start: float = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps: int = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    grad_clip_norm: float = float(os.environ.get("GRAD_CLIP_NORM", 0.0))

    out_dir: str = os.environ.get("OUT_DIR", "logs")
    counted_code_path: str = os.environ.get("COUNTED_CODE_PATH", "./train_gpt.py")
    artifact_codecs: str = os.environ.get("ARTIFACT_CODECS", "zlib:9,lzma:6")
    primary_artifact_codec: str = os.environ.get("PRIMARY_ARTIFACT_CODEC", "zlib")
    write_run_summary_json: bool = bool(int(os.environ.get("WRITE_RUN_SUMMARY_JSON", "1")))

    @property
    def train_files(self) -> str:
        return f"{self.data_path}/fineweb_train_*.bin"

    @property
    def val_files(self) -> str:
        return f"{self.data_path}/fineweb_val_*.bin"

    @property
    def microbatch_tokens(self) -> int:
        return self.train_batch_tokens // self.grad_accum_steps

    def lr_mul(self, step: int, elapsed_ms: float) -> float:
        if self.warmdown_iters <= 0:
            return 1.0
        if self.max_wallclock_seconds <= 0:
            warmdown_start = max(self.iterations - self.warmdown_iters, 0)
            return max((self.iterations - step) / max(self.warmdown_iters, 1), 0.0) if warmdown_start <= step < self.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = self.warmdown_iters * step_ms
        remaining_ms = max(1000.0 * self.max_wallclock_seconds - elapsed_ms, 0.0)
        return remaining_ms / max(warmdown_ms, 1e-9) if remaining_ms <= warmdown_ms else 1.0


CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,xsa_gate,parallel_branch_mix,hemisphere_gate,council_mix_logits,council_prior_logits,council_confidence_scale,council_entropy_threshold,council_entropy_sharpness,micro_recur_gate,bigram.scale,smear,ve_layer_scales,ve_shared.scale,pass_embeddings,pass_role_embeddings,pass_role_scales,depth_role_q_scales,depth_role_v_scales,step_embeddings,context_blend,residual_gate,delta_scale,coil.,loop_index_embeddings,loop_index_scale,recur_inject_log_a,recur_inject_log_b,route_phase_embeddings,route_phase_scales",
    ).split(",")
    if pattern
)
ADAPTER_TENSOR_NAME_PATTERNS = (
    "depth_adapters.",
    "refiner.",
    "delta_sidecar.",
)
AUX_TRAINABLE_TENSOR_NAME_PATTERNS = (
    "bigram.",
    "ve_shared.",
    "ve_layer_scales",
)
TRAINABLE_MODE_PATTERN_MAP = {
    "control": CONTROL_TENSOR_NAME_PATTERNS,
    "adapters": ADAPTER_TENSOR_NAME_PATTERNS,
    "control_adapters": CONTROL_TENSOR_NAME_PATTERNS + ADAPTER_TENSOR_NAME_PATTERNS + AUX_TRAINABLE_TENSOR_NAME_PATTERNS,
    "sidecar": ("refiner.", "delta_sidecar."),
    "refiner": ("refiner.",),
    "delta": ("delta_sidecar.",),
}
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
HOST_TOKEN_DTYPE_MAP = {
    "uint16": np.uint16,
    "int32": np.int32,
}


def parse_pattern_csv(spec: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in spec.split(",") if part.strip())


def env_flag_explicit(name: str) -> bool:
    return name in os.environ


def gb_to_bytes(value_gb: float) -> int:
    return int(max(float(value_gb), 0.0) * (1024**3))


def mlx_memory_snapshot() -> dict[str, int]:
    return {
        "active_bytes": int(mx.get_active_memory()),
        "peak_bytes": int(mx.get_peak_memory()),
        "cache_bytes": int(mx.get_cache_memory()),
    }


def maybe_log_mlx_memory(
    enabled: bool,
    label: str,
    log_fn: Callable[[str], None],
    snapshots: dict[str, dict[str, int]] | None = None,
) -> None:
    if not enabled:
        return
    snap = mlx_memory_snapshot()
    if snapshots is not None:
        snapshots[label] = snap
    mib = 1024.0 * 1024.0
    log_fn(
        f"mlx_memory:{label} active_mib:{snap['active_bytes'] / mib:.1f} "
        f"peak_mib:{snap['peak_bytes'] / mib:.1f} cache_mib:{snap['cache_bytes'] / mib:.1f}"
    )


def cast_host_tokens(tokens: np.ndarray, host_token_dtype: str) -> np.ndarray:
    dtype = HOST_TOKEN_DTYPE_MAP.get(host_token_dtype)
    if dtype is None:
        raise ValueError(
            f"HOST_TOKEN_DTYPE must be one of {','.join(sorted(HOST_TOKEN_DTYPE_MAP))}, got {host_token_dtype!r}"
        )
    if tokens.dtype == dtype:
        return tokens
    return tokens.astype(dtype, copy=False)


def apply_runtime_mode_overrides(args: Hyperparameters) -> None:
    if args.scout_mode == "none":
        return
    if args.scout_mode not in {"fast", "adapter"}:
        raise ValueError(f"SCOUT_MODE must be one of none|fast|adapter, got {args.scout_mode!r}")
    if not env_flag_explicit("SKIP_POST_AVERAGE_EVAL"):
        args.skip_post_average_eval = True
    if not env_flag_explicit("SKIP_PROBE_METRICS"):
        args.skip_probe_metrics = True
    if not env_flag_explicit("SKIP_FINAL_ARTIFACTS"):
        args.skip_final_artifacts = True
    if not env_flag_explicit("SKIP_FINAL_QUANT_EVAL"):
        args.skip_final_quant_eval = True
    if not env_flag_explicit("VAL_TOKENS_LIMIT") and args.val_tokens_limit <= 0:
        args.val_tokens_limit = 131_072
    if not env_flag_explicit("VAL_BATCH_SIZE") and args.val_batch_size >= 524_288:
        args.val_batch_size = 65_536
    if not env_flag_explicit("WARMUP_STEPS"):
        args.warmup_steps = min(args.warmup_steps, 4)
    if not env_flag_explicit("TRAIN_LOG_EVERY"):
        args.train_log_every = min(max(args.train_log_every, 1), 25)
    if args.scout_mode == "adapter" and not env_flag_explicit("TRAINABLE_MODE") and not args.trainable_include_patterns:
        args.trainable_mode = "control_adapters"
    if not env_flag_explicit("TRAIN_SHARD_PREFETCH"):
        args.train_shard_prefetch = max(args.train_shard_prefetch, 1)


def resolve_trainable_patterns(args: Hyperparameters) -> tuple[tuple[str, ...], tuple[str, ...]]:
    include_patterns = parse_pattern_csv(args.trainable_include_patterns)
    exclude_patterns = parse_pattern_csv(args.trainable_exclude_patterns)
    if args.trainable_mode == "all":
        return include_patterns, exclude_patterns
    mode_patterns = TRAINABLE_MODE_PATTERN_MAP.get(args.trainable_mode)
    if mode_patterns is None:
        raise ValueError(
            "TRAINABLE_MODE must be one of all|control|adapters|control_adapters|sidecar|refiner|delta, "
            f"got {args.trainable_mode!r}"
        )
    return tuple(mode_patterns) + include_patterns, exclude_patterns


def _resolve_module_path(root: nn.Module, path_parts: list[str]):
    target = root
    for part in path_parts:
        if isinstance(target, (list, tuple)):
            target = target[int(part)]
        else:
            target = getattr(target, part)
    return target


def _resolve_unfreeze_target(root: nn.Module, path_parts: list[str]) -> tuple[nn.Module, str]:
    if not path_parts:
        raise ValueError("Cannot resolve unfreeze target for empty parameter path")
    if len(path_parts) == 1:
        return root, path_parts[0]

    parent = _resolve_module_path(root, path_parts[:-1])
    leaf = path_parts[-1]
    if isinstance(parent, nn.Module):
        return parent, leaf
    if isinstance(parent, (list, tuple)) and leaf.isdigit():
        owner = _resolve_module_path(root, path_parts[:-2]) if len(path_parts) > 1 else root
        owner_key = path_parts[-2]
        if not isinstance(owner, nn.Module):
            raise TypeError(
                "Selective training expected an nn.Module owner for list-backed tensor "
                f"path={'.'.join(path_parts)!r}, got {type(owner).__name__}"
            )
        return owner, owner_key
    raise TypeError(
        "Selective training could not map parameter path to an nn.Module field: "
        f"path={'.'.join(path_parts)!r} parent_type={type(parent).__name__}"
    )


def configure_trainable_parameters(
    model: nn.Module,
    args: Hyperparameters,
) -> tuple[list[str], tuple[str, ...], tuple[str, ...]]:
    all_param_names = [name for name, _ in tree_flatten(model.parameters())]
    include_patterns, exclude_patterns = resolve_trainable_patterns(args)
    if args.trainable_mode == "all" and not include_patterns and not exclude_patterns:
        return sorted(all_param_names), include_patterns, exclude_patterns

    if args.trainable_mode == "all":
        if include_patterns:
            selected_names = [name for name in all_param_names if any(pattern in name for pattern in include_patterns)]
        else:
            selected_names = list(all_param_names)
    else:
        selected_names = [
            name for name in all_param_names
            if any(pattern in name for pattern in include_patterns)
        ]
    if exclude_patterns:
        selected_names = [
            name for name in selected_names
            if not any(pattern in name for pattern in exclude_patterns)
        ]
    selected_names = sorted(dict.fromkeys(selected_names))
    if not selected_names:
        raise ValueError(
            "Selective training left zero trainable tensors. "
            f"mode={args.trainable_mode!r} include={include_patterns!r} exclude={exclude_patterns!r}"
        )

    model.freeze()
    unfreeze_requests: list[tuple[nn.Module, str]] = []
    seen_requests: set[tuple[int, str]] = set()
    for name in selected_names:
        target_module, target_key = _resolve_unfreeze_target(model, name.split("."))
        dedupe_key = (id(target_module), target_key)
        if dedupe_key in seen_requests:
            continue
        seen_requests.add(dedupe_key)
        unfreeze_requests.append((target_module, target_key))
    for target_module, target_key in unfreeze_requests:
        target_module.unfreeze(keys=target_key, recurse=False, strict=True)

    actual_names = sorted(name for name, _ in tree_flatten(model.trainable_parameters()))
    if actual_names != selected_names:
        missing = sorted(set(selected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(selected_names))
        raise ValueError(
            "Selective training mismatch after freeze/unfreeze. "
            f"missing={missing[:8]!r} extra={extra[:8]!r}"
        )
    return actual_names, include_patterns, exclude_patterns
INT8_KEEP_FLOAT_FP16_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP16_NAME_PATTERNS",
        "",
    ).split(",")
    if pattern
)


# ==============================================================================
# MATH HELPERS
# ==============================================================================

def rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return (x * mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + eps)).astype(x.dtype)


def parse_layer_indices(raw: str, num_layers: int) -> list[int]:
    vals: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            idx = int(item)
        except ValueError as exc:
            raise ValueError(f"VE_LAYERS must be a comma-separated int list, got {raw!r}") from exc
        if idx < 0 or idx >= num_layers:
            raise ValueError(f"VE_LAYERS index {idx} out of range for NUM_LAYERS={num_layers}")
        vals.append(idx)
    return vals


def parse_unique_block_indices(raw: str, num_unique_blocks: int, field_name: str) -> list[int]:
    if not raw:
        return []
    vals: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            idx = int(item)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a comma-separated int list, got {raw!r}") from exc
        if idx < 0 or idx >= num_unique_blocks:
            raise ValueError(f"{field_name} index {idx} out of range for NUM_UNIQUE_BLOCKS={num_unique_blocks}")
        vals.append(idx)
    return vals


HRC_DEPTH_SCHEDULE_ALIASES = {
    "sequential_prime_cycle": "prime_skip_superloop",
}
HRC_DEPTH_SCHEDULE_MODES = {
    "cycle",
    "palindrome",
    "edge_palindrome",
    "anchored_palindrome",
    "recursive_palindrome",
    "transition_recursive_palindrome",
    "transition_recursive_cycle",
    "prime_skip_superloop",
}


def normalize_hrc_depth_schedule_mode(mode: str) -> str:
    raw_mode = str(mode).strip().lower()
    return HRC_DEPTH_SCHEDULE_ALIASES.get(raw_mode, raw_mode)


def parse_superloop_skip_ids(prime_width: int, raw_skip_schedule: str) -> list[int]:
    max_skip_count = max(int(prime_width) // 2, 0)
    if max_skip_count <= 0:
        return []
    if str(raw_skip_schedule).strip():
        requested = parse_csv_int_list(raw_skip_schedule, "HRC_SUPERLOOP_SKIP_SCHEDULE")
    else:
        requested = list(range(max_skip_count))
    ordered: list[int] = []
    seen: set[int] = set()
    for skip_id in requested:
        if skip_id < 0 or skip_id >= max_skip_count:
            raise ValueError(
                "HRC_SUPERLOOP_SKIP_SCHEDULE entries must be in "
                f"[0, {max_skip_count - 1}] for prime interior width {prime_width}; got {skip_id}"
            )
        if skip_id not in seen:
            ordered.append(int(skip_id))
            seen.add(int(skip_id))
    if not ordered:
        raise ValueError(
            "HRC_SUPERLOOP_SKIP_SCHEDULE produced an empty skip program; "
            "provide at least one legal skip family"
        )
    return ordered


def build_prime_skip_superloop_route_metadata(
    effective_depth: int,
    num_unique_blocks: int,
    shell_width: int,
    laps_per_skip: int,
    raw_skip_schedule: str = "",
) -> dict[str, object]:
    depth = int(effective_depth)
    unique_blocks = int(num_unique_blocks)
    if unique_blocks <= 1:
        schedule = [0] * max(depth, 0)
        zeros = [0] * len(schedule)
        neg_ones = [-1] * len(schedule)
        return {
            "block_schedule": schedule,
            "route_phase_schedule": zeros,
            "route_phase_position_schedule": zeros,
            "route_skip_id_schedule": neg_ones,
            "route_skip_hop_schedule": zeros,
            "superloop_skip_ids": [],
            "prime_width": 1,
            "shell_width": 0,
        }
    shell_width = int(shell_width)
    if shell_width < 1 or shell_width >= unique_blocks:
        raise ValueError(
            "prime_skip_superloop requires 1 <= HRC_RECURSIVE_CORE_START < NUM_UNIQUE_BLOCKS; "
            f"got HRC_RECURSIVE_CORE_START={shell_width}, NUM_UNIQUE_BLOCKS={unique_blocks}"
        )
    prime_width = unique_blocks - shell_width
    if prime_width < 3:
        raise ValueError(
            "prime_skip_superloop requires an interior width >= 3 after the mirrored shell; "
            f"got interior width {prime_width} from NUM_UNIQUE_BLOCKS={unique_blocks}, "
            f"HRC_RECURSIVE_CORE_START={shell_width}"
        )
    if not is_prime(prime_width):
        raise ValueError(
            "prime_skip_superloop requires NUM_UNIQUE_BLOCKS - HRC_RECURSIVE_CORE_START to be prime; "
            f"got {prime_width} from NUM_UNIQUE_BLOCKS={unique_blocks}, "
            f"HRC_RECURSIVE_CORE_START={shell_width}"
        )
    skip_ids = parse_superloop_skip_ids(prime_width, raw_skip_schedule)
    laps_per_skip = max(int(laps_per_skip), 1)

    schedule: list[int] = []
    phase_schedule: list[int] = []
    phase_position_schedule: list[int] = []
    skip_id_schedule: list[int] = []
    skip_hop_schedule: list[int] = []

    for shell_pos, block_idx in enumerate(range(shell_width)):
        schedule.append(int(block_idx))
        phase_schedule.append(0)
        phase_position_schedule.append(int(shell_pos))
        skip_id_schedule.append(-1)
        skip_hop_schedule.append(0)

    for phase_idx, skip_id in enumerate(skip_ids, start=1):
        hop = int(skip_id) + 1
        for lap_idx in range(laps_per_skip):
            ring_pos = 0
            for step_idx in range(prime_width):
                schedule.append(shell_width + ring_pos)
                phase_schedule.append(int(phase_idx))
                phase_position_schedule.append(int(lap_idx * prime_width + step_idx))
                skip_id_schedule.append(int(skip_id))
                skip_hop_schedule.append(int(hop))
                ring_pos = (ring_pos + hop) % prime_width

    for shell_pos, block_idx in enumerate(reversed(range(shell_width))):
        schedule.append(int(block_idx))
        phase_schedule.append(0)
        phase_position_schedule.append(int(shell_pos))
        skip_id_schedule.append(-1)
        skip_hop_schedule.append(0)

    if len(schedule) != depth:
        raise ValueError(
            "prime_skip_superloop requires EFFECTIVE_DEPTH to match the explicit route length; "
            f"got EFFECTIVE_DEPTH={depth}, expected {len(schedule)} for "
            f"NUM_UNIQUE_BLOCKS={unique_blocks}, HRC_RECURSIVE_CORE_START={shell_width}, "
            f"HRC_ROUTE_REPEATS={laps_per_skip}, HRC_SUPERLOOP_SKIP_SCHEDULE="
            f"{','.join(str(x) for x in skip_ids)}"
        )
    return {
        "block_schedule": schedule,
        "route_phase_schedule": phase_schedule,
        "route_phase_position_schedule": phase_position_schedule,
        "route_skip_id_schedule": skip_id_schedule,
        "route_skip_hop_schedule": skip_hop_schedule,
        "superloop_skip_ids": [int(x) for x in skip_ids],
        "prime_width": int(prime_width),
        "shell_width": int(shell_width),
    }


def build_hrc_route_metadata(
    effective_depth: int,
    num_unique_blocks: int,
    mode: str,
    route_repeats: int = 1,
    recursive_core_start: int = 2,
    superloop_skip_schedule: str = "",
) -> dict[str, object]:
    schedule_mode = normalize_hrc_depth_schedule_mode(mode)
    if schedule_mode == "prime_skip_superloop":
        return build_prime_skip_superloop_route_metadata(
            effective_depth=effective_depth,
            num_unique_blocks=num_unique_blocks,
            shell_width=recursive_core_start,
            laps_per_skip=route_repeats,
            raw_skip_schedule=superloop_skip_schedule,
        )
    block_schedule = build_hrc_block_schedule(
        effective_depth=effective_depth,
        num_unique_blocks=num_unique_blocks,
        mode=schedule_mode,
        route_repeats=route_repeats,
        recursive_core_start=recursive_core_start,
        superloop_skip_schedule=superloop_skip_schedule,
    )
    zeros = [0] * len(block_schedule)
    neg_ones = [-1] * len(block_schedule)
    return {
        "block_schedule": block_schedule,
        "route_phase_schedule": zeros,
        "route_phase_position_schedule": zeros,
        "route_skip_id_schedule": neg_ones,
        "route_skip_hop_schedule": zeros,
        "superloop_skip_ids": [],
        "prime_width": 0,
        "shell_width": 0,
    }


def build_hrc_block_schedule(
    effective_depth: int,
    num_unique_blocks: int,
    mode: str,
    route_repeats: int = 1,
    recursive_core_start: int = 2,
    superloop_skip_schedule: str = "",
) -> list[int]:
    depth = int(effective_depth)
    unique_blocks = int(num_unique_blocks)
    schedule_mode = normalize_hrc_depth_schedule_mode(mode)
    if depth <= 0:
        return []
    if schedule_mode == "cycle":
        return [idx % unique_blocks for idx in range(depth)]
    if schedule_mode == "palindrome":
        if unique_blocks <= 1:
            return [0] * depth
        period = 2 * unique_blocks
        schedule: list[int] = []
        for idx in range(depth):
            pos = idx % period
            block_idx = pos if pos < unique_blocks else (period - 1 - pos)
            schedule.append(int(block_idx))
        return schedule
    if schedule_mode == "edge_palindrome":
        if unique_blocks <= 1:
            return [0] * depth
        max_dist = max(min(idx, depth - 1 - idx) for idx in range(depth))
        if max_dist <= 0:
            return [0] * depth
        scale = float(unique_blocks - 1) / float(max_dist)
        schedule: list[int] = []
        for idx in range(depth):
            dist = min(idx, depth - 1 - idx)
            block_idx = min(int(round(dist * scale)), unique_blocks - 1)
            schedule.append(block_idx)
        return schedule
    if schedule_mode == "anchored_palindrome":
        if unique_blocks <= 1:
            return [0] * depth
        last = depth - 1
        schedule: list[int] = []
        for idx in range(depth):
            dist = min(idx, last - idx)
            schedule.append(min(dist, unique_blocks - 1))
        return schedule
    if schedule_mode == "recursive_palindrome":
        if unique_blocks <= 1:
            return [0] * depth
        repeats = max(int(route_repeats), 1)
        inner = list(range(1, unique_blocks)) + list(range(unique_blocks - 1, 0, -1))
        schedule = [0]
        for repeat_idx in range(repeats):
            schedule.extend(inner if repeat_idx == 0 else inner[1:])
        schedule.append(0)
        if len(schedule) != depth:
            raise ValueError(
                "recursive_palindrome requires EFFECTIVE_DEPTH to match the explicit route length; "
                f"got EFFECTIVE_DEPTH={depth}, expected {len(schedule)} for "
                f"NUM_UNIQUE_BLOCKS={unique_blocks} and HRC_ROUTE_REPEATS={repeats}"
            )
        return schedule
    if schedule_mode == "transition_recursive_palindrome":
        if unique_blocks <= 1:
            return [0] * depth
        repeats = max(int(route_repeats), 1)
        core_start = int(recursive_core_start)
        if core_start < 1 or core_start >= unique_blocks:
            raise ValueError(
                "transition_recursive_palindrome requires 1 <= HRC_RECURSIVE_CORE_START < NUM_UNIQUE_BLOCKS; "
                f"got HRC_RECURSIVE_CORE_START={core_start}, NUM_UNIQUE_BLOCKS={unique_blocks}"
            )
        prefix = list(range(core_start))
        core = list(range(core_start, unique_blocks)) + list(range(unique_blocks - 2, core_start - 1, -1))
        schedule = list(prefix)
        for repeat_idx in range(repeats):
            schedule.extend(core if repeat_idx == 0 else core[1:])
        schedule.extend(reversed(prefix))
        if len(schedule) != depth:
            raise ValueError(
                "transition_recursive_palindrome requires EFFECTIVE_DEPTH to match the explicit route length; "
                f"got EFFECTIVE_DEPTH={depth}, expected {len(schedule)} for "
                f"NUM_UNIQUE_BLOCKS={unique_blocks}, HRC_ROUTE_REPEATS={repeats}, "
                f"HRC_RECURSIVE_CORE_START={core_start}"
            )
        return schedule
    if schedule_mode == "transition_recursive_cycle":
        if unique_blocks <= 1:
            return [0] * depth
        repeats = max(int(route_repeats), 1)
        core_start = int(recursive_core_start)
        if core_start < 1 or core_start >= unique_blocks:
            raise ValueError(
                "transition_recursive_cycle requires 1 <= HRC_RECURSIVE_CORE_START < NUM_UNIQUE_BLOCKS; "
                f"got HRC_RECURSIVE_CORE_START={core_start}, NUM_UNIQUE_BLOCKS={unique_blocks}"
            )
        prefix = list(range(core_start))
        core = list(range(core_start, unique_blocks))
        schedule = list(prefix)
        for _ in range(repeats):
            schedule.extend(core)
        schedule.extend(reversed(prefix))
        if len(schedule) != depth:
            raise ValueError(
                "transition_recursive_cycle requires EFFECTIVE_DEPTH to match the explicit route length; "
                f"got EFFECTIVE_DEPTH={depth}, expected {len(schedule)} for "
                f"NUM_UNIQUE_BLOCKS={unique_blocks}, HRC_ROUTE_REPEATS={repeats}, "
                f"HRC_RECURSIVE_CORE_START={core_start}"
            )
        return schedule
    if schedule_mode == "prime_skip_superloop":
        return list(
            build_prime_skip_superloop_route_metadata(
                effective_depth=depth,
                num_unique_blocks=unique_blocks,
                shell_width=recursive_core_start,
                laps_per_skip=route_repeats,
                raw_skip_schedule=superloop_skip_schedule,
            )["block_schedule"]
        )
    raise ValueError(
        "HRC_DEPTH_SCHEDULE_MODE must be one of cycle|palindrome|edge_palindrome|anchored_palindrome|recursive_palindrome|transition_recursive_palindrome|transition_recursive_cycle|prime_skip_superloop|sequential_prime_cycle, "
        f"got {mode!r}"
    )


def build_block_repeat_index_schedule(block_schedule: list[int]) -> list[int]:
    counts: dict[int, int] = {}
    schedule: list[int] = []
    for block_idx in block_schedule:
        count = counts.get(int(block_idx), 0)
        schedule.append(count)
        counts[int(block_idx)] = count + 1
    return schedule


def build_loop_index_embedding_table(max_index: int, dim: int, loop_dim: int, theta: float = 10000.0) -> mx.array:
    table = np.zeros((max(int(max_index), 0) + 1, int(dim)), dtype=np.float32)
    loop_dim = max(0, min(int(loop_dim), int(dim)))
    loop_dim -= loop_dim % 2
    if loop_dim <= 0:
        return mx.array(table, dtype=mx.float32)
    freqs = 1.0 / (
        float(theta)
        ** (np.arange(0, loop_dim, 2, dtype=np.float32) / float(loop_dim))
    )
    for loop_t in range(table.shape[0]):
        angles = float(loop_t) * freqs
        emb = np.concatenate((np.sin(angles), np.cos(angles) - 1.0), axis=0).astype(np.float32)
        table[loop_t, :loop_dim] = emb[:loop_dim]
    return mx.array(table, dtype=mx.float32)


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
        return True
    if (n % 2) == 0 or (n % 3) == 0:
        return False
    i = 5
    while i * i <= n:
        if (n % i) == 0 or (n % (i + 2)) == 0:
            return False
        i += 6
    return True


def choose_coil_tap_offsets(window: int, tap_count: int) -> list[int]:
    max_offset = max(int(window) - 2, 1)
    candidates = [idx for idx in range(1, max_offset + 1) if math.gcd(idx, max(int(window), 1)) == 1]
    if not candidates:
        return [1]
    prime_offsets = [idx for idx in candidates if is_prime(idx)]
    nonprime_offsets = [idx for idx in candidates if idx not in prime_offsets and idx != 1]
    ordered_primary = [1] + prime_offsets
    ordered_primary = [idx for i, idx in enumerate(ordered_primary) if idx not in ordered_primary[:i]]
    if tap_count <= len(ordered_primary):
        if tap_count >= len(ordered_primary):
            return ordered_primary
        picks: list[int] = []
        seen: set[int] = set()
        for raw_idx in np.linspace(0, len(ordered_primary) - 1, tap_count, dtype=int):
            idx = ordered_primary[int(raw_idx)]
            if idx not in seen:
                picks.append(idx)
                seen.add(idx)
        for idx in ordered_primary:
            if len(picks) >= tap_count:
                break
            if idx not in seen:
                picks.append(idx)
                seen.add(idx)
        return picks
    ordered = ordered_primary + nonprime_offsets
    picks: list[int] = []
    seen: set[int] = set()
    for raw_idx in np.linspace(0, len(ordered) - 1, tap_count, dtype=int):
        idx = ordered[int(raw_idx)]
        if idx not in seen:
            picks.append(idx)
            seen.add(idx)
    for idx in ordered:
        if len(picks) >= tap_count:
            break
        if idx not in seen:
            picks.append(idx)
            seen.add(idx)
    return picks


def choose_coil_act_pairs(tap_offsets: list[int], pair_count: int) -> list[tuple[int, int]]:
    ordered = sorted({int(x) for x in tap_offsets if int(x) > 0})
    if len(ordered) < 2 or pair_count <= 0:
        return []
    target = min(int(pair_count), max(1, len(ordered) // 2))
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    left = 0
    right = len(ordered) - 1
    while left < right and len(pairs) < target:
        pair = (ordered[left], ordered[right])
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)
        left += 1
        right -= 1

    idx = 0
    while len(pairs) < target and idx + 1 < len(ordered):
        pair = (ordered[idx], ordered[idx + 1])
        if pair not in seen:
            pairs.append(pair)
            seen.add(pair)
        idx += 1

    return pairs


def build_layer_tie_schedule(
    num_layers: int,
    mode: str,
    block_schedule: list[int] | None = None,
) -> list[int]:
    depth = int(num_layers)
    tie_mode = str(mode).strip().lower()
    if tie_mode == "none":
        return list(range(depth))
    if tie_mode == "palindrome":
        last = depth - 1
        return [min(idx, last - idx) for idx in range(depth)]
    if tie_mode == "block":
        if block_schedule is None:
            raise ValueError("tie mode 'block' requires a block_schedule")
        if len(block_schedule) != depth:
            raise ValueError(
                f"tie mode 'block' requires len(block_schedule)==num_layers, got {len(block_schedule)} vs {depth}"
            )
        return [int(idx) for idx in block_schedule]
    raise ValueError(f"tie mode must be one of none|palindrome|block, got {mode!r}")


def build_pass_role_schedule(num_layers: int, mode: str) -> list[int]:
    depth = int(num_layers)
    role_mode = str(mode).strip().lower()
    if role_mode == "none":
        return []
    if role_mode == "edge2":
        if depth <= 0:
            return []
        last = depth - 1
        return [0 if idx == 0 or idx == last else 1 for idx in range(depth)]
    if role_mode == "phase4":
        if depth <= 0:
            return []
        if depth == 1:
            return [0]
        last = depth - 1
        encoder_limit = max(depth // 2, 1)
        schedule: list[int] = []
        for idx in range(depth):
            if idx == 0:
                schedule.append(0)
            elif idx == last:
                schedule.append(3)
            elif idx < encoder_limit:
                schedule.append(1)
            else:
                schedule.append(2)
        return schedule
    if role_mode == "phase5":
        if depth <= 0:
            return []
        if depth == 1:
            return [0]
        if depth == 2:
            return [0, 4]
        last = depth - 1
        if depth % 2 == 0:
            center_band = {max((depth // 2) - 1, 1), depth // 2}
        else:
            center_band = {depth // 2}
        schedule: list[int] = []
        for idx in range(depth):
            if idx == 0:
                schedule.append(0)
            elif idx == last:
                schedule.append(4)
            elif idx in center_band:
                schedule.append(2)
            elif idx < min(center_band):
                schedule.append(1)
            else:
                schedule.append(3)
        return schedule
    raise ValueError(f"HRC_PASS_ROLE_MODE must be one of none|edge2|phase4|phase5, got {mode!r}")


def apply_rope(x: mx.array, rope: nn.RoPE, rope_dims: int) -> mx.array:
    if rope_dims <= 0 or rope_dims >= x.shape[-1]:
        return rope(x)
    x_rope = rope(x[..., :rope_dims])
    return mx.concatenate([x_rope, x[..., rope_dims:]], axis=-1)


def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # Muon uses this to normalize matrix-shaped gradients before applying them.
    # Background on Muon: https://kellerjordan.github.io/posts/muon/
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(mx.float32)
    x = x / (mx.sqrt(mx.sum(x * x)) + eps)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a_mat = x @ x.T
        b_mat = b * a_mat + c * (a_mat @ a_mat)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.astype(g.dtype)


def load_data_shard(path: Path) -> np.ndarray:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {path}")
    num_tokens = int(header[2])
    if path.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch for {path}")
    tokens = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens.size != num_tokens:
        raise ValueError(f"Short read for {path}")
    return tokens


def load_data_shard_host(path: Path, host_token_dtype: str) -> np.ndarray:
    return cast_host_tokens(load_data_shard(path), host_token_dtype)


# ==============================================================================
# TOKEN STREAMING / BATCHING
# ==============================================================================


class TokenStream:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
        prefetch_shards: int = 1,
        host_token_dtype: str = "uint16",
    ):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.epoch = 1
        self.file_idx = 0
        self.log_fn = log_fn
        self.dataset_name = dataset_name
        self.host_token_dtype = host_token_dtype
        self.tokens = cast_host_tokens(load_data_shard(self.files[0]), self.host_token_dtype)
        self.pos = 0
        self.prefetch_shards = max(int(prefetch_shards), 0)
        self._executor: ThreadPoolExecutor | None = None
        self._prefetch_depth = min(self.prefetch_shards, max(len(self.files) - 1, 0))
        self._prefetch_futures: dict[int, Future[np.ndarray]] = {}
        if self.prefetch_shards > 0 and len(self.files) > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=max(1, min(self._prefetch_depth, 4)),
                thread_name_prefix="golf-shard-prefetch",
            )
            self._schedule_prefetch()

    def _schedule_prefetch(self) -> None:
        if self._executor is None or len(self.files) <= 1:
            return
        for offset in range(1, self._prefetch_depth + 1):
            next_idx = (self.file_idx + offset) % len(self.files)
            if next_idx in self._prefetch_futures:
                continue
            self._prefetch_futures[next_idx] = self._executor.submit(
                load_data_shard_host,
                self.files[next_idx],
                self.host_token_dtype,
            )

    def next_file(self) -> None:
        next_idx = (self.file_idx + 1) % len(self.files)
        next_future = self._prefetch_futures.pop(next_idx, None)
        if next_future is not None:
            self.tokens = next_future.result()
        else:
            self.tokens = cast_host_tokens(load_data_shard(self.files[next_idx]), self.host_token_dtype)
        self.file_idx = next_idx
        if self.file_idx == 0:
            self.epoch += 1
            if self.log_fn is not None:
                self.log_fn(
                    f"WARNING: starting epoch:{self.epoch} "
                    f"dataset:{self.dataset_name} train_shards:{len(self.files)}"
                )
        self.pos = 0
        self._schedule_prefetch()

    def take(self, n: int) -> np.ndarray:
        chunks: list[np.ndarray] = []
        left = n
        while left > 0:
            if self.pos >= self.tokens.size:
                self.next_file()
            k = min(left, int(self.tokens.size - self.pos))
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            left -= k
        return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)

    def close(self) -> None:
        if self._executor is not None:
            for future in self._prefetch_futures.values():
                try:
                    future.cancel()
                except Exception:
                    pass
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._prefetch_futures = {}


class TokenLoader:
    def __init__(
        self,
        pattern: str,
        log_fn: Callable[[str], None] | None = None,
        dataset_name: str = "",
        prefetch_shards: int = 1,
        host_token_dtype: str = "uint16",
    ):
        self.stream = TokenStream(
            pattern,
            log_fn=log_fn,
            dataset_name=dataset_name,
            prefetch_shards=prefetch_shards,
            host_token_dtype=host_token_dtype,
        )

    def next_batch(self, batch_tokens: int, seq_len: int) -> tuple[mx.array, mx.array]:
        usable = (batch_tokens // seq_len) * seq_len
        if usable <= 0:
            raise ValueError(f"token budget too small for seq_len={seq_len}")
        chunk = self.stream.take(usable + 1)
        x = chunk[:-1].reshape(-1, seq_len)
        y = chunk[1:].reshape(-1, seq_len)
        return mx.array(x, dtype=mx.int32), mx.array(y, dtype=mx.int32)

    def close(self) -> None:
        self.stream.close()


# ==============================================================================
# MODEL BLOCKS
# ==============================================================================

class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array, mirror_mode: str = "none") -> mx.array:
        weight = self.weight.astype(x.dtype)
        mode = str(mirror_mode).strip().lower()
        if mode.startswith("hybrid"):
            _, _, base_mode = mode.partition(":")
            mirrored = mirror_weight(weight, base_mode or "signperm")
            return 0.5 * (x @ weight.T + x @ mirrored.T)
        if mode not in {"", "none"}:
            weight = mirror_weight(weight, mode)
        return x @ weight.T


class RMSNormNoWeight(nn.Module):
    # MLX module wrapper around the functional RMSNorm helper so it composes nicely in blocks.
    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x)


class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        gate = mx.sigmoid(self.gate.astype(x.dtype))[None, None, :]
        x_prev = mx.concatenate([mx.zeros_like(x[:, :1, :]), x[:, :-1, :]], axis=1)
        return (1.0 - gate) * x + gate * x_prev


class ValueEmbedding(nn.Module):
    def __init__(self, vocab_size: int, ve_dim: int, out_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, int(ve_dim))
        self.embed.weight = (
            mx.random.normal(self.embed.weight.shape, dtype=mx.float32) * 0.01
        ).astype(COMPUTE_DTYPE)
        self.proj = None if int(ve_dim) == int(out_dim) else nn.Linear(int(ve_dim), int(out_dim), bias=False)
        if self.proj is not None:
            self.proj.weight = mx.zeros_like(self.proj.weight)
        self.scale = mx.array([0.1], dtype=mx.float32)

    def __call__(self, token_ids: mx.array) -> mx.array:
        h = self.embed(token_ids)
        if self.proj is not None:
            h = h @ self.proj.weight.astype(h.dtype).T
        return h * self.scale.astype(h.dtype)[0]


class CoilTemporalFeatures(nn.Module):
    def __init__(
        self,
        dim: int,
        mode: str,
        prime_window: int,
        tap_count: int,
        sparse_topk: int,
        pctm_mode: str,
        pctm_basis_count: int,
        anti_enabled: bool,
        probe_mode: str,
        probe_bias_init: float,
        probe_scale_init: float,
        act_enabled: bool,
        act_pair_count: int,
        act_scale_init: float,
        residual_gate_init: float,
    ):
        super().__init__()
        self.mode = str(mode).strip().lower()
        if self.mode not in {"none", "clf", "pctm", "clf_pctm"}:
            raise ValueError(f"COIL_MODE must be one of none|clf|pctm|clf_pctm, got {self.mode!r}")
        self.prime_window = max(3, int(prime_window))
        self.tap_offsets = choose_coil_tap_offsets(self.prime_window, max(1, int(tap_count)))
        self.tap_index_by_offset = {offset: idx for idx, offset in enumerate(self.tap_offsets)}
        self.sparse_topk = max(0, int(sparse_topk))
        self.pctm_mode = str(pctm_mode).strip().lower()
        if self.pctm_mode not in {"weighted", "circulant"}:
            raise ValueError(f"COIL_PCTM_MODE must be one of weighted|circulant, got {self.pctm_mode!r}")
        self.pctm_basis_count = max(1, int(pctm_basis_count))
        self.anti_enabled = bool(anti_enabled)
        self.probe_mode = str(probe_mode).strip().lower()
        if self.probe_mode not in {"none", "coherence"}:
            raise ValueError(f"COIL_PROBE_MODE must be one of none|coherence, got {self.probe_mode!r}")
        self.act_enabled = bool(act_enabled)
        self.act_pairs = choose_coil_act_pairs(self.tap_offsets, max(0, int(act_pair_count))) if self.act_enabled else []
        n_taps = len(self.tap_offsets)
        init_bias = -np.linspace(0.0, 1.0, n_taps, dtype=np.float32)
        self.lag_logits = mx.array(init_bias, dtype=mx.float32)
        self.delta_logits = mx.array(init_bias, dtype=mx.float32)
        self.pctm_logits = mx.array(init_bias, dtype=mx.float32)
        self.anti_logits = mx.array(init_bias, dtype=mx.float32) if self.anti_enabled else None
        if self.pctm_mode == "circulant":
            self.pctm_basis = self._build_circulant_basis()
            basis_count = int(self.pctm_basis.shape[0])
            self.pctm_basis_logits = mx.array(-np.linspace(0.0, 1.0, basis_count, dtype=np.float32), dtype=mx.float32)
        else:
            self.pctm_basis = None
            self.pctm_basis_logits = None
        if self.probe_mode != "none":
            self.probe_bias = mx.array([probe_bias_init], dtype=mx.float32)
            self.probe_scale = mx.array([probe_scale_init], dtype=mx.float32)
        else:
            self.probe_bias = None
            self.probe_scale = None
        if self.act_pairs:
            self.act_logits = mx.array(-np.linspace(0.0, 1.0, len(self.act_pairs), dtype=np.float32), dtype=mx.float32)
            self.act_scale = mx.array([act_scale_init], dtype=mx.float32)
        else:
            self.act_logits = None
            self.act_scale = None
        self.clf_scale = mx.array([0.08], dtype=mx.float32)
        self.pctm_scale = mx.array([0.08], dtype=mx.float32)
        self.anti_scale = mx.array([0.04], dtype=mx.float32)
        self.residual_gate = mx.array([residual_gate_init], dtype=mx.float32)

    def _build_circulant_basis(self) -> mx.array:
        offsets = np.array(self.tap_offsets, dtype=np.float32)
        basis_count = max(1, min(self.pctm_basis_count, max(1, len(offsets))))
        rows: list[np.ndarray] = []
        for basis_idx in range(basis_count):
            if basis_idx == 0:
                row = np.ones_like(offsets, dtype=np.float32)
            else:
                phase = (2.0 * np.pi * basis_idx * offsets) / float(max(self.prime_window, 1))
                row = np.cos(phase).astype(np.float32) if (basis_idx % 2) == 1 else np.sin(phase).astype(np.float32)
            denom = max(float(np.sum(np.abs(row))), 1e-6)
            rows.append((row / denom).astype(np.float32))
        return mx.array(np.stack(rows, axis=0), dtype=mx.float32)

    def _shift(self, x: mx.array, offset: int) -> mx.array:
        bsz, seqlen, dim = x.shape
        if offset <= 0:
            return x
        if offset >= seqlen:
            return mx.zeros((bsz, seqlen, dim), dtype=x.dtype)
        return mx.concatenate(
            [mx.zeros((bsz, offset, dim), dtype=x.dtype), x[:, :-offset, :]],
            axis=1,
        )

    def _sparse_weights(self, logits: mx.array, dtype: mx.Dtype) -> mx.array:
        weights = mx.softmax(logits.astype(dtype), axis=0)
        if 0 < self.sparse_topk < int(weights.shape[0]):
            keep = mx.argsort(weights)[-self.sparse_topk :]
            mask = mx.zeros_like(weights)
            base_idx = mx.arange(int(weights.shape[0]))
            for i in range(self.sparse_topk):
                mask = mask + (base_idx == keep[i]).astype(weights.dtype)
            weights = weights * mx.clip(mask, 0.0, 1.0)
            weights = weights / mx.maximum(mx.sum(weights), mx.array(1e-9, dtype=weights.dtype))
        return weights

    def __call__(self, x: mx.array) -> mx.array:
        x_norm = rms_norm(x)
        taps = mx.stack([self._shift(x_norm, offset) for offset in self.tap_offsets], axis=0)
        out = mx.zeros_like(x_norm)
        lag_mix = None
        pctm_mix = None
        if self.mode in {"clf", "clf_pctm"}:
            lag_w = self._sparse_weights(self.lag_logits, x.dtype)
            delta_w = self._sparse_weights(self.delta_logits, x.dtype)
            lag_mix = mx.sum(lag_w[:, None, None, None] * taps, axis=0)
            delta_mix = mx.sum(delta_w[:, None, None, None] * (x_norm[None, ...] - taps), axis=0)
            out = out + self.clf_scale.astype(x.dtype)[0] * (lag_mix + 0.5 * delta_mix)
        if self.mode in {"pctm", "clf_pctm"}:
            if self.pctm_mode == "circulant" and self.pctm_basis is not None and self.pctm_basis_logits is not None:
                basis = self.pctm_basis.astype(x.dtype)
                basis_responses = mx.sum(basis[:, :, None, None, None] * taps[None, ...], axis=1)
                basis_w = self._sparse_weights(self.pctm_basis_logits, x.dtype)
                pctm_mix = mx.sum(basis_w[:, None, None, None] * basis_responses, axis=0)
            else:
                pctm_w = self._sparse_weights(self.pctm_logits, x.dtype)
                pctm_mix = mx.sum(pctm_w[:, None, None, None] * taps, axis=0)
            out = out + self.pctm_scale.astype(x.dtype)[0] * pctm_mix
        if self.anti_logits is not None:
            anti_w = self._sparse_weights(self.anti_logits, x.dtype)
            periodicity = mx.sum(anti_w[:, None, None, None] * (x_norm[None, ...] * taps), axis=0)
            out = out - self.anti_scale.astype(x.dtype)[0] * periodicity
        if self.act_logits is not None and self.act_scale is not None:
            pair_terms = []
            for left_offset, right_offset in self.act_pairs:
                left_idx = self.tap_index_by_offset[left_offset]
                right_idx = self.tap_index_by_offset[right_offset]
                pair_terms.append(taps[left_idx] * taps[right_idx])
            if pair_terms:
                pair_stack = mx.stack(pair_terms, axis=0)
                pair_w = self._sparse_weights(self.act_logits, x.dtype)
                pair_mix = mx.sum(pair_w[:, None, None, None] * pair_stack, axis=0)
                out = out + self.act_scale.astype(x.dtype)[0] * rms_norm(pair_mix)
        probe_gate = mx.ones(x_norm.shape[:-1] + (1,), dtype=x.dtype)
        if self.probe_bias is not None and self.probe_scale is not None:
            probe_source = lag_mix if lag_mix is not None else pctm_mix
            if probe_source is None:
                probe_source = x_norm
            coherence = mx.mean(x_norm * probe_source, axis=-1, keepdims=True)
            probe_logits = (
                self.probe_scale.astype(x.dtype)[0] * coherence
                + self.probe_bias.astype(x.dtype)[0]
            )
            probe_gate = mx.sigmoid(probe_logits)
        return mx.sigmoid(self.residual_gate.astype(x.dtype))[0] * probe_gate * out


class MicroDiffusionResidualRefiner(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int,
        steps: int,
        context_window: int,
        gate_mode: str,
        gate_init: float,
        entropy_threshold: float,
        entropy_sharpness: float,
    ):
        super().__init__()
        self.rank = max(0, int(rank))
        self.steps = max(1, int(steps))
        self.context_window = max(0, int(context_window))
        self.gate_mode = str(gate_mode).strip().lower()
        if self.gate_mode not in {"none", "entropy"}:
            raise ValueError(f"REFINER_GATE_MODE must be one of none|entropy, got {self.gate_mode!r}")
        self.hidden_down = CastedLinear(dim, self.rank)
        self.context_down = CastedLinear(dim, self.rank)
        self.state_proj = CastedLinear(self.rank, self.rank)
        self.update_proj = CastedLinear(self.rank, self.rank)
        self.delta_up = CastedLinear(self.rank, dim)
        self.delta_up.weight = mx.zeros_like(self.delta_up.weight)
        self.step_embeddings = (
            mx.random.normal((self.steps, self.rank), dtype=mx.float32) * 0.02
        )
        self.context_blend = mx.array([0.0], dtype=mx.float32)
        self.residual_gate = mx.array([gate_init], dtype=mx.float32)
        self.delta_scale = mx.array([0.10], dtype=mx.float32)
        self.entropy_threshold = float(entropy_threshold)
        self.entropy_sharpness = float(entropy_sharpness)

    def _context_summary(self, x: mx.array) -> mx.array:
        if self.context_window <= 0:
            return mx.zeros_like(x)
        bsz, seqlen, dim = x.shape
        context = mx.zeros((bsz, seqlen, dim), dtype=x.dtype)
        counts = mx.zeros((bsz, seqlen, 1), dtype=x.dtype)
        for offset in range(1, self.context_window + 1):
            if offset >= seqlen:
                break
            shifted = mx.concatenate(
                [mx.zeros((bsz, offset, dim), dtype=x.dtype), x[:, :-offset, :]],
                axis=1,
            )
            mask = mx.concatenate(
                [mx.zeros((bsz, offset, 1), dtype=x.dtype), mx.ones((bsz, seqlen - offset, 1), dtype=x.dtype)],
                axis=1,
            )
            context = context + shifted
            counts = counts + mask
        return context / mx.maximum(counts, mx.array(1.0, dtype=x.dtype))

    def _entropy_gate(self, logits: mx.array) -> mx.array:
        if self.gate_mode != "entropy":
            return mx.ones(logits.shape[:-1] + (1,), dtype=logits.dtype)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        entropy = -mx.sum(probs * mx.log(mx.maximum(probs, 1e-9)), axis=-1, keepdims=True)
        norm = entropy / max(math.log(max(int(logits.shape[-1]), 2)), 1e-6)
        gate_logits = (norm - self.entropy_threshold) * self.entropy_sharpness
        return mx.sigmoid(gate_logits.astype(logits.dtype))

    def __call__(self, x: mx.array, emb: mx.array, peer_mode: str = "none") -> mx.array:
        base_logits = x @ emb.T
        context = self._context_summary(x)
        context_blend = mx.sigmoid(self.context_blend.astype(x.dtype))[0]
        cond = self.hidden_down(rms_norm(x), mirror_mode=peer_mode)
        cond = cond + context_blend * self.context_down(rms_norm(context), mirror_mode=peer_mode)
        latent = mx.zeros(cond.shape, dtype=x.dtype)
        for step_idx in range(self.steps):
            step_embed = self.step_embeddings[step_idx].astype(x.dtype)[None, None, :]
            update = silu(cond + self.state_proj(rms_norm(latent), mirror_mode=peer_mode) + step_embed)
            latent = latent + self.update_proj(update, mirror_mode=peer_mode)
        delta = self.delta_up(rms_norm(latent), mirror_mode=peer_mode)
        delta = delta * self.delta_scale.astype(delta.dtype)[0]
        gate = mx.sigmoid(self.residual_gate.astype(delta.dtype))[0] * self._entropy_gate(base_logits).astype(delta.dtype)
        return (x + gate * delta) @ emb.T


class DeltaLogitSidecar(nn.Module):
    def __init__(
        self,
        dim: int,
        vocab_size: int,
        rank: int,
        context_window: int,
        gate_mode: str,
        gate_init: float,
        entropy_threshold: float,
        entropy_sharpness: float,
        scale_init: float,
    ):
        super().__init__()
        self.rank = max(0, int(rank))
        self.context_window = max(0, int(context_window))
        self.gate_mode = str(gate_mode).strip().lower()
        if self.gate_mode not in {"none", "entropy"}:
            raise ValueError(f"DELTA_SIDECAR_GATE_MODE must be one of none|entropy, got {self.gate_mode!r}")
        self.hidden_down = CastedLinear(dim, self.rank)
        self.context_down = CastedLinear(dim, self.rank)
        self.rank_proj = CastedLinear(self.rank, self.rank)
        self.vocab_up = CastedLinear(self.rank, vocab_size)
        self.vocab_up.weight = mx.zeros_like(self.vocab_up.weight)
        self.context_blend = mx.array([0.0], dtype=mx.float32)
        self.residual_gate = mx.array([gate_init], dtype=mx.float32)
        self.delta_scale = mx.array([scale_init], dtype=mx.float32)
        self.entropy_threshold = float(entropy_threshold)
        self.entropy_sharpness = float(entropy_sharpness)

    def _context_summary(self, x: mx.array) -> mx.array:
        if self.context_window <= 0:
            return mx.zeros_like(x)
        bsz, seqlen, dim = x.shape
        context = mx.zeros((bsz, seqlen, dim), dtype=x.dtype)
        counts = mx.zeros((bsz, seqlen, 1), dtype=x.dtype)
        for offset in range(1, self.context_window + 1):
            if offset >= seqlen:
                break
            shifted = mx.concatenate(
                [mx.zeros((bsz, offset, dim), dtype=x.dtype), x[:, :-offset, :]],
                axis=1,
            )
            mask = mx.concatenate(
                [mx.zeros((bsz, offset, 1), dtype=x.dtype), mx.ones((bsz, seqlen - offset, 1), dtype=x.dtype)],
                axis=1,
            )
            context = context + shifted
            counts = counts + mask
        return context / mx.maximum(counts, mx.array(1.0, dtype=x.dtype))

    def _entropy_gate(self, logits: mx.array) -> mx.array:
        if self.gate_mode != "entropy":
            return mx.ones(logits.shape[:-1] + (1,), dtype=logits.dtype)
        probs = mx.softmax(logits.astype(mx.float32), axis=-1)
        entropy = -mx.sum(probs * mx.log(mx.maximum(probs, 1e-9)), axis=-1, keepdims=True)
        norm = entropy / max(math.log(max(int(logits.shape[-1]), 2)), 1e-6)
        gate_logits = (norm - self.entropy_threshold) * self.entropy_sharpness
        return mx.sigmoid(gate_logits.astype(logits.dtype))

    def __call__(self, x: mx.array, base_logits: mx.array, peer_mode: str = "none") -> mx.array:
        context = self._context_summary(x)
        context_blend = mx.sigmoid(self.context_blend.astype(x.dtype))[0]
        latent = self.hidden_down(rms_norm(x), mirror_mode=peer_mode)
        latent = latent + context_blend * self.context_down(rms_norm(context), mirror_mode=peer_mode)
        latent = silu(latent + self.rank_proj(rms_norm(latent), mirror_mode=peer_mode))
        delta = self.vocab_up(rms_norm(latent), mirror_mode=peer_mode)
        gate = mx.sigmoid(self.residual_gate.astype(delta.dtype))[0]
        gate = gate * self._entropy_gate(base_logits).astype(delta.dtype)
        return base_logits + gate * self.delta_scale.astype(delta.dtype)[0] * delta


class CausalSelfAttention(nn.Module):
    # - separate q/k/v projections
    # - RMSNorm on q and k before attention
    # - RoPE on q and k
    # - causal masked SDPA
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        rope_dims: int,
        qk_gain_init: float,
        qk_norm_enabled: bool = True,
        rbf_qk_norm_enabled: bool = False,
        rbf_disable_rope: bool = True,
        use_xsa: bool = False,
        xsa_gate: bool = False,
        xsa_gate_init: float = 2.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim)
        self.c_k = CastedLinear(dim, kv_dim)
        self.c_v = CastedLinear(dim, kv_dim)
        self.proj = CastedLinear(dim, dim)
        self.q_gain = mx.ones((num_heads,), dtype=mx.float32) * qk_gain_init
        self.qk_norm_enabled = bool(qk_norm_enabled)
        self.rbf_qk_norm_enabled = bool(rbf_qk_norm_enabled)
        self.rbf_disable_rope = bool(rbf_disable_rope)
        if rope_dims < 0 or rope_dims > self.head_dim or rope_dims % 2 != 0:
            raise ValueError(
                f"ROPE_DIMS must be an even value in [0, {self.head_dim}], got {rope_dims}"
            )
        self.rope_dims = self.head_dim if rope_dims <= 0 else rope_dims
        self.rope = nn.RoPE(self.rope_dims, traditional=False, base=rope_base)
        self.scale = self.head_dim ** -0.5
        self.use_xsa = bool(use_xsa)
        self.xsa_gate = mx.array([xsa_gate_init], dtype=mx.float32) if xsa_gate else None

    def __call__(
        self,
        x: mx.array,
        v_embed: mx.array | None = None,
        q_delta: mx.array | None = None,
        v_delta: mx.array | None = None,
        mirror_mode: str = "none",
        use_rbf_override: bool | None = None,
        use_xsa_override: bool | None = None,
    ) -> mx.array:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x, mirror_mode=mirror_mode)
        if q_delta is not None:
            q = q + q_delta.astype(q.dtype)
        k = self.c_k(x, mirror_mode=mirror_mode)
        v = self.c_v(x, mirror_mode=mirror_mode)
        if v_delta is not None:
            v = v + v_delta.astype(v.dtype)
        q = q.reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        if v_embed is not None:
            v = v + v_embed.astype(v.dtype).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        use_rbf = bool(use_rbf_override) if use_rbf_override is not None else False
        use_qk_norm = self.rbf_qk_norm_enabled if use_rbf else self.qk_norm_enabled
        if use_qk_norm:
            q = rms_norm(q).astype(COMPUTE_DTYPE)
            k = rms_norm(k).astype(COMPUTE_DTYPE)
        else:
            q = q.astype(COMPUTE_DTYPE)
            k = k.astype(COMPUTE_DTYPE)

        if not (use_rbf and self.rbf_disable_rope):
            q = apply_rope(q, self.rope, self.rope_dims)
            k = apply_rope(k, self.rope, self.rope_dims)

        if use_rbf:
            # RBF attention can be written as SDPA with an augmented key term:
            # softmax((2 q·k - ||k||^2) / sqrt(d)).
            gain = self.q_gain.astype(q.dtype)[None, :, None, None]
            k_sq = mx.sum(k.astype(mx.float32) * k.astype(mx.float32), axis=-1, keepdims=True).astype(q.dtype)
            q_prime = mx.concatenate([q * gain, mx.ones_like(q[..., :1]) * gain], axis=-1)
            k_prime = mx.concatenate([k, -0.5 * k_sq], axis=-1)
            y = mx.fast.scaled_dot_product_attention(
                q_prime,
                k_prime,
                v,
                scale=(2.0 / math.sqrt(self.head_dim)),
                mask="causal",
            )
        else:
            q = q * self.q_gain.astype(q.dtype)[None, :, None, None]
            y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        y_native = y.transpose(0, 2, 1, 3)
        if self.use_xsa if use_xsa_override is None else bool(use_xsa_override):
            y_xsa = apply_xsa(y_native, v.transpose(0, 2, 1, 3))
            if self.xsa_gate is not None:
                gate = mx.sigmoid(self.xsa_gate.astype(y_xsa.dtype))[0]
                y_native = y_native + gate * (y_xsa - y_native)
            else:
                y_native = y_xsa
        y = y_native.reshape(bsz, seqlen, dim)
        return self.proj(y, mirror_mode=mirror_mode)


class MLP(nn.Module):
    # Baseline MLP uses relu^2 instead of GELU/SiLU. It is cheap and works well in this setup.
    def __init__(
        self,
        dim: int,
        mlp_mult: float,
        activation_kind: str,
        negative_slope: float,
        qsparse_enabled: bool = False,
        qsparse_topk: int = 0,
        qsparse_gate_init: float = -1.5,
        monarch_enabled: bool = False,
        monarch_preferred_blocks: int = 0,
        monarch_gate_init: float = -1.5,
    ):
        super().__init__()
        self.activation_kind = activation_kind
        self.negative_slope = float(negative_slope)
        self.qsparse_enabled = bool(qsparse_enabled) and int(qsparse_topk) > 0
        self.qsparse_topk = max(0, int(qsparse_topk))
        self.qsparse_gate = (
            mx.array([qsparse_gate_init], dtype=mx.float32)
            if self.qsparse_enabled
            else None
        )
        hidden = round_hidden_dim(dim * mlp_mult)
        self.monarch = (
            MonarchMixer(hidden, preferred_blocks=monarch_preferred_blocks, gate_init=monarch_gate_init)
            if bool(monarch_enabled)
            else None
        )
        if self.activation_kind == "swiglu":
            hidden = round_hidden_dim(hidden * (2.0 / 3.0))
            self.fc = CastedLinear(dim, hidden)
            self.gate = CastedLinear(dim, hidden)
        else:
            self.fc = CastedLinear(dim, hidden)
            self.gate = None
        self.proj = CastedLinear(hidden, dim)

    def _apply_qsparse(self, x: mx.array) -> mx.array:
        if not self.qsparse_enabled:
            return x
        width = int(x.shape[-1])
        topk = min(self.qsparse_topk, width)
        if topk <= 0 or topk >= width:
            return x
        scores = mx.abs(x.astype(mx.float32))
        threshold = mx.topk(scores, topk)[..., :1].astype(scores.dtype)
        mask = (scores >= threshold).astype(x.dtype)
        sparse = x * mask
        dense_rms = mx.sqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        sparse_rms = mx.sqrt(mx.mean(sparse * sparse, axis=-1, keepdims=True) + 1e-6)
        sparse = sparse * (dense_rms / sparse_rms)
        gate = mx.sigmoid(self.qsparse_gate.astype(x.dtype))[0]
        return x + gate * (sparse - x)

    def __call__(
        self,
        x: mx.array,
        mirror_mode: str = "none",
        use_qsparse_override: bool | None = None,
    ) -> mx.array:
        use_qsparse = self.qsparse_enabled if use_qsparse_override is None else (self.qsparse_enabled and bool(use_qsparse_override))
        if self.activation_kind == "swiglu":
            x = self.fc(x, mirror_mode=mirror_mode) * silu(self.gate(x, mirror_mode=mirror_mode))
            if use_qsparse:
                x = self._apply_qsparse(x)
            if self.monarch is not None:
                x = self.monarch(x, mirror_mode=mirror_mode)
            return self.proj(x, mirror_mode=mirror_mode)
        x = self.fc(x, mirror_mode=mirror_mode)
        if self.activation_kind == "leaky_relu2":
            x = leaky_relu(x, self.negative_slope)
        else:
            x = nn.relu(x)
        x = x * x
        if use_qsparse:
            x = self._apply_qsparse(x)
        if self.monarch is not None:
            x = self.monarch(x, mirror_mode=mirror_mode)
        return self.proj(x, mirror_mode=mirror_mode)


def choose_monarch_factors(width: int, preferred_blocks: int = 0) -> tuple[int, int]:
    width = int(width)
    preferred_blocks = int(preferred_blocks)
    if preferred_blocks > 0 and width % preferred_blocks == 0:
        return preferred_blocks, width // preferred_blocks
    best_p = 1
    best_q = width
    best_score = abs(best_q - best_p)
    limit = int(math.sqrt(width))
    for p in range(1, limit + 1):
        if width % p != 0:
            continue
        q = width // p
        score = abs(q - p)
        if score < best_score:
            best_p = p
            best_q = q
            best_score = score
    return best_p, best_q


class MonarchMixer(nn.Module):
    def __init__(self, width: int, preferred_blocks: int = 0, gate_init: float = -1.5):
        super().__init__()
        self.width = int(width)
        self.left_blocks, self.left_block_dim = choose_monarch_factors(self.width, preferred_blocks)
        self.right_blocks = self.left_block_dim
        self.right_block_dim = self.left_blocks
        self.left = mx.random.normal(
            (self.left_blocks, self.left_block_dim, self.left_block_dim),
            dtype=mx.float32,
        ) * 0.02
        self.right = mx.random.normal(
            (self.right_blocks, self.right_block_dim, self.right_block_dim),
            dtype=mx.float32,
        ) * 0.02
        self.gate = mx.array([gate_init], dtype=mx.float32)

    def _apply_blockdiag(self, x: mx.array, weights: mx.array, mirror_mode: str) -> mx.array:
        out_blocks: list[mx.array] = []
        for block_idx in range(int(weights.shape[0])):
            weight = weights[block_idx].astype(x.dtype)
            mode = str(mirror_mode).strip().lower()
            if mode not in {"", "none"}:
                weight = mirror_weight(weight, mode)
            out_blocks.append(x[..., block_idx, :] @ weight.T)
        return mx.stack(out_blocks, axis=-2)

    def __call__(self, x: mx.array, mirror_mode: str = "none") -> mx.array:
        orig_shape = x.shape
        y = x.reshape(*orig_shape[:-1], self.left_blocks, self.left_block_dim)
        y = self._apply_blockdiag(y, self.left, mirror_mode)
        y = mx.swapaxes(y, -1, -2).reshape(*orig_shape[:-1], self.right_blocks, self.right_block_dim)
        y = self._apply_blockdiag(y, self.right, mirror_mode).reshape(*orig_shape[:-1], self.width)
        gate = mx.sigmoid(self.gate.astype(y.dtype))[0]
        return x + gate * (y - x)


class DepthLoRA(nn.Module):
    def __init__(self, dim: int, kv_dim: int, rank: int):
        super().__init__()
        self.enabled = int(rank) > 0
        if not self.enabled:
            self.rank = 0
            return
        self.rank = int(rank)
        self.q_down = CastedLinear(dim, self.rank)
        self.q_up = CastedLinear(self.rank, dim)
        self.v_down = CastedLinear(dim, self.rank)
        self.v_up = CastedLinear(self.rank, kv_dim)
        self.q_up.weight = mx.zeros_like(self.q_up.weight)
        self.v_up.weight = mx.zeros_like(self.v_up.weight)

    def q_delta(self, x: mx.array) -> mx.array | None:
        if not self.enabled:
            return None
        return self.q_up(self.q_down(x))

    def v_delta(self, x: mx.array) -> mx.array | None:
        if not self.enabled:
            return None
        return self.v_up(self.v_down(x))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        rope_base: float,
        rope_dims: int,
        qk_gain_init: float,
        layer_idx: int,
        ln_scale: bool,
        activation_kind: str,
        activation_negative_slope: float,
        qk_norm_enabled: bool = True,
        rbf_qk_norm_enabled: bool = False,
        rbf_disable_rope: bool = True,
        parallel_residual: bool = False,
        hemisphere_enabled: bool = False,
        hemisphere_mix_init: float = -2.0,
        use_xsa: bool = False,
        xsa_gate: bool = False,
        xsa_gate_init: float = 2.0,
        qsparse_enabled: bool = False,
        qsparse_topk: int = 0,
        qsparse_gate_init: float = -1.5,
        monarch_enabled: bool = False,
        monarch_preferred_blocks: int = 0,
        monarch_gate_init: float = -1.5,
        attn_enabled: bool = True,
        mlp_enabled: bool = True,
    ):
        super().__init__()
        self.attn_enabled = bool(attn_enabled)
        self.mlp_enabled = bool(mlp_enabled)
        if not self.attn_enabled and not self.mlp_enabled:
            raise ValueError("Block must enable at least one of attention or MLP")
        self.attn_norm = RMSNormNoWeight()
        self.mlp_norm = RMSNormNoWeight()
        self.attn = (
            CausalSelfAttention(
                dim,
                num_heads,
                num_kv_heads,
                rope_base,
                rope_dims,
                qk_gain_init,
                qk_norm_enabled=qk_norm_enabled,
                rbf_qk_norm_enabled=rbf_qk_norm_enabled,
                rbf_disable_rope=rbf_disable_rope,
                use_xsa=use_xsa,
                xsa_gate=xsa_gate,
                xsa_gate_init=xsa_gate_init,
            )
            if self.attn_enabled
            else None
        )
        self.mlp = (
            MLP(
                dim,
                mlp_mult,
                activation_kind,
                activation_negative_slope,
                qsparse_enabled=qsparse_enabled,
                qsparse_topk=qsparse_topk,
                qsparse_gate_init=qsparse_gate_init,
                monarch_enabled=monarch_enabled,
                monarch_preferred_blocks=monarch_preferred_blocks,
                monarch_gate_init=monarch_gate_init,
            )
            if self.mlp_enabled
            else None
        )
        self.attn_scale = mx.ones((dim,), dtype=mx.float32) if self.attn_enabled else None
        self.mlp_scale = mx.ones((dim,), dtype=mx.float32) if self.mlp_enabled else None
        self.resid_mix = mx.array(np.stack((np.ones((dim,), dtype=np.float32), np.zeros((dim,), dtype=np.float32))))
        self.ln_scale_factor = float(1.0 / math.sqrt(layer_idx + 1)) if ln_scale else 1.0
        self.parallel_residual = bool(parallel_residual) and self.attn_enabled and self.mlp_enabled
        self.parallel_branch_mix = mx.array([0.0, 0.0], dtype=mx.float32) if self.parallel_residual else None
        self.hemisphere_gate = mx.array([hemisphere_mix_init], dtype=mx.float32) if hemisphere_enabled else None

    def mix_hemisphere(self, y: mx.array) -> mx.array:
        if self.hemisphere_gate is None:
            return y
        gate = mx.sigmoid(self.hemisphere_gate.astype(y.dtype))[0]
        return y + gate * (hemisphere_transform(y) - y)

    def __call__(
        self,
        x: mx.array,
        x0: mx.array,
        v_embed: mx.array | None = None,
        depth_adapter: DepthLoRA | None = None,
        q_role_scale: mx.array | None = None,
        v_role_scale: mx.array | None = None,
        mirror_mode: str = "none",
        use_rbf_override: bool | None = None,
        use_xsa_override: bool | None = None,
        use_qsparse_override: bool | None = None,
    ) -> mx.array:
        mix = self.resid_mix.astype(x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_input = self.attn_norm(x) * self.ln_scale_factor
        q_delta = (
            depth_adapter.q_delta(attn_input)
            if self.attn_enabled and depth_adapter is not None and depth_adapter.enabled
            else None
        )
        if q_delta is not None and q_role_scale is not None:
            q_delta = q_delta * q_role_scale[None, None, :]
        v_delta = (
            depth_adapter.v_delta(attn_input)
            if self.attn_enabled and depth_adapter is not None and depth_adapter.enabled
            else None
        )
        if v_delta is not None and v_role_scale is not None:
            v_delta = v_delta * v_role_scale[None, None, :]
        if self.parallel_residual:
            attn_out = self.mix_hemisphere(
                self.attn(
                    attn_input,
                    v_embed=v_embed,
                    q_delta=q_delta,
                    v_delta=v_delta,
                    mirror_mode=mirror_mode,
                    use_rbf_override=use_rbf_override,
                    use_xsa_override=use_xsa_override,
                )
            )
            mlp_out = self.mix_hemisphere(
                self.mlp(attn_input, mirror_mode=mirror_mode, use_qsparse_override=use_qsparse_override)
            )
            branch_mix = mx.softmax(self.parallel_branch_mix.astype(x.dtype), axis=0)
            x = x + branch_mix[0] * self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
            x = x + branch_mix[1] * self.mlp_scale.astype(x.dtype)[None, None, :] * mlp_out
            return x
        if self.attn_enabled:
            attn_out = self.mix_hemisphere(
                self.attn(
                    attn_input,
                    v_embed=v_embed,
                    q_delta=q_delta,
                    v_delta=v_delta,
                    mirror_mode=mirror_mode,
                    use_rbf_override=use_rbf_override,
                    use_xsa_override=use_xsa_override,
                )
            )
            x = x + self.attn_scale.astype(x.dtype)[None, None, :] * attn_out
        if self.mlp_enabled:
            mlp_out = self.mix_hemisphere(
                self.mlp(
                    self.mlp_norm(x) * self.ln_scale_factor,
                    mirror_mode=mirror_mode,
                    use_qsparse_override=use_qsparse_override,
                )
            )
            x = x + self.mlp_scale.astype(x.dtype)[None, None, :] * mlp_out
        return x


class GPT(nn.Module):
    # - token embedding + RMSNorm
    # - encoder half accumulates skip tensors
    # - decoder half consumes reversed skips with learned skip_weights
    # - tied embeddings for the LM head (the baseline default setup)
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
        rope_dims: int,
        tied_embed_init_std: float,
        qk_gain_init: float,
        ln_scale: bool,
        smear_enabled: bool,
        activation_kind: str,
        activation_negative_slope: float,
        rbf_last_n: int = 0,
        rbf_qk_norm_enabled: bool = False,
        rbf_disable_rope: bool = True,
        qsparse_enabled: bool = False,
        qsparse_topk: int = 0,
        qsparse_last_n: int = 0,
        qsparse_gate_init: float = -1.5,
        monarch_enabled: bool = False,
        monarch_last_n: int = 0,
        monarch_preferred_blocks: int = 0,
        monarch_gate_init: float = -1.5,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        coil_mode: str = "none",
        coil_prime_window: int = 17,
        coil_tap_count: int = 6,
        coil_sparse_topk: int = 0,
        coil_pctm_mode: str = "weighted",
        coil_pctm_basis_count: int = 4,
        coil_anti_enabled: bool = False,
        coil_probe_mode: str = "none",
        coil_probe_bias_init: float = -2.0,
        coil_probe_scale_init: float = 4.0,
        coil_act_enabled: bool = False,
        coil_act_pair_count: int = 2,
        coil_act_scale_init: float = 0.03,
        coil_residual_gate_init: float = -2.0,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "9,10",
        layer_mlp_mults: list[float] | None = None,
        parallel_residual_flags: list[bool] | None = None,
        hemisphere_layer_flags: list[bool] | None = None,
        hemisphere_mix_init: float = -2.0,
        xsa_layer_flags: list[bool] | None = None,
        xsa_gate_layer_flags: list[bool] | None = None,
        xsa_gate_init: float = 2.0,
        council_mode: str = "none",
        micro_recur_last_n: int = 0,
        micro_recur_steps: int = 1,
        micro_recur_mirror: bool = False,
        micro_recur_gate_init: float = -2.0,
        delta_sidecar_enabled: bool = False,
        delta_sidecar_rank: int = 24,
        delta_sidecar_context_window: int = 4,
        delta_sidecar_gate_mode: str = "none",
        delta_sidecar_gate_init: float = -1.5,
        delta_sidecar_entropy_threshold: float = 0.60,
        delta_sidecar_entropy_sharpness: float = 12.0,
        delta_sidecar_scale_init: float = 0.10,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = logit_chunk_tokens
        self.logit_softcap = logit_softcap
        self.council_mode = str(council_mode).strip().lower()
        if self.council_mode not in {"none", "mirror2", "mirror3"}:
            raise ValueError(f"COUNCIL_MODE must be one of none|mirror2|mirror3, got {self.council_mode!r}")

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, dim) if bigram_vocab_size > 0 else None
        self.coil = (
            CoilTemporalFeatures(
                dim=dim,
                mode=coil_mode,
                prime_window=coil_prime_window,
                tap_count=coil_tap_count,
                sparse_topk=coil_sparse_topk,
                pctm_mode=coil_pctm_mode,
                pctm_basis_count=coil_pctm_basis_count,
                anti_enabled=coil_anti_enabled,
                probe_mode=coil_probe_mode,
                probe_bias_init=coil_probe_bias_init,
                probe_scale_init=coil_probe_scale_init,
                act_enabled=coil_act_enabled,
                act_pair_count=coil_act_pair_count,
                act_scale_init=coil_act_scale_init,
                residual_gate_init=coil_residual_gate_init,
            )
            if str(coil_mode).strip().lower() != "none"
            else None
        )
        self.smear = SmearGate(dim) if smear_enabled else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.ve_layer_indices = parse_layer_indices(ve_layers, num_layers) if ve_enabled else []
        self.ve_shared = ValueEmbedding(vocab_size, ve_dim, num_kv_heads * (dim // num_heads)) if self.ve_layer_indices else None
        self.ve_layer_scales = (
            mx.ones((len(self.ve_layer_indices),), dtype=mx.float32)
            if self.ve_layer_indices
            else None
        )
        layer_mlp_mults = [float(mlp_mult)] * num_layers if layer_mlp_mults is None else layer_mlp_mults
        parallel_residual_flags = [False] * num_layers if parallel_residual_flags is None else parallel_residual_flags
        hemisphere_layer_flags = [False] * num_layers if hemisphere_layer_flags is None else hemisphere_layer_flags
        xsa_layer_flags = [False] * num_layers if xsa_layer_flags is None else xsa_layer_flags
        xsa_gate_layer_flags = [False] * num_layers if xsa_gate_layer_flags is None else xsa_gate_layer_flags
        self.rbf_layer_flags = (
            build_tail_flags(num_layers, rbf_last_n)
            if int(rbf_last_n) > 0
            else [False] * num_layers
        )
        if qsparse_enabled:
            self.qsparse_layer_flags = (
                build_tail_flags(num_layers, qsparse_last_n)
                if int(qsparse_last_n) > 0
                else [True] * num_layers
            )
        else:
            self.qsparse_layer_flags = [False] * num_layers
        if monarch_enabled:
            self.monarch_layer_flags = (
                build_tail_flags(num_layers, monarch_last_n)
                if int(monarch_last_n) > 0
                else [True] * num_layers
            )
        else:
            self.monarch_layer_flags = [False] * num_layers
        self.blocks = [
            Block(
                dim,
                num_heads,
                num_kv_heads,
                layer_mlp_mults[i],
                rope_base,
                rope_dims,
                qk_gain_init,
                layer_idx=i,
                qk_norm_enabled=True,
                rbf_qk_norm_enabled=rbf_qk_norm_enabled,
                rbf_disable_rope=rbf_disable_rope,
                ln_scale=ln_scale,
                activation_kind=activation_kind,
                activation_negative_slope=activation_negative_slope,
                parallel_residual=parallel_residual_flags[i],
                hemisphere_enabled=hemisphere_layer_flags[i],
                hemisphere_mix_init=hemisphere_mix_init,
                use_xsa=xsa_layer_flags[i],
                xsa_gate=xsa_gate_layer_flags[i],
                xsa_gate_init=xsa_gate_init,
                qsparse_enabled=qsparse_enabled,
                qsparse_topk=qsparse_topk,
                qsparse_gate_init=qsparse_gate_init,
                monarch_enabled=self.monarch_layer_flags[i],
                monarch_preferred_blocks=monarch_preferred_blocks,
                monarch_gate_init=monarch_gate_init,
            )
            for i in range(num_layers)
        ]
        self.final_norm = RMSNormNoWeight()
        self.micro_recur_last_n = max(0, min(int(micro_recur_last_n), num_layers))
        self.micro_recur_steps = max(0, int(micro_recur_steps))
        self.micro_recur_mirror = bool(micro_recur_mirror)
        self.micro_recur_gate = (
            mx.array([micro_recur_gate_init], dtype=mx.float32)
            if self.micro_recur_last_n > 0 and self.micro_recur_steps > 0
            else None
        )
        self.delta_sidecar = (
            DeltaLogitSidecar(
                dim=dim,
                vocab_size=vocab_size,
                rank=delta_sidecar_rank,
                context_window=delta_sidecar_context_window,
                gate_mode=delta_sidecar_gate_mode,
                gate_init=delta_sidecar_gate_init,
                entropy_threshold=delta_sidecar_entropy_threshold,
                entropy_sharpness=delta_sidecar_entropy_sharpness,
                scale_init=delta_sidecar_scale_init,
            )
            if bool(delta_sidecar_enabled) and int(delta_sidecar_rank) > 0
            else None
        )
        if self.council_mode == "mirror2":
            self.council_mix_logits = mx.array([0.0, -2.0], dtype=mx.float32)
        elif self.council_mode == "mirror3":
            self.council_mix_logits = mx.array([0.0, -2.0, -2.0], dtype=mx.float32)
        else:
            self.council_mix_logits = None

        for b in self.blocks:
            b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def __call__(self, input_ids: mx.array) -> mx.array:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        if self.bigram is not None:
            x = x + self.bigram(input_ids).astype(x.dtype)
        if self.coil is not None:
            x = x + self.coil(x).astype(x.dtype)
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        skips: list[mx.array] = []
        ve_cache: mx.array | None = None

        def ve_for_layer(layer_idx: int) -> mx.array | None:
            nonlocal ve_cache
            if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
                return None
            if ve_cache is None:
                ve_cache = self.ve_shared(input_ids).astype(x.dtype)
            ve_idx = self.ve_layer_indices.index(layer_idx)
            return ve_cache * self.ve_layer_scales[ve_idx].astype(x.dtype)

        for i in range(self.num_encoder_layers):
            x = self.blocks[i](
                x,
                x0,
                v_embed=ve_for_layer(i),
                use_rbf_override=self.rbf_layer_flags[i],
                use_qsparse_override=self.qsparse_layer_flags[i],
            )
            skips.append(x)
        for i in range(self.num_decoder_layers):
            # Odd layer counts have one more decoder block than encoder block. The baseline only
            # applies a skip connection when one exists, then runs the remaining decoder block(s)
            # without an added skip.
            if skips:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            block_idx = self.num_encoder_layers + i
            x = self.blocks[block_idx](
                x,
                x0,
                v_embed=ve_for_layer(block_idx),
                use_rbf_override=self.rbf_layer_flags[block_idx],
                use_qsparse_override=self.qsparse_layer_flags[block_idx],
            )
        if self.micro_recur_gate is not None:
            recur_x = x
            tail_start = len(self.blocks) - self.micro_recur_last_n
            for _ in range(self.micro_recur_steps):
                for block_idx in range(tail_start, len(self.blocks)):
                    block_input = hemisphere_transform(recur_x) if self.micro_recur_mirror else recur_x
                    recur_x = self.blocks[block_idx](
                        block_input,
                        x0,
                        v_embed=ve_for_layer(block_idx),
                        use_rbf_override=self.rbf_layer_flags[block_idx],
                        use_qsparse_override=self.qsparse_layer_flags[block_idx],
                    )
                    if self.micro_recur_mirror:
                        recur_x = hemisphere_transform(recur_x)
            recur_gate = mx.sigmoid(self.micro_recur_gate.astype(x.dtype))[0]
            x = x + recur_gate * (recur_x - x)
        return self.final_norm(x)

    def project_logits(self, x: mx.array) -> mx.array:
        emb_t = self.tok_emb.weight.astype(x.dtype).T
        base_logits = x @ emb_t
        if self.council_mix_logits is None:
            if self.delta_sidecar is not None:
                base_logits = self.delta_sidecar(x, base_logits, peer_mode="none")
            return self.softcap(base_logits)

        peers = [base_logits]
        mirror_x = hemisphere_transform(x)
        peers.append(mirror_x @ emb_t)
        if self.council_mode == "mirror3":
            peers.append(((x + mirror_x) * 0.5) @ emb_t)

        mix = mx.softmax(self.council_mix_logits.astype(x.dtype), axis=0)
        logits_proj = peers[0] * mix[0]
        for idx in range(1, len(peers)):
            logits_proj = logits_proj + peers[idx] * mix[idx]
        if self.delta_sidecar is not None:
            logits_proj = self.delta_sidecar(x, logits_proj, peer_mode="none")
        return self.softcap(logits_proj)

    def loss(self, input_ids: mx.array, target_ids: mx.array, token_weights: mx.array | None = None) -> mx.array:
        # Cross-entropy over flattened tokens. We keep optional logit chunking because it is a useful
        # memory knob on Macs, but the common path is chunk_tokens=0 (single matmul + CE).
        x = self(input_ids).reshape(-1, self.tok_emb.weight.shape[1])
        y = target_ids.reshape(-1)
        weights = token_weights.reshape(-1).astype(mx.float32) if token_weights is not None else None
        if self.logit_chunk_tokens <= 0 or x.shape[0] <= self.logit_chunk_tokens:
            logits = self.project_logits(x)
            if weights is None:
                return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")
            token_loss = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")
            return mx.sum(token_loss * weights) / mx.maximum(mx.sum(weights), mx.array(1e-9, dtype=mx.float32))

        loss_sum = mx.array(0.0, dtype=mx.float32)
        weight_sum = mx.array(0.0, dtype=mx.float32)
        n = int(x.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            logits = self.project_logits(x[s:e])
            if weights is None:
                loss_sum = loss_sum + nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="sum")
            else:
                token_loss = nn.losses.cross_entropy(logits.astype(mx.float32), y[s:e], reduction="none")
                loss_sum = loss_sum + mx.sum(token_loss * weights[s:e])
                weight_sum = weight_sum + mx.sum(weights[s:e])
        if weights is None:
            return loss_sum / float(n)
        return loss_sum / mx.maximum(weight_sum, mx.array(1e-9, dtype=mx.float32))


class HRCGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_unique_blocks: int,
        effective_depth: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: float,
        logit_chunk_tokens: int,
        logit_softcap: float,
        rope_base: float,
        rope_dims: int,
        tied_embed_init_std: float,
        qk_gain_init: float,
        ln_scale: bool,
        activation_kind: str,
        activation_negative_slope: float,
        rbf_last_n: int = 0,
        rbf_qk_norm_enabled: bool = False,
        rbf_disable_rope: bool = True,
        depth_lora_rank: int = 0,
        qsparse_enabled: bool = False,
        qsparse_topk: int = 0,
        qsparse_last_n: int = 0,
        qsparse_gate_init: float = -1.5,
        monarch_enabled: bool = False,
        monarch_last_n: int = 0,
        monarch_preferred_blocks: int = 0,
        monarch_gate_init: float = -1.5,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
        smear_enabled: bool = False,
        coil_mode: str = "none",
        coil_prime_window: int = 17,
        coil_tap_count: int = 6,
        coil_sparse_topk: int = 0,
        coil_pctm_mode: str = "weighted",
        coil_pctm_basis_count: int = 4,
        coil_anti_enabled: bool = False,
        coil_probe_mode: str = "none",
        coil_probe_bias_init: float = -2.0,
        coil_probe_scale_init: float = 4.0,
        coil_act_enabled: bool = False,
        coil_act_pair_count: int = 2,
        coil_act_scale_init: float = 0.03,
        coil_residual_gate_init: float = -2.0,
        ve_enabled: bool = False,
        ve_dim: int = 128,
        ve_layers: str = "",
        layer_mlp_mults: list[float] | None = None,
        parallel_residual_flags: list[bool] | None = None,
        xsa_last_n: int = 0,
        xsa_gate_mode: str = "none",
        xsa_gate_init: float = 2.0,
        hrc_mirror_mode: str = "signperm",
        hrc_depth_schedule_mode: str = "cycle",
        hrc_route_repeats: int = 1,
        hrc_recursive_core_start: int = 2,
        hrc_superloop_skip_schedule: str = "",
        hrc_council_mode: str = "none",
        hrc_council_train_mode: str = "always",
        hrc_council_depth_offsets: str = "",
        hrc_council_conf_scale_init: float = 1.0,
        hrc_base_peer_mode: str = "none",
        hrc_pass_embed_enabled: bool = False,
        hrc_pass_embed_init_std: float = 0.01,
        hrc_pass_embed_mode: str = "shared",
        hrc_pass_role_mode: str = "none",
        hrc_pass_role_init_std: float = 0.003,
        hrc_depth_adapter_tie_mode: str = "none",
        hrc_route_phase_enabled: bool = False,
        hrc_route_phase_init_std: float = 0.003,
        hrc_loop_index_enabled: bool = False,
        hrc_loop_index_dim: int = 0,
        hrc_loop_index_scale_init: float = 0.03,
        hrc_recur_inject_enabled: bool = False,
        hrc_recur_inject_log_a_init: float = 0.0,
        hrc_recur_inject_log_b_init: float = -2.0,
        hrc_council_hard_gate: bool = False,
        hrc_council_entropy_threshold: float = 6.0,
        hrc_council_entropy_sharpness: float = 8.0,
        hrc_attn_only_blocks: str = "",
        hrc_mlp_only_blocks: str = "",
        refiner_enabled: bool = False,
        refiner_rank: int = 32,
        refiner_steps: int = 2,
        refiner_context_window: int = 4,
        refiner_gate_mode: str = "none",
        refiner_gate_init: float = -1.5,
        refiner_entropy_threshold: float = 0.60,
        refiner_entropy_sharpness: float = 12.0,
        refiner_aux_base_loss: float = 0.10,
        delta_sidecar_enabled: bool = False,
        delta_sidecar_rank: int = 24,
        delta_sidecar_context_window: int = 4,
        delta_sidecar_gate_mode: str = "none",
        delta_sidecar_gate_init: float = -1.5,
        delta_sidecar_entropy_threshold: float = 0.60,
        delta_sidecar_entropy_sharpness: float = 12.0,
        delta_sidecar_scale_init: float = 0.10,
        micro_recur_last_n: int = 0,
        micro_recur_steps: int = 1,
        micro_recur_mirror: bool = False,
        micro_recur_gate_init: float = -2.0,
    ):
        super().__init__()
        if effective_depth < num_unique_blocks:
            raise ValueError(
                f"EFFECTIVE_DEPTH ({effective_depth}) must be >= NUM_UNIQUE_BLOCKS ({num_unique_blocks})"
            )
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.logit_chunk_tokens = int(logit_chunk_tokens)
        self.logit_softcap = float(logit_softcap)
        self.num_unique_blocks = int(num_unique_blocks)
        self.effective_depth = int(effective_depth)
        self.mirror_mode = str(hrc_mirror_mode).strip().lower()
        if self.mirror_mode not in {"signperm", "householder"}:
            raise ValueError(f"HRC_MIRROR_MODE must be one of signperm|householder, got {self.mirror_mode!r}")
        self.depth_schedule_mode = normalize_hrc_depth_schedule_mode(hrc_depth_schedule_mode)
        self.route_repeats = max(int(hrc_route_repeats), 1)
        if self.depth_schedule_mode not in HRC_DEPTH_SCHEDULE_MODES:
            raise ValueError(
                "HRC_DEPTH_SCHEDULE_MODE must be one of cycle|palindrome|edge_palindrome|anchored_palindrome|recursive_palindrome|transition_recursive_palindrome|transition_recursive_cycle|prime_skip_superloop|sequential_prime_cycle, "
                f"got {self.depth_schedule_mode!r}"
            )
        self.recursive_core_start = max(int(hrc_recursive_core_start), 1)
        self.superloop_skip_schedule = str(hrc_superloop_skip_schedule).strip()
        self.council_mode = str(hrc_council_mode).strip().lower()
        if self.council_mode not in {"none", "base_mirror", "base_mirror_hybrid"}:
            raise ValueError(
                f"HRC_COUNCIL_MODE must be one of none|base_mirror|base_mirror_hybrid, got {self.council_mode!r}"
            )
        self.council_train_mode = str(hrc_council_train_mode).strip().lower()
        if self.council_train_mode not in {"always", "eval_only"}:
            raise ValueError(
                f"HRC_COUNCIL_TRAIN_MODE must be one of always|eval_only, got {self.council_train_mode!r}"
            )
        self.base_peer_mode = str(hrc_base_peer_mode).strip().lower() or "none"
        self.pass_embed_mode = str(hrc_pass_embed_mode).strip().lower()
        if self.pass_embed_mode not in {"shared", "peer", "palindrome", "palindrome_peer", "block", "block_peer"}:
            raise ValueError(
                f"HRC_PASS_EMBED_MODE must be one of shared|peer|palindrome|palindrome_peer|block|block_peer, "
                f"got {self.pass_embed_mode!r}"
            )
        if self.pass_embed_mode.startswith("palindrome"):
            self.pass_embed_tie_mode = "palindrome"
        elif self.pass_embed_mode.startswith("block"):
            self.pass_embed_tie_mode = "block"
        else:
            self.pass_embed_tie_mode = "none"
        self.pass_embed_peer_transform = self.pass_embed_mode in {"peer", "palindrome_peer", "block_peer"}
        self.pass_role_mode = str(hrc_pass_role_mode).strip().lower()
        if self.pass_role_mode not in {"none", "edge2", "phase4", "phase5"}:
            raise ValueError(
                f"HRC_PASS_ROLE_MODE must be one of none|edge2|phase4|phase5, got {self.pass_role_mode!r}"
            )
        self.pass_role_schedule = build_pass_role_schedule(self.effective_depth, self.pass_role_mode)
        self.depth_adapter_tie_mode = str(hrc_depth_adapter_tie_mode).strip().lower()
        if self.depth_adapter_tie_mode not in {"none", "palindrome", "block"}:
            raise ValueError(
                f"HRC_DEPTH_ADAPTER_TIE_MODE must be one of none|palindrome|block, got {self.depth_adapter_tie_mode!r}"
            )
        self.route_phase_enabled = bool(hrc_route_phase_enabled)
        self.route_phase_init_std = max(0.0, float(hrc_route_phase_init_std))

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.bigram = BigramHashEmbedding(bigram_vocab_size, bigram_dim, dim) if bigram_vocab_size > 0 else None
        self.coil = (
            CoilTemporalFeatures(
                dim=dim,
                mode=coil_mode,
                prime_window=coil_prime_window,
                tap_count=coil_tap_count,
                sparse_topk=coil_sparse_topk,
                pctm_mode=coil_pctm_mode,
                pctm_basis_count=coil_pctm_basis_count,
                anti_enabled=coil_anti_enabled,
                probe_mode=coil_probe_mode,
                probe_bias_init=coil_probe_bias_init,
                probe_scale_init=coil_probe_scale_init,
                act_enabled=coil_act_enabled,
                act_pair_count=coil_act_pair_count,
                act_scale_init=coil_act_scale_init,
                residual_gate_init=coil_residual_gate_init,
            )
            if str(coil_mode).strip().lower() != "none"
            else None
        )
        self.smear = SmearGate(dim) if smear_enabled else None
        self.num_encoder_layers = self.effective_depth // 2
        self.num_decoder_layers = self.effective_depth - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = mx.ones((self.num_skip_weights, dim), dtype=mx.float32)
        self.ve_layer_indices = parse_layer_indices(ve_layers, self.effective_depth) if ve_enabled and ve_layers else []
        self.ve_shared = ValueEmbedding(vocab_size, ve_dim, num_kv_heads * (dim // num_heads)) if self.ve_layer_indices else None
        self.ve_layer_scales = (
            mx.ones((len(self.ve_layer_indices),), dtype=mx.float32)
            if self.ve_layer_indices
            else None
        )

        shared_mlp_mults = [float(mlp_mult)] * self.num_unique_blocks
        if layer_mlp_mults is not None:
            if len(layer_mlp_mults) < self.num_unique_blocks:
                raise ValueError(
                    f"layer_mlp_mults must provide at least NUM_UNIQUE_BLOCKS={self.num_unique_blocks} values, "
                    f"got {len(layer_mlp_mults)}"
                )
            shared_mlp_mults = [float(x) for x in layer_mlp_mults[: self.num_unique_blocks]]
        shared_parallel_flags = [False] * self.num_unique_blocks
        if parallel_residual_flags is not None:
            shared_parallel_flags = list(parallel_residual_flags[: self.num_unique_blocks])
            if len(shared_parallel_flags) < self.num_unique_blocks:
                shared_parallel_flags.extend([False] * (self.num_unique_blocks - len(shared_parallel_flags)))
        attn_only_blocks = set(
            parse_unique_block_indices(hrc_attn_only_blocks, self.num_unique_blocks, "HRC_ATTN_ONLY_BLOCKS")
        )
        mlp_only_blocks = set(
            parse_unique_block_indices(hrc_mlp_only_blocks, self.num_unique_blocks, "HRC_MLP_ONLY_BLOCKS")
        )
        if attn_only_blocks & mlp_only_blocks:
            overlap = ",".join(str(idx) for idx in sorted(attn_only_blocks & mlp_only_blocks))
            raise ValueError(f"HRC_ATTN_ONLY_BLOCKS and HRC_MLP_ONLY_BLOCKS overlap on block(s): {overlap}")
        self.shared_attn_enabled = [idx not in mlp_only_blocks for idx in range(self.num_unique_blocks)]
        self.shared_mlp_enabled = [idx not in attn_only_blocks for idx in range(self.num_unique_blocks)]

        xsa_gated = str(xsa_gate_mode).strip().lower() != "none" and int(xsa_last_n) > 0
        if qsparse_enabled:
            self.qsparse_layer_flags = (
                build_tail_flags(self.effective_depth, qsparse_last_n)
                if int(qsparse_last_n) > 0
                else [True] * self.effective_depth
            )
        else:
            self.qsparse_layer_flags = [False] * self.effective_depth
        if monarch_enabled:
            self.monarch_layer_flags = (
                build_tail_flags(self.effective_depth, monarch_last_n)
                if int(monarch_last_n) > 0
                else [True] * self.effective_depth
            )
        else:
            self.monarch_layer_flags = [False] * self.effective_depth
        self.rbf_layer_flags = (
            build_tail_flags(self.effective_depth, rbf_last_n)
            if int(rbf_last_n) > 0
            else [False] * self.effective_depth
        )
        self.blocks = [
            Block(
                dim,
                num_heads,
                num_kv_heads,
                shared_mlp_mults[i],
                rope_base,
                rope_dims,
                qk_gain_init,
                layer_idx=i,
                qk_norm_enabled=True,
                rbf_qk_norm_enabled=rbf_qk_norm_enabled,
                rbf_disable_rope=rbf_disable_rope,
                ln_scale=ln_scale,
                activation_kind=activation_kind,
                activation_negative_slope=activation_negative_slope,
                parallel_residual=shared_parallel_flags[i],
                use_xsa=False,
                xsa_gate=xsa_gated,
                xsa_gate_init=xsa_gate_init,
                qsparse_enabled=qsparse_enabled,
                qsparse_topk=qsparse_topk,
                qsparse_gate_init=qsparse_gate_init,
                monarch_enabled=self.monarch_layer_flags[i],
                monarch_preferred_blocks=monarch_preferred_blocks,
                monarch_gate_init=monarch_gate_init,
                attn_enabled=self.shared_attn_enabled[i],
                mlp_enabled=self.shared_mlp_enabled[i],
            )
            for i in range(self.num_unique_blocks)
        ]
        route_metadata = build_hrc_route_metadata(
            effective_depth=self.effective_depth,
            num_unique_blocks=self.num_unique_blocks,
            mode=self.depth_schedule_mode,
            route_repeats=self.route_repeats,
            recursive_core_start=self.recursive_core_start,
            superloop_skip_schedule=self.superloop_skip_schedule,
        )
        self.block_schedule = list(route_metadata["block_schedule"])
        self.block_repeat_schedule = build_block_repeat_index_schedule(self.block_schedule)
        self.route_phase_schedule = [int(x) for x in route_metadata["route_phase_schedule"]]
        self.route_phase_position_schedule = [int(x) for x in route_metadata["route_phase_position_schedule"]]
        self.route_skip_id_schedule = [int(x) for x in route_metadata["route_skip_id_schedule"]]
        self.route_skip_hop_schedule = [int(x) for x in route_metadata["route_skip_hop_schedule"]]
        self.superloop_skip_ids = [int(x) for x in route_metadata["superloop_skip_ids"]]
        self.superloop_prime_width = int(route_metadata["prime_width"])
        self.superloop_shell_width = int(route_metadata["shell_width"])
        self.repeat_layer_flags = [idx > 0 for idx in self.block_repeat_schedule]
        kv_dim = num_kv_heads * (dim // num_heads)
        self.depth_adapter_schedule = build_layer_tie_schedule(
            self.effective_depth,
            self.depth_adapter_tie_mode,
            self.block_schedule,
        )
        self.depth_adapters = [
            DepthLoRA(dim, kv_dim, depth_lora_rank)
            for _ in range((max(self.depth_adapter_schedule) + 1) if self.depth_adapter_schedule else 0)
        ]
        self.xsa_layer_flags = build_tail_flags(self.effective_depth, xsa_last_n)
        self.pass_embed_schedule = build_layer_tie_schedule(
            self.effective_depth,
            self.pass_embed_tie_mode,
            self.block_schedule,
        )
        self.pass_embeddings = (
            mx.random.normal(((max(self.pass_embed_schedule) + 1), dim), dtype=mx.float32)
            * float(hrc_pass_embed_init_std)
        ) if hrc_pass_embed_enabled else None
        role_count = (max(self.pass_role_schedule) + 1) if self.pass_role_schedule else 0
        self.pass_role_embeddings = (
            mx.random.normal((role_count, dim), dtype=mx.float32) * float(hrc_pass_role_init_std)
        ) if role_count > 0 else None
        self.pass_role_scales = (
            mx.ones((role_count,), dtype=mx.float32)
        ) if role_count > 0 else None
        phase_count = (max(self.route_phase_schedule) + 1) if self.route_phase_schedule else 0
        self.route_phase_embeddings = (
            mx.random.normal((phase_count, dim), dtype=mx.float32) * self.route_phase_init_std
        ) if self.route_phase_enabled and phase_count > 1 else None
        self.route_phase_scales = (
            mx.ones((phase_count,), dtype=mx.float32)
        ) if self.route_phase_embeddings is not None else None
        self.depth_role_q_scales = (
            mx.zeros((role_count, dim), dtype=mx.float32)
        ) if role_count > 0 else None
        self.depth_role_v_scales = (
            mx.zeros((role_count, kv_dim), dtype=mx.float32)
        ) if role_count > 0 else None
        loop_dim = int(hrc_loop_index_dim)
        if bool(hrc_loop_index_enabled) and loop_dim <= 0:
            loop_dim = max(2, (min(int(dim // 8), int(dim)) // 2) * 2)
        loop_dim = max(0, min(int(loop_dim), int(dim)))
        loop_dim -= loop_dim % 2
        max_loop_index = max(
            max(self.block_repeat_schedule, default=0),
            max(self.route_phase_position_schedule, default=0),
        )
        self.loop_index_embeddings = (
            build_loop_index_embedding_table(max_loop_index, dim, loop_dim)
            if bool(hrc_loop_index_enabled) and loop_dim > 0
            else None
        )
        self.loop_index_scale = (
            mx.array([hrc_loop_index_scale_init], dtype=mx.float32)
            if self.loop_index_embeddings is not None
            else None
        )
        self.loop_index_dim = int(loop_dim) if self.loop_index_embeddings is not None else 0
        self.recur_inject_log_a = (
            mx.array([hrc_recur_inject_log_a_init], dtype=mx.float32)
            if bool(hrc_recur_inject_enabled) and any(self.repeat_layer_flags)
            else None
        )
        self.recur_inject_log_b = (
            mx.array([hrc_recur_inject_log_b_init], dtype=mx.float32)
            if self.recur_inject_log_a is not None
            else None
        )
        self.refiner = (
            MicroDiffusionResidualRefiner(
                dim=dim,
                rank=refiner_rank,
                steps=refiner_steps,
                context_window=refiner_context_window,
                gate_mode=refiner_gate_mode,
                gate_init=refiner_gate_init,
                entropy_threshold=refiner_entropy_threshold,
                entropy_sharpness=refiner_entropy_sharpness,
            )
            if bool(refiner_enabled) and int(refiner_rank) > 0 and int(refiner_steps) > 0
            else None
        )
        self.refiner_aux_base_loss = max(0.0, float(refiner_aux_base_loss))
        self.delta_sidecar = (
            DeltaLogitSidecar(
                dim=dim,
                vocab_size=vocab_size,
                rank=delta_sidecar_rank,
                context_window=delta_sidecar_context_window,
                gate_mode=delta_sidecar_gate_mode,
                gate_init=delta_sidecar_gate_init,
                entropy_threshold=delta_sidecar_entropy_threshold,
                entropy_sharpness=delta_sidecar_entropy_sharpness,
                scale_init=delta_sidecar_scale_init,
            )
            if bool(delta_sidecar_enabled) and int(delta_sidecar_rank) > 0
            else None
        )
        self.micro_recur_last_n = max(0, min(int(micro_recur_last_n), self.effective_depth))
        self.micro_recur_steps = max(0, int(micro_recur_steps))
        self.micro_recur_mirror = bool(micro_recur_mirror)
        self.micro_recur_gate = (
            mx.array([micro_recur_gate_init], dtype=mx.float32)
            if self.micro_recur_last_n > 0 and self.micro_recur_steps > 0
            else None
        )
        self.final_norm = RMSNormNoWeight()

        self.peer_modes = self._build_peer_modes()
        self.peer_depth_limits = self._build_peer_depth_limits(hrc_council_depth_offsets)
        if len(self.peer_modes) > 1:
            prior_init = np.zeros((len(self.peer_modes),), dtype=np.float32)
            prior_init[0] = 0.0
            prior_init[1:] = -0.5
            self.council_prior_logits = mx.array(prior_init, dtype=mx.float32)
            self.council_confidence_scale = mx.array([hrc_council_conf_scale_init], dtype=mx.float32)
        else:
            self.council_prior_logits = None
            self.council_confidence_scale = None
        self.council_hard_gate = bool(hrc_council_hard_gate) and len(self.peer_modes) > 1
        self.council_entropy_threshold = (
            mx.array([hrc_council_entropy_threshold], dtype=mx.float32)
            if self.council_hard_gate
            else None
        )
        self.council_entropy_sharpness = (
            mx.array([hrc_council_entropy_sharpness], dtype=mx.float32)
            if self.council_hard_gate
            else None
        )

        for b in self.blocks:
            if b.attn is not None:
                b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
            if b.mlp is not None:
                b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)
        self.tok_emb.weight = (
            mx.random.normal(self.tok_emb.weight.shape, dtype=mx.float32) * tied_embed_init_std
        ).astype(COMPUTE_DTYPE)

    def _build_peer_modes(self) -> list[str]:
        if self.council_mode == "none":
            return [self.base_peer_mode]
        if self.council_mode == "base_mirror":
            return ["none", self.mirror_mode]
        return ["none", self.mirror_mode, f"hybrid:{self.mirror_mode}"]

    def _build_peer_depth_limits(self, raw_offsets: str) -> list[int]:
        if raw_offsets:
            offsets = parse_csv_int_list(raw_offsets, "HRC_COUNCIL_DEPTH_OFFSETS")
            if len(offsets) != len(self.peer_modes):
                raise ValueError(
                    f"HRC_COUNCIL_DEPTH_OFFSETS must provide one offset per peer ({len(self.peer_modes)}), "
                    f"got {len(offsets)}"
                )
        elif len(self.peer_modes) == 3:
            offsets = [0, 0, -1]
        else:
            offsets = [0] * len(self.peer_modes)
        return [max(1, min(self.effective_depth + int(offset), self.effective_depth)) for offset in offsets]

    def softcap(self, logits: mx.array) -> mx.array:
        c = self.logit_softcap
        return c * mx.tanh(logits / c)

    def _transform_lastdim(self, tensor: mx.array, peer_mode: str) -> mx.array:
        mode = str(peer_mode).strip().lower()
        if mode in {"", "none"}:
            return tensor
        tensor_in = tensor[None, :] if tensor.ndim == 1 else tensor
        if mode.startswith("hybrid"):
            _, _, base_mode = mode.partition(":")
            mirrored = mirror_weight(tensor_in, base_mode or self.mirror_mode)
            out = 0.5 * (tensor_in + mirrored)
        else:
            out = mirror_weight(tensor_in, mode)
        return out[0] if tensor.ndim == 1 else out

    def _layer_peer_mode(self, peer_mode: str, layer_idx: int, depth_limit: int) -> str:
        mode = str(peer_mode).strip().lower()
        if mode in {"", "none", "signperm", "householder"} or mode.startswith("hybrid:"):
            return "none" if mode in {"", "none"} else mode
        prefix, sep, base_mode = mode.partition(":")
        if sep == "":
            return mode
        base_mode = base_mode or self.mirror_mode
        if prefix == "alt":
            return base_mode if (int(layer_idx) % 2) == 1 else "none"
        if prefix == "split":
            return base_mode if int(layer_idx) >= max(int(depth_limit) // 2, 1) else "none"
        if prefix == "splithybrid":
            return f"hybrid:{base_mode}" if int(layer_idx) >= max(int(depth_limit) // 2, 1) else "none"
        if prefix.startswith("tailhybrid"):
            raw_n = prefix[len("tailhybrid") :]
            tail_n = int(raw_n) if raw_n else 1
            return f"hybrid:{base_mode}" if int(layer_idx) >= max(int(depth_limit) - tail_n, 0) else "none"
        if prefix.startswith("tail"):
            raw_n = prefix[len("tail") :]
            tail_n = int(raw_n) if raw_n else 1
            return base_mode if int(layer_idx) >= max(int(depth_limit) - tail_n, 0) else "none"
        return mode

    def _output_peer_mode(self, peer_mode: str, depth_limit: int) -> str:
        return self._layer_peer_mode(peer_mode, max(int(depth_limit) - 1, 0), depth_limit)

    def _project_hidden(
        self,
        x: mx.array,
        peer_mode: str,
        depth_limit: int,
        apply_refiner: bool = True,
        apply_sidecar: bool = True,
    ) -> mx.array:
        emb = self._transform_lastdim(
            self.tok_emb.weight.astype(x.dtype),
            self._output_peer_mode(peer_mode, depth_limit),
        )
        if self.refiner is None or not apply_refiner:
            logits = x @ emb.T
        else:
            logits = self.refiner(x, emb, peer_mode=self._output_peer_mode(peer_mode, depth_limit))
        if self.delta_sidecar is not None and apply_sidecar:
            logits = self.delta_sidecar(x, logits, peer_mode=self._output_peer_mode(peer_mode, depth_limit))
        return logits

    def _stem(self, input_ids: mx.array) -> tuple[mx.array, Callable[[int], mx.array | None]]:
        x = rms_norm(self.tok_emb(input_ids).astype(COMPUTE_DTYPE))
        if self.bigram is not None:
            x = x + self.bigram(input_ids).astype(x.dtype)
        if self.coil is not None:
            x = x + self.coil(x).astype(x.dtype)
        if self.smear is not None:
            x = self.smear(x)
        ve_cache: mx.array | None = None

        def ve_for_layer(layer_idx: int) -> mx.array | None:
            nonlocal ve_cache
            if self.ve_shared is None or layer_idx not in self.ve_layer_indices:
                return None
            if ve_cache is None:
                ve_cache = self.ve_shared(input_ids).astype(x.dtype)
            ve_idx = self.ve_layer_indices.index(layer_idx)
            return ve_cache * self.ve_layer_scales[ve_idx].astype(x.dtype)

        return x, ve_for_layer

    def _pass_embedding_for_layer(
        self,
        layer_idx: int,
        depth_limit: int,
        dtype: mx.Dtype,
        layer_peer_mode: str,
    ) -> mx.array | None:
        pe: mx.array | None = None
        if self.pass_embeddings is not None:
            pe = self.pass_embeddings[self.pass_embed_schedule[layer_idx]].astype(dtype)
            if self.pass_embed_peer_transform:
                pe = self._transform_lastdim(pe, layer_peer_mode)
        if self.pass_role_embeddings is not None and self.pass_role_scales is not None:
            role_schedule = build_pass_role_schedule(depth_limit, self.pass_role_mode)
            if role_schedule:
                role_idx = role_schedule[layer_idx]
                role = self.pass_role_embeddings[role_idx].astype(dtype)
                role = self._transform_lastdim(role, layer_peer_mode)
                role = self.pass_role_scales[role_idx].astype(dtype) * role
                pe = role if pe is None else (pe + role)
        if self.route_phase_embeddings is not None and self.route_phase_scales is not None:
            phase_idx = self.route_phase_schedule[layer_idx]
            if phase_idx > 0:
                phase = self.route_phase_embeddings[phase_idx].astype(dtype)
                phase = self._transform_lastdim(phase, layer_peer_mode)
                phase = self.route_phase_scales[phase_idx].astype(dtype) * phase
                pe = phase if pe is None else (pe + phase)
        if self.loop_index_embeddings is not None and self.loop_index_scale is not None:
            repeat_idx = self.block_repeat_schedule[layer_idx]
            if self.route_phase_schedule[layer_idx] > 0:
                repeat_idx = self.route_phase_position_schedule[layer_idx]
            if repeat_idx > 0:
                loop_emb = self.loop_index_embeddings[repeat_idx].astype(dtype)
                loop_emb = self.loop_index_scale.astype(dtype)[0] * self._transform_lastdim(loop_emb, layer_peer_mode)
                pe = loop_emb if pe is None else (pe + loop_emb)
        return pe

    def _depth_adapter_for_layer(self, layer_idx: int) -> DepthLoRA | None:
        if not self.depth_adapters:
            return None
        return self.depth_adapters[self.depth_adapter_schedule[layer_idx]]

    def _apply_recur_injection(
        self,
        layer_idx: int,
        prev_x: mx.array,
        new_x: mx.array,
        anchor_x: mx.array,
    ) -> mx.array:
        if (
            self.recur_inject_log_a is None
            or self.recur_inject_log_b is None
            or not self.repeat_layer_flags[layer_idx]
        ):
            return new_x
        dtype = new_x.dtype
        recur_a = mx.exp(-mx.exp(self.recur_inject_log_a.astype(dtype)))[0]
        recur_b = (1.0 - recur_a) * mx.sigmoid(self.recur_inject_log_b.astype(dtype))[0]
        return recur_a * prev_x + recur_b * anchor_x + (1.0 - recur_a - recur_b) * new_x

    def _depth_role_scales_for_layer(
        self,
        layer_idx: int,
        depth_limit: int,
        dtype: mx.Dtype,
    ) -> tuple[mx.array | None, mx.array | None]:
        if self.depth_role_q_scales is None or self.depth_role_v_scales is None:
            return None, None
        role_schedule = build_pass_role_schedule(depth_limit, self.pass_role_mode)
        if not role_schedule:
            return None, None
        role_idx = role_schedule[layer_idx]
        q_scale = 1.0 + self.depth_role_q_scales[role_idx].astype(dtype)
        v_scale = 1.0 + self.depth_role_v_scales[role_idx].astype(dtype)
        return q_scale, v_scale

    def _run_peer(self, input_ids: mx.array, peer_mode: str, depth_limit: int) -> mx.array:
        x, ve_for_layer = self._stem(input_ids)
        x0 = x
        skips: list[mx.array] = []
        enc_layers = int(depth_limit) // 2
        dec_layers = int(depth_limit) - enc_layers
        for i in range(enc_layers):
            block_idx = self.block_schedule[i]
            layer_peer_mode = self._layer_peer_mode(peer_mode, i, depth_limit)
            pass_control = self._pass_embedding_for_layer(i, depth_limit, x.dtype, layer_peer_mode)
            q_role_scale, v_role_scale = self._depth_role_scales_for_layer(i, depth_limit, x.dtype)
            prev_x = x
            if pass_control is not None:
                x = x + pass_control[None, None, :]
            x = self.blocks[block_idx](
                x,
                x0,
                v_embed=ve_for_layer(i),
                depth_adapter=self._depth_adapter_for_layer(i),
                q_role_scale=q_role_scale,
                v_role_scale=v_role_scale,
                mirror_mode=layer_peer_mode,
                use_rbf_override=self.rbf_layer_flags[i],
                use_xsa_override=self.xsa_layer_flags[i],
                use_qsparse_override=self.qsparse_layer_flags[i],
            )
            x = self._apply_recur_injection(i, prev_x, x, x0)
            skips.append(x)
        for i in range(dec_layers):
            vi = enc_layers + i
            if skips and i < self.skip_weights.shape[0]:
                x = x + self.skip_weights[i].astype(x.dtype)[None, None, :] * skips.pop()
            block_idx = self.block_schedule[vi]
            layer_peer_mode = self._layer_peer_mode(peer_mode, vi, depth_limit)
            pass_control = self._pass_embedding_for_layer(vi, depth_limit, x.dtype, layer_peer_mode)
            q_role_scale, v_role_scale = self._depth_role_scales_for_layer(vi, depth_limit, x.dtype)
            prev_x = x
            if pass_control is not None:
                x = x + pass_control[None, None, :]
            x = self.blocks[block_idx](
                x,
                x0,
                v_embed=ve_for_layer(vi),
                depth_adapter=self._depth_adapter_for_layer(vi),
                q_role_scale=q_role_scale,
                v_role_scale=v_role_scale,
                mirror_mode=layer_peer_mode,
                use_rbf_override=self.rbf_layer_flags[vi],
                use_xsa_override=self.xsa_layer_flags[vi],
                use_qsparse_override=self.qsparse_layer_flags[vi],
            )
            x = self._apply_recur_injection(vi, prev_x, x, x0)
        if self.micro_recur_gate is not None:
            recur_x = x
            tail_start = max(int(depth_limit) - self.micro_recur_last_n, 0)
            for _ in range(self.micro_recur_steps):
                for vi in range(tail_start, int(depth_limit)):
                    block_idx = self.block_schedule[vi]
                    layer_peer_mode = self._layer_peer_mode(peer_mode, vi, depth_limit)
                    prev_x = recur_x
                    block_input = hemisphere_transform(recur_x) if self.micro_recur_mirror else recur_x
                    pass_control = self._pass_embedding_for_layer(vi, depth_limit, block_input.dtype, layer_peer_mode)
                    q_role_scale, v_role_scale = self._depth_role_scales_for_layer(vi, depth_limit, block_input.dtype)
                    if pass_control is not None:
                        block_input = block_input + pass_control[None, None, :]
                    recur_x = self.blocks[block_idx](
                        block_input,
                        x0,
                        v_embed=ve_for_layer(vi),
                        depth_adapter=self._depth_adapter_for_layer(vi),
                        q_role_scale=q_role_scale,
                        v_role_scale=v_role_scale,
                        mirror_mode=layer_peer_mode,
                        use_rbf_override=self.rbf_layer_flags[vi],
                        use_xsa_override=self.xsa_layer_flags[vi],
                        use_qsparse_override=self.qsparse_layer_flags[vi],
                    )
                    recur_x = self._apply_recur_injection(vi, prev_x, recur_x, x0)
                    if self.micro_recur_mirror:
                        recur_x = hemisphere_transform(recur_x)
            recur_gate = mx.sigmoid(self.micro_recur_gate.astype(x.dtype))[0]
            x = x + recur_gate * (recur_x - x)
        return self.final_norm(x)

    def _peer_hidden_states(self, input_ids: mx.array) -> list[mx.array]:
        return [
            self._run_peer(input_ids, mode, depth_limit)
            for mode, depth_limit in zip(self.peer_modes, self.peer_depth_limits)
        ]

    def _base_peer_hidden_state(self, input_ids: mx.array) -> tuple[mx.array, str, int]:
        base_mode = self.peer_modes[0]
        base_depth_limit = self.peer_depth_limits[0]
        return self._run_peer(input_ids, base_mode, base_depth_limit), base_mode, base_depth_limit

    def _peer_logits_from_hidden_states(self, peer_states: list[mx.array], apply_refiner: bool = True) -> list[mx.array]:
        return [
            self._project_hidden(hidden, mode, depth_limit, apply_refiner=apply_refiner)
            for hidden, mode, depth_limit in zip(peer_states, self.peer_modes, self.peer_depth_limits)
        ]

    def _synthesize_council(self, peer_logits: list[mx.array]) -> mx.array:
        if len(peer_logits) == 1:
            return self.softcap(peer_logits[0])
        stack = mx.stack(peer_logits, axis=0).astype(mx.float32)
        probs = mx.softmax(stack, axis=-1)
        entropy = -mx.sum(probs * mx.log(mx.maximum(probs, 1e-9)), axis=-1)
        conf_scale = mx.abs(self.council_confidence_scale.astype(stack.dtype))[0]
        mix_logits = self.council_prior_logits.astype(stack.dtype)[:, None, None] - conf_scale * entropy
        mix = mx.softmax(mix_logits, axis=0)
        mixed = mx.sum(stack * mix[..., None], axis=0)
        if not self.council_hard_gate:
            return self.softcap(mixed)
        base_entropy = entropy[0]
        threshold = self.council_entropy_threshold.astype(stack.dtype)[0]
        sharpness = mx.maximum(mx.abs(self.council_entropy_sharpness.astype(stack.dtype))[0], 1e-3)
        gate = mx.sigmoid((base_entropy - threshold) * sharpness)
        combined = stack[0] + gate[..., None] * (mixed - stack[0])
        return self.softcap(combined)

    def __call__(self, input_ids: mx.array) -> mx.array:
        return self._run_peer(input_ids, "none", self.effective_depth)

    def loss(
        self,
        input_ids: mx.array,
        target_ids: mx.array,
        token_weights: mx.array | None = None,
        use_council: bool | None = None,
    ) -> mx.array:
        y = target_ids.reshape(-1)
        weights = token_weights.reshape(-1).astype(mx.float32) if token_weights is not None else None
        if use_council is None:
            use_council = len(self.peer_modes) > 1 and self.council_train_mode != "eval_only"
        if use_council:
            peer_states = self._peer_hidden_states(input_ids)
            logits = self._synthesize_council(
                self._peer_logits_from_hidden_states(peer_states, apply_refiner=True)
            ).reshape(-1, self.tok_emb.weight.shape[0])
        else:
            base_hidden, base_mode, base_depth_limit = self._base_peer_hidden_state(input_ids)
            logits = self.softcap(
                self._project_hidden(base_hidden, base_mode, base_depth_limit, apply_refiner=True)
            ).reshape(-1, self.tok_emb.weight.shape[0])
        if self.logit_chunk_tokens <= 0 or logits.shape[0] <= self.logit_chunk_tokens:
            if weights is None:
                loss = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")
            else:
                token_loss = nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="none")
                loss = mx.sum(token_loss * weights) / mx.maximum(mx.sum(weights), mx.array(1e-9, dtype=mx.float32))
            if self.refiner is None or self.refiner_aux_base_loss <= 0.0:
                return loss
            if use_council:
                base_logits = self._synthesize_council(
                    self._peer_logits_from_hidden_states(peer_states, apply_refiner=False)
                ).reshape(-1, self.tok_emb.weight.shape[0])
            else:
                base_hidden, base_mode, base_depth_limit = self._base_peer_hidden_state(input_ids)
                base_logits = self.softcap(
                    self._project_hidden(base_hidden, base_mode, base_depth_limit, apply_refiner=False)
                ).reshape(-1, self.tok_emb.weight.shape[0])
            if weights is None:
                base_loss = nn.losses.cross_entropy(base_logits.astype(mx.float32), y, reduction="mean")
            else:
                base_token_loss = nn.losses.cross_entropy(base_logits.astype(mx.float32), y, reduction="none")
                base_loss = mx.sum(base_token_loss * weights) / mx.maximum(mx.sum(weights), mx.array(1e-9, dtype=mx.float32))
            return loss + self.refiner_aux_base_loss * base_loss

        loss_sum = mx.array(0.0, dtype=mx.float32)
        weight_sum = mx.array(0.0, dtype=mx.float32)
        n = int(logits.shape[0])
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            if weights is None:
                loss_sum = loss_sum + nn.losses.cross_entropy(logits[s:e].astype(mx.float32), y[s:e], reduction="sum")
            else:
                token_loss = nn.losses.cross_entropy(logits[s:e].astype(mx.float32), y[s:e], reduction="none")
                loss_sum = loss_sum + mx.sum(token_loss * weights[s:e])
                weight_sum = weight_sum + mx.sum(weights[s:e])
        if weights is None:
            loss = loss_sum / float(n)
        else:
            loss = loss_sum / mx.maximum(weight_sum, mx.array(1e-9, dtype=mx.float32))
        if self.refiner is None or self.refiner_aux_base_loss <= 0.0:
            return loss
        base_logits = self._synthesize_council(
            self._peer_logits_from_hidden_states(peer_states, apply_refiner=False)
        ).reshape(-1, self.tok_emb.weight.shape[0])
        base_loss_sum = mx.array(0.0, dtype=mx.float32)
        base_weight_sum = mx.array(0.0, dtype=mx.float32)
        for s in range(0, n, self.logit_chunk_tokens):
            e = min(s + self.logit_chunk_tokens, n)
            if weights is None:
                base_loss_sum = base_loss_sum + nn.losses.cross_entropy(
                    base_logits[s:e].astype(mx.float32), y[s:e], reduction="sum"
                )
            else:
                base_token_loss = nn.losses.cross_entropy(base_logits[s:e].astype(mx.float32), y[s:e], reduction="none")
                base_loss_sum = base_loss_sum + mx.sum(base_token_loss * weights[s:e])
                base_weight_sum = base_weight_sum + mx.sum(weights[s:e])
        if weights is None:
            base_loss = base_loss_sum / float(n)
        else:
            base_loss = base_loss_sum / mx.maximum(base_weight_sum, mx.array(1e-9, dtype=mx.float32))
        return loss + self.refiner_aux_base_loss * base_loss

# ==============================================================================
# OPTIMIZERS (MUON + ADAM SPLIT)
# ==============================================================================
class Muon:
    # Muon applies SGD-momentum to matrix gradients, then orthogonalizes the result before the
    # parameter update.
    def __init__(self, keys: list[str], params: dict[str, mx.array], args: Hyperparameters):
        self.keys = keys
        self.args = args
        self.buffers = {k: mx.zeros_like(params[k]) for k in keys}

    def step(self, params: dict[str, mx.array], grads: dict[str, mx.array], step: int, lr_mul: float) -> dict[str, mx.array]:
        if self.args.muon_momentum_warmup_steps:
            t = min(step / self.args.muon_momentum_warmup_steps, 1.0)
            momentum = (1.0 - t) * self.args.muon_momentum_warmup_start + t * self.args.muon_momentum
        else:
            momentum = self.args.muon_momentum
        lr = self.args.matrix_lr * lr_mul
        out: dict[str, mx.array] = {}
        for k in self.keys:
            p = params[k]
            g = grads[k]
            buf = momentum * self.buffers[k] + g
            self.buffers[k] = buf
            g_eff = g + momentum * buf
            g_ortho = zeropower_newtonschulz5(g_eff, self.args.muon_backend_steps)
            scale = math.sqrt(max(1.0, float(p.shape[0]) / float(p.shape[1])))
            out[k] = p - lr * (g_ortho * scale).astype(p.dtype)
        return out


class SplitOptimizers:
    # - embeddings: Adam with the tied-embedding LR
    # - block matrices (2D): Muon
    # - block scalars + skip weights: Adam
    # This preserves the high-level optimization behavior even though MLX internals differ.
    def __init__(self, model: nn.Module, args: Hyperparameters):
        self.args = args
        params = dict(tree_flatten(model.trainable_parameters()))
        self.trainable_keys = sorted(params)
        if not self.trainable_keys:
            raise ValueError("SplitOptimizers requires at least one trainable tensor")
        self.embed_keys = [k for k in ("tok_emb.weight", "bigram.embed.weight", "ve_shared.embed.weight") if k in params]
        self.matrix_keys = [
            k
            for k, p in params.items()
            if (
                k.startswith("blocks.")
                or k.startswith("depth_adapters.")
                or k.startswith("refiner.")
                or k.startswith("delta_sidecar.")
                or k in {"bigram.proj.weight", "ve_shared.proj.weight"}
            )
            and p.ndim == 2
            and not any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k
            for k, p in params.items()
            if k == "skip_weights"
            or k in {
                "council_mix_logits",
                "council_prior_logits",
                "council_confidence_scale",
                "council_entropy_threshold",
                "council_entropy_sharpness",
                "micro_recur_gate",
                "pass_embeddings",
                "pass_role_embeddings",
                "pass_role_scales",
                "depth_role_q_scales",
                "depth_role_v_scales",
            }
            or (k.startswith("blocks.") and (p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
            or (k.startswith("depth_adapters.") and (p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
            or (k.startswith("refiner.") and (p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
            or (k.startswith("delta_sidecar.") and (p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
            or (k.startswith("bigram.") and (p.ndim < 2 or any(pattern in k for pattern in CONTROL_TENSOR_NAME_PATTERNS)))
        ]

        self.muon = Muon(self.matrix_keys, params, args)
        self.adam_embed = optim.Adam(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

    def step(self, model: nn.Module, grads_tree: dict, step: int, lr_mul: float) -> None:
        params = dict(tree_flatten(model.trainable_parameters()))
        grads = dict(tree_flatten(grads_tree))
        updated = dict(params)

        if self.matrix_keys:
            updated.update(self.muon.step(params, grads, step=step, lr_mul=lr_mul))

        if self.embed_keys:
            self.adam_embed.learning_rate = self.args.tied_embed_lr * lr_mul
            updated.update(
                self.adam_embed.apply_gradients(
                    {k: grads[k] for k in self.embed_keys},
                    {k: params[k] for k in self.embed_keys},
                )
            )

        if self.scalar_keys:
            self.adam_scalar.learning_rate = self.args.scalar_lr * lr_mul
            scalar_grads = {k: grads[k] for k in self.scalar_keys}
            scalar_params = {k: params[k] for k in self.scalar_keys}
            updated.update(self.adam_scalar.apply_gradients(scalar_grads, scalar_params))

        model.update(tree_unflatten(list(updated.items())))

# ==============================================================================
# QUANTIZATION (INT8 + ZLIB)
# ==============================================================================
# - per-row int8 for 2D float tensors
# - per-tensor int8 for other float tensors
# - fp16 passthrough for small float tensors
# - exact passthrough for non-floats

MX_DTYPE_FROM_NAME = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}

INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = np.float16
INT8_PER_ROW_SCALE_DTYPE = np.float16
INT8_CLIP_PERCENTILE = float(os.environ.get("INT8_CLIP_PERCENTILE", 99.99984))
INT8_CLIP_PERCENTILE_ATTN = float(os.environ.get("INT8_CLIP_PERCENTILE_ATTN", INT8_CLIP_PERCENTILE))
INT8_CLIP_PERCENTILE_MLP = float(os.environ.get("INT8_CLIP_PERCENTILE_MLP", INT8_CLIP_PERCENTILE))
INT8_CLIP_PERCENTILE_EMBED = float(os.environ.get("INT8_CLIP_PERCENTILE_EMBED", INT8_CLIP_PERCENTILE))


def clip_percentile_for_name(name: str) -> float:
    lname = name.lower()
    if "tok_emb" in lname or ".embed." in lname or "bigram." in lname or "ve_" in lname:
        return INT8_CLIP_PERCENTILE_EMBED
    if ".attn." in lname:
        return INT8_CLIP_PERCENTILE_ATTN
    if ".mlp." in lname:
        return INT8_CLIP_PERCENTILE_MLP
    return INT8_CLIP_PERCENTILE


def _np_float32(arr: mx.array) -> np.ndarray:
    return np.array(arr.astype(mx.float32), dtype=np.float32, copy=False)


def keep_float_array(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> np.ndarray:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return np.ascontiguousarray(_np_float32(arr))
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP16_NAME_PATTERNS):
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = str(arr.dtype).split(".")[-1]
        return np.ascontiguousarray(np.array(arr.astype(mx.float16), dtype=INT8_KEEP_FLOAT_STORE_DTYPE, copy=False))
    return np.ascontiguousarray(np.array(arr, copy=True))


def quantize_float_array(name: str, arr: mx.array) -> tuple[np.ndarray, np.ndarray]:
    f32 = _np_float32(arr)
    clip_q = clip_percentile_for_name(name) / 100.0
    if f32.ndim == 2:
        # Matrices get one scale per row, which usually tracks output-channel
        # ranges much better than a single tensor-wide scale.
        clip_abs = np.quantile(np.abs(f32), clip_q, axis=1) if f32.size else np.empty((f32.shape[0],), dtype=np.float32)
        clipped = np.clip(f32, -clip_abs[:, None], clip_abs[:, None])
        scale = np.maximum(clip_abs / 127.0, 1.0 / 127.0).astype(np.float32, copy=False)
        q = np.clip(np.round(clipped / scale[:, None]), -127, 127).astype(np.int8, copy=False)
        return np.ascontiguousarray(q), np.ascontiguousarray(scale.astype(INT8_PER_ROW_SCALE_DTYPE, copy=False))

    # Vectors / scalars use a simpler per-tensor scale.
    clip_abs = float(np.quantile(np.abs(f32).reshape(-1), clip_q)) if f32.size else 0.0
    scale = np.array(clip_abs / 127.0 if clip_abs > 0.0 else 1.0, dtype=np.float32)
    q = np.clip(np.round(np.clip(f32, -clip_abs, clip_abs) / scale), -127, 127).astype(np.int8, copy=False)
    return np.ascontiguousarray(q), scale


def quantize_state_dict_int8(flat_state: dict[str, mx.array]) -> tuple[dict[str, object], dict[str, int]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors", "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )
    for name, arr in flat_state.items():
        stats["param_count"] += int(arr.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += int(arr.nbytes)
        if not mx.issubdtype(arr.dtype, mx.floating):
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = np.ascontiguousarray(np.array(arr))
            stats["int8_payload_bytes"] += int(passthrough[name].nbytes)
            continue

        # Explicit name-pattern passthrough should work for large tensors too, so
        # we can reserve higher precision for selected route-critical layers
        # (for example the shared entry/exit block in mirrored stacks).
        if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS) or any(
            pattern in name for pattern in INT8_KEEP_FLOAT_FP16_NAME_PATTERNS
        ):
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if int(arr.size) <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_array(name, arr, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += int(kept.nbytes)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_array(name, arr)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0, "clip_percentile": clip_percentile_for_name(name)}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(arr.dtype).split(".")[-1]
        stats["int8_payload_bytes"] += int(q.nbytes + s.nbytes)
    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def dequantize_state_dict_int8(quant_obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    qmeta = quant_obj.get("qmeta", {})
    passthrough_orig_dtypes = quant_obj.get("passthrough_orig_dtypes", {})
    for name, q in quant_obj["quantized"].items():
        q_np = np.asarray(q, dtype=np.int8)
        dtype_name = quant_obj["dtypes"][name]
        scale = np.asarray(quant_obj["scales"][name], dtype=np.float32)
        if qmeta.get(name, {}).get("scheme") == "per_row" or scale.ndim > 0:
            # Broadcast the saved row scale back across trailing dimensions.
            out_arr = q_np.astype(np.float32) * scale.reshape((q_np.shape[0],) + (1,) * (q_np.ndim - 1))
        else:
            out_arr = q_np.astype(np.float32) * float(scale)
        out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[dtype_name])
    for name, arr in quant_obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_arr = np.array(arr, copy=True)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out[name] = mx.array(out_arr, dtype=MX_DTYPE_FROM_NAME[orig_dtype])
        else:
            out[name] = mx.array(out_arr)
    return out


def quantize_dequantize_flat_state_int8(flat_state: dict[str, mx.array]) -> tuple[dict[str, mx.array], dict[str, int]]:
    quant_obj, stats = quantize_state_dict_int8(flat_state)
    return dequantize_state_dict_int8(quant_obj), stats


def flatten_array_state(state_tree) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    for name, value in tree_flatten(state_tree):
        if hasattr(value, "dtype") and hasattr(value, "shape"):
            out[name] = value
    return out


def parse_artifact_codecs(spec: str) -> list[tuple[str, int]]:
    codecs: list[tuple[str, int]] = []
    for item in (x.strip() for x in spec.split(",")):
        if not item:
            continue
        if ":" in item:
            codec, level_s = item.split(":", 1)
            level = int(level_s)
        else:
            codec = item
            level = 9 if codec == "zlib" else 6
        codec = codec.strip().lower()
        if codec not in {"zlib", "lzma"}:
            raise ValueError(f"Unsupported artifact codec {codec!r}; expected zlib or lzma")
        if level < 0 or level > 9:
            raise ValueError(f"Compression level for {codec} must be in [0, 9], got {level}")
        codecs.append((codec, level))
    if not codecs:
        raise ValueError("No valid artifact codecs found in ARTIFACT_CODECS")
    return codecs


def compress_payload(payload: bytes, codec: str, level: int) -> bytes:
    if codec == "zlib":
        return zlib.compress(payload, level=level)
    if codec == "lzma":
        return lzma.compress(payload, preset=level)
    raise ValueError(f"Unsupported codec {codec!r}")


def decompress_payload(blob: bytes, codec: str) -> bytes:
    if codec == "zlib":
        return zlib.decompress(blob)
    if codec == "lzma":
        return lzma.decompress(blob)
    raise ValueError(f"Unsupported codec {codec!r}")


def resolve_counted_code_path(code_path: str) -> Path:
    path = Path(code_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"COUNTED_CODE_PATH does not exist: {path}")
    return path


def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_lut = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_lut = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_lut = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_lut[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_lut[token_id] = True
            piece = piece[1:]
        base_bytes_lut[token_id] = len(piece.encode("utf-8"))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


def validate_dataset_tokenizer_pair(data_path: str, tokenizer_path: str) -> tuple[str, int, int | None]:
    # The shard directory and tokenizer are coupled: val_bpb is only meaningful if we
    # decode bytes with the exact tokenizer that produced the shards. The manifest
    # lets the training script fail fast on accidental dataset/tokenizer mismatches.
    dataset_dir = Path(data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    if len(dataset_dir.parents) < 2:
        return dataset_dir.name, actual_train_files, None
    manifest_path = dataset_dir.parents[1] / "manifest.json"
    if not manifest_path.is_file():
        return dataset_dir.name, actual_train_files, None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_dir.name), None)
    if dataset_entry is None:
        return dataset_dir.name, actual_train_files, None

    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = (
        next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
        if tokenizer_name
        else None
    )
    expected_name = Path((tokenizer_entry or {}).get("model_path") or (tokenizer_entry or {}).get("path") or "").name
    if expected_name and Path(tokenizer_path).name != expected_name:
        raise ValueError(f"{dataset_dir.name} expects tokenizer {expected_name}, got {Path(tokenizer_path).name}")
    expected_train_files = (dataset_entry.get("stats") or {}).get("files_train")
    if expected_train_files is not None:
        expected_train_files = int(expected_train_files)
        if actual_train_files > expected_train_files:
            raise ValueError(
                f"{dataset_dir.name} has more train shards than expected: found {actual_train_files}, "
                f"manifest says {expected_train_files}"
            )
    return dataset_dir.name, actual_train_files, expected_train_files


def load_validation_tokens(
    pattern: str,
    seq_len: int,
    val_tokens_limit: int = 0,
    host_token_dtype: str = "uint16",
) -> np.ndarray:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    tokens = np.ascontiguousarray(
        np.concatenate([load_data_shard_host(file, host_token_dtype) for file in files], axis=0)
    )
    if val_tokens_limit > 0:
        keep = min(val_tokens_limit + 1, tokens.size)
        if keep <= 1:
            raise ValueError(f"VAL_TOKENS_LIMIT={val_tokens_limit} is too small")
        tokens = tokens[:keep]
    usable = ((tokens.size - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def loss_and_grad_chunked(
    args: Hyperparameters,
    train_loader: TokenLoader,
    compiled_loss_and_grad,
    batch_plan: list[tuple[int, float]],
) -> tuple[mx.array, dict]:
    loss_value = mx.array(0.0, dtype=mx.float32)
    grad_accum: dict[str, mx.array] | None = None
    for chunk_tokens, chunk_scale in batch_plan:
        x, y = train_loader.next_batch(chunk_tokens, args.train_seq_len)
        loss, grads = compiled_loss_and_grad(x, y)
        loss_value = loss_value + loss.astype(mx.float32) * chunk_scale
        grad_accum = accumulate_flat_grads(grad_accum, grads, chunk_scale, tree_flatten)
        if args.mlx_eager_eval:
            mx.eval(loss_value, grad_accum)  # materialize each chunk to cap peak memory
    return loss_value, tree_unflatten(list(grad_accum.items()))


def eval_val(
    args: Hyperparameters,
    compiled_loss,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    val_batch_tokens = args.val_batch_size // args.grad_accum_steps
    if val_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
            f"TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    val_batch_seqs = val_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.size - 1) // args.train_seq_len
    total_batches = max((total_seqs + val_batch_seqs - 1) // val_batch_seqs, 1)
    total_loss_sum = 0.0
    total_tokens = 0.0
    total_bytes = 0.0
    for batch_idx, batch_seq_start in enumerate(range(0, total_seqs, val_batch_seqs), start=1):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * args.train_seq_len
        raw_end = batch_seq_end * args.train_seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, args.train_seq_len)
        y_np = chunk[1:].reshape(-1, args.train_seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        y = mx.array(y_np, dtype=mx.int32)
        chunk_token_count = float(y.size)
        batch_loss = compiled_loss(x, y).astype(mx.float32)
        mx.eval(batch_loss)
        total_loss_sum += float(batch_loss.item()) * chunk_token_count
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        bytes_np += (
            has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        ).astype(np.int16, copy=False)
        total_tokens += chunk_token_count
        total_bytes += float(bytes_np.astype(np.float64).sum())
        if log_fn is not None and total_batches > 1 and (
            batch_idx == 1 or batch_idx == total_batches or batch_idx % 25 == 0
        ):
            log_fn(f"val_progress:{batch_idx}/{total_batches}")
    val_loss = total_loss_sum / total_tokens
    bits_per_token = val_loss / math.log(2.0)
    val_bpb = bits_per_token * (total_tokens / total_bytes)
    return val_loss, val_bpb


def forward_model_logits(
    args: Hyperparameters,
    model: nn.Module,
    input_ids: mx.array,
    apply_refiner: bool = True,
) -> mx.array:
    if args.model_family == "hrc":
        if len(getattr(model, "peer_modes", [])) > 1:
            peer_states = model._peer_hidden_states(input_ids)
            return model._synthesize_council(
                model._peer_logits_from_hidden_states(peer_states, apply_refiner=apply_refiner)
            )
        base_hidden, base_mode, base_depth_limit = model._base_peer_hidden_state(input_ids)
        return model.softcap(
            model._project_hidden(
                base_hidden,
                base_mode,
                base_depth_limit,
                apply_refiner=apply_refiner,
            )
        )
    hidden = model(input_ids)
    return model.project_logits(hidden)


def collect_probe_metrics(
    args: Hyperparameters,
    model: nn.Module,
    val_tokens: np.ndarray,
    base_bytes_lut: np.ndarray,
    has_leading_space_lut: np.ndarray,
    is_boundary_token_lut: np.ndarray,
    log_fn: Callable[[str], None] | None = None,
    label: str = "probe",
) -> dict[str, object]:
    probe_limit = min(max(int(args.probe_tokens_limit), 0), int(val_tokens.size - 1))
    usable = (probe_limit // args.train_seq_len) * args.train_seq_len
    if usable <= 0:
        return {}

    seq_len = int(args.train_seq_len)
    total_seqs = usable // seq_len
    val_batch_tokens = max(seq_len, int(args.val_batch_size // max(args.grad_accum_steps, 1)))
    val_batch_seqs = max(1, val_batch_tokens // seq_len)

    total_tokens = 0
    total_bytes = 0.0
    total_correct = 0.0
    total_loss_nat = 0.0
    total_entropy_nat = 0.0
    total_top1_prob = 0.0
    total_hard = 0.0
    entropy_threshold = float(args.probe_entropy_threshold)

    slice_stats: dict[str, dict[str, float]] = {
        "byte1": {"loss_nat": 0.0, "bytes": 0.0, "tokens": 0.0, "correct": 0.0},
        "byte2plus": {"loss_nat": 0.0, "bytes": 0.0, "tokens": 0.0, "correct": 0.0},
        "leading_space_cont": {"loss_nat": 0.0, "bytes": 0.0, "tokens": 0.0, "correct": 0.0},
        "other": {"loss_nat": 0.0, "bytes": 0.0, "tokens": 0.0, "correct": 0.0},
    }

    for batch_seq_start in range(0, total_seqs, val_batch_seqs):
        batch_seq_end = min(batch_seq_start + val_batch_seqs, total_seqs)
        raw_start = batch_seq_start * seq_len
        raw_end = batch_seq_end * seq_len + 1
        chunk = val_tokens[raw_start:raw_end]
        x_np = chunk[:-1].reshape(-1, seq_len)
        y_np = chunk[1:].reshape(-1, seq_len)
        x = mx.array(x_np, dtype=mx.int32)
        tgt = mx.array(y_np, dtype=mx.int32).reshape(-1)
        logits = forward_model_logits(args, model, x).astype(mx.float32)
        probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(mx.maximum(probs, 1e-9))
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_probs = probs.reshape(-1, probs.shape[-1])
        flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
        token_loss = nn.losses.cross_entropy(flat_logits, tgt, reduction="none").astype(mx.float32)
        pred = mx.argmax(flat_logits, axis=-1)
        entropy = -mx.sum(flat_probs * flat_log_probs, axis=-1)
        top1_prob = mx.max(flat_probs, axis=-1)
        correct = (pred == tgt).astype(mx.float32)
        mx.eval(token_loss, entropy, top1_prob, correct)

        token_loss_np = np.array(token_loss, dtype=np.float32, copy=False)
        entropy_np = np.array(entropy, dtype=np.float32, copy=False)
        top1_prob_np = np.array(top1_prob, dtype=np.float32, copy=False)
        correct_np = np.array(correct, dtype=np.float32, copy=False)
        prev_ids = x_np.reshape(-1)
        tgt_ids = y_np.reshape(-1)
        bytes_np = base_bytes_lut[tgt_ids].astype(np.int16, copy=True)
        leading_space_mask = has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
        bytes_np += leading_space_mask.astype(np.int16, copy=False)
        byte1_mask = bytes_np == 1
        byte2plus_mask = bytes_np >= 2
        other_mask = ~leading_space_mask

        total_tokens += int(token_loss_np.size)
        total_bytes += float(bytes_np.astype(np.float64).sum())
        total_correct += float(correct_np.astype(np.float64).sum())
        total_loss_nat += float(token_loss_np.astype(np.float64).sum())
        total_entropy_nat += float(entropy_np.astype(np.float64).sum())
        total_top1_prob += float(top1_prob_np.astype(np.float64).sum())
        total_hard += float((entropy_np >= entropy_threshold).astype(np.float64).sum())

        for name, mask in (
            ("byte1", byte1_mask),
            ("byte2plus", byte2plus_mask),
            ("leading_space_cont", leading_space_mask),
            ("other", other_mask),
        ):
            if not np.any(mask):
                continue
            mask64 = mask.astype(np.float64, copy=False)
            slice_stats[name]["loss_nat"] += float((token_loss_np.astype(np.float64) * mask64).sum())
            slice_stats[name]["bytes"] += float((bytes_np.astype(np.float64) * mask64).sum())
            slice_stats[name]["tokens"] += float(mask64.sum())
            slice_stats[name]["correct"] += float((correct_np.astype(np.float64) * mask64).sum())

    metrics: dict[str, object] = {
        "label": label,
        "tokens": total_tokens,
        "bytes": total_bytes,
        "accuracy": (total_correct / total_tokens) if total_tokens > 0 else None,
        "mean_entropy_bits": ((total_entropy_nat / total_tokens) / math.log(2.0)) if total_tokens > 0 else None,
        "mean_top1_prob": (total_top1_prob / total_tokens) if total_tokens > 0 else None,
        "hard_entropy_fraction": (total_hard / total_tokens) if total_tokens > 0 else None,
        "probe_val_loss": (total_loss_nat / total_tokens) if total_tokens > 0 else None,
        "probe_val_bpb": ((total_loss_nat / math.log(2.0)) / total_bytes) if total_bytes > 0 else None,
        "slices": {},
    }
    for name, stats in slice_stats.items():
        tokens = stats["tokens"]
        bytes_sum = stats["bytes"]
        metrics["slices"][name] = {
            "tokens": int(round(tokens)),
            "bytes": bytes_sum,
            "accuracy": (stats["correct"] / tokens) if tokens > 0 else None,
            "bpb": ((stats["loss_nat"] / math.log(2.0)) / bytes_sum) if bytes_sum > 0 else None,
        }
    if log_fn is not None and total_tokens > 0:
        log_fn(
            f"{label}_summary tokens:{total_tokens} "
            f"acc:{metrics['accuracy']:.4f} "
            f"entropy_bits:{metrics['mean_entropy_bits']:.4f} "
            f"top1_prob:{metrics['mean_top1_prob']:.4f} "
            f"hard_frac:{metrics['hard_entropy_fraction']:.4f} "
            f"probe_bpb:{metrics['probe_val_bpb']:.4f}"
        )
    return metrics


def build_bpb_token_weights(
    input_ids: mx.array,
    target_ids: mx.array,
    base_bytes_lut_mx: mx.array,
    has_leading_space_lut_mx: mx.array,
    is_boundary_token_lut_mx: mx.array,
) -> mx.array:
    weights = base_bytes_lut_mx[target_ids].astype(mx.float32)
    leading_space = has_leading_space_lut_mx[target_ids]
    prev_is_boundary = is_boundary_token_lut_mx[input_ids]
    return weights + mx.logical_and(leading_space, mx.logical_not(prev_is_boundary)).astype(mx.float32)

# -----------------------------
# TRAINING
# -----------------------------

def clip_grad_tree(grads_tree: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return grads_tree
    flat = dict(tree_flatten(grads_tree))
    total_sq = 0.0
    for grad in flat.values():
        total_sq += float(np.sum(np.square(_np_float32(grad)), dtype=np.float64))
    if total_sq <= 0.0:
        return grads_tree
    total_norm = math.sqrt(total_sq)
    if total_norm <= max_norm:
        return grads_tree
    scale = max_norm / (total_norm + 1e-12)
    return tree_unflatten([(k, g * scale) for k, g in flat.items()])


def main() -> None:
    # ==============================================================================
    # TOKENIZER + VALIDATION METRIC SETUP
    # ==============================================================================
    args = Hyperparameters()
    apply_runtime_mode_overrides(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open("a", encoding="utf-8") as f:
            print(msg, file=f)

    code = Path(__file__).read_text(encoding="utf-8")
    log(code, console=False)
    log("=" * 100, console=False)
    log(f"Running Python {sys.version}", console=False)
    log(f"Running MLX {mx.__version__}", console=False)
    log("=" * 100, console=False)

    stage_ms: dict[str, float] = {}
    train_tok_s_samples: list[float] = []
    train_step_ms_samples: list[float] = []
    artifact_variants: list[dict[str, object]] = []
    run_status = "ok"
    failure_kind = ""
    failure_detail = ""
    final_val_loss: float | None = None
    final_val_bpb: float | None = None
    final_quant_val_loss: float | None = None
    final_quant_val_bpb: float | None = None
    final_quant_eval_ms: float | None = None
    float_probe_metrics: dict[str, object] = {}
    quant_probe_metrics: dict[str, object] = {}
    raw_model_bytes: int | None = None
    quant_serialized_bytes: int | None = None
    counted_code_bytes: int | None = None
    quant_stats: dict[str, int] = {}
    step = 0
    train_time_ms = 0.0
    stop_after_step: int | None = None
    dataset_name = ""
    actual_train_files = 0
    expected_train_files: int | None = None
    train_loader: TokenLoader | None = None
    trainable_names: list[str] = []
    trainable_include_patterns: tuple[str, ...] = ()
    trainable_exclude_patterns: tuple[str, ...] = ()
    n_trainable_params = 0
    quant_train_rounds = 0
    quant_train_last_payload_bytes: int | None = None
    mlx_device_info: dict[str, object] = {}
    mlx_governor_config: dict[str, int | None] = {
        "memory_limit_bytes": None,
        "cache_limit_bytes": None,
        "wired_limit_bytes": None,
    }
    mlx_memory_snapshots: dict[str, dict[str, int]] = {}

    try:
        setup_t0 = time.perf_counter()
        if args.coil_mode not in {"none", "clf", "pctm", "clf_pctm"}:
            raise ValueError(f"COIL_MODE must be one of none|clf|pctm|clf_pctm, got {args.coil_mode!r}")
        if args.scout_mode not in {"none", "fast", "adapter"}:
            raise ValueError(f"SCOUT_MODE must be one of none|fast|adapter, got {args.scout_mode!r}")
        if args.host_token_dtype not in HOST_TOKEN_DTYPE_MAP:
            raise ValueError(
                f"HOST_TOKEN_DTYPE must be one of {','.join(sorted(HOST_TOKEN_DTYPE_MAP))}, got {args.host_token_dtype!r}"
            )
        mlx_device_info = dict(mx.device_info(mx.default_device()))
        if args.mlx_memory_limit_gb > 0:
            mlx_governor_config["memory_limit_bytes"] = gb_to_bytes(args.mlx_memory_limit_gb)
            mx.set_memory_limit(mlx_governor_config["memory_limit_bytes"])
        if args.mlx_cache_limit_gb > 0:
            mlx_governor_config["cache_limit_bytes"] = gb_to_bytes(args.mlx_cache_limit_gb)
            mx.set_cache_limit(mlx_governor_config["cache_limit_bytes"])
        if args.mlx_wired_limit_gb > 0:
            mlx_governor_config["wired_limit_bytes"] = gb_to_bytes(args.mlx_wired_limit_gb)
            mx.set_wired_limit(mlx_governor_config["wired_limit_bytes"])
        if args.mlx_memory_telemetry:
            mx.reset_peak_memory()
        if args.train_shard_prefetch < 0:
            raise ValueError(f"TRAIN_SHARD_PREFETCH must be >= 0, got {args.train_shard_prefetch}")
        if args.coil_prime_window < 3:
            raise ValueError(f"COIL_PRIME_WINDOW must be >= 3, got {args.coil_prime_window}")
        if args.coil_tap_count <= 0:
            raise ValueError(f"COIL_TAP_COUNT must be >= 1, got {args.coil_tap_count}")
        if args.coil_sparse_topk < 0:
            raise ValueError(f"COIL_SPARSE_TOPK must be >= 0, got {args.coil_sparse_topk}")
        if args.qsparse_topk < 0:
            raise ValueError(f"QSPARSE_TOPK must be >= 0, got {args.qsparse_topk}")
        if args.qsparse_last_n < 0:
            raise ValueError(f"QSPARSE_LAST_N must be >= 0, got {args.qsparse_last_n}")
        if args.rbf_attn_last_n < 0:
            raise ValueError(f"RBF_ATTN_LAST_N must be >= 0, got {args.rbf_attn_last_n}")
        if args.monarch_last_n < 0:
            raise ValueError(f"MONARCH_LAST_N must be >= 0, got {args.monarch_last_n}")
        if args.monarch_preferred_blocks < 0:
            raise ValueError(f"MONARCH_PREFERRED_BLOCKS must be >= 0, got {args.monarch_preferred_blocks}")
        if args.coil_pctm_mode not in {"weighted", "circulant"}:
            raise ValueError(
                f"COIL_PCTM_MODE must be one of weighted|circulant, got {args.coil_pctm_mode!r}"
            )
        if args.coil_pctm_basis_count <= 0:
            raise ValueError(f"COIL_PCTM_BASIS_COUNT must be >= 1, got {args.coil_pctm_basis_count}")
        if args.coil_probe_mode not in {"none", "coherence"}:
            raise ValueError(
                f"COIL_PROBE_MODE must be one of none|coherence, got {args.coil_probe_mode!r}"
            )
        if args.coil_act_pair_count < 0:
            raise ValueError(f"COIL_ACT_PAIR_COUNT must be >= 0, got {args.coil_act_pair_count}")
        if args.hrc_pass_role_mode not in {"none", "edge2", "phase4", "phase5"}:
            raise ValueError(
                f"HRC_PASS_ROLE_MODE must be one of none|edge2|phase4|phase5, got {args.hrc_pass_role_mode!r}"
            )
        if args.hrc_pass_embed_mode not in {"shared", "peer", "palindrome", "palindrome_peer", "block", "block_peer"}:
            raise ValueError(
                "HRC_PASS_EMBED_MODE must be one of shared|peer|palindrome|palindrome_peer|block|block_peer, "
                f"got {args.hrc_pass_embed_mode!r}"
            )
        args.hrc_depth_schedule_mode = normalize_hrc_depth_schedule_mode(args.hrc_depth_schedule_mode)
        if args.hrc_depth_schedule_mode not in HRC_DEPTH_SCHEDULE_MODES:
            raise ValueError(
                "HRC_DEPTH_SCHEDULE_MODE must be one of cycle|palindrome|edge_palindrome|anchored_palindrome|recursive_palindrome|transition_recursive_palindrome|transition_recursive_cycle|prime_skip_superloop|sequential_prime_cycle, "
                f"got {args.hrc_depth_schedule_mode!r}"
            )
        if args.hrc_route_repeats <= 0:
            raise ValueError(f"HRC_ROUTE_REPEATS must be >= 1, got {args.hrc_route_repeats}")
        if args.hrc_recursive_core_start <= 0:
            raise ValueError(
                f"HRC_RECURSIVE_CORE_START must be >= 1, got {args.hrc_recursive_core_start}"
            )
        if args.hrc_depth_adapter_tie_mode not in {"none", "palindrome", "block"}:
            raise ValueError(
                f"HRC_DEPTH_ADAPTER_TIE_MODE must be one of none|palindrome|block, got {args.hrc_depth_adapter_tie_mode!r}"
            )
        if args.hrc_route_phase_init_std < 0.0:
            raise ValueError(f"HRC_ROUTE_PHASE_INIT_STD must be >= 0, got {args.hrc_route_phase_init_std}")
        if args.hrc_loop_index_dim < 0:
            raise ValueError(f"HRC_LOOP_INDEX_DIM must be >= 0, got {args.hrc_loop_index_dim}")
        if args.hrc_pass_role_init_std < 0.0:
            raise ValueError(f"HRC_PASS_ROLE_INIT_STD must be >= 0, got {args.hrc_pass_role_init_std}")
        if args.hrc_council_train_mode not in {"always", "eval_only"}:
            raise ValueError(
                f"HRC_COUNCIL_TRAIN_MODE must be one of always|eval_only, got {args.hrc_council_train_mode!r}"
            )
        if args.quant_train_mode not in {"none", "roundtrip"}:
            raise ValueError(f"QUANT_TRAIN_MODE must be one of none|roundtrip, got {args.quant_train_mode!r}")
        if args.quant_train_every <= 0:
            raise ValueError(f"QUANT_TRAIN_EVERY must be >= 1, got {args.quant_train_every}")
        if not 0.0 <= args.quant_train_start_fraction <= 1.0:
            raise ValueError(
                f"QUANT_TRAIN_START_FRACTION must be in [0, 1], got {args.quant_train_start_fraction}"
            )
        if args.hrc_council_entropy_sharpness <= 0.0:
            raise ValueError(
                f"HRC_COUNCIL_ENTROPY_SHARPNESS must be > 0, got {args.hrc_council_entropy_sharpness}"
            )
        if args.delta_sidecar_gate_mode not in {"none", "entropy"}:
            raise ValueError(
                f"DELTA_SIDECAR_GATE_MODE must be one of none|entropy, got {args.delta_sidecar_gate_mode!r}"
            )
        if args.delta_sidecar_entropy_sharpness <= 0.0:
            raise ValueError(
                f"DELTA_SIDECAR_ENTROPY_SHARPNESS must be > 0, got {args.delta_sidecar_entropy_sharpness}"
            )
        if not args.tie_embeddings:
            raise NotImplementedError("train_gpt_mlx.py only supports tied embeddings")
        if not args.tokenizer_path.endswith(".model"):
            raise ValueError(f"TOKENIZER_PATH must point to a SentencePiece .model file: {args.tokenizer_path}")
        sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
        if int(sp.vocab_size()) != args.vocab_size:
            raise ValueError(
                f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
            )
        dataset_name, actual_train_files, expected_train_files = validate_dataset_tokenizer_pair(
            args.data_path,
            args.tokenizer_path,
        )
        val_tokens = load_validation_tokens(
            args.val_files,
            args.train_seq_len,
            args.val_tokens_limit,
            host_token_dtype=args.host_token_dtype,
        )

        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
            sp, args.vocab_size
        )
        base_bytes_lut_mx = mx.array(base_bytes_lut, dtype=mx.int32)
        has_leading_space_lut_mx = mx.array(has_leading_space_lut)
        is_boundary_token_lut_mx = mx.array(is_boundary_token_lut)

        # ==============================================================================
        # TRAINING SETUP
        # ==============================================================================
        mx.random.seed(args.seed)
        train_loader = TokenLoader(
            args.train_files,
            log_fn=log,
            dataset_name=dataset_name,
            prefetch_shards=args.train_shard_prefetch,
            host_token_dtype=args.host_token_dtype,
        )

        # ==============================================================================
        # MODEL + OPTIMIZER SETUP
        # ==============================================================================
        layer_mlp_mults = build_layer_mlp_mults(args)
        xsa_layer_flags = build_xsa_layer_flags(args)
        xsa_gate_layer_flags = build_xsa_gate_flags(args, xsa_layer_flags)
        parallel_residual_flags = build_parallel_residual_flags(args)
        hemisphere_layer_flags = build_hemisphere_layer_flags(args)
        if args.model_family == "hrc":
            model = HRCGPT(
                vocab_size=args.vocab_size,
                num_unique_blocks=args.num_unique_blocks,
                effective_depth=args.effective_depth,
                dim=args.model_dim,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                mlp_mult=args.mlp_mult,
                logit_chunk_tokens=args.logit_chunk_tokens,
                logit_softcap=args.logit_softcap,
                rope_base=args.rope_base,
                rope_dims=args.rope_dims,
                tied_embed_init_std=args.tied_embed_init_std,
                qk_gain_init=args.qk_gain_init,
                rbf_last_n=args.rbf_attn_last_n,
                rbf_qk_norm_enabled=args.rbf_attn_qk_norm,
                rbf_disable_rope=args.rbf_attn_disable_rope,
                ln_scale=args.ln_scale,
                activation_kind=args.activation_kind,
                activation_negative_slope=args.activation_negative_slope,
                depth_lora_rank=args.depth_lora_rank,
                qsparse_enabled=args.qsparse_enabled,
                qsparse_topk=args.qsparse_topk,
                qsparse_last_n=args.qsparse_last_n,
                qsparse_gate_init=args.qsparse_gate_init,
                monarch_enabled=args.monarch_enabled,
                monarch_last_n=args.monarch_last_n,
                monarch_preferred_blocks=args.monarch_preferred_blocks,
                monarch_gate_init=args.monarch_gate_init,
                bigram_vocab_size=args.bigram_vocab_size,
                bigram_dim=args.bigram_dim,
                smear_enabled=args.smear_enabled,
                coil_mode=args.coil_mode,
                coil_prime_window=args.coil_prime_window,
                coil_tap_count=args.coil_tap_count,
                coil_sparse_topk=args.coil_sparse_topk,
                coil_pctm_mode=args.coil_pctm_mode,
                coil_pctm_basis_count=args.coil_pctm_basis_count,
                coil_anti_enabled=args.coil_anti_enabled,
                coil_probe_mode=args.coil_probe_mode,
                coil_probe_bias_init=args.coil_probe_bias_init,
                coil_probe_scale_init=args.coil_probe_scale_init,
                coil_act_enabled=args.coil_act_enabled,
                coil_act_pair_count=args.coil_act_pair_count,
                coil_act_scale_init=args.coil_act_scale_init,
                coil_residual_gate_init=args.coil_residual_gate_init,
                ve_enabled=args.ve_enabled,
                ve_dim=args.ve_dim,
                ve_layers=args.ve_layers,
                layer_mlp_mults=layer_mlp_mults,
                parallel_residual_flags=parallel_residual_flags,
                xsa_last_n=args.xsa_last_n,
                xsa_gate_mode=args.xsa_gate_mode,
                xsa_gate_init=args.xsa_gate_init,
                hrc_mirror_mode=args.hrc_mirror_mode,
                hrc_depth_schedule_mode=args.hrc_depth_schedule_mode,
                hrc_route_repeats=args.hrc_route_repeats,
                hrc_recursive_core_start=args.hrc_recursive_core_start,
                hrc_superloop_skip_schedule=args.hrc_superloop_skip_schedule,
                hrc_council_mode=args.hrc_council_mode,
                hrc_council_train_mode=args.hrc_council_train_mode,
                hrc_council_depth_offsets=args.hrc_council_depth_offsets,
                hrc_council_conf_scale_init=args.hrc_council_conf_scale_init,
                hrc_base_peer_mode=args.hrc_base_peer_mode,
                hrc_pass_embed_enabled=args.hrc_pass_embed_enabled,
                hrc_pass_embed_init_std=args.hrc_pass_embed_init_std,
                hrc_pass_embed_mode=args.hrc_pass_embed_mode,
                hrc_pass_role_mode=args.hrc_pass_role_mode,
                hrc_pass_role_init_std=args.hrc_pass_role_init_std,
                hrc_depth_adapter_tie_mode=args.hrc_depth_adapter_tie_mode,
                hrc_route_phase_enabled=args.hrc_route_phase_enabled,
                hrc_route_phase_init_std=args.hrc_route_phase_init_std,
                hrc_loop_index_enabled=args.hrc_loop_index_enabled,
                hrc_loop_index_dim=args.hrc_loop_index_dim,
                hrc_loop_index_scale_init=args.hrc_loop_index_scale_init,
                hrc_recur_inject_enabled=args.hrc_recur_inject_enabled,
                hrc_recur_inject_log_a_init=args.hrc_recur_inject_log_a_init,
                hrc_recur_inject_log_b_init=args.hrc_recur_inject_log_b_init,
                hrc_council_hard_gate=args.hrc_council_hard_gate,
                hrc_council_entropy_threshold=args.hrc_council_entropy_threshold,
                hrc_council_entropy_sharpness=args.hrc_council_entropy_sharpness,
                hrc_attn_only_blocks=args.hrc_attn_only_blocks,
                hrc_mlp_only_blocks=args.hrc_mlp_only_blocks,
                refiner_enabled=args.refiner_enabled,
                refiner_rank=args.refiner_rank,
                refiner_steps=args.refiner_steps,
                refiner_context_window=args.refiner_context_window,
                refiner_gate_mode=args.refiner_gate_mode,
                refiner_gate_init=args.refiner_gate_init,
                refiner_entropy_threshold=args.refiner_entropy_threshold,
                refiner_entropy_sharpness=args.refiner_entropy_sharpness,
                refiner_aux_base_loss=args.refiner_aux_base_loss,
                delta_sidecar_enabled=args.delta_sidecar_enabled,
                delta_sidecar_rank=args.delta_sidecar_rank,
                delta_sidecar_context_window=args.delta_sidecar_context_window,
                delta_sidecar_gate_mode=args.delta_sidecar_gate_mode,
                delta_sidecar_gate_init=args.delta_sidecar_gate_init,
                delta_sidecar_entropy_threshold=args.delta_sidecar_entropy_threshold,
                delta_sidecar_entropy_sharpness=args.delta_sidecar_entropy_sharpness,
                delta_sidecar_scale_init=args.delta_sidecar_scale_init,
                micro_recur_last_n=args.micro_recur_last_n,
                micro_recur_steps=args.micro_recur_steps,
                micro_recur_mirror=args.micro_recur_mirror,
                micro_recur_gate_init=args.micro_recur_gate_init,
            )
        else:
            model = GPT(
                vocab_size=args.vocab_size,
                num_layers=args.num_layers,
                dim=args.model_dim,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                mlp_mult=args.mlp_mult,
                logit_chunk_tokens=args.logit_chunk_tokens,
                logit_softcap=args.logit_softcap,
                rope_base=args.rope_base,
                rope_dims=args.rope_dims,
                tied_embed_init_std=args.tied_embed_init_std,
                qk_gain_init=args.qk_gain_init,
                rbf_last_n=args.rbf_attn_last_n,
                rbf_qk_norm_enabled=args.rbf_attn_qk_norm,
                rbf_disable_rope=args.rbf_attn_disable_rope,
                ln_scale=args.ln_scale,
                smear_enabled=args.smear_enabled,
                activation_kind=args.activation_kind,
                activation_negative_slope=args.activation_negative_slope,
                qsparse_enabled=args.qsparse_enabled,
                qsparse_topk=args.qsparse_topk,
                qsparse_last_n=args.qsparse_last_n,
                qsparse_gate_init=args.qsparse_gate_init,
                monarch_enabled=args.monarch_enabled,
                monarch_last_n=args.monarch_last_n,
                monarch_preferred_blocks=args.monarch_preferred_blocks,
                monarch_gate_init=args.monarch_gate_init,
                bigram_vocab_size=args.bigram_vocab_size,
                bigram_dim=args.bigram_dim,
                coil_mode=args.coil_mode,
                coil_prime_window=args.coil_prime_window,
                coil_tap_count=args.coil_tap_count,
                coil_sparse_topk=args.coil_sparse_topk,
                coil_pctm_mode=args.coil_pctm_mode,
                coil_pctm_basis_count=args.coil_pctm_basis_count,
                coil_anti_enabled=args.coil_anti_enabled,
                coil_probe_mode=args.coil_probe_mode,
                coil_probe_bias_init=args.coil_probe_bias_init,
                coil_probe_scale_init=args.coil_probe_scale_init,
                coil_act_enabled=args.coil_act_enabled,
                coil_act_pair_count=args.coil_act_pair_count,
                coil_act_scale_init=args.coil_act_scale_init,
                coil_residual_gate_init=args.coil_residual_gate_init,
                ve_enabled=args.ve_enabled,
                ve_dim=args.ve_dim,
                ve_layers=args.ve_layers,
                layer_mlp_mults=layer_mlp_mults,
                parallel_residual_flags=parallel_residual_flags,
                hemisphere_layer_flags=hemisphere_layer_flags,
                hemisphere_mix_init=args.hemisphere_mix_init,
                xsa_layer_flags=xsa_layer_flags,
                xsa_gate_layer_flags=xsa_gate_layer_flags,
                xsa_gate_init=args.xsa_gate_init,
                council_mode=args.council_mode,
                micro_recur_last_n=args.micro_recur_last_n,
                micro_recur_steps=args.micro_recur_steps,
                micro_recur_mirror=args.micro_recur_mirror,
                micro_recur_gate_init=args.micro_recur_gate_init,
                delta_sidecar_enabled=args.delta_sidecar_enabled,
                delta_sidecar_rank=args.delta_sidecar_rank,
                delta_sidecar_context_window=args.delta_sidecar_context_window,
                delta_sidecar_gate_mode=args.delta_sidecar_gate_mode,
                delta_sidecar_gate_init=args.delta_sidecar_gate_init,
                delta_sidecar_entropy_threshold=args.delta_sidecar_entropy_threshold,
                delta_sidecar_entropy_sharpness=args.delta_sidecar_entropy_sharpness,
                delta_sidecar_scale_init=args.delta_sidecar_scale_init,
            )
        trainable_names, trainable_include_patterns, trainable_exclude_patterns = configure_trainable_parameters(
            model, args
        )
        n_trainable_params = sum(
            int(np.prod(param.shape))
            for _, param in tree_flatten(model.trainable_parameters())
        )
        opt = SplitOptimizers(model, args)

        # ==============================================================================
        # COMPILED TRAIN / EVAL FUNCTIONS (MLX)
        # ==============================================================================
        # The crucial MLX detail is capture scope: this model contains non-trainable arrays too (for example
        # inside RoPE modules), so compiling only against trainable parameters throws "uncaptured inputs".
        # Compiling the model-bound functions and capturing the full model state fixes that while still
        # returning gradients only for trainable parameters via nn.value_and_grad(...).
        def eval_loss_fn(x, y):
            if args.model_family == "hrc":
                return model.loss(x, y, use_council=True)
            return model.loss(x, y)

        def train_loss_fn(x, y):
            if not args.bpb_weighted_loss:
                if args.model_family == "hrc":
                    return model.loss(x, y, use_council=(args.hrc_council_train_mode != "eval_only"))
                return model.loss(x, y)
            token_weights = build_bpb_token_weights(
                x,
                y,
                base_bytes_lut_mx,
                has_leading_space_lut_mx,
                is_boundary_token_lut_mx,
            )
            if args.model_family == "hrc":
                return model.loss(
                    x,
                    y,
                    token_weights=token_weights,
                    use_council=(args.hrc_council_train_mode != "eval_only"),
                )
            return model.loss(x, y, token_weights=token_weights)

        if args.disable_compile:
            compiled_loss = eval_loss_fn
            compiled_loss_and_grad = nn.value_and_grad(model, train_loss_fn)
        else:
            compiled_loss = mx.compile(eval_loss_fn, inputs=model.state, outputs=model.state)
            compiled_loss_and_grad = mx.compile(
                nn.value_and_grad(model, train_loss_fn),
                inputs=model.state,
                outputs=model.state,
            )

        # Print config once so logs are self-describing.
        n_params = sum(int(np.prod(p.shape)) for _, p in tree_flatten(model.parameters()))
        log(f"run_id:{args.run_id}")
        log(f"mlx_version:{mx.__version__}")
        log(
            f"mlx_async_eval:{int(args.mlx_async_eval)} mlx_memory_telemetry:{int(args.mlx_memory_telemetry)} "
            f"mlx_memory_limit_gb:{args.mlx_memory_limit_gb:.3f} "
            f"mlx_cache_limit_gb:{args.mlx_cache_limit_gb:.3f} "
            f"mlx_wired_limit_gb:{args.mlx_wired_limit_gb:.3f}"
        )
        if mlx_device_info:
            log(
                f"mlx_device:{mlx_device_info.get('device_name', 'unknown')} "
                f"arch:{mlx_device_info.get('architecture', 'unknown')} "
                f"memory_size:{mlx_device_info.get('memory_size', 0)} "
                f"recommended_wss:{mlx_device_info.get('max_recommended_working_set_size', 0)} "
                f"max_buffer_length:{mlx_device_info.get('max_buffer_length', 0)}"
            )
        maybe_log_mlx_memory(args.mlx_memory_telemetry, "post_setup", log, mlx_memory_snapshots)
        log(
            f"scout_mode:{args.scout_mode} skip_post_average_eval:{int(args.skip_post_average_eval)} "
            f"skip_probe_metrics:{int(args.skip_probe_metrics)} "
            f"skip_final_artifacts:{int(args.skip_final_artifacts)} "
            f"skip_final_quant_eval:{int(args.skip_final_quant_eval)}"
        )
        log(f"train_loader:shards pattern={args.train_files}")
        log(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.size - 1}")
        log(f"host_token_dtype:{args.host_token_dtype} train_shard_prefetch:{args.train_shard_prefetch}")
        if args.val_tokens_limit > 0:
            log(
                f"WARNING: using VAL_TOKENS_LIMIT={args.val_tokens_limit} "
                f"(effective_tokens:{val_tokens.size - 1}) for local proxy evaluation"
            )
        if expected_train_files is None:
            log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}")
        elif actual_train_files < expected_train_files:
            log(
                f"WARNING: train_loader:subset dataset:{dataset_name} "
                f"train_shards:{actual_train_files}/{expected_train_files} "
                f"new epochs will arrive sooner than the full dataset"
            )
        else:
            log(f"train_loader:dataset:{dataset_name} train_shards:{actual_train_files}/{expected_train_files}")
        log(f"tokenizer_path:{args.tokenizer_path}")
        log(
            f"trainable_mode:{args.trainable_mode} trainable_params:{n_trainable_params}/{n_params} "
            f"trainable_tensors:{len(trainable_names)}/{len(list(tree_flatten(model.parameters())))} "
            f"trainable_include:{','.join(trainable_include_patterns) or 'none'} "
            f"trainable_exclude:{','.join(trainable_exclude_patterns) or 'none'}"
        )
        if args.trainable_mode != "all" or trainable_include_patterns or trainable_exclude_patterns:
            preview = trainable_names[:32]
            suffix = "..." if len(trainable_names) > len(preview) else ""
            log(f"trainable_tensor_preview:{','.join(preview)}{suffix}")
        if args.model_family == "hrc":
            log(
                f"model_params:{n_params} model_family:{args.model_family} vocab_size:{args.vocab_size} "
                f"unique_blocks:{args.num_unique_blocks} effective_depth:{args.effective_depth} "
                f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
                f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
            )
        else:
            log(
                f"model_params:{n_params} model_family:{args.model_family} vocab_size:{args.vocab_size} layers:{args.num_layers} "
                f"dim:{args.model_dim} heads:{args.num_heads} kv_heads:{args.num_kv_heads} "
                f"seq_len:{args.train_seq_len} tie_embeddings:{args.tie_embeddings}"
            )
        log(
            f"activation_kind:{args.activation_kind} activation_negative_slope:{args.activation_negative_slope} "
            f"qsparse_enabled:{int(args.qsparse_enabled)} qsparse_topk:{args.qsparse_topk} "
            f"qsparse_last_n:{args.qsparse_last_n} qsparse_gate_init:{args.qsparse_gate_init} "
            f"monarch_enabled:{int(args.monarch_enabled)} monarch_last_n:{args.monarch_last_n} "
            f"monarch_preferred_blocks:{args.monarch_preferred_blocks} monarch_gate_init:{args.monarch_gate_init} "
            f"bigram_vocab_size:{args.bigram_vocab_size} bigram_dim:{args.bigram_dim} "
            f"coil_mode:{args.coil_mode} coil_prime_window:{args.coil_prime_window} "
            f"coil_tap_count:{args.coil_tap_count} coil_sparse_topk:{args.coil_sparse_topk} "
            f"coil_pctm_mode:{args.coil_pctm_mode} coil_pctm_basis_count:{args.coil_pctm_basis_count} "
            f"coil_anti_enabled:{int(args.coil_anti_enabled)} coil_probe_mode:{args.coil_probe_mode} "
            f"coil_act_enabled:{int(args.coil_act_enabled)} coil_act_pair_count:{args.coil_act_pair_count} "
            f"rope_dims:{args.rope_dims} ln_scale:{int(args.ln_scale)} smear_enabled:{int(args.smear_enabled)} "
            f"rbf_attn_last_n:{args.rbf_attn_last_n} rbf_attn_qk_norm:{int(args.rbf_attn_qk_norm)} "
            f"rbf_attn_disable_rope:{int(args.rbf_attn_disable_rope)} "
            f"ve_enabled:{int(args.ve_enabled)} ve_dim:{args.ve_dim} ve_layers:{args.ve_layers}"
        )
        if getattr(model, "coil", None) is not None:
            log(f"coil_taps:{','.join(str(x) for x in model.coil.tap_offsets)}")
            if getattr(model.coil, "act_pairs", None):
                log(
                    "coil_act_pairs:"
                    + ",".join(f"{left}-{right}" for left, right in getattr(model.coil, "act_pairs", []))
                )
        if args.model_family == "hrc":
            hrc_xsa_flags = build_tail_flags(args.effective_depth, args.xsa_last_n)
            shared_branch_modes = [
                "full"
                if model.shared_attn_enabled[i] and model.shared_mlp_enabled[i]
                else ("attn_only" if model.shared_attn_enabled[i] else "mlp_only")
                for i in range(args.num_unique_blocks)
            ]
            log(
                f"layer_mlp_mult_schedule:{','.join(f'{v:.3f}' for v in layer_mlp_mults[:args.num_unique_blocks])} "
                f"shared_parallel_flags:{','.join('1' if f else '0' for f in parallel_residual_flags[:args.num_unique_blocks])} "
                f"shared_branch_modes:{','.join(shared_branch_modes)} "
                f"virtual_block_schedule:{','.join(str(idx) for idx in model.block_schedule)} "
                f"virtual_block_repeats:{','.join(str(idx) for idx in getattr(model, 'block_repeat_schedule', []))} "
                f"virtual_route_phases:{','.join(str(idx) for idx in getattr(model, 'route_phase_schedule', []))} "
                f"virtual_route_phase_positions:{','.join(str(idx) for idx in getattr(model, 'route_phase_position_schedule', []))} "
                f"virtual_route_skip_ids:{','.join(str(idx) for idx in getattr(model, 'route_skip_id_schedule', []))} "
                f"virtual_route_skip_hops:{','.join(str(idx) for idx in getattr(model, 'route_skip_hop_schedule', []))} "
                f"pass_embed_schedule:{','.join(str(idx) for idx in model.pass_embed_schedule)} "
                f"pass_role_schedule:{','.join(str(idx) for idx in model.pass_role_schedule) or 'none'} "
                f"depth_adapter_schedule:{','.join(str(idx) for idx in model.depth_adapter_schedule)} "
                f"virtual_qsparse_layers:{','.join('1' if f else '0' for f in getattr(model, 'qsparse_layer_flags', []))} "
                f"virtual_monarch_layers:{','.join('1' if f else '0' for f in getattr(model, 'monarch_layer_flags', []))} "
                f"virtual_rbf_layers:{','.join('1' if f else '0' for f in getattr(model, 'rbf_layer_flags', []))} "
                f"virtual_xsa_layers:{','.join('1' if f else '0' for f in hrc_xsa_flags)}"
            )
            log(
                f"depth_lora_rank:{args.depth_lora_rank} hrc_mirror_mode:{args.hrc_mirror_mode} "
                f"hrc_depth_schedule_mode:{args.hrc_depth_schedule_mode} "
                f"hrc_route_repeats:{args.hrc_route_repeats} "
                f"hrc_recursive_core_start:{args.hrc_recursive_core_start} "
                f"hrc_superloop_skip_schedule:{args.hrc_superloop_skip_schedule or 'default'} "
                f"hrc_route_phase_enabled:{int(args.hrc_route_phase_enabled)} "
                f"hrc_route_phase_init_std:{args.hrc_route_phase_init_std} "
                f"hrc_council_mode:{args.hrc_council_mode} hrc_council_train_mode:{args.hrc_council_train_mode} "
                f"hrc_council_depth_offsets:{args.hrc_council_depth_offsets or 'default'} "
                f"hrc_council_conf_scale_init:{args.hrc_council_conf_scale_init} "
                f"hrc_council_hard_gate:{int(args.hrc_council_hard_gate)} "
                f"hrc_council_entropy_threshold:{args.hrc_council_entropy_threshold} "
                f"hrc_council_entropy_sharpness:{args.hrc_council_entropy_sharpness} "
                f"hrc_base_peer_mode:{args.hrc_base_peer_mode} "
                f"hrc_pass_embed_enabled:{int(args.hrc_pass_embed_enabled)} hrc_pass_embed_init_std:{args.hrc_pass_embed_init_std} "
                f"hrc_pass_embed_mode:{args.hrc_pass_embed_mode} "
                f"hrc_pass_role_mode:{args.hrc_pass_role_mode} hrc_pass_role_init_std:{args.hrc_pass_role_init_std} "
                f"hrc_depth_adapter_tie_mode:{args.hrc_depth_adapter_tie_mode} "
                f"hrc_loop_index_enabled:{int(args.hrc_loop_index_enabled)} "
                f"hrc_loop_index_dim:{args.hrc_loop_index_dim} "
                f"hrc_loop_index_scale_init:{args.hrc_loop_index_scale_init} "
                f"hrc_recur_inject_enabled:{int(args.hrc_recur_inject_enabled)} "
                f"hrc_recur_inject_log_a_init:{args.hrc_recur_inject_log_a_init} "
                f"hrc_recur_inject_log_b_init:{args.hrc_recur_inject_log_b_init} "
                f"hrc_attn_only_blocks:{args.hrc_attn_only_blocks or 'none'} "
                f"hrc_mlp_only_blocks:{args.hrc_mlp_only_blocks or 'none'} "
                f"delta_sidecar_enabled:{int(args.delta_sidecar_enabled)} "
                f"delta_sidecar_rank:{args.delta_sidecar_rank} "
                f"delta_sidecar_context_window:{args.delta_sidecar_context_window} "
                f"delta_sidecar_gate_mode:{args.delta_sidecar_gate_mode} "
                f"delta_sidecar_gate_init:{args.delta_sidecar_gate_init} "
                f"delta_sidecar_entropy_threshold:{args.delta_sidecar_entropy_threshold} "
                f"delta_sidecar_entropy_sharpness:{args.delta_sidecar_entropy_sharpness} "
                f"delta_sidecar_scale_init:{args.delta_sidecar_scale_init} "
                f"micro_recur_last_n:{args.micro_recur_last_n} micro_recur_steps:{args.micro_recur_steps} "
                f"micro_recur_mirror:{int(args.micro_recur_mirror)} micro_recur_gate_init:{args.micro_recur_gate_init}"
            )
        else:
            log(
                f"layer_mlp_mult_schedule:{','.join(f'{v:.3f}' for v in layer_mlp_mults)} "
                f"parallel_residual_layers:{','.join('1' if f else '0' for f in parallel_residual_flags)} "
                f"hemisphere_layers:{','.join('1' if f else '0' for f in hemisphere_layer_flags)} "
                f"rbf_layers:{','.join('1' if f else '0' for f in getattr(model, 'rbf_layer_flags', []))} "
                f"xsa_layers:{','.join('1' if f else '0' for f in xsa_layer_flags)} "
                f"qsparse_layers:{','.join('1' if f else '0' for f in getattr(model, 'qsparse_layer_flags', []))} "
                f"monarch_layers:{','.join('1' if f else '0' for f in getattr(model, 'monarch_layer_flags', []))} "
                f"xsa_gate_layers:{','.join('1' if f else '0' for f in xsa_gate_layer_flags)}"
            )
            log(
                f"council_mode:{args.council_mode} hemisphere_mix_init:{args.hemisphere_mix_init} "
                f"delta_sidecar_enabled:{int(args.delta_sidecar_enabled)} delta_sidecar_rank:{args.delta_sidecar_rank} "
                f"delta_sidecar_context_window:{args.delta_sidecar_context_window} "
                f"delta_sidecar_gate_mode:{args.delta_sidecar_gate_mode} "
                f"delta_sidecar_gate_init:{args.delta_sidecar_gate_init} "
                f"delta_sidecar_entropy_threshold:{args.delta_sidecar_entropy_threshold} "
                f"delta_sidecar_entropy_sharpness:{args.delta_sidecar_entropy_sharpness} "
                f"delta_sidecar_scale_init:{args.delta_sidecar_scale_init} "
                f"micro_recur_last_n:{args.micro_recur_last_n} micro_recur_steps:{args.micro_recur_steps} "
                f"micro_recur_mirror:{int(args.micro_recur_mirror)} micro_recur_gate_init:{args.micro_recur_gate_init}"
            )
        log(
            f"iterations:{args.iterations} train_batch_tokens:{args.train_batch_tokens} grad_accum_steps:{args.grad_accum_steps} "
            f"microbatch_tokens:{args.microbatch_tokens} microbatch_batch_size:{args.microbatch_tokens // args.train_seq_len} "
            f"val_batch_size:{args.val_batch_size} "
            f"warmup_steps:{args.warmup_steps} max_wallclock_seconds:{args.max_wallclock_seconds:.3f}"
        )
        log(f"mlx_max_microbatch_tokens:{args.mlx_max_microbatch_tokens}")
        log(
            f"optimizer:muon+adam trainable_keys:{len(opt.trainable_keys)} "
            f"muon_matrix_params:{len(opt.matrix_keys)} scalar_params:{len(opt.scalar_keys)} "
            f"embed_lr:{args.tied_embed_lr} "
            f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
            f"muon_momentum:{args.muon_momentum} muon_steps:{args.muon_backend_steps}"
        )
        log(f"ema_enabled:{int(args.ema_enabled)} ema_decay:{args.ema_decay}")
        log(
            f"swa_enabled:{int(args.swa_enabled)} swa_start_fraction:{args.swa_start_fraction:.3f} "
            f"swa_every:{args.swa_every} ema_swa_blend:{args.ema_swa_blend:.3f}"
        )
        log(
            "quant_clip_percentiles "
            f"default:{INT8_CLIP_PERCENTILE:.8f} attn:{INT8_CLIP_PERCENTILE_ATTN:.8f} "
            f"mlp:{INT8_CLIP_PERCENTILE_MLP:.8f} embed:{INT8_CLIP_PERCENTILE_EMBED:.8f}"
        )
        log(
            f"int8_keep_float_fp16_patterns:{','.join(INT8_KEEP_FLOAT_FP16_NAME_PATTERNS) or 'none'} "
            f"int8_keep_float_fp32_patterns:{','.join(INT8_KEEP_FLOAT_FP32_NAME_PATTERNS) or 'none'}"
        )
        log(
            f"quant_train_mode:{args.quant_train_mode} "
            f"quant_train_start_fraction:{args.quant_train_start_fraction:.3f} "
            f"quant_train_every:{args.quant_train_every}"
        )
        log(f"bpb_weighted_loss:{int(args.bpb_weighted_loss)}")
        log(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
        log(f"compute_dtype:{COMPUTE_DTYPE} compile:{int(not args.disable_compile)}")
        first_attn_dtype = next(
            (block.attn.c_q.weight.dtype for block in model.blocks if getattr(block, "attn", None) is not None),
            "none",
        )
        log(
            f"dtypes tok_emb:{model.tok_emb.weight.dtype} "
            f"linear_weight:{first_attn_dtype} "
            f"skip_weights:{model.skip_weights.dtype}"
        )
        log(f"artifact_codecs:{args.artifact_codecs} primary_artifact_codec:{args.primary_artifact_codec}")
        stage_ms["setup"] = 1000.0 * (time.perf_counter() - setup_t0)

        # ==============================================================================
        # TRAINING LOOP
        # ==============================================================================
        train_batch_plan = microbatch_plan(
            args.microbatch_tokens,
            args.train_seq_len,
            args.mlx_max_microbatch_tokens,
        )
        log(f"microbatch_plan:{','.join(f'{tokens}x{weight:.4f}' for tokens, weight in train_batch_plan)}")
        warmup_t0 = time.perf_counter()
        if args.warmup_steps > 0:
            # Warmup should only prime MLX compile/allocation paths. Updating parameters here forces us
            # to snapshot and restore model/optimizer state, which is expensive on unified-memory Macs.
            # Instead we run the real train shapes, force the loss/grads to materialize, and then reset
            # the loader so measured training still starts from the true init and token window.
            for warmup_step in range(args.warmup_steps):
                accum: dict[str, mx.array] | None = None
                warmup_loss = mx.array(0.0, dtype=mx.float32)
                grad_scale = 1.0 / args.grad_accum_steps
                for _ in range(args.grad_accum_steps):
                    warmup_loss, grads = loss_and_grad_chunked(
                        args,
                        train_loader,
                        compiled_loss_and_grad,
                        train_batch_plan,
                    )
                    accum = accumulate_flat_grads(accum, grads, grad_scale, tree_flatten)
                if args.mlx_async_eval:
                    mx.async_eval(warmup_loss, accum)
                else:
                    mx.eval(warmup_loss, accum)
                mx.synchronize()
                if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                    log(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")

            # Prime the standalone eval graph once too. It is compiled separately from value_and_grad.
            val_batch_tokens = args.val_batch_size // args.grad_accum_steps
            if val_batch_tokens < args.train_seq_len:
                raise ValueError(
                    "VAL_BATCH_SIZE must provide at least one sequence; "
                    f"got VAL_BATCH_SIZE={args.val_batch_size}, GRAD_ACCUM_STEPS={args.grad_accum_steps}, "
                    f"TRAIN_SEQ_LEN={args.train_seq_len}"
                )
            warm_val_seqs = min(val_batch_tokens // args.train_seq_len, (val_tokens.size - 1) // args.train_seq_len)
            warm_chunk = val_tokens[: warm_val_seqs * args.train_seq_len + 1]
            x_val = mx.array(warm_chunk[:-1].reshape(-1, args.train_seq_len), dtype=mx.int32)
            y_val = mx.array(warm_chunk[1:].reshape(-1, args.train_seq_len), dtype=mx.int32)
            warm_val_loss = compiled_loss(x_val, y_val)
            mx.eval(warm_val_loss)
            mx.synchronize()

            train_loader.close()
            train_loader = TokenLoader(
                args.train_files,
                log_fn=log,
                dataset_name=dataset_name,
                prefetch_shards=args.train_shard_prefetch,
                host_token_dtype=args.host_token_dtype,
            )
        maybe_log_mlx_memory(args.mlx_memory_telemetry, "post_warmup", log, mlx_memory_snapshots)
        stage_ms["warmup"] = 1000.0 * (time.perf_counter() - warmup_t0)
        ema_state: dict[str, mx.array] | None = None
        if args.ema_enabled:
            ema_state = {
                k: mx.array(np.array(v.astype(mx.float32), dtype=np.float32, copy=True))
                for k, v in tree_flatten(model.trainable_parameters())
            }
        swa_state: dict[str, mx.array] | None = None
        swa_count = 0
        swa_start_step = max(1, int(round(args.iterations * max(min(args.swa_start_fraction, 1.0), 0.0))))
        quant_train_start_step = max(1, int(round(args.iterations * args.quant_train_start_fraction)))
        quant_train_active_logged = False
        if args.swa_enabled:
            swa_state = {
                k: mx.array(np.array(v.astype(mx.float32), dtype=np.float32, copy=True))
                for k, v in tree_flatten(model.trainable_parameters())
            }
            swa_count = 1
            log(f"swa_initialized:1 swa_start_step:{swa_start_step}")

        train_t0 = time.perf_counter()
        max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
        t0 = time.perf_counter()
        while True:
            last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)
            if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
                train_time_ms += 1000.0 * (time.perf_counter() - t0)
                # Validation always scans the same fixed full validation split.
                val_loss, val_bpb = eval_val(
                    args,
                    compiled_loss,
                    val_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                    log_fn=log,
                )
                if not math.isfinite(val_loss) or not math.isfinite(val_bpb):
                    raise FloatingPointError(
                        f"Non-finite validation metrics at step {step}: val_loss={val_loss}, val_bpb={val_bpb}"
                    )
                final_val_loss, final_val_bpb = val_loss, val_bpb
                if step % 25 == 0 or last_step:
                    log(
                        f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                        f"train_time:{train_time_ms:.0f}ms step_avg:{train_time_ms / max(step, 1):.2f}ms"
                    )
                t0 = time.perf_counter()
            if last_step:
                if stop_after_step is not None and step < args.iterations:
                    log(f"stopping_early: wallclock_cap train_time:{train_time_ms:.0f}ms step:{step}/{args.iterations}")
                break

            lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (time.perf_counter() - t0))
            step_t0 = time.perf_counter()

            accum: dict[str, mx.array] | None = None
            train_loss = mx.array(0.0, dtype=mx.float32)
            grad_scale = 1.0 / args.grad_accum_steps
            for _ in range(args.grad_accum_steps):
                loss, grads = loss_and_grad_chunked(
                    args,
                    train_loader,
                    compiled_loss_and_grad,
                    train_batch_plan,
                )
                accum = accumulate_flat_grads(accum, grads, grad_scale, tree_flatten)
                train_loss = train_loss + loss.astype(mx.float32) * grad_scale
                if args.mlx_eager_eval:
                    if args.mlx_async_eval:
                        mx.async_eval(train_loss, accum)
                    else:
                        mx.eval(train_loss, accum)  # materialize each microbatch to cap peak memory

            grads = tree_unflatten(list(accum.items()))
            grads = clip_grad_tree(grads, args.grad_clip_norm)
            train_loss_value = float(train_loss.item())
            if not math.isfinite(train_loss_value):
                raise FloatingPointError(f"Non-finite train_loss at step {step}: {train_loss_value}")

            opt.step(model, grads, step=step, lr_mul=lr_mul)
            completed_step = step + 1
            quant_train_active = (
                args.quant_train_mode == "roundtrip"
                and completed_step >= quant_train_start_step
                and (completed_step - quant_train_start_step) % args.quant_train_every == 0
            )
            if quant_train_active:
                if not quant_train_active_logged:
                    log(
                        f"quant_train:activated mode:{args.quant_train_mode} "
                        f"start_step:{quant_train_start_step} every:{args.quant_train_every}"
                    )
                    quant_train_active_logged = True
                flat_params_latest = dict(tree_flatten(model.parameters()))
                projected_flat, quant_train_stats = quantize_dequantize_flat_state_int8(flat_params_latest)
                model.update(tree_unflatten(list(projected_flat.items())))
                quant_train_rounds += 1
                quant_train_last_payload_bytes = quant_train_stats["int8_payload_bytes"]
                mx.eval(*projected_flat.values())
            flat_params = dict(tree_flatten(model.trainable_parameters()))
            if ema_state is not None:
                keep: dict[str, mx.array] = {}
                keep_frac = 1.0 - args.ema_decay
                for name, tensor in flat_params.items():
                    keep[name] = ema_state[name] * args.ema_decay + tensor.astype(mx.float32) * keep_frac
                ema_state = keep
                mx.eval(*ema_state.values())
            if swa_state is not None and completed_step >= swa_start_step and (completed_step - swa_start_step) % max(args.swa_every, 1) == 0:
                next_count = swa_count + 1
                inv_count = 1.0 / float(next_count)
                keep: dict[str, mx.array] = {}
                for name, tensor in flat_params.items():
                    target = tensor.astype(mx.float32)
                    keep[name] = swa_state[name] + (target - swa_state[name]) * inv_count
                swa_state = keep
                swa_count = next_count
                mx.eval(*swa_state.values())
            mx.synchronize()

            step_ms = 1000.0 * (time.perf_counter() - step_t0)
            approx_train_time_ms = train_time_ms + 1000.0 * (time.perf_counter() - t0)
            tok_s = args.train_batch_tokens / max(step_ms / 1000.0, 1e-9)
            if not math.isfinite(tok_s) or tok_s <= 0.0:
                raise FloatingPointError(f"Invalid throughput at step {step}: tok_s={tok_s} step_ms={step_ms}")

            train_step_ms_samples.append(step_ms)
            train_tok_s_samples.append(tok_s)
            step += 1
            if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None):
                msg = (
                    f"step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} "
                    f"train_time:{approx_train_time_ms:.0f}ms step_avg:{approx_train_time_ms / step:.2f}ms tok_s:{tok_s:.0f}"
                )
                if quant_train_active:
                    msg += (
                        f" quant_train:1 quant_rounds:{quant_train_rounds}"
                        f" quant_payload:{quant_train_last_payload_bytes}"
                    )
                log(msg)
            if max_wallclock_ms is not None and stop_after_step is None and approx_train_time_ms >= max_wallclock_ms:
                stop_after_step = step
        maybe_log_mlx_memory(args.mlx_memory_telemetry, "post_train", log, mlx_memory_snapshots)
        stage_ms["train"] = 1000.0 * (time.perf_counter() - train_t0)

        # ==============================================================================
        # FINAL SERIALIZATION + QUANTIZED ROUNDTRIP EVAL
        # ==============================================================================
        export_t0 = time.perf_counter()
        export_average_mode = "latest"
        export_state: dict[str, mx.array] | None = None
        if ema_state is not None and swa_state is not None and swa_count > 0:
            blend = min(max(args.ema_swa_blend, 0.0), 1.0)
            log(f"ema_swa:blending ema_weight:{blend:.3f} swa_weight:{1.0 - blend:.3f} swa_count:{swa_count}")
            export_state = {
                k: ema_state[k] * blend + swa_state[k] * (1.0 - blend)
                for k in ema_state
            }
            export_average_mode = "ema_swa_blend"
        elif ema_state is not None:
            log("ema:applying EMA weights")
            export_state = ema_state
            export_average_mode = "ema"
        elif swa_state is not None and swa_count > 0:
            log(f"swa:applying SWA weights swa_count:{swa_count}")
            export_state = swa_state
            export_average_mode = "swa"
        if export_state is not None:
            dtype_map = {k: v.dtype for k, v in tree_flatten(model.parameters())}
            model.update(tree_unflatten([(k, export_state[k].astype(dtype_map[k])) for k in export_state]))
            if args.skip_post_average_eval:
                log(f"DIAGNOSTIC post_{export_average_mode} eval_skipped:1")
            else:
                diag_t0 = time.perf_counter()
                diag_val_loss, diag_val_bpb = eval_val(
                    args,
                    compiled_loss,
                    val_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                    log_fn=log,
                )
                final_val_loss, final_val_bpb = diag_val_loss, diag_val_bpb
                log(
                    f"DIAGNOSTIC post_{export_average_mode} val_loss:{diag_val_loss:.4f} val_bpb:{diag_val_bpb:.4f} "
                    f"eval_time:{1000.0 * (time.perf_counter() - diag_t0):.0f}ms"
                )
        if args.skip_probe_metrics:
            log("probe_metrics:skipped")
        else:
            float_probe_metrics = collect_probe_metrics(
                args,
                model,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                log_fn=log,
                label=f"probe_post_{export_average_mode}",
            )

        if args.skip_final_artifacts:
            log("final_artifacts:skipped")
            stage_ms["export"] = 1000.0 * (time.perf_counter() - export_t0)
        else:
            # We always write a raw artifact and quantized artifacts, then validate one
            # selected quantized roundtrip directly.
            out_path = out_dir / f"{args.run_id}_mlx_model.npz"
            flat_state = flatten_array_state(model.state)
            mx.savez(str(out_path), **flat_state)
            raw_model_bytes = int(out_path.stat().st_size)
            log(f"saved_model:{out_path} bytes:{raw_model_bytes}")

            quant_obj, quant_stats = quantize_state_dict_int8(flat_state)
            quant_raw = pickle.dumps(quant_obj, protocol=pickle.HIGHEST_PROTOCOL)
            quant_serialized_bytes = int(len(quant_raw))
            ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
            log(
                f"serialized_model_int8_raw:{quant_serialized_bytes} bytes "
                f"(payload:{quant_stats['int8_payload_bytes']} payload_ratio:{ratio:.2f}x)"
            )

            codecs = parse_artifact_codecs(args.artifact_codecs)
            code_path = resolve_counted_code_path(args.counted_code_path)
            counted_code_bytes = int(code_path.stat().st_size)
            log(f"counted_code_path:{code_path} counted_code_bytes:{counted_code_bytes}")
            blobs_by_codec: dict[tuple[str, int], bytes] = {}
            for codec, level in codecs:
                blob = compress_payload(quant_raw, codec, level)
                blobs_by_codec[(codec, level)] = blob
                quant_path = out_dir / f"{args.run_id}_mlx_model.int8.{codec}.ptz"
                with quant_path.open("wb") as f:
                    f.write(blob)
                quant_file_bytes = int(quant_path.stat().st_size)
                counted_total_bytes = counted_code_bytes + quant_file_bytes
                variant = {
                    "codec": codec,
                    "level": level,
                    "path": str(quant_path),
                    "compressed_model_bytes": quant_file_bytes,
                    "counted_total_bytes": counted_total_bytes,
                }
                artifact_variants.append(variant)
                log(
                    f"artifact_variant codec:{codec} level:{level} "
                    f"compressed_model_bytes:{quant_file_bytes} counted_total_bytes:{counted_total_bytes}"
                )

            selected_variant = next(
                (v for v in artifact_variants if v["codec"] == args.primary_artifact_codec),
                artifact_variants[0],
            )
            selected_key = (str(selected_variant["codec"]), int(selected_variant["level"]))
            selected_blob = blobs_by_codec[selected_key]
            selected_codec = selected_key[0]
            selected_quant_path = Path(str(selected_variant["path"]))
            canonical_quant_path = out_dir / f"{args.run_id}_mlx_model.int8.ptz"
            with canonical_quant_path.open("wb") as f:
                f.write(selected_blob)
            log(
                f"selected_artifact codec:{selected_key[0]} level:{selected_key[1]} "
                f"path:{selected_quant_path} alias:{canonical_quant_path}"
            )
            stage_ms["export"] = 1000.0 * (time.perf_counter() - export_t0)

            if args.skip_final_quant_eval:
                log("quant_roundtrip_eval:skipped")
            else:
                quant_eval_t0 = time.perf_counter()
                with selected_quant_path.open("rb") as f:
                    quant_blob_disk = f.read()
                dequant_obj = pickle.loads(decompress_payload(quant_blob_disk, selected_codec))
                quant_flat = dequantize_state_dict_int8(dequant_obj)
                model.update(tree_unflatten(list(quant_flat.items())))
                q_t0 = time.perf_counter()
                q_val_loss, q_val_bpb = eval_val(
                    args,
                    compiled_loss,
                    val_tokens,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                    log_fn=log,
                )
                q_eval_ms = 1000.0 * (time.perf_counter() - q_t0)
                if not math.isfinite(q_val_loss) or not math.isfinite(q_val_bpb):
                    raise FloatingPointError(
                        f"Non-finite quantized validation metrics: val_loss={q_val_loss}, val_bpb={q_val_bpb}"
                    )
                final_quant_val_loss = q_val_loss
                final_quant_val_bpb = q_val_bpb
                final_quant_eval_ms = q_eval_ms
                log(
                    f"final_int8_{selected_codec}_roundtrip val_loss:{q_val_loss:.4f} "
                    f"val_bpb:{q_val_bpb:.4f} eval_time:{q_eval_ms:.0f}ms"
                )
                log(f"final_int8_{selected_codec}_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")
                if args.skip_probe_metrics:
                    log("quant_probe_metrics:skipped")
                else:
                    quant_probe_metrics = collect_probe_metrics(
                        args,
                        model,
                        val_tokens,
                        base_bytes_lut,
                        has_leading_space_lut,
                        is_boundary_token_lut,
                        log_fn=log,
                        label=f"probe_post_quant_{selected_codec}",
                    )
                stage_ms["quant_eval"] = 1000.0 * (time.perf_counter() - quant_eval_t0)
        maybe_log_mlx_memory(args.mlx_memory_telemetry, "post_export", log, mlx_memory_snapshots)

    except Exception as exc:
        run_status = "failed"
        failure_detail = f"{type(exc).__name__}: {exc}"
        message_lower = failure_detail.lower()
        if "out of memory" in message_lower or "oom" in message_lower:
            failure_kind = "oom"
        elif isinstance(exc, FloatingPointError):
            failure_kind = "non_finite"
        else:
            failure_kind = type(exc).__name__.lower()
        log(f"run_failed:{failure_detail}")
        log(f"failure_kind:{failure_kind}")
        log(traceback.format_exc(), console=False)
        raise
    finally:
        if train_loader is not None:
            train_loader.close()
        stage_ms["total"] = float(sum(stage_ms.values()))
        if train_tok_s_samples:
            tok_s_np = np.array(train_tok_s_samples, dtype=np.float64)
            step_ms_np = np.array(train_step_ms_samples, dtype=np.float64)
            log(
                f"throughput_summary tok_s_mean:{tok_s_np.mean():.2f} tok_s_p50:{np.percentile(tok_s_np, 50):.2f} "
                f"tok_s_p95:{np.percentile(tok_s_np, 95):.2f} step_ms_mean:{step_ms_np.mean():.2f}"
            )
        log(
            "stage_times_ms "
            + " ".join(f"{name}:{value:.1f}" for name, value in sorted(stage_ms.items()))
        )
        log(
            f"quant_train_summary mode:{args.quant_train_mode} rounds:{quant_train_rounds} "
            f"last_payload:{quant_train_last_payload_bytes if quant_train_last_payload_bytes is not None else 'none'}"
        )
        if args.write_run_summary_json:
            summary_path = out_dir / f"{args.run_id}.summary.json"
            summary_obj = {
                "run_id": args.run_id,
                "status": run_status,
                "failure_kind": failure_kind,
                "failure_detail": failure_detail,
                "seed": args.seed,
                "dataset_name": dataset_name,
                "actual_train_shards": actual_train_files,
                "expected_train_shards": expected_train_files,
                "steps_completed": step,
                "iterations": args.iterations,
                "stopped_early": stop_after_step is not None and step < args.iterations,
                "train_time_ms": train_time_ms,
                "stage_ms": stage_ms,
                "scout_mode": args.scout_mode,
                "skip_post_average_eval": args.skip_post_average_eval,
                "skip_probe_metrics": args.skip_probe_metrics,
                "skip_final_artifacts": args.skip_final_artifacts,
                "skip_final_quant_eval": args.skip_final_quant_eval,
                "export_average_mode": export_average_mode,
                "swa_count": swa_count,
                "quant_train_mode": args.quant_train_mode,
                "quant_train_start_fraction": args.quant_train_start_fraction,
                "quant_train_every": args.quant_train_every,
                "quant_train_rounds": quant_train_rounds,
                "quant_train_last_payload_bytes": quant_train_last_payload_bytes,
                "final_val_loss": final_val_loss,
                "final_val_bpb": final_val_bpb,
                "final_quant_val_loss": final_quant_val_loss,
                "final_quant_val_bpb": final_quant_val_bpb,
                "final_quant_eval_ms": final_quant_eval_ms,
                "probe_tokens_limit": args.probe_tokens_limit,
                "probe_entropy_threshold": args.probe_entropy_threshold,
                "float_probe_metrics": float_probe_metrics,
                "quant_probe_metrics": quant_probe_metrics,
                "probe_delta": {
                    "bpb": (
                        (quant_probe_metrics.get("probe_val_bpb") or 0.0)
                        - (float_probe_metrics.get("probe_val_bpb") or 0.0)
                    ) if float_probe_metrics and quant_probe_metrics else None,
                    "accuracy": (
                        (quant_probe_metrics.get("accuracy") or 0.0)
                        - (float_probe_metrics.get("accuracy") or 0.0)
                    ) if float_probe_metrics and quant_probe_metrics else None,
                    "entropy_bits": (
                        (quant_probe_metrics.get("mean_entropy_bits") or 0.0)
                        - (float_probe_metrics.get("mean_entropy_bits") or 0.0)
                    ) if float_probe_metrics and quant_probe_metrics else None,
                },
                "n_params": n_params,
                "n_trainable_params": n_trainable_params,
                "trainable_mode": args.trainable_mode,
                "trainable_include_patterns": list(trainable_include_patterns),
                "trainable_exclude_patterns": list(trainable_exclude_patterns),
                "trainable_tensor_count": len(trainable_names),
                "trainable_tensor_preview": trainable_names[:128],
                "host_token_dtype": args.host_token_dtype,
                "train_shard_prefetch": args.train_shard_prefetch,
                "mlx_async_eval": args.mlx_async_eval,
                "mlx_memory_telemetry": args.mlx_memory_telemetry,
                "mlx_memory_limit_gb": args.mlx_memory_limit_gb,
                "mlx_cache_limit_gb": args.mlx_cache_limit_gb,
                "mlx_wired_limit_gb": args.mlx_wired_limit_gb,
                "mlx_device_info": mlx_device_info,
                "mlx_governor_config": mlx_governor_config,
                "mlx_memory_snapshots": mlx_memory_snapshots,
                "raw_model_bytes": raw_model_bytes,
                "quant_serialized_bytes": quant_serialized_bytes,
                "counted_code_bytes": counted_code_bytes,
                "selected_artifact_codec": selected_variant["codec"] if artifact_variants else None,
                "selected_artifact_level": selected_variant["level"] if artifact_variants else None,
                "artifact_counted_bytes": selected_variant["counted_total_bytes"] if artifact_variants else None,
                "artifact_model_bytes": selected_variant["compressed_model_bytes"] if artifact_variants else None,
                "artifact_variants": artifact_variants,
                "quant_stats": quant_stats,
                "num_layers": args.num_layers,
                "num_ve_layers": len(getattr(model, "ve_layer_indices", [])),
                "ve_layers": args.ve_layers,
                "qsparse_enabled": args.qsparse_enabled,
                "qsparse_topk": args.qsparse_topk,
                "qsparse_last_n": args.qsparse_last_n,
                "qsparse_gate_init": args.qsparse_gate_init,
                "qsparse_layer_flags": list(getattr(model, "qsparse_layer_flags", [])),
                "rbf_attn_last_n": args.rbf_attn_last_n,
                "rbf_attn_qk_norm": args.rbf_attn_qk_norm,
                "rbf_attn_disable_rope": args.rbf_attn_disable_rope,
                "rbf_layer_flags": list(getattr(model, "rbf_layer_flags", [])),
                "monarch_enabled": args.monarch_enabled,
                "monarch_last_n": args.monarch_last_n,
                "monarch_preferred_blocks": args.monarch_preferred_blocks,
                "monarch_gate_init": args.monarch_gate_init,
                "monarch_layer_flags": list(getattr(model, "monarch_layer_flags", [])),
                "coil_mode": args.coil_mode,
                "coil_prime_window": args.coil_prime_window,
                "coil_tap_count": args.coil_tap_count,
                "coil_sparse_topk": args.coil_sparse_topk,
                "coil_pctm_mode": args.coil_pctm_mode,
                "coil_pctm_basis_count": args.coil_pctm_basis_count,
                "coil_anti_enabled": args.coil_anti_enabled,
                "coil_probe_mode": args.coil_probe_mode,
                "coil_act_enabled": args.coil_act_enabled,
                "coil_act_pair_count": args.coil_act_pair_count,
                "coil_taps": list(getattr(model.coil, "tap_offsets", [])) if getattr(model, "coil", None) is not None else [],
                "coil_act_pairs": list(getattr(model.coil, "act_pairs", [])) if getattr(model, "coil", None) is not None else [],
                "hrc_depth_schedule_mode": args.hrc_depth_schedule_mode,
                "hrc_route_repeats": args.hrc_route_repeats,
                "hrc_recursive_core_start": args.hrc_recursive_core_start,
                "hrc_superloop_skip_schedule": args.hrc_superloop_skip_schedule,
                "hrc_route_phase_enabled": args.hrc_route_phase_enabled,
                "hrc_route_phase_init_std": args.hrc_route_phase_init_std,
                "hrc_depth_adapter_tie_mode": args.hrc_depth_adapter_tie_mode,
                "hrc_mirror_mode": args.hrc_mirror_mode,
                "hrc_council_train_mode": args.hrc_council_train_mode,
                "hrc_pass_role_mode": args.hrc_pass_role_mode,
                "hrc_pass_role_init_std": args.hrc_pass_role_init_std,
                "hrc_pass_role_schedule": list(getattr(model, "pass_role_schedule", [])),
                "hrc_route_phase_schedule": list(getattr(model, "route_phase_schedule", [])),
                "hrc_route_phase_position_schedule": list(getattr(model, "route_phase_position_schedule", [])),
                "hrc_route_skip_id_schedule": list(getattr(model, "route_skip_id_schedule", [])),
                "hrc_route_skip_hop_schedule": list(getattr(model, "route_skip_hop_schedule", [])),
                "hrc_superloop_skip_ids": list(getattr(model, "superloop_skip_ids", [])),
                "hrc_superloop_prime_width": getattr(model, "superloop_prime_width", 0),
                "hrc_superloop_shell_width": getattr(model, "superloop_shell_width", 0),
                "hrc_loop_index_enabled": args.hrc_loop_index_enabled,
                "hrc_loop_index_dim": getattr(model, "loop_index_dim", 0),
                "hrc_loop_index_scale_init": args.hrc_loop_index_scale_init,
                "hrc_block_repeat_schedule": list(getattr(model, "block_repeat_schedule", [])),
                "hrc_repeat_layer_flags": list(getattr(model, "repeat_layer_flags", [])),
                "hrc_recur_inject_enabled": args.hrc_recur_inject_enabled,
                "hrc_recur_inject_log_a_init": args.hrc_recur_inject_log_a_init,
                "hrc_recur_inject_log_b_init": args.hrc_recur_inject_log_b_init,
                "hrc_council_hard_gate": args.hrc_council_hard_gate,
                "hrc_council_entropy_threshold": args.hrc_council_entropy_threshold,
                "hrc_council_entropy_sharpness": args.hrc_council_entropy_sharpness,
                "delta_sidecar_enabled": args.delta_sidecar_enabled,
                "delta_sidecar_rank": args.delta_sidecar_rank,
                "delta_sidecar_context_window": args.delta_sidecar_context_window,
                "delta_sidecar_gate_mode": args.delta_sidecar_gate_mode,
                "delta_sidecar_gate_init": args.delta_sidecar_gate_init,
                "delta_sidecar_entropy_threshold": args.delta_sidecar_entropy_threshold,
                "delta_sidecar_entropy_sharpness": args.delta_sidecar_entropy_sharpness,
                "delta_sidecar_scale_init": args.delta_sidecar_scale_init,
                "int8_keep_float_fp16_patterns": list(INT8_KEEP_FLOAT_FP16_NAME_PATTERNS),
                "int8_keep_float_fp32_patterns": list(INT8_KEEP_FLOAT_FP32_NAME_PATTERNS),
                "throughput": {
                    "samples": len(train_tok_s_samples),
                    "tok_s_mean": float(np.mean(train_tok_s_samples)) if train_tok_s_samples else None,
                    "tok_s_p50": float(np.percentile(np.array(train_tok_s_samples), 50)) if train_tok_s_samples else None,
                    "tok_s_p95": float(np.percentile(np.array(train_tok_s_samples), 95)) if train_tok_s_samples else None,
                },
            }
            summary_path.write_text(json.dumps(summary_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            log(f"run_summary_json:{summary_path}")


if __name__ == "__main__":
    main()
