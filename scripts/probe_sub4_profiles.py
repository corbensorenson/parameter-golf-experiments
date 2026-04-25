"""Probe sub-4MB ternary profiles for params and compressed artifact size.

Examples:
  python scripts/probe_sub4_profiles.py
  python scripts/probe_sub4_profiles.py i4l8r2_d384_e128 i6l10r2_d320_e96
"""

from __future__ import annotations

import argparse
import bz2
import gzip
import io
import json
import lzma
import os
import subprocess
import sys
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_OVERRIDE_KEYS = {
    "MODEL_DIM",
    "FACTORED_EMBED_DIM",
    "NUM_HEADS",
    "NUM_KV_HEADS",
    "MLP_MULT",
    "NUM_UNIQUE_BLOCKS",
    "EFFECTIVE_DEPTH",
    "HRC_RECURSIVE_CORE_START",
    "HRC_ROUTE_REPEATS",
    "HRC_ATTN_ONLY_BLOCKS",
    "HRC_MLP_ONLY_BLOCKS",
    "HRC_FROZEN_CARRY_ENABLED",
    "HRC_FROZEN_CARRY_BLOCKS",
    "HRC_FROZEN_CARRY_BETA",
    "HRC_FROZEN_CARRY_ALPHA",
    "HRC_FROZEN_CARRY_DETACH",
    "TRAIN_TERNARY_GROUP_SIZE",
    "TRAIN_TERNARY_SCALE_STAT",
    "TRAIN_TERNARY_PARAM_DTYPE",
    "TRAIN_TERNARY_PACKED_KERNEL",
    "TRAIN_TERNARY_DENSE_KERNEL",
    "TRAIN_CASTED_LINEAR_PARAM_DTYPE",
    "KEEP_CONTROL_PARAMS_FP32",
    "RESID_MIX_ENABLED",
    "BRANCH_SCALE_ENABLED",
    "BRANCH_SCALE_KIND",
    "QUANT_TERNARY_GROUP_SIZE",
    "QUANT_TERNARY_SCALE_STAT",
    "LQER_ENABLED",
    "LQER_RANK",
    "LQER_TOP_K",
    "LQER_FACTOR_BITS",
    "LQER_ASYM_ENABLED",
    "LQER_ASYM_GROUP",
    "LQER_INCLUDE_PATTERNS",
    "LQER_EXCLUDE_PATTERNS",
    "BITNET_V2_HADAMARD",
    "LOGIT_SOFTCAP",
    "LOSS_FP32",
    "LOSS_TOKEN_STRIDE",
    "LOSS_TOKEN_RANDOM_OFFSET",
    "LOSS_VOCAB_SAMPLE_SIZE",
    "LOSS_VOCAB_SAMPLE_CORRECTION",
    "TIED_EMBED_LR",
    "MATRIX_LR",
    "SCALAR_LR",
    "LR_WARMUP_ITERS",
    "WARMDOWN_ITERS",
    "LR_MIN_SCALE",
    "MUON_NS_VARIANT",
    "MUON_WEIGHT_DECAY",
    "MUON_WEIGHT_DECAY_MODE",
    "MUON_WEIGHT_DECAY_HUBER_DELTA_SCALE",
    "MUON_WD",
    "MUON_ROW_NORMALIZE",
    "QK_GAIN_INIT",
    "DEPTH_SCALE_INIT_ENABLED",
    "DEPTH_SCALE_INIT_START",
    "DEPTH_SCALE_INIT_END",
    "ACTIVATION_KIND",
    "ATTN_QK_NORM_ENABLED",
    "BLOCK_NORM_ENABLED",
    "BIGRAM_VOCAB_SIZE",
    "BIGRAM_DIM",
    "BIGRAM_INIT_STD",
    "BIGRAM_SCALE_INIT",
    "VE_ENABLED",
    "VE_DIM",
    "VE_LAYERS",
    "SMEAR_GATE_ENABLED",
    "SMEAR_GATE_WIDTH",
    "SMEAR_GATE_MODE",
    "ATTN_OUT_GATE_ENABLED",
    "ATTN_OUT_GATE_WIDTH",
    "SPARSE_ATTN_GATE_ENABLED",
    "SPARSE_ATTN_GATE_INIT_STD",
    "SPARSE_ATTN_GATE_SCALE",
    "QSPARSE_ENABLED",
    "QSPARSE_TOPK",
    "QSPARSE_GATE_INIT",
}


def compress_payload(data: bytes, codec: str, level: int) -> bytes:
    if codec == "zlib":
        return zlib.compress(data, level)
    if codec == "gzip":
        return gzip.compress(data, compresslevel=level)
    if codec == "bz2":
        return bz2.compress(data, compresslevel=level)
    if codec == "lzma":
        return lzma.compress(data, preset=level)
    raise ValueError(f"Unsupported codec: {codec}")


def run_worker(profile: str, vocab_size: int) -> dict[str, object]:
    env = os.environ.copy()
    for key in PROFILE_OVERRIDE_KEYS:
        env.pop(key, None)
    env["SUB4_PROFILE"] = profile
    env["VOCAB_SIZE"] = str(vocab_size)
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", profile, "--vocab-size", str(vocab_size)]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"profile {profile} failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    return json.loads(proc.stdout)


def build_model(train_gpt, vocab_size: int):
    hp = train_gpt.Hyperparameters
    return train_gpt.GPT(
        vocab_size=vocab_size,
        num_layers=hp.num_layers,
        model_dim=hp.model_dim,
        num_heads=hp.num_heads,
        num_kv_heads=hp.num_kv_heads,
        mlp_mult=hp.mlp_mult,
        activation_kind=hp.activation_kind,
        activation_negative_slope=hp.activation_negative_slope,
        tie_embeddings=hp.tie_embeddings,
        factored_embed_dim=hp.factored_embed_dim,
        tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap,
        rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
        qk_norm_enabled=hp.attn_qk_norm_enabled,
        model_family=hp.model_family,
        num_unique_blocks=hp.num_unique_blocks,
        effective_depth=hp.effective_depth,
        depth_lora_rank=hp.depth_lora_rank,
        hrc_mirror_mode=hp.hrc_mirror_mode,
        hrc_depth_schedule_mode=hp.hrc_depth_schedule_mode,
        hrc_route_repeats=hp.hrc_route_repeats,
        hrc_recursive_core_start=hp.hrc_recursive_core_start,
        hrc_attn_only_blocks=hp.hrc_attn_only_blocks,
        hrc_mlp_only_blocks=hp.hrc_mlp_only_blocks,
        hrc_recur_inject_enabled=hp.hrc_recur_inject_enabled,
        hrc_loop_index_enabled=hp.hrc_loop_index_enabled,
        hrc_loop_index_dim=hp.hrc_loop_index_dim,
        hrc_pass_embed_enabled=hp.hrc_pass_embed_enabled,
        hrc_pass_embed_mode=hp.hrc_pass_embed_mode,
        hrc_pass_role_mode=hp.hrc_pass_role_mode,
        hrc_frozen_carry_enabled=hp.hrc_frozen_carry_enabled,
        hrc_frozen_carry_blocks=hp.hrc_frozen_carry_blocks,
        hrc_frozen_carry_beta=hp.hrc_frozen_carry_beta,
        hrc_frozen_carry_alpha=hp.hrc_frozen_carry_alpha,
        hrc_frozen_carry_detach=hp.hrc_frozen_carry_detach,
        block_norm_enabled=hp.block_norm_enabled,
        resid_mix_enabled=hp.resid_mix_enabled,
        branch_scale_enabled=hp.branch_scale_enabled,
        branch_scale_kind=hp.branch_scale_kind,
        train_ternary_blocks=hp.train_ternary_blocks,
        train_ternary_group_size=hp.train_ternary_group_size,
        train_ternary_forward_cache=hp.train_ternary_forward_cache,
        train_ternary_scale_stat=hp.train_ternary_scale_stat,
        train_fused_qkv=hp.train_fused_qkv,
        bitnet_v2_hadamard=hp.bitnet_v2_hadamard,
        bigram_vocab_size=hp.bigram_vocab_size,
        bigram_dim=hp.bigram_dim,
        bigram_init_std=hp.bigram_init_std,
        bigram_scale_init=hp.bigram_scale_init,
        ve_enabled=hp.ve_enabled,
        ve_dim=hp.ve_dim,
        ve_layers=hp.ve_layers,
        smear_gate_enabled=hp.smear_gate_enabled,
        smear_gate_width=hp.smear_gate_width,
        smear_gate_mode=hp.smear_gate_mode,
        attn_out_gate_enabled=hp.attn_out_gate_enabled,
        attn_out_gate_width=hp.attn_out_gate_width,
        sparse_attn_gate_enabled=hp.sparse_attn_gate_enabled,
        sparse_attn_gate_init_std=hp.sparse_attn_gate_init_std,
        sparse_attn_gate_scale=hp.sparse_attn_gate_scale,
        qsparse_enabled=hp.qsparse_enabled,
        qsparse_topk=hp.qsparse_topk,
        qsparse_gate_init=hp.qsparse_gate_init,
    )


def worker(profile: str, vocab_size: int) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch

    import train_gpt_ternary  # noqa: F401 - applies profile defaults
    import train_gpt
    from train_gpt import CausalSelfAttention, TernaryLinear

    hp = train_gpt.Hyperparameters
    param_dtype = train_gpt.resolve_param_dtype(hp, torch.float16)
    model = build_model(train_gpt, vocab_size).to(dtype=param_dtype)
    for module in model.modules():
        if (
            isinstance(module, train_gpt.CastedLinear)
            and hp.train_casted_linear_param_dtype == "fp32"
        ) or (
            isinstance(module, TernaryLinear) and hp.train_ternary_param_dtype == "fp32"
        ):
            module.float()
    if hp.keep_control_params_fp32:
        train_gpt.restore_low_dim_params_to_fp32(model)

    quant_obj, quant_stats = train_gpt.quantize_state_dict_int8(model.state_dict())
    raw_buf = io.BytesIO()
    torch.save(quant_obj, raw_buf)
    raw = raw_buf.getvalue()
    compressed = compress_payload(raw, hp.model_codec, max(0, min(hp.model_codec_level, 9)))
    code_bytes = len(train_gpt.read_counted_code_snapshot().encode("utf-8"))
    base = model
    result = {
        "profile": profile,
        "route": "".join(str(i) for i in base.block_schedule),
        "effective_depth": len(base.block_schedule),
        "params": sum(p.numel() for p in base.parameters()),
        "ternary_layers": sum(1 for module in base.modules() if isinstance(module, TernaryLinear)),
        "fused_qkv": sum(
            1 for module in base.modules() if isinstance(module, CausalSelfAttention) and module.qkv is not None
        ),
        "train_group": hp.train_ternary_group_size,
        "train_scale_stat": hp.train_ternary_scale_stat,
        "train_param_dtype": hp.train_ternary_param_dtype,
        "train_packed_kernel": hp.train_ternary_packed_kernel,
        "train_dense_kernel": hp.train_ternary_dense_kernel,
        "casted_linear_param_dtype": hp.train_casted_linear_param_dtype,
        "keep_control_params_fp32": bool(hp.keep_control_params_fp32),
        "resid_mix_enabled": bool(hp.resid_mix_enabled),
        "branch_scale_enabled": bool(hp.branch_scale_enabled),
        "branch_scale_kind": hp.branch_scale_kind,
        "quant_group": hp.quant_ternary_group_size,
        "quant_scale_stat": hp.quant_ternary_scale_stat,
        "lqer_enabled": bool(hp.lqer_enabled),
        "lqer_rank": hp.lqer_rank,
        "lqer_top_k": hp.lqer_top_k,
        "lqer_factor_bits": hp.lqer_factor_bits,
        "lqer_asym_enabled": bool(hp.lqer_asym_enabled),
        "lqer_asym_group": hp.lqer_asym_group,
        "hadamard": bool(hp.bitnet_v2_hadamard),
        "loss_fp32": bool(hp.loss_fp32),
        "loss_token_stride": hp.loss_token_stride,
        "loss_vocab_sample_size": hp.loss_vocab_sample_size,
        "tied_embed_lr": hp.tied_embed_lr,
        "matrix_lr": hp.matrix_lr,
        "scalar_lr": hp.scalar_lr,
        "lr_warmup_iters": hp.lr_warmup_iters,
        "warmdown_iters": hp.warmdown_iters,
        "lr_min_scale": hp.lr_min_scale,
        "muon_ns_variant": hp.muon_ns_variant,
        "muon_weight_decay": hp.muon_weight_decay,
        "muon_weight_decay_mode": hp.muon_weight_decay_mode,
        "muon_weight_decay_huber_delta_scale": hp.muon_weight_decay_huber_delta_scale,
        "muon_row_normalize": int(hp.muon_row_normalize),
        "qk_gain_init": hp.qk_gain_init,
        "depth_scale_init_enabled": bool(hp.depth_scale_init_enabled),
        "depth_scale_init_start": hp.depth_scale_init_start,
        "depth_scale_init_end": hp.depth_scale_init_end,
        "logit_softcap": hp.logit_softcap,
        "activation_kind": hp.activation_kind,
        "block_norm_enabled": bool(hp.block_norm_enabled),
        "attn_qk_norm_enabled": bool(hp.attn_qk_norm_enabled),
        "factored_embed_dim": hp.factored_embed_dim,
        "bigram_vocab_size": hp.bigram_vocab_size,
        "bigram_dim": hp.bigram_dim,
        "bigram_init_std": hp.bigram_init_std,
        "bigram_scale_init": hp.bigram_scale_init,
        "ve_enabled": bool(hp.ve_enabled),
        "ve_dim": hp.ve_dim,
        "ve_layers": hp.ve_layers,
        "hrc_frozen_carry_enabled": bool(getattr(base, "hrc_frozen_carry_enabled", False)),
        "hrc_frozen_carry_blocks": ",".join(
            str(v) for v in getattr(base, "hrc_frozen_carry_block_ids", ())
        ),
        "smear_gate_enabled": bool(hp.smear_gate_enabled),
        "smear_gate_width": hp.smear_gate_width,
        "smear_gate_mode": hp.smear_gate_mode,
        "attn_out_gate_enabled": bool(hp.attn_out_gate_enabled),
        "attn_out_gate_width": hp.attn_out_gate_width,
        "sparse_attn_gate_enabled": bool(hp.sparse_attn_gate_enabled),
        "sparse_attn_gate_init_std": hp.sparse_attn_gate_init_std,
        "sparse_attn_gate_scale": hp.sparse_attn_gate_scale,
        "attn_only_blocks": hp.hrc_attn_only_blocks,
        "mlp_only_blocks": hp.hrc_mlp_only_blocks,
        "raw_payload": int(quant_stats.get("export_payload_bytes", quant_stats["int8_payload_bytes"])),
        "torch_save_bytes": len(raw),
        "compressed_model_bytes": len(compressed),
        "code_bytes": code_bytes,
        "total_submission_bytes": len(compressed) + code_bytes,
        "headroom": hp.submission_size_cap_bytes - (len(compressed) + code_bytes),
        "codec": hp.model_codec,
    }
    print(json.dumps(result, sort_keys=True))


def print_table(rows: list[dict[str, object]]) -> None:
    print("| Profile | Params | Route Depth | Ternary | QKV | Groups | Variant | Raw Payload | Model Bytes | Total | Headroom |")
    print("| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    f"`{row['profile']}`",
                    str(row["params"]),
                    str(row["effective_depth"]),
                    str(row["ternary_layers"]),
                    str(row["fused_qkv"]),
                    f"{row['train_group']}/{row['quant_group']}",
                    (
                        f"scale={row['train_scale_stat']}/{row['quant_scale_stat']} "
                        f"tdtype={row['train_param_dtype']} "
                        f"pkern={row.get('train_packed_kernel', '0')} "
                        f"dkern={row.get('train_dense_kernel', '0')} "
                        f"cdtype={row.get('casted_linear_param_dtype', 'fp32')} "
                        f"ctrlfp32={int(bool(row.get('keep_control_params_fp32', True)))} "
                        f"resmix={int(bool(row.get('resid_mix_enabled', True)))} "
                        f"bscale={int(bool(row.get('branch_scale_enabled', True)))} "
                        f"bskind={row.get('branch_scale_kind', 'vector')} "
                        f"lqer={int(bool(row.get('lqer_enabled', False)))}:{row.get('lqer_rank', 0)}:{row.get('lqer_top_k', 0)} "
                        f"had={int(bool(row['hadamard']))} "
                        f"lossfp32={int(bool(row['loss_fp32']))} "
                        f"tstride={row['loss_token_stride']} "
                        f"vsample={row['loss_vocab_sample_size']} "
                        f"lr={row.get('tied_embed_lr', '?')}/{row.get('matrix_lr', '?')}/{row.get('scalar_lr', '?')} "
                        f"warm={row.get('lr_warmup_iters', '?')}/{row.get('warmdown_iters', '?')} "
                        f"minlr={row.get('lr_min_scale', 0.0)} "
                        f"ns={row.get('muon_ns_variant', 'default')} "
                        f"mwd={row.get('muon_weight_decay', 0.0)} "
                        f"wdmode={row.get('muon_weight_decay_mode', 'decoupled')} "
                        f"mrow={row.get('muon_row_normalize', 0)} "
                        f"qkgain={row.get('qk_gain_init', '?')} "
                        f"dscale={int(bool(row.get('depth_scale_init_enabled', False)))}:"
                        f"{row.get('depth_scale_init_start', '?')}-{row.get('depth_scale_init_end', '?')} "
                        f"softcap={row['logit_softcap']} "
                        f"act={row.get('activation_kind', 'relu2')} "
                        f"bnorm={int(bool(row.get('block_norm_enabled', True)))} "
                        f"qknorm={int(bool(row.get('attn_qk_norm_enabled', True)))} "
                        f"edim={row.get('factored_embed_dim', '?')} "
                        f"bigram={row.get('bigram_vocab_size', 0)}x{row.get('bigram_dim', 0)} "
                        f"binit={row.get('bigram_init_std', 0.0)}/{row.get('bigram_scale_init', 0.0)} "
                        f"ve={int(bool(row.get('ve_enabled', False)))}:{row.get('ve_dim', 0)}:{row.get('ve_layers', '') or 'none'} "
                        f"fcarry={int(bool(row.get('hrc_frozen_carry_enabled', False)))}:{row.get('hrc_frozen_carry_blocks', '') or 'none'} "
                        f"smear={int(bool(row.get('smear_gate_enabled', False)))}:{row.get('smear_gate_width', 0)}:{row.get('smear_gate_mode', 'vector')} "
                        f"attng={int(bool(row.get('attn_out_gate_enabled', False)))}:{int(bool(row.get('sparse_attn_gate_enabled', False)))}:{row.get('attn_out_gate_width', 0)} "
                        f"attn_only={row['attn_only_blocks'] or 'none'} "
                        f"mlp_only={row['mlp_only_blocks'] or 'none'}"
                    ),
                    str(row["raw_payload"]),
                    str(row["compressed_model_bytes"]),
                    str(row["total_submission_bytes"]),
                    str(row["headroom"]),
                ]
            )
            + " |"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles", nargs="*")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    if args.worker:
        if len(args.profiles) != 1:
            raise SystemExit("--worker expects exactly one profile")
        worker(args.profiles[0], args.vocab_size)
        return 0

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from train_gpt_ternary import SUB4_PROFILES

    profiles = args.profiles or list(SUB4_PROFILES)
    rows = [run_worker(profile, args.vocab_size) for profile in profiles]
    rows.sort(key=lambda item: int(item["total_submission_bytes"]))
    print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
