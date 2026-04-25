"""Quick synthetic training-step benchmark for the sub-4MB ternary profiles.

Set SUB4_PROFILE before running, for example:
  SUB4_PROFILE=i2l3r2_d512_e128 python scripts/bench_sub4_ternary.py
"""

from __future__ import annotations

import os
import sys
import time
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import train_gpt_ternary  # noqa: F401 - sets defaults before train_gpt import
import train_gpt
from train_gpt import CausalSelfAttention, GPT, TernaryLinear


def configure_cuda(device: torch.device) -> str:
    if device.type != "cuda":
        return "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = train_gpt.Hyperparameters.cudnn_benchmark
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    requested = train_gpt.Hyperparameters.sdp_backend
    if requested not in {"auto", "flash", "mem_efficient", "cudnn", "math"}:
        raise ValueError(f"SDP_BACKEND must be auto|flash|mem_efficient|cudnn|math, got {requested!r}")
    capability = torch.cuda.get_device_capability(device)
    flash_supported = capability[0] >= 8
    head_dim = train_gpt.Hyperparameters.model_dim // max(train_gpt.Hyperparameters.num_heads, 1)
    mem_efficient_supported_shape = (
        train_gpt.Hyperparameters.num_kv_heads == train_gpt.Hyperparameters.num_heads
        and train_gpt.Hyperparameters.model_dim % max(train_gpt.Hyperparameters.num_heads, 1) == 0
        and head_dim % 8 == 0
    )
    selected = (
        "flash"
        if flash_supported
        else ("mem_efficient" if mem_efficient_supported_shape else "math")
    ) if requested == "auto" else requested
    if selected == "flash" and not flash_supported:
        selected = "math"
    if selected == "cudnn" and capability[0] < 8:
        selected = "mem_efficient" if train_gpt.Hyperparameters.num_kv_heads == train_gpt.Hyperparameters.num_heads else "math"
    if selected == "mem_efficient" and train_gpt.Hyperparameters.num_kv_heads != train_gpt.Hyperparameters.num_heads:
        selected = "math"
    if selected == "mem_efficient" and (
        train_gpt.Hyperparameters.model_dim % max(train_gpt.Hyperparameters.num_heads, 1) != 0
        or head_dim % 8 != 0
    ):
        selected = "math"
    enable_cudnn_sdp(selected == "cudnn")
    enable_flash_sdp(selected == "flash")
    enable_mem_efficient_sdp(selected == "mem_efficient")
    enable_math_sdp(selected == "math")
    return selected


def build_model(vocab_size: int) -> GPT:
    hp = train_gpt.Hyperparameters
    return GPT(
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
    )


def build_optimizers(
    model: GPT,
    device: torch.device,
    optimizer_name: str,
    lr: float,
    cuda_graph: bool = False,
) -> list[torch.optim.Optimizer]:
    optimizer_name = optimizer_name.strip().lower()
    if optimizer_name == "adamw":
        return [
            torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                fused=(device.type == "cuda"),
                capturable=(device.type == "cuda" and cuda_graph),
            )
        ]
    if optimizer_name == "sgd":
        return [torch.optim.SGD(model.parameters(), lr=lr, foreach=(device.type == "cuda"))]
    if optimizer_name == "sgd_momentum":
        return [torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, foreach=(device.type == "cuda"))]
    if optimizer_name != "hybrid":
        raise ValueError(f"BENCH_OPTIMIZER must be adamw|hybrid|sgd|sgd_momentum, got {optimizer_name!r}")
    token_params = [model.tok_emb.weight]
    excluded = {id(p) for p in token_params}
    if model.lm_head is not None:
        excluded.add(id(model.lm_head.weight))
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad and id(p) not in excluded]
    matrix_params = [
        p
        for name, p in named
        if p.ndim == 2 and not any(pattern in name for pattern in train_gpt.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in named
        if p.ndim < 2 or any(pattern in name for pattern in train_gpt.CONTROL_TENSOR_NAME_PATTERNS)
    ]
    optimizers: list[torch.optim.Optimizer] = [
        torch.optim.Adam(model.tok_emb.parameters(), lr=lr, fused=(device.type == "cuda"))
    ]
    if matrix_params:
        muon_dtype = torch.float16 if device.type == "cuda" else torch.float32
        optimizers.append(
            train_gpt.Muon(
                matrix_params,
                lr=lr,
                momentum=train_gpt.Hyperparameters.muon_momentum,
                backend_steps=int(os.environ.get("BENCH_MUON_BACKEND_STEPS", str(train_gpt.Hyperparameters.muon_backend_steps))),
                backend_dtype=muon_dtype,
            )
        )
    if scalar_params:
        optimizers.append(torch.optim.Adam(scalar_params, lr=lr, fused=(device.type == "cuda")))
    return optimizers


def main() -> None:
    hp = train_gpt.Hyperparameters
    requested_device = os.environ.get("BENCH_DEVICE", "auto").strip().lower()
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    seq_len = int(os.environ.get("BENCH_SEQ_LEN", str(hp.train_seq_len)))
    batch_tokens = int(os.environ.get("BENCH_BATCH_TOKENS", str(hp.train_batch_tokens)))
    steps = int(os.environ.get("BENCH_STEPS", "20"))
    warmup = int(os.environ.get("BENCH_WARMUP", "5"))
    vocab_size = int(os.environ.get("VOCAB_SIZE", str(hp.vocab_size)))
    compile_model = bool(int(os.environ.get("BENCH_COMPILE", "0")))
    compile_requested = compile_model
    cuda_graph = bool(int(os.environ.get("BENCH_CUDA_GRAPH", "0")))
    optimizer_name = os.environ.get("BENCH_OPTIMIZER", "adamw")
    lr = float(os.environ.get("BENCH_LR", "1e-4"))
    phase_timing = bool(int(os.environ.get("BENCH_PHASES", "0")))
    use_loader = bool(int(os.environ.get("BENCH_USE_LOADER", "0")))
    if compile_model and device.type == "cuda" and importlib.util.find_spec("triton") is None:
        compile_model = False
    if cuda_graph and device.type != "cuda":
        cuda_graph = False
    if cuda_graph and use_loader:
        raise ValueError("BENCH_CUDA_GRAPH=1 requires BENCH_USE_LOADER=0")
    if cuda_graph and optimizer_name.strip().lower() != "adamw":
        raise ValueError("BENCH_CUDA_GRAPH=1 currently supports BENCH_OPTIMIZER=adamw")
    batch = max(batch_tokens // seq_len, 1)
    selected_sdp_backend = configure_cuda(device)
    autocast_dtype = train_gpt.resolve_autocast_dtype(hp, device)
    param_dtype = train_gpt.resolve_param_dtype(hp, autocast_dtype)

    base_model = build_model(vocab_size).to(device=device, dtype=param_dtype)
    for module in base_model.modules():
        if (
            isinstance(module, train_gpt.CastedLinear)
            and hp.train_casted_linear_param_dtype == "fp32"
        ) or (
            isinstance(module, TernaryLinear) and hp.train_ternary_param_dtype == "fp32"
        ):
            module.float()
    if hp.keep_control_params_fp32:
        train_gpt.restore_low_dim_params_to_fp32(base_model)
    model = base_model
    if compile_model:
        model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model.train()
    optimizers = build_optimizers(base_model, device, optimizer_name, lr, cuda_graph=cuda_graph)
    loader = train_gpt.DistributedTokenLoader(hp.train_files, 0, 1, device) if use_loader else None
    static_x = torch.randint(0, vocab_size, (batch, seq_len), device=device)
    static_y = torch.randint(0, vocab_size, (batch, seq_len), device=device)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()

    def current_batch() -> tuple[torch.Tensor, torch.Tensor]:
        if loader is not None:
            return loader.next_batch(batch * seq_len, seq_len, 1)
        return static_x, static_y

    def one_step(timed: bool = False) -> tuple[float, dict[str, float]]:
        timings: dict[str, float] = {}
        if timed:
            sync()
            t0 = time.perf_counter()
        x, y = current_batch()
        if timed:
            sync()
            t1 = time.perf_counter()
            timings["data_ms"] = 1000.0 * (t1 - t0)
            t0 = t1
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        if timed:
            t1 = time.perf_counter()
            timings["zero_ms"] = 1000.0 * (t1 - t0)
            t0 = t1
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(device.type == "cuda")):
            loss = model(x, y)
        if timed:
            sync()
            t1 = time.perf_counter()
            timings["forward_ms"] = 1000.0 * (t1 - t0)
            t0 = t1
        loss.backward()
        if timed:
            sync()
            t1 = time.perf_counter()
            timings["backward_ms"] = 1000.0 * (t1 - t0)
            t0 = t1
        for optimizer in optimizers:
            optimizer.step()
        if timed:
            sync()
            t1 = time.perf_counter()
            timings["optimizer_ms"] = 1000.0 * (t1 - t0)
        return float(loss.detach().float().item()), timings

    graph = None
    graph_loss = None
    if cuda_graph:
        warm_stream = torch.cuda.Stream(device=device)
        torch.cuda.current_stream(device=device).wait_stream(warm_stream)
        with torch.cuda.stream(warm_stream):
            for _ in range(max(warmup, 3)):
                one_step()
        torch.cuda.current_stream(device=device).wait_stream(warm_stream)
        graph = torch.cuda.CUDAGraph()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph):
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=True):
                graph_loss = model(static_x, static_y)
            graph_loss.backward()
            for optimizer in optimizers:
                optimizer.step()
        def graph_step() -> float:
            assert graph is not None
            assert graph_loss is not None
            graph.replay()
            return float(graph_loss.detach().float().item())
    else:
        for _ in range(warmup):
            one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss = 0.0
    phase_sums: dict[str, float] = {}
    for _ in range(steps):
        if cuda_graph:
            loss = graph_step()
        else:
            loss, timings = one_step(timed=phase_timing)
            for key, value in timings.items():
                phase_sums[key] = phase_sums.get(key, 0.0) + value
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = 1000.0 * elapsed / max(steps, 1)
    phase_avgs = {key: round(value / max(steps, 1), 3) for key, value in phase_sums.items()}
    base = base_model
    ternary_count = sum(1 for module in base.modules() if isinstance(module, TernaryLinear))
    fused_qkv_count = sum(
        1 for module in base.modules() if isinstance(module, CausalSelfAttention) and module.qkv is not None
    )
    print(
        {
            "profile": os.environ.get("SUB4_PROFILE", "i3l5r2_d384_e128"),
            "device": str(device),
            "route": "".join(str(i) for i in base.block_schedule),
            "params": sum(p.numel() for p in base.parameters()),
            "ternary_layers": ternary_count,
            "fused_qkv": fused_qkv_count,
            "train_scale_stat": hp.train_ternary_scale_stat,
            "train_param_dtype": hp.train_ternary_param_dtype,
            "train_packed_kernel": hp.train_ternary_packed_kernel,
            "train_dense_kernel": hp.train_ternary_dense_kernel,
            "casted_linear_param_dtype": hp.train_casted_linear_param_dtype,
            "keep_control_params_fp32": hp.keep_control_params_fp32,
            "block_norm_enabled": hp.block_norm_enabled,
            "attn_qk_norm_enabled": hp.attn_qk_norm_enabled,
            "resid_mix_enabled": hp.resid_mix_enabled,
            "branch_scale_enabled": hp.branch_scale_enabled,
            "branch_scale_kind": hp.branch_scale_kind,
            "hadamard": bool(hp.bitnet_v2_hadamard),
            "loss_fp32": bool(hp.loss_fp32),
            "loss_token_stride": hp.loss_token_stride,
            "loss_vocab_sample_size": hp.loss_vocab_sample_size,
            "logit_softcap": hp.logit_softcap,
            "attn_only_blocks": hp.hrc_attn_only_blocks or "",
            "mlp_only_blocks": hp.hrc_mlp_only_blocks or "",
            "batch_tokens": batch * seq_len,
            "ms_per_step": round(ms, 3),
            "tokens_per_sec": round((batch * seq_len) / (ms / 1000.0), 1),
            "loss": round(loss, 4),
            "compiled": compile_model,
            "compile_requested": compile_requested,
            "cuda_graph": cuda_graph,
            "optimizer": optimizer_name,
            "use_loader": use_loader,
            "phase_ms": phase_avgs,
            "autocast_dtype": train_gpt.dtype_name(autocast_dtype),
            "param_dtype": train_gpt.dtype_name(param_dtype),
            "sdp_backend": selected_sdp_backend,
        }
    )


if __name__ == "__main__":
    main()
