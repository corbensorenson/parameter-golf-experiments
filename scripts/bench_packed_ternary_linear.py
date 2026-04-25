"""Microbenchmark experimental train-time ternary CUDA linear kernels."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cuda126_env import configure_cuda126_env

configure_cuda126_env()

from ternary_golf.layers import TernaryLinear, ternary_ste_weight


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_kernel_mode(mode: str) -> None:
    mode = mode.strip().lower()
    os.environ["TRAIN_TERNARY_PACKED_KERNEL"] = "1" if mode == "packed" else "0"
    os.environ["TRAIN_TERNARY_DENSE_KERNEL"] = "1" if mode == "cuda_dense" else "0"


def run_case(layer: TernaryLinear, x: torch.Tensor, mode: str, steps: int, warmup: int, backward: bool) -> float:
    set_kernel_mode(mode)

    def step() -> None:
        layer.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        y = layer(x)
        if backward:
            y.float().square().mean().backward()

    for _ in range(warmup):
        step()
    sync()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync()
    return 1000.0 * (time.perf_counter() - t0) / max(steps, 1)


def correctness(layer: TernaryLinear, x: torch.Tensor) -> dict[str, float]:
    set_kernel_mode("torch_dense")
    ref_layer = TernaryLinear(
        layer.in_features,
        layer.out_features,
        bias=False,
        group_size=layer.group_size,
        scale_stat=layer.scale_stat,
    ).to(device=x.device, dtype=layer.weight.dtype)
    ref_layer.weight.data.copy_(layer.weight.data)
    x_ref = x.detach().clone().requires_grad_(True)
    y_ref = ref_layer(x_ref)
    y_ref.float().square().mean().backward()

    out: dict[str, float] = {}
    for mode in ("cuda_dense", "packed"):
        set_kernel_mode(mode)
        test_layer = TernaryLinear(
            layer.in_features,
            layer.out_features,
            bias=False,
            group_size=layer.group_size,
            scale_stat=layer.scale_stat,
        ).to(device=x.device, dtype=layer.weight.dtype)
        test_layer.weight.data.copy_(layer.weight.data)
        x_test = x.detach().clone().requires_grad_(True)
        y = test_layer(x_test)
        y.float().square().mean().backward()
        sync()
        out[f"{mode}_forward_max_abs"] = float((y.float() - y_ref.float()).abs().max().item())
        out[f"{mode}_grad_x_max_abs"] = float((x_test.grad.float() - x_ref.grad.float()).abs().max().item())
        out[f"{mode}_grad_w_max_abs"] = float(
            (test_layer.weight.grad.float() - ref_layer.weight.grad.float()).abs().max().item()
        )
    return out


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    torch.manual_seed(int(os.environ.get("BENCH_SEED", "123")))
    m = int(os.environ.get("BENCH_M", "4096"))
    k = int(os.environ.get("BENCH_K", "128"))
    n = int(os.environ.get("BENCH_N", "128"))
    group_size = int(os.environ.get("BENCH_GROUP_SIZE", "256"))
    steps = int(os.environ.get("BENCH_STEPS", "100"))
    warmup = int(os.environ.get("BENCH_WARMUP", "20"))
    backward = bool(int(os.environ.get("BENCH_BACKWARD", "1")))
    dtype_name = os.environ.get("BENCH_DTYPE", "fp16").strip().lower()
    dtype = torch.float32 if dtype_name == "fp32" else (torch.bfloat16 if dtype_name == "bf16" else torch.float16)
    x = torch.randn(m, k, device="cuda", dtype=dtype, requires_grad=backward)
    layer = TernaryLinear(k, n, bias=False, group_size=group_size).to(device="cuda", dtype=dtype)
    layer.weight.data.normal_(0, 0.02)

    # Force extension build before timing.
    set_kernel_mode("packed")
    _ = layer(x[: min(m, 8)])
    set_kernel_mode("cuda_dense")
    _ = layer(x[: min(m, 8)])
    sync()

    diffs = correctness(layer, x[: min(m, 128)])
    torch_dense_ms = run_case(layer, x, mode="torch_dense", steps=steps, warmup=warmup, backward=backward)
    cuda_dense_ms = run_case(layer, x, mode="cuda_dense", steps=steps, warmup=warmup, backward=backward)
    packed_ms = run_case(layer, x, mode="packed", steps=steps, warmup=warmup, backward=backward)
    torch_dense_tokens = m / (torch_dense_ms / 1000.0)
    cuda_dense_tokens = m / (cuda_dense_ms / 1000.0)
    packed_tokens = m / (packed_ms / 1000.0)
    print(
        {
            "m": m,
            "k": k,
            "n": n,
            "group_size": group_size,
            "dtype": str(dtype).removeprefix("torch."),
            "backward": backward,
            "torch_dense_ms": round(torch_dense_ms, 4),
            "cuda_dense_ms": round(cuda_dense_ms, 4),
            "packed_ms": round(packed_ms, 4),
            "cuda_dense_speedup": round(torch_dense_ms / cuda_dense_ms, 4) if cuda_dense_ms > 0 else 0.0,
            "packed_speedup": round(torch_dense_ms / packed_ms, 4) if packed_ms > 0 else 0.0,
            "torch_dense_rows_per_sec": round(torch_dense_tokens, 1),
            "cuda_dense_rows_per_sec": round(cuda_dense_tokens, 1),
            "packed_rows_per_sec": round(packed_tokens, 1),
            **diffs,
            "reference_dense_weight_max_abs": float(
                ternary_ste_weight(layer.weight, group_size, work_dtype=dtype, scale_stat="mean").abs().max().item()
            ),
        }
    )


if __name__ == "__main__":
    main()
