"""Compile and run a tiny CUDA extension against the local torch+cu126 install."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cuda126_env import configure_cuda126_env

configure_cuda126_env()
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")

import torch
from torch.utils.cpp_extension import load_inline


CPP_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

extern "C" void add_one_launch(float* x, long long n, cudaStream_t stream);

torch::Tensor add_one_cuda(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, "x must be float32");
  auto y = x.contiguous().clone();
  add_one_launch(y.data_ptr<float>(), y.numel(), at::cuda::getCurrentCUDAStream().stream());
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &add_one_cuda, "add one CUDA smoke");
}
"""

CUDA_SRC = r"""
#include <cuda_runtime.h>

__global__ void add_one_kernel(float* x, long long n) {
  long long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += 1.0f;
  }
}

extern "C" void add_one_launch(float* x, long long n, cudaStream_t stream) {
  int threads = 256;
  int blocks = (int)((n + threads - 1) / threads);
  add_one_kernel<<<blocks, threads, 0, stream>>>(x, n);
}
"""


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    build_dir = ROOT / "tmp_cuda_extensions" / "cuda126_add_one"
    build_dir.mkdir(parents=True, exist_ok=True)
    ext = load_inline(
        name="cuda126_add_one_ext",
        cpp_sources=CPP_SRC,
        cuda_sources=CUDA_SRC,
        build_directory=str(build_dir),
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    x = torch.arange(8, device="cuda", dtype=torch.float32)
    y = ext.add_one(x)
    torch.testing.assert_close(y.cpu(), torch.arange(1, 9, dtype=torch.float32))
    print("cuda126_extension_build:ok")


if __name__ == "__main__":
    main()
