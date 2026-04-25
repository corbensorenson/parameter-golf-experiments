"""Preflight for local PyTorch CUDA extension builds."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.cuda126_env import configure_cuda126_env

configure_cuda126_env()

import torch
from torch.utils.cpp_extension import CUDA_HOME


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout.strip()


print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"cpp_extension.CUDA_HOME={CUDA_HOME}")
print(f"nvcc={shutil.which('nvcc')}")
print(run(["nvcc", "--version"]).splitlines()[-1])
print(f"cl={shutil.which('cl.exe')}")
if torch.version.cuda != "12.6":
    raise SystemExit(f"Expected torch.version.cuda == 12.6, got {torch.version.cuda!r}")
if CUDA_HOME is None or "v12.6" not in str(CUDA_HOME):
    raise SystemExit(f"Expected CUDA_HOME to point at v12.6, got {CUDA_HOME!r}")
if shutil.which("nvcc") is None:
    raise SystemExit("nvcc not found on PATH")
