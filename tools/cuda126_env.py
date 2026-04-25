"""Project-local CUDA 12.6 environment helpers for PyTorch extension builds."""

from __future__ import annotations

import os
from pathlib import Path


CUDA126_HOME = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")


def configure_cuda126_env() -> Path:
    """Point this Python process at the CUDA 12.6 toolkit used by torch+cu126."""

    if not CUDA126_HOME.exists():
        raise FileNotFoundError(f"CUDA 12.6 toolkit not found at {CUDA126_HOME}")
    bin_dir = CUDA126_HOME / "bin"
    if not (bin_dir / "nvcc.exe").exists():
        raise FileNotFoundError(f"nvcc.exe not found under {bin_dir}")

    os.environ["CUDA_HOME"] = str(CUDA126_HOME)
    os.environ["CUDA_PATH"] = str(CUDA126_HOME)
    path_parts = [
        part
        for part in os.environ.get("PATH", "").split(os.pathsep)
        if part and str(bin_dir).lower() != part.lower() and r"cuda\v11.7\bin" not in part.lower()
    ]
    os.environ["PATH"] = os.pathsep.join([str(bin_dir), *path_parts])
    return CUDA126_HOME


__all__ = ["CUDA126_HOME", "configure_cuda126_env"]
