#!/usr/bin/env python3
"""
Create a deterministic local proxy split from an official cached FineWeb export.

The proxy split keeps all validation shards and only a prefix of training shards,
so local iteration remains directionally aligned while being cheaper to rerun.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def sorted_shards(root: Path, pattern: str) -> list[Path]:
    return sorted(root.glob(pattern))


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            mode = "copy"
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"unknown mode: {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a local proxy FineWeb split")
    parser.add_argument(
        "--source-dataset",
        type=Path,
        required=True,
        help="Source dataset directory containing fineweb_train_*.bin and fineweb_val_*.bin",
    )
    parser.add_argument(
        "--dest-dataset",
        type=Path,
        required=True,
        help="Destination dataset directory to create/update",
    )
    parser.add_argument(
        "--train-shards",
        type=int,
        required=True,
        help="Number of training shards to keep in the proxy split",
    )
    parser.add_argument(
        "--mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="Materialization mode. hardlink is fastest and falls back to copy on failure.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Optional JSON output path for split metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = args.source_dataset.resolve()
    dst = args.dest_dataset.resolve()

    if not src.is_dir():
        raise FileNotFoundError(f"source dataset not found: {src}")
    if args.train_shards < 0:
        raise ValueError("--train-shards must be non-negative")

    train_shards = sorted_shards(src, "fineweb_train_*.bin")
    val_shards = sorted_shards(src, "fineweb_val_*.bin")
    if not train_shards:
        raise FileNotFoundError(f"no training shards found in {src}")
    if not val_shards:
        raise FileNotFoundError(f"no validation shards found in {src}")
    if args.train_shards > len(train_shards):
        raise ValueError(
            f"requested {args.train_shards} train shards but source only has {len(train_shards)}"
        )

    selected_train = train_shards[: args.train_shards]
    if any("fineweb_val_" in p.name for p in selected_train):
        raise ValueError("proxy split construction would include validation shards in train set")

    dst.mkdir(parents=True, exist_ok=True)
    existing_bins = list(dst.glob("fineweb_*.bin"))
    for path in existing_bins:
        path.unlink()

    for shard in selected_train:
        link_or_copy(shard, dst / shard.name, args.mode)
    for shard in val_shards:
        link_or_copy(shard, dst / shard.name, args.mode)

    metadata = {
        "source_dataset": str(src),
        "dest_dataset": str(dst),
        "mode": args.mode,
        "selected_train_shards": [p.name for p in selected_train],
        "num_train_shards": len(selected_train),
        "num_val_shards": len(val_shards),
    }
    metadata_json = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if args.metadata_out is not None:
        args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_out.write_text(metadata_json, encoding="utf-8")
    print(metadata_json, end="")


if __name__ == "__main__":
    main()
