from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_datafile_body(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype="<u2")
    if data.size < 512:
        raise ValueError(f"datafile too small: {path}")
    header = data[:512].view("<i4")
    n_tokens = int(header[2])
    body = data[512 : 512 + n_tokens]
    if body.size != n_tokens:
        raise ValueError(f"truncated datafile: {path}")
    return body


def resolve_glob(root: Path, pattern: str | None) -> list[Path]:
    if not pattern:
        return []
    path = Path(pattern)
    if not path.is_absolute():
        path = root / path
    return sorted(path.parent.glob(path.name))


def sum_val_bytes(root: Path, dataset: dict[str, Any]) -> int | None:
    paths = resolve_glob(root, dataset.get("val_bytes_glob"))
    if not paths:
        return None
    return int(sum(int(read_datafile_body(path).sum()) for path in paths))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize tokenizer export fertility from a Parameter Golf manifest."
    )
    parser.add_argument("manifest", help="Path to manifest.json from a tokenizer export")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    tokenizers = {item["name"]: item for item in manifest.get("tokenizers", [])}
    datasets = manifest.get("datasets", [])

    print("| Dataset | Vocab | Transform | Val Tokens | Val Bytes | Tokens/Byte | Bytes/Token | Train Tokens |")
    print("| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for dataset in datasets:
        tok = tokenizers.get(dataset.get("tokenizer_name"), {})
        stats = dataset.get("stats") or {}
        val_tokens = int(stats.get("tokens_val", 0))
        train_tokens = int(stats.get("tokens_train", 0))
        val_bytes = sum_val_bytes(root, dataset)
        if val_bytes and val_tokens:
            tokens_per_byte = val_tokens / val_bytes
            bytes_per_token = val_bytes / val_tokens
            tpb = f"{tokens_per_byte:.6f}"
            bpt = f"{bytes_per_token:.6f}"
            val_bytes_text = str(val_bytes)
        else:
            tpb = ""
            bpt = ""
            val_bytes_text = ""
        print(
            "| "
            + " | ".join(
                [
                    f"`{dataset.get('name', '')}`",
                    str(dataset.get("vocab_size", tok.get("vocab_size", ""))),
                    str(tok.get("text_transform", dataset.get("text_transform", "identity"))),
                    str(val_tokens),
                    val_bytes_text,
                    tpb,
                    bpt,
                    str(train_tokens),
                ]
            )
            + " |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
