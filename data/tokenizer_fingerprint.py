#!/usr/bin/env python3
"""
Deterministic tokenizer fingerprint utility for local reproducibility.

This script emits stable JSON metadata for tokenizer artifacts and, optionally,
the manifest dataset/tokenizer pairing that generated local shards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    import sentencepiece as spm
except Exception:  # pragma: no cover - optional dependency at utility-runtime
    spm = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


def load_manifest_pairing(manifest_path: Path, dataset_name: str) -> dict[str, object]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_entry = next((x for x in manifest.get("datasets", []) if x.get("name") == dataset_name), None)
    if dataset_entry is None:
        raise ValueError(f"dataset {dataset_name!r} not found in {manifest_path}")
    tokenizer_name = dataset_entry.get("tokenizer_name")
    tokenizer_entry = next((x for x in manifest.get("tokenizers", []) if x.get("name") == tokenizer_name), None)
    return {
        "dataset_name": dataset_name,
        "dataset_tokenizer_name": tokenizer_name,
        "expected_train_shards": maybe_int((dataset_entry.get("stats") or {}).get("files_train")),
        "expected_val_shards": maybe_int((dataset_entry.get("stats") or {}).get("files_val")),
        "tokenizer_manifest_entry": tokenizer_entry,
    }


def build_fingerprint(
    *,
    tokenizer_model_path: Path,
    tokenizer_vocab_path: Path | None,
    manifest_path: Path | None,
    dataset_name: str | None,
) -> dict[str, object]:
    if not tokenizer_model_path.is_file():
        raise FileNotFoundError(f"tokenizer model not found: {tokenizer_model_path}")
    if tokenizer_vocab_path is not None and not tokenizer_vocab_path.is_file():
        raise FileNotFoundError(f"tokenizer vocab not found: {tokenizer_vocab_path}")

    vocab_size = None
    if spm is not None:
        proc = spm.SentencePieceProcessor(model_file=str(tokenizer_model_path))
        vocab_size = int(proc.vocab_size())

    payload: dict[str, object] = {
        "tokenizer_model_path": str(tokenizer_model_path.resolve()),
        "tokenizer_model_sha256": sha256_file(tokenizer_model_path),
        "tokenizer_model_bytes": tokenizer_model_path.stat().st_size,
        "tokenizer_vocab_size": vocab_size,
    }
    if tokenizer_vocab_path is not None:
        payload.update(
            {
                "tokenizer_vocab_path": str(tokenizer_vocab_path.resolve()),
                "tokenizer_vocab_sha256": sha256_file(tokenizer_vocab_path),
                "tokenizer_vocab_bytes": tokenizer_vocab_path.stat().st_size,
            }
        )
    if manifest_path is not None:
        if dataset_name is None:
            raise ValueError("--dataset-name is required when --manifest is set")
        payload["manifest_path"] = str(manifest_path.resolve())
        payload["manifest_sha256"] = sha256_file(manifest_path)
        payload["manifest_pairing"] = load_manifest_pairing(manifest_path, dataset_name)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit deterministic tokenizer fingerprints")
    parser.add_argument(
        "--tokenizer-model",
        required=True,
        type=Path,
        help="Path to SentencePiece .model file",
    )
    parser.add_argument(
        "--tokenizer-vocab",
        type=Path,
        default=None,
        help="Optional path to .vocab sidecar file",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to data/manifest.json for dataset/tokenizer pairing metadata",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset name in manifest (for example fineweb10B_sp1024)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. If omitted, prints JSON to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_fingerprint(
        tokenizer_model_path=args.tokenizer_model,
        tokenizer_vocab_path=args.tokenizer_vocab,
        manifest_path=args.manifest,
        dataset_name=args.dataset_name,
    )
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(text, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
