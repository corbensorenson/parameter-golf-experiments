from __future__ import annotations

import numpy as np

import mlx.core as mx
import mlx.nn as nn


def parse_csv_float_list(raw: str, field_name: str) -> list[float]:
    vals: list[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            vals.append(float(item))
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a comma-separated float list, got {raw!r}") from exc
    if not vals:
        raise ValueError(f"{field_name} is empty")
    return vals


def parse_csv_int_list(raw: str, field_name: str) -> list[int]:
    vals: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            vals.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{field_name} must be a comma-separated int list, got {raw!r}") from exc
    if not vals:
        raise ValueError(f"{field_name} is empty")
    return vals


def linear_schedule(length: int, start: float, end: float) -> list[float]:
    if length <= 0:
        return []
    if length == 1:
        return [float(end)]
    return [float(start + (end - start) * idx / (length - 1)) for idx in range(length)]


def build_tail_flags(num_layers: int, last_n: int) -> list[bool]:
    clamped = max(0, min(int(last_n), int(num_layers)))
    return [layer_idx >= int(num_layers) - clamped for layer_idx in range(int(num_layers))]


def round_hidden_dim(value: float) -> int:
    return max(8, int(round(float(value) / 8.0) * 8))


def build_layer_mlp_mults(args) -> list[float]:
    schedule_len = int(args.num_unique_blocks) if getattr(args, "model_family", "").strip().lower() == "hrc" else int(args.num_layers)
    if getattr(args, "layer_mlp_mult_schedule", ""):
        vals = parse_csv_float_list(args.layer_mlp_mult_schedule, "LAYER_MLP_MULT_SCHEDULE")
        if len(vals) != schedule_len:
            raise ValueError(
                f"LAYER_MLP_MULT_SCHEDULE length must match "
                f"{'NUM_UNIQUE_BLOCKS' if getattr(args, 'model_family', '').strip().lower() == 'hrc' else 'NUM_LAYERS'}="
                f"{schedule_len}, got {len(vals)}"
            )
        return vals
    if getattr(args, "nonuniform_mlp_schedule", False):
        return linear_schedule(schedule_len, args.nonuniform_mlp_mult_min, args.nonuniform_mlp_mult_max)
    return [float(args.mlp_mult)] * schedule_len


def build_xsa_layer_flags(args) -> list[bool]:
    return build_tail_flags(int(args.num_layers), int(getattr(args, "xsa_last_n", 0)))


def build_xsa_gate_flags(args, xsa_flags: list[bool]) -> list[bool]:
    mode = str(getattr(args, "xsa_gate_mode", "none")).strip().lower()
    if mode not in {"none", "all", "final"}:
        raise ValueError(f"XSA_GATE_MODE must be one of none/all/final, got {mode!r}")
    if mode == "none":
        return [False] * len(xsa_flags)
    if mode == "all":
        return list(xsa_flags)
    out = [False] * len(xsa_flags)
    for idx in reversed(range(len(xsa_flags))):
        if xsa_flags[idx]:
            out[idx] = True
            break
    return out


def build_parallel_residual_flags(args) -> list[bool]:
    tail_len = int(args.num_unique_blocks) if getattr(args, "model_family", "").strip().lower() == "hrc" else int(args.num_layers)
    return build_tail_flags(tail_len, int(getattr(args, "parallel_residual_last_n", 0)))


def build_hemisphere_layer_flags(args) -> list[bool]:
    return build_tail_flags(int(args.num_layers), int(getattr(args, "hemisphere_last_n", 0)))


def leaky_relu(x: mx.array, negative_slope: float) -> mx.array:
    return mx.maximum(x, 0) + float(negative_slope) * mx.minimum(x, 0)


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def hemisphere_transform(x: mx.array) -> mx.array:
    dim = int(x.shape[-1])
    if dim <= 1 or dim % 2 != 0:
        return x
    half = dim // 2
    return mx.concatenate([x[..., half:], x[..., :half]], axis=-1)


def mirror_weight(weight: mx.array, mirror_mode: str) -> mx.array:
    mode = str(mirror_mode).strip().lower()
    if mode in {"", "none"}:
        return weight
    if mode == "signperm":
        row_idx = mx.array(np.arange(weight.shape[0] - 1, -1, -1, dtype=np.int32))
        col_idx = mx.array(np.arange(weight.shape[1] - 1, -1, -1, dtype=np.int32))
        row_signs = 1.0 - 2.0 * (mx.arange(weight.shape[0], dtype=mx.float32) % 2)
        flipped = weight[row_idx][:, col_idx]
        return flipped * row_signs[:, None].astype(weight.dtype)
    if mode == "householder":
        cols = int(weight.shape[1])
        if cols <= 1:
            return weight
        idx = mx.arange(cols, dtype=mx.float32) + 1.0
        vec = mx.cos(idx * 0.61803398875)
        vec = vec * mx.rsqrt(mx.sum(vec * vec) + 1e-6)
        return weight - 2.0 * (weight @ vec[:, None]) * vec[None, :]
    raise ValueError(f"MIRROR mode must be one of none|signperm|householder, got {mirror_mode!r}")


def apply_xsa(y_native: mx.array, v_native: mx.array, eps: float = 1e-6) -> mx.array:
    bsz, seqlen, heads, head_dim = y_native.shape
    kv_heads = v_native.shape[2]
    group = heads // kv_heads
    y_grouped = y_native.reshape(bsz, seqlen, kv_heads, group, head_dim)
    v_norm = v_native * mx.rsqrt(mx.sum(v_native * v_native, axis=-1, keepdims=True) + eps)
    v_norm = v_norm[:, :, :, None, :]
    proj = mx.sum(y_grouped * v_norm, axis=-1, keepdims=True) * v_norm
    return (y_grouped - proj).reshape(bsz, seqlen, heads, head_dim)


def token_chunks(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[int]:
    usable_total = (total_tokens // seq_len) * seq_len
    if usable_total <= 0:
        raise ValueError(f"token budget too small for seq_len={seq_len}")
    usable_chunk = max((max_chunk_tokens // seq_len) * seq_len, seq_len)
    chunks: list[int] = []
    remaining = usable_total
    while remaining > 0:
        chunk = min(remaining, usable_chunk)
        chunks.append(chunk)
        remaining -= chunk
    return chunks


def microbatch_plan(total_tokens: int, seq_len: int, max_chunk_tokens: int) -> list[tuple[int, float]]:
    chunks = token_chunks(total_tokens, seq_len, max_chunk_tokens)
    total = float(sum(chunks))
    return [(chunk_tokens, float(chunk_tokens) / total) for chunk_tokens in chunks]


def accumulate_flat_grads(
    accum: dict[str, mx.array] | None,
    grads_tree: dict,
    scale: float,
    tree_flatten_fn,
) -> dict[str, mx.array]:
    flat = dict(tree_flatten_fn(grads_tree))
    if accum is None:
        return {k: g * scale for k, g in flat.items()}
    for k, g in flat.items():
        accum[k] = accum[k] + g * scale
    return accum


class BigramHashEmbedding(nn.Module):
    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
        super().__init__()
        if bigram_vocab_size <= 1:
            raise ValueError(f"BIGRAM_VOCAB_SIZE must be > 1, got {bigram_vocab_size}")
        self.bigram_vocab_size = int(bigram_vocab_size)
        self.embed = nn.Embedding(self.bigram_vocab_size, int(bigram_dim))
        self.embed.weight = mx.zeros_like(self.embed.weight)
        self.proj = None if int(bigram_dim) == int(model_dim) else nn.Linear(int(bigram_dim), int(model_dim), bias=False)
        if self.proj is not None:
            self.proj.weight = mx.zeros_like(self.proj.weight)
        self.scale = mx.array([0.05], dtype=mx.float32)

    def bigram_hash(self, token_ids: mx.array) -> mx.array:
        t = token_ids.astype(mx.int32)
        mod = self.bigram_vocab_size - 1
        first = mx.full((t.shape[0], 1), mod, dtype=mx.int32)
        rest = mx.bitwise_xor(36313 * t[:, 1:], 27191 * t[:, :-1]) % mod
        return mx.concatenate([first, rest], axis=1)

    def __call__(self, token_ids: mx.array) -> mx.array:
        h = self.embed(self.bigram_hash(token_ids))
        if self.proj is not None:
            h = h @ self.proj.weight.astype(h.dtype).T
        return h * self.scale.astype(h.dtype)[0]
