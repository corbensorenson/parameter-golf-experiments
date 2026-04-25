"""Fast local CUDA launcher tuned for the RTX 2060 SUPER scout loop.

This wrapper keeps `train_gpt.py` as the source of truth while choosing
2060-friendly defaults for short architecture sweeps. Any environment variable
explicitly supplied by the caller still wins.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def _setdefault(name: str, value: str) -> None:
    os.environ.setdefault(name, value)


def _default_data_path(profile: str) -> str | None:
    root = Path(__file__).resolve().parent
    preferred: list[Path] = []
    if profile in {
        "nearcap1120",
        "nearcap1152balanced",
        "nearcap1152coreonly",
        "nearcap1152balancedstable",
        "nearcap1152balancedqt3",
        "nearcap1152balancedbest",
        "nearcap1152balancedlong10k",
        "nearcap1152balancedlong20k",
        "nearcap1152coreonlystable",
    }:
        preferred.append(root / "data" / "datasets" / "fineweb10B_sp1024_proxy01")
    preferred.extend(
        [
            root / "data" / "datasets" / "fineweb10B_sp1024_cuda_rank256k",
            root / "data" / "datasets" / "fineweb10B_sp1024_cuda_smoke",
            root / "data" / "datasets" / "fineweb10B_sp1024_proxy01",
        ]
    )
    for candidate in preferred:
        if candidate.exists():
            return str(candidate)
    return None


def apply_2060_defaults() -> None:
    profile = os.environ.get("CUDA2060_PROFILE", "scout").strip().lower()
    if profile not in {
        "scout",
        "artifact",
        "loopscout",
        "long1k",
        "long3kprobe",
        "long10kmain",
        "long10kreserve",
        "councilsmoke",
        "council3kstable",
        "nearcap1120",
        "nearcap1152balanced",
        "nearcap1152coreonly",
        "nearcap1152balancedstable",
        "nearcap1152balancedqt3",
        "nearcap1152balancedbest",
        "nearcap1152balancedlong10k",
        "nearcap1152balancedlong20k",
        "nearcap1152coreonlystable",
    }:
        raise ValueError(
            "CUDA2060_PROFILE must be one of "
            "scout|artifact|loopscout|long1k|long3kprobe|long10kmain|long10kreserve|"
            "councilsmoke|council3kstable|nearcap1120|nearcap1152balanced|nearcap1152coreonly|"
            "nearcap1152balancedstable|nearcap1152balancedqt3|nearcap1152balancedbest|"
            "nearcap1152balancedlong10k|nearcap1152balancedlong20k|nearcap1152coreonlystable, "
            f"got {profile!r}"
        )

    user_env_keys = set(os.environ.keys())
    if "DATA_PATH" not in user_env_keys:
        data_path = _default_data_path(profile)
        if data_path is not None:
            os.environ["DATA_PATH"] = data_path

    common_defaults = {
        "DISABLE_COMPILE": "1",
        "SDP_BACKEND": "math",
        "CUDNN_BENCHMARK": "1",
        "PIN_HOST_MEMORY": "1",
        "PREFETCH_HOST_TO_DEVICE": "1",
        "PERSISTENT_BATCH_BUFFERS": "1",
        "HRC_LOOP_INDEX_DIM": "64",
        "AMP_DTYPE": "auto",
        "USE_GRAD_SCALER": "0",
        "MUON_DTYPE": "auto",
        "GRAD_ACCUM_STEPS": "1",
        "TRAIN_SEQ_LEN": "1024",
        "VAL_BATCH_SIZE": "65536",
        "TRAIN_LOG_EVERY": "10",
        "LOG_CODE_SNAPSHOT": "0",
        "LOG_NVIDIA_SMI": "0",
    }
    scout_defaults = {
        "ITERATIONS": "30",
        "WARMUP_STEPS": "0",
        "VAL_TOKENS_LIMIT": "32768",
        "SKIP_FINAL_ARTIFACTS": "1",
        "ADAMW_WEIGHT_DECAY": "0",
        "TRAIN_BATCH_TOKENS": "8192",
    }
    artifact_defaults = {
        "ITERATIONS": "60",
        "WARMUP_STEPS": "4",
        "VAL_TOKENS_LIMIT": "131072",
        "SKIP_FINAL_ARTIFACTS": "0",
        "TRAIN_BATCH_TOKENS": "8192",
    }
    serious_amp_defaults = {
        # Serious confirmation runs should use the proper AMP path on the 2060:
        # keep trainable params in fp32, use fp16 autocast, let GradScaler stay
        # enabled, and run the Muon backend math in fp32.
        "PARAM_DTYPE": "fp32",
        "USE_GRAD_SCALER": "1",
        "MUON_DTYPE": "fp32",
    }
    loopscout_defaults = {
        "MODEL_FAMILY": "hrc",
        "ITERATIONS": "20",
        "WARMUP_STEPS": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "VAL_LOSS_EVERY": "10",
        "TRAIN_LOG_EVERY": "10",
        "VAL_TOKENS_LIMIT": "32768",
        "SKIP_FINAL_ARTIFACTS": "1",
        "ADAMW_WEIGHT_DECAY": "0",
        "TRAIN_BATCH_TOKENS": "10240",
        "NUM_KV_HEADS": "1",
        "BIGRAM_VOCAB_SIZE": "2048",
        "BIGRAM_DIM": "96",
        "QK_GAIN_INIT": "5.25",
        "LOGIT_SOFTCAP": "15",
        "DEPTH_LORA_RANK": "8",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "7",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "1",
        "HRC_DEPTH_SCHEDULE_MODE": "prime_skip_superloop",
        "HRC_SUPERLOOP_SKIP_SCHEDULE": "0",
        "LR_WARMDOWN_STYLE": "cosine",
        "HRC_DEPTH_ADAPTER_TIE_MODE": "block",
        "HRC_LOOP_INDEX_ENABLED": "1",
        "XSA_LAST_N": "1",
        "HRC_PASS_EMBED_ENABLED": "1",
        "HRC_PASS_EMBED_MODE": "block_peer",
        "HRC_PASS_ROLE_MODE": "phase5",
        "HRC_ROUTE_PHASE_ENABLED": "1",
        "HRC_RECUR_INJECT_ENABLED": "1",
        "PARALLEL_RESIDUAL_LAST_N": "2",
    }
    long1k_defaults = {
        **serious_amp_defaults,
        "MODEL_FAMILY": "hrc",
        "ITERATIONS": "1000",
        "WARMUP_STEPS": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "WARMDOWN_ITERS": "1200",
        "VAL_LOSS_EVERY": "50",
        "TRAIN_LOG_EVERY": "50",
        "VAL_TOKENS_LIMIT": "32768",
        "SKIP_FINAL_ARTIFACTS": "1",
        "ADAMW_WEIGHT_DECAY": "0",
        "TRAIN_BATCH_TOKENS": "10240",
        "NUM_KV_HEADS": "1",
        "BIGRAM_VOCAB_SIZE": "2048",
        "BIGRAM_DIM": "96",
        "QK_GAIN_INIT": "5.0",
        "LOGIT_SOFTCAP": "12",
        "DEPTH_LORA_RANK": "8",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "7",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "1",
        "HRC_DEPTH_SCHEDULE_MODE": "prime_skip_superloop",
        "HRC_SUPERLOOP_SKIP_SCHEDULE": "0",
        "LR_WARMDOWN_STYLE": "cosine",
        "HRC_DEPTH_ADAPTER_TIE_MODE": "block",
        "HRC_LOOP_INDEX_ENABLED": "1",
        "XSA_LAST_N": "1",
        "HRC_PASS_EMBED_ENABLED": "1",
        "HRC_PASS_EMBED_MODE": "block_peer",
        "HRC_PASS_ROLE_MODE": "phase5",
        "HRC_ROUTE_PHASE_ENABLED": "1",
        "HRC_RECUR_INJECT_ENABLED": "1",
        "PARALLEL_RESIDUAL_LAST_N": "2",
        "HRC_MLP_ONLY_BLOCKS": "1",
        "TIED_EMBED_LR": "0.03",
        "MATRIX_LR": "0.02",
        "SCALAR_LR": "0.02",
        "GRAD_CLIP_NORM": "1.0",
    }
    long3kprobe_defaults = {
        **serious_amp_defaults,
        "MODEL_FAMILY": "hrc",
        "ITERATIONS": "3000",
        "WARMUP_STEPS": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "WARMDOWN_ITERS": "3000",
        "VAL_LOSS_EVERY": "100",
        "TRAIN_LOG_EVERY": "100",
        "VAL_TOKENS_LIMIT": "32768",
        "SKIP_FINAL_ARTIFACTS": "1",
        "ADAMW_WEIGHT_DECAY": "0",
        "TRAIN_BATCH_TOKENS": "10240",
        "NUM_KV_HEADS": "1",
        "BIGRAM_VOCAB_SIZE": "2048",
        "BIGRAM_DIM": "96",
        "QK_GAIN_INIT": "5.0",
        "LOGIT_SOFTCAP": "12",
        "DEPTH_LORA_RANK": "8",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "7",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "1",
        "HRC_DEPTH_SCHEDULE_MODE": "prime_skip_superloop",
        "HRC_SUPERLOOP_SKIP_SCHEDULE": "0",
        "LR_WARMDOWN_STYLE": "cosine",
        "HRC_DEPTH_ADAPTER_TIE_MODE": "block",
        "HRC_LOOP_INDEX_ENABLED": "1",
        "XSA_LAST_N": "1",
        "HRC_PASS_EMBED_ENABLED": "1",
        "HRC_PASS_EMBED_MODE": "block_peer",
        "HRC_PASS_ROLE_MODE": "phase5",
        "HRC_ROUTE_PHASE_ENABLED": "1",
        "HRC_RECUR_INJECT_ENABLED": "1",
        "PARALLEL_RESIDUAL_LAST_N": "2",
        "HRC_MLP_ONLY_BLOCKS": "1",
        "HRC_ATTN_ONLY_BLOCKS": "3,4",
        "TIED_EMBED_LR": "0.025",
        "MATRIX_LR": "0.0175",
        "SCALAR_LR": "0.0175",
        "GRAD_CLIP_NORM": "0.75",
    }
    long10kmain_defaults = {
        **serious_amp_defaults,
        "MODEL_FAMILY": "hrc",
        "ITERATIONS": "10000",
        "WARMUP_STEPS": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "WARMDOWN_ITERS": "1200",
        "VAL_LOSS_EVERY": "200",
        "TRAIN_LOG_EVERY": "200",
        "VAL_TOKENS_LIMIT": "32768",
        "SKIP_FINAL_ARTIFACTS": "1",
        "ADAMW_WEIGHT_DECAY": "0",
        "TRAIN_BATCH_TOKENS": "10240",
        "NUM_KV_HEADS": "1",
        "BIGRAM_VOCAB_SIZE": "2048",
        "BIGRAM_DIM": "96",
        "QK_GAIN_INIT": "5.16",
        "LOGIT_SOFTCAP": "12",
        "DEPTH_LORA_RANK": "8",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "7",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "1",
        "HRC_DEPTH_SCHEDULE_MODE": "prime_skip_superloop",
        "HRC_SUPERLOOP_SKIP_SCHEDULE": "0",
        "LR_WARMDOWN_STYLE": "cosine",
        "HRC_DEPTH_ADAPTER_TIE_MODE": "block",
        "HRC_LOOP_INDEX_ENABLED": "1",
        "XSA_LAST_N": "1",
        "HRC_PASS_EMBED_ENABLED": "1",
        "HRC_PASS_EMBED_MODE": "block_peer",
        "HRC_PASS_ROLE_MODE": "phase5",
        "HRC_ROUTE_PHASE_ENABLED": "1",
        "HRC_RECUR_INJECT_ENABLED": "1",
        "PARALLEL_RESIDUAL_LAST_N": "2",
        "HRC_MLP_ONLY_BLOCKS": "1",
        "TIED_EMBED_LR": "0.002",
        "MATRIX_LR": "0.0016",
        "SCALAR_LR": "0.0016",
        "GRAD_CLIP_NORM": "1.0",
    }
    long10kreserve_defaults = {
        **long10kmain_defaults,
        "HRC_ATTN_ONLY_BLOCKS": "3,4",
    }
    councilsmoke_defaults = {
        **long1k_defaults,
        "NUM_UNIQUE_BLOCKS": "2",
        "EFFECTIVE_DEPTH": "8",
        "HRC_DEPTH_SCHEDULE_MODE": "cycle",
        "BIGRAM_VOCAB_SIZE": "3072",
        "BIGRAM_DIM": "112",
        "QK_GAIN_INIT": "5.25",
        "LOGIT_SOFTCAP": "15",
        "VE_ENABLED": "1",
        "VE_DIM": "96",
        "VE_LAYERS": "6,7",
        "HRC_PASS_EMBED_INIT_STD": "0.003",
        "HRC_PASS_EMBED_MODE": "shared",
        # Keep the tested shell1 specialization so councilsmoke matches the
        # faithful CUDA council probes already in the logs.
        "HRC_MLP_ONLY_BLOCKS": "1",
        "HRC_MIRROR_MODE": "householder",
        "HRC_COUNCIL_MODE": "base_mirror",
        "HRC_COUNCIL_TRAIN_MODE": "eval_only",
        "HRC_COUNCIL_DEPTH_OFFSETS": "0,-2",
        "HRC_COUNCIL_CONF_SCALE_INIT": "1.0",
        "HRC_COUNCIL_HARD_GATE": "0",
        "HRC_COUNCIL_ENTROPY_THRESHOLD": "6.0",
        "HRC_COUNCIL_ENTROPY_SHARPNESS": "8.0",
        # Council eval can overflow on mirrored peer logits in fp16 on the 2060.
        # Keep the council-only sanitize/clamp defaults enabled in this narrow
        # diagnostic profile so medium-rung council probes stay finite.
        "HRC_COUNCIL_SANITIZE": "1",
        "HRC_COUNCIL_LOGIT_CLAMP": "60",
    }
    council3kstable_defaults = {
        **councilsmoke_defaults,
        "ITERATIONS": "3000",
        "VAL_LOSS_EVERY": "500",
        "TRAIN_LOG_EVERY": "500",
        "WARMDOWN_ITERS": "3000",
        "TIED_EMBED_LR": "0.01",
        "MATRIX_LR": "0.0075",
        "SCALAR_LR": "0.0075",
    }
    nearcap_common_defaults = {
        **serious_amp_defaults,
        "MODEL_FAMILY": "hrc",
        # Match the promoted MLX proxy recipe before we reopen the broader CUDA
        # search: shorter seq_len, batch/token geometry, and faithful council +
        # late-tail qsparse + roundtrip quant-train.
        "ITERATIONS": "1000",
        "WARMUP_STEPS": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "WARMDOWN_ITERS": "1200",
        "LR_WARMDOWN_STYLE": "linear",
        "VAL_LOSS_EVERY": "50",
        "TRAIN_LOG_EVERY": "50",
        "VAL_TOKENS_LIMIT": "65536",
        "VAL_BATCH_SIZE": "2048",
        "SKIP_FINAL_ARTIFACTS": "1",
        "ADAMW_WEIGHT_DECAY": "0",
        "TRAIN_BATCH_TOKENS": "2048",
        "GRAD_ACCUM_STEPS": "4",
        "TRAIN_SEQ_LEN": "128",
        "NUM_UNIQUE_BLOCKS": "2",
        "EFFECTIVE_DEPTH": "8",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "ACTIVATION_KIND": "relu2",
        "BIGRAM_VOCAB_SIZE": "3072",
        "VE_ENABLED": "1",
        "VE_LAYERS": "6,7",
        "DEPTH_LORA_RANK": "12",
        "HRC_DEPTH_SCHEDULE_MODE": "cycle",
        "HRC_ROUTE_REPEATS": "1",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_MIRROR_MODE": "householder",
        "HRC_COUNCIL_MODE": "base_mirror",
        "HRC_COUNCIL_TRAIN_MODE": "eval_only",
        "HRC_COUNCIL_CONF_SCALE_INIT": "1.0",
        "HRC_COUNCIL_HARD_GATE": "1",
        "HRC_COUNCIL_ENTROPY_THRESHOLD": "6.0",
        "HRC_COUNCIL_ENTROPY_SHARPNESS": "8.0",
        # The raw faithful council path overflowed in fp16 eval on CUDA.
        # Keep the council-only sanitize/clamp guard rails on by default here
        # so we test the promoted family instead of rediscovering that failure.
        "HRC_COUNCIL_SANITIZE": "1",
        "HRC_COUNCIL_LOGIT_CLAMP": "60",
        "HRC_PASS_EMBED_ENABLED": "1",
        "HRC_PASS_EMBED_MODE": "shared",
        "HRC_PASS_EMBED_INIT_STD": "0.003",
        "XSA_LAST_N": "1",
        "QSPARSE_ENABLED": "1",
        "QSPARSE_TOPK": "128",
        "QSPARSE_LAST_N": "3",
        "QUANT_TRAIN_MODE": "roundtrip",
        "QUANT_TRAIN_START_FRACTION": "0.80",
        "QUANT_TRAIN_EVERY": "2",
        "TIED_EMBED_LR": "0.05",
        "MATRIX_LR": "0.04",
        "SCALAR_LR": "0.04",
    }
    nearcap1120_defaults = {
        **nearcap_common_defaults,
        "MODEL_DIM": "1120",
        "BIGRAM_DIM": "240",
        "VE_DIM": "240",
    }
    nearcap1152balanced_defaults = {
        **nearcap_common_defaults,
        "MODEL_DIM": "1152",
        "BIGRAM_DIM": "256",
        "VE_DIM": "256",
    }
    nearcap1152coreonly_defaults = {
        **nearcap_common_defaults,
        "MODEL_DIM": "1152",
        "BIGRAM_DIM": "240",
        "VE_DIM": "240",
    }
    nearcap_stable_common_defaults = {
        **nearcap_common_defaults,
        # Stabilized long-rung recipe discovered on CUDA: decay across the whole
        # run and trim the faithful MLX LRs moderately so the promoted family
        # survives past the old 900-1k failure zone on the 2060.
        "ITERATIONS": "3000",
        "WARMDOWN_ITERS": "3000",
        "TIED_EMBED_LR": "0.04",
        "MATRIX_LR": "0.032",
        "SCALAR_LR": "0.032",
    }
    nearcap1152balancedstable_defaults = {
        **nearcap_stable_common_defaults,
        "MODEL_DIM": "1152",
        "BIGRAM_DIM": "256",
        "VE_DIM": "256",
    }
    nearcap1152balancedqt3_defaults = {
        **nearcap1152balancedstable_defaults,
        "QUANT_TRAIN_EVERY": "3",
    }
    nearcap1152balancedbest_defaults = {
        **nearcap1152balancedqt3_defaults,
        # Best local near-cap CUDA lane so far: hold the stable LR recipe,
        # keep reduced roundtrip cadence, and wait until the final step window
        # before applying the quant-train projection.
        "QUANT_TRAIN_START_FRACTION": "0.99",
    }
    nearcap1152balancedlong10k_defaults = {
        **nearcap1152balancedbest_defaults,
        # Promote the current near-cap winner into a real longer local rung
        # using the latest repaired 10k schedule frontier: full-run decay, the
        # cooler LR trim, and the bracketed Muon momentum winner that gives the
        # best completed local 10k curve on the 2060.
        "ITERATIONS": "10000",
        "WARMDOWN_ITERS": "10000",
        "TIED_EMBED_LR": "0.025",
        "MATRIX_LR": "0.020",
        "SCALAR_LR": "0.020",
        "MUON_MOMENTUM": "0.88",
        "VAL_LOSS_EVERY": "200",
        "TRAIN_LOG_EVERY": "200",
    }
    nearcap1152balancedlong20k_defaults = {
        **nearcap1152balancedlong10k_defaults,
        # Extended local replay profile for candidates that earn it after the
        # repaired 10k rung. Keep the same repaired schedule shape and scale it
        # conservatively to a longer horizon.
        "ITERATIONS": "20000",
        "WARMDOWN_ITERS": "10000",
        "VAL_LOSS_EVERY": "400",
        "TRAIN_LOG_EVERY": "400",
    }
    nearcap1152coreonlystable_defaults = {
        **nearcap_stable_common_defaults,
        "MODEL_DIM": "1152",
        "BIGRAM_DIM": "240",
        "VE_DIM": "240",
    }
    profile_defaults = {
        "scout": scout_defaults,
        "artifact": artifact_defaults,
        "loopscout": loopscout_defaults,
        "long1k": long1k_defaults,
        "long3kprobe": long3kprobe_defaults,
        "long10kmain": long10kmain_defaults,
        "long10kreserve": long10kreserve_defaults,
        "councilsmoke": councilsmoke_defaults,
        "council3kstable": council3kstable_defaults,
        "nearcap1120": nearcap1120_defaults,
        "nearcap1152balanced": nearcap1152balanced_defaults,
        "nearcap1152coreonly": nearcap1152coreonly_defaults,
        "nearcap1152balancedstable": nearcap1152balancedstable_defaults,
        "nearcap1152balancedqt3": nearcap1152balancedqt3_defaults,
        "nearcap1152balancedbest": nearcap1152balancedbest_defaults,
        "nearcap1152balancedlong10k": nearcap1152balancedlong10k_defaults,
        "nearcap1152balancedlong20k": nearcap1152balancedlong20k_defaults,
        "nearcap1152coreonlystable": nearcap1152coreonlystable_defaults,
    }

    for key, value in common_defaults.items():
        if key not in user_env_keys:
            os.environ[key] = value
    for key, value in profile_defaults[profile].items():
        if key not in user_env_keys:
            os.environ[key] = value
    model_family = os.environ.get("MODEL_FAMILY", "").strip().lower()
    if "OPTIMIZER_PRESET" not in os.environ:
        os.environ["OPTIMIZER_PRESET"] = "hybrid" if model_family == "hrc" else "adamw"

    if "RUN_ID" not in os.environ:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.environ["RUN_ID"] = f"cuda2060-{profile}-{stamp}"


def main() -> None:
    apply_2060_defaults()
    import train_gpt

    train_gpt.main()


if __name__ == "__main__":
    main()
