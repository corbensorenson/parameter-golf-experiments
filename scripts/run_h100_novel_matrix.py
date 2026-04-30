"""Run H100 contender probes for the novel HRC/VocabMoE lane.

This is intentionally separate from the local RTX 2060 matrix runner. The 2060
runner calls the CUDA2060 wrapper and uses tiny local batches/contexts. This
script launches train_gpt.py directly through torch.distributed.run so the same
candidate definitions can be tested under the real 8xH100-style budget.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from run_16mb_vocab_moe_matrix import (
    BEST_CLEAN_VOCABMOE,
    DATASET_DIR,
    ROOT,
    TOKENIZER_PATH,
    bigram_hash_env,
    cap16_lqer_env,
    cap16_nospeed_base,
    cap16_taper_env,
    dynamic_council_env,
    dual_stream_env,
    leaderboard_schedule_env,
    legal_ttt_env,
    parse_train,
    q8_core_io_ladder_env,
    q8_train_export_env,
    rlm_memory_env,
    run_command_live,
    sparse_gate_env,
    spike_vocab_moe_env,
    vocab_moe_env,
    write_candidate_plan,
    write_summary,
)
from run_sub4_micro_matrix import configure_env, merged_train_output, write_csv


TRAINER = ROOT / "train_gpt.py"


H100_COMMON: dict[str, str] = {
    # Official-style distributed run. With WORLD_SIZE=8, GRAD_ACCUM_STEPS=0 lets
    # train_gpt.py choose one microstep per rank.
    "MODEL_FAMILY": "hrc",
    "TRAIN_BATCH_TOKENS": "524288",
    "TRAIN_SEQ_LEN": "1024",
    "GRAD_ACCUM_STEPS": "0",
    "VAL_BATCH_SIZE": "524288",
    "MAX_WALLCLOCK_SECONDS": "600",
    "ITERATIONS": "1000000",
    "WARMDOWN_ITERS": "3500",
    # Warmup restores the initial model/optimizer state, so it is useful for
    # compile path priming but just burns real wallclock in the no-compile H100
    # scout profile. Keep the paid runs training immediately.
    "WARMUP_STEPS": "0",
    "VAL_LOSS_EVERY": "0",
    "TRAIN_LOG_EVERY": "250",
    "SKIP_INITIAL_VAL": "1",
    "SKIP_FINAL_ARTIFACTS": "0",
    "TRAIN_QUANT_FORWARD": "1",
    "TRAIN_QUANT_EMBEDDINGS": "1",
    "QUANT_TRAIN_MODE": "none",
    # Same parameterization as separate Q/K/V, fewer projection launches and
    # one train-time quant materialization instead of three.
    "TRAIN_FUSED_QKV": "1",
    "LOG_CODE_SNAPSHOT": "0",
    "LOG_NVIDIA_SMI": "1",
    # Reliability first for rented time. We can add compile only after a smoke
    # proves the fullgraph path is stable for this custom HRC stack.
    "DISABLE_COMPILE": "1",
    "SDP_BACKEND": "auto",
    "PARAM_DTYPE": "fp32",
    "USE_GRAD_SCALER": "1",
    "MUON_DTYPE": "fp32",
    "LOSS_FP32": "1",
    # The H100 scout profile has already smoked the candidates; avoid repeated
    # every-step safety scans and redundant post-step grad clearing. Gradients
    # are still cleared at the start of each step.
    "TRAIN_DEBUG_NONFINITE": "0",
    "TRAIN_ABORT_ON_NONFINITE": "0",
    "POST_STEP_ZERO_GRAD": "0",
    "MODEL_CODEC": "lzma",
    "MODEL_CODEC_LEVEL": "9",
    "SAVE_RAW_MODEL": "0",
    "SUBMISSION_SIZE_CAP_BYTES": "16000000",
    "FAIL_ON_ARTIFACT_CAP": "0",
}


H100_1X_DEFAULTS: dict[str, str] = {
    # Match the per-rank microbatch of the 8xH100 profile. The earlier 131k
    # scout batch was fine for the accidentally-baseline path, but real HRC
    # plus an 8192-way loss crosses 80GB.
    "TRAIN_BATCH_TOKENS": "65536",
    "GRAD_ACCUM_STEPS": "1",
    "VAL_BATCH_SIZE": "65536",
    "TRAIN_LOG_EVERY": "250",
}


def h100_env(env: dict[str, str]) -> dict[str, str]:
    out = dict(env)
    # Remove local-only defaults when the base row came from the 2060 runner.
    out.pop("CUDA2060_PROFILE", None)
    out.update(H100_COMMON)
    return out


def h100_env_with_overrides(env: dict[str, str], overrides: dict[str, str]) -> dict[str, str]:
    out = h100_env(env)
    out.update(overrides)
    return out


def hrc_core_mlp_env(io_width: int, loop_width: int, keep_attn_blocks: tuple[int, ...] = ()) -> dict[str, str]:
    keep = {int(block) for block in keep_attn_blocks}
    core_blocks = [
        str(block)
        for block in range(int(io_width), int(io_width) + int(loop_width))
        if block not in keep
    ]
    return {"HRC_MLP_ONLY_BLOCKS": ",".join(core_blocks)}


H100_NOVEL_ROUND1: list[dict[str, Any]] = [
    {
        "name": "h100_hrc_i3l5r2_d896e2688_q8_coremlp_embedq",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=896, embed_dim=2688, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
            }
        ),
        "notes": (
            "real HRC cap-spend mainline: mirrored IO tail, two loop passes, "
            "MLP-only recurrent core for speed, wide d896/e2688 q8 token interface"
        ),
    },
    {
        "name": "h100_hrc_i3l5r2_d768e2688_q8_coreattn1_lqer12t24_embedq",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=2688, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
            }
        ),
        "notes": (
            "same real HRC route but lets the first recurrent block keep attention; "
            "tests the quality/speed tradeoff of occasional core token mixing"
        ),
    },
    {
        "name": "h100_hrc_i3l7r2_d768e2560_q8_coremlp_lqer12t24_embedq",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=2560, io_width=3, loop_width=7, repeats=2),
                **hrc_core_mlp_env(3, 7),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
            }
        ),
        "notes": (
            "prime-ish loop-width row: more unique recurrent MLP blocks rather than "
            "more repeats, matching the strongest signal from local loop sweeps"
        ),
    },
    {
        "name": "h100_hrc_dual_i3l5r2_d768e2560_q8_coremlp_left320_embedq",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=2560, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **dual_stream_env(left_dim=320, rank=24, sites="input,loop_first,loop_exit,pre_output"),
            }
        ),
        "notes": (
            "trained left/right advisor bridge on the best HRC/VocabMoE spine; "
            "this is the novel dual-stream scout, kept as one row instead of a broad sweep"
        ),
    },
]


H100_NOVEL_ROUND2_SEEDS: list[dict[str, Any]] = [
    {
        "name": "h100_hrc_i3l5r2_d896e2688_q8_coremlp_lqer12t24_embedq",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=896, embed_dim=2688, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
            }
        ),
        "notes": "repair row for the real HRC near-cap mainline if round1 promotes it",
    },
]


H100_CAPLEGAL_FOLLOWUP: list[dict[str, Any]] = [
    {
        "name": "h100_caplegal_i3l5r2_d768e1728_q8_coreattn1_lqer12t24_vocabmoe_qk525",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1728, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.25",
            }
        ),
        "notes": (
            "cap-safe version of the best over-cap H100 signal: d768, one "
            "attention core block, QK 5.25, and LQER repair"
        ),
    },
    {
        "name": "h100_caplegal_dual_i3l5r2_d768e1536_q8_coremlp_left256_vocabmoe_qk525",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1536, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **dual_stream_env(left_dim=256, rank=16, sites="input,loop_first,loop_exit,pre_output"),
                "QK_GAIN_INIT": "5.25",
            }
        ),
        "notes": (
            "cap-safe version of the best over-cap dual-stream row; keeps the "
            "advisor idea but trims token rank enough to make size plausible"
        ),
    },
    {
        "name": "h100_caplegal_i3l5r2_d896e896_q866_q4core_vocabmoe_lqer16t32_qk525",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=896, embed_dim=896, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5),
                **cap16_taper_env(io_width=3, loop_width=5, io_bits=(8, 6, 6), core_bits=4),
                **cap16_lqer_env(16, 32),
                **BEST_CLEAN_VOCABMOE,
                **leaderboard_schedule_env(min_lr=0.10),
                "QK_GAIN_INIT": "5.25",
            }
        ),
        "notes": (
            "mixed-precision-from-start candidate: spends bytes on d896 while "
            "using q8/q6/q6 IO-tail and q4 recurrent core"
        ),
    },
]


H100_1X_ARCH_SCOUT: list[dict[str, Any]] = [
    {
        "name": "h100_1x_i3l5r2_d768e1536_q8_coreattn1_lqer12t24_vocabmoe_qk525",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1536, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.25",
            }
        ),
        "notes": (
            "cap-margin sibling of the current d768/e1728 winner; tests how "
            "much quality the extra token rank is buying"
        ),
    },
    {
        "name": "h100_1x_i3l5r2_d768e1856_q8_coreattn1_lqer12t24_vocabmoe_qk525",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1856, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.25",
            }
        ),
        "notes": (
            "token-interface spend above e1728; proxy-only unless the final "
            "artifact from e1728 proves enough cap headroom"
        ),
    },
    {
        "name": "h100_1x_i3l5r2_d768e1536_q8_coreattn2_lqer12t24_vocabmoe_qk525",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1536, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3, 4)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.25",
            }
        ),
        "notes": (
            "adds a second attention-enabled recurrent block; tests whether "
            "the loop needs more token mixing than the one-attention winner"
        ),
    },
    {
        "name": "h100_1x_i3l7r2_d768e1536_q8_coreattn1_lqer12t24_vocabmoe_qk525",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1536, io_width=3, loop_width=7, repeats=2),
                **hrc_core_mlp_env(3, 7, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.25",
            }
        ),
        "notes": (
            "tests the user's more-unique-loop-block hypothesis with the "
            "same first-core-attention fix that helped the cap-legal row"
        ),
    },
    {
        "name": "h100_1x_dual_i3l5r2_d768e1536_q8_coreattn1_left256_vocabmoe_qk525",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1536, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(8, 12),
                **dual_stream_env(left_dim=256, rank=16, sites="input,loop_first,loop_exit,pre_output"),
                "QK_GAIN_INIT": "5.25",
            }
        ),
        "notes": (
            "dual-stream retest with the one-attention recurrent core; the "
            "previous dual row was faster but lacked core attention"
        ),
    },
    {
        "name": "h100_1x_i3l5r2_d768e1536_q8_coreattn1_vocabmoe_r4_qk525",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1536, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **vocab_moe_env(
                    experts=16,
                    rank=4,
                    mode="hybrid",
                    layers="input,loop_first",
                    train_quant_bits=8,
                ),
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.25",
            }
        ),
        "notes": (
            "spends bytes on richer token-conditioned experts instead of "
            "embedding rank; high-value VocabMoE capacity test"
        ),
    },
    {
        "name": "h100_1x_i3l5r2_d768e1536_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1536, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.5",
            }
        ),
        "notes": (
            "tests whether the public-leaderboard QK-gain push transfers "
            "above 5.25 on this HRC/VocabMoE shape"
        ),
    },
]


H100_1X_PRECISION_WIDTH_SCOUT: list[dict[str, Any]] = [
    {
        "name": "h100_pw_i3l5r5_d640e512_q8_polar_vocabmoe_qk50",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
                **hrc_core_mlp_env(3, 5),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                "QK_GAIN_INIT": "5.0",
            }
        ),
        "notes": (
            "H100 rerun of the strongest completed local spine: q8 "
            "train/export, i3/l5/r5 HRC, e512 token rank, VocabMoE, Polar "
            "Muon, and 10% min-LR"
        ),
    },
    {
        "name": "h100_pw_i3l5r5_d640e640_q8_polar_vocabmoe_qk50",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=640, embed_dim=640, io_width=3, loop_width=5, repeats=5),
                **hrc_core_mlp_env(3, 5),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                "QK_GAIN_INIT": "5.0",
            }
        ),
        "notes": (
            "cap-spend sibling of the local winner: spends the remaining "
            "bytes on the factored tied embedding interface, where local "
            "e640 already improved the pre-Polar cap-fill branch"
        ),
    },
    {
        "name": "h100_pw_i3l5r5_d704e512_q8_polar_vocabmoe_qk50",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=704, embed_dim=512, io_width=3, loop_width=5, repeats=5),
                **hrc_core_mlp_env(3, 5),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                "QK_GAIN_INIT": "5.0",
            }
        ),
        "notes": (
            "residual-width spend row: d704 keeps 64-wide heads while moving "
            "closer to the 16MB cap without changing the recurrent program"
        ),
    },
    {
        "name": "h100_pw_i3l5r5_d640e512_q8_width400480560640_vocabmoe_qk50",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
                **hrc_core_mlp_env(3, 5),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                "BASIS_XSA_ENABLED": "0",
                "DEPTH_LORA_RANK": "0",
                "VE_ENABLED": "0",
                "LAYER_WIDTH_SCHEDULE": "400,480,560,640,640,640,640,640",
                "QK_GAIN_INIT": "5.0",
            }
        ),
        "notes": (
            "pure width-ladder/hourglass row: higher-precision token-facing "
            "blocks run narrower, while the low-precision recurrent semantic "
            "core keeps full d640 capacity"
        ),
    },
    {
        "name": "h100_pw_i3l5r5_d640e512_q16q8q8io_q8core_width_vocabmoe_qk50",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
                **hrc_core_mlp_env(3, 5),
                **BEST_CLEAN_VOCABMOE,
                **q8_core_io_ladder_env((16, 8, 8)),
                **leaderboard_schedule_env(min_lr=0.10),
                "BASIS_XSA_ENABLED": "0",
                "DEPTH_LORA_RANK": "0",
                "VE_ENABLED": "0",
                "LAYER_WIDTH_SCHEDULE": "400,480,560,640,640,640,640,640",
                "QK_GAIN_INIT": "5.0",
            }
        ),
        "notes": (
            "q8-floor data-density ladder: IO-tail blocks are trained from "
            "step zero at q16/q8/q8 while the recurrent core stays q8 and "
            "full width"
        ),
    },
    {
        "name": "h100_pw_i3l5r5_d640e512_q16q8q4io_q4core_width_vocabmoe_qk50",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=640, embed_dim=512, io_width=3, loop_width=5, repeats=5),
                **hrc_core_mlp_env(3, 5),
                **BEST_CLEAN_VOCABMOE,
                **cap16_taper_env(io_width=3, loop_width=5, io_bits=(16, 8, 4), core_bits=4),
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                "BASIS_XSA_ENABLED": "0",
                "DEPTH_LORA_RANK": "0",
                "VE_ENABLED": "0",
                "LAYER_WIDTH_SCHEDULE": "400,480,560,640,640,640,640,640",
                "QK_GAIN_INIT": "5.0",
            }
        ),
        "notes": (
            "q4-floor data-density ladder: IO-tail blocks are trained from "
            "step zero at q16/q8/q4 and the recurrent core is q4, so the "
            "core precision never exceeds the narrowest IO-tail precision"
        ),
    },
]


H100_1X_CAPFIT_SCOUT: list[dict[str, Any]] = [
    {
        "name": "h100_capfit_i3l5r2_d768e1024_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1024, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.5",
            }
        ),
        "notes": "safe cap-fit version of the d768/e1536 QK 5.5 winner",
    },
    {
        "name": "h100_capfit_i3l5r2_d768e1088_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1088, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.5",
            }
        ),
        "notes": "middle cap-fit row; should keep most e1536 quality while staying under 16MB",
    },
    {
        "name": "h100_capfit_i3l5r2_d768e1120_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1120, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.5",
            }
        ),
        "notes": "trained-export-slope cap-fit row; targets ~15.9MB instead of trusting 1-step compression",
    },
    {
        "name": "h100_capfit_i3l5r2_d768e1152_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1152, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.5",
            }
        ),
        "notes": "near-cap estimate from the e1536/e1728 export slope",
    },
    {
        "name": "h100_capfit_i3l5r2_d768e1216_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=1216, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "QK_GAIN_INIT": "5.5",
            }
        ),
        "notes": "aggressive cap-fit row; only train if 1-step export preflight passes",
    },
]


def leaderboard_transfer_tail(*, embed_dim: int) -> dict[str, str]:
    return {
        **cap16_nospeed_base(model_dim=768, embed_dim=embed_dim, io_width=3, loop_width=5, repeats=2),
        **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
        **BEST_CLEAN_VOCABMOE,
        **q8_train_export_env(),
        **leaderboard_schedule_env(min_lr=0.10),
        **cap16_lqer_env(12, 24),
        **sparse_gate_env(width=12, scale=0.5),
        "BETA2": "0.99",
        "WARMDOWN_ITERS": "2200",
        "QK_GAIN_INIT": "5.5",
    }


H100_1X_LEADER_TRANSFER: list[dict[str, Any]] = [
    {
        "name": "h100_lt_i3l5r2_d768e1088_q8_sparsegate05_lqer12t24_vocabmoe_qk55_b99",
        "env": h100_env(leaderboard_transfer_tail(embed_dim=1088)),
        "notes": (
            "late-leaderboard transfer onto our HRC/VocabMoE spine: BOS-safe "
            "SmearGate, sparse attention gate scale 0.5, beta2 0.99, and "
            "1x-appropriate warmdown"
        ),
    },
    {
        "name": "h100_lt_i3l5r2_d768e1120_q8_sparsegate05_lqer12t24_vocabmoe_qk55_b99",
        "env": h100_env(leaderboard_transfer_tail(embed_dim=1120)),
        "notes": "same as e1088, nudged toward the trained-export slope cap boundary",
    },
    {
        "name": "h100_lt_i3l5r2_d768e1088_q8_sparsegate05_bigram10240_lqer12t24_vocabmoe_qk55_b99",
        "env": h100_env(
            {
                **leaderboard_transfer_tail(embed_dim=1088),
                **bigram_hash_env(vocab_size=10240, dim=32),
            }
        ),
        "notes": "tests the recurring public BigramHash side channel without changing the HRC spine",
    },
    {
        "name": "h100_lt_i3l5r2_d768e1088_q8_sparsegate05_ttt24_lqer12t24_vocabmoe_qk55_b99",
        "env": h100_env(
            {
                **leaderboard_transfer_tail(embed_dim=1088),
                **legal_ttt_env(lr=0.005, updates=24),
            }
        ),
        "notes": "score-first TTT canary on the cap-fit HRC/VocabMoE row; training path is unchanged",
    },
]


def e1088_best_tail() -> dict[str, str]:
    return {
        **cap16_nospeed_base(model_dim=768, embed_dim=1088, io_width=3, loop_width=5, repeats=2),
        **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
        **BEST_CLEAN_VOCABMOE,
        **q8_train_export_env(),
        **leaderboard_schedule_env(min_lr=0.10),
        **cap16_lqer_env(12, 24),
        "QK_GAIN_INIT": "5.5",
    }


def warmdown_capfit_tail(*, embed_dim: int, weight_decay: float = 0.0) -> dict[str, str]:
    env = {
        **cap16_nospeed_base(model_dim=768, embed_dim=embed_dim, io_width=3, loop_width=5, repeats=2),
        **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
        **BEST_CLEAN_VOCABMOE,
        **q8_train_export_env(),
        **leaderboard_schedule_env(min_lr=0.10),
        **cap16_lqer_env(12, 24),
        "QK_GAIN_INIT": "5.5",
    }
    if weight_decay > 0.0:
        env.update(
            {
                "MUON_WEIGHT_DECAY": f"{weight_decay:g}",
                "MUON_WEIGHT_DECAY_MODE": "huber",
            }
        )
    return env


def baseline_chase_tail(
    *,
    model_dim: int,
    embed_dim: int,
    io_width: int = 3,
    loop_width: int = 5,
    repeats: int = 2,
    io_bits: tuple[int, ...] = (8, 8, 8),
    core_bits: int = 8,
    keep_attn_blocks: tuple[int, ...] = (3,),
    lqer_rank: int = 12,
    lqer_top_k: int = 24,
    qk_gain: float = 5.5,
    weight_decay: float = 0.0,
) -> dict[str, str]:
    env = {
        **cap16_nospeed_base(
            model_dim=model_dim,
            embed_dim=embed_dim,
            io_width=io_width,
            loop_width=loop_width,
            repeats=repeats,
        ),
        **hrc_core_mlp_env(io_width, loop_width, keep_attn_blocks=keep_attn_blocks),
        **BEST_CLEAN_VOCABMOE,
        **q8_train_export_env(),
        **leaderboard_schedule_env(min_lr=0.10),
        **cap16_lqer_env(lqer_rank, lqer_top_k),
        "QK_GAIN_INIT": f"{qk_gain:g}",
    }
    if weight_decay > 0.0:
        env.update(
            {
                "MUON_WEIGHT_DECAY": f"{weight_decay:g}",
                "MUON_WEIGHT_DECAY_MODE": "huber",
            }
        )
    return env


def loop_index_env(*, dim: int = 64, scale: float = 0.02) -> dict[str, str]:
    return {
        "HRC_LOOP_INDEX_ENABLED": "1",
        "HRC_LOOP_INDEX_DIM": str(dim),
        "HRC_LOOP_INDEX_SCALE_INIT": f"{scale:g}",
    }


H100_1X_LOSS_NEXT: list[dict[str, Any]] = [
    {
        "name": "h100_loss_e1088_bpbweighted_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env(
            {
                **e1088_best_tail(),
                "BPB_WEIGHTED_LOSS": "1",
            }
        ),
        "notes": "same best under-cap HRC/VocabMoE spine, but train CE is weighted toward original-byte BPB",
    },
    {
        "name": "h100_loss_e1088_warm2600_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env(
            {
                **e1088_best_tail(),
                "WARMDOWN_ITERS": "2600",
            }
        ),
        "notes": "schedule-only probe: align warmdown to the roughly 2600-step 1xH100 10-minute horizon",
    },
    {
        "name": "h100_loss_e1088_bpbweighted_warm2600_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env(
            {
                **e1088_best_tail(),
                "BPB_WEIGHTED_LOSS": "1",
                "WARMDOWN_ITERS": "2600",
            }
        ),
        "notes": "most likely schedule/loss interaction row for lowering actual validation BPB",
    },
    {
        "name": "h100_loss_d832e640_bpbweighted_warm2600_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=832, embed_dim=640, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "BPB_WEIGHTED_LOSS": "1",
                "WARMDOWN_ITERS": "2600",
                "QK_GAIN_INIT": "5.5",
            }
        ),
        "notes": "spend bytes on body width rather than more embedding rank; preflight rejects if size is unsafe",
    },
    {
        "name": "h100_loss_e896_bpbweighted_warm2600_q16q8q8io_q8core_lqer12t24_vocabmoe_qk55",
        "env": h100_env(
            {
                **cap16_nospeed_base(model_dim=768, embed_dim=896, io_width=3, loop_width=5, repeats=2),
                **hrc_core_mlp_env(3, 5, keep_attn_blocks=(3,)),
                **BEST_CLEAN_VOCABMOE,
                **cap16_taper_env(io_width=3, loop_width=5, io_bits=(16, 8, 8), core_bits=8),
                **q8_train_export_env(),
                **leaderboard_schedule_env(min_lr=0.10),
                **cap16_lqer_env(12, 24),
                "BPB_WEIGHTED_LOSS": "1",
                "WARMDOWN_ITERS": "2600",
                "QK_GAIN_INIT": "5.5",
            }
        ),
        "notes": "clean precision-ladder row: q16/q8/q8 IO tail and q8 recurrent core from step zero",
    },
]


H100_1X_WARMDOWN_FIX: list[dict[str, Any]] = [
    {
        "name": "h100_fix_e1088_warm2600_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **e1088_best_tail(),
            },
            {"WARMDOWN_ITERS": "2600"},
        ),
        "notes": "corrected schedule-only row after fixing h100_env override order",
    },
    {
        "name": "h100_fix_e1088_warm2200_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **e1088_best_tail(),
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": "more aggressive warmdown point around the observed 1xH100 step horizon",
    },
]


H100_1X_WARMDOWN_CAPFIT: list[dict[str, Any]] = [
    {
        "name": "h100_capwarm_e832_w2200_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            warmdown_capfit_tail(embed_dim=832),
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "cap-fit version of the best warmdown signal: trims factored "
            "embedding rank enough that the trained artifact should land near "
            "the 16MB line"
        ),
    },
    {
        "name": "h100_capwarm_e896_w2200_wd02_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            warmdown_capfit_tail(embed_dim=896, weight_decay=0.02),
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "tests whether a little compressibility-aware Muon WD lets us keep "
            "more token rank while preserving the strong 2200 warmdown quality"
        ),
    },
    {
        "name": "h100_capwarm_e1088_w2200_wd04_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            warmdown_capfit_tail(embed_dim=1088, weight_decay=0.04),
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "pure entropy-control row: same capacity as the 1.4045 BPB "
            "over-cap result, using Muon WD as the byte-saving lever"
        ),
    },
    {
        "name": "h100_capwarm_e1088_w2200_wd08_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            warmdown_capfit_tail(embed_dim=1088, weight_decay=0.08),
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "aggressive leaderboard-style WD row; likely size-safe if WD "
            "behaves like the public SP8192 transformer stacks"
        ),
    },
]


H100_1X_BASELINE_CHASE: list[dict[str, Any]] = [
    {
        "name": "h100_chase_i3l5r3_d768e832_q8_coreattn1_lidx_lqer12t24_vocabmoe_qk55_w1900",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=3),
                **loop_index_env(),
            },
            {"WARMDOWN_ITERS": "1900"},
        ),
        "notes": (
            "clean continuation of the 1.4052 BPB row: same e832 q8 train/export, "
            "one attention-capable core block, VocabMoE, QK 5.5, Polar/MIN_LR, "
            "and LQER r12/t24, changing only r3 plus loop index"
        ),
    },
    {
        "name": "h100_chase_i3l5r4_d768e832_q8_coreattn1_lidx_lqer12t24_vocabmoe_qk55_w1700",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=4),
                **loop_index_env(),
            },
            {"WARMDOWN_ITERS": "1700"},
        ),
        "notes": (
            "same proven q8/e832 spine as the 1.4052 row, but with r4 recurrence "
            "and loop index; warmdown is shortened for the slower virtual depth"
        ),
    },
    {
        "name": "h100_chase_i3l5r5_d768e832_q8_coreattn1_lidx_lqer12t24_vocabmoe_qk55_w1500",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=5),
                **loop_index_env(),
            },
            {"WARMDOWN_ITERS": "1500"},
        ),
        "notes": (
            "highest-repeat loop-index row; still preserves the same q8/e832 "
            "core-attention/VocabMoE/LQER stack that produced the 1.4052 result"
        ),
    },
    {
        "name": "h100_chase_asym_i3l5r4_d768e832_q8_coreattn1_lidx_lqer12t24_vocabmoe_qk55_w1700",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=4),
                **loop_index_env(),
                "ASYM_LOGIT_RESCALE": "1",
            },
            {"WARMDOWN_ITERS": "1700"},
        ),
        "notes": (
            "direct late-leaderboard Asymmetric Logit Rescale A/B on the r4 "
            "loop-index row, while preserving the same 1.4052 q8/e832 spine"
        ),
    },
    {
        "name": "h100_chase_dual_i3l5r4_d768e768_q8_coreattn1_lidx_left256_lqer12t24_vocabmoe_qk55_w1700",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=768,
                    embed_dim=768,
                    repeats=4,
                    lqer_rank=12,
                    lqer_top_k=24,
                ),
                **loop_index_env(),
                **dual_stream_env(left_dim=256, rank=16, sites="input,loop_first,loop_exit,pre_output"),
            },
            {"WARMDOWN_ITERS": "1700"},
        ),
        "notes": (
            "trained left/right advisor row on the r4 loop-index precision "
            "ladder; this keeps the novelty hedge aligned with the stronger "
            "deeper-loop hypothesis"
        ),
    },
]


H100_1X_BEST_R2_FOLLOWUP: list[dict[str, Any]] = [
    {
        "name": "h100_best_r2_asym_e832_w2200_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=2),
                "ASYM_LOGIT_RESCALE": "1",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "clean A/B for Asymmetric Logit Rescale on the actual 1.4052 BPB "
            "winner; the earlier asym row used the now-weaker r4 loop"
        ),
    },
    {
        "name": "h100_best_r2_ttt24_e832_w2200_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=2),
                **legal_ttt_env(lr=0.005, updates=24),
                "H100_1X_VAL_BATCH_SIZE": "4096",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "legal score-first control-parameter TTT on the best r2 spine; "
            "tests a current-leaderboard eval-time lever without changing the "
            "training architecture"
        ),
    },
    {
        "name": "h100_best_r2_dual_e768_left128_w2200_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=768, repeats=2),
                **dual_stream_env(left_dim=128, rank=16, sites="input,loop_first,loop_exit,pre_output"),
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "conservative trained left/right advisor retest on the fast r2 "
            "spine; e768/left128 is chosen so 1-step preflight can reject it "
            "cleanly if the bridge pushes the artifact over 16MB"
        ),
    },
]


H100_1X_BEAT14_QUEUE: list[dict[str, Any]] = [
    {
        "name": "h100_beat14_r2_asym_ttt24_e832_w2200_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=2),
                **legal_ttt_env(lr=0.005, updates=24),
                "ASYM_LOGIT_RESCALE": "1",
                "H100_1X_VAL_BATCH_SIZE": "4096",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "combo row: asymmetric logits plus meaningful score-first TTT on "
            "the current 1.4052 BPB r2/e832 winner"
        ),
    },
    {
        "name": "h100_beat14_r2_seq512_e832_w2200_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=2),
            },
            {
                "WARMDOWN_ITERS": "2200",
                "TRAIN_SEQ_LEN": "512",
            },
        ),
        "notes": (
            "training-shape row: shorter chunks may buy more optimizer updates "
            "inside the same wall clock without changing the artifact"
        ),
    },
    {
        "name": "h100_beat14_r2_batch32k_e832_w2200_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=2),
                "H100_1X_TRAIN_BATCH_TOKENS": "32768",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "training-shape row: smaller batch trades fewer tokens per step for "
            "more optimizer updates"
        ),
    },
    {
        "name": "h100_beat14_r2_hourglass_e832_width512640768_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=2),
                "LAYER_WIDTH_SCHEDULE": "512,640,768,768,768,768,768,768",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "hourglass architecture: narrower token-facing IO blocks and full "
            "width recurrent core"
        ),
    },
    {
        "name": "h100_beat14_r2_q16q8q8io_width512640768_e832_coreq8_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=2),
                **q8_core_io_ladder_env((16, 8, 8)),
                "LAYER_WIDTH_SCHEDULE": "512,640,768,768,768,768,768,768",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "precision-plus-width ladder: q16/q8/q8 IO tail from step zero, "
            "q8 core, and hourglass block widths"
        ),
    },
    {
        "name": "h100_beat14_i4l4r2_d768e832_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            baseline_chase_tail(
                model_dim=768,
                embed_dim=832,
                io_width=4,
                loop_width=4,
                repeats=2,
                keep_attn_blocks=(4,),
            ),
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "different HRC partition: more mirrored IO-tail capacity and a "
            "shorter recurrent core, same total unique blocks"
        ),
    },
    {
        "name": "h100_beat14_r2_spike32top4_e832_q8_coreattn1_lqer12t24_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=2),
                **spike_vocab_moe_env(
                    experts=32,
                    rank=2,
                    mode="hybrid",
                    layers="input,loop_first",
                    train_quant_bits=8,
                    spike_top_k=4,
                ),
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "spiking/self-election VocabMoE row: more experts with hard top-k "
            "token election instead of only dense token prior mixing"
        ),
    },
    {
        "name": "h100_beat14_r2_bigram10240_d768e768_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=768, repeats=2),
                **bigram_hash_env(vocab_size=10240, dim=32),
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "token-feature architecture: adds compact BigramHash side features "
            "on the HRC/VocabMoE spine"
        ),
    },
    {
        "name": "h100_beat14_r2_rlm_loopfirst_e832_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=832, repeats=2),
                **rlm_memory_env(inject="loop_first", decay=0.90, scale=0.02),
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "legal RLM-lite architecture: recurrent document memory injects "
            "already-scored prefix information into later chunks"
        ),
    },
    {
        "name": "h100_beat14_r2_dynamic_council_e768_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=768, embed_dim=768, repeats=2),
                **dynamic_council_env(threshold=6.0, min_gate=0.01),
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "conditional council architecture: only asks a mirrored peer when "
            "the distribution is uncertain"
        ),
    },
]


H100_1X_BATCH32K_LEGAL: list[dict[str, Any]] = [
    {
        "name": "h100_batch32k_legal_d704e832_w2200_q8_coreattn1_lqer12t24_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(model_dim=704, embed_dim=832, repeats=2),
                "H100_1X_TRAIN_BATCH_TOKENS": "32768",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "cap-safe follow-up to the 1.3611 BPB but over-cap batch32k row: "
            "preserves the smaller-batch/more-optimizer-steps recipe, q8 "
            "train/export, VocabMoE, LQER r12/t24, QK 5.5, and one attention "
            "core block, but shrinks the body from d768 to d704 so the final "
            "lzma artifact should fit under 16MB"
        ),
    },
]


H100_1X_BATCH_STEP_SWEEP: list[dict[str, Any]] = [
    {
        "name": "h100_batch32k_d704e832_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=832,
                    repeats=2,
                    lqer_rank=10,
                    lqer_top_k=20,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "32768",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "legalized anchor for the 1.3595 BPB d704/e832 batch32k row: "
            "keeps the fast smaller-batch recipe but trims LQER from r12/t24 "
            "to r10/t20 to save the final ~13KB cap miss"
        ),
    },
    {
        "name": "h100_batch24k_d704e832_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=832,
                    repeats=2,
                    lqer_rank=10,
                    lqer_top_k=20,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "24576",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "tests whether going below 32k buys enough extra optimizer steps "
            "to beat the current 1.3595 over-cap signal without starving the H100"
        ),
    },
    {
        "name": "h100_batch16k_d704e832_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=832,
                    repeats=2,
                    lqer_rank=10,
                    lqer_top_k=20,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "16384",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "aggressive optimizer-step row: 16k batch should maximize updates "
            "inside 10 minutes; this tells us if noise/underutilization becomes "
            "worse than the extra steps"
        ),
    },
]


H100_1X_BATCH_REPEAT_SWEEP: list[dict[str, Any]] = [
    {
        "name": "h100_batch32k_r3_d704e832_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=832,
                    repeats=3,
                    lqer_rank=10,
                    lqer_top_k=20,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "32768",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "tests the user's hypothesis that the smaller-batch speed gain "
            "lets us afford more recurrent passes again; same legalized "
            "d704/e832/lqer10t20 stack as the batch sweep, but r3"
        ),
    },
    {
        "name": "h100_batch32k_r4_d704e832_w2000_q8_coreattn1_lqer10t20_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=832,
                    repeats=4,
                    lqer_rank=10,
                    lqer_top_k=20,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "32768",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "2000"},
        ),
        "notes": (
            "higher-repeat stress row after batch32k proved fast; warmdown is "
            "slightly shorter because r4 should take fewer optimizer steps"
        ),
    },
]


H100_1X_BATCH24_LEGALIZE: list[dict[str, Any]] = [
    {
        "name": "h100_batch24k_d704e768_w3000_q8_coreattn1_lqer10t20_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=768,
                    repeats=2,
                    lqer_rank=10,
                    lqer_top_k=20,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "24576",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "3000"},
        ),
        "notes": (
            "legalized version of the best raw 24k result: spend less on the "
            "factored tied embedding (e768 instead of e832) and lengthen "
            "warmdown for the ~6200-step 24k horizon"
        ),
    },
    {
        "name": "h100_batch24k_d704e832_w3000_q8_coreattn1_lqer8t16_wd02_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=832,
                    repeats=2,
                    lqer_rank=8,
                    lqer_top_k=16,
                    weight_decay=0.02,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "24576",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "3000"},
        ),
        "notes": (
            "tries to keep the e832 token interface from the 1.3555 raw row "
            "while saving bytes through smaller LQER and mild compression-aware "
            "Muon weight decay"
        ),
    },
    {
        "name": "h100_batch20k_d704e768_w3200_q8_coreattn1_lqer10t20_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=768,
                    repeats=2,
                    lqer_rank=10,
                    lqer_top_k=20,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "20480",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "3200"},
        ),
        "notes": (
            "middle-point batch test between the strong 24k and weaker 16k rows; "
            "keeps the legal e768/e10t20 package and gives the longer run a "
            "slightly longer warmdown"
        ),
    },
]


H100_1X_BATCH12_REPEAT: list[dict[str, Any]] = [
    {
        "name": "h100_batch12k_r3_d704e768_w3200_q8_coreattn1_lqer8t16_wd02_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=768,
                    repeats=3,
                    lqer_rank=8,
                    lqer_top_k=16,
                    weight_decay=0.02,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "12288",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "3200"},
        ),
        "notes": (
            "tests the user's 12k plus extra-recurrence idea while keeping the "
            "artifact likely legal via e768, smaller LQER, and mild Muon WD"
        ),
    },
    {
        "name": "h100_batch12k_r4_d704e768_w3000_q8_coreattn1_lqer8t16_wd02_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=768,
                    repeats=4,
                    lqer_rank=8,
                    lqer_top_k=16,
                    weight_decay=0.02,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "12288",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "3000"},
        ),
        "notes": (
            "r4 companion to the 12k/r3 row; this should only be worth running "
            "if r3 shows the smaller-batch/extra-depth interaction is promising"
        ),
    },
]


H100_1X_BEAT135_EXPORTFIX: list[dict[str, Any]] = [
    {
        "name": "h100_beat135_32k_d704e864_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=864,
                    repeats=2,
                    lqer_rank=10,
                    lqer_top_k=20,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "32768",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "current best legal row is d704/e832/r10t20 at 1.3569 BPB with "
            "about 342KB headroom; this spends that headroom on the factored "
            "token interface without changing the proven 32k optimizer rhythm"
        ),
    },
    {
        "name": "h100_beat135_24k_d704e800_w2200_q8_coreattn1_lqer10t20_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=800,
                    repeats=2,
                    lqer_rank=10,
                    lqer_top_k=20,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "24576",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "2200"},
        ),
        "notes": (
            "minimal legalizer for the raw 24k winner: the over-cap e832 row "
            "hit 1.3555 BPB, so this only trims factored embed rank while "
            "preserving the original warmdown, LQER r10/t20, and no-WD recipe"
        ),
    },
    {
        "name": "h100_beat135_20k_d704e768_w3200_q8_coreattn1_lqer11t22_vocabmoe_qk55",
        "env": h100_env_with_overrides(
            {
                **baseline_chase_tail(
                    model_dim=704,
                    embed_dim=768,
                    repeats=2,
                    lqer_rank=11,
                    lqer_top_k=22,
                ),
                "H100_1X_TRAIN_BATCH_TOKENS": "20480",
                "H100_1X_VAL_BATCH_SIZE": "32768",
            },
            {"WARMDOWN_ITERS": "3200"},
        ),
        "notes": (
            "targets the 20k/e768 export gap directly: the proxy scored "
            "1.3533 BPB, but the r10/t20 export roundtrip fell to 1.3642; "
            "r11/t22 should spend the remaining legal headroom on repair"
        ),
    },
]


GROUPS: dict[str, list[dict[str, Any]]] = {
    "h100_novel_round1": H100_NOVEL_ROUND1,
    "h100_novel_round2_seeds": H100_NOVEL_ROUND2_SEEDS,
    "h100_caplegal_followup": H100_CAPLEGAL_FOLLOWUP,
    "h100_1x_arch_scout": H100_1X_ARCH_SCOUT,
    "h100_1x_precision_width_scout": H100_1X_PRECISION_WIDTH_SCOUT,
    "h100_1x_capfit_scout": H100_1X_CAPFIT_SCOUT,
    "h100_1x_leader_transfer": H100_1X_LEADER_TRANSFER,
    "h100_1x_loss_next": H100_1X_LOSS_NEXT,
    "h100_1x_warmdown_fix": H100_1X_WARMDOWN_FIX,
    "h100_1x_warmdown_capfit": H100_1X_WARMDOWN_CAPFIT,
    "h100_1x_baseline_chase": H100_1X_BASELINE_CHASE,
    "h100_1x_best_r2_followup": H100_1X_BEST_R2_FOLLOWUP,
    "h100_1x_beat14_queue": H100_1X_BEAT14_QUEUE,
    "h100_1x_batch32k_legal": H100_1X_BATCH32K_LEGAL,
    "h100_1x_batch_step_sweep": H100_1X_BATCH_STEP_SWEEP,
    "h100_1x_batch_repeat_sweep": H100_1X_BATCH_REPEAT_SWEEP,
    "h100_1x_batch24_legalize": H100_1X_BATCH24_LEGALIZE,
    "h100_1x_batch12_repeat": H100_1X_BATCH12_REPEAT,
    "h100_1x_beat135_exportfix": H100_1X_BEAT135_EXPORTFIX,
    "all": (
        H100_NOVEL_ROUND1
        + H100_NOVEL_ROUND2_SEEDS
        + H100_CAPLEGAL_FOLLOWUP
        + H100_1X_ARCH_SCOUT
        + H100_1X_PRECISION_WIDTH_SCOUT
        + H100_1X_CAPFIT_SCOUT
        + H100_1X_LEADER_TRANSFER
        + H100_1X_LOSS_NEXT
        + H100_1X_WARMDOWN_FIX
        + H100_1X_WARMDOWN_CAPFIT
        + H100_1X_BASELINE_CHASE
        + H100_1X_BEST_R2_FOLLOWUP
        + H100_1X_BEAT14_QUEUE
        + H100_1X_BATCH32K_LEGAL
        + H100_1X_BATCH_STEP_SWEEP
        + H100_1X_BATCH_REPEAT_SWEEP
        + H100_1X_BATCH24_LEGALIZE
        + H100_1X_BATCH12_REPEAT
        + H100_1X_BEAT135_EXPORTFIX
    ),
}


def selected_candidates(raw: str, group: str) -> list[dict[str, Any]]:
    if group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; choices: {', '.join(sorted(GROUPS))}")
    pool = GROUPS[group]
    if not raw.strip():
        return pool
    wanted = {item.strip() for item in raw.split(",") if item.strip()}
    out = [candidate for candidate in pool if candidate["name"] in wanted]
    missing = wanted - {candidate["name"] for candidate in out}
    if missing:
        raise ValueError(f"unknown candidates for {group}: {', '.join(sorted(missing))}")
    return out


def run_matrix(
    *,
    out_dir: Path,
    candidates: list[dict[str, Any]],
    data_path: Path,
    tokenizer_path: Path,
    nproc_per_node: int,
    wallclock_seconds: int,
    val_tokens: int,
    timeout: int,
    final_artifacts: bool,
    iterations: int | None,
    warmdown_iters: int | None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        run_id = f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
        env = configure_env()
        env.update(candidate["env"])
        env.update(
            {
                "RUN_ID": run_id,
                "DATA_PATH": str(data_path),
                "TOKENIZER_PATH": str(tokenizer_path),
                "VOCAB_SIZE": "8192",
                "MAX_WALLCLOCK_SECONDS": str(wallclock_seconds),
                "VAL_TOKENS_LIMIT": str(val_tokens),
                "SKIP_FINAL_ARTIFACTS": "0" if final_artifacts else "1",
                "PYTHONUNBUFFERED": "1",
            }
        )
        if nproc_per_node == 1:
            env.update(
                {
                    key: env.get(f"H100_1X_{key}", os.environ.get(f"H100_1X_{key}", value))
                    for key, value in H100_1X_DEFAULTS.items()
                }
            )
        if iterations is not None:
            env["ITERATIONS"] = str(iterations)
        if warmdown_iters is not None:
            env["WARMDOWN_ITERS"] = str(warmdown_iters)
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(nproc_per_node),
            str(TRAINER),
        ]
        raw_path = out_dir / f"train_{name}.txt"
        started = time.perf_counter()
        proc = run_command_live(cmd, env, timeout=timeout, live_path=raw_path, label=name)
        stdout = merged_train_output(proc.stdout, run_id)
        raw_path.write_text(stdout, encoding="utf-8")
        row: dict[str, object] = {
            "candidate": name,
            "nproc_per_node": nproc_per_node,
            "wallclock_seconds": wallclock_seconds,
            "iterations": env.get("ITERATIONS", ""),
            "warmdown_iters": env.get("WARMDOWN_ITERS", ""),
            "val_tokens": val_tokens,
            "final_artifacts": int(final_artifacts),
            "returncode": proc.returncode,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "raw_log": raw_path.name,
            "run_id": run_id,
            "env_overrides": ",".join(f"{key}={value}" for key, value in sorted(candidate["env"].items())),
        }
        parsed = parse_train(stdout)
        if parsed:
            row.update(parsed)
        if proc.returncode != 0:
            row["error_tail"] = "\n".join(stdout.splitlines()[-12:])
        rows.append(row)
        write_csv(out_dir / "train.csv", rows)
        write_summary(out_dir, rows)
        print(f"train {name} rc={proc.returncode}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--candidate-group", default="h100_novel_round1", choices=sorted(GROUPS))
    parser.add_argument("--candidates", default="")
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--wallclock-seconds", type=int, default=600)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--warmdown-iters", type=int, default=-1)
    parser.add_argument("--val-tokens", type=int, default=131072)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--data-path", default=str(DATASET_DIR))
    parser.add_argument("--tokenizer-path", default=str(TOKENIZER_PATH))
    parser.add_argument("--final-artifacts", action="store_true", default=True)
    parser.add_argument("--skip-final-artifacts", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    candidates = selected_candidates(args.candidates, args.candidate_group)
    if args.list:
        for candidate in candidates:
            print(candidate["name"])
        return 0
    if not TRAINER.is_file():
        print(f"Missing trainer: {TRAINER}", file=sys.stderr)
        return 1
    data_path = Path(args.data_path)
    tokenizer_path = Path(args.tokenizer_path)
    if not data_path.is_dir() or not tokenizer_path.is_file():
        print(f"Missing CaseOps SP8192 data/tokenizer:\n  data={data_path}\n  tokenizer={tokenizer_path}")
        return 1
    out_dir = Path(args.out) if args.out else ROOT / "records" / f"h100-novel-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    iterations = args.iterations if args.iterations > 0 else None
    warmdown_iters = args.warmdown_iters if args.warmdown_iters >= 0 else None
    write_candidate_plan(out_dir, candidates, iterations or 1_000_000, args.val_tokens)
    run_matrix(
        out_dir=out_dir,
        candidates=candidates,
        data_path=data_path,
        tokenizer_path=tokenizer_path,
        nproc_per_node=args.nproc_per_node,
        wallclock_seconds=args.wallclock_seconds,
        val_tokens=args.val_tokens,
        timeout=args.timeout,
        final_artifacts=bool(args.final_artifacts and not args.skip_final_artifacts),
        iterations=iterations,
        warmdown_iters=warmdown_iters,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
