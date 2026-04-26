"""Sub-4MB ternary entrypoint.

This wrapper keeps the experimental defaults out of the main trainer. It sets
the small-artifact ternary defaults before importing train_gpt, whose
Hyperparameters are evaluated at import time.
"""

from __future__ import annotations

import os


USER_ENV_KEYS = set(os.environ)


SUB4_PROFILES = {
    "i3l5r2_d384_e128": {
        "MODEL_DIM": "384",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "8",
        "EFFECTIVE_DEPTH": "16",
        "HRC_RECURSIVE_CORE_START": "3",
        "HRC_ROUTE_REPEATS": "2",
    },
    "i4l5r2_d384_e128": {
        "MODEL_DIM": "384",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "9",
        "EFFECTIVE_DEPTH": "18",
        "HRC_RECURSIVE_CORE_START": "4",
        "HRC_ROUTE_REPEATS": "2",
    },
    "i2l3r2_d512_e128": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
    },
    "i2l3r2_d512_e128_medscale": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "TRAIN_TERNARY_SCALE_STAT": "median",
        "QUANT_TERNARY_SCALE_STAT": "median",
    },
    "i2l3r2_d512_e128_hadamard": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "BITNET_V2_HADAMARD": "1",
    },
    "i2l3r2_d512_e128_fp16ternary": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "TRAIN_TERNARY_PARAM_DTYPE": "model",
    },
    "i2l3r2_d512_e128_mlp25": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "2.5",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
    },
    "i2l3r2_d512_e128_mlp2": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "2.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
    },
    "i2l3r2_d448_e128": {
        "MODEL_DIM": "448",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "7",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "TRAIN_TERNARY_GROUP_SIZE": "64",
        "QUANT_TERNARY_GROUP_SIZE": "64",
    },
    "i2l3r2_d384_e128": {
        "MODEL_DIM": "384",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i2l3r2_d384_e128_h6": {
        "MODEL_DIM": "384",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "6",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i2l3r2_d384_e128_h6mha": {
        "MODEL_DIM": "384",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "6",
        "NUM_KV_HEADS": "6",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i2l3r2_d320_e96_h5": {
        "MODEL_DIM": "320",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "5",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i2l3r2_d320_e96_h5mha": {
        "MODEL_DIM": "320",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "5",
        "NUM_KV_HEADS": "5",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i2l3r2_d256_e96_h4": {
        "MODEL_DIM": "256",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i2l3r2_d256_e96_h4mha": {
        "MODEL_DIM": "256",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i2l2r2_d320_e96_h5mha": {
        "MODEL_DIM": "320",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "5",
        "NUM_KV_HEADS": "5",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "4",
        "EFFECTIVE_DEPTH": "8",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d320_e96_h5mha": {
        "MODEL_DIM": "320",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "5",
        "NUM_KV_HEADS": "5",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d320_e96_h5mha_mlpinner": {
        "MODEL_DIM": "320",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "5",
        "NUM_KV_HEADS": "5",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i2l2r2_d256_e96_h4mha": {
        "MODEL_DIM": "256",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "4",
        "EFFECTIVE_DEPTH": "8",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d256_e96_h4mha_mlpinner": {
        "MODEL_DIM": "256",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d256_e96_h4mha_attninner": {
        "MODEL_DIM": "256",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_ATTN_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d256_e96_h4mha": {
        "MODEL_DIM": "256",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d224_e80_h4mha_mlpinner": {
        "MODEL_DIM": "224",
        "FACTORED_EMBED_DIM": "80",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d192_e80_h3mha_mlpinner": {
        "MODEL_DIM": "192",
        "FACTORED_EMBED_DIM": "80",
        "NUM_HEADS": "3",
        "NUM_KV_HEADS": "3",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d192_e80_h3mha_mlpinner_mlp15": {
        "MODEL_DIM": "192",
        "FACTORED_EMBED_DIM": "80",
        "NUM_HEADS": "3",
        "NUM_KV_HEADS": "3",
        "MLP_MULT": "1.5",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d160_e64_h5mha_mlpinner_mlp15": {
        "MODEL_DIM": "160",
        "FACTORED_EMBED_DIM": "64",
        "NUM_HEADS": "5",
        "NUM_KV_HEADS": "5",
        "MLP_MULT": "1.5",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d128_e64_h4mha_mlpinner_mlp15": {
        "MODEL_DIM": "128",
        "FACTORED_EMBED_DIM": "64",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "1.5",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d128_e64_h4mha_mlpinner_mlp2": {
        "MODEL_DIM": "128",
        "FACTORED_EMBED_DIM": "64",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r2_d96_e48_h3mha_mlpinner_mlp2": {
        "MODEL_DIM": "96",
        "FACTORED_EMBED_DIM": "48",
        "NUM_HEADS": "3",
        "NUM_KV_HEADS": "3",
        "MLP_MULT": "2.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "6",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r3_d96_e48_h3mha_mlpinner_mlp2": {
        "MODEL_DIM": "96",
        "FACTORED_EMBED_DIM": "48",
        "NUM_HEADS": "3",
        "NUM_KV_HEADS": "3",
        "MLP_MULT": "2.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "8",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "3",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l3r2_d96_e48_h3mha_mlpinner_mlp2": {
        "MODEL_DIM": "96",
        "FACTORED_EMBED_DIM": "48",
        "NUM_HEADS": "3",
        "NUM_KV_HEADS": "3",
        "MLP_MULT": "2.0",
        "NUM_UNIQUE_BLOCKS": "4",
        "EFFECTIVE_DEPTH": "8",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "1,2,3",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i2l3r2_d96_e48_h3mha_mlpinner_mlp2": {
        "MODEL_DIM": "96",
        "FACTORED_EMBED_DIM": "48",
        "NUM_HEADS": "3",
        "NUM_KV_HEADS": "3",
        "MLP_MULT": "2.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "2,3,4",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r3_d128_e64_h4mha_mlpinner_mlp2": {
        "MODEL_DIM": "128",
        "FACTORED_EMBED_DIM": "64",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "8",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "3",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l3r2_d128_e64_h4mha_mlpinner_mlp2": {
        "MODEL_DIM": "128",
        "FACTORED_EMBED_DIM": "64",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2.0",
        "NUM_UNIQUE_BLOCKS": "4",
        "EFFECTIVE_DEPTH": "8",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "1,2,3",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i2l3r2_d128_e64_h4mha_mlpinner_mlp2": {
        "MODEL_DIM": "128",
        "FACTORED_EMBED_DIM": "64",
        "NUM_HEADS": "4",
        "NUM_KV_HEADS": "4",
        "MLP_MULT": "2.0",
        "NUM_UNIQUE_BLOCKS": "5",
        "EFFECTIVE_DEPTH": "10",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "2,3,4",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i1l2r3_d192_e80_h3mha_mlpinner": {
        "MODEL_DIM": "192",
        "FACTORED_EMBED_DIM": "80",
        "NUM_HEADS": "3",
        "NUM_KV_HEADS": "3",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "3",
        "EFFECTIVE_DEPTH": "8",
        "HRC_RECURSIVE_CORE_START": "1",
        "HRC_ROUTE_REPEATS": "3",
        "HRC_MLP_ONLY_BLOCKS": "1,2",
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    },
    "i3l5r2_d512_e128_attnloop": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "8",
        "EFFECTIVE_DEPTH": "16",
        "HRC_RECURSIVE_CORE_START": "3",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_ATTN_ONLY_BLOCKS": "3,5,7",
    },
    "i3l5r2_d512_e128_mlploop": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "8",
        "EFFECTIVE_DEPTH": "16",
        "HRC_RECURSIVE_CORE_START": "3",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "3,5,7",
    },
    "i4l8r2_d512_e128_attnloop": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "12",
        "EFFECTIVE_DEPTH": "24",
        "HRC_RECURSIVE_CORE_START": "4",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_ATTN_ONLY_BLOCKS": "4,6,8,10",
    },
    "i4l8r2_d512_e128_mlploop": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "12",
        "EFFECTIVE_DEPTH": "24",
        "HRC_RECURSIVE_CORE_START": "4",
        "HRC_ROUTE_REPEATS": "2",
        "HRC_MLP_ONLY_BLOCKS": "4,6,8,10",
    },
    "i2l4r2_d512_e128": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "6",
        "EFFECTIVE_DEPTH": "12",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
    },
    "i3l4r2_d512_e128": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "7",
        "EFFECTIVE_DEPTH": "14",
        "HRC_RECURSIVE_CORE_START": "3",
        "HRC_ROUTE_REPEATS": "2",
    },
    "i2l5r2_d448_e128": {
        "MODEL_DIM": "448",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "7",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "7",
        "EFFECTIVE_DEPTH": "14",
        "HRC_RECURSIVE_CORE_START": "2",
        "HRC_ROUTE_REPEATS": "2",
        "TRAIN_TERNARY_GROUP_SIZE": "64",
        "QUANT_TERNARY_GROUP_SIZE": "64",
    },
    "i3l3r3_d448_e128": {
        "MODEL_DIM": "448",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "7",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "6",
        "EFFECTIVE_DEPTH": "15",
        "HRC_RECURSIVE_CORE_START": "3",
        "HRC_ROUTE_REPEATS": "3",
        "TRAIN_TERNARY_GROUP_SIZE": "64",
        "QUANT_TERNARY_GROUP_SIZE": "64",
    },
    "i5l7r2_d320_e96": {
        "MODEL_DIM": "320",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "12",
        "EFFECTIVE_DEPTH": "24",
        "HRC_RECURSIVE_CORE_START": "5",
        "HRC_ROUTE_REPEATS": "2",
        "TRAIN_TERNARY_GROUP_SIZE": "64",
        "QUANT_TERNARY_GROUP_SIZE": "64",
    },
    "i6l9r2_d256_e96": {
        "MODEL_DIM": "256",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "15",
        "EFFECTIVE_DEPTH": "30",
        "HRC_RECURSIVE_CORE_START": "6",
        "HRC_ROUTE_REPEATS": "2",
    },
    "i4l6r2_d448_e128": {
        "MODEL_DIM": "448",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "7",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "10",
        "EFFECTIVE_DEPTH": "20",
        "HRC_RECURSIVE_CORE_START": "4",
        "HRC_ROUTE_REPEATS": "2",
        "TRAIN_TERNARY_GROUP_SIZE": "64",
        "QUANT_TERNARY_GROUP_SIZE": "64",
    },
    "i4l8r2_d384_e128": {
        "MODEL_DIM": "384",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "12",
        "EFFECTIVE_DEPTH": "24",
        "HRC_RECURSIVE_CORE_START": "4",
        "HRC_ROUTE_REPEATS": "2",
    },
    "i3l6r3_d384_e128": {
        "MODEL_DIM": "384",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "9",
        "EFFECTIVE_DEPTH": "24",
        "HRC_RECURSIVE_CORE_START": "3",
        "HRC_ROUTE_REPEATS": "3",
    },
    "i5l9r2_d384_e128": {
        "MODEL_DIM": "384",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "14",
        "EFFECTIVE_DEPTH": "28",
        "HRC_RECURSIVE_CORE_START": "5",
        "HRC_ROUTE_REPEATS": "2",
    },
    "i4l6r2_d512_e128": {
        "MODEL_DIM": "512",
        "FACTORED_EMBED_DIM": "128",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "10",
        "EFFECTIVE_DEPTH": "20",
        "HRC_RECURSIVE_CORE_START": "4",
        "HRC_ROUTE_REPEATS": "2",
    },
    "i6l10r2_d320_e96": {
        "MODEL_DIM": "320",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "16",
        "EFFECTIVE_DEPTH": "32",
        "HRC_RECURSIVE_CORE_START": "6",
        "HRC_ROUTE_REPEATS": "2",
        "TRAIN_TERNARY_GROUP_SIZE": "64",
        "QUANT_TERNARY_GROUP_SIZE": "64",
    },
    "i8l12r2_d256_e96": {
        "MODEL_DIM": "256",
        "FACTORED_EMBED_DIM": "96",
        "NUM_HEADS": "8",
        "NUM_KV_HEADS": "1",
        "MLP_MULT": "3.0",
        "NUM_UNIQUE_BLOCKS": "20",
        "EFFECTIVE_DEPTH": "40",
        "HRC_RECURSIVE_CORE_START": "8",
        "HRC_ROUTE_REPEATS": "2",
    },
}


def _i1l2r2_micro_profile(
    model_dim: int,
    embed_dim: int,
    heads: int,
    *,
    mlp_mult: float = 2.0,
    inner_mode: str = "mlp_only",
    num_unique_blocks: int = 3,
    effective_depth: int = 6,
    recursive_core_start: int = 1,
    route_repeats: int = 2,
    inner_blocks: str = "1,2",
    bigram_vocab_size: int = 0,
    bigram_dim: int | None = None,
    ve_layers: str = "",
    ve_dim: int | None = None,
) -> dict[str, str]:
    profile = {
        "MODEL_DIM": str(model_dim),
        "FACTORED_EMBED_DIM": str(embed_dim),
        "NUM_HEADS": str(heads),
        "NUM_KV_HEADS": str(heads),
        "MLP_MULT": str(float(mlp_mult)),
        "NUM_UNIQUE_BLOCKS": str(num_unique_blocks),
        "EFFECTIVE_DEPTH": str(effective_depth),
        "HRC_RECURSIVE_CORE_START": str(recursive_core_start),
        "HRC_ROUTE_REPEATS": str(route_repeats),
        "SDP_BACKEND": "mem_efficient",
        "TRAIN_TERNARY_GROUP_SIZE": "128",
        "QUANT_TERNARY_GROUP_SIZE": "128",
    }
    if inner_mode == "mlp_only":
        profile["HRC_MLP_ONLY_BLOCKS"] = inner_blocks
    elif inner_mode == "attn_only":
        profile["HRC_ATTN_ONLY_BLOCKS"] = inner_blocks
    elif inner_mode != "full":
        raise ValueError(f"Unsupported inner_mode={inner_mode!r}")
    if bigram_vocab_size > 0:
        profile.update(
            {
                "BIGRAM_VOCAB_SIZE": str(bigram_vocab_size),
                "BIGRAM_DIM": str(model_dim if bigram_dim is None else bigram_dim),
                "BIGRAM_INIT_STD": "0.01",
                "BIGRAM_SCALE_INIT": "0.1",
            }
        )
    if ve_layers:
        profile.update(
            {
                "VE_ENABLED": "1",
                "VE_DIM": str(model_dim if ve_dim is None else ve_dim),
                "VE_LAYERS": ve_layers,
            }
        )
    return profile


def _hrc_profile(
    model_dim: int,
    embed_dim: int,
    heads: int,
    *,
    kv_heads: int | None = None,
    mlp_mult: float = 2.0,
    num_unique_blocks: int = 3,
    effective_depth: int = 6,
    recursive_core_start: int = 1,
    route_repeats: int = 2,
    inner_mode: str = "mlp_only",
    inner_blocks: str = "1,2",
    group_size: int = 128,
    sdp_backend: str = "mem_efficient",
) -> dict[str, str]:
    profile = {
        "MODEL_DIM": str(model_dim),
        "FACTORED_EMBED_DIM": str(embed_dim),
        "NUM_HEADS": str(heads),
        "NUM_KV_HEADS": str(heads if kv_heads is None else kv_heads),
        "MLP_MULT": str(float(mlp_mult)),
        "NUM_UNIQUE_BLOCKS": str(num_unique_blocks),
        "EFFECTIVE_DEPTH": str(effective_depth),
        "HRC_RECURSIVE_CORE_START": str(recursive_core_start),
        "HRC_ROUTE_REPEATS": str(route_repeats),
        "SDP_BACKEND": sdp_backend,
        "TRAIN_TERNARY_GROUP_SIZE": str(group_size),
        "QUANT_TERNARY_GROUP_SIZE": str(group_size),
    }
    if inner_mode == "mlp_only":
        profile["HRC_MLP_ONLY_BLOCKS"] = inner_blocks
    elif inner_mode == "attn_only":
        profile["HRC_ATTN_ONLY_BLOCKS"] = inner_blocks
    elif inner_mode != "full":
        raise ValueError(f"Unsupported inner_mode={inner_mode!r}")
    return profile


SUB4_PROFILES.update(
    {
        # Spend headroom on the tied factored IO path first. This costs some
        # output-softmax time, but directly attacks the low-rank vocabulary cap.
        "i1l2r2_d96_e64_h3mha_mlpinner_mlp2": _i1l2r2_micro_profile(96, 64, 3),
        "i1l2r2_d96_e72_h3mha_mlpinner_mlp2": _i1l2r2_micro_profile(96, 72, 3),
        "i1l2r2_d96_e80_h3mha_mlpinner_mlp2": _i1l2r2_micro_profile(96, 80, 3),
        "i1l2r2_d96_e88_h3mha_mlpinner_mlp2": _i1l2r2_micro_profile(96, 88, 3),
        "i1l2r2_d96_e96_h3mha_mlpinner_mlp2": _i1l2r2_micro_profile(96, 96, 3),
        # Quality rescue lanes: keep the fast IO-tail route, but restore
        # repeated attention in the middle. The mlpinner family is probably too
        # attention-starved to approach the larger 16MB losses.
        "i1l2r2_d96_e80_h3mha_attninner_mlp2": _i1l2r2_micro_profile(
            96, 80, 3, inner_mode="attn_only"
        ),
        "i1l2r2_d96_e80_h3mha_fullinner_mlp2": _i1l2r2_micro_profile(
            96, 80, 3, inner_mode="full"
        ),
        "i1l2r2_d128_e80_h4mha_attninner_mlp2": _i1l2r2_micro_profile(
            128, 80, 4, inner_mode="attn_only"
        ),
        "i1l2r2_d128_e80_h4mha_fullinner_mlp2": _i1l2r2_micro_profile(
            128, 80, 4, inner_mode="full"
        ),
        "i1l2r2_d160_e96_h5mha_attninner_mlp2": _i1l2r2_micro_profile(
            160, 96, 5, inner_mode="attn_only"
        ),
        "i1l2r2_d160_e96_h5mha_fullinner_mlp2": _i1l2r2_micro_profile(
            160, 96, 5, inner_mode="full"
        ),
        # Wider/deeper Muon-first lanes. These are intended to spend bytes on
        # reusable trunk capacity while retaining the fast MLP-only loop body.
        "i1l2r2_d192_e96_h3mha_mlpinner_mlp15": _i1l2r2_micro_profile(
            192, 96, 3, mlp_mult=1.5
        ),
        "i1l2r3_d192_e80_h3mha_mlpinner_mlp15": _i1l2r2_micro_profile(
            192, 80, 3, mlp_mult=1.5, effective_depth=8, route_repeats=3
        ),
        "i1l3r2_d192_e80_h3mha_mlpinner_mlp15": _i1l2r2_micro_profile(
            192,
            80,
            3,
            mlp_mult=1.5,
            num_unique_blocks=4,
            effective_depth=8,
            inner_blocks="1,2,3",
        ),
        "i2l3r2_d192_e80_h3mha_mlpinner_mlp15": _i1l2r2_micro_profile(
            192,
            80,
            3,
            mlp_mult=1.5,
            num_unique_blocks=5,
            effective_depth=10,
            recursive_core_start=2,
            inner_blocks="2,3,4",
        ),
        "i1l2r2_d224_e80_h4mha_mlpinner_mlp15": _i1l2r2_micro_profile(
            224, 80, 4, mlp_mult=1.5
        ),
        "i1l2r2_d224_e96_h4mha_mlpinner_mlp15": _i1l2r2_micro_profile(
            224, 96, 4, mlp_mult=1.5
        ),
        "i1l2r2_d256_e96_h4mha_mlpinner_mlp15": _i1l2r2_micro_profile(
            256, 96, 4, mlp_mult=1.5
        ),
        # Spend bytes with almost no matmul cost: one learned previous-token hash
        # table, kept at model width so it is lookup+add rather than lookup+proj.
        "i1l2r2_d96_e48_h3mha_mlpinner_mlp2_bigram8k": _i1l2r2_micro_profile(
            96, 48, 3, bigram_vocab_size=8192
        ),
        "i1l2r2_d96_e48_h3mha_mlpinner_mlp2_bigram16k": _i1l2r2_micro_profile(
            96, 48, 3, bigram_vocab_size=16384
        ),
        "i1l2r2_d96_e48_h3mha_mlpinner_mlp2_bigram32k": _i1l2r2_micro_profile(
            96, 48, 3, bigram_vocab_size=32768
        ),
        # Combined lane: modestly richer IO plus a collision-reducing bigram table.
        "i1l2r2_d96_e64_h3mha_mlpinner_mlp2_bigram16k": _i1l2r2_micro_profile(
            96, 64, 3, bigram_vocab_size=16384
        ),
        "i1l2r2_d96_e64_h3mha_mlpinner_mlp2_bigram32k": _i1l2r2_micro_profile(
            96, 64, 3, bigram_vocab_size=32768
        ),
        # Shared value-embedding lanes spend bytes inside attention values while
        # avoiding extra output vocabulary matmul.
        "i1l2r2_d96_e80_h3mha_mlpinner_mlp2_veenc": _i1l2r2_micro_profile(
            96, 80, 3, ve_layers="0,1,2"
        ),
        "i1l2r2_d96_e80_h3mha_mlpinner_mlp2_veall": _i1l2r2_micro_profile(
            96, 80, 3, ve_layers="0,1,2,3,4,5"
        ),
        "i1l2r2_d96_e80_h3mha_mlpinner_mlp2_bigram16k_veall": _i1l2r2_micro_profile(
            96, 80, 3, bigram_vocab_size=16384, ve_layers="0,1,2,3,4,5"
        ),
        # A cautious width step between the current d96 winner and the slower d128.
        "i1l2r2_d112_e56_h4mha_mlpinner_mlp2": _i1l2r2_micro_profile(112, 56, 4),
        "i1l2r2_d112_e56_h4mha_mlpinner_mlp2_bigram16k": _i1l2r2_micro_profile(
            112, 56, 4, bigram_vocab_size=16384
        ),
        # Large-but-damped lanes. The earlier d384/d512 full HRC bodies fit the
        # byte cap but exploded under sprint training. These spend the headroom
        # on width while keeping the recurrent middle cheap and mostly MLP-only.
        "i1l2r2_d320_e96_h8kv1_mlpinner_mlp10": _hrc_profile(
            320, 96, 8, kv_heads=1, mlp_mult=1.0
        ),
        "i1l2r2_d320_e96_h8kv1_mlpinner_mlp15": _hrc_profile(
            320, 96, 8, kv_heads=1, mlp_mult=1.5
        ),
        "i1l2r2_d384_e128_h8kv1_mlpinner_mlp10": _hrc_profile(
            384, 128, 8, kv_heads=1, mlp_mult=1.0
        ),
        "i1l2r2_d384_e128_h8kv1_mlpinner_mlp15": _hrc_profile(
            384, 128, 8, kv_heads=1, mlp_mult=1.5
        ),
        "i1l2r2_d448_e128_h7kv1_mlpinner_mlp10": _hrc_profile(
            448, 128, 7, kv_heads=1, mlp_mult=1.0, group_size=64
        ),
        "i1l2r2_d448_e128_h7kv1_mlpinner_mlp15": _hrc_profile(
            448, 128, 7, kv_heads=1, mlp_mult=1.5, group_size=64
        ),
        "i2l3r2_d320_e96_h8kv1_mlpinner_mlp10": _hrc_profile(
            320,
            96,
            8,
            kv_heads=1,
            mlp_mult=1.0,
            num_unique_blocks=5,
            effective_depth=10,
            recursive_core_start=2,
            inner_blocks="2,3,4",
        ),
        "i2l3r2_d320_e96_h8kv1_mlpinner_mlp15": _hrc_profile(
            320,
            96,
            8,
            kv_heads=1,
            mlp_mult=1.5,
            num_unique_blocks=5,
            effective_depth=10,
            recursive_core_start=2,
            inner_blocks="2,3,4",
        ),
        "i2l3r2_d384_e128_h8kv1_mlpinner_mlp10": _hrc_profile(
            384,
            128,
            8,
            kv_heads=1,
            mlp_mult=1.0,
            num_unique_blocks=5,
            effective_depth=10,
            recursive_core_start=2,
            inner_blocks="2,3,4",
        ),
        "i2l3r2_d384_e128_h8kv1_mlpinner_mlp15": _hrc_profile(
            384,
            128,
            8,
            kv_heads=1,
            mlp_mult=1.5,
            num_unique_blocks=5,
            effective_depth=10,
            recursive_core_start=2,
            inner_blocks="2,3,4",
        ),
        "i1l2r2_d512_e128_h8kv1_mlpinner_mlp075": _hrc_profile(
            512, 128, 8, kv_heads=1, mlp_mult=0.75, group_size=64
        ),
        "i1l2r2_d512_e128_h8kv1_mlpinner_mlp10": _hrc_profile(
            512, 128, 8, kv_heads=1, mlp_mult=1.0, group_size=64
        ),
        "i1l2r2_d512_e160_h8kv1_mlpinner_mlp10": _hrc_profile(
            512, 160, 8, kv_heads=1, mlp_mult=1.0, group_size=64
        ),
        "i1l2r2_d640_e160_h10kv1_mlpinner_mlp075": _hrc_profile(
            640, 160, 10, kv_heads=1, mlp_mult=0.75, group_size=64
        ),
        "i1l2r2_d640_e160_h10kv1_mlpinner_mlp10": _hrc_profile(
            640, 160, 10, kv_heads=1, mlp_mult=1.0, group_size=64
        ),
        "i1l2r2_d640_e224_h10kv1_mlpinner_mlp050": _hrc_profile(
            640, 224, 10, kv_heads=1, mlp_mult=0.5, group_size=64
        ),
        "i1l2r2_d640_e256_h10kv1_mlpinner_mlp050": _hrc_profile(
            640, 256, 10, kv_heads=1, mlp_mult=0.5, group_size=64
        ),
        # Near-cap lanes: spend the unused byte budget on width and a less
        # bottlenecked tied IO rank while keeping only the entry/exit block
        # attention-heavy. These are aimed more at H100-style throughput than
        # the local 2060 sprint path.
        "i1l2r2_d768_e192_h12kv1_mlpinner_mlp050": _hrc_profile(
            768, 192, 12, kv_heads=1, mlp_mult=0.5, group_size=64
        ),
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075": _hrc_profile(
            768, 256, 12, kv_heads=1, mlp_mult=0.75, group_size=64
        ),
        "i1l2r2_d768_e320_h12kv1_mlpinner_mlp050": _hrc_profile(
            768, 320, 12, kv_heads=1, mlp_mult=0.5, group_size=64
        ),
        "i1l2r2_d896_e256_h14kv1_mlpinner_mlp050": _hrc_profile(
            896, 256, 14, kv_heads=1, mlp_mult=0.5, group_size=64
        ),
        "i1l2r2_d1024_e256_h16kv1_mlpinner_mlp050": _hrc_profile(
            1024, 256, 16, kv_heads=1, mlp_mult=0.5, group_size=64
        ),
        "i1l2r2_d1280_e320_h20kv1_mlpinner_mlp050": _hrc_profile(
            1280, 320, 20, kv_heads=1, mlp_mult=0.5, group_size=64
        ),
        "i1l2r2_d1536_e384_h24kv1_mlpinner_mlp050": _hrc_profile(
            1536, 384, 24, kv_heads=1, mlp_mult=0.5, group_size=64
        ),
        "i1l2r2_d1536_e512_h24kv1_mlpinner_mlp025": _hrc_profile(
            1536, 512, 24, kv_heads=1, mlp_mult=0.25, group_size=64
        ),
        "i1l2r2_d1792_e448_h28kv1_mlpinner_mlp050": _hrc_profile(
            1792, 448, 28, kv_heads=1, mlp_mult=0.5, group_size=64
        ),
        "i1l2r2_d2048_e384_h32kv1_mlpinner_mlp025": _hrc_profile(
            2048, 384, 32, kv_heads=1, mlp_mult=0.25, group_size=64
        ),
        "i1l2r2_d2048_e512_h32kv1_mlpinner_mlp025": _hrc_profile(
            2048, 512, 32, kv_heads=1, mlp_mult=0.25, group_size=64
        ),
    }
)


SUB4_TERNARY_COMMON_DEFAULTS = {
    "WARMUP_STEPS": "0",
    "SKIP_INITIAL_VAL": "1",
    "LOG_NVIDIA_SMI": "0",
    "LOG_CODE_SNAPSHOT": "0",
    "OPTIMIZER_PRESET": "adamw",
    "MODEL_FAMILY": "hrc",
    "HRC_DEPTH_SCHEDULE_MODE": "transition_recursive_cycle",
    "HRC_RECUR_INJECT_ENABLED": "1",
    "HRC_LOOP_INDEX_ENABLED": "1",
    "HRC_LOOP_INDEX_DIM": "32",
    "HRC_PASS_EMBED_ENABLED": "1",
    "HRC_PASS_EMBED_MODE": "block_peer",
    "HRC_PASS_ROLE_MODE": "phase5",
    "TRAIN_TERNARY_BLOCKS": "1",
    "TRAIN_TERNARY_GROUP_SIZE": "128",
    "TRAIN_TERNARY_FORWARD_CACHE": "1",
    "TRAIN_TERNARY_SCALE_STAT": "mean",
    "TRAIN_TERNARY_PARAM_DTYPE": "fp32",
    "TRAIN_TERNARY_PACKED_KERNEL": "0",
    "TRAIN_TERNARY_DENSE_KERNEL": "0",
    "TRAIN_FUSED_QKV": "1",
    "BITNET_V2_HADAMARD": "0",
    "FACTORED_EMBED_DIM": "128",
    "MODEL_CODEC": "lzma",
    "MODEL_CODEC_LEVEL": "9",
    "SUBMISSION_SIZE_CAP_BYTES": "4000000",
    "FAIL_ON_ARTIFACT_CAP": "1",
    "QUANT_TERNARY_PATTERNS": "blocks.",
    "QUANT_TERNARY_EXCLUDE_PATTERNS": "tok_emb.weight,lm_head.weight,embed_proj",
    "QUANT_TERNARY_GROUP_SIZE": "128",
    "QUANT_TERNARY_SCALE_STAT": "mean",
    "QUANT_TERNARY_SHRINKAGE_FIX": "1",
    "INT8_KEEP_FLOAT_MAX_NUMEL": "4096",
    "CODE_SNAPSHOT_PATHS": "train_gpt.py,train_gpt_ternary.py,ternary_golf/__init__.py,ternary_golf/layers.py,ternary_golf/packed_cuda.py",
}

SUB4_SPEED_PRESETS = {
    "2060": {
        "TRAIN_SEQ_LEN": "128",
        "TRAIN_BATCH_TOKENS": "4096",
        "VAL_BATCH_SIZE": "4096",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
    },
    "local2060": {
        "TRAIN_SEQ_LEN": "128",
        "TRAIN_BATCH_TOKENS": "4096",
        "VAL_BATCH_SIZE": "4096",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
    },
    "2060max": {
        "TRAIN_SEQ_LEN": "64",
        "TRAIN_BATCH_TOKENS": "4096",
        "VAL_BATCH_SIZE": "4096",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
        "LOSS_FP32": "0",
        "LOSS_TOKEN_STRIDE": "1",
        "LOSS_VOCAB_SAMPLE_SIZE": "0",
        "LOGIT_SOFTCAP": "0",
        "TRAIN_TERNARY_GROUP_SIZE": "256",
        "QUANT_TERNARY_GROUP_SIZE": "256",
    },
    "2060ultra": {
        "TRAIN_SEQ_LEN": "64",
        "TRAIN_BATCH_TOKENS": "4096",
        "VAL_BATCH_SIZE": "4096",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
        "LOSS_FP32": "0",
        "LOSS_TOKEN_STRIDE": "2",
        "LOSS_VOCAB_SAMPLE_SIZE": "0",
        "LOGIT_SOFTCAP": "0",
        "TRAIN_TERNARY_GROUP_SIZE": "256",
        "QUANT_TERNARY_GROUP_SIZE": "256",
    },
    "2060sprint": {
        "TRAIN_SEQ_LEN": "64",
        "TRAIN_BATCH_TOKENS": "4096",
        "VAL_BATCH_SIZE": "4096",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
        "LOSS_FP32": "0",
        "LOSS_TOKEN_STRIDE": "1",
        "LOSS_VOCAB_SAMPLE_SIZE": "0",
        "LOGIT_SOFTCAP": "0",
        "TRAIN_TERNARY_PARAM_DTYPE": "model",
        "TRAIN_CASTED_LINEAR_PARAM_DTYPE": "model",
        "KEEP_CONTROL_PARAMS_FP32": "0",
        "RESID_MIX_ENABLED": "0",
        "BRANCH_SCALE_ENABLED": "1",
        "BRANCH_SCALE_KIND": "vector",
        "HRC_RECUR_INJECT_ENABLED": "0",
        "HRC_LOOP_INDEX_ENABLED": "0",
        "HRC_PASS_EMBED_ENABLED": "0",
        "HRC_PASS_ROLE_MODE": "none",
        "TRAIN_TERNARY_GROUP_SIZE": "256",
        "QUANT_TERNARY_GROUP_SIZE": "256",
    },
    "2060sprint_steps": {
        "TRAIN_SEQ_LEN": "64",
        "TRAIN_BATCH_TOKENS": "2048",
        "VAL_BATCH_SIZE": "2048",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
        "LOSS_FP32": "0",
        "LOSS_TOKEN_STRIDE": "1",
        "LOSS_VOCAB_SAMPLE_SIZE": "0",
        "LOGIT_SOFTCAP": "0",
        "TRAIN_TERNARY_PARAM_DTYPE": "model",
        "TRAIN_CASTED_LINEAR_PARAM_DTYPE": "model",
        "KEEP_CONTROL_PARAMS_FP32": "0",
        "RESID_MIX_ENABLED": "0",
        "BRANCH_SCALE_ENABLED": "1",
        "BRANCH_SCALE_KIND": "vector",
        "HRC_RECUR_INJECT_ENABLED": "0",
        "HRC_LOOP_INDEX_ENABLED": "0",
        "HRC_PASS_EMBED_ENABLED": "0",
        "HRC_PASS_ROLE_MODE": "none",
        "TRAIN_TERNARY_GROUP_SIZE": "256",
        "QUANT_TERNARY_GROUP_SIZE": "256",
    },
    "2060sprint_ultra": {
        "TRAIN_SEQ_LEN": "64",
        "TRAIN_BATCH_TOKENS": "4096",
        "VAL_BATCH_SIZE": "4096",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
        "LOSS_FP32": "0",
        "LOSS_TOKEN_STRIDE": "2",
        "LOSS_VOCAB_SAMPLE_SIZE": "0",
        "LOGIT_SOFTCAP": "0",
        "TRAIN_TERNARY_PARAM_DTYPE": "model",
        "TRAIN_CASTED_LINEAR_PARAM_DTYPE": "model",
        "KEEP_CONTROL_PARAMS_FP32": "0",
        "RESID_MIX_ENABLED": "0",
        "BRANCH_SCALE_ENABLED": "1",
        "BRANCH_SCALE_KIND": "vector",
        "HRC_RECUR_INJECT_ENABLED": "0",
        "HRC_LOOP_INDEX_ENABLED": "0",
        "HRC_PASS_EMBED_ENABLED": "0",
        "HRC_PASS_ROLE_MODE": "none",
        "TRAIN_TERNARY_GROUP_SIZE": "256",
        "QUANT_TERNARY_GROUP_SIZE": "256",
    },
    "2060sprint_turbo": {
        "TRAIN_SEQ_LEN": "64",
        "TRAIN_BATCH_TOKENS": "2048",
        "VAL_BATCH_SIZE": "2048",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
        "LOSS_FP32": "0",
        "LOSS_TOKEN_STRIDE": "4",
        "LOSS_VOCAB_SAMPLE_SIZE": "0",
        "LOGIT_SOFTCAP": "0",
        "TRAIN_TERNARY_PARAM_DTYPE": "model",
        "TRAIN_CASTED_LINEAR_PARAM_DTYPE": "model",
        "KEEP_CONTROL_PARAMS_FP32": "0",
        "RESID_MIX_ENABLED": "0",
        "BRANCH_SCALE_ENABLED": "1",
        "BRANCH_SCALE_KIND": "vector",
        "HRC_RECUR_INJECT_ENABLED": "0",
        "HRC_LOOP_INDEX_ENABLED": "0",
        "HRC_PASS_EMBED_ENABLED": "0",
        "HRC_PASS_ROLE_MODE": "none",
        "TRAIN_TERNARY_GROUP_SIZE": "256",
        "QUANT_TERNARY_GROUP_SIZE": "256",
    },
    "2060sprint_throughput": {
        "TRAIN_SEQ_LEN": "64",
        "TRAIN_BATCH_TOKENS": "8192",
        "VAL_BATCH_SIZE": "8192",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
        "LOSS_FP32": "0",
        "LOSS_TOKEN_STRIDE": "2",
        "LOSS_VOCAB_SAMPLE_SIZE": "0",
        "LOGIT_SOFTCAP": "0",
        "TRAIN_TERNARY_PARAM_DTYPE": "model",
        "TRAIN_CASTED_LINEAR_PARAM_DTYPE": "model",
        "KEEP_CONTROL_PARAMS_FP32": "0",
        "RESID_MIX_ENABLED": "0",
        "BRANCH_SCALE_ENABLED": "1",
        "BRANCH_SCALE_KIND": "vector",
        "HRC_RECUR_INJECT_ENABLED": "0",
        "HRC_LOOP_INDEX_ENABLED": "0",
        "HRC_PASS_EMBED_ENABLED": "0",
        "HRC_PASS_ROLE_MODE": "none",
        "TRAIN_TERNARY_GROUP_SIZE": "256",
        "QUANT_TERNARY_GROUP_SIZE": "256",
    },
    "2060sprint_micro": {
        "TRAIN_SEQ_LEN": "64",
        "TRAIN_BATCH_TOKENS": "4096",
        "VAL_BATCH_SIZE": "4096",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
        "LOSS_FP32": "0",
        "LOSS_TOKEN_STRIDE": "2",
        "LOSS_VOCAB_SAMPLE_SIZE": "0",
        "LOGIT_SOFTCAP": "0",
        "TRAIN_TERNARY_PARAM_DTYPE": "model",
        "TRAIN_CASTED_LINEAR_PARAM_DTYPE": "model",
        "KEEP_CONTROL_PARAMS_FP32": "0",
        "RESID_MIX_ENABLED": "0",
        "BRANCH_SCALE_ENABLED": "1",
        "BRANCH_SCALE_KIND": "vector",
        "HRC_RECUR_INJECT_ENABLED": "0",
        "HRC_LOOP_INDEX_ENABLED": "0",
        "HRC_PASS_EMBED_ENABLED": "0",
        "HRC_PASS_ROLE_MODE": "none",
        "TRAIN_TERNARY_GROUP_SIZE": "256",
        "QUANT_TERNARY_GROUP_SIZE": "256",
        "TIED_EMBED_LR": "0.0125",
        "MATRIX_LR": "0.01",
        "SCALAR_LR": "0.01",
    },
    "2060sprint_micro_throughput": {
        "TRAIN_SEQ_LEN": "64",
        "TRAIN_BATCH_TOKENS": "8192",
        "VAL_BATCH_SIZE": "8192",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
        "LOSS_FP32": "0",
        "LOSS_TOKEN_STRIDE": "2",
        "LOSS_VOCAB_SAMPLE_SIZE": "0",
        "LOGIT_SOFTCAP": "0",
        "TRAIN_TERNARY_PARAM_DTYPE": "model",
        "TRAIN_CASTED_LINEAR_PARAM_DTYPE": "model",
        "KEEP_CONTROL_PARAMS_FP32": "0",
        "RESID_MIX_ENABLED": "0",
        "BRANCH_SCALE_ENABLED": "1",
        "BRANCH_SCALE_KIND": "vector",
        "HRC_RECUR_INJECT_ENABLED": "0",
        "HRC_LOOP_INDEX_ENABLED": "0",
        "HRC_PASS_EMBED_ENABLED": "0",
        "HRC_PASS_ROLE_MODE": "none",
        "TRAIN_TERNARY_GROUP_SIZE": "256",
        "QUANT_TERNARY_GROUP_SIZE": "256",
        "TIED_EMBED_LR": "0.0125",
        "MATRIX_LR": "0.01",
        "SCALAR_LR": "0.01",
    },
    "2060sprint_micro_tokens": {
        "TRAIN_SEQ_LEN": "64",
        "TRAIN_BATCH_TOKENS": "16384",
        "VAL_BATCH_SIZE": "16384",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "POST_STEP_ZERO_GRAD": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
        "LOSS_FP32": "0",
        "LOSS_TOKEN_STRIDE": "4",
        "LOSS_VOCAB_SAMPLE_SIZE": "0",
        "LOGIT_SOFTCAP": "0",
        "TRAIN_TERNARY_PARAM_DTYPE": "model",
        "TRAIN_CASTED_LINEAR_PARAM_DTYPE": "model",
        "KEEP_CONTROL_PARAMS_FP32": "0",
        "RESID_MIX_ENABLED": "0",
        "BRANCH_SCALE_ENABLED": "1",
        "BRANCH_SCALE_KIND": "vector",
        "HRC_RECUR_INJECT_ENABLED": "0",
        "HRC_LOOP_INDEX_ENABLED": "0",
        "HRC_PASS_EMBED_ENABLED": "0",
        "HRC_PASS_ROLE_MODE": "none",
        "TRAIN_TERNARY_GROUP_SIZE": "256",
        "QUANT_TERNARY_GROUP_SIZE": "256",
        "TIED_EMBED_LR": "0.0125",
        "MATRIX_LR": "0.01",
        "SCALAR_LR": "0.01",
    },
    "2060steps": {
        "TRAIN_SEQ_LEN": "64",
        "TRAIN_BATCH_TOKENS": "2048",
        "VAL_BATCH_SIZE": "2048",
        "VAL_LOSS_EVERY": "0",
        "TRAIN_LOG_EVERY": "0",
        "GRAD_ACCUM_STEPS": "1",
        "OPTIMIZER_PRESET": "adamw",
        "SDP_BACKEND": "auto",
        "DISABLE_COMPILE": "1",
        "LOSS_FP32": "0",
        "LOSS_TOKEN_STRIDE": "1",
        "LOSS_VOCAB_SAMPLE_SIZE": "0",
        "LOGIT_SOFTCAP": "0",
        "TRAIN_TERNARY_GROUP_SIZE": "256",
        "QUANT_TERNARY_GROUP_SIZE": "256",
    },
}

for _stable_base_preset in (
    "2060sprint_micro",
    "2060sprint_micro_throughput",
    "2060sprint_micro_tokens",
):
    _stable_preset = dict(SUB4_SPEED_PRESETS[_stable_base_preset])
    _stable_preset.update(
        {
            "TIED_EMBED_LR": "0.004",
            "MATRIX_LR": "0.003",
            "SCALAR_LR": "0.003",
            "GRAD_CLIP_NORM": "0.0",
        }
    )
    SUB4_SPEED_PRESETS[f"{_stable_base_preset}_stable"] = _stable_preset

for _cool_base_preset in (
    "2060sprint_micro",
    "2060sprint_micro_throughput",
    "2060sprint_micro_tokens",
):
    _cool_preset = dict(SUB4_SPEED_PRESETS[_cool_base_preset])
    _cool_preset.update(
        {
            "TIED_EMBED_LR": "0.0015",
            "MATRIX_LR": "0.001",
            "SCALAR_LR": "0.001",
            "WARMDOWN_ITERS": "1000",
        }
    )
    SUB4_SPEED_PRESETS[f"{_cool_base_preset}_cool1k"] = _cool_preset

_cool_full_loss = dict(SUB4_SPEED_PRESETS["2060sprint_micro_cool1k"])
_cool_full_loss["LOSS_TOKEN_STRIDE"] = "1"
SUB4_SPEED_PRESETS["2060sprint_micro_cool1k_full"] = _cool_full_loss

_quality_muon = dict(SUB4_SPEED_PRESETS["2060sprint_micro_cool1k"])
_quality_muon.update(
    {
        "OPTIMIZER_PRESET": "hybrid",
        "TIED_EMBED_LR": "0.002",
        "MATRIX_LR": "0.003",
        "SCALAR_LR": "0.003",
        "MUON_BACKEND_STEPS": "5",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_quality"] = _quality_muon

_quality_muon_damped = dict(_quality_muon)
_quality_muon_damped.update(
    {
        "LR_WARMUP_ITERS": "200",
        "DEPTH_SCALE_INIT_ENABLED": "1",
        "DEPTH_SCALE_INIT_START": "0.25",
        "DEPTH_SCALE_INIT_END": "0.75",
        "QK_GAIN_INIT": "1.0",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_damped"] = _quality_muon_damped

_quality_muon_damped_full = dict(_quality_muon_damped)
_quality_muon_damped_full["LOSS_TOKEN_STRIDE"] = "1"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_damped_full"] = _quality_muon_damped_full

_quality_muon_damped_full_vsample2k = dict(_quality_muon_damped_full)
_quality_muon_damped_full_vsample2k["LOSS_VOCAB_SAMPLE_SIZE"] = "2048"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_damped_full_vsample2k"] = _quality_muon_damped_full_vsample2k

_quality_muon_damped_full_tokens8k_vsample2k = dict(_quality_muon_damped_full_vsample2k)
_quality_muon_damped_full_tokens8k_vsample2k.update(
    {
        "TRAIN_BATCH_TOKENS": "8192",
        "VAL_BATCH_SIZE": "8192",
    }
)
SUB4_SPEED_PRESETS[
    "2060sprint_micro_muon_damped_full_tokens8k_vsample2k"
] = _quality_muon_damped_full_tokens8k_vsample2k

_quality_muon_cooltaper5k = dict(_quality_muon_damped_full)
_quality_muon_cooltaper5k.update(
    {
        "TIED_EMBED_LR": "0.001",
        "MATRIX_LR": "0.0015",
        "SCALAR_LR": "0.0015",
        "WARMDOWN_ITERS": "5000",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_cooltaper5k"] = _quality_muon_cooltaper5k

_quality_muon_cooltaper5k_low = dict(_quality_muon_damped_full)
_quality_muon_cooltaper5k_low.update(
    {
        "TIED_EMBED_LR": "0.00075",
        "MATRIX_LR": "0.001",
        "SCALAR_LR": "0.001",
        "WARMDOWN_ITERS": "5000",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_cooltaper5k_low"] = _quality_muon_cooltaper5k_low

_quality_muon_cooltaper5k_cold = dict(_quality_muon_damped_full)
_quality_muon_cooltaper5k_cold.update(
    {
        "TIED_EMBED_LR": "0.0003",
        "MATRIX_LR": "0.0004",
        "SCALAR_LR": "0.0004",
        "WARMDOWN_ITERS": "5000",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_cooltaper5k_cold"] = _quality_muon_cooltaper5k_cold

_quality_muon_cooltaper5k_cold_tokens8k = dict(_quality_muon_cooltaper5k_cold)
_quality_muon_cooltaper5k_cold_tokens8k.update(
    {
        "TRAIN_BATCH_TOKENS": "8192",
        "VAL_BATCH_SIZE": "8192",
    }
)
SUB4_SPEED_PRESETS[
    "2060sprint_micro_muon_cooltaper5k_cold_tokens8k"
] = _quality_muon_cooltaper5k_cold_tokens8k

_quality_muon_minlr5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_minlr5k["LR_MIN_SCALE"] = "0.10"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_minlr5k"] = _quality_muon_minlr5k

_quality_muon_polar5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_polar5k["MUON_NS_VARIANT"] = "polar_express"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_polar5k"] = _quality_muon_polar5k

_quality_muon_turbogram5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_turbogram5k.update(
    {
        "MUON_NS_VARIANT": "gram_polar",
        "MUON_BACKEND_STEPS": "4",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_turbogram5k"] = _quality_muon_turbogram5k

_quality_muon_qk525_5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_qk525_5k["QK_GAIN_INIT"] = "5.25"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_qk525_5k"] = _quality_muon_qk525_5k

_quality_muon_rownorm5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_rownorm5k["MUON_ROW_NORMALIZE"] = "1"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_rownorm5k"] = _quality_muon_rownorm5k

_quality_muon_rownorm_wd5k = dict(_quality_muon_rownorm5k)
_quality_muon_rownorm_wd5k["MUON_WEIGHT_DECAY"] = "0.095"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_rownorm_wd5k"] = _quality_muon_rownorm_wd5k

_quality_muon_attngate5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_attngate5k.update(
    {
        "ATTN_OUT_GATE_ENABLED": "1",
        "ATTN_OUT_GATE_WIDTH": "12",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_attngate5k"] = _quality_muon_attngate5k

_quality_muon_smear_scalar5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_smear_scalar5k.update(
    {
        "SMEAR_GATE_ENABLED": "1",
        "SMEAR_GATE_WIDTH": "12",
        "SMEAR_GATE_MODE": "scalar",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_smear_scalar5k"] = _quality_muon_smear_scalar5k

_quality_muon_sparsegate5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_sparsegate5k.update(
    {
        "SPARSE_ATTN_GATE_ENABLED": "1",
        "ATTN_OUT_GATE_WIDTH": "12",
        "SPARSE_ATTN_GATE_INIT_STD": "0.0",
        "SPARSE_ATTN_GATE_SCALE": "1.0",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_sparsegate5k"] = _quality_muon_sparsegate5k

_quality_muon_smear_sparse5k = dict(_quality_muon_smear_scalar5k)
_quality_muon_smear_sparse5k.update(
    {
        "SPARSE_ATTN_GATE_ENABLED": "1",
        "ATTN_OUT_GATE_WIDTH": "12",
        "SPARSE_ATTN_GATE_INIT_STD": "0.0",
        "SPARSE_ATTN_GATE_SCALE": "1.0",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_smear_sparse5k"] = _quality_muon_smear_sparse5k

_quality_muon_huberwd5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_huberwd5k.update(
    {
        "MUON_WEIGHT_DECAY": "0.095",
        "MUON_WEIGHT_DECAY_MODE": "huber",
        "MUON_WEIGHT_DECAY_HUBER_DELTA_SCALE": "3.0",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_huberwd5k"] = _quality_muon_huberwd5k

_quality_muon_ttt_control5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_ttt_control5k.update(
    {
        "TTT_SCORE_FIRST_ENABLED": "1",
        "TTT_SCORE_FIRST_PARAM_MODE": "control",
        "TTT_SCORE_FIRST_OPTIMIZER": "sgd",
        "TTT_SCORE_FIRST_LR": "0.0005",
        "TTT_SCORE_FIRST_GRAD_CLIP": "1.0",
        "TTT_SCORE_FIRST_MAX_UPDATES": "8",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_ttt_control5k"] = _quality_muon_ttt_control5k

_quality_muon_competitor_meta5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_competitor_meta5k.update(
    {
        "LR_MIN_SCALE": "0.10",
        "MUON_NS_VARIANT": "polar_express",
        "QK_GAIN_INIT": "5.25",
    }
)
SUB4_SPEED_PRESETS[
    "2060sprint_micro_muon_competitor_meta5k"
] = _quality_muon_competitor_meta5k

_quality_muon_lqer5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_lqer5k.update(
    {
        "LQER_ENABLED": "1",
        "LQER_RANK": "4",
        "LQER_TOP_K": "8",
        "LQER_ASYM_ENABLED": "1",
        "LQER_ASYM_GROUP": "64",
        "LQER_INCLUDE_PATTERNS": "blocks.",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_lqer5k"] = _quality_muon_lqer5k

_quality_muon_lqer_r8t16_5k = dict(_quality_muon_lqer5k)
_quality_muon_lqer_r8t16_5k.update(
    {
        "LQER_RANK": "8",
        "LQER_TOP_K": "16",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_lqer_r8t16_5k"] = _quality_muon_lqer_r8t16_5k

_quality_muon_lqerio_r8t16_5k = dict(_quality_muon_lqer_r8t16_5k)
_quality_muon_lqerio_r8t16_5k.update(
    {
        "LQER_INCLUDE_PATTERNS": "tok_emb.weight,embed_proj,blocks.",
        "LQER_EXCLUDE_PATTERNS": "lm_head.weight,token_smear,attn_gate_w,attn_out_gate",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_lqerio_r8t16_5k"] = _quality_muon_lqerio_r8t16_5k

_quality_muon_lqerio_r16t24_5k = dict(_quality_muon_lqerio_r8t16_5k)
_quality_muon_lqerio_r16t24_5k.update(
    {
        "LQER_RANK": "16",
        "LQER_TOP_K": "24",
    }
)
SUB4_SPEED_PRESETS[
    "2060sprint_micro_muon_lqerio_r16t24_5k"
] = _quality_muon_lqerio_r16t24_5k

_quality_muon_lqerio_r16t32_5k = dict(_quality_muon_lqerio_r16t24_5k)
_quality_muon_lqerio_r16t32_5k["LQER_TOP_K"] = "32"
SUB4_SPEED_PRESETS[
    "2060sprint_micro_muon_lqerio_r16t32_5k"
] = _quality_muon_lqerio_r16t32_5k

_quality_muon_fcarry5k = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_fcarry5k.update(
    {
        "HRC_FROZEN_CARRY_ENABLED": "1",
        "HRC_FROZEN_CARRY_BLOCKS": "",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_fcarry5k"] = _quality_muon_fcarry5k

_quality_muon_fcarry_lqer5k = dict(_quality_muon_lqer5k)
_quality_muon_fcarry_lqer5k.update(
    {
        "HRC_FROZEN_CARRY_ENABLED": "1",
        "HRC_FROZEN_CARRY_BLOCKS": "",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_fcarry_lqer5k"] = _quality_muon_fcarry_lqer5k

_quality_muon_fcarry_lqerio_r8t16_5k = dict(_quality_muon_lqerio_r8t16_5k)
_quality_muon_fcarry_lqerio_r8t16_5k.update(
    {
        "HRC_FROZEN_CARRY_ENABLED": "1",
        "HRC_FROZEN_CARRY_BLOCKS": "",
    }
)
SUB4_SPEED_PRESETS[
    "2060sprint_micro_muon_fcarry_lqerio_r8t16_5k"
] = _quality_muon_fcarry_lqerio_r8t16_5k

_quality_muon_fcarry_lqerio_r16t24_5k = dict(_quality_muon_lqerio_r16t24_5k)
_quality_muon_fcarry_lqerio_r16t24_5k.update(
    {
        "HRC_FROZEN_CARRY_ENABLED": "1",
        "HRC_FROZEN_CARRY_BLOCKS": "",
    }
)
SUB4_SPEED_PRESETS[
    "2060sprint_micro_muon_fcarry_lqerio_r16t24_5k"
] = _quality_muon_fcarry_lqerio_r16t24_5k

_quality_muon_fcarry_lqerio_r16t32_5k = dict(_quality_muon_lqerio_r16t32_5k)
_quality_muon_fcarry_lqerio_r16t32_5k.update(
    {
        "HRC_FROZEN_CARRY_ENABLED": "1",
        "HRC_FROZEN_CARRY_BLOCKS": "",
    }
)
SUB4_SPEED_PRESETS[
    "2060sprint_micro_muon_fcarry_lqerio_r16t32_5k"
] = _quality_muon_fcarry_lqerio_r16t32_5k

_quality_muon_fcarry_lqerio_nodetach5k = dict(_quality_muon_fcarry_lqerio_r8t16_5k)
_quality_muon_fcarry_lqerio_nodetach5k["HRC_FROZEN_CARRY_DETACH"] = "0"
SUB4_SPEED_PRESETS[
    "2060sprint_micro_muon_fcarry_lqerio_nodetach5k"
] = _quality_muon_fcarry_lqerio_nodetach5k

_quality_muon_publicstack5k = dict(_quality_muon_competitor_meta5k)
_quality_muon_publicstack5k.update(
    {
        "MUON_WEIGHT_DECAY": "0.095",
        "MUON_WEIGHT_DECAY_MODE": "huber",
        "MUON_WEIGHT_DECAY_HUBER_DELTA_SCALE": "3.0",
        "SPARSE_ATTN_GATE_ENABLED": "1",
        "ATTN_OUT_GATE_WIDTH": "12",
        "SPARSE_ATTN_GATE_INIT_STD": "0.0",
        "SPARSE_ATTN_GATE_SCALE": "1.0",
        "HRC_FROZEN_CARRY_ENABLED": "1",
        "HRC_FROZEN_CARRY_BLOCKS": "",
        "LQER_ENABLED": "1",
        "LQER_RANK": "4",
        "LQER_TOP_K": "8",
        "LQER_ASYM_ENABLED": "1",
        "LQER_ASYM_GROUP": "64",
        "LQER_INCLUDE_PATTERNS": "blocks.",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_publicstack5k"] = _quality_muon_publicstack5k

_quality_muon_publicstack_smear5k = dict(_quality_muon_publicstack5k)
_quality_muon_publicstack_smear5k.update(
    {
        "SMEAR_GATE_ENABLED": "1",
        "SMEAR_GATE_WIDTH": "12",
        "SMEAR_GATE_MODE": "scalar",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_publicstack_smear5k"] = _quality_muon_publicstack_smear5k

_quality_muon_publicstack_lqerio5k = dict(_quality_muon_publicstack5k)
_quality_muon_publicstack_lqerio5k.update(
    {
        "LQER_RANK": "8",
        "LQER_TOP_K": "16",
        "LQER_INCLUDE_PATTERNS": "tok_emb.weight,embed_proj,blocks.",
        "LQER_EXCLUDE_PATTERNS": "lm_head.weight,token_smear,attn_gate_w,attn_out_gate",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_publicstack_lqerio5k"] = _quality_muon_publicstack_lqerio5k

_quality_muon_cooltaper_wallclock = dict(_quality_muon_cooltaper5k)
_quality_muon_cooltaper_wallclock["WARMDOWN_ITERS"] = "20000"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_cooltaper_wallclock"] = _quality_muon_cooltaper_wallclock

_quality_muon_cooltaper_low_wallclock = dict(_quality_muon_cooltaper5k_low)
_quality_muon_cooltaper_low_wallclock["WARMDOWN_ITERS"] = "20000"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_cooltaper_low_wallclock"] = _quality_muon_cooltaper_low_wallclock

_quality_muon_cooltaper_cold_wallclock = dict(_quality_muon_cooltaper5k_cold)
_quality_muon_cooltaper_cold_wallclock["WARMDOWN_ITERS"] = "12000"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_cooltaper_cold_wallclock"] = _quality_muon_cooltaper_cold_wallclock

_quality_muon_cooltaper_cold_wallclock16k = dict(_quality_muon_cooltaper5k_cold)
_quality_muon_cooltaper_cold_wallclock16k["WARMDOWN_ITERS"] = "16000"
SUB4_SPEED_PRESETS[
    "2060sprint_micro_muon_cooltaper_cold_wallclock16k"
] = _quality_muon_cooltaper_cold_wallclock16k

_quality_muon_cooltaper_cold_tokens8k_wallclock = dict(_quality_muon_cooltaper5k_cold_tokens8k)
_quality_muon_cooltaper_cold_tokens8k_wallclock["WARMDOWN_ITERS"] = "5200"
SUB4_SPEED_PRESETS[
    "2060sprint_micro_muon_cooltaper_cold_tokens8k_wallclock"
] = _quality_muon_cooltaper_cold_tokens8k_wallclock

_quality_muon_wallclock = dict(_quality_muon_damped_full)
_quality_muon_wallclock["WARMDOWN_ITERS"] = "10000"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_wallclock"] = _quality_muon_wallclock

# Quality guard rail lane for the wide sub-4MB family. The raw sprint presets
# intentionally strip these controls for speed, but that also removes the cheap
# pass/loop identity signals that make the repeated HRC middle less ambiguous.
_quality_muon_guarded = dict(_quality_muon_damped_full)
_quality_muon_guarded.update(
    {
        "LOGIT_SOFTCAP": "30",
        "LOSS_FP32": "1",
        "HRC_RECUR_INJECT_ENABLED": "1",
        "HRC_LOOP_INDEX_ENABLED": "1",
        "HRC_PASS_EMBED_ENABLED": "1",
        "HRC_PASS_EMBED_MODE": "block_peer",
        "HRC_PASS_ROLE_MODE": "phase5",
        "KEEP_CONTROL_PARAMS_FP32": "1",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_guarded"] = _quality_muon_guarded

_quality_muon_guarded_seq128 = dict(_quality_muon_guarded)
_quality_muon_guarded_seq128.update(
    {
        "TRAIN_SEQ_LEN": "128",
        "VAL_BATCH_SIZE": "4096",
    }
)
SUB4_SPEED_PRESETS["2060sprint_micro_muon_guarded_seq128"] = _quality_muon_guarded_seq128

_quality_muon_guarded_wallclock = dict(_quality_muon_guarded)
_quality_muon_guarded_wallclock["WARMDOWN_ITERS"] = "10000"
SUB4_SPEED_PRESETS["2060sprint_micro_muon_guarded_wallclock"] = _quality_muon_guarded_wallclock

for _packed_base_preset in (
    "2060sprint_micro",
    "2060sprint_micro_throughput",
    "2060sprint_micro_tokens",
    "2060sprint_micro_stable",
    "2060sprint_micro_throughput_stable",
    "2060sprint_micro_tokens_stable",
    "2060sprint_micro_cool1k",
    "2060sprint_micro_throughput_cool1k",
    "2060sprint_micro_tokens_cool1k",
    "2060sprint_micro_cool1k_full",
    "2060sprint_micro_muon_quality",
    "2060sprint_micro_muon_damped",
    "2060sprint_micro_muon_damped_full",
    "2060sprint_micro_muon_damped_full_vsample2k",
    "2060sprint_micro_muon_damped_full_tokens8k_vsample2k",
    "2060sprint_micro_muon_cooltaper5k",
    "2060sprint_micro_muon_cooltaper5k_low",
    "2060sprint_micro_muon_cooltaper5k_cold",
    "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
    "2060sprint_micro_muon_minlr5k",
    "2060sprint_micro_muon_polar5k",
    "2060sprint_micro_muon_turbogram5k",
    "2060sprint_micro_muon_qk525_5k",
    "2060sprint_micro_muon_rownorm5k",
    "2060sprint_micro_muon_rownorm_wd5k",
    "2060sprint_micro_muon_attngate5k",
    "2060sprint_micro_muon_smear_scalar5k",
    "2060sprint_micro_muon_sparsegate5k",
    "2060sprint_micro_muon_smear_sparse5k",
    "2060sprint_micro_muon_huberwd5k",
    "2060sprint_micro_muon_ttt_control5k",
    "2060sprint_micro_muon_competitor_meta5k",
    "2060sprint_micro_muon_lqer5k",
    "2060sprint_micro_muon_lqer_r8t16_5k",
    "2060sprint_micro_muon_lqerio_r8t16_5k",
    "2060sprint_micro_muon_lqerio_r16t24_5k",
    "2060sprint_micro_muon_lqerio_r16t32_5k",
    "2060sprint_micro_muon_fcarry5k",
    "2060sprint_micro_muon_fcarry_lqer5k",
    "2060sprint_micro_muon_fcarry_lqerio_r8t16_5k",
    "2060sprint_micro_muon_fcarry_lqerio_r16t24_5k",
    "2060sprint_micro_muon_fcarry_lqerio_r16t32_5k",
    "2060sprint_micro_muon_fcarry_lqerio_nodetach5k",
    "2060sprint_micro_muon_publicstack5k",
    "2060sprint_micro_muon_publicstack_smear5k",
    "2060sprint_micro_muon_publicstack_lqerio5k",
    "2060sprint_micro_muon_cooltaper_wallclock",
    "2060sprint_micro_muon_cooltaper_low_wallclock",
    "2060sprint_micro_muon_cooltaper_cold_wallclock",
    "2060sprint_micro_muon_cooltaper_cold_wallclock16k",
    "2060sprint_micro_muon_cooltaper_cold_tokens8k_wallclock",
    "2060sprint_micro_muon_wallclock",
    "2060sprint_micro_muon_guarded",
    "2060sprint_micro_muon_guarded_seq128",
    "2060sprint_micro_muon_guarded_wallclock",
):
    _packed_preset = dict(SUB4_SPEED_PRESETS[_packed_base_preset])
    _packed_preset["TRAIN_TERNARY_PACKED_KERNEL"] = "1"
    _packed_preset["TRAIN_TERNARY_DENSE_KERNEL"] = "0"
    SUB4_SPEED_PRESETS[f"{_packed_base_preset}_packed"] = _packed_preset

for _dense_base_preset in (
    "2060sprint_micro",
    "2060sprint_micro_throughput",
    "2060sprint_micro_tokens",
    "2060sprint_micro_stable",
    "2060sprint_micro_throughput_stable",
    "2060sprint_micro_tokens_stable",
    "2060sprint_micro_cool1k",
    "2060sprint_micro_throughput_cool1k",
    "2060sprint_micro_tokens_cool1k",
    "2060sprint_micro_cool1k_full",
    "2060sprint_micro_muon_quality",
    "2060sprint_micro_muon_damped",
    "2060sprint_micro_muon_damped_full",
    "2060sprint_micro_muon_damped_full_vsample2k",
    "2060sprint_micro_muon_damped_full_tokens8k_vsample2k",
    "2060sprint_micro_muon_cooltaper5k",
    "2060sprint_micro_muon_cooltaper5k_low",
    "2060sprint_micro_muon_cooltaper5k_cold",
    "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
    "2060sprint_micro_muon_minlr5k",
    "2060sprint_micro_muon_polar5k",
    "2060sprint_micro_muon_turbogram5k",
    "2060sprint_micro_muon_qk525_5k",
    "2060sprint_micro_muon_rownorm5k",
    "2060sprint_micro_muon_rownorm_wd5k",
    "2060sprint_micro_muon_attngate5k",
    "2060sprint_micro_muon_smear_scalar5k",
    "2060sprint_micro_muon_sparsegate5k",
    "2060sprint_micro_muon_smear_sparse5k",
    "2060sprint_micro_muon_huberwd5k",
    "2060sprint_micro_muon_ttt_control5k",
    "2060sprint_micro_muon_competitor_meta5k",
    "2060sprint_micro_muon_lqer5k",
    "2060sprint_micro_muon_lqer_r8t16_5k",
    "2060sprint_micro_muon_lqerio_r8t16_5k",
    "2060sprint_micro_muon_lqerio_r16t24_5k",
    "2060sprint_micro_muon_lqerio_r16t32_5k",
    "2060sprint_micro_muon_fcarry5k",
    "2060sprint_micro_muon_fcarry_lqer5k",
    "2060sprint_micro_muon_fcarry_lqerio_r8t16_5k",
    "2060sprint_micro_muon_fcarry_lqerio_r16t24_5k",
    "2060sprint_micro_muon_fcarry_lqerio_r16t32_5k",
    "2060sprint_micro_muon_fcarry_lqerio_nodetach5k",
    "2060sprint_micro_muon_publicstack5k",
    "2060sprint_micro_muon_publicstack_smear5k",
    "2060sprint_micro_muon_publicstack_lqerio5k",
    "2060sprint_micro_muon_cooltaper_wallclock",
    "2060sprint_micro_muon_cooltaper_low_wallclock",
    "2060sprint_micro_muon_cooltaper_cold_wallclock",
    "2060sprint_micro_muon_cooltaper_cold_wallclock16k",
    "2060sprint_micro_muon_cooltaper_cold_tokens8k_wallclock",
    "2060sprint_micro_muon_wallclock",
    "2060sprint_micro_muon_guarded",
    "2060sprint_micro_muon_guarded_seq128",
    "2060sprint_micro_muon_guarded_wallclock",
):
    _dense_preset = dict(SUB4_SPEED_PRESETS[_dense_base_preset])
    _dense_preset["TRAIN_TERNARY_PACKED_KERNEL"] = "0"
    _dense_preset["TRAIN_TERNARY_DENSE_KERNEL"] = "1"
    SUB4_SPEED_PRESETS[f"{_dense_base_preset}_dense"] = _dense_preset

profile_name = os.environ.get("SUB4_PROFILE", "i1l2r2_d384_e128_h8kv1_mlpinner_mlp10").strip().lower()
if profile_name not in SUB4_PROFILES:
    raise ValueError(f"Unknown SUB4_PROFILE={profile_name!r}; choose one of {sorted(SUB4_PROFILES)}")

for key, value in SUB4_PROFILES[profile_name].items():
    os.environ.setdefault(key, value)

speed_preset_name = os.environ.get("SUB4_SPEED_PRESET", "2060sprint_micro_muon_damped_full").strip().lower()
if speed_preset_name:
    if speed_preset_name not in SUB4_SPEED_PRESETS:
        raise ValueError(
            f"Unknown SUB4_SPEED_PRESET={speed_preset_name!r}; "
            f"choose one of {sorted(SUB4_SPEED_PRESETS)}"
        )
    for key, value in SUB4_SPEED_PRESETS[speed_preset_name].items():
        if key not in USER_ENV_KEYS:
            os.environ[key] = value

for key, value in SUB4_TERNARY_COMMON_DEFAULTS.items():
    os.environ.setdefault(key, value)

from train_gpt import main


if __name__ == "__main__":
    main()
