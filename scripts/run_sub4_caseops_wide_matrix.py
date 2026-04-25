"""Run the current wide-shallow CaseOps matrix for the sub-4MB ternary lane."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from run_sub4_micro_matrix import ROOT, make_summary, run_probe, run_train_matrix, write_csv


CASEOPS_DATA_PATH = (
    "upstream_records/records/track_10min_16mb/2026-04-18_PR1626_CaseOps_Taper/"
    "datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
)
CASEOPS_TOKENIZER_PATH = (
    "upstream_records/records/track_10min_16mb/2026-04-18_PR1626_CaseOps_Taper/"
    "tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"
)

LOCAL_5K_PAIRS = [
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_ttt_control5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_minlr5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_polar5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_turbogram5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_qk525_5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_rownorm5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_rownorm_wd5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_attngate5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_smear_scalar5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_sparsegate5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_smear_sparse5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_huberwd5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_competitor_meta5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_lqer5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_fcarry5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_fcarry_lqer5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_publicstack5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
    ),
    ("i1l2r2_d768_e192_h12kv1_mlpinner_mlp050", "2060sprint_micro_muon_cooltaper5k_cold"),
    ("i1l2r2_d768_e256_h12kv1_mlpinner_mlp075", "2060sprint_micro_muon_cooltaper5k_cold"),
    ("i1l2r2_d896_e256_h14kv1_mlpinner_mlp050", "2060sprint_micro_muon_cooltaper5k_cold"),
]

LOCAL_EXPORT_PAIRS = [
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_fcarry5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_lqer_r8t16_5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_lqerio_r8t16_5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_lqerio_r16t24_5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_lqerio_r16t32_5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_fcarry_lqerio_r8t16_5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_fcarry_lqerio_r16t24_5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_fcarry_lqerio_r16t32_5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_fcarry_lqerio_nodetach5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_publicstack_lqerio5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_publicstack_smear5k",
    ),
]

LOCAL_10K_PAIRS = [
    # Baseline control from the prior export-aware matrix.
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
    ),
    # Best 1k export row, promoted to the real 10k proof lane.
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_publicstack_lqerio5k",
    ),
    # Spend the remaining byte headroom on bigger IO-aware LQER sidecars.
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_lqerio_r16t24_5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_lqerio_r16t32_5k",
    ),
    # Re-test carry only where the 1k screen said the carry idea still has signal.
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_fcarry5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_fcarry_lqerio_r16t24_5k",
    ),
    # Architecture spend: thinner IO rank versus the winning lever stack.
    (
        "i1l2r2_d768_e192_h12kv1_mlpinner_mlp050",
        "2060sprint_micro_muon_publicstack_lqerio5k",
    ),
    # Architecture spend: wider body at the same e256 tied interface.
    (
        "i1l2r2_d896_e256_h14kv1_mlpinner_mlp050",
        "2060sprint_micro_muon_publicstack_lqerio5k",
    ),
    (
        "i1l2r2_d896_e256_h14kv1_mlpinner_mlp050",
        "2060sprint_micro_muon_lqerio_r16t24_5k",
    ),
    # Larger body stress test. Probe/export will catch cap misses.
    (
        "i1l2r2_d1024_e256_h16kv1_mlpinner_mlp050",
        "2060sprint_micro_muon_publicstack_lqerio5k",
    ),
    # Different loop geometry: deeper recursive middle at lower width.
    (
        "i2l3r2_d384_e128_h8kv1_mlpinner_mlp10",
        "2060sprint_micro_muon_publicstack_lqerio5k",
    ),
    (
        "i2l3r2_d384_e128_h8kv1_mlpinner_mlp15",
        "2060sprint_micro_muon_publicstack_lqerio5k",
    ),
]

LOCAL_10K_GUARDED_PAIRS = [
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_cooltaper5k_cold_tokens8k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_lqerio_r16t32_5k",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_publicstack_lqerio5k",
    ),
    (
        "i1l2r2_d896_e256_h14kv1_mlpinner_mlp050",
        "2060sprint_micro_muon_publicstack_lqerio5k",
    ),
    (
        "i2l3r2_d384_e128_h8kv1_mlpinner_mlp10",
        "2060sprint_micro_muon_publicstack_lqerio5k",
    ),
    (
        "i2l3r2_d384_e128_h8kv1_mlpinner_mlp15",
        "2060sprint_micro_muon_publicstack_lqerio5k",
    ),
]

LOCAL_10K_GUARDED_ENV = {
    # The first 10k pass showed good train-time BPB but catastrophic export
    # round-trip BPB. Keep this lane export-honest throughout the back half.
    "QUANT_TRAIN_MODE": "roundtrip",
    "QUANT_TRAIN_START_FRACTION": "0.30",
    "QUANT_TRAIN_EVERY": "100",
    "TRAIN_TERNARY_GROUP_SIZE": "128",
    "QUANT_TERNARY_GROUP_SIZE": "128",
    "LOGIT_SOFTCAP": "12",
    "LOSS_FP32": "1",
    "KEEP_CONTROL_PARAMS_FP32": "1",
    "WARMDOWN_ITERS": "10000",
    "LR_WARMDOWN_STYLE": "cosine",
    "LR_MIN_SCALE": "0",
    "TIED_EMBED_LR": "0.00015",
    "MATRIX_LR": "0.0002",
    "SCALAR_LR": "0.0002",
    "VAL_LOSS_EVERY": "1000",
}

LOCAL_WALLCLOCK_PAIRS = [
    (
        "i1l2r2_d768_e192_h12kv1_mlpinner_mlp050",
        "2060sprint_micro_muon_cooltaper_cold_wallclock",
    ),
    (
        "i1l2r2_d768_e256_h12kv1_mlpinner_mlp075",
        "2060sprint_micro_muon_cooltaper_cold_wallclock16k",
    ),
]

H100_PROBE_PROFILES = [
    "i1l2r2_d1280_e320_h20kv1_mlpinner_mlp050",
    "i1l2r2_d1536_e384_h24kv1_mlpinner_mlp050",
    "i1l2r2_d2048_e512_h32kv1_mlpinner_mlp025",
]


def parse_mode(value: str) -> list[tuple[str, str]]:
    if value == "local-5k":
        return LOCAL_5K_PAIRS
    if value == "local-export1k":
        return LOCAL_EXPORT_PAIRS
    if value == "local-10k":
        return LOCAL_10K_PAIRS
    if value == "local-10k-guarded":
        return LOCAL_10K_GUARDED_PAIRS
    if value == "local-wallclock":
        return LOCAL_WALLCLOCK_PAIRS
    raise ValueError(f"unknown mode {value!r}")


def make_out_dir(prefix: str, explicit: str) -> Path:
    if explicit:
        out_dir = Path(explicit)
    else:
        out_dir = ROOT / "records" / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "local-5k",
            "local-export1k",
            "local-10k",
            "local-10k-guarded",
            "local-wallclock",
            "h100-probe",
        ),
        default="local-5k",
    )
    parser.add_argument("--out", default="")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--wallclock-seconds", type=int, default=600)
    parser.add_argument("--val-tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--final-artifacts", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    args = parser.parse_args()

    out_dir = make_out_dir(f"sub4_caseops_wide_{args.mode}", args.out)
    env_overrides = {
        "DATA_PATH": CASEOPS_DATA_PATH,
        "TOKENIZER_PATH": CASEOPS_TOKENIZER_PATH,
    }

    if args.mode == "h100-probe":
        if not args.skip_probe:
            run_probe(out_dir, H100_PROBE_PROFILES, "2060sprint_micro_muon_cooltaper5k_cold", 8192)
        write_csv(out_dir / "bench.csv", [])
        write_csv(out_dir / "train.csv", [])
        make_summary(out_dir, [], [])
        print(out_dir)
        return

    pairs = parse_mode(args.mode)
    profiles = sorted({profile for profile, _preset in pairs})
    if not args.skip_probe:
        run_probe(out_dir, profiles, pairs[0][1], 8192)
    else:
        (out_dir / "probe_profiles.md").write_text("# Probe skipped\n", encoding="utf-8")
        (out_dir / "probe_selected.md").write_text("# Probe skipped\n", encoding="utf-8")

    iterations = args.iterations
    timeout = args.timeout
    final_artifacts = args.final_artifacts
    if args.mode == "local-export1k":
        iterations = args.iterations if args.iterations != 5000 else 1000
        timeout = max(timeout, 1800)
        final_artifacts = True
    if args.mode == "local-10k":
        iterations = args.iterations if args.iterations != 5000 else 10000
        timeout = max(timeout, 1800)
        final_artifacts = True
    if args.mode == "local-10k-guarded":
        iterations = args.iterations if args.iterations != 5000 else 10000
        timeout = max(timeout, 2400)
        final_artifacts = True
        env_overrides.update(LOCAL_10K_GUARDED_ENV)
    if args.mode == "local-wallclock":
        iterations = max(iterations, 100000)
        timeout = max(timeout, args.wallclock_seconds + 240)
        env_overrides["MAX_WALLCLOCK_SECONDS"] = str(args.wallclock_seconds)

    write_csv(out_dir / "bench.csv", [])
    train_rows = run_train_matrix(
        out_dir,
        profiles,
        [],
        iterations,
        args.val_tokens,
        8192,
        pairs=pairs,
        timeout=timeout,
        env_overrides=env_overrides,
        final_artifacts=final_artifacts,
    )
    make_summary(out_dir, [], train_rows)
    print(out_dir)


if __name__ == "__main__":
    main()
