from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD_DIR = (
    ROOT
    / "upstream_records"
    / "records"
    / "track_10min_16mb"
    / "2026-04-18_PR1626_CaseOps_Taper"
)
TRAINER = RECORD_DIR / "train_gpt_2060compat.py"
DATASET_DIR = RECORD_DIR / "datasets" / "fineweb10B_sp8192_lossless_caps_caseops_v1_reserved"
TOKENIZER_PATH = RECORD_DIR / "tokenizers" / "fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model"


PROFILES: dict[str, dict[str, str]] = {
    "trainonly_smoke": {
        "ITERATIONS": "1",
        "WARMUP_STEPS": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "TRAIN_BATCH_TOKENS": "16384",
        "VAL_BATCH_TOKENS": "16384",
        "TRAIN_SEQ_LEN": "2048",
        "EVAL_SEQ_LEN": "2048",
        "TRAIN_LOG_EVERY": "1",
        "VAL_LOSS_EVERY": "0",
        "SKIP_INITIAL_VAL": "1",
        "SKIP_POST_TRAIN_EVAL": "1",
    },
    "compare1k": {
        "ITERATIONS": "1000",
        "WARMUP_STEPS": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "TRAIN_BATCH_TOKENS": "16384",
        "VAL_BATCH_TOKENS": "16384",
        "TRAIN_SEQ_LEN": "2048",
        "EVAL_SEQ_LEN": "2048",
        "TRAIN_LOG_EVERY": "50",
        "VAL_LOSS_EVERY": "100",
        "VAL_DOC_FRACTION": "0.001",
        "SKIP_INITIAL_VAL": "1",
        "SKIP_POST_TRAIN_EVAL": "1",
    },
    "compare3k": {
        "ITERATIONS": "3000",
        "WARMUP_STEPS": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "TRAIN_BATCH_TOKENS": "16384",
        "VAL_BATCH_TOKENS": "16384",
        "TRAIN_SEQ_LEN": "2048",
        "EVAL_SEQ_LEN": "2048",
        "TRAIN_LOG_EVERY": "100",
        "VAL_LOSS_EVERY": "200",
        "VAL_DOC_FRACTION": "0.001",
        "SKIP_INITIAL_VAL": "1",
        "SKIP_POST_TRAIN_EVAL": "1",
    },
    "compare10k": {
        "ITERATIONS": "10000",
        "WARMUP_STEPS": "0",
        "MAX_WALLCLOCK_SECONDS": "0",
        "TRAIN_BATCH_TOKENS": "16384",
        "VAL_BATCH_TOKENS": "16384",
        "TRAIN_SEQ_LEN": "2048",
        "EVAL_SEQ_LEN": "2048",
        "TRAIN_LOG_EVERY": "200",
        "VAL_LOSS_EVERY": "200",
        "VAL_DOC_FRACTION": "0.001",
        "SKIP_INITIAL_VAL": "1",
        "SKIP_POST_TRAIN_EVAL": "1",
    },
}


def main() -> int:
    profile = os.environ.get("WINNER2060_PROFILE", "compare1k").strip().lower()
    if profile not in PROFILES:
        print(
            f"WINNER2060_PROFILE must be one of {', '.join(sorted(PROFILES))}, got {profile!r}",
            file=sys.stderr,
        )
        return 2
    if not TRAINER.is_file():
        print(f"Missing compat trainer: {TRAINER}", file=sys.stderr)
        return 1
    if not DATASET_DIR.is_dir() or not TOKENIZER_PATH.is_file():
        print(
            "Missing CaseOps data/tokenizer. First run the upstream downloader:\n"
            "  MATCHED_FINEWEB_REPO_ID=romeerp/parameter-golf-caseops-v1\n"
            "  MATCHED_FINEWEB_REMOTE_ROOT_PREFIX=datasets\n"
            "  python upstream_records/records/track_10min_16mb/2026-04-18_PR1626_CaseOps_Taper/cached_challenge_fineweb.py "
            "--variant sp8192_lossless_caps_caseops_v1_reserved --train-shards 1",
            file=sys.stderr,
        )
        return 1

    env = os.environ.copy()
    run_id = env.get(
        "RUN_ID",
        f"caseops-winner2060-{profile}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    env.setdefault("RUN_ID", run_id)
    env.setdefault("AMP_DTYPE", "fp16")
    env.setdefault("USE_FUSED_MLP", "0")
    env.setdefault("DISABLE_COMPILE", "1")
    env.setdefault("LOG_SOURCE_SNAPSHOT", "0")
    env.setdefault("DATASETS_DIR", str(DATASET_DIR))
    env.setdefault("TOKENIZER_PATH", str(TOKENIZER_PATH))

    for key, value in PROFILES[profile].items():
        env.setdefault(key, value)

    cmd = [sys.executable, str(TRAINER)]
    print(f"record_dir={RECORD_DIR}")
    print(f"profile={profile}")
    print(f"run_id={env['RUN_ID']}")
    print(f"trainer={TRAINER}")
    print(f"dataset_dir={env['DATASETS_DIR']}")
    print(f"tokenizer_path={env['TOKENIZER_PATH']}")
    print("launching...")
    return subprocess.run(cmd, cwd=str(RECORD_DIR), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
