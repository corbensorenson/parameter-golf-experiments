from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WINNER_LAUNCHER = ROOT / "scripts" / "run_caseops_winner_2060_compare.py"
CANDIDATE_LAUNCHER = ROOT / "scripts" / "run_caseops_candidate_2060_compare.py"

STAGES: dict[str, dict[str, str]] = {
    "winner3k": {
        "launcher": str(WINNER_LAUNCHER),
        "WINNER2060_PROFILE": "compare3k",
    },
    "candidate3k": {
        "launcher": str(CANDIDATE_LAUNCHER),
        "CANDIDATE2060_PROFILE": "compare3k",
    },
    "winner10k": {
        "launcher": str(WINNER_LAUNCHER),
        "WINNER2060_PROFILE": "compare10k",
    },
    "candidate10k": {
        "launcher": str(CANDIDATE_LAUNCHER),
        "CANDIDATE2060_PROFILE": "compare10k",
    },
}


def main() -> int:
    stages_raw = os.environ.get(
        "CASEOPS_HEAD2HEAD_STAGES",
        "winner3k,candidate3k,winner10k,candidate10k",
    )
    stages = [stage.strip().lower() for stage in stages_raw.split(",") if stage.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        print(f"Unknown CASEOPS_HEAD2HEAD_STAGES entries: {', '.join(unknown)}", file=sys.stderr)
        return 2

    suite_id = os.environ.get(
        "RUN_ID",
        f"caseops-head2head-2060-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    print(f"suite_id={suite_id}")
    print(f"stages={','.join(stages)}")
    print("launching_serial_suite...")

    for idx, stage in enumerate(stages, start=1):
        spec = STAGES[stage]
        env = os.environ.copy()
        env["RUN_ID"] = f"{suite_id}-{stage}"
        for key, value in spec.items():
            if key != "launcher":
                env[key] = value
        print(f"[{idx}/{len(stages)}] stage={stage} run_id={env['RUN_ID']}")
        result = subprocess.run([sys.executable, "-u", spec["launcher"]], cwd=str(ROOT), env=env)
        print(f"[{idx}/{len(stages)}] stage={stage} returncode={result.returncode}")
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
