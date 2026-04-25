from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"


def split_csv(raw: str) -> list[str]:
    return [item.strip().lower() for item in raw.replace(";", ",").split(",") if item.strip()]


def run_stage(
    suite_id: str,
    stage: str,
    cmd: list[str],
    env: dict[str, str],
    suite_log,
) -> int:
    out_path = LOG_DIR / f"{suite_id}-{stage}.out.txt"
    err_path = LOG_DIR / f"{suite_id}-{stage}.err.txt"
    suite_log.write(f"stage_start name={stage} out={out_path} err={err_path}\n")
    suite_log.write(f"stage_cmd {' '.join(cmd)}\n")
    suite_log.flush()
    with out_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=err)
    suite_log.write(f"stage_done name={stage} returncode={proc.returncode}\n")
    suite_log.flush()
    return int(proc.returncode)


def main() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    suite_id = os.environ.get(
        "COMP_MATRIX_RUN_ID",
        f"competitive-matrix-2060-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
    )
    stages = split_csv(os.environ.get("COMP_MATRIX_STAGES", "sub4_export1k,sub16_loopproof"))
    valid = {"sub4_export1k", "sub16_loopproof"}
    unknown = [stage for stage in stages if stage not in valid]
    if unknown:
        print(f"unknown COMP_MATRIX_STAGES entries: {', '.join(unknown)}", file=sys.stderr)
        return 2

    suite_path = LOG_DIR / f"{suite_id}.suite.txt"
    with suite_path.open("w", encoding="utf-8") as suite_log:
        suite_log.write(f"suite_id={suite_id}\n")
        suite_log.write(f"stages={','.join(stages)}\n")
        suite_log.flush()

        for stage in stages:
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            if stage == "sub4_export1k":
                sub4_iters = os.environ.get("COMP_MATRIX_SUB4_ITERS", "1000")
                sub4_val_tokens = os.environ.get("COMP_MATRIX_SUB4_VAL_TOKENS", "32768")
                sub4_timeout = os.environ.get("COMP_MATRIX_SUB4_TIMEOUT", "3600")
                out_dir = ROOT / "records" / f"{suite_id}-sub4-export"
                cmd = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts" / "run_sub4_caseops_wide_matrix.py"),
                    "--mode",
                    "local-export1k",
                    "--iterations",
                    sub4_iters,
                    "--val-tokens",
                    sub4_val_tokens,
                    "--timeout",
                    sub4_timeout,
                    "--out",
                    str(out_dir),
                    "--final-artifacts",
                ]
            else:
                env["LOOP_LADDER_RUN_ID"] = f"{suite_id}-sub16-loopproof"
                sub16_profiles = os.environ.get("COMP_MATRIX_SUB16_PROFILES", "").strip()
                sub16_mode = os.environ.get("COMP_MATRIX_SUB16_MODE", "quality").strip().lower()
                if sub16_profiles:
                    env["LOOP_LADDER_PROFILES"] = sub16_profiles
                elif sub16_mode == "speed":
                    env.pop("LOOP_LADDER_PROFILES", None)
                    env["LOOP_LADDER_MODE"] = "speed"
                elif sub16_mode == "quality":
                    env["LOOP_LADDER_PROFILES"] = ",".join(
                        [
                            "loopplain5k_i3l3r3_q6proof",
                            "loopplain5k_i3l3r3_lqer_fcarry_q6proof",
                            "loopplain5k_i3l3r3_lqerio_fcarry_q6proof",
                            "loopplain5k_i3l3r3_lqerio_r16t24_fcarry_q6proof",
                            "loopplain5k_i3l3r3_publicstack_q6proof",
                        ]
                    )
                else:
                    print("COMP_MATRIX_SUB16_MODE must be quality|speed", file=sys.stderr)
                    return 2
                cmd = [
                    sys.executable,
                    "-u",
                    str(ROOT / "scripts" / "run_caseops_loop_ladder_2060.py"),
                ]
            rc = run_stage(suite_id, stage, cmd, env, suite_log)
            if rc != 0:
                suite_log.write(f"suite_stop stage={stage} returncode={rc}\n")
                suite_log.flush()
                print(f"suite_log={suite_path}")
                return rc
        suite_log.write("suite_done returncode=0\n")
    print(f"suite_log={suite_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
