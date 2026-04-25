from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOLF_ROOT = ROOT.parents[1]
LOG_DIR = GOLF_ROOT / "logs"
LAUNCHER = ROOT / "scripts" / "run_caseops_candidate_2060_compare.py"

DEFAULT_LADDER = (
    "loopplain5k_i3l3r3_q6proof",
    "loopplain5k_i3l3r3_lqer_fcarry_q6proof",
    "loopplain5k_i3l3r3_lqerio_fcarry_q6proof",
    "loopplain5k_i3l3r3_lqerio_r16t24_fcarry_q6proof",
    "loopplain5k_i3l3r3_publicstack_q6proof",
    "loopplain5k_i3l3r3_q6allproof",
    "loopplain5k_i3l3r3_q6emb8proof",
    "loopplain5k_i3l3r3_q8proof",
)

SPEED_LADDER = (
    "loopplain1k_i3l3r3_q6proof_speedprobe",
    "loopplain1k_i3l3r3_q6proof_fusedqkv_speedprobe",
    "loopplain1k_i3l3r3_q6proof_fp16_speedprobe",
    "loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_speedprobe",
    "loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_mb2_speedprobe",
    "loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_lvs4096_speedprobe",
    "loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_nosoftcap_speedprobe",
    "loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_adamw_speedprobe",
    "loopplain1k_i3l3r3_q6proof_fp16_fusedqkv_seq64_speedprobe",
)


FINAL_BPB_RE = re.compile(r"final_export_roundtrip(?:_exact)? val_loss:[0-9.]+ val_bpb:([0-9.]+)")
CAP_RE = re.compile(r"submission_cap:(\d+) size:(\d+) headroom:([-0-9]+) status:(\w+)")
VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:[0-9.]+ val_bpb:([0-9.]+)")


def parse_profiles() -> list[str]:
    raw = os.environ.get("LOOP_LADDER_PROFILES", "").strip()
    if not raw:
        mode = os.environ.get("LOOP_LADDER_MODE", "quality").strip().lower()
        if mode == "speed":
            return list(SPEED_LADDER)
        if mode != "quality":
            raise ValueError("LOOP_LADDER_MODE must be quality|speed")
        return list(DEFAULT_LADDER)
    return [item.strip().lower() for item in raw.replace(";", ",").split(",") if item.strip()]


def summarize_stage(log_path: Path) -> str:
    if not log_path.is_file():
        return "missing_log=1"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    final_bpbs = FINAL_BPB_RE.findall(text)
    cap_matches = CAP_RE.findall(text)
    val_matches = VAL_RE.findall(text)
    parts: list[str] = []
    if val_matches:
        step, total, bpb = val_matches[-1]
        parts.append(f"last_val={step}/{total}:{bpb}")
    if final_bpbs:
        parts.append(f"final_export_bpb={final_bpbs[-1]}")
    if cap_matches:
        cap, size, headroom, status = cap_matches[-1]
        parts.append(f"cap={status}:{size}/{cap}:headroom={headroom}")
    if "non-finite" in text.lower() or "nan" in text.lower():
        parts.append("nonfinite_signal=1")
    return " ".join(parts) if parts else "no_summary_signal=1"


def run_stage(profile: str, suite_id: str, suite_log) -> int:
    run_id = f"{suite_id}-{profile}"
    out_path = LOG_DIR / f"{run_id}.txt"
    err_path = LOG_DIR / f"{run_id}.err.txt"
    env = os.environ.copy()
    env["CANDIDATE2060_PROFILE"] = profile
    env["RUN_ID"] = run_id
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [sys.executable, "-u", str(LAUNCHER)]
    suite_log.write(f"stage_start profile={profile} out={out_path} err={err_path}\n")
    suite_log.flush()
    with out_path.open("w", encoding="utf-8") as out, err_path.open("w", encoding="utf-8") as err:
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=err)
    summary = summarize_stage(out_path)
    suite_log.write(f"stage_done profile={profile} returncode={proc.returncode} {summary}\n")
    suite_log.flush()
    return proc.returncode


def main() -> int:
    if not LAUNCHER.is_file():
        print(f"missing launcher: {LAUNCHER}", file=sys.stderr)
        return 1
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    suite_id = os.environ.get("LOOP_LADDER_RUN_ID", "").strip()
    if not suite_id:
        suite_id = f"caseops-loopladder2060-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    profiles = parse_profiles()
    suite_path = LOG_DIR / f"{suite_id}.suite.txt"
    with suite_path.open("w", encoding="utf-8") as suite_log:
        suite_log.write(f"suite_id={suite_id}\n")
        suite_log.write(f"profiles={','.join(profiles)}\n")
        suite_log.flush()
        for profile in profiles:
            rc = run_stage(profile, suite_id, suite_log)
            if rc != 0:
                suite_log.write(f"suite_stop profile={profile} returncode={rc}\n")
                suite_log.flush()
                return rc
        suite_log.write("suite_done returncode=0\n")
    print(f"suite_log={suite_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
