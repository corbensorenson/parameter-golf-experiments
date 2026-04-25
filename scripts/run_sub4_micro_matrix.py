"""Run a bounded benchmark matrix for the sub-4MB nano/micro family."""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-cuda313" / "Scripts" / "python.exe"
_MSVC_ENV_CACHE: dict[str, str] | None = None

DEFAULT_PROFILES = [
    "i1l2r2_d384_e128_h8kv1_mlpinner_mlp10",
    "i1l2r2_d512_e128_h8kv1_mlpinner_mlp075",
    "i1l2r2_d320_e96_h8kv1_mlpinner_mlp10",
    "i1l2r2_d192_e80_h3mha_mlpinner_mlp15",
    "i1l2r2_d96_e80_h3mha_mlpinner_mlp2",
    "i1l2r2_d320_e96_h8kv1_mlpinner_mlp15",
    "i1l2r2_d384_e128_h8kv1_mlpinner_mlp15",
    "i2l3r2_d320_e96_h8kv1_mlpinner_mlp10",
    "i2l3r2_d384_e128_h8kv1_mlpinner_mlp10",
]

DEFAULT_BENCH_PRESETS = [
    "2060sprint_micro",
    "2060sprint_micro_dense",
    "2060sprint_micro_throughput_dense",
    "2060sprint_micro_tokens_dense",
]

DEFAULT_TRAIN_PRESETS = [
    "2060sprint_micro_muon_damped_full",
]

FLOAT_VALUE_RAW = r"[-+]?(?:nan|inf|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
FLOAT_VALUE = rf"(?P<value>{FLOAT_VALUE_RAW})"

TRAIN_PATTERNS = {
    "train_step": re.compile(rf"step:(?P<value>\d+)/\d+ train_loss:{FLOAT_VALUE_RAW}"),
    "val_step": re.compile(rf"step:(?P<value>\d+)/\d+ val_loss:{FLOAT_VALUE_RAW}"),
    "wallclock_stop_step": re.compile(r"stopping_early: wallclock_cap .*?step:(?P<value>\d+)/\d+"),
    "step_avg_ms": re.compile(rf"step:\d+/\d+ .*?step_avg:{FLOAT_VALUE}ms"),
    "train_loss": re.compile(rf"step:\d+/\d+ train_loss:{FLOAT_VALUE}"),
    "val_loss": re.compile(rf"val_loss:{FLOAT_VALUE}"),
    "val_bpb": re.compile(rf"val_bpb:{FLOAT_VALUE}"),
    "final_export_val_loss": re.compile(rf"final_export_roundtrip val_loss:{FLOAT_VALUE}"),
    "final_export_val_bpb": re.compile(rf"final_export_roundtrip val_loss:{FLOAT_VALUE_RAW} val_bpb:{FLOAT_VALUE}"),
    "artifact_model_bytes": re.compile(r"Serialized model int\d+(?:/ternary\d+)?(?:/lqer\d+)?\+\w+: (?P<value>\d+) bytes"),
    "artifact_total_bytes": re.compile(r"Total submission size int\d+(?:/ternary\d+)?(?:/lqer\d+)?\+\w+: (?P<value>\d+) bytes"),
    "artifact_headroom": re.compile(r"submission_cap:\d+ size:\d+ headroom:(?P<value>-?\d+) status:\w+"),
    "peak_alloc_mib": re.compile(r"peak memory allocated: (?P<value>\d+) MiB"),
    "peak_reserved_mib": re.compile(r"reserved: (?P<value>\d+) MiB"),
}


def msvc_build_env() -> dict[str, str]:
    global _MSVC_ENV_CACHE
    if _MSVC_ENV_CACHE is not None:
        return _MSVC_ENV_CACHE

    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"),
    ]
    devcmd = next((path for path in candidates if path.exists()), None)
    if devcmd is None:
        _MSVC_ENV_CACHE = {}
        return _MSVC_ENV_CACHE

    cmd = f'cmd.exe /d /s /c "call "{devcmd}" -arch=x64 -host_arch=x64 >nul && set"'
    proc = subprocess.run(
        cmd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0:
        _MSVC_ENV_CACHE = {}
        return _MSVC_ENV_CACHE

    parsed: dict[str, str] = {}
    keep = {
        "PATH": "PATH",
        "INCLUDE": "INCLUDE",
        "LIB": "LIB",
        "LIBPATH": "LIBPATH",
        "VCTOOLSINSTALLDIR": "VCToolsInstallDir",
        "VCINSTALLDIR": "VCINSTALLDIR",
        "WINDOWSSDKDIR": "WindowsSdkDir",
        "WINDOWSSDKVERSION": "WindowsSDKVersion",
    }
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        canonical = keep.get(key.upper())
        if canonical is not None:
            parsed[canonical] = value
    _MSVC_ENV_CACHE = parsed
    return _MSVC_ENV_CACHE


def configure_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(msvc_build_env())
    venv_scripts = PYTHON.parent
    cuda_home = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6")
    cuda_bin = cuda_home / "bin"
    old_path = env.get("PATH", "")
    parts = [part for part in old_path.split(os.pathsep) if part and r"cuda\v11.7\bin" not in part.lower()]
    if cuda_bin.exists():
        env["CUDA_HOME"] = str(cuda_home)
        env["CUDA_PATH"] = str(cuda_home)
        env["CMAKE_CUDA_COMPILER"] = str(cuda_bin / "nvcc.exe")
        env["PATH"] = os.pathsep.join([str(venv_scripts), str(cuda_bin), *parts])
    else:
        env["PATH"] = os.pathsep.join([str(venv_scripts), *parts])
    env.setdefault("TORCH_CUDA_ARCH_LIST", "7.5")
    return env


def run_command(args: list[str], env: dict[str, str], timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        stdout += f"\nTIMEOUT after {timeout}s\n"
        return subprocess.CompletedProcess(args, 124, stdout)


def parse_bench(stdout: str) -> dict[str, object]:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            obj = ast.literal_eval(line)
            if isinstance(obj, dict):
                return obj
    raise ValueError("bench output did not include a result dict")


def parse_train(stdout: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, pattern in TRAIN_PATTERNS.items():
        matches = list(pattern.finditer(stdout))
        if not matches:
            continue
        raw = matches[-1].group("value")
        out[key] = int(raw) if raw.isdigit() else float(raw)
    return out


def merged_train_output(stdout: str, run_id: str) -> str:
    """Combine captured stdout with the trainer's logfile when subprocess output is sparse."""

    log_path = ROOT / "logs" / f"{run_id}.txt"
    if not log_path.exists():
        return stdout
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return stdout
    if not log_text:
        return stdout
    if stdout and log_text in stdout:
        return stdout
    return f"{stdout.rstrip()}\n\n--- trainer logfile: {log_path} ---\n{log_text}".lstrip()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_train_pairs(value: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for item in split_csv(value):
        if ":" not in item:
            raise ValueError(f"train pair must be PROFILE:PRESET, got {item!r}")
        profile, preset = item.split(":", 1)
        profile = profile.strip()
        preset = preset.strip()
        if not profile or not preset:
            raise ValueError(f"train pair must be PROFILE:PRESET, got {item!r}")
        pairs.append((profile, preset))
    return pairs


def parse_env_assignments(value: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in split_csv(value):
        if "=" not in item:
            raise ValueError(f"env override must be KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"env override must be KEY=VALUE, got {item!r}")
        out[key] = val.strip()
    return out


def run_probe(out_dir: Path, profiles: list[str], preset: str, vocab_size: int) -> None:
    env = configure_env()
    env["SUB4_SPEED_PRESET"] = preset
    env["VOCAB_SIZE"] = str(vocab_size)
    proc = run_command([str(PYTHON), "scripts/probe_sub4_profiles.py", *profiles], env, timeout=240)
    (out_dir / "probe_profiles.md").write_text(proc.stdout, encoding="utf-8")
    selected = [line for line in proc.stdout.splitlines() if any(f"`{profile}`" in line for profile in profiles)]
    (out_dir / "probe_selected.md").write_text("\n".join(selected) + "\n", encoding="utf-8")


def run_bench_matrix(
    out_dir: Path,
    profiles: list[str],
    presets: list[str],
    steps: int,
    warmup: int,
    vocab_size: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile in profiles:
        for preset in presets:
            env = configure_env()
            env.update(
                {
                    "SUB4_PROFILE": profile,
                    "SUB4_SPEED_PRESET": preset,
                    "VOCAB_SIZE": str(vocab_size),
                    "BENCH_STEPS": str(steps),
                    "BENCH_WARMUP": str(warmup),
                    "BENCH_OPTIMIZER": "adamw",
                    "BENCH_PHASES": "1",
                }
            )
            started = time.perf_counter()
            proc = run_command([str(PYTHON), "scripts/bench_sub4_ternary.py"], env, timeout=360)
            raw_path = out_dir / f"bench_{profile}__{preset}.txt"
            raw_path.write_text(proc.stdout, encoding="utf-8")
            row: dict[str, object] = {
                "profile": profile,
                "preset": preset,
                "returncode": proc.returncode,
                "elapsed_s": round(time.perf_counter() - started, 3),
                "raw_log": raw_path.name,
            }
            if proc.returncode == 0:
                try:
                    row.update(parse_bench(proc.stdout))
                except Exception as exc:
                    row["parse_error"] = str(exc)
            else:
                row["error_tail"] = "\n".join(proc.stdout.splitlines()[-12:])
            rows.append(row)
            print(f"bench {profile} {preset} rc={proc.returncode}")
    write_csv(out_dir / "bench.csv", rows)
    return rows


def run_train_matrix(
    out_dir: Path,
    profiles: list[str],
    presets: list[str],
    iterations: int,
    val_tokens: int,
    vocab_size: int,
    pairs: list[tuple[str, str]] | None = None,
    timeout: int = 360,
    env_overrides: dict[str, str] | None = None,
    final_artifacts: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    jobs = pairs if pairs is not None else [(profile, preset) for profile in profiles for preset in presets]
    for profile, preset in jobs:
        run_id = f"matrix_{profile}__{preset}__{iterations}"
        env = configure_env()
        env.update(
            {
                "SUB4_PROFILE": profile,
                "SUB4_SPEED_PRESET": preset,
                "VOCAB_SIZE": str(vocab_size),
                "ITERATIONS": str(iterations),
                "VAL_TOKENS_LIMIT": str(val_tokens),
                "TRAIN_LOG_EVERY": str(iterations),
                "RUN_ID": run_id,
                "SKIP_FINAL_ARTIFACTS": "0" if final_artifacts else "1",
                "MAX_WALLCLOCK_SECONDS": "0",
                "PYTHONUNBUFFERED": "1",
            }
        )
        if iterations > 1000:
            env["TRAIN_LOG_EVERY"] = str(min(1000, max(100, iterations // 5)))
        if env_overrides:
            env.update(env_overrides)
        started = time.perf_counter()
        proc = run_command([str(PYTHON), "train_gpt_ternary.py"], env, timeout=timeout)
        stdout = merged_train_output(proc.stdout, run_id)
        raw_path = out_dir / f"train_{profile}__{preset}.txt"
        raw_path.write_text(stdout, encoding="utf-8")
        row: dict[str, object] = {
            "profile": profile,
            "preset": preset,
            "iterations": iterations,
            "final_artifacts": int(final_artifacts),
            "returncode": proc.returncode,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "raw_log": raw_path.name,
            "run_id": run_id,
        }
        if env_overrides:
            row["env_overrides"] = ",".join(f"{key}={value}" for key, value in sorted(env_overrides.items()))
        if proc.returncode == 0:
            row.update(parse_train(stdout))
        else:
            parsed = parse_train(stdout)
            if parsed:
                row.update(parsed)
            row["error_tail"] = "\n".join(stdout.splitlines()[-12:])
        rows.append(row)
        print(f"train {profile} {preset} rc={proc.returncode}")
    write_csv(out_dir / "train.csv", rows)
    return rows


def make_summary(out_dir: Path, bench_rows: list[dict[str, object]], train_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Sub-4MB Nano/Micro Matrix",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Fastest Synthetic Bench",
        "",
    ]
    good_bench = [row for row in bench_rows if row.get("returncode") == 0 and "ms_per_step" in row]
    for row in sorted(good_bench, key=lambda r: float(r["ms_per_step"]))[:12]:
        lines.append(
            f"- {row['profile']} / {row['preset']}: {row['ms_per_step']} ms, "
            f"{row.get('tokens_per_sec')} tok/s, params={row.get('params')}"
        )
    lines.extend(["", "## Best Real Proxy Val", ""])
    good_train = [row for row in train_rows if row.get("returncode") == 0 and "val_loss" in row]
    def score_key(row: dict[str, object]) -> float:
        return float(row.get("final_export_val_bpb", row.get("val_bpb", row["val_loss"])))
    for row in sorted(good_train, key=score_key)[:12]:
        score = (
            f"final_bpb={row['final_export_val_bpb']}"
            if "final_export_val_bpb" in row
            else f"val_bpb={row.get('val_bpb', 'n/a')}"
        )
        lines.append(
            f"- {row['profile']} / {row['preset']}: val={row['val_loss']}, {score}, "
            f"step_avg={row.get('step_avg_ms')} ms, peak={row.get('peak_alloc_mib')} MiB, "
            f"artifact={row.get('artifact_total_bytes', 'n/a')} headroom={row.get('artifact_headroom', 'n/a')}"
        )
    lines.extend(["", "## Files", "", "- `bench.csv`", "- `train.csv`", "- `probe_selected.md`"])
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="")
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--bench-presets", default=",".join(DEFAULT_BENCH_PRESETS))
    parser.add_argument("--train-presets", default=",".join(DEFAULT_TRAIN_PRESETS))
    parser.add_argument("--bench-steps", type=int, default=40)
    parser.add_argument("--bench-warmup", type=int, default=10)
    parser.add_argument("--train-iterations", type=int, default=40)
    parser.add_argument("--train-timeout", type=int, default=360)
    parser.add_argument("--val-tokens", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--probe-vocab-size", type=int, default=8192)
    parser.add_argument("--train-pairs", default="")
    parser.add_argument("--train-env", default="")
    parser.add_argument("--final-artifacts", action="store_true")
    parser.add_argument("--skip-probe", action="store_true")
    parser.add_argument("--skip-bench", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    profiles = split_csv(args.profiles)
    bench_presets = split_csv(args.bench_presets)
    train_presets = split_csv(args.train_presets)
    train_pairs = parse_train_pairs(args.train_pairs) if args.train_pairs else None
    train_env = parse_env_assignments(args.train_env) if args.train_env else None
    out_dir = Path(args.out) if args.out else ROOT / "records" / f"sub4_micro_matrix_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_probe:
        (out_dir / "probe_profiles.md").write_text("# Probe skipped\n", encoding="utf-8")
        (out_dir / "probe_selected.md").write_text("# Probe skipped\n", encoding="utf-8")
    else:
        run_probe(out_dir, profiles, bench_presets[0], args.probe_vocab_size)
    if args.skip_bench:
        bench_rows: list[dict[str, object]] = []
        write_csv(out_dir / "bench.csv", bench_rows)
    else:
        bench_rows = run_bench_matrix(out_dir, profiles, bench_presets, args.bench_steps, args.bench_warmup, args.vocab_size)
    train_rows: list[dict[str, object]] = []
    if not args.skip_train:
        train_rows = run_train_matrix(
            out_dir,
            profiles,
            train_presets,
            args.train_iterations,
            args.val_tokens,
            args.vocab_size,
            train_pairs,
            args.train_timeout,
            train_env,
            args.final_artifacts,
        )
    make_summary(out_dir, bench_rows, train_rows)
    print(out_dir)


if __name__ == "__main__":
    main()
