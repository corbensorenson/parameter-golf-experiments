from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


PROFILE_RE = re.compile(r"^profile=([A-Za-z0-9_]+)", re.MULTILINE)
VAL_RE = re.compile(r"step:(\d+)/(\d+) val_loss:[0-9.]+ val_bpb:([0-9.]+)")
FINAL_RE = re.compile(
    r"final_export_roundtrip(?:_exact)? val_loss:[0-9.]+ val_bpb:([0-9.]+)"
)
CAP_RE = re.compile(r"submission_cap:(\d+) size:(\d+) headroom:([-0-9]+) status:(\w+)")
STEP_AVG_RE = re.compile(r"step_avg:([0-9.]+)ms")


@dataclass
class Result:
    profile: str
    log_name: str
    last_step: int | None
    total_steps: int | None
    last_val_bpb: float | None
    final_bpb: float | None
    cap_status: str | None
    cap_size: int | None
    cap_headroom: int | None
    step_avg_ms: float | None
    nonfinite: bool

    @property
    def rank_key(self) -> tuple[int, float, float]:
        cap_bad = 0 if self.cap_status == "ok" else 1
        final = self.final_bpb if self.final_bpb is not None else 999.0
        last = self.last_val_bpb if self.last_val_bpb is not None else 999.0
        return (cap_bad, final, last)


def parse_log(path: Path) -> Result | None:
    if path.name.endswith(".err.txt"):
        return None
    if path.suffix != ".txt":
        return None
    if any(marker in path.name for marker in (".suite.", ".status.", ".host.")):
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    profile_match = PROFILE_RE.search(text)
    if not profile_match:
        return None
    vals = VAL_RE.findall(text)
    finals = FINAL_RE.findall(text)
    caps = CAP_RE.findall(text)
    avgs = STEP_AVG_RE.findall(text)
    last_step = total_steps = None
    last_val_bpb = None
    if vals:
        step, total, bpb = vals[-1]
        last_step = int(step)
        total_steps = int(total)
        last_val_bpb = float(bpb)
    cap_status = None
    cap_size = None
    cap_headroom = None
    if caps:
        _, size, headroom, status = caps[-1]
        cap_status = status
        cap_size = int(size)
        cap_headroom = int(headroom)
    return Result(
        profile=profile_match.group(1),
        log_name=path.name,
        last_step=last_step,
        total_steps=total_steps,
        last_val_bpb=last_val_bpb,
        final_bpb=float(finals[-1]) if finals else None,
        cap_status=cap_status,
        cap_size=cap_size,
        cap_headroom=cap_headroom,
        step_avg_ms=float(avgs[-1]) if avgs else None,
        nonfinite=("non-finite" in text.lower() or "nan" in text.lower()),
    )


def fmt_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def fmt_int(value: int | None) -> str:
    return "" if value is None else str(value)


def build_markdown(results: list[Result]) -> str:
    completed = [item for item in results if item.final_bpb is not None]
    running = [item for item in results if item.final_bpb is None]
    ranked = sorted(completed, key=lambda item: item.rank_key)
    lines: list[str] = []
    lines.append("# CaseOps 2060 Ladder Results")
    lines.append("")
    if ranked:
        leader = ranked[0]
        lines.append(
            f"Current cap-aware leader: `{leader.profile}` with "
            f"`final_export_roundtrip val_bpb {leader.final_bpb:.6f}` "
            f"and cap `{leader.cap_status}`."
        )
    else:
        lines.append("No completed export-proof stages found yet.")
    lines.append("")
    lines.append("## Completed")
    lines.append("")
    lines.append(
        "| Rank | Profile | Final Export BPB | Cap | Size | Headroom | Last Local BPB | Step | Avg ms | Nonfinite | Log |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for idx, item in enumerate(ranked, start=1):
        step = ""
        if item.last_step is not None and item.total_steps is not None:
            step = f"{item.last_step}/{item.total_steps}"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    f"`{item.profile}`",
                    fmt_float(item.final_bpb),
                    item.cap_status or "",
                    fmt_int(item.cap_size),
                    fmt_int(item.cap_headroom),
                    fmt_float(item.last_val_bpb),
                    step,
                    fmt_float(item.step_avg_ms),
                    "yes" if item.nonfinite else "",
                    f"`{item.log_name}`",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Incomplete Or Running")
    lines.append("")
    lines.append("| Profile | Last Local BPB | Step | Avg ms | Nonfinite | Log |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in sorted(running, key=lambda r: (r.profile, r.log_name)):
        step = ""
        if item.last_step is not None and item.total_steps is not None:
            step = f"{item.last_step}/{item.total_steps}"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.profile}`",
                    fmt_float(item.last_val_bpb),
                    step,
                    fmt_float(item.step_avg_ms),
                    "yes" if item.nonfinite else "",
                    f"`{item.log_name}`",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Reference targets:")
    lines.append("")
    lines.append("- Near-cap q6-all legal fallback: `1.62205130` roundtrip BPB.")
    lines.append("- Near-cap q6 quality reference, over cap: `1.57438639` roundtrip BPB.")
    lines.append("- A loop winner must either beat `1.62205130` directly or get close while leaving enough bytes to widen the model.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, default=Path(__file__).resolve().parents[3] / "logs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    results: list[Result] = []
    for path in args.logs.glob("caseops-*.txt"):
        parsed = parse_log(path)
        if parsed is not None:
            results.append(parsed)
    markdown = build_markdown(results)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
