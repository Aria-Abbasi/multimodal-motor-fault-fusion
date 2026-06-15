#!/usr/bin/env python3
"""Print a compact status dashboard for the NLN validation pilot."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CACHE_PATTERN = re.compile(
    r"Protocol RAM cache:\s+(\d+)%.*?\|\s*(\d+)/(\d+)"
    r"\s+\[(.*?)<([^,]+),\s*([0-9.]+)it/s\]"
)
JOB_PATTERN = re.compile(r"\[(\d+)/(\d+)\]\s+(running|skipping completed)\s+(.+)")
EPOCH_PATTERN = re.compile(
    r"(\S+)\s+epoch\s+(\d+)/(\d+)\s+\|\s+loss=([0-9.eE+-]+)"
    r"\s+\|\s+val_f1=(\S+).*?grad_norm=(\S+)"
)
FOLD_PATTERN = re.compile(r"Fold\s+(\S+)\s+cache ready")


def command_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def read_log_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    return [
        ANSI_ESCAPE.sub("", line).strip()
        for line in text.splitlines()
        if line.strip()
    ]


def latest_match(lines: list[str], pattern: re.Pattern[str]):
    for line in reversed(lines):
        match = pattern.search(line)
        if match:
            return match
    return None


def completed_rows(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    completed = sum(row.get("status") == "COMPLETED" for row in rows)
    failed = sum(row.get("status") == "FAILED" for row in rows)
    return completed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log", default="artifacts/logs/nln_validation_pilot.log"
    )
    parser.add_argument(
        "--results", default="results/tables/nln_validation_pilot.csv"
    )
    parser.add_argument("--total-jobs", type=int, default=48)
    args = parser.parse_args()

    lines = read_log_lines(Path(args.log))
    completed, failed = completed_rows(Path(args.results))
    session = command_output(["tmux", "has-session", "-t", "nln_pilot"])
    session_state = "RUNNING" if session == "" else "STOPPED"

    cache = latest_match(lines, CACHE_PATTERN)
    job = latest_match(lines, JOB_PATTERN)
    epoch = latest_match(lines, EPOCH_PATTERN)
    fold = latest_match(lines, FOLD_PATTERN)

    if completed >= args.total_jobs:
        phase = "COMPLETE"
    elif cache and (not job or int(cache.group(2)) < int(cache.group(3))):
        phase = "CACHING TENSORS"
    elif job:
        phase = "TRAINING"
    else:
        phase = "STARTING"

    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    gpu_parts = [part.strip() for part in gpu.split(",")]
    memory = command_output(["free", "-h", "--si"])
    memory_line = next(
        (line for line in memory.splitlines() if line.startswith("Mem:")),
        "Mem: unavailable",
    )
    memory_parts = memory_line.split()

    print("NLN Validation Pilot")
    print("=" * 52)
    print(f"Session       : {session_state}")
    print(f"Phase         : {phase}")
    print(
        f"Results       : {completed}/{args.total_jobs} completed"
        f" | {failed} failed"
    )
    if fold:
        print(f"Cached fold   : {fold.group(1)}")
    if cache and phase == "CACHING TENSORS":
        print(
            f"Cache         : {cache.group(1)}% "
            f"({int(cache.group(2)):,}/{int(cache.group(3)):,})"
        )
        print(f"Cache ETA     : {cache.group(5)} at {cache.group(6)} tensors/s")
    if job:
        print(
            f"Current job   : {job.group(1)}/{job.group(2)} "
            f"{job.group(4)}"
        )
    if epoch:
        print(
            f"Latest epoch  : {epoch.group(1)} {epoch.group(2)}/{epoch.group(3)}"
        )
        print(
            f"Metrics       : loss={epoch.group(4)} "
            f"val_f1={epoch.group(5)} grad_norm={epoch.group(6)}"
        )
    if len(gpu_parts) == 4:
        print(
            f"GPU           : {gpu_parts[0]}% | "
            f"{gpu_parts[1]}/{gpu_parts[2]} MiB | {gpu_parts[3]} C"
        )
    else:
        print(f"GPU           : {gpu}")
    if len(memory_parts) >= 7:
        print(
            f"RAM           : {memory_parts[2]} used | "
            f"{memory_parts[6]} available"
        )
    else:
        print(f"RAM           : {memory_line}")

    if lines:
        event = next(
            (
                line
                for line in reversed(lines)
                if "Protocol RAM cache:" not in line
            ),
            lines[-1],
        )
        print(f"Last event    : {event[:120]}")


if __name__ == "__main__":
    main()
