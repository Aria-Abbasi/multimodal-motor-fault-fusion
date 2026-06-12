"""Download and validate the complete four-load CWRU early-fault benchmark."""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import scipy.io as sio


OFFICIAL_BASE_URL = "https://engineering.case.edu/sites/default/files"


@dataclass(frozen=True)
class CWRUFile:
    remote_name: str
    local_name: str
    load: int
    condition: str


def expected_cwru_files() -> tuple[CWRUFile, ...]:
    rows = []
    groups = (
        ("normal", (97, 98, 99, 100)),
        ("inner_007", (105, 106, 107, 108)),
        ("ball_007", (118, 119, 120, 121)),
        ("outer_007", (130, 131, 132, 133)),
    )
    for condition, identifiers in groups:
        for load, identifier in enumerate(identifiers):
            if condition == "normal":
                local = f"Time_Normal_{load}_{identifier}.mat"
            elif condition == "inner_007":
                local = f"IR007_{load}_{identifier}.mat"
            elif condition == "ball_007":
                local = f"B007_{load}_{identifier}.mat"
            else:
                local = f"OR007_6_{load}_{identifier}.mat"
            rows.append(CWRUFile(f"{identifier}.mat", local, load, condition))
    return tuple(rows)


def validate_cwru_file(path: Path) -> None:
    """Reject HTML/error downloads and MAT files without drive-end vibration."""
    if not path.exists() or path.stat().st_size < 1_000:
        raise ValueError(f"CWRU file is missing or implausibly small: {path}")
    try:
        mat = sio.loadmat(path)
    except Exception as error:
        raise ValueError(f"Unreadable CWRU MAT file {path}: {error}") from error
    if not any("DE_time" in key for key in mat):
        raise ValueError(f"No DE_time vibration channel found in {path}")


def validate_cwru_benchmark(output_dir: Path) -> dict[int, int]:
    """Validate all 16 expected files and return counts by motor load."""
    counts = {load: 0 for load in range(4)}
    missing = []
    for specification in expected_cwru_files():
        path = output_dir / specification.local_name
        if not path.exists():
            missing.append(path.name)
            continue
        validate_cwru_file(path)
        counts[specification.load] += 1
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} CWRU benchmark files: {missing}"
        )
    if set(counts.values()) != {4}:
        raise ValueError(f"Expected four conditions for each load, found {counts}")
    return counts


def download_cwru_benchmark(
    output_dir: Path,
    base_url: str = OFFICIAL_BASE_URL,
) -> dict[int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for specification in expected_cwru_files():
        output_path = output_dir / specification.local_name
        if output_path.exists():
            validate_cwru_file(output_path)
            continue
        temporary = output_path.with_suffix(".mat.part")
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/{specification.remote_name}",
            headers={"User-Agent": "early-fault-fusion-paper/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                temporary.write_bytes(response.read())
            validate_cwru_file(temporary)
            temporary.replace(output_path)
        finally:
            temporary.unlink(missing_ok=True)
    return validate_cwru_benchmark(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/raw/cwru/raw")
    parser.add_argument("--base-url", default=OFFICIAL_BASE_URL)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    counts = (
        validate_cwru_benchmark(output_dir)
        if args.validate_only
        else download_cwru_benchmark(output_dir, args.base_url)
    )
    print(f"Validated CWRU loads: {counts}")


if __name__ == "__main__":
    main()
