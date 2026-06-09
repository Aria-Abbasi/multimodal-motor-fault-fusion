"""Regenerate the CWRU leave-one-load-out split without fallback leakage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.generate_splits import cwru_leave_one_load_out


def regenerate_cwru_split(
    metadata_path: Path, output_path: Path
) -> pd.DataFrame:
    """Create a genuine LOSO split, refusing an unsupported one-load dataset."""
    metadata = pd.read_csv(metadata_path)
    cwru = metadata[metadata["dataset"] == "cwru"].copy()
    loads = sorted(
        {
            str(load)
            for load in cwru["load"].dropna().tolist()
            if str(load).strip().lower() not in {"", "unknown", "nan"}
        }
    )
    if len(loads) < 2:
        raise ValueError(
            "CWRU leave-one-load-out requires at least two distinct loads. "
            f"Found {loads}. Download loads 0, 2, and 3 before regenerating."
        )

    split = cwru_leave_one_load_out(metadata)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(output_path, index=False)
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", default="data/metadata/metadata_master.csv"
    )
    parser.add_argument(
        "--output", default="data/splits/cwru_leave_one_load_out.csv"
    )
    args = parser.parse_args()

    try:
        split = regenerate_cwru_split(Path(args.metadata), Path(args.output))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Wrote {args.output}: {len(split)} rows")


if __name__ == "__main__":
    main()
