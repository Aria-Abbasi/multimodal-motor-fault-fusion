"""Validate leakage-safe split files.

Checks:
- required files exist
- required columns exist
- no recording_id appears in more than one split within a protocol/fold
- no base_recording_id appears in more than one split within a protocol/fold
- no excluded/unreadable rows are included
- reports class and condition counts
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "protocol",
    "fold_id",
    "dataset",
    "recording_id",
    "base_recording_id",
    "split",
    "health_label",
    "fault_family",
    "operating_condition_id",
    "exclude_from_training",
}


DEFAULT_SPLIT_FILES = [
    "data/splits/nln_emp_leave_one_speed_out.csv",
    "data/splits/paderborn_condition_generalization.csv",
    "data/splits/paderborn_artificial_to_natural.csv",
    "data/splits/cwru_leave_one_load_out.csv",
]


def read_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")

    return df


def validate_no_overlap(df: pd.DataFrame, path: Path) -> list[str]:
    errors = []

    for (protocol, fold_id), group in df.groupby(["protocol", "fold_id"], dropna=False):
        for key_col in ["recording_id", "base_recording_id"]:
            split_sets = {
                split: set(sub[key_col].astype(str))
                for split, sub in group.groupby("split", dropna=False)
            }

            splits = sorted(split_sets)
            for i, a in enumerate(splits):
                for b in splits[i + 1 :]:
                    overlap = split_sets[a] & split_sets[b]
                    if overlap:
                        examples = sorted(overlap)[:10]
                        errors.append(
                            f"{path}: leakage in {protocol}/{fold_id}: "
                            f"{key_col} overlap between {a} and {b}: "
                            f"{len(overlap)} examples={examples}"
                        )

    return errors


def validate_excluded(df: pd.DataFrame, path: Path) -> list[str]:
    errors = []
    excluded = df["exclude_from_training"].astype(str).str.lower() == "true"
    if excluded.any():
        errors.append(f"{path}: contains {excluded.sum()} exclude_from_training rows")
    return errors


def report(df: pd.DataFrame, path: Path) -> None:
    print("\n" + "=" * 100)
    print(path)
    print("=" * 100)
    print(f"Rows: {len(df)}")

    print("\nRows by protocol/fold/split:")
    print(
        df.groupby(["protocol", "fold_id", "split"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .to_string(index=False)
    )

    print("\nBase recording counts by split:")
    print(
        df.groupby(["protocol", "fold_id", "split"], dropna=False)["base_recording_id"]
        .nunique()
        .rename("base_recordings")
        .reset_index()
        .to_string(index=False)
    )

    print("\nHealth label counts:")
    print(
        df.groupby(["protocol", "fold_id", "split", "health_label"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .to_string(index=False)
    )

    print("\nOperating condition counts:")
    print(
        df.groupby(["protocol", "fold_id", "split", "operating_condition_id"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .head(80)
        .to_string(index=False)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate split CSV files.")
    parser.add_argument(
        "--split-files",
        nargs="*",
        default=DEFAULT_SPLIT_FILES,
        help="Split CSV files to validate.",
    )
    args = parser.parse_args()

    all_errors = []

    for split_file in args.split_files:
        path = Path(split_file)
        df = read_split(path)

        report(df, path)

        all_errors.extend(validate_no_overlap(df, path))
        all_errors.extend(validate_excluded(df, path))

    print("\n" + "=" * 100)
    print("Validation result")
    print("=" * 100)

    if all_errors:
        print(f"FAILED with {len(all_errors)} error(s):")
        for error in all_errors:
            print(" -", error)
        raise SystemExit(1)

    print("PASSED: zero recording/base-recording overlap across train/val/test.")


if __name__ == "__main__":
    main()
