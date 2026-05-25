"""Validate complete metadata_master.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "dataset",
    "machine_id",
    "recording_id",
    "base_recording_id",
    "sensor_type",
    "sensor_types_present",
    "speed",
    "speed_rpm",
    "load",
    "pressure_bar",
    "flow_m3h",
    "operating_condition_id",
    "fault_family",
    "severity",
    "health_label",
    "original_label",
    "damage_source",
    "run_id",
    "source_path",
    "source_files",
    "n_source_files",
    "mat_top_level_key",
    "mat_signal_keys",
    "mat_channel_names",
    "is_readable",
    "exclude_from_training",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/metadata/metadata_master.csv")
    args = parser.parse_args()

    path = Path(args.metadata)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    print("Metadata master validation report")
    print("=" * 40)
    print(f"Rows: {len(df)}")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    print(f"Missing required columns: {len(missing_cols)}")
    if missing_cols:
        print(missing_cols)

    duplicate_recording_ids = df["recording_id"].duplicated().sum()
    print(f"Duplicate recording_id values: {duplicate_recording_ids}")

    print("\nDataset counts:")
    print(df["dataset"].value_counts(dropna=False).to_string())

    print("\nDataset x health label:")
    print(pd.crosstab(df["dataset"], df["health_label"], dropna=False).to_string())

    print("\nDataset x sensor type:")
    print(pd.crosstab(df["dataset"], df["sensor_type"], dropna=False).to_string())

    print("\nSensor types present:")
    print(df["sensor_types_present"].value_counts(dropna=False).head(30).to_string())

    print("\nReadable counts:")
    print(df["is_readable"].value_counts(dropna=False).to_string())

    print("\nExcluded from training:")
    print(df["exclude_from_training"].value_counts(dropna=False).to_string())

    print("\nUnknown counts in important columns:")
    for col in [
        "machine_id",
        "sensor_type",
        "sensor_types_present",
        "speed",
        "load",
        "operating_condition_id",
        "fault_family",
        "severity",
        "health_label",
        "damage_source",
    ]:
        unknown = (df[col].astype(str).str.lower() == "unknown").sum()
        blank = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
        print(f"  {col}: unknown={unknown}, blank={blank}")

    bad_sources = []
    for source_path in df["source_path"].astype(str):
        for part in source_path.split("|"):
            if part and not Path(part).exists():
                bad_sources.append(part)

    print(f"\nMissing source path entries: {len(bad_sources)}")
    if bad_sources:
        print("Examples:")
        for x in bad_sources[:10]:
            print(" ", x)

    if missing_cols or duplicate_recording_ids or bad_sources:
        raise SystemExit("Validation failed.")

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
