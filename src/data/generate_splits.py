"""Generate leakage-safe recording-level split files.

Input:
    data/metadata/metadata_master.csv

Outputs:
    data/splits/nln_emp_leave_one_speed_out.csv
    data/splits/paderborn_condition_generalization.csv
    data/splits/paderborn_artificial_to_natural.csv
    data/splits/cwru_leave_one_load_out.csv
    data/splits/split_summary.csv

Important:
    Splits are assigned at recording/base-recording level, not window level.
    No preprocessing or windowing is performed here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


SPLIT_COLUMNS = [
    "protocol",
    "fold_id",
    "dataset",
    "recording_id",
    "base_recording_id",
    "split",
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
    "run_id",
    "source_path",
    "exclude_from_training",
    "notes",
]


def ensure_split_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in SPLIT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[SPLIT_COLUMNS]


def usable(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows usable for split generation."""
    out = df.copy()

    if "exclude_from_training" in out.columns:
        out = out[out["exclude_from_training"].astype(str).str.lower() != "true"]

    if "is_readable" in out.columns:
        out = out[out["is_readable"].astype(str).str.lower() != "false"]

    return out.copy()


def add_val_from_train(train_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Mark a small deterministic validation subset inside train.

    We do this by base_recording_id, never by row/window.
    """
    train_df = train_df.copy()
    train_df["split"] = "train"

    base_ids = sorted(train_df["base_recording_id"].astype(str).unique())
    if len(base_ids) < 5:
        return train_df

    base_table = pd.DataFrame({"base_recording_id": base_ids})
    base_table = base_table.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    n_val = max(1, int(round(0.15 * len(base_table))))
    val_ids = set(base_table.head(n_val)["base_recording_id"])

    train_df.loc[train_df["base_recording_id"].isin(val_ids), "split"] = "val"
    return train_df


def nln_emp_leave_one_speed_out(meta: pd.DataFrame) -> pd.DataFrame:
    """NLN-EMP leave-one-speed-out.

    For each observed speed token, test on that speed and train/val on all others.
    """
    df = usable(meta)
    df = df[df["dataset"] == "nln_emp"].copy()

    rows = []
    speeds = sorted(df["speed"].astype(str).unique())

    for speed in speeds:
        fold_id = f"test_speed_{speed}"

        test_df = df[df["speed"].astype(str) == speed].copy()
        train_df = df[df["speed"].astype(str) != speed].copy()
        train_df = add_val_from_train(train_df, seed=42)

        test_df["split"] = "test"

        fold = pd.concat([train_df, test_df], ignore_index=True)
        fold["protocol"] = "nln_emp_leave_one_speed_out"
        fold["fold_id"] = fold_id
        rows.append(fold)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return ensure_split_columns(out)


def paderborn_condition_generalization(meta: pd.DataFrame) -> pd.DataFrame:
    """Paderborn P1: train on 3 operating conditions, test on the held-out 4th.

    Uses operating_condition_id from metadata_master:
      n09_m07_f10
      n15_m01_f10
      n15_m07_f04
      n15_m07_f10
    """
    df = usable(meta)
    df = df[df["dataset"] == "paderborn"].copy()

    rows = []
    conditions = sorted(df["operating_condition_id"].astype(str).unique())

    for condition in conditions:
        fold_id = f"test_condition_{condition}"

        test_df = df[df["operating_condition_id"].astype(str) == condition].copy()
        train_df = df[df["operating_condition_id"].astype(str) != condition].copy()
        train_df = add_val_from_train(train_df, seed=42)

        test_df["split"] = "test"

        fold = pd.concat([train_df, test_df], ignore_index=True)
        fold["protocol"] = "paderborn_condition_generalization"
        fold["fold_id"] = fold_id
        rows.append(fold)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return ensure_split_columns(out)


def paderborn_artificial_to_natural(meta: pd.DataFrame) -> pd.DataFrame:
    """Paderborn P2: train on healthy + artificial, test on healthy + real.

    To avoid leakage from the same healthy recording appearing in both train and test,
    healthy recordings are partitioned by bearing ID:
      train healthy: K001, K002, ...
      test healthy: held-out healthy bearings by deterministic split

    Artificial faults:
      damage_source == artificial -> train/val

    Real faults:
      damage_source == real -> test
    """
    df = usable(meta)
    df = df[df["dataset"] == "paderborn"].copy()

    artificial = df[df["damage_source"].astype(str) == "artificial"].copy()
    real = df[df["damage_source"].astype(str) == "real"].copy()
    healthy = df[df["damage_source"].astype(str) == "healthy"].copy()

    healthy_bearings = sorted(healthy["machine_id"].astype(str).unique())
    healthy_table = pd.DataFrame({"machine_id": healthy_bearings}).sample(
        frac=1.0, random_state=42
    )

    n_test_healthy = max(1, int(round(0.25 * len(healthy_table)))) if len(healthy_table) else 0
    test_healthy_ids = set(healthy_table.head(n_test_healthy)["machine_id"])

    healthy_test = healthy[healthy["machine_id"].isin(test_healthy_ids)].copy()
    healthy_train = healthy[~healthy["machine_id"].isin(test_healthy_ids)].copy()

    train_df = pd.concat([healthy_train, artificial], ignore_index=True)
    train_df = add_val_from_train(train_df, seed=42)

    test_df = pd.concat([healthy_test, real], ignore_index=True)
    test_df["split"] = "test"

    out = pd.concat([train_df, test_df], ignore_index=True)
    out["protocol"] = "paderborn_artificial_to_natural"
    out["fold_id"] = "artificial_to_real_damage"

    return ensure_split_columns(out)


def cwru_leave_one_load_out(meta: pd.DataFrame) -> pd.DataFrame:
    """CWRU leave-one-load-out.

    Your current CWRU subset appears to contain load=1 only. If only one load exists,
    this script still creates one fold, but train/val will be empty. That is a data
    limitation, not leakage. You may later add loads 0, 2, 3 to make this protocol useful.
    """
    df = usable(meta)
    df = df[df["dataset"] == "cwru"].copy()

    rows = []
    loads = sorted(df["load"].astype(str).unique())

    for load in loads:
        fold_id = f"test_load_{load}"

        test_df = df[df["load"].astype(str) == load].copy()
        train_df = df[df["load"].astype(str) != load].copy()
        if len(train_df):
            train_df = add_val_from_train(train_df, seed=42)
        else:
            train_df["split"] = "train"

        test_df["split"] = "test"

        fold = pd.concat([train_df, test_df], ignore_index=True)
        fold["protocol"] = "cwru_leave_one_load_out"
        fold["fold_id"] = fold_id
        rows.append(fold)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return ensure_split_columns(out)


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    group_cols = ["protocol", "fold_id", "split"]

    for keys, group in df.groupby(group_cols, dropna=False):
        protocol, fold_id, split = keys

        rows.append(
            {
                "protocol": protocol,
                "fold_id": fold_id,
                "split": split,
                "n_rows": len(group),
                "n_recording_id": group["recording_id"].nunique(),
                "n_base_recording_id": group["base_recording_id"].nunique(),
                "n_health_labels": group["health_label"].nunique(),
                "health_label_counts": group["health_label"].value_counts().to_dict(),
                "fault_family_counts": group["fault_family"].value_counts().to_dict(),
                "condition_counts": group["operating_condition_id"].value_counts().to_dict(),
            }
        )

    return pd.DataFrame(rows)


def write_split(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}: {len(df)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leakage-safe split CSVs.")
    parser.add_argument("--metadata", default="data/metadata/metadata_master.csv")
    parser.add_argument("--output-dir", default="data/splits")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_tables = {
        "nln_emp_leave_one_speed_out.csv": nln_emp_leave_one_speed_out(metadata),
        "paderborn_condition_generalization.csv": paderborn_condition_generalization(metadata),
        "paderborn_artificial_to_natural.csv": paderborn_artificial_to_natural(metadata),
        "cwru_leave_one_load_out.csv": cwru_leave_one_load_out(metadata),
    }

    all_splits = []

    for filename, table in split_tables.items():
        path = output_dir / filename
        write_split(table, path)
        all_splits.append(table)

    combined = pd.concat(all_splits, ignore_index=True)
    summary = summarize_split(combined)
    summary_path = output_dir / "split_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}: {len(summary)} rows")

    print("\nSplit summary:")
    print(summary[["protocol", "fold_id", "split", "n_rows", "n_base_recording_id"]].to_string(index=False))


if __name__ == "__main__":
    main()
