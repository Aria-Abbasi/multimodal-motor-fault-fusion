"""Validate raw_manifest.csv.

Checks:
- manifest exists
- all manifest paths exist
- unreadable files
- duplicate checksums
- empty files
- missing dataset / recording ID / sensor type
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {
    "dataset",
    "original_filename",
    "absolute_path",
    "relative_path",
    "extension",
    "file_size",
    "checksum_sha256",
    "likely_sensor_type",
    "candidate_recording_id",
    "notes",
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw data manifest.")
    parser.add_argument(
        "--manifest",
        default="data/metadata/raw_manifest.csv",
        help="Path to raw manifest CSV.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = read_manifest(manifest_path)

    missing_paths: list[str] = []
    unreadable_paths: list[str] = []
    empty_files: list[str] = []
    missing_dataset: list[str] = []
    missing_sensor: list[str] = []
    missing_recording_id: list[str] = []

    checksum_to_paths: dict[str, list[str]] = defaultdict(list)
    dataset_counts: Counter[str] = Counter()
    sensor_counts: Counter[str] = Counter()
    extension_counts: Counter[str] = Counter()

    for row in rows:
        abs_path = row["absolute_path"]
        path = Path(abs_path)

        dataset = row["dataset"].strip()
        sensor = row["likely_sensor_type"].strip()
        extension = row["extension"].strip()
        recording_id = row["candidate_recording_id"].strip()
        checksum = row["checksum_sha256"].strip()

        dataset_counts[dataset] += 1
        sensor_counts[sensor] += 1
        extension_counts[extension] += 1

        if not dataset:
            missing_dataset.append(abs_path)
        if not sensor:
            missing_sensor.append(abs_path)
        if not recording_id:
            missing_recording_id.append(abs_path)

        if not path.exists():
            missing_paths.append(abs_path)
            continue

        try:
            with path.open("rb") as f:
                f.read(1)
        except OSError:
            unreadable_paths.append(abs_path)

        try:
            size = int(row["file_size"])
            if size == 0:
                empty_files.append(abs_path)
        except ValueError:
            empty_files.append(abs_path)

        if checksum:
            checksum_to_paths[checksum].append(abs_path)

    duplicate_checksum_groups = {
        checksum: paths
        for checksum, paths in checksum_to_paths.items()
        if len(paths) > 1
    }

    print("Raw manifest validation report")
    print("=" * 32)
    print(f"Rows: {len(rows)}")
    print(f"Missing paths: {len(missing_paths)}")
    print(f"Unreadable files: {len(unreadable_paths)}")
    print(f"Empty files: {len(empty_files)}")
    print(f"Duplicate checksum groups: {len(duplicate_checksum_groups)}")
    print(f"Missing dataset values: {len(missing_dataset)}")
    print(f"Missing sensor values: {len(missing_sensor)}")
    print(f"Missing recording ID values: {len(missing_recording_id)}")

    print("\nDataset counts:")
    for key, value in sorted(dataset_counts.items()):
        print(f"  {key}: {value}")

    print("\nLikely sensor type counts:")
    for key, value in sorted(sensor_counts.items()):
        print(f"  {key}: {value}")

    print("\nTop extension counts:")
    for key, value in extension_counts.most_common(20):
        print(f"  .{key}: {value}")

    if duplicate_checksum_groups:
        print("\nDuplicate checksum examples:")
        for checksum, paths in list(duplicate_checksum_groups.items())[:10]:
            print(f"  {checksum}")
            for path in paths[:5]:
                print(f"    {path}")
            if len(paths) > 5:
                print(f"    ... {len(paths) - 5} more")

    if missing_paths or unreadable_paths:
        raise SystemExit("Validation failed: missing or unreadable files found.")

    print("\nValidation completed.")


if __name__ == "__main__":
    main()
