"""Create a raw-data manifest for NLN-EMP, Paderborn, and CWRU.

Output:
    data/metadata/raw_manifest.csv

The manifest is intentionally raw-file-level only. It does not preprocess,
window, split, or modify any dataset files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


DATASET_DIR_NAMES = {
    "nln_emp": "nln_emp",
    "nln-emp": "nln_emp",
    "nln": "nln_emp",
    "cwru": "cwru",
    "paderborn": "paderborn",
    "pu": "paderborn",
}


@dataclass(frozen=True)
class ManifestRow:
    dataset: str
    original_filename: str
    absolute_path: str
    relative_path: str
    extension: str
    file_size: int
    checksum_sha256: str
    likely_sensor_type: str
    candidate_recording_id: str
    notes: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def detect_dataset(path: Path, raw_root: Path) -> str:
    """Detect dataset name from first directory below data/raw."""
    try:
        first_part = path.relative_to(raw_root).parts[0].lower()
    except Exception:
        return "unknown"

    return DATASET_DIR_NAMES.get(first_part, first_part)


def infer_sensor_type(path: Path, dataset: str) -> str:
    """Infer likely sensor type using path and filename patterns.

    This is conservative. Ambiguous files are marked as unknown rather than guessed.
    """
    text = str(path).lower()
    name = path.name.lower()

    if dataset == "nln_emp":
        if "/electric/" in text or "\\electric\\" in text or name.startswith("electric_"):
            return "current"
        if "/vibration/" in text or "\\vibration\\" in text or name.startswith("vibration_"):
            return "vibration"
        return "metadata_or_documentation"

    if dataset == "cwru":
        if path.suffix.lower() == ".mat":
            # CWRU .mat files commonly contain drive-end/fan-end/base accelerometer signals.
            return "vibration"
        if "feature" in name:
            return "derived_features"
        if path.suffix.lower() == ".npz":
            return "derived_or_packaged_data"
        return "unknown"

    if dataset == "paderborn":
        # Paderborn bearing data is usually vibration/current rich in MATLAB files,
        # but file naming varies by release. Mark multi-sensor when likely.
        if path.suffix.lower() in {".mat", ".npz", ".tdms", ".csv"}:
            if any(token in text for token in ["current", "curr", "phase", "electric"]):
                return "current"
            if any(token in text for token in ["vibration", "vib", "acc", "accelerometer"]):
                return "vibration"
            return "unknown_or_multisensor"
        return "metadata_or_documentation"

    # Generic fallback.
    if any(token in text for token in ["electric", "current", "phase"]):
        return "current"
    if any(token in text for token in ["vibration", "vib", "acc", "accelerometer"]):
        return "vibration"
    return "unknown"


def normalize_recording_id(value: str) -> str:
    """Make a stable, filesystem-safe recording ID candidate."""
    value = value.strip()
    value = re.sub(r"\.[^.]+$", "", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value.lower()


def recover_nln_recording_id(path: Path) -> str:
    """Recover NLN-EMP recording ID from filename or directory structure.

    Example:
      Electric_Motor-2_100_time-bearing bpfi 1-ch3.csv
    should become:
      motor_2_100_bearing_bpfi_1

    Channel is intentionally removed so ch1/ch2/... from the same condition share
    a recording-level ID candidate.
    """
    stem = path.stem

    # Strip modality prefix.
    stem = re.sub(r"^(Electric|Vibration)_", "", stem, flags=re.IGNORECASE)

    # Strip channel suffix.
    stem = re.sub(r"-ch\d+$", "", stem, flags=re.IGNORECASE)

    # Remove literal time marker while preserving condition.
    stem = stem.replace("_time-", "_")
    stem = stem.replace("_time_", "_")

    return normalize_recording_id(stem)


def recover_cwru_recording_id(path: Path) -> str:
    """Recover CWRU recording ID from common filename pattern."""
    return normalize_recording_id(path.stem)


def recover_paderborn_recording_id(path: Path) -> str:
    """Recover Paderborn recording ID.

    Paderborn filenames often encode condition and run IDs. Since variants exist,
    keep the stem as the first stable candidate.
    """
    return normalize_recording_id(path.stem)


def recover_recording_id(path: Path, dataset: str) -> str:
    if dataset == "nln_emp":
        return recover_nln_recording_id(path)
    if dataset == "cwru":
        return recover_cwru_recording_id(path)
    if dataset == "paderborn":
        return recover_paderborn_recording_id(path)
    return normalize_recording_id(path.stem)


def notes_for_file(path: Path, dataset: str, likely_sensor_type: str) -> str:
    notes: list[str] = []

    suffix = path.suffix.lower()
    if suffix in {".pdf", ".xlsx", ".xls", ".txt", ".doc", ".docx"}:
        notes.append("documentation_or_metadata_file")

    if dataset == "nln_emp":
        if likely_sensor_type == "current":
            notes.append("nln_emp_electric_branch")
        elif likely_sensor_type == "vibration":
            notes.append("nln_emp_vibration_branch")
        if re.search(r"-ch\d+\.csv$", path.name, flags=re.IGNORECASE):
            notes.append("channel_file")

    if dataset == "cwru":
        if suffix == ".mat":
            notes.append("cwru_mat_recording")
        elif suffix in {".csv", ".npz"}:
            notes.append("derived_or_prepackaged_file")

    if likely_sensor_type in {"unknown", "unknown_or_multisensor"}:
        notes.append("sensor_type_needs_manual_check")

    return ";".join(notes)


def iter_files(raw_root: Path) -> Iterable[Path]:
    """Yield all files below raw root in stable order."""
    yield from sorted((p for p in raw_root.rglob("*") if p.is_file()), key=lambda x: str(x))


def build_manifest(raw_root: Path, project_root: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []

    for path in iter_files(raw_root):
        dataset = detect_dataset(path, raw_root)
        likely_sensor_type = infer_sensor_type(path, dataset)
        candidate_recording_id = recover_recording_id(path, dataset)
        checksum = sha256_file(path)

        try:
            relative_path = str(path.relative_to(project_root))
        except ValueError:
            relative_path = str(path)

        row = ManifestRow(
            dataset=dataset,
            original_filename=path.name,
            absolute_path=str(path.resolve()),
            relative_path=relative_path,
            extension=path.suffix.lower().lstrip("."),
            file_size=path.stat().st_size,
            checksum_sha256=checksum,
            likely_sensor_type=likely_sensor_type,
            candidate_recording_id=candidate_recording_id,
            notes=notes_for_file(path, dataset, likely_sensor_type),
        )
        rows.append(row)

    return rows


def write_csv(rows: list[ManifestRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(asdict(rows[0]).keys()) if rows else [
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
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create raw-data manifest CSV.")
    parser.add_argument("--raw-root", default="data/raw", help="Raw data root directory.")
    parser.add_argument(
        "--output",
        default="data/metadata/raw_manifest.csv",
        help="Output manifest CSV path.",
    )
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    raw_root = Path(args.raw_root).resolve()
    output_path = Path(args.output)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root does not exist: {raw_root}")

    rows = build_manifest(raw_root=raw_root, project_root=project_root)
    write_csv(rows, output_path)

    dataset_counts: dict[str, int] = {}
    sensor_counts: dict[str, int] = {}

    for row in rows:
        dataset_counts[row.dataset] = dataset_counts.get(row.dataset, 0) + 1
        sensor_counts[row.likely_sensor_type] = sensor_counts.get(row.likely_sensor_type, 0) + 1

    print(f"Wrote manifest: {output_path}")
    print(f"Total files: {len(rows)}")
    print("Dataset counts:")
    for dataset, count in sorted(dataset_counts.items()):
        print(f"  {dataset}: {count}")

    print("Likely sensor type counts:")
    for sensor, count in sorted(sensor_counts.items()):
        print(f"  {sensor}: {count}")


if __name__ == "__main__":
    main()
