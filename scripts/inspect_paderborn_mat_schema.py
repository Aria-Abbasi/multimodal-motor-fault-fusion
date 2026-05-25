"""Inspect Paderborn .mat files and summarize internal signal keys."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd


def try_loadmat_keys(path: Path) -> list[str]:
    """Read MATLAB keys using scipy for v7 MAT files, h5py for v7.3 files."""
    try:
        import scipy.io as sio

        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        return sorted(k for k in mat.keys() if not k.startswith("__"))
    except Exception:
        pass

    try:
        import h5py

        keys: list[str] = []

        def visitor(name, obj):
            keys.append(name)

        with h5py.File(path, "r") as f:
            f.visititems(visitor)

        return sorted(keys)
    except Exception as e:
        return [f"READ_ERROR:{type(e).__name__}:{e}"]


def infer_sensor_types_from_keys(keys: list[str]) -> str:
    text = " ".join(keys).lower()
    sensors = []

    if any(x in text for x in ["vib", "acc", "accelerometer", "bearing"]):
        sensors.append("vibration")
    if any(x in text for x in ["current", "curr", "phase", "i1", "i2"]):
        sensors.append("current")
    if any(x in text for x in ["speed", "rpm", "n_"]):
        sensors.append("speed")
    if any(x in text for x in ["torque", "load", "m_"]):
        sensors.append("torque_or_load")
    if any(x in text for x in ["force", "radial"]):
        sensors.append("force")

    return "|".join(sorted(set(sensors))) if sensors else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/metadata/raw_manifest.csv")
    parser.add_argument("--output", default="data/metadata/paderborn_mat_schema.csv")
    parser.add_argument("--max-files", type=int, default=0, help="0 means inspect all Paderborn MAT files.")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    paderborn = manifest[
        (manifest["dataset"] == "paderborn")
        & (manifest["extension"] == "mat")
    ].copy()

    if args.max_files > 0:
        paderborn = paderborn.head(args.max_files)

    rows = []
    pattern_counter = Counter()
    key_examples = defaultdict(list)

    for i, row in paderborn.iterrows():
        path = Path(row["absolute_path"])
        keys = try_loadmat_keys(path)
        key_signature = "|".join(keys)
        sensor_types = infer_sensor_types_from_keys(keys)

        pattern_counter[key_signature] += 1
        if len(key_examples[key_signature]) < 5:
            key_examples[key_signature].append(row["relative_path"])

        rows.append(
            {
                "relative_path": row["relative_path"],
                "original_filename": row["original_filename"],
                "candidate_recording_id": row["candidate_recording_id"],
                "mat_keys": key_signature,
                "n_mat_keys": len(keys),
                "sensor_types_present": sensor_types,
                "read_error": any(k.startswith("READ_ERROR:") for k in keys),
            }
        )

        if len(rows) % 100 == 0:
            print(f"Inspected {len(rows)} / {len(paderborn)} files")

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nWrote: {args.output}")
    print(f"Rows: {len(out)}")
    print("\nSensor type counts:")
    print(out["sensor_types_present"].value_counts(dropna=False).to_string())

    print("\nUnique MAT key signatures:", len(pattern_counter))
    print("\nTop MAT key signatures:")
    for signature, count in pattern_counter.most_common(10):
        print("=" * 80)
        print("Count:", count)
        print("Keys:", signature[:1000])
        print("Examples:")
        for ex in key_examples[signature]:
            print(" ", ex)


if __name__ == "__main__":
    main()
