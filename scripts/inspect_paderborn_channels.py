"""Inspect Paderborn MAT files and extract nested channel metadata.

Output:
    data/metadata/paderborn_channel_schema.csv

This reads the nested MATLAB structure:
    top_level_recording.X
    top_level_recording.Y

and extracts Y channel names, paths, devices, raster, shape, min/max.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import scipy.io as sio


def safe_str(value: Any) -> str:
    """Convert MATLAB/scipy values to a safe compact string."""
    if value is None:
        return ""
    try:
        if hasattr(value, "shape") and getattr(value, "size", 1) == 0:
            return ""
    except Exception:
        pass
    return str(value)


def infer_channel_sensor_type(text: str) -> str:
    text = text.lower()

    if any(x in text for x in ["vib", "acc", "accelerometer", "bearing", "schwing", "beschleun"]):
        return "vibration"
    if any(x in text for x in ["current", "strom", "phase", "i_1", "i_2", "i1", "i2"]):
        return "current"
    if any(x in text for x in ["speed", "rpm", "drehzahl", "n_"]):
        return "speed"
    if any(x in text for x in ["torque", "moment", "load", "m_"]):
        return "torque_or_load"
    if any(x in text for x in ["force", "kraft", "radial", "f_"]):
        return "force"

    return "unknown"


def load_paderborn_mat(path: Path) -> tuple[str | None, Any | None, str]:
    """Return top-level recording key, object, and read error."""
    try:
        mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if not keys:
            return None, None, "NO_TOP_LEVEL_KEYS"
        key = keys[0]
        return key, mat[key], ""
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def extract_channels(path: Path, relative_path: str, original_filename: str) -> list[dict[str, Any]]:
    top_key, root, read_error = load_paderborn_mat(path)

    if read_error:
        return [
            {
                "relative_path": relative_path,
                "original_filename": original_filename,
                "top_level_key": "",
                "channel_index": -1,
                "channel_name": "",
                "channel_path": "",
                "channel_device": "",
                "channel_raster": "",
                "channel_unit": "",
                "channel_type_code": "",
                "channel_x_index": "",
                "channel_downsampling": "",
                "n_samples": 0,
                "channel_min": "",
                "channel_max": "",
                "inferred_sensor_type": "unreadable",
                "read_error": read_error,
            }
        ]

    rows: list[dict[str, Any]] = []

    y = getattr(root, "Y", None)
    if y is None:
        return [
            {
                "relative_path": relative_path,
                "original_filename": original_filename,
                "top_level_key": top_key or "",
                "channel_index": -1,
                "channel_name": "",
                "channel_path": "",
                "channel_device": "",
                "channel_raster": "",
                "channel_unit": "",
                "channel_type_code": "",
                "channel_x_index": "",
                "channel_downsampling": "",
                "n_samples": 0,
                "channel_min": "",
                "channel_max": "",
                "inferred_sensor_type": "unknown",
                "read_error": "NO_Y_FIELD",
            }
        ]

    # y may be ndarray or a single mat_struct.
    if hasattr(y, "flat"):
        channels = list(y.flat)
    else:
        channels = [y]

    for idx, channel in enumerate(channels):
        name = safe_str(getattr(channel, "Name", ""))
        path_value = safe_str(getattr(channel, "Path", ""))
        device = safe_str(getattr(channel, "Device", ""))
        raster = safe_str(getattr(channel, "Raster", ""))
        unit = safe_str(getattr(channel, "Unit", ""))
        type_code = safe_str(getattr(channel, "Type", ""))
        x_index = safe_str(getattr(channel, "XIndex", ""))
        downsampling = safe_str(getattr(channel, "DownSampling", ""))

        data = getattr(channel, "Data", None)
        n_samples = int(getattr(data, "shape", [0])[0]) if hasattr(data, "shape") and len(data.shape) > 0 else 0

        channel_min = safe_str(getattr(channel, "Min", ""))
        channel_max = safe_str(getattr(channel, "Max", ""))

        combined_text = " ".join([name, path_value, device, raster, unit])
        inferred_sensor_type = infer_channel_sensor_type(combined_text)

        rows.append(
            {
                "relative_path": relative_path,
                "original_filename": original_filename,
                "top_level_key": top_key or "",
                "channel_index": idx,
                "channel_name": name,
                "channel_path": path_value,
                "channel_device": device,
                "channel_raster": raster,
                "channel_unit": unit,
                "channel_type_code": type_code,
                "channel_x_index": x_index,
                "channel_downsampling": downsampling,
                "n_samples": n_samples,
                "channel_min": channel_min,
                "channel_max": channel_max,
                "inferred_sensor_type": inferred_sensor_type,
                "read_error": "",
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/metadata/raw_manifest.csv")
    parser.add_argument("--output", default="data/metadata/paderborn_channel_schema.csv")
    parser.add_argument("--max-files", type=int, default=0)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    paderborn = manifest[
        (manifest["dataset"] == "paderborn")
        & (manifest["extension"] == "mat")
    ].copy()

    if args.max_files > 0:
        paderborn = paderborn.head(args.max_files)

    all_rows: list[dict[str, Any]] = []

    for i, row in enumerate(paderborn.itertuples(index=False), start=1):
        rows = extract_channels(
            path=Path(row.absolute_path),
            relative_path=row.relative_path,
            original_filename=row.original_filename,
        )
        all_rows.extend(rows)

        if i % 100 == 0:
            print(f"Inspected {i} / {len(paderborn)} files")

    out = pd.DataFrame(all_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Wrote: {args.output}")
    print(f"Files inspected: {len(paderborn)}")
    print(f"Channel rows: {len(out)}")

    print("\nRead errors:")
    print(out["read_error"].replace("", "OK").value_counts(dropna=False).head(20).to_string())

    print("\nInferred sensor type counts:")
    print(out["inferred_sensor_type"].value_counts(dropna=False).to_string())

    print("\nTop channel names:")
    print(out["channel_name"].value_counts(dropna=False).head(30).to_string())

    print("\nTop channel paths:")
    print(out["channel_path"].value_counts(dropna=False).head(30).to_string())


if __name__ == "__main__":
    main()
