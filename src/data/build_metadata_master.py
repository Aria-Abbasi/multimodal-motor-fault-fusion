"""Build complete recording-level metadata_master.csv.

Inputs:
    data/metadata/raw_manifest.csv
    data/metadata/paderborn_channel_schema.csv
    data/raw/nln_emp/Appendices/Other/measurement overview.xlsx

Output:
    data/metadata/metadata_master.csv

Rules:
    - one row per recording/modality for NLN-EMP
    - one row per raw CWRU MAT recording
    - one row per Paderborn MAT recording
    - no windowing
    - no preprocessing
    - preserve original labels
    - keep stable base_recording_id for leakage-safe splitting
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

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


def slug(text: Any) -> str:
    text = "" if pd.isna(text) else str(text)
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def clean_text(text: Any) -> str:
    if pd.isna(text):
        return ""
    return str(text).strip()


def normalize_fault_label(label: Any) -> tuple[str, str, str]:
    raw = clean_text(label).lower()
    clean = re.sub(r"\s+", " ", raw.replace("-", " ")).strip()

    if not clean:
        return "unknown", "unknown", "unknown"

    if any(x in clean for x in ["healthy", "normal", "new motor", "original motor"]):
        return "healthy", "0", "healthy"

    if "bpfi" in clean or "inner" in clean or clean.startswith("ir"):
        family = "bearing_inner_race"
    elif "bpfo" in clean or "outer" in clean or clean.startswith("or"):
        family = "bearing_outer_race"
    elif "bsf" in clean:
        family = "bearing_ball_spin"
    elif "bearing contaminated" in clean or "contaminated" in clean:
        family = "bearing_contamination"
    elif "rotor" in clean:
        family = "rotor_fault"
    elif "bent shaft" in clean:
        family = "bent_shaft"
    elif "stator" in clean or "winding" in clean:
        family = "stator_winding_fault"
    elif "impeller" in clean:
        family = "impeller_fault"
    elif "soft foot" in clean or "loose foot" in clean:
        family = "looseness_soft_foot"
    elif "align" in clean or "angular" in clean or "parallel" in clean or "combination" in clean:
        family = "misalignment"
    elif "unbalance" in clean:
        family = "unbalance"
    elif "cavitation" in clean:
        family = "cavitation"
    elif "coupling" in clean:
        family = "coupling_fault"
    elif "bearing" in clean or clean.startswith("b"):
        family = "bearing_ball"
    else:
        family = slug(clean)

    severity_match = re.search(r"(?:^|\s)(\d+)(?:\s|$)", clean)
    severity = severity_match.group(1) if severity_match else "unknown"

    return family, severity, "fault"


def read_nln_overview(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_excel(path, sheet_name="Ordered Measurements", header=1)
    df = df.rename(
        columns={
            "Order": "order",
            "Setup": "setup",
            "Failure description": "failure_description",
            "Severity": "severity_verified",
            "Speed (%)": "speed_percent_verified",
            "Speed (RPM) +- 5": "speed_rpm_verified",
            "Pressure (bar) +- 0.1": "pressure_bar_verified",
            "Flow (m3/h) +- 5": "flow_m3h_verified",
            "Alignment report": "alignment_report",
            "Comments": "comments",
        }
    )

    keep = [
        "order",
        "setup",
        "failure_description",
        "severity_verified",
        "speed_percent_verified",
        "speed_rpm_verified",
        "pressure_bar_verified",
        "flow_m3h_verified",
        "alignment_report",
        "comments",
    ]

    existing = [c for c in keep if c in df.columns]
    df = df[existing].copy()

    df = df[df["failure_description"].notna()]
    df["setup_norm"] = df["setup"].astype(str).str.strip().str.lower()
    df["label_norm"] = df["failure_description"].astype(str).map(slug)

    return df


def parse_nln_filename(filename: str) -> dict[str, str]:
    stem = Path(filename).stem

    pattern = re.compile(
        r"^(?P<sensor>Electric|Vibration)_"
        r"(?P<machine>Motor-\d+)_"
        r"(?P<speed>\d+)_"
        r"time-"
        r"(?P<label>.+?)"
        r"-ch(?P<channel>\d+)$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(stem)

    if not match:
        return {
            "machine_id": "unknown",
            "sensor_type": "unknown",
            "speed": "unknown",
            "original_label": stem,
            "channel": "unknown",
        }

    sensor_raw = match.group("sensor").lower()
    sensor_type = "current" if sensor_raw == "electric" else "vibration"

    return {
        "machine_id": match.group("machine"),
        "sensor_type": sensor_type,
        "speed": match.group("speed"),
        "original_label": match.group("label").strip(),
        "channel": match.group("channel"),
    }


def find_nln_verified_row(parsed: dict[str, str], overview: pd.DataFrame) -> dict[str, str]:
    if overview.empty:
        return {}

    machine_norm = parsed["machine_id"].replace("-", " ").lower()
    label_norm = slug(parsed["original_label"])

    candidates = overview[overview["setup_norm"] == machine_norm].copy()
    if candidates.empty:
        return {}

    candidates["score"] = candidates["label_norm"].map(
        lambda x: 2 if x == label_norm else (1 if x in label_norm or label_norm in x else 0)
    )
    candidates = candidates.sort_values("score", ascending=False)

    if candidates.iloc[0]["score"] <= 0:
        return {}

    row = candidates.iloc[0].to_dict()
    return {k: clean_text(v) for k, v in row.items()}


def build_nln_metadata(raw: pd.DataFrame, overview: pd.DataFrame) -> pd.DataFrame:
    df = raw[
        (raw["dataset"] == "nln_emp")
        & (raw["extension"] == "csv")
        & (raw["relative_path"].str.contains("/Dataset/", regex=False))
    ].copy()

    records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        parsed = parse_nln_filename(row["original_filename"])
        verified = find_nln_verified_row(parsed, overview)

        original_label = parsed["original_label"]
        fault_family, severity_from_name, health_label = normalize_fault_label(original_label)

        severity = clean_text(verified.get("severity_verified", "")) or severity_from_name
        if health_label == "healthy":
            severity = "0"

        speed_rpm = clean_text(verified.get("speed_rpm_verified", ""))
        pressure_bar = clean_text(verified.get("pressure_bar_verified", ""))
        flow_m3h = clean_text(verified.get("flow_m3h_verified", ""))

        base_recording_id = slug(
            f"nln_emp_{parsed['machine_id']}_{parsed['speed']}_{original_label}"
        )
        recording_id = slug(f"{base_recording_id}_{parsed['sensor_type']}")

        records.append(
            {
                "dataset": "nln_emp",
                "machine_id": parsed["machine_id"],
                "recording_id": recording_id,
                "base_recording_id": base_recording_id,
                "sensor_type": parsed["sensor_type"],
                "sensor_types_present": parsed["sensor_type"],
                "speed": parsed["speed"],
                "speed_rpm": speed_rpm or "unknown",
                "load": "unknown",
                "pressure_bar": pressure_bar or "unknown",
                "flow_m3h": flow_m3h or "unknown",
                "operating_condition_id": slug(
                    f"{parsed['machine_id']}_speed_{parsed['speed']}"
                ),
                "fault_family": fault_family,
                "severity": severity if severity else "unknown",
                "health_label": health_label,
                "original_label": original_label,
                "damage_source": "experimental",
                "run_id": "unknown",
                "source_path": row["relative_path"],
                "source_file": row["original_filename"],
                "mat_top_level_key": "",
                "mat_signal_keys": "",
                "mat_channel_names": "",
                "is_readable": True,
                "exclude_from_training": False,
                "notes": f"channel_{parsed['channel']};nln_overview_verified={bool(verified)}",
            }
        )

    parsed_df = pd.DataFrame(records)
    if parsed_df.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    group_cols = [
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
        "mat_top_level_key",
        "mat_signal_keys",
        "mat_channel_names",
        "is_readable",
        "exclude_from_training",
    ]

    grouped = (
        parsed_df.groupby(group_cols, dropna=False)
        .agg(
            source_path=("source_path", lambda x: "|".join(sorted(set(map(str, x))))),
            source_files=("source_file", lambda x: "|".join(sorted(set(map(str, x))))),
            n_source_files=("source_file", "count"),
            notes=("notes", lambda x: "|".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )

    return grouped[REQUIRED_COLUMNS]


def parse_cwru_filename(filename: str) -> dict[str, Any]:
    stem = Path(filename).stem

    if stem.lower().startswith("time_normal"):
        parts = stem.split("_")
        load = parts[2] if len(parts) >= 3 and parts[2].isdigit() else "unknown"
        return {
            "machine_id": "cwru_bearing_rig",
            "base_recording_id": slug(f"cwru_{stem}"),
            "recording_id": slug(f"cwru_{stem}_vibration"),
            "sensor_type": "vibration",
            "sensor_types_present": "vibration",
            "speed": "unknown",
            "speed_rpm": "unknown",
            "load": load,
            "pressure_bar": "",
            "flow_m3h": "",
            "operating_condition_id": slug(f"load_{load}"),
            "fault_family": "healthy",
            "severity": "0",
            "health_label": "healthy",
            "original_label": "normal",
            "damage_source": "healthy",
            "run_id": parts[-1] if parts else "unknown",
            "notes": "cwru_raw_mat",
        }

    pattern = re.compile(
        r"^(?P<fault>B|IR|OR)(?P<size>\d+)_(?:(?P<or_position>\d+)_)?(?P<load>\d+)_(?P<run>\d+)$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(stem)

    if not match:
        return {
            "machine_id": "cwru_bearing_rig",
            "base_recording_id": slug(f"cwru_{stem}"),
            "recording_id": slug(f"cwru_{stem}_vibration"),
            "sensor_type": "vibration",
            "sensor_types_present": "vibration",
            "speed": "unknown",
            "speed_rpm": "unknown",
            "load": "unknown",
            "pressure_bar": "",
            "flow_m3h": "",
            "operating_condition_id": "unknown",
            "fault_family": "unknown",
            "severity": "unknown",
            "health_label": "unknown",
            "original_label": stem,
            "damage_source": "unknown",
            "run_id": "unknown",
            "notes": "cwru_parse_failed",
        }

    fault = match.group("fault").upper()
    size = match.group("size")
    load = match.group("load")
    run = match.group("run")
    or_position = match.group("or_position") or "none"

    fault_family = {
        "B": "bearing_ball",
        "IR": "bearing_inner_race",
        "OR": "bearing_outer_race",
    }[fault]

    original_label = f"{fault}{size}"
    if fault == "OR":
        original_label += f"_position_{or_position}"

    return {
        "machine_id": "cwru_bearing_rig",
        "base_recording_id": slug(f"cwru_{stem}"),
        "recording_id": slug(f"cwru_{stem}_vibration"),
        "sensor_type": "vibration",
        "sensor_types_present": "vibration",
        "speed": "unknown",
        "speed_rpm": "unknown",
        "load": load,
        "pressure_bar": "",
        "flow_m3h": "",
        "operating_condition_id": slug(f"load_{load}"),
        "fault_family": fault_family,
        "severity": size,
        "health_label": "fault",
        "original_label": original_label,
        "damage_source": "seeded_fault",
        "run_id": run,
        "notes": f"cwru_raw_mat;or_position_{or_position}",
    }


def build_cwru_metadata(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[
        (raw["dataset"] == "cwru")
        & (raw["extension"] == "mat")
        & (raw["relative_path"].str.contains("/raw/", regex=False))
    ].copy()

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        parsed = parse_cwru_filename(row["original_filename"])
        rows.append(
            {
                "dataset": "cwru",
                **parsed,
                "source_path": row["relative_path"],
                "source_files": row["original_filename"],
                "n_source_files": 1,
                "mat_top_level_key": "",
                "mat_signal_keys": "",
                "mat_channel_names": "",
                "is_readable": True,
                "exclude_from_training": False,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return out[REQUIRED_COLUMNS]


def parse_paderborn_filename(filename: str, relative_path: str) -> dict[str, Any]:
    stem = Path(filename).stem

    pattern = re.compile(
        r"^N(?P<n>\d+)_M(?P<m>\d+)_F(?P<f>\d+)_(?P<bearing>[A-Z]+\d+)_(?P<run>\d+)$",
        flags=re.IGNORECASE,
    )
    match = pattern.match(stem)

    folder_bearing = Path(relative_path).parent.name.upper()

    if match:
        n = match.group("n")
        m = match.group("m")
        f = match.group("f")
        bearing = match.group("bearing").upper()
        run = match.group("run")
    else:
        n, m, f = "unknown", "unknown", "unknown"
        bearing = folder_bearing
        run = "unknown"

    if re.fullmatch(r"K\d+", bearing):
        fault_family = "healthy"
        severity = "0"
        health_label = "healthy"
        damage_source = "healthy"
    elif bearing.startswith("KA"):
        fault_family = "bearing_artificial_damage"
        severity = re.sub(r"[^0-9]", "", bearing) or "unknown"
        health_label = "fault"
        damage_source = "artificial"
    elif bearing.startswith("KI"):
        fault_family = "bearing_real_damage"
        severity = re.sub(r"[^0-9]", "", bearing) or "unknown"
        health_label = "fault"
        damage_source = "real"
    elif bearing.startswith("KB"):
        fault_family = "bearing_real_damage"
        severity = re.sub(r"[^0-9]", "", bearing) or "unknown"
        health_label = "fault"
        damage_source = "real"
    else:
        fault_family = "bearing_unknown"
        severity = re.sub(r"[^0-9]", "", bearing) or "unknown"
        health_label = "unknown"
        damage_source = "unknown"

    return {
        "machine_id": bearing,
        "base_recording_id": slug(f"paderborn_{stem}"),
        "recording_id": slug(f"paderborn_{stem}_multisensor"),
        "speed": f"N{n}" if n != "unknown" else "unknown",
        "speed_rpm": "unknown",
        "load": f"M{m}_F{f}" if m != "unknown" and f != "unknown" else "unknown",
        "pressure_bar": "",
        "flow_m3h": "",
        "operating_condition_id": slug(f"N{n}_M{m}_F{f}"),
        "fault_family": fault_family,
        "severity": severity,
        "health_label": health_label,
        "original_label": bearing,
        "damage_source": damage_source,
        "run_id": run,
    }


def sensor_type_from_paderborn_channels(channels: pd.DataFrame) -> str:
    if channels.empty:
        return "unknown"

    names = set(channels["channel_name"].fillna("").astype(str).str.lower())
    inferred = set(channels["inferred_sensor_type"].fillna("").astype(str).str.lower())

    sensors: set[str] = set()

    if "vibration_1" in names or "vibration" in inferred:
        sensors.add("vibration")
    if "phase_current_1" in names or "phase_current_2" in names or "current" in inferred:
        sensors.add("current")
    if "speed" in names or "speed" in inferred:
        sensors.add("speed")
    if "torque" in names or "torque_or_load" in inferred:
        sensors.add("torque")
    if "force" in names or "force" in inferred:
        sensors.add("force")
    if "temp_2_bearing_module" in names or "temperature" in inferred:
        sensors.add("temperature")

    return "|".join(sorted(sensors)) if sensors else "unknown"


def build_paderborn_metadata(raw: pd.DataFrame, channel_schema: pd.DataFrame) -> pd.DataFrame:
    df = raw[(raw["dataset"] == "paderborn") & (raw["extension"] == "mat")].copy()

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        parsed = parse_paderborn_filename(row["original_filename"], row["relative_path"])

        ch = channel_schema[channel_schema["relative_path"] == row["relative_path"]].copy()

        read_errors = sorted(
            set(
                str(x)
                for x in ch["read_error"].fillna("").tolist()
                if str(x).strip()
            )
        ) if not ch.empty and "read_error" in ch.columns else []

        is_readable = len(read_errors) == 0
        exclude_from_training = not is_readable

        readable_ch = ch[ch["read_error"].fillna("").astype(str).str.strip() == ""] if not ch.empty else pd.DataFrame()

        channel_name_set = set(readable_ch["channel_name"].fillna("").astype(str)) - {""}
        channel_names = "|".join(sorted(channel_name_set)) if not readable_ch.empty else ""

        inferred_sensors = sensor_type_from_paderborn_channels(readable_ch)
        top_keys = "|".join(sorted(set(readable_ch["top_level_key"].fillna("").astype(str)))) if not readable_ch.empty else ""

        signal_keys = "|".join(
            sorted(
                set(
                    f"{r.channel_index}:{r.channel_name}:{r.n_samples}"
                    for r in readable_ch.itertuples(index=False)
                )
            )
        ) if not readable_ch.empty else ""

        rows.append(
            {
                "dataset": "paderborn",
                "machine_id": parsed["machine_id"],
                "recording_id": parsed["recording_id"],
                "base_recording_id": parsed["base_recording_id"],
                "sensor_type": "multisensor" if inferred_sensors != "unknown" else "unknown",
                "sensor_types_present": inferred_sensors,
                "speed": parsed["speed"],
                "speed_rpm": parsed["speed_rpm"],
                "load": parsed["load"],
                "pressure_bar": parsed["pressure_bar"],
                "flow_m3h": parsed["flow_m3h"],
                "operating_condition_id": parsed["operating_condition_id"],
                "fault_family": parsed["fault_family"],
                "severity": parsed["severity"],
                "health_label": parsed["health_label"],
                "original_label": parsed["original_label"],
                "damage_source": parsed["damage_source"],
                "run_id": parsed["run_id"],
                "source_path": row["relative_path"],
                "source_files": row["original_filename"],
                "n_source_files": 1,
                "mat_top_level_key": top_keys,
                "mat_signal_keys": signal_keys,
                "mat_channel_names": channel_names,
                "is_readable": is_readable,
                "exclude_from_training": exclude_from_training,
                "notes": ";".join(read_errors) if read_errors else "paderborn_channels_verified",
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    return out[REQUIRED_COLUMNS]


def validate_metadata(df: pd.DataFrame) -> None:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    duplicate_count = df["recording_id"].duplicated().sum()
    if duplicate_count:
        examples = df.loc[df["recording_id"].duplicated(), "recording_id"].head(10).tolist()
        raise ValueError(f"Duplicate recording_id values: {duplicate_count}; examples={examples}")

    must_not_be_empty = [
        "dataset",
        "machine_id",
        "recording_id",
        "base_recording_id",
        "sensor_type",
        "sensor_types_present",
        "operating_condition_id",
        "fault_family",
        "health_label",
        "source_path",
    ]

    for col in must_not_be_empty:
        empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if empty.any():
            raise ValueError(f"Column {col} has {empty.sum()} empty values")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build complete metadata_master.csv.")
    parser.add_argument("--manifest", default="data/metadata/raw_manifest.csv")
    parser.add_argument("--paderborn-channels", default="data/metadata/paderborn_channel_schema.csv")
    parser.add_argument("--nln-overview", default="data/raw/nln_emp/Appendices/Other/measurement overview.xlsx")
    parser.add_argument("--output", default="data/metadata/metadata_master.csv")
    args = parser.parse_args()

    raw = pd.read_csv(args.manifest)

    paderborn_channels_path = Path(args.paderborn_channels)
    if not paderborn_channels_path.exists():
        raise FileNotFoundError(
            f"Missing {paderborn_channels_path}. Run scripts/inspect_paderborn_channels.py first."
        )

    paderborn_channels = pd.read_csv(paderborn_channels_path)
    nln_overview = read_nln_overview(Path(args.nln_overview))

    metadata = pd.concat(
        [
            build_nln_metadata(raw, nln_overview),
            build_cwru_metadata(raw),
            build_paderborn_metadata(raw, paderborn_channels),
        ],
        ignore_index=True,
    )

    metadata = metadata[REQUIRED_COLUMNS].sort_values(
        ["dataset", "machine_id", "operating_condition_id", "recording_id"]
    )

    validate_metadata(metadata)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(output_path, index=False)

    print(f"Wrote: {output_path}")
    print(f"Rows: {len(metadata)}")

    print("\nDataset counts:")
    print(metadata["dataset"].value_counts().to_string())

    print("\nHealth label counts:")
    print(metadata["health_label"].value_counts().to_string())

    print("\nSensor types present:")
    print(metadata["sensor_types_present"].value_counts().head(20).to_string())

    print("\nReadable counts:")
    print(metadata["is_readable"].value_counts(dropna=False).to_string())

    print("\nExcluded from training:")
    print(metadata["exclude_from_training"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
