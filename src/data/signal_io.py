"""Dataset-specific raw signal loading with explicit modality selection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.io as sio


NLN_DEFAULT_VIBRATION_CHANNEL = 2
NLN_DEFAULT_CURRENT_CHANNELS = (1, 2, 3)
PADERBORN_VIBRATION_CHANNEL = "vibration_1"
PADERBORN_CURRENT_CHANNELS = ("phase_current_1", "phase_current_2")


@dataclass
class NLNSignalCache:
    """Cache all measurement columns for one NLN condition at a time."""

    _signature: tuple[str, ...] | None = None
    _channels: dict[Path, dict[str, np.ndarray]] = field(default_factory=dict)

    def load(
        self,
        row: Mapping[str, Any],
        vibration_channel: int,
        current_channels: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        vibration_paths = select_nln_channel_paths(
            row["vibration_source_path"], (vibration_channel,)
        )
        current_paths = select_nln_channel_paths(
            row["current_source_path"], tuple(current_channels)
        )
        paths = vibration_paths + current_paths
        signature = tuple(str(path) for path in paths)
        if signature != self._signature:
            self._channels = {
                path: _read_nln_measurements(path)
                for path in paths
            }
            self._signature = signature

        measurement_column = str(row["measurement_column"])
        missing = [
            str(path)
            for path in paths
            if measurement_column not in self._channels[path]
        ]
        if missing:
            raise ValueError(
                f"NLN measurement column {measurement_column!r} is missing "
                f"from {missing}"
            )
        vibration = self._channels[vibration_paths[0]][measurement_column]
        current = np.stack(
            [self._channels[path][measurement_column] for path in current_paths]
        )
        return vibration, current


def _path_list(value: Any) -> list[Path]:
    return [Path(item) for item in str(value).split("|") if item.strip()]


def _channel_number(path: Path) -> int | None:
    match = re.search(r"-ch(\d+)\.csv$", path.name, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def select_nln_channel_paths(
    source_paths: Any, channels: Sequence[int]
) -> list[Path]:
    """Select exact NLN channel files from a pipe-delimited metadata field."""
    by_channel = {
        channel: path
        for path in _path_list(source_paths)
        if (channel := _channel_number(path)) is not None
    }
    missing = [channel for channel in channels if channel not in by_channel]
    if missing:
        raise ValueError(
            f"Missing NLN channels {missing}; available={sorted(by_channel)}"
        )
    return [by_channel[channel] for channel in channels]


def nln_measurement_columns(paths: Sequence[Path]) -> tuple[str, ...]:
    """Return measurement columns shared by all selected NLN channel files."""
    if not paths:
        raise ValueError("At least one NLN channel path is required")
    shared: set[str] | None = None
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing NLN channel file: {path}")
        columns = {
            str(column)
            for column in pd.read_csv(path, nrows=0).columns
            if str(column).strip().lower()
            not in {"time", "timestamp", "index", "unnamed: 0"}
        }
        shared = columns if shared is None else shared & columns
    if not shared:
        raise ValueError(f"No shared measurement columns in {[str(p) for p in paths]}")

    def sort_key(value: str) -> tuple[int, float | str]:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value)

    return tuple(sorted(shared, key=sort_key))


def _read_nln_measurements(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NLN channel file: {path}")
    frame = pd.read_csv(path)
    measurement_columns = [
        str(column)
        for column in frame.columns
        if str(column).strip().lower()
        not in {"time", "timestamp", "index", "unnamed: 0"}
    ]
    if not measurement_columns:
        raise ValueError(f"No measurement columns found in {path}")
    numeric = frame[measurement_columns].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(numeric).all():
        raise ValueError(
            f"Non-numeric or missing measurement values in {path}"
        )
    return {
        column: numeric[:, index]
        for index, column in enumerate(measurement_columns)
    }


def _matlab_string(value: Any) -> str:
    array = np.asarray(value).squeeze()
    if array.size == 1:
        return str(array.item())
    return str(array)


def _load_paderborn_channels(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing Paderborn file: {path}")
    mat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    keys = [key for key in mat if not key.startswith("__")]
    if not keys:
        raise ValueError(f"No recording structure found in {path}")
    root = mat[path.stem] if path.stem in mat else mat[keys[0]]
    channels = np.atleast_1d(getattr(root, "Y", None))
    if channels.size == 0:
        raise ValueError(f"No Y channels found in {path}")

    loaded: dict[str, np.ndarray] = {}
    for channel in channels.flat:
        name = _matlab_string(getattr(channel, "Name", "")).strip()
        data = np.asarray(getattr(channel, "Data", [])).squeeze()
        if name and data.size:
            loaded[name.lower()] = data.astype(np.float32, copy=False).reshape(-1)
    return loaded


def load_recording_signals(
    source: Path | Mapping[str, Any],
    dataset_name: str,
    *,
    nln_vibration_channel: int = NLN_DEFAULT_VIBRATION_CHANNEL,
    nln_current_channels: Sequence[int] = NLN_DEFAULT_CURRENT_CHANNELS,
    nln_cache: NLNSignalCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load vibration and current.

    Current is returned as ``(n_phases, n_samples)``. NLN rows must already be
    paired and include ``vibration_source_path``, ``current_source_path`` and
    ``measurement_column``.
    """
    dataset_name = str(dataset_name).strip().lower()
    row: Mapping[str, Any] = source if isinstance(source, Mapping) else {}

    if dataset_name == "nln_emp":
        if not row:
            raise TypeError("NLN loading requires a paired metadata row")
        cache = nln_cache or NLNSignalCache()
        return cache.load(
            row,
            vibration_channel=nln_vibration_channel,
            current_channels=nln_current_channels,
        )

    path = Path(row.get("source_path", source))
    if dataset_name == "paderborn":
        channels = _load_paderborn_channels(path)
        vibration_key = PADERBORN_VIBRATION_CHANNEL.lower()
        missing = [
            name for name in PADERBORN_CURRENT_CHANNELS if name.lower() not in channels
        ]
        if vibration_key not in channels or missing:
            raise ValueError(
                f"Required Paderborn channels missing in {path}: "
                f"vibration={vibration_key in channels}, current_missing={missing}"
            )
        vibration = channels[vibration_key]
        current = np.stack(
            [channels[name.lower()] for name in PADERBORN_CURRENT_CHANNELS]
        )
        return vibration, current

    if dataset_name == "cwru":
        if not path.exists():
            raise FileNotFoundError(f"Missing CWRU file: {path}")
        mat = sio.loadmat(path)
        de_keys = [key for key in mat if "DE_time" in key]
        if not de_keys:
            raise ValueError(f"Could not find DE_time channel in {path}")
        vibration = np.asarray(mat[de_keys[0]], dtype=np.float32).reshape(-1)
        return vibration, np.zeros((1, len(vibration)), dtype=np.float32)

    raise ValueError(f"Unknown dataset: {dataset_name}")
