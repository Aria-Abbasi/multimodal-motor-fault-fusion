"""Build deterministic, leakage-safe multimodal spectrogram tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.signal
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from .preprocessing_config import (
    SPECTROGRAM_SIZE,
    STFT_NOVERLAP,
    STFT_NPERSEG,
    WINDOW_OVERLAP,
    WINDOW_SIZE,
)
from .signal_io import (
    NLN_DEFAULT_CURRENT_CHANNELS,
    NLN_DEFAULT_VIBRATION_CHANNEL,
    NLNSignalCache,
    load_recording_signals,
    nln_measurement_columns,
    select_nln_channel_paths,
)


CONSISTENCY_COLUMNS = (
    "fold_id",
    "split",
    "speed",
    "fault_family",
    "severity",
    "health_label",
)


def compute_stft_spectrogram(
    signal: np.ndarray, target_size: tuple[int, int] = SPECTROGRAM_SIZE
) -> torch.Tensor:
    """Compute a resized log-magnitude STFT."""
    _, _, zxx = scipy.signal.stft(
        signal,
        window="hann",
        nperseg=STFT_NPERSEG,
        noverlap=STFT_NOVERLAP,
    )
    log_spec = np.log(np.abs(zxx) + 1e-8)
    tensor = torch.from_numpy(log_spec).unsqueeze(0).unsqueeze(0).float()
    return F.interpolate(
        tensor, size=target_size, mode="bilinear", align_corners=False
    ).squeeze(0).squeeze(0)


def compute_phase_averaged_spectrogram(
    phase_signals: np.ndarray,
    target_size: tuple[int, int] = SPECTROGRAM_SIZE,
) -> torch.Tensor:
    """Average phase log-magnitude spectrograms without cancelling AC phases."""
    phases = np.atleast_2d(phase_signals)
    return torch.stack(
        [compute_stft_spectrogram(phase, target_size) for phase in phases]
    ).mean(dim=0)


def _assert_consistent(group: pd.DataFrame, base_id: str) -> None:
    for column in CONSISTENCY_COLUMNS:
        if column in group and group[column].dropna().astype(str).nunique() > 1:
            raise ValueError(
                f"Inconsistent {column} values for NLN base recording {base_id}"
            )


def pair_nln_rows(
    dataframe: pd.DataFrame,
    vibration_channel: int = NLN_DEFAULT_VIBRATION_CHANNEL,
    current_channels: tuple[int, ...] = NLN_DEFAULT_CURRENT_CHANNELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse separate NLN modality rows into weakly aligned condition pairs."""
    paired: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for base_id, group in dataframe.groupby("base_recording_id", sort=True):
        _assert_consistent(group, str(base_id))
        vibration_rows = group[group["sensor_type"].astype(str) == "vibration"]
        current_rows = group[group["sensor_type"].astype(str) == "current"]
        if len(vibration_rows) != 1 or len(current_rows) != 1:
            exclusions.append(
                {
                    "base_recording_id": base_id,
                    "reason": "missing_or_duplicate_modality",
                    "vibration_rows": len(vibration_rows),
                    "current_rows": len(current_rows),
                }
            )
            continue

        vibration_row = vibration_rows.iloc[0]
        current_row = current_rows.iloc[0]
        vibration_paths = select_nln_channel_paths(
            vibration_row["source_path"], (vibration_channel,)
        )
        current_paths = select_nln_channel_paths(
            current_row["source_path"], current_channels
        )
        columns = nln_measurement_columns(vibration_paths + current_paths)
        for measurement_column in columns:
            row = vibration_row.to_dict()
            row.update(
                {
                    "recording_id": (
                        f"{base_id}_measurement_{measurement_column}"
                    ),
                    "sensor_type": "multimodal",
                    "sensor_types_present": "current|vibration",
                    "vibration_source_path": str(vibration_row["source_path"]),
                    "current_source_path": str(current_row["source_path"]),
                    "measurement_column": str(measurement_column),
                    "nln_vibration_channel": vibration_channel,
                    "nln_current_channels": "|".join(map(str, current_channels)),
                    "alignment_policy": "condition_level_weak_alignment",
                }
            )
            paired.append(row)
    return pd.DataFrame(paired), pd.DataFrame(exclusions)


def prepare_recording_rows(
    dataframe: pd.DataFrame,
    dataset: str,
    vibration_channel: int = NLN_DEFAULT_VIBRATION_CHANNEL,
    current_channels: tuple[int, ...] = NLN_DEFAULT_CURRENT_CHANNELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset == "nln_emp":
        return pair_nln_rows(
            dataframe,
            vibration_channel=vibration_channel,
            current_channels=current_channels,
        )
    prepared = dataframe.copy()
    prepared["alignment_policy"] = (
        "synchronized_channels" if dataset == "paderborn" else "vibration_only"
    )
    return prepared, pd.DataFrame()


def _window_count(length: int) -> int:
    step = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))
    return max(0, (length - WINDOW_SIZE) // step + 1)


def _iter_window_pairs(
    vibration: np.ndarray, current: np.ndarray
):
    step = int(WINDOW_SIZE * (1 - WINDOW_OVERLAP))
    current = np.atleast_2d(current)
    count = min(_window_count(len(vibration)), _window_count(current.shape[1]))
    for index in range(count):
        start = index * step
        end = start + WINDOW_SIZE
        yield index, vibration[start:end], current[:, start:end]


def _load_row(
    row: pd.Series,
    dataset: str,
    vibration_channel: int,
    current_channels: tuple[int, ...],
    nln_cache: NLNSignalCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return load_recording_signals(
        row.to_dict(),
        dataset,
        nln_vibration_channel=vibration_channel,
        nln_current_channels=current_channels,
        nln_cache=nln_cache,
    )


def build_fold_spectrograms(
    dataframe: pd.DataFrame,
    dataset: str,
    base_out_dir: Path,
    *,
    vibration_channel: int = NLN_DEFAULT_VIBRATION_CHANNEL,
    current_channels: tuple[int, ...] = NLN_DEFAULT_CURRENT_CHANNELS,
    skip_bad_recordings: bool = False,
    tensor_dtype: torch.dtype = torch.float16,
) -> None:
    """Build one fold and save all provenance needed to reproduce it."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print(
            "Warning: matplotlib is unavailable; tensor generation will continue "
            "without QC plots."
        )

    tensor_dir = base_out_dir / "tensors"
    qc_dir = base_out_dir / "qc_plots"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    fold_id = (
        str(dataframe["fold_id"].iloc[0])
        if "fold_id" in dataframe and len(dataframe)
        else "single_fold"
    )
    rows, exclusions = prepare_recording_rows(
        dataframe,
        dataset,
        vibration_channel=vibration_channel,
        current_channels=current_channels,
    )
    if rows.empty:
        raise ValueError(f"No usable paired recordings for {dataset}/{fold_id}")
    provenance_columns = [
        column
        for column in (
            "recording_id",
            "base_recording_id",
            "split",
            "fold_id",
            "vibration_source_path",
            "current_source_path",
            "source_path",
            "measurement_column",
            "nln_vibration_channel",
            "nln_current_channels",
            "alignment_policy",
        )
        if column in rows.columns
    ]
    rows[provenance_columns].to_csv(
        base_out_dir / "paired_recordings.csv", index=False
    )

    errors: list[dict[str, Any]] = []

    def handle_error(row: pd.Series, phase: str, error: Exception) -> None:
        errors.append(
            {
                "fold_id": fold_id,
                "recording_id": row.get("recording_id", ""),
                "base_recording_id": row.get("base_recording_id", ""),
                "phase": phase,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        if not skip_bad_recordings:
            raise error

    train = rows[rows["split"].astype(str).str.lower() == "train"]
    if train.empty:
        raise ValueError("No training recordings are available")

    sums = {"vibration": 0.0, "current": 0.0}
    square_sums = {"vibration": 0.0, "current": 0.0}
    counts = {"vibration": 0, "current": 0}
    valid_train_recordings = 0
    nln_cache = NLNSignalCache() if dataset == "nln_emp" else None
    for _, row in tqdm(train.iterrows(), total=len(train), desc="Train statistics"):
        try:
            vibration, current = _load_row(
                row,
                dataset,
                vibration_channel,
                current_channels,
                nln_cache,
            )
            windows = list(_iter_window_pairs(vibration, current))
            if not windows:
                raise ValueError("Recording is shorter than one complete window")
            valid_train_recordings += 1
            for _, vibration_window, current_window in windows:
                sums["vibration"] += float(vibration_window.sum(dtype=np.float64))
                square_sums["vibration"] += float(
                    np.square(vibration_window, dtype=np.float64).sum()
                )
                counts["vibration"] += vibration_window.size
                sums["current"] += float(current_window.sum(dtype=np.float64))
                square_sums["current"] += float(
                    np.square(current_window, dtype=np.float64).sum()
                )
                counts["current"] += current_window.size
        except Exception as error:
            handle_error(row, "train_statistics", error)

    if not all(counts.values()):
        raise ValueError("No valid training samples were available for normalization")

    means = {key: sums[key] / counts[key] for key in sums}
    stds = {
        key: max(
            (
                square_sums[key] / counts[key] - means[key] ** 2
            ) ** 0.5,
            1e-6,
        )
        for key in sums
    }
    stats = {
        "dataset": dataset,
        "fold_id": fold_id,
        "train_only": True,
        "valid_train_recordings": valid_train_recordings,
        "counts": counts,
        "mean": means,
        "std": stds,
    }
    (base_out_dir / "normalization_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )

    index_rows: list[dict[str, Any]] = []
    global_index = 0
    nln_cache = NLNSignalCache() if dataset == "nln_emp" else None
    for _, row in tqdm(rows.iterrows(), total=len(rows), desc="Spectrograms"):
        try:
            vibration, current = _load_row(
                row,
                dataset,
                vibration_channel,
                current_channels,
                nln_cache,
            )
            pairs = _iter_window_pairs(vibration, current)
            saved_for_recording = 0
            for window_index, vibration_window, current_window in pairs:
                vibration_normalized = (
                    vibration_window - means["vibration"]
                ) / stds["vibration"]
                current_normalized = (
                    current_window - means["current"]
                ) / stds["current"]
                vibration_spec = compute_stft_spectrogram(vibration_normalized)
                current_spec = compute_phase_averaged_spectrogram(current_normalized)
                tensor = torch.stack([vibration_spec, current_spec]).to(tensor_dtype)
                tensor_id = f"{row['recording_id']}_w{window_index}.pt"
                torch.save(tensor, tensor_dir / tensor_id)
                index_rows.append(
                    {
                        "tensor_id": tensor_id,
                        "recording_id": row["recording_id"],
                        "base_recording_id": row.get(
                            "base_recording_id", row["recording_id"]
                        ),
                        "split": row["split"],
                        "dataset": dataset,
                        "fold_id": row.get("fold_id", fold_id),
                        "fault_family": row.get("fault_family", "unknown"),
                        "severity": row.get("severity", "unknown"),
                        "damage_source": row.get("damage_source", "unknown"),
                        "health_label": row.get("health_label", "unknown"),
                        "window_index": window_index,
                    }
                )
                if plt is not None and global_index % 10000 == 0:
                    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].imshow(vibration_spec.numpy(), origin="lower")
                    axes[1].imshow(current_spec.numpy(), origin="lower")
                    axes[0].set_title("Vibration")
                    axes[1].set_title("Phase-averaged current")
                    figure.tight_layout()
                    figure.savefig(qc_dir / f"qc_plot_{global_index}.png")
                    plt.close(figure)
                global_index += 1
                saved_for_recording += 1
            if not saved_for_recording:
                raise ValueError("Recording is shorter than one complete window")
        except Exception as error:
            handle_error(row, "spectrogram_generation", error)

    if not index_rows:
        raise ValueError("Preprocessing produced zero tensors")
    pd.DataFrame(index_rows).to_csv(base_out_dir / "windows_index.csv", index=False)
    if not exclusions.empty:
        exclusions.to_csv(base_out_dir / "preprocessing_exclusions.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(
            base_out_dir / "preprocessing_errors.csv", index=False
        )

    manifest = {
        "dataset": dataset,
        "fold_id": fold_id,
        "alignment_policy": {
            "nln_emp": "condition_level_weak_alignment",
            "paderborn": "synchronized_channels",
            "cwru": "vibration_only",
        }[dataset],
        "window_size": WINDOW_SIZE,
        "window_overlap": WINDOW_OVERLAP,
        "stft_nperseg": STFT_NPERSEG,
        "stft_noverlap": STFT_NOVERLAP,
        "spectrogram_size": list(SPECTROGRAM_SIZE),
        "tensor_dtype": str(tensor_dtype).replace("torch.", ""),
        "nln_vibration_channel": vibration_channel if dataset == "nln_emp" else None,
        "nln_current_channels": (
            list(current_channels) if dataset == "nln_emp" else None
        ),
        "current_representation": (
            "not_applicable"
            if dataset == "cwru"
            else "mean_of_phase_log_magnitude_spectrograms"
        ),
        "recordings_input": len(dataframe),
        "paired_measurements": len(rows),
        "excluded_base_recordings": len(exclusions),
        "saved_windows": len(index_rows),
        "qc_plots_enabled": plt is not None,
    }
    (base_out_dir / "preprocessing_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def select_fold_frames(
    dataframe: pd.DataFrame, fold_id: str | None, all_folds: bool
) -> list[tuple[str, pd.DataFrame]]:
    if fold_id and all_folds:
        raise ValueError("--fold-id and --all-folds cannot be used together")
    if "fold_id" not in dataframe:
        return [("single_fold", dataframe.copy())]
    fold_ids = sorted(dataframe["fold_id"].dropna().astype(str).unique())
    if fold_id:
        if fold_id not in fold_ids:
            raise ValueError(f"Unknown fold {fold_id}; available={fold_ids}")
        return [(fold_id, dataframe[dataframe["fold_id"].astype(str) == fold_id])]
    if all_folds:
        return [
            (value, dataframe[dataframe["fold_id"].astype(str) == value].copy())
            for value in fold_ids
        ]
    if len(fold_ids) > 1:
        raise ValueError(
            "Split file contains multiple folds. Pass --all-folds or --fold-id."
        )
    return [(fold_ids[0] if fold_ids else "single_fold", dataframe.copy())]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split-file", "--split_file", dest="split_file", required=True)
    parser.add_argument("--dataset", required=True, choices=("nln_emp", "paderborn", "cwru"))
    parser.add_argument("--fold-id")
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument(
        "--nln-vibration-channel",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--nln-current-channels",
        nargs="+",
        type=int,
        default=list(NLN_DEFAULT_CURRENT_CHANNELS),
    )
    parser.add_argument("--skip-bad-recordings", action="store_true")
    parser.add_argument(
        "--tensor-dtype", choices=("float16", "float32"), default="float16"
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    preprocessing_config = config.get("preprocessing", {})
    vibration_channel = args.nln_vibration_channel
    if vibration_channel is None:
        vibration_channel = preprocessing_config.get("nln_vibration_channel")
    if args.dataset == "nln_emp" and vibration_channel is None:
        raise ValueError(
            "NLN vibration channel is not frozen. Verify the sensor-location "
            "documentation, then pass --nln-vibration-channel or set it in config."
        )
    if vibration_channel is None:
        vibration_channel = NLN_DEFAULT_VIBRATION_CHANNEL
    dataframe = pd.read_csv(args.split_file)
    dataframe["dataset"] = dataframe.get("dataset", args.dataset)
    dataframe["split"] = dataframe["split"].astype(str).str.strip().str.lower()
    dataframe = dataframe[
        dataframe["dataset"].astype(str).str.lower() == args.dataset
    ].copy()
    folds = select_fold_frames(dataframe, args.fold_id, args.all_folds)
    protocol_dir = (
        Path(config["paths"]["processed"]) / args.dataset / Path(args.split_file).stem
    )
    use_subdirectories = len(folds) > 1 or dataframe.get(
        "fold_id", pd.Series(dtype=str)
    ).nunique() > 1
    for current_fold, fold_dataframe in folds:
        output_dir = (
            protocol_dir / current_fold if use_subdirectories else protocol_dir
        )
        build_fold_spectrograms(
            fold_dataframe,
            args.dataset,
            output_dir,
            vibration_channel=int(vibration_channel),
            current_channels=tuple(args.nln_current_channels),
            skip_bad_recordings=args.skip_bad_recordings,
            tensor_dtype=(
                torch.float16 if args.tensor_dtype == "float16" else torch.float32
            ),
        )


if __name__ == "__main__":
    main()
