"""Regression tests for corrected modality construction and real attention."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.data.build_spectrograms import pair_nln_rows
from src.data.signal_io import NLNSignalCache, load_recording_signals
from src.evaluation.metrics import (
    aggregate_recording_predictions,
    select_decision_threshold,
)
from src.models.multimodal_cross_attention import MultimodalMotorModel


def _write_channel(path: Path, offset: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "time": np.arange(32),
            "0": np.arange(32, dtype=float) + offset,
            "1": np.arange(32, dtype=float) + offset + 100,
        }
    ).to_csv(path, index=False)


def test_nln_pairing_uses_separate_branches_and_phase_channels(
    tmp_path: Path,
) -> None:
    vibration_paths = []
    current_paths = []
    for channel in range(1, 6):
        path = tmp_path / "Vibration" / f"sample-ch{channel}.csv"
        _write_channel(path, channel * 10)
        vibration_paths.append(path)
    for channel in range(1, 7):
        path = tmp_path / "Electric" / f"sample-ch{channel}.csv"
        _write_channel(path, channel * 100)
        current_paths.append(path)

    common = {
        "dataset": "nln_emp",
        "base_recording_id": "condition_a",
        "fold_id": "fold_a",
        "split": "train",
        "speed": "50",
        "fault_family": "healthy",
        "severity": "0",
        "health_label": "healthy",
    }
    dataframe = pd.DataFrame(
        [
            {
                **common,
                "recording_id": "condition_a_vibration",
                "sensor_type": "vibration",
                "source_path": "|".join(map(str, vibration_paths)),
            },
            {
                **common,
                "recording_id": "condition_a_current",
                "sensor_type": "current",
                "source_path": "|".join(map(str, current_paths)),
            },
        ]
    )

    paired, exclusions = pair_nln_rows(dataframe)
    assert exclusions.empty
    assert set(paired["measurement_column"]) == {"0", "1"}
    vibration, current = load_recording_signals(
        paired.iloc[0].to_dict(), "nln_emp"
    )
    assert vibration.shape == (32,)
    assert current.shape == (3, 32)
    assert vibration[0] == pytest.approx(20)
    assert current[:, 0].tolist() == pytest.approx([100, 200, 300])


def test_nln_cache_reads_each_selected_csv_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vibration_paths = []
    current_paths = []
    for channel in range(1, 6):
        path = tmp_path / "Vibration" / f"sample-ch{channel}.csv"
        _write_channel(path, channel * 10)
        vibration_paths.append(path)
    for channel in range(1, 7):
        path = tmp_path / "Electric" / f"sample-ch{channel}.csv"
        _write_channel(path, channel * 100)
        current_paths.append(path)

    rows = [
        {
            "vibration_source_path": "|".join(map(str, vibration_paths)),
            "current_source_path": "|".join(map(str, current_paths)),
            "measurement_column": column,
        }
        for column in ("0", "1")
    ]
    original_read_csv = pd.read_csv
    reads: list[Path] = []

    def counted_read_csv(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        reads.append(Path(path))
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", counted_read_csv)
    cache = NLNSignalCache()
    for row in rows:
        load_recording_signals(row, "nln_emp", nln_cache=cache)

    assert len(reads) == 4
    assert set(reads) == {
        vibration_paths[1],
        current_paths[0],
        current_paths[1],
        current_paths[2],
    }


def test_attention_operates_on_multiple_spatial_tokens() -> None:
    model = MultimodalMotorModel(
        embed_dim=32,
        num_fault_families=3,
        num_attention_heads=4,
    ).eval()
    model(torch.randn(2, 2, 64, 64))
    shapes = model.fusion.blocks[0].last_attention_shapes
    assert shapes is not None
    assert shapes[0][-2:] == (16, 16)
    assert shapes[1][-2:] == (16, 16)


def test_threshold_and_recording_aggregation_are_validation_safe() -> None:
    threshold = select_decision_threshold(
        [0, 0, 1, 1], [0.1, 0.4, 0.45, 0.9]
    )
    assert 0.4 < threshold <= 0.45
    labels, probabilities, early = aggregate_recording_predictions(
        ["a", "a", "b", "b"],
        [0, 0, 1, 1],
        [0.1, 0.3, 0.7, 0.9],
        [False, False, True, True],
    )
    assert labels == [0, 1]
    assert probabilities == pytest.approx([0.2, 0.8])
    assert early == [False, True]
