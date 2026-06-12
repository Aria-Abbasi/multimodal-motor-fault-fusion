"""Checks for preprocessing constants and fold-selection safety."""

import pandas as pd
import pytest

from src.data.build_spectrograms import select_fold_frames
from src.data.preprocessing_config import (
    SPECTROGRAM_SIZE,
    STFT_NOVERLAP,
    STFT_NPERSEG,
    WINDOW_OVERLAP,
    WINDOW_SIZE,
)
from src.training.train_multimodal import DEFAULT_MODALITY_DROPOUT


def test_preprocessing_constants_match_project_plan() -> None:
    assert WINDOW_SIZE in {2048, 4096}
    assert WINDOW_OVERLAP == 0.5
    assert STFT_NPERSEG == 256
    assert STFT_NOVERLAP == 128
    assert SPECTROGRAM_SIZE == (128, 128)
    assert DEFAULT_MODALITY_DROPOUT == 0.2


def test_all_folds_are_selected_without_mixing_rows() -> None:
    dataframe = pd.DataFrame(
        {
            "fold_id": ["test_speed_50", "test_speed_50", "test_speed_70"],
            "recording_id": ["a", "b", "c"],
        }
    )

    folds = select_fold_frames(dataframe, fold_id=None, all_folds=True)

    assert [fold_id for fold_id, _ in folds] == [
        "test_speed_50",
        "test_speed_70",
    ]
    assert [len(frame) for _, frame in folds] == [2, 1]
    assert all(frame["fold_id"].nunique() == 1 for _, frame in folds)


def test_multiple_folds_require_an_explicit_selection_mode() -> None:
    dataframe = pd.DataFrame(
        {
            "fold_id": ["test_speed_50", "test_speed_70"],
            "recording_id": ["a", "b"],
        }
    )

    with pytest.raises(ValueError, match="--all-folds"):
        select_fold_frames(dataframe, fold_id=None, all_folds=False)
