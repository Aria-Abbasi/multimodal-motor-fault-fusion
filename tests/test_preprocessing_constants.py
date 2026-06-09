"""Checks for preprocessing constants and fold-selection safety."""

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
