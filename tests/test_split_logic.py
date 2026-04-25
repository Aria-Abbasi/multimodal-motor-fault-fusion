"""Tests for recording-level split leakage validation."""

from __future__ import annotations

import pytest

from src.data.split_validation import assert_no_recording_overlap


def test_no_recording_overlap_passes() -> None:
    assert_no_recording_overlap(
        train_recording_ids=["r1", "r2"],
        val_recording_ids=["r3"],
        test_recording_ids=["r4", "r5"],
    )


def test_recording_overlap_raises() -> None:
    with pytest.raises(ValueError):
        assert_no_recording_overlap(
            train_recording_ids=["r1", "r2"],
            val_recording_ids=["r2", "r3"],
            test_recording_ids=["r4"],
        )
