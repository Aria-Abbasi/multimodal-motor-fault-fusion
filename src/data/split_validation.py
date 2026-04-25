"""Split validation helpers enforcing recording-level leakage safety."""

from __future__ import annotations

from typing import Iterable


def assert_no_recording_overlap(
    train_recording_ids: Iterable[str],
    val_recording_ids: Iterable[str],
    test_recording_ids: Iterable[str],
) -> None:
    """Validate that split recording IDs are mutually exclusive.

    Args:
        train_recording_ids: Recording IDs assigned to train split.
        val_recording_ids: Recording IDs assigned to validation split.
        test_recording_ids: Recording IDs assigned to test split.

    Raises:
        ValueError: If any overlap exists across train/val/test.
    """
    train_set = set(train_recording_ids)
    val_set = set(val_recording_ids)
    test_set = set(test_recording_ids)

    overlaps = {
        "train_val": train_set.intersection(val_set),
        "train_test": train_set.intersection(test_set),
        "val_test": val_set.intersection(test_set),
    }

    if any(overlaps.values()):
        raise ValueError(f"Recording-level leakage detected: {overlaps}")
