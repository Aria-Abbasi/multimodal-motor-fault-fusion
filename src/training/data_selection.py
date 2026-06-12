"""Recording-level data selection shared by paper experiments."""

from __future__ import annotations

import hashlib

import pandas as pd


def recording_column(dataframe: pd.DataFrame) -> str:
    """Return the leakage-safe prediction and selection unit."""
    return (
        "base_recording_id"
        if "base_recording_id" in dataframe.columns
        else "recording_id"
    )


def select_label_budget(
    train_dataframe: pd.DataFrame,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Select a deterministic stratified fraction of complete recordings."""
    if not 0 < fraction <= 1:
        raise ValueError("label budget fraction must be in (0, 1]")
    if fraction == 1:
        return train_dataframe.copy()

    unit = recording_column(train_dataframe)
    recording_table = (
        train_dataframe[[unit, "health_label"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    if recording_table[unit].duplicated().any():
        raise ValueError("One recording has conflicting health labels")

    selected: set[str] = set()
    for _, group in recording_table.groupby("health_label", dropna=False):
        count = max(1, int(round(len(group) * fraction)))
        ranked = group.assign(
            _rank=group[unit].astype(str).map(
                lambda value: hashlib.sha256(
                    f"{seed}:{value}".encode("utf-8")
                ).hexdigest()
            )
        ).sort_values("_rank")
        selected.update(ranked.head(count)[unit].astype(str))
    return train_dataframe[
        train_dataframe[unit].astype(str).isin(selected)
    ].copy()
