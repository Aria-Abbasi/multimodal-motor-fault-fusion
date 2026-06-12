"""Versioned, resumable result storage for corrected experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.training.train_multimodal import PIPELINE_VERSION


DEFAULT_RESULT_KEY = (
    "paper_experiment",
    "protocol",
    "fold_id",
    "model",
    "configuration",
    "seed",
    "label_budget",
)


def read_current_results(path: Path) -> pd.DataFrame:
    """Read corrected results and reject legacy schemas."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    dataframe = pd.read_csv(path)
    if "pipeline_version" not in dataframe.columns:
        raise ValueError(
            f"{path} is a legacy result file without pipeline_version"
        )
    invalid = dataframe[
        dataframe["pipeline_version"].astype(str) != PIPELINE_VERSION
    ]
    if len(invalid):
        raise ValueError(f"{path} mixes incompatible pipeline versions")
    return dataframe


def result_completed(
    path: Path,
    identity: dict[str, Any],
    key_columns: Iterable[str] = DEFAULT_RESULT_KEY,
) -> bool:
    dataframe = read_current_results(path)
    if dataframe.empty or "status" not in dataframe:
        return False
    match = pd.Series(True, index=dataframe.index)
    for column in key_columns:
        if column not in dataframe or column not in identity:
            return False
        match &= dataframe[column].astype(str) == str(identity[column])
    return bool((dataframe.loc[match, "status"] == "COMPLETED").any())


def bank_result(
    result: dict[str, Any],
    path: Path,
    key_columns: Iterable[str] = DEFAULT_RESULT_KEY,
) -> None:
    """Atomically insert or replace one versioned result row."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**result, "pipeline_version": PIPELINE_VERSION}
    existing = read_current_results(path)
    if not existing.empty:
        keep = pd.Series(True, index=existing.index)
        for column in key_columns:
            if column not in row:
                raise ValueError(f"Result is missing identity column {column}")
            if column not in existing:
                keep &= True
            else:
                keep &= existing[column].astype(str) == str(row[column])
        existing = existing[~keep]
    combined = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False, na_rep="N/A")
    temporary.replace(path)
