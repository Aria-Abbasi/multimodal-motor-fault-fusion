"""Utilities for selecting valid runs without test-set tuning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.training.result_store import read_current_results


def select_validation_run(
    results_path: Path,
    *,
    paper_experiment: str | None = None,
    model: str = "proposed",
    dataset: str = "nln_emp",
    metric: str = "validation_recording_macro_f1",
) -> pd.Series:
    """Select a checkpoint using validation results only."""
    dataframe = read_current_results(results_path)
    if dataframe.empty:
        raise ValueError(f"No corrected results found in {results_path}")
    if metric not in dataframe:
        raise ValueError(f"Selection metric {metric} is missing")
    selected = dataframe[
        (dataframe["status"] == "COMPLETED")
        & (dataframe["model"].astype(str) == model)
        & (dataframe["dataset"].astype(str) == dataset)
    ].copy()
    if paper_experiment is not None:
        selected = selected[
            selected["paper_experiment"].astype(str) == paper_experiment
        ]
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    selected = selected.dropna(subset=[metric, "checkpoint_path", "processed_dir"])
    if selected.empty:
        raise ValueError("No validation-selectable corrected run was found")
    return selected.loc[selected[metric].idxmax()]
