"""Validation-only pilot selection and frozen configuration safeguards."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.training.experiment_runner import build_experiment_matrix
from src.training.paper_experiment_runner import resolve_frozen_configuration
from src.training.pilot_selection import (
    select_configuration,
    summarize_pilot,
)
from src.training.train_multimodal import PIPELINE_VERSION


def build_pilot_results() -> pd.DataFrame:
    rows = []
    for experiment_index, experiment in enumerate(build_experiment_matrix()):
        for fold_index in range(4):
            rows.append(
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "status": "COMPLETED",
                    "protocol": "nln_emp",
                    "fold_id": f"fold_{fold_index}",
                    "experiment": experiment["experiment"],
                    "seed": 42,
                    "loss_name": experiment["loss_name"],
                    "modality_gate": experiment["use_modality_gate"],
                    "validation_recording_macro_f1": (
                        0.70 + experiment_index / 100 + fold_index / 1000
                    ),
                    "validation_recording_early_fault_recall": (
                        0.96 if experiment_index else 0.90
                    ),
                    "validation_recording_fault_precision": (
                        0.60 + experiment_index / 100
                    ),
                    "validation_recording_mcc": (
                        0.50 + experiment_index / 100
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_pilot_selection_uses_complete_validation_matrix() -> None:
    summary = summarize_pilot(
        build_pilot_results(), expected_seed=42, expected_folds=4
    )
    selected = select_configuration(summary, minimum_early_recall=0.95)

    assert len(summary) == 12
    assert selected["experiment"] == summary.sort_values(
        "validation_recording_macro_f1_mean"
    ).iloc[-1]["experiment"]


def test_pilot_selection_rejects_missing_fold() -> None:
    results = build_pilot_results().iloc[:-1]

    with pytest.raises(ValueError, match="must contain 4 folds"):
        summarize_pilot(results, expected_seed=42, expected_folds=4)


def test_final_runner_requires_versioned_frozen_config(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "frozen.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "pipeline_version": PIPELINE_VERSION,
                "frozen_loss": "ce_2.0",
                "frozen_gate": False,
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        frozen_config=str(config_path),
        frozen_loss=None,
        frozen_gate=None,
        allow_explicit_frozen_config=False,
    )

    assert resolve_frozen_configuration(args) == (
        "ce_2.0",
        False,
        str(config_path),
    )


def test_final_runner_rejects_missing_frozen_config(tmp_path: Path) -> None:
    args = argparse.Namespace(
        frozen_config=str(tmp_path / "missing.yaml"),
        frozen_loss=None,
        frozen_gate=None,
        allow_explicit_frozen_config=False,
    )

    with pytest.raises(FileNotFoundError, match="pilot selection"):
        resolve_frozen_configuration(args)
