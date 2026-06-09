"""Regression tests for early-fault metric availability."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("sklearn")

from src.evaluation.metrics import (
    compute_binary_metrics,
    format_optional_metric,
    is_early_fault,
)


def test_early_recall_is_na_when_paderborn_has_no_severity_labels() -> None:
    early_mask = [
        is_early_fault("", health)
        for health in ("healthy", "fault", "fault", "healthy")
    ]
    metrics = compute_binary_metrics(
        targets=[0, 1, 1, 0],
        predictions=[0, 1, 0, 1],
        fault_probabilities=[0.1, 0.8, 0.4, 0.7],
        early_fault_mask=early_mask,
    )

    assert metrics["early_fault_support"] == 0
    assert metrics["early_fault_recall_available"] is False
    assert math.isnan(metrics["early_fault_recall"])
    assert format_optional_metric(metrics["early_fault_recall"]) == "N/A"


def test_nln_severity_one_has_valid_early_recall() -> None:
    early_mask = [
        is_early_fault(severity, health)
        for severity, health in ((0, "healthy"), (1, "fault"), (1, "fault"), (2, "fault"))
    ]
    metrics = compute_binary_metrics(
        targets=[0, 1, 1, 1],
        predictions=[0, 1, 0, 1],
        early_fault_mask=early_mask,
    )

    assert metrics["early_fault_support"] == 2
    assert metrics["early_fault_recall"] == pytest.approx(0.5)
    assert metrics["balanced_acc"] != metrics["accuracy"]
