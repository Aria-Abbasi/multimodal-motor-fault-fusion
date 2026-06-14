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
        is_early_fault("", health, "paderborn")
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
        is_early_fault(severity, health, "nln_emp")
        for severity, health in ((0, "healthy"), (1, "fault"), (1, "fault"), (2, "fault"))
    ]
    metrics = compute_binary_metrics(
        targets=[0, 1, 1, 1],
        predictions=[0, 1, 0, 1],
        early_fault_mask=early_mask,
    )

    assert metrics["early_fault_support"] == 2
    assert metrics["early_fault_recall"] == pytest.approx(0.5)


def test_fault_precision_tracks_false_positive_collapse() -> None:
    metrics = compute_binary_metrics(
        targets=[0, 0, 0, 1],
        predictions=[1, 1, 0, 1],
        fault_probabilities=[0.8, 0.7, 0.2, 0.9],
    )

    assert metrics["fault_precision"] == pytest.approx(1 / 3)
    assert metrics["balanced_acc"] != metrics["accuracy"]


def test_paderborn_bearing_ids_are_not_severity_labels() -> None:
    for bearing_id in ("01", "05", "07", "16", "30"):
        assert not is_early_fault(bearing_id, "fault", "paderborn")


def test_cwru_smallest_defect_is_early() -> None:
    assert is_early_fault("007", "fault", "cwru")
    assert is_early_fault("0.007", "fault", "cwru")
    assert not is_early_fault("014", "fault", "cwru")
