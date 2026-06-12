"""Binary fault-detection metrics and severity-label handling."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


MISSING_SEVERITY_TOKENS = {"", "nan", "none", "null", "unknown", "n/a", "na"}
EXPLICIT_EARLY_TOKENS = {"early", "early_fault", "lowest", "low"}
DATASET_EARLY_SEVERITY_TOKENS = {
    "nln_emp": {"1", "1.0"},
    "cwru": {"007", "0.007"},
}


def is_early_fault(
    severity: Any, health_label: Any, dataset: Any = None
) -> bool:
    """Return whether a row is an early fault under its dataset's label schema."""
    if "fault" not in str(health_label).strip().lower():
        return False

    token = str(severity).strip().lower()
    if token in MISSING_SEVERITY_TOKENS:
        return False
    if token in EXPLICIT_EARLY_TOKENS or "early" in token:
        return True

    dataset_name = str(dataset).strip().lower()
    return token in DATASET_EARLY_SEVERITY_TOKENS.get(dataset_name, set())


def severity_is_available(severity: Any, dataset: Any = None) -> bool:
    """Return whether a sample has a usable granular severity annotation."""
    if str(dataset).strip().lower() == "paderborn":
        return False
    return str(severity).strip().lower() not in MISSING_SEVERITY_TOKENS


def _safe_probability_metric(function: Any, targets: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(targets)) < 2:
        return float("nan")
    return float(function(targets, scores))


def compute_binary_metrics(
    targets: Iterable[int],
    predictions: Iterable[int],
    fault_probabilities: Optional[Iterable[float]] = None,
    early_fault_mask: Optional[Iterable[bool]] = None,
) -> dict[str, Any]:
    """Compute paper metrics, returning NaN when a metric is not identifiable."""
    y_true = np.asarray(list(targets), dtype=np.int64)
    y_pred = np.asarray(list(predictions), dtype=np.int64)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("targets and predictions must be non-empty and equally sized")

    scores = (
        np.asarray(list(fault_probabilities), dtype=np.float64)
        if fault_probabilities is not None
        else y_pred.astype(np.float64)
    )
    if scores.shape != y_true.shape:
        raise ValueError("fault_probabilities must match targets")

    if early_fault_mask is None:
        early_mask = np.zeros_like(y_true, dtype=bool)
    else:
        early_mask = np.asarray(list(early_fault_mask), dtype=bool)
        if early_mask.shape != y_true.shape:
            raise ValueError("early_fault_mask must match targets")
    early_mask &= y_true == 1

    early_count = int(early_mask.sum())
    early_recall = (
        float((y_pred[early_mask] == 1).mean())
        if early_count
        else float("nan")
    )

    has_both_classes = len(np.unique(y_true)) == 2
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": (
            float(balanced_accuracy_score(y_true, y_pred))
            if has_both_classes
            else float("nan")
        ),
        "accuracy": float((y_true == y_pred).mean()),
        "early_fault_recall": early_recall,
        "early_fault_support": early_count,
        "early_fault_recall_available": bool(early_count),
        "auroc": _safe_probability_metric(roc_auc_score, y_true, scores),
        "auprc": _safe_probability_metric(
            average_precision_score, y_true, scores
        ),
        "mcc": (
            float(matthews_corrcoef(y_true, y_pred))
            if has_both_classes
            else float("nan")
        ),
    }


def select_decision_threshold(
    targets: Iterable[int],
    fault_probabilities: Iterable[float],
) -> float:
    """Choose the validation threshold that maximizes Macro F1."""
    y_true = np.asarray(list(targets), dtype=np.int64)
    scores = np.asarray(list(fault_probabilities), dtype=np.float64)
    if not len(y_true) or y_true.shape != scores.shape:
        raise ValueError("targets and probabilities must be non-empty and aligned")
    if len(np.unique(y_true)) < 2:
        return 0.5
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    ranked = []
    for threshold in candidates:
        predictions = (scores >= threshold).astype(np.int64)
        macro_f1 = f1_score(
            y_true, predictions, average="macro", zero_division=0
        )
        ranked.append(
            (float(macro_f1), -abs(float(threshold) - 0.5), float(threshold))
        )
    return max(ranked)[2]


def aggregate_recording_predictions(
    recording_ids: Iterable[str],
    targets: Iterable[int],
    fault_probabilities: Iterable[float],
    early_fault_mask: Optional[Iterable[bool]] = None,
) -> tuple[list[int], list[float], list[bool]]:
    """Mean-pool window probabilities into one unit per recording."""
    identifiers = list(recording_ids)
    labels = list(targets)
    probabilities = list(fault_probabilities)
    early = (
        [False] * len(labels)
        if early_fault_mask is None
        else list(early_fault_mask)
    )
    lengths = {len(identifiers), len(labels), len(probabilities), len(early)}
    if lengths == {0} or len(lengths) != 1:
        raise ValueError("recording aggregation inputs must be non-empty and aligned")

    grouped: dict[str, list[int]] = {}
    for index, recording_id in enumerate(identifiers):
        grouped.setdefault(str(recording_id), []).append(index)

    output_labels: list[int] = []
    output_probabilities: list[float] = []
    output_early: list[bool] = []
    for recording_id in sorted(grouped):
        indices = grouped[recording_id]
        unique_labels = {int(labels[index]) for index in indices}
        if len(unique_labels) != 1:
            raise ValueError(f"Inconsistent labels within recording {recording_id}")
        output_labels.append(unique_labels.pop())
        output_probabilities.append(
            float(np.mean([probabilities[index] for index in indices]))
        )
        output_early.append(any(bool(early[index]) for index in indices))
    return output_labels, output_probabilities, output_early


def format_optional_metric(value: Any) -> str:
    """Format an unavailable numeric metric as N/A for logs and reports."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "N/A"
    return "N/A" if math.isnan(numeric) else f"{numeric:.4f}"
