"""Freeze one loss/gate configuration using validation-only pilot metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.training.experiment_runner import build_experiment_matrix
from src.training.train_multimodal import PIPELINE_VERSION


REQUIRED_VALIDATION_METRICS = (
    "validation_recording_macro_f1",
    "validation_recording_early_fault_recall",
    "validation_recording_fault_precision",
    "validation_recording_mcc",
)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def summarize_pilot(
    results: pd.DataFrame,
    *,
    expected_seed: int,
    expected_folds: int = 4,
) -> pd.DataFrame:
    """Validate and summarize the complete 12-configuration NLN pilot."""
    required = {
        "pipeline_version",
        "status",
        "protocol",
        "fold_id",
        "experiment",
        "seed",
        "loss_name",
        "modality_gate",
        *REQUIRED_VALIDATION_METRICS,
    }
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Pilot results are missing columns: {sorted(missing)}")

    pilot = results[
        (results["protocol"].astype(str) == "nln_emp")
        & (results["seed"].astype(str) == str(expected_seed))
    ].copy()
    invalid_versions = pilot[
        pilot["pipeline_version"].astype(str) != PIPELINE_VERSION
    ]
    if len(invalid_versions):
        raise ValueError("Pilot results contain an incompatible pipeline version")
    if not len(pilot):
        raise ValueError("No NLN pilot rows were found")
    if pilot.duplicated(["experiment", "fold_id", "seed"]).any():
        raise ValueError("Pilot results contain duplicate configuration/fold/seed rows")
    if not (pilot["status"] == "COMPLETED").all():
        failed = pilot.loc[pilot["status"] != "COMPLETED", "experiment"].tolist()
        raise ValueError(f"Pilot contains incomplete rows: {failed}")

    expected_experiments = {
        row["experiment"] for row in build_experiment_matrix()
    }
    observed_experiments = set(pilot["experiment"].astype(str))
    if observed_experiments != expected_experiments:
        raise ValueError(
            "Pilot experiment set is incomplete: "
            f"missing={sorted(expected_experiments - observed_experiments)}, "
            f"unexpected={sorted(observed_experiments - expected_experiments)}"
        )

    counts = pilot.groupby("experiment")["fold_id"].nunique()
    bad_counts = counts[counts != expected_folds]
    if len(bad_counts):
        raise ValueError(
            f"Every pilot configuration must contain {expected_folds} folds: "
            f"{bad_counts.to_dict()}"
        )

    rows: list[dict[str, Any]] = []
    for experiment, group in pilot.groupby("experiment", sort=True):
        row: dict[str, Any] = {
            "experiment": experiment,
            "loss_name": str(group["loss_name"].iloc[0]),
            "modality_gate": _as_bool(group["modality_gate"].iloc[0]),
            "n_folds": int(group["fold_id"].nunique()),
            "seed": expected_seed,
        }
        for metric in REQUIRED_VALIDATION_METRICS:
            values = pd.to_numeric(group[metric], errors="coerce")
            if values.isna().any():
                raise ValueError(
                    f"{experiment} has unavailable validation metric {metric}"
                )
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_between_fold_std"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def select_configuration(
    summary: pd.DataFrame,
    *,
    minimum_early_recall: float,
) -> pd.Series:
    """Apply the predeclared validation-only selection rule."""
    if not 0 <= minimum_early_recall <= 1:
        raise ValueError("minimum_early_recall must be between 0 and 1")
    early_column = "validation_recording_early_fault_recall_mean"
    eligible = summary[summary[early_column] >= minimum_early_recall].copy()
    if eligible.empty:
        raise ValueError(
            "No pilot configuration meets the frozen minimum validation "
            f"early-fault recall of {minimum_early_recall:.4f}"
        )
    ranked = eligible.sort_values(
        [
            "validation_recording_macro_f1_mean",
            "validation_recording_fault_precision_mean",
            "validation_recording_mcc_mean",
            "validation_recording_macro_f1_between_fold_std",
            "experiment",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    )
    return ranked.iloc[0]


def freeze_pilot_configuration(
    results_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    expected_seed: int = 42,
    expected_folds: int = 4,
    minimum_early_recall: float = 0.95,
) -> dict[str, Any]:
    """Select and write a reproducible frozen configuration."""
    results_path = Path(results_path)
    results = pd.read_csv(results_path)
    summary = summarize_pilot(
        results,
        expected_seed=expected_seed,
        expected_folds=expected_folds,
    )
    selected = select_configuration(
        summary, minimum_early_recall=minimum_early_recall
    )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "code_revision": _git_revision(),
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_results": str(results_path),
        "source_results_sha256": _sha256(results_path),
        "expected_seed": expected_seed,
        "expected_folds": expected_folds,
        "minimum_validation_recording_early_fault_recall": minimum_early_recall,
        "selection_rule": [
            "Require mean validation recording early-fault recall at or above "
            "the frozen minimum",
            "Maximize mean validation recording Macro F1",
            "Tie-break by fault precision, MCC, lower between-fold Macro F1 "
            "standard deviation, then experiment name",
            "Never use test metrics for configuration selection",
        ],
        "selected_experiment": str(selected["experiment"]),
        "frozen_loss": str(selected["loss_name"]),
        "frozen_gate": bool(selected["modality_gate"]),
        "validation_summary": {
            key: (
                value.item() if hasattr(value, "item") else value
            )
            for key, value in selected.to_dict().items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default="results/tables/nln_validation_pilot.csv"
    )
    parser.add_argument(
        "--output", default="configs/frozen_l4_selection.yaml"
    )
    parser.add_argument(
        "--summary",
        default="results/tables/nln_validation_pilot_selection.csv",
    )
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--expected-folds", type=int, default=4)
    parser.add_argument(
        "--minimum-early-recall", type=float, default=0.95
    )
    args = parser.parse_args()
    payload = freeze_pilot_configuration(
        Path(args.results),
        Path(args.output),
        Path(args.summary),
        expected_seed=args.expected_seed,
        expected_folds=args.expected_folds,
        minimum_early_recall=args.minimum_early_recall,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
