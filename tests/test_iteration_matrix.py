"""Regression tests for the master experiment matrix."""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")

from src.training.experiment_runner import build_experiment_matrix, build_parser


def test_matrix_contains_six_losses_with_and_without_gate() -> None:
    matrix = build_experiment_matrix()

    assert len(matrix) == 12
    assert len({row["experiment"] for row in matrix}) == 12
    assert sum(row["use_modality_gate"] for row in matrix) == 6
    assert {row["loss_name"] for row in matrix} == {
        "ce_1.0",
        "ce_1.5",
        "ce_2.0",
        "ce_3.0",
        "ce_4.0",
        "dynamic_focal",
    }


def test_runner_accepts_sequential_protocols() -> None:
    args = build_parser().parse_args(
        [
            "--protocols",
            "nln_emp",
            "paderborn_artificial_to_natural",
            "--cache-max-gb",
            "36",
        ]
    )

    assert args.protocols == [
        "nln_emp",
        "paderborn_artificial_to_natural",
    ]
    assert args.cache_max_gb == 36.0
