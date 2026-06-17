"""Regression tests for the master experiment matrix."""

from __future__ import annotations

import pytest
import argparse
from pathlib import Path

pytest.importorskip("pandas")
pd = pytest.importorskip("pandas")

from src.training.experiment_runner import (
    bank_result,
    build_experiment_matrix,
    build_parser,
    discover_processed_folds,
    is_completed,
    run_protocol_matrix,
    write_aggregate_summary,
)


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


def test_matrix_accepts_explicit_low_pressure_losses() -> None:
    matrix = build_experiment_matrix(
        ("ce_1.0", "ce_1.25", "ce_1.5", "dynamic_focal")
    )

    assert len(matrix) == 8
    assert {row["loss_name"] for row in matrix} == {
        "ce_1.0",
        "ce_1.25",
        "ce_1.5",
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


def test_runner_accepts_stage2_checkpoint_selection() -> None:
    args = build_parser().parse_args(
        ["--checkpoint-selection", "best_stage2"]
    )

    assert args.checkpoint_selection == "best_stage2"


def test_paderborn_rejects_early_weight_grid(tmp_path) -> None:
    args = argparse.Namespace(
        protocol="paderborn_artificial_to_natural",
        data_root=str(tmp_path),
        output_file=str(tmp_path / "results.csv"),
        seeds=[42],
        smoke_test=True,
        losses=["ce_2.0"],
    )

    with pytest.raises(ValueError, match="no granular severity labels"):
        run_protocol_matrix(args, "paderborn_artificial_to_natural")


def write_nln_split_file(root: Path) -> tuple[str, ...]:
    folds = (
        "test_speed_100",
        "test_speed_50",
        "test_speed_70",
        "test_speed_75",
    )
    root.mkdir(parents=True)
    pd.DataFrame({"fold_id": folds}).to_csv(
        root / "nln_emp_leave_one_speed_out.csv", index=False
    )
    return folds


def write_processed_fold(root: Path, fold_id: str) -> None:
    fold_dir = root / "nln_emp" / "nln_emp_leave_one_speed_out" / fold_id
    (fold_dir / "tensors").mkdir(parents=True)
    pd.DataFrame(
        {
            "tensor_id": ["sample.pt"],
            "split": ["train"],
            "fold_id": [fold_id],
        }
    ).to_csv(fold_dir / "windows_index.csv", index=False)


def test_fold_discovery_requires_complete_loso(tmp_path: Path) -> None:
    split_root = tmp_path / "splits"
    folds = write_nln_split_file(split_root)
    write_processed_fold(tmp_path / "processed", folds[0])

    with pytest.raises(FileNotFoundError, match="Missing processed folds"):
        discover_processed_folds(
            tmp_path / "processed",
            split_root,
            "nln_emp",
        )

    resolved = discover_processed_folds(
        tmp_path / "processed",
        split_root,
        "nln_emp",
        allow_partial=True,
    )
    assert [fold_id for fold_id, _ in resolved] == [folds[0]]


def test_result_identity_includes_fold_and_summary_averages_runs(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "results.csv"
    summary_path = tmp_path / "summary.csv"
    base = {
        "protocol": "nln_emp",
        "dataset": "nln_emp",
        "experiment": "fusion_gate_on_ce_1p0",
        "loss_name": "ce_1.0",
        "modality_gate": True,
        "seed": 42,
        "status": "COMPLETED",
        "balanced_acc": 0.75,
        "accuracy": 0.75,
        "early_fault_recall": 1.0,
        "auroc": 0.8,
        "auprc": 0.8,
        "mcc": 0.5,
        "current_gate_mean": 0.6,
    }
    bank_result(
        {**base, "fold_id": "test_speed_50", "macro_f1": 0.6},
        output_path,
    )
    bank_result(
        {**base, "fold_id": "test_speed_70", "macro_f1": 0.8},
        output_path,
    )

    assert is_completed(
        output_path,
        "nln_emp",
        "test_speed_50",
        "fusion_gate_on_ce_1p0",
        42,
    )
    assert len(pd.read_csv(output_path)) == 2

    summary = write_aggregate_summary(output_path, summary_path)
    assert summary.iloc[0]["n_folds"] == 2
    assert summary.iloc[0]["n_runs"] == 2
    assert summary.iloc[0]["macro_f1_mean"] == pytest.approx(0.7)
    assert summary.iloc[0]["macro_f1_std"] == pytest.approx(2**0.5 / 10)


def test_runner_executes_every_nln_fold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_root = tmp_path / "splits"
    folds = write_nln_split_file(split_root)
    for fold_id in folds:
        write_processed_fold(tmp_path / "processed", fold_id)

    def fake_train(args: argparse.Namespace) -> dict[str, object]:
        assert args.checkpoint_selection == "best_stage2"
        return {
            "run_id": args.run_id,
            "experiment": args.experiment,
            "dataset": args.dataset,
            "fold_id": args.fold_id,
            "seed": args.seed,
            "loss_name": args.loss_name,
            "modality_gate": args.use_modality_gate,
            "macro_f1": 0.75,
            "balanced_acc": 0.75,
            "accuracy": 0.75,
            "early_fault_recall": 1.0,
            "auroc": 0.8,
            "auprc": 0.8,
            "mcc": 0.5,
            "current_gate_mean": 0.6,
        }

    monkeypatch.setattr(
        "src.training.train_multimodal.train_multimodal",
        fake_train,
    )
    output_path = tmp_path / "results.csv"
    args = build_parser().parse_args(
        [
            "--protocol",
            "nln_emp",
            "--data-root",
            str(tmp_path / "processed"),
            "--split-root",
            str(split_root),
            "--losses",
            "ce_1.0",
            "--seeds",
            "42",
            "--smoke-test",
            "--no-preload",
            "--output-file",
            str(output_path),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
            "--checkpoint-selection",
            "best_stage2",
        ]
    )

    run_protocol_matrix(args, "nln_emp")

    results = pd.read_csv(output_path)
    summary = pd.read_csv(tmp_path / "results_summary.csv")
    assert len(results) == 8
    assert set(results["fold_id"]) == set(folds)
    assert len(summary) == 2
    assert set(summary["n_folds"]) == {4}
    assert set(summary["n_runs"]) == {4}
