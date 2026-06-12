"""Synthetic CPU verification for the complete paper experiment stack."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

torch = pytest.importorskip("torch")

from scripts.download_cwru_benchmark import (
    expected_cwru_files,
    validate_cwru_benchmark,
)
from src.evaluation.explainability import explain_sample
from src.evaluation.reporting import summarize_results
from src.models.classical_baselines import CLASSICAL_BASELINE_NAMES
from src.models.deep_baselines import DEEP_BASELINE_NAMES
from src.models.multimodal_cross_attention import MultimodalMotorModel
from src.training.baseline_runner import run_baseline
from src.training.data_selection import select_label_budget
from src.training.paper_experiment_runner import (
    PaperJob,
    build_paper_jobs,
    training_signature,
)
from src.training.train_multimodal import PIPELINE_VERSION


def build_processed_fold(root: Path) -> None:
    tensor_dir = root / "tensors"
    tensor_dir.mkdir(parents=True)
    generator = torch.Generator().manual_seed(20260612)
    rows = []
    sample_index = 0
    for split, recordings in (("train", 6), ("val", 4), ("test", 4)):
        for recording_index in range(recordings):
            is_fault = recording_index % 2 == 1
            recording_id = f"{split}_recording_{recording_index}"
            for window_index in range(2):
                tensor = torch.randn(2, 16, 16, generator=generator) * 0.1
                if is_fault:
                    tensor[0, 4:12, 4:12] += 1
                tensor_id = f"tensor_{sample_index}.pt"
                torch.save(tensor, tensor_dir / tensor_id)
                rows.append(
                    {
                        "tensor_id": tensor_id,
                        "recording_id": recording_id,
                        "base_recording_id": recording_id,
                        "split": split,
                        "health_label": "fault" if is_fault else "healthy",
                        "fault_family": "bearing" if is_fault else "healthy",
                        "severity": "1" if is_fault else "0",
                        "window_index": window_index,
                    }
                )
                sample_index += 1
    pd.DataFrame(rows).to_csv(root / "windows_index.csv", index=False)


@pytest.mark.parametrize(
    "model", CLASSICAL_BASELINE_NAMES + DEEP_BASELINE_NAMES
)
def test_every_baseline_runs_on_complete_synthetic_splits(
    tmp_path: Path, model: str
) -> None:
    processed = tmp_path / "processed"
    build_processed_fold(processed)
    result = run_baseline(
        argparse.Namespace(
            processed_dir=str(processed),
            dataset="nln_emp",
            model=model,
            modality="vibration",
            seed=42,
            label_budget=1.0,
            epochs=1,
            patience=1,
            batch_size=4,
            num_workers=0,
            smoke_test=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            run_id=f"smoke_{model}",
            paper_experiment="E1",
            protocol="nln_emp",
            fold_id="fold",
            configuration="smoke",
        )
    )
    assert result["pipeline_version"] == PIPELINE_VERSION
    assert result["model"] == model
    assert result["n_train_windows"] == 12
    assert 0 <= result["recording_macro_f1"] <= 1
    assert Path(result["checkpoint_path"]).exists()


def test_label_budgets_keep_complete_recordings() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "base_recording_id": f"r{recording}",
                "health_label": "fault" if recording % 2 else "healthy",
                "window": window,
            }
            for recording in range(20)
            for window in range(3)
        ]
    )
    selected = select_label_budget(dataframe, 0.25, seed=42)
    counts = selected.groupby("base_recording_id").size()
    assert len(counts) == selected["base_recording_id"].nunique()
    assert set(counts.tolist()) == {3}
    assert set(selected["health_label"]) == {"healthy", "fault"}


def _write_protocol(
    data_root: Path,
    split_root: Path,
    folder: str,
    split_filename: str,
    folds: list[str],
) -> None:
    pd.DataFrame({"fold_id": folds}).to_csv(
        split_root / split_filename, index=False
    )
    for fold in folds:
        directory = data_root / folder / fold
        (directory / "tensors").mkdir(parents=True)
        pd.DataFrame(
            {
                "tensor_id": ["dummy.pt"],
                "recording_id": ["dummy"],
                "split": ["train"],
                "health_label": ["healthy"],
            }
        ).to_csv(directory / "windows_index.csv", index=False)


def test_complete_e1_to_e6_plan_contains_all_job_families(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "processed"
    split_root = tmp_path / "splits"
    split_root.mkdir()
    _write_protocol(
        data_root,
        split_root,
        "nln_emp/nln_emp_leave_one_speed_out",
        "nln_emp_leave_one_speed_out.csv",
        ["n1", "n2", "n3", "n4"],
    )
    _write_protocol(
        data_root,
        split_root,
        "paderborn/paderborn_condition_generalization",
        "paderborn_condition_generalization.csv",
        ["p1", "p2", "p3", "p4"],
    )
    _write_protocol(
        data_root,
        split_root,
        "paderborn/paderborn_artificial_to_natural",
        "paderborn_artificial_to_natural.csv",
        ["p2_transfer"],
    )
    _write_protocol(
        data_root,
        split_root,
        "cwru/cwru_leave_one_load_out",
        "cwru_leave_one_load_out.csv",
        ["c0", "c1", "c2", "c3"],
    )
    jobs = build_paper_jobs(
        data_root=data_root,
        split_root=split_root,
        experiments=("E1", "E2", "E3", "E4", "E5", "E6"),
        seeds=(42,),
    )
    assert len(jobs) == 105
    assert {job.paper_experiment for job in jobs} == {
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
        "E6",
    }
    assert set(DEEP_BASELINE_NAMES).issubset(
        {job.model for job in jobs if job.paper_experiment == "E1"}
    )
    assert {job.label_budget for job in jobs if job.paper_experiment == "E6"} == {
        0.1,
        0.25,
        0.5,
        1.0,
    }


def test_identical_cross_experiment_jobs_have_reusable_signatures() -> None:
    common = {
        "protocol": "nln_emp",
        "fold_id": "fold",
        "processed_dir": "/tmp/fold",
        "dataset": "nln_emp",
        "model": "proposed",
        "seed": 42,
        "use_gate": True,
    }
    e1 = PaperJob(
        paper_experiment="E1",
        configuration="frozen_proposed",
        **common,
    )
    e6 = PaperJob(
        paper_experiment="E6",
        configuration="limited_labels",
        label_budget=1.0,
        **common,
    )
    args = argparse.Namespace(frozen_loss="ce_1.0", baseline_epochs=20)
    assert training_signature(e1, args) == training_signature(e6, args)


def test_e7_explanation_contains_gradcam_and_attention() -> None:
    model = MultimodalMotorModel(
        embed_dim=32,
        num_fault_families=3,
        num_attention_heads=4,
        use_modality_gate=True,
    ).eval()
    explanation = explain_sample(model, torch.randn(1, 2, 32, 32))
    assert explanation["vibration_gradcam"].shape == (32, 32)
    assert explanation["current_gradcam"].shape == (32, 32)
    assert "attention_vibration_to_current_last" in explanation


def test_cwru_validator_requires_all_four_loads(tmp_path: Path) -> None:
    for specification in expected_cwru_files():
        sio.savemat(
            tmp_path / specification.local_name,
            {
                f"X{specification.remote_name[:-4]}_DE_time": np.arange(
                    2048, dtype=float
                )
            },
        )
    assert validate_cwru_benchmark(tmp_path) == {0: 4, 1: 4, 2: 4, 3: 4}


def test_reporting_uses_fold_and_seed_structure() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "pipeline_version": PIPELINE_VERSION,
                "status": "COMPLETED",
                "paper_experiment": "E1",
                "protocol": "nln_emp",
                "dataset": "nln_emp",
                "model": "proposed",
                "configuration": "frozen",
                "label_budget": 1.0,
                "fold_id": fold,
                "seed": seed,
                "recording_macro_f1": value,
            }
            for fold, values in (("f1", (0.8, 0.9)), ("f2", (0.6, 0.7)))
            for seed, value in enumerate(values)
        ]
    )
    summary = summarize_results(dataframe)
    assert summary.iloc[0]["n_folds"] == 2
    assert summary.iloc[0]["n_seeds"] == 2
    assert summary.iloc[0][
        "recording_macro_f1_mean_of_fold_means"
    ] == pytest.approx(0.75)
