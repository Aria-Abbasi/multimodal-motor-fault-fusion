"""Small end-to-end CPU training test using synthetic cached tensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from src.training.train_multimodal import (
    load_protocol_tensor_cache,
    select_curriculum_stage_two_rows,
    train_multimodal,
)


def build_synthetic_dataset(root: Path) -> None:
    """Write a balanced train/validation/test tensor dataset."""
    tensor_dir = root / "tensors"
    tensor_dir.mkdir(parents=True)
    generator = torch.Generator().manual_seed(20260609)
    rows = []
    sample_index = 0

    for split, count in (("train", 12), ("val", 4), ("test", 4)):
        for local_index in range(count):
            is_fault = local_index % 2 == 1
            tensor = torch.randn(2, 16, 16, generator=generator) * 0.1
            if is_fault:
                tensor[0, 4:12, 4:12] += 1.0
                tensor[1, 6:10, 6:10] += 0.25

            tensor_id = f"sample_{sample_index:03d}.pt"
            torch.save(tensor, tensor_dir / tensor_id)
            rows.append(
                {
                    "tensor_id": tensor_id,
                    "recording_id": f"recording_{sample_index:03d}",
                    "split": split,
                    "fault_family": (
                        "bearing_inner_race" if is_fault else "healthy"
                    ),
                    "severity": "1" if is_fault else "0",
                    "health_label": "fault" if is_fault else "healthy",
                }
            )
            sample_index += 1

    pd.DataFrame(rows).to_csv(root / "windows_index.csv", index=False)


def test_cpu_training_pipeline_end_to_end(tmp_path: Path) -> None:
    data_dir = tmp_path / "synthetic"
    build_synthetic_dataset(data_dir)
    checkpoint_dir = tmp_path / "checkpoints"
    metrics_path = tmp_path / "metrics.csv"

    result = train_multimodal(
        argparse.Namespace(
            processed_dir=str(data_dir),
            dataset="nln_emp",
            seed=42,
            loss_name="dynamic_focal",
            use_modality_gate=True,
            use_curriculum=True,
            ablation=None,
            stage1_epochs=1,
            stage2_epochs=1,
            batch_size=4,
            num_workers=0,
            learning_rate=1e-3,
            minimum_learning_rate=1e-5,
            weight_decay=1e-4,
            gradient_clip_norm=1.0,
            modality_dropout=0.2,
            warmup_ratio=0.1,
            family_loss_weight=0.5,
            preload=False,
            amp=False,
            smoke_test=True,
            checkpoint_dir=str(checkpoint_dir),
            metrics_file=str(metrics_path),
            write_detailed_metrics=True,
            run_id="cpu_smoke",
            experiment="cpu_smoke",
        )
    )

    assert result["dataset"] == "nln_emp"
    assert result["modality_gate"] is True
    assert result["loss_name"] == "dynamic_focal"
    assert 0.0 <= result["macro_f1"] <= 1.0
    assert Path(result["checkpoint_path"]).exists()
    assert metrics_path.exists()


def test_stage_two_contains_only_healthy_and_early_faults(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "synthetic"
    build_synthetic_dataset(data_dir)
    dataframe = pd.read_csv(data_dir / "windows_index.csv")
    later_fault = dataframe.iloc[[1]].copy()
    later_fault["severity"] = "2"
    dataframe = pd.concat([dataframe, later_fault], ignore_index=True)

    stage_two = select_curriculum_stage_two_rows(dataframe, "nln_emp")

    assert not (
        (stage_two["health_label"] == "fault")
        & (stage_two["severity"].astype(str) == "2")
    ).any()
    assert (stage_two["severity"].astype(str) == "1").any()


def test_shared_protocol_cache_prevents_per_run_disk_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "synthetic"
    build_synthetic_dataset(data_dir)
    cache = load_protocol_tensor_cache(
        data_dir, maximum_cache_gb=1.0, smoke_test=True
    )
    assert len(cache.tensors) == 20
    assert cache.cached_splits == ("train", "val", "test")

    def fail_if_loaded_from_disk(*args: object, **kwargs: object) -> None:
        raise AssertionError("Tensor was reloaded after shared cache creation")

    monkeypatch.setattr(torch, "load", fail_if_loaded_from_disk)
    result = train_multimodal(
        argparse.Namespace(
            processed_dir=str(data_dir),
            dataset="nln_emp",
            seed=42,
            loss_name="ce_1.0",
            use_modality_gate=False,
            use_curriculum=True,
            ablation=None,
            batch_size=4,
            num_workers=0,
            learning_rate=1e-3,
            minimum_learning_rate=1e-5,
            weight_decay=1e-4,
            gradient_clip_norm=1.0,
            modality_dropout=0.2,
            warmup_ratio=0.1,
            family_loss_weight=0.5,
            preload=False,
            shared_tensor_cache=cache,
            amp=False,
            smoke_test=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            write_detailed_metrics=False,
            run_id="shared_cache_smoke",
            experiment="shared_cache_smoke",
        )
    )

    assert result["shared_cache_splits"] == "train,val,test"
    assert len(cache.tensors) == 20
    cache.clear()
    assert not cache.tensors
