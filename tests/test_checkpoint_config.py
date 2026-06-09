"""Checkpoint reconstruction must not assume a fixed family-head size."""

from pathlib import Path

import torch

from src.evaluation.checkpoint import load_model_from_checkpoint
from src.models.multimodal_cross_attention import MultimodalMotorModel


def test_checkpoint_loader_restores_dynamic_model_config(tmp_path: Path) -> None:
    model = MultimodalMotorModel(
        embed_dim=32,
        num_fault_families=7,
        use_modality_gate=True,
        num_attention_heads=4,
        dropout=0.2,
    )
    checkpoint_path = tmp_path / "model.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "embed_dim": 32,
                "num_fault_families": 7,
                "ablation_mode": None,
                "use_modality_gate": True,
                "num_attention_heads": 4,
                "dropout": 0.2,
            },
            "family_to_index": {"unknown": 0, "healthy": 1},
        },
        checkpoint_path,
    )

    restored, checkpoint = load_model_from_checkpoint(
        checkpoint_path, torch.device("cpu")
    )

    assert restored.head_fault_family[-1].out_features == 7
    assert restored.current_gate is not None
    assert restored.dropout == 0.2
    assert checkpoint["family_to_index"]["healthy"] == 1
