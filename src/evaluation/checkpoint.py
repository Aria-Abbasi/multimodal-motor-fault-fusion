"""Checkpoint-aware model reconstruction for evaluation scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from src.models.multimodal_cross_attention import (
    MODEL_ATTENTION_HEADS,
    MODEL_DROPOUT,
    MultimodalMotorModel,
)


def infer_model_config(
    checkpoint: dict[str, Any], ablation_mode: Optional[str] = None
) -> dict[str, Any]:
    """Read model configuration, with shape-based support for older checkpoints."""
    state_dict = checkpoint["state_dict"]
    saved_config = dict(checkpoint.get("model_config", {}))

    family_weight = state_dict["head_fault_family.3.weight"]
    projection_weight = state_dict["vib_encoder.projection.weight"]
    use_gate = any(key.startswith("current_gate.") for key in state_dict)

    return {
        "embed_dim": int(saved_config.get("embed_dim", projection_weight.shape[0])),
        "num_fault_families": int(
            saved_config.get("num_fault_families", family_weight.shape[0])
        ),
        "ablation_mode": (
            ablation_mode
            if ablation_mode is not None
            else saved_config.get("ablation_mode")
        ),
        "use_modality_gate": bool(
            saved_config.get("use_modality_gate", use_gate)
        ),
        "num_attention_heads": int(
            saved_config.get("num_attention_heads", MODEL_ATTENTION_HEADS)
        ),
        "dropout": float(saved_config.get("dropout", MODEL_DROPOUT)),
    }


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    ablation_mode: Optional[str] = None,
) -> tuple[MultimodalMotorModel, dict[str, Any]]:
    """Load a model using the exact saved head and architecture configuration."""
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    config = infer_model_config(checkpoint, ablation_mode=ablation_mode)
    model = MultimodalMotorModel(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
