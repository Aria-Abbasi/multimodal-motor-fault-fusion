"""Regression tests for the loss/gating iteration."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.models.multimodal_cross_attention import MultimodalMotorModel
from src.training.losses import LOSS_NAMES, build_health_loss


def test_all_six_loss_configurations_are_finite() -> None:
    logits = torch.tensor([[0.2, 0.8], [0.8, 0.2]], requires_grad=True)
    targets = torch.tensor([1, 0])
    early_mask = torch.tensor([True, False])

    assert len(LOSS_NAMES) == 6
    for name in LOSS_NAMES:
        loss = build_health_loss(name)(logits, targets, early_mask)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_modality_gate_is_optional_and_sample_wise() -> None:
    batch = torch.randn(2, 2, 32, 32)
    gated_model = MultimodalMotorModel(
        embed_dim=32, num_fault_families=3, use_modality_gate=True
    ).eval()
    ungated_model = MultimodalMotorModel(
        embed_dim=32, num_fault_families=3, use_modality_gate=False
    ).eval()

    gated_health, gated_family = gated_model(batch)
    ungated_health, _ = ungated_model(batch)

    assert gated_health.shape == ungated_health.shape == (2, 2)
    assert gated_family.shape == (2, 3)
    assert gated_model.last_current_gate is not None
    assert gated_model.last_current_gate.shape == (2, 1)
    assert torch.all(
        (gated_model.last_current_gate >= 0)
        & (gated_model.last_current_gate <= 1)
    )
    assert ungated_model.last_current_gate is None
