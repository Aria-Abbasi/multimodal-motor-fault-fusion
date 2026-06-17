"""Health-classification losses used by the experiment matrix."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.experiment_config import LOSS_NAMES


@dataclass(frozen=True)
class LossConfig:
    name: str
    early_weight: float = 1.0
    focal_gamma: float = 2.0
    focal_early_focus: float = 1.0


LOSS_CONFIGS = {
    "ce_1.0": LossConfig("ce_1.0", early_weight=1.0),
    "ce_1.25": LossConfig("ce_1.25", early_weight=1.25),
    "ce_1.5": LossConfig("ce_1.5", early_weight=1.5),
    "ce_2.0": LossConfig("ce_2.0", early_weight=2.0),
    "ce_3.0": LossConfig("ce_3.0", early_weight=3.0),
    "ce_4.0": LossConfig("ce_4.0", early_weight=4.0),
    "dynamic_focal": LossConfig(
        "dynamic_focal", focal_gamma=2.0, focal_early_focus=1.0
    ),
}


class SeverityWeightedCrossEntropy(nn.Module):
    """Cross entropy with a configurable multiplier for early-fault samples."""

    def __init__(self, early_weight: float) -> None:
        super().__init__()
        if early_weight <= 0:
            raise ValueError("early_weight must be positive")
        self.early_weight = early_weight

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        early_fault_mask: torch.Tensor,
    ) -> torch.Tensor:
        sample_loss = F.cross_entropy(logits, targets, reduction="none")
        weights = torch.ones_like(sample_loss)
        weights = torch.where(
            early_fault_mask.bool(),
            torch.full_like(weights, self.early_weight),
            weights,
        )
        return (sample_loss * weights).mean()


class DynamicFocalLoss(nn.Module):
    """Focal loss with extra emphasis that grows for hard early-fault samples."""

    def __init__(self, gamma: float = 2.0, early_focus: float = 1.0) -> None:
        super().__init__()
        if gamma < 0 or early_focus < 0:
            raise ValueError("gamma and early_focus must be non-negative")
        self.gamma = gamma
        self.early_focus = early_focus

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        early_fault_mask: torch.Tensor,
    ) -> torch.Tensor:
        cross_entropy = F.cross_entropy(logits, targets, reduction="none")
        target_probability = torch.exp(-cross_entropy)
        difficulty = 1.0 - target_probability
        focal_factor = difficulty.pow(self.gamma)

        # Hard early faults receive up to a 2x multiplier by default. Unlike the
        # old fixed 5x penalty, the multiplier vanishes as a sample becomes easy.
        early_boost = 1.0 + (
            early_fault_mask.to(logits.dtype)
            * self.early_focus
            * difficulty.detach()
        )
        return (focal_factor * early_boost * cross_entropy).mean()


def build_health_loss(name: str) -> nn.Module:
    """Create one of the six loss configurations."""
    if name not in LOSS_CONFIGS:
        raise ValueError(f"Unknown loss '{name}'. Expected one of {LOSS_NAMES}")
    config = LOSS_CONFIGS[name]
    if name == "dynamic_focal":
        return DynamicFocalLoss(
            gamma=config.focal_gamma, early_focus=config.focal_early_focus
        )
    return SeverityWeightedCrossEntropy(early_weight=config.early_weight)
