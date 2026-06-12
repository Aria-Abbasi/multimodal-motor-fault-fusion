"""Multimodal vibration-current model with spatial-token cross-attention."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


MODEL_EMBED_DIM = 256
MODEL_ATTENTION_HEADS = 4
MODEL_ATTENTION_BLOCKS = 2
MODEL_DROPOUT = 0.1


def _two_dimensional_position_encoding(
    height: int,
    width: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create deterministic 2D sinusoidal positions with shape (1, H*W, D)."""
    quarter = max(1, dimension // 4)
    frequencies = torch.exp(
        torch.arange(quarter, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / max(1, quarter - 1))
    )
    y = torch.arange(height, device=device, dtype=torch.float32)[:, None]
    x = torch.arange(width, device=device, dtype=torch.float32)[:, None]
    y_encoding = torch.cat([torch.sin(y * frequencies), torch.cos(y * frequencies)], 1)
    x_encoding = torch.cat([torch.sin(x * frequencies), torch.cos(x * frequencies)], 1)
    grid_y = y_encoding[:, None, :].expand(height, width, -1)
    grid_x = x_encoding[None, :, :].expand(height, width, -1)
    encoding = torch.cat([grid_y, grid_x], dim=-1).reshape(1, height * width, -1)
    if encoding.shape[-1] < dimension:
        encoding = nn.functional.pad(encoding, (0, dimension - encoding.shape[-1]))
    return encoding[..., :dimension].to(dtype=dtype)


class SpectrogramEncoder(nn.Module):
    """Encode one spectrogram while retaining its spatial-frequency tokens."""

    def __init__(
        self, in_channels: int = 1, embed_dim: int = MODEL_EMBED_DIM
    ) -> None:
        super().__init__()

        def block(input_channels: int, output_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.GELU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
        )
        self.projection = nn.Conv2d(256, embed_dim, kernel_size=1)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        feature_map = self.projection(self.features(signal))
        height, width = feature_map.shape[-2:]
        tokens = feature_map.flatten(2).transpose(1, 2)
        positions = _two_dimensional_position_encoding(
            height,
            width,
            tokens.shape[-1],
            tokens.device,
            tokens.dtype,
        )
        return self.output_norm(tokens + positions)


class DynamicCurrentGate(nn.Module):
    """Predict a sample-wise current volume from both modality summaries."""

    def __init__(self, embed_dim: int = MODEL_EMBED_DIM) -> None:
        super().__init__()
        hidden_dim = max(32, embed_dim // 2)
        self.network = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        final_linear = self.network[-2]
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, 1.38629436112)

    def forward(
        self, vibration_tokens: torch.Tensor, current_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        summaries = torch.cat(
            [vibration_tokens.mean(1), current_tokens.mean(1)], dim=-1
        )
        gate = self.network(summaries)
        return current_tokens * gate.unsqueeze(-1), gate


class BidirectionalCrossAttentionBlock(nn.Module):
    """One pre-norm vibration-current cross-attention and feed-forward block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.vibration_norm = nn.LayerNorm(embed_dim)
        self.current_norm = nn.LayerNorm(embed_dim)
        self.vibration_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.current_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.vibration_ffn_norm = nn.LayerNorm(embed_dim)
        self.current_ffn_norm = nn.LayerNorm(embed_dim)

        def feed_forward() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.Dropout(dropout),
            )

        self.vibration_ffn = feed_forward()
        self.current_ffn = feed_forward()
        self.dropout = nn.Dropout(dropout)
        self.last_attention_shapes: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self.last_attention_weights: (
            tuple[torch.Tensor, torch.Tensor] | None
        ) = None

    def forward(
        self, vibration: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vibration_normalized = self.vibration_norm(vibration)
        current_normalized = self.current_norm(current)
        vibration_update, vibration_weights = self.vibration_attention(
            vibration_normalized,
            current_normalized,
            current_normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        current_update, current_weights = self.current_attention(
            current_normalized,
            vibration_normalized,
            vibration_normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        vibration = vibration + self.dropout(vibration_update)
        current = current + self.dropout(current_update)
        vibration = vibration + self.vibration_ffn(self.vibration_ffn_norm(vibration))
        current = current + self.current_ffn(self.current_ffn_norm(current))
        self.last_attention_shapes = (
            tuple(vibration_weights.shape),
            tuple(current_weights.shape),
        )
        self.last_attention_weights = (
            vibration_weights.detach(),
            current_weights.detach(),
        )
        return vibration, current


class CrossAttentionFusion(nn.Module):
    """Fuse token sequences using two bidirectional cross-attention blocks."""

    def __init__(
        self,
        embed_dim: int = MODEL_EMBED_DIM,
        num_heads: int = MODEL_ATTENTION_HEADS,
        dropout: float = MODEL_DROPOUT,
        num_blocks: int = MODEL_ATTENTION_BLOCKS,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                BidirectionalCrossAttentionBlock(embed_dim, num_heads, dropout)
                for _ in range(num_blocks)
            ]
        )
        self.output_norm = nn.LayerNorm(embed_dim * 2)

    def forward(
        self, vibration: torch.Tensor, current: torch.Tensor
    ) -> torch.Tensor:
        for block in self.blocks:
            vibration, current = block(vibration, current)
        return self.output_norm(
            torch.cat([vibration.mean(1), current.mean(1)], dim=-1)
        )


class MultimodalMotorModel(nn.Module):
    """Cross-attention model with optional sample-wise current gating."""

    def __init__(
        self,
        embed_dim: int = MODEL_EMBED_DIM,
        num_fault_families: int = 5,
        ablation_mode: Optional[str] = None,
        use_modality_gate: bool = False,
        num_attention_heads: int = MODEL_ATTENTION_HEADS,
        dropout: float = MODEL_DROPOUT,
        num_attention_blocks: int = MODEL_ATTENTION_BLOCKS,
    ) -> None:
        super().__init__()
        if ablation_mode not in {None, "vibration_only", "current_only"}:
            raise ValueError(f"Unsupported ablation mode: {ablation_mode}")
        if embed_dim % num_attention_heads:
            raise ValueError("embed_dim must be divisible by num_attention_heads")

        self.ablation_mode = ablation_mode
        self.use_modality_gate = use_modality_gate and ablation_mode is None
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.num_attention_blocks = num_attention_blocks
        self.dropout = dropout
        self.last_current_gate: Optional[torch.Tensor] = None

        self.vib_encoder = SpectrogramEncoder(1, embed_dim)
        self.curr_encoder = SpectrogramEncoder(1, embed_dim)
        self.current_gate = (
            DynamicCurrentGate(embed_dim) if self.use_modality_gate else None
        )
        self.fusion = CrossAttentionFusion(
            embed_dim,
            num_attention_heads,
            dropout,
            num_attention_blocks,
        )

        fusion_dim = embed_dim if ablation_mode else embed_dim * 2

        def head(output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, output_dim),
            )

        self.head_early_fault = head(2)
        self.head_fault_family = head(num_fault_families)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_current_gate = None
        if self.ablation_mode == "vibration_only":
            features = self.vib_encoder(inputs[:, 0:1]).mean(1)
        elif self.ablation_mode == "current_only":
            features = self.curr_encoder(inputs[:, 1:2]).mean(1)
        else:
            vibration = self.vib_encoder(inputs[:, 0:1])
            current = self.curr_encoder(inputs[:, 1:2])
            if self.current_gate is not None:
                current, gate = self.current_gate(vibration, current)
                self.last_current_gate = gate.detach()
            features = self.fusion(vibration, current)
        return self.head_early_fault(features), self.head_fault_family(features)
