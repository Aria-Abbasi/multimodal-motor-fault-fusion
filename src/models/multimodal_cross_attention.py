"""Multimodal vibration-current cross-attention model."""

from __future__ import annotations

import time
from typing import Optional

import torch
import torch.nn as nn


MODEL_EMBED_DIM = 256
MODEL_ATTENTION_HEADS = 4
MODEL_DROPOUT = 0.1


class SpectrogramEncoder(nn.Module):
    """Encode one 128x128 spectrogram into a compact embedding."""

    def __init__(
        self, in_channels: int = 1, embed_dim: int = MODEL_EMBED_DIM
    ) -> None:
        super().__init__()

        def conv_block(in_features: int, out_features: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_features),
                nn.GELU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            conv_block(in_channels, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        return self.projection(x)


class DynamicCurrentGate(nn.Module):
    """Predict a sample-wise current-signal volume in the range [0, 1]."""

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

        # Start near 0.8, preserving current information while still allowing
        # the network to quickly turn it down when vibration is sufficient.
        final_linear = self.network[-2]
        nn.init.zeros_(final_linear.weight)
        nn.init.constant_(final_linear.bias, 1.38629436112)

    def forward(
        self, vibration_embedding: torch.Tensor, current_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.network(torch.cat([vibration_embedding, current_embedding], dim=-1))
        return current_embedding * gate, gate


class CrossAttentionFusion(nn.Module):
    """Fuse vibration and current embeddings with bidirectional attention."""

    def __init__(
        self,
        embed_dim: int = MODEL_EMBED_DIM,
        num_heads: int = MODEL_ATTENTION_HEADS,
        dropout: float = MODEL_DROPOUT,
    ) -> None:
        super().__init__()
        self.attn_v_c = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_c_v = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim * 2)

    def forward(
        self, vibration_embedding: torch.Tensor, current_embedding: torch.Tensor
    ) -> torch.Tensor:
        vibration_sequence = vibration_embedding.unsqueeze(1)
        current_sequence = current_embedding.unsqueeze(1)

        vibration_fused, _ = self.attn_v_c(
            query=vibration_sequence,
            key=current_sequence,
            value=current_sequence,
            need_weights=False,
        )
        current_fused, _ = self.attn_c_v(
            query=current_sequence,
            key=vibration_sequence,
            value=vibration_sequence,
            need_weights=False,
        )

        fused = torch.cat(
            [vibration_fused.squeeze(1), current_fused.squeeze(1)], dim=-1
        )
        return self.norm(fused)


class MultimodalMotorModel(nn.Module):
    """Cross-attention model with optional dynamic current gating."""

    def __init__(
        self,
        embed_dim: int = MODEL_EMBED_DIM,
        num_fault_families: int = 5,
        ablation_mode: Optional[str] = None,
        use_modality_gate: bool = False,
        num_attention_heads: int = MODEL_ATTENTION_HEADS,
        dropout: float = MODEL_DROPOUT,
    ) -> None:
        super().__init__()
        if ablation_mode not in {None, "vibration_only", "current_only"}:
            raise ValueError(f"Unsupported ablation mode: {ablation_mode}")
        if embed_dim % num_attention_heads != 0:
            raise ValueError("embed_dim must be divisible by num_attention_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.ablation_mode = ablation_mode
        self.use_modality_gate = use_modality_gate and ablation_mode is None
        self.last_current_gate: Optional[torch.Tensor] = None

        self.vib_encoder = SpectrogramEncoder(in_channels=1, embed_dim=embed_dim)
        self.curr_encoder = SpectrogramEncoder(in_channels=1, embed_dim=embed_dim)
        self.current_gate = (
            DynamicCurrentGate(embed_dim=embed_dim) if self.use_modality_gate else None
        )
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.fusion = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        fusion_dim = embed_dim * 2 if ablation_mode is None else embed_dim
        self.head_early_fault = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        self.head_fault_family = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_fault_families),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vibration_signal = x[:, 0:1, :, :]
        current_signal = x[:, 1:2, :, :]
        self.last_current_gate = None

        if self.ablation_mode == "vibration_only":
            features = self.vib_encoder(vibration_signal)
        elif self.ablation_mode == "current_only":
            features = self.curr_encoder(current_signal)
        else:
            vibration_embedding = self.vib_encoder(vibration_signal)
            current_embedding = self.curr_encoder(current_signal)
            if self.current_gate is not None:
                current_embedding, gate = self.current_gate(
                    vibration_embedding, current_embedding
                )
                self.last_current_gate = gate.detach()
            features = self.fusion(vibration_embedding, current_embedding)

        return self.head_early_fault(features), self.head_fault_family(features)


if __name__ == "__main__":
    batch_size = 8
    dummy_input = torch.randn(batch_size, 2, 128, 128)

    for gate_enabled in (False, True):
        model = MultimodalMotorModel(
            num_fault_families=5, use_modality_gate=gate_enabled
        )
        start_time = time.time()
        health_output, family_output = model(dummy_input)
        elapsed_ms = (time.time() - start_time) * 1000
        gate_mean = (
            model.last_current_gate.mean().item()
            if model.last_current_gate is not None
            else None
        )
        print(
            f"gate={gate_enabled} health={tuple(health_output.shape)} "
            f"family={tuple(family_output.shape)} gate_mean={gate_mean} "
            f"forward_ms={elapsed_ms:.2f}"
        )
