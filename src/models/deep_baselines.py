"""Deep baseline architectures with a common spectrogram input contract."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


DEEP_BASELINE_NAMES = (
    "cnn",
    "lstm",
    "cnn_lstm",
    "transformer",
    "autoencoder",
)


def select_modality(inputs: torch.Tensor, modality: str) -> torch.Tensor:
    if modality == "vibration":
        return inputs[:, 0:1]
    if modality == "current":
        return inputs[:, 1:2]
    if modality == "both":
        return inputs
    raise ValueError(f"Unknown modality: {modality}")


class CNNBaseline(nn.Module):
    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 32, 9, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 7, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.network(inputs.flatten(2)).squeeze(-1))


class LSTMBaseline(nn.Module):
    def __init__(self, input_channels: int = 1, hidden_dim: int = 64) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.lstm = nn.LSTM(32 * input_channels, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(inputs, (32, 32))
        sequence = pooled.permute(0, 3, 1, 2).flatten(2)
        _, (hidden, _) = self.lstm(sequence)
        return self.classifier(hidden[-1])


class CNNLSTMBaseline(nn.Module):
    def __init__(self, input_channels: int = 1, hidden_dim: int = 64) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(input_channels, 32, 9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(4),
        )
        self.lstm = nn.LSTM(32, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.convolution(inputs.flatten(2)).transpose(1, 2)
        _, (hidden, _) = self.lstm(features)
        return self.classifier(hidden[-1])


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(32 * input_channels, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.classifier = nn.Linear(embed_dim, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(inputs, (32, 32))
        sequence = pooled.permute(0, 3, 1, 2).flatten(2)
        encoded = self.encoder(self.projection(sequence))
        return self.classifier(encoded.mean(dim=1))


class HealthyAutoencoder(nn.Module):
    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, input_channels, 4, stride=2, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))

    def anomaly_score(self, inputs: torch.Tensor) -> torch.Tensor:
        reconstruction = self(inputs)
        return (reconstruction - inputs).square().flatten(1).mean(dim=1)


def build_deep_baseline(name: str, modality: str = "vibration") -> nn.Module:
    input_channels = 2 if modality == "both" else 1
    if name == "cnn":
        return CNNBaseline(input_channels)
    if name == "lstm":
        return LSTMBaseline(input_channels)
    if name == "cnn_lstm":
        return CNNLSTMBaseline(input_channels)
    if name == "transformer":
        return TransformerBaseline(input_channels)
    if name == "autoencoder":
        return HealthyAutoencoder(input_channels)
    raise ValueError(f"Unknown deep baseline: {name}")
