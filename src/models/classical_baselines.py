"""Classical baseline definitions and deterministic tensor features."""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kurtosis, skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASSICAL_BASELINE_NAMES = ("svm", "random_forest")


def extract_tensor_features(
    tensor: torch.Tensor, modality: str = "both"
) -> np.ndarray:
    """Extract fixed spectral-statistical features from selected channels."""
    tensor = tensor.float()
    if modality == "vibration":
        tensor = tensor[0:1]
    elif modality == "current":
        tensor = tensor[1:2]
    elif modality != "both":
        raise ValueError(f"Unknown modality: {modality}")

    features: list[float] = []
    for channel in tensor.numpy():
        flattened = channel.reshape(-1)
        frequency_profile = channel.mean(axis=1)
        power = np.square(frequency_profile)
        probability = power / max(float(power.sum()), 1e-12)
        spectral_entropy = -float(
            np.sum(probability * np.log(probability + 1e-12))
        )
        band_energies = [
            float(np.square(band).mean())
            for band in np.array_split(frequency_profile, 4)
        ]
        rms = float(np.sqrt(np.square(flattened).mean()))
        peak = float(np.abs(flattened).max())
        features.extend(
            [
                float(flattened.mean()),
                float(flattened.std()),
                rms,
                float(kurtosis(flattened)),
                float(skew(flattened)),
                peak / max(rms, 1e-12),
                spectral_entropy,
                *band_energies,
            ]
        )
    return np.nan_to_num(np.asarray(features, dtype=np.float64))


def build_classical_baseline(name: str, seed: int):
    if name == "svm":
        return make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", class_weight="balanced", random_state=seed),
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classical baseline: {name}")
