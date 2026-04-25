"""Seed utilities for deterministic and reproducible experiments."""

from __future__ import annotations

import os
import random

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    if seed < 0:
        raise ValueError("Seed must be non-negative.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str = "auto") -> str:
    """Resolve runtime device selection.

    Args:
        device: Requested device string, one of "auto", "cpu", "cuda".

    Returns:
        Final device string.
    """
    normalized = device.lower().strip()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")

    has_cuda = torch is not None and torch.cuda.is_available()

    if normalized == "auto":
        return "cuda" if has_cuda else "cpu"
    if normalized == "cuda" and not has_cuda:
        return "cpu"
    return normalized
