"""Tests for seed and device utilities."""

from __future__ import annotations

import random

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from src.utils.seed import resolve_device, set_global_seed


def test_set_global_seed_reproducible_python_numpy_torch() -> None:
    set_global_seed(123)
    python_first = random.random()
    numpy_first = float(np.random.rand())
    torch_first = float(torch.rand(1).item())

    set_global_seed(123)
    python_second = random.random()
    numpy_second = float(np.random.rand())
    torch_second = float(torch.rand(1).item())

    assert python_first == python_second
    assert numpy_first == numpy_second
    assert torch_first == torch_second


def test_set_global_seed_negative_raises() -> None:
    try:
        set_global_seed(-1)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative seed")


def test_resolve_device_values() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}
