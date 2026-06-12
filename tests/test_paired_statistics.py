"""Tests for fold-and-seed matched statistical comparisons."""

from __future__ import annotations

import pandas as pd

from src.evaluation.generate_table6_stats import paired_comparisons


def test_statistics_pair_identical_fold_and_seed_cells() -> None:
    rows = []
    for fold in ("f1", "f2"):
        for seed in (1, 2, 3):
            rows.append(
                {
                    "experiment": "proposed",
                    "fold_id": fold,
                    "seed": seed,
                    "recording_macro_f1": 0.9,
                    "status": "COMPLETED",
                }
            )
            rows.append(
                {
                    "experiment": "baseline",
                    "fold_id": fold,
                    "seed": seed,
                    "recording_macro_f1": 0.7,
                    "status": "COMPLETED",
                }
            )
    result = paired_comparisons(pd.DataFrame(rows), "proposed")
    assert len(result) == 1
    assert result.iloc[0]["n_pairs"] == 6
    assert result.iloc[0]["mean_paired_difference"] > 0
    assert 0 <= result.iloc[0]["p_value_holm"] <= 1
