"""CWRU must not masquerade a random split as leave-one-load-out."""

from pathlib import Path

import pandas as pd
import pytest

from scripts.fix_cwru_splits import regenerate_cwru_split


def test_cwru_loso_requires_multiple_loads(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.csv"
    output_path = tmp_path / "split.csv"
    pd.DataFrame(
        [
            {
                "dataset": "cwru",
                "recording_id": "cwru_1",
                "load": "1",
            },
            {
                "dataset": "cwru",
                "recording_id": "cwru_2",
                "load": "1",
            },
        ]
    ).to_csv(metadata_path, index=False)

    with pytest.raises(ValueError, match="at least two distinct loads"):
        regenerate_cwru_split(metadata_path, output_path)

    assert not output_path.exists()
