"""Generate dataset and model-complexity tables without fixed result rows."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from src.models.multimodal_cross_attention import MultimodalMotorModel


def measure_inference_ms(model: torch.nn.Module, inputs: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(inputs)
        start = time.perf_counter()
        for _ in range(20):
            model(inputs)
    return (time.perf_counter() - start) * 1000 / 20


def generate_tables(metadata_path: Path, output_dir: Path) -> tuple[Path, Path]:
    metadata = pd.read_csv(metadata_path)
    dataset_rows = []
    protocols = {
        "nln_emp": "Leave-one-speed-out",
        "paderborn": "Condition and artificial-to-natural",
        "cwru": "Leave-one-load-out",
    }
    for dataset, group in metadata.groupby("dataset"):
        dataset_rows.append(
            {
                "dataset": dataset,
                "sensors": "|".join(
                    sorted(group["sensor_types_present"].astype(str).unique())
                ),
                "fault_families": group[
                    group["health_label"].astype(str).str.lower() == "fault"
                ]["fault_family"].nunique(),
                "protocol": protocols.get(str(dataset), "unknown"),
                "recordings": group["base_recording_id"].nunique(),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = output_dir / "table1_datasets.csv"
    pd.DataFrame(dataset_rows).to_csv(dataset_path, index=False)

    family_count = max(
        2,
        metadata[metadata["dataset"] == "nln_emp"]["fault_family"].nunique(),
    )
    dummy = torch.randn(1, 2, 128, 128)
    complexity_rows = []
    for name, ablation in (
        ("proposed", None),
        ("vibration_only", "vibration_only"),
        ("current_only", "current_only"),
    ):
        model = MultimodalMotorModel(
            num_fault_families=family_count,
            ablation_mode=ablation,
        )
        complexity_rows.append(
            {
                "model": name,
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "cpu_inference_ms": measure_inference_ms(model, dummy),
            }
        )
    complexity_path = output_dir / "table5_complexity.csv"
    pd.DataFrame(complexity_rows).to_csv(complexity_path, index=False)
    return dataset_path, complexity_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata", default="data/metadata/metadata_master.csv"
    )
    parser.add_argument("--output-dir", default="results/tables")
    args = parser.parse_args()
    print(generate_tables(Path(args.metadata), Path(args.output_dir)))


if __name__ == "__main__":
    main()
