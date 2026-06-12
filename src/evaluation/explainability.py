"""E7 Grad-CAM, input saliency, and cross-attention artifact generation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.evaluation.checkpoint import load_model_from_checkpoint
from src.evaluation.result_utils import select_validation_run


def _normalize(array: np.ndarray) -> np.ndarray:
    array = np.maximum(array, 0)
    array = array - array.min()
    maximum = array.max()
    return array / maximum if maximum > 0 else array


def _attention_grid(weights: torch.Tensor) -> np.ndarray:
    """Average heads and queries, then reshape attended key tokens."""
    token_scores = weights[0].mean(dim=(0, 1))
    side = int(round(math.sqrt(token_scores.numel())))
    if side * side != token_scores.numel():
        return token_scores.cpu().numpy()[None, :]
    return token_scores.reshape(side, side).cpu().numpy()


def explain_sample(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int | None = None,
) -> dict[str, np.ndarray | float | int]:
    """Return modality Grad-CAM, saliency, and attention arrays."""
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(name: str):
        def hook(_module, _inputs, output):
            activations[name] = output
        return hook

    def backward_hook(name: str):
        def hook(_module, _grad_input, grad_output):
            gradients[name] = grad_output[0]
        return hook

    handles = []
    for name, encoder in (
        ("vibration", model.vib_encoder),
        ("current", model.curr_encoder),
    ):
        layer = encoder.features[-1][0]
        handles.append(layer.register_forward_hook(forward_hook(name)))
        handles.append(layer.register_full_backward_hook(backward_hook(name)))

    sample = inputs.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    health_logits, _ = model(sample)
    probability = torch.softmax(health_logits, dim=1)[0, 1]
    chosen_class = (
        int(health_logits.argmax(dim=1).item())
        if target_class is None
        else int(target_class)
    )
    health_logits[0, chosen_class].backward()

    output: dict[str, Any] = {
        "fault_probability": float(probability.detach().cpu()),
        "target_class": chosen_class,
        "vibration_saliency": _normalize(
            sample.grad[0, 0].abs().detach().cpu().numpy()
        ),
        "current_saliency": _normalize(
            sample.grad[0, 1].abs().detach().cpu().numpy()
        ),
    }
    for name in ("vibration", "current"):
        weights = gradients[name].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[name]).sum(dim=1, keepdim=True).relu()
        cam = F.interpolate(
            cam,
            size=sample.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        output[f"{name}_gradcam"] = _normalize(
            cam[0, 0].detach().cpu().numpy()
        )

    if model.ablation_mode is None:
        first_block = model.fusion.blocks[0]
        last_block = model.fusion.blocks[-1]
        if first_block.last_attention_weights is not None:
            output["attention_vibration_to_current_first"] = _normalize(
                _attention_grid(first_block.last_attention_weights[0])
            )
        if last_block.last_attention_weights is not None:
            output["attention_vibration_to_current_last"] = _normalize(
                _attention_grid(last_block.last_attention_weights[0])
            )
            output["attention_current_to_vibration_last"] = _normalize(
                _attention_grid(last_block.last_attention_weights[1])
            )
    for handle in handles:
        handle.remove()
    return output


def _plot_explanation(
    input_tensor: torch.Tensor,
    explanation: dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    figure, axes = plt.subplots(2, 3, figsize=(12, 7))
    vibration = input_tensor[0, 0].cpu().numpy()
    current = input_tensor[0, 1].cpu().numpy()
    panels = (
        (vibration, "Vibration input"),
        (explanation["vibration_gradcam"], "Vibration Grad-CAM"),
        (explanation["vibration_saliency"], "Vibration saliency"),
        (current, "Current input"),
        (explanation["current_gradcam"], "Current Grad-CAM"),
        (
            explanation.get(
                "attention_vibration_to_current_last",
                explanation["current_saliency"],
            ),
            "Cross-attention",
        ),
    )
    for axis, (image, panel_title) in zip(axes.flat, panels):
        axis.imshow(image, aspect="auto", origin="lower", cmap="viridis")
        axis.set_title(panel_title)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def generate_e7_artifacts(
    results_path: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate E7 artifacts from a validation-selected proposed checkpoint."""
    selected = select_validation_run(
        results_path, paper_experiment="E1", model="proposed"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_from_checkpoint(
        Path(selected["checkpoint_path"]), device
    )
    processed_dir = Path(selected["processed_dir"])
    index = pd.read_csv(processed_dir / "windows_index.csv")
    test = index[index["split"] == "test"]
    samples = {
        "healthy": test[
            test["health_label"].astype(str).str.lower() == "healthy"
        ],
        "fault": test[
            test["health_label"].astype(str).str.lower().str.contains("fault")
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    metadata = {
        "selected_run_id": str(selected["run_id"]),
        "selection_metric": "validation_recording_macro_f1",
        "decision_threshold": float(checkpoint.get("decision_threshold", 0.5)),
    }
    for label, candidates in samples.items():
        if candidates.empty:
            continue
        row = candidates.iloc[0]
        tensor = torch.load(
            processed_dir / "tensors" / str(row["tensor_id"]),
            map_location=device,
            weights_only=True,
        ).float().unsqueeze(0)
        explanation = explain_sample(model, tensor)
        array_path = output_dir / f"e7_{label}_explanation.npz"
        np.savez(
            array_path,
            **{
                key: value
                for key, value in explanation.items()
                if isinstance(value, np.ndarray)
            },
        )
        written.append(array_path)
        figure_path = output_dir / f"e7_{label}_explanation.pdf"
        _plot_explanation(
            tensor.detach().cpu(),
            explanation,
            figure_path,
            f"{label.title()} sample",
        )
        if figure_path.exists():
            written.append(figure_path)
    metadata_path = output_dir / "e7_selection.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    written.append(metadata_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default="results/tables/corrected_paper_experiments.csv"
    )
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    for path in generate_e7_artifacts(
        Path(args.results), Path(args.output_dir)
    ):
        print(path)


if __name__ == "__main__":
    main()
