"""Generate checkpoint-aligned predictions and diagnostic figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.checkpoint import load_model_from_checkpoint
from src.evaluation.result_utils import select_validation_run
from src.training.data_selection import recording_column


def collect_selected_predictions(
    results_path: Path,
    paper_experiment: str = "E1",
) -> tuple[pd.Series, pd.DataFrame]:
    selected = select_validation_run(
        results_path,
        paper_experiment=paper_experiment,
        model="proposed",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_model_from_checkpoint(
        Path(selected["checkpoint_path"]), device
    )
    threshold = float(checkpoint.get("decision_threshold", 0.5))
    processed_dir = Path(selected["processed_dir"])
    index = pd.read_csv(processed_dir / "windows_index.csv")
    test = index[index["split"] == "test"].reset_index(drop=True)
    rows = []
    with torch.no_grad():
        for row in test.to_dict("records"):
            tensor = torch.load(
                processed_dir / "tensors" / str(row["tensor_id"]),
                map_location=device,
                weights_only=True,
            ).float().unsqueeze(0)
            health, family = model(tensor)
            probability = torch.softmax(health, dim=1)[0, 1].item()
            rows.append(
                {
                    **row,
                    "target": int("fault" in str(row["health_label"]).lower()),
                    "fault_probability": probability,
                    "prediction": int(probability >= threshold),
                    "family_prediction": int(family.argmax(dim=1).item()),
                    "latent": "|".join(
                        map(
                            str,
                            torch.cat([health, family], dim=1)[0]
                            .float()
                            .cpu()
                            .tolist(),
                        )
                    ),
                }
            )
    return selected, pd.DataFrame(rows)


def generate_prediction_artifacts(
    results_path: Path,
    output_dir: Path,
) -> list[Path]:
    selected, predictions = collect_selected_predictions(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "selected_test_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    written = [prediction_path]
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.manifold import TSNE
        from sklearn.metrics import (
            confusion_matrix,
            precision_recall_curve,
            roc_curve,
        )
    except ImportError:
        return written

    matrix = confusion_matrix(
        predictions["target"], predictions["prediction"], labels=[0, 1]
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=("Healthy", "Fault"),
        yticklabels=("Healthy", "Fault"),
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    figure.tight_layout()
    path = output_dir / "fig_confusion_matrix.pdf"
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    written.append(path)

    checkpoint = torch.load(
        Path(selected["checkpoint_path"]),
        map_location="cpu",
        weights_only=True,
    )
    family_mapping = checkpoint.get("family_to_index", {})
    if family_mapping and "fault_family" in predictions:
        labels = list(range(len(family_mapping)))
        inverse = {value: key for key, value in family_mapping.items()}
        family_targets = [
            family_mapping.get(str(value), 0)
            for value in predictions["fault_family"]
        ]
        family_matrix = confusion_matrix(
            family_targets,
            predictions["family_prediction"],
            labels=labels,
        )
        figure, axis = plt.subplots(figsize=(9, 7))
        names = [inverse.get(index, str(index)) for index in labels]
        sns.heatmap(
            family_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=names,
            yticklabels=names,
            ax=axis,
        )
        axis.set_xlabel("Predicted family")
        axis.set_ylabel("True family")
        figure.tight_layout()
        path = output_dir / "fig_family_confusion_matrix.pdf"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

    if predictions["target"].nunique() == 2:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            predictions["target"], predictions["fault_probability"]
        )
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(false_positive_rate, true_positive_rate)
        axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
        axis.set_xlabel("False positive rate")
        axis.set_ylabel("True positive rate")
        figure.tight_layout()
        path = output_dir / "fig_roc_curve.pdf"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

        precision, recall, _ = precision_recall_curve(
            predictions["target"], predictions["fault_probability"]
        )
        prevalence = float(predictions["target"].mean())
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.plot(recall, precision)
        axis.axhline(prevalence, linestyle="--", color="grey")
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        figure.tight_layout()
        path = output_dir / "fig_precision_recall_curve.pdf"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        written.append(path)

    latent = np.asarray(
        [
            [float(value) for value in text.split("|")]
            for text in predictions["latent"]
        ]
    )
    if len(latent) >= 5:
        perplexity = min(30, max(2, len(latent) // 4))
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            init="pca",
        ).fit_transform(latent)
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=predictions["target"],
            cmap="coolwarm",
            s=12,
        )
        figure.tight_layout()
        path = output_dir / "fig_tsne.pdf"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", default="results/tables/corrected_paper_experiments.csv"
    )
    parser.add_argument("--output-dir", default="results/figures")
    args = parser.parse_args()
    for path in generate_prediction_artifacts(
        Path(args.results), Path(args.output_dir)
    ):
        print(path)


if __name__ == "__main__":
    main()
