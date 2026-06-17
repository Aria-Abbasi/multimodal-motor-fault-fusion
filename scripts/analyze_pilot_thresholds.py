"""Sweep validation-selected thresholds for corrected NLN pilot checkpoints."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef

from src.evaluation.checkpoint import infer_model_config
from src.evaluation.metrics import compute_binary_metrics
from src.models.multimodal_cross_attention import MultimodalMotorModel
from src.training.train_multimodal import (
    MultimodalDataset,
    evaluate_model,
    load_protocol_tensor_cache,
    make_loader,
    prediction_count_summary,
    select_index_splits,
)


def candidate_thresholds(scores: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(scores), dtype=np.float64)
    return np.unique(np.concatenate(([0.0, 0.5, 1.0], values)))


def recording_metrics_at(
    targets: list[int],
    scores: list[float],
    early: list[bool],
    threshold: float,
) -> dict[str, Any]:
    predictions = [int(score >= threshold) for score in scores]
    return compute_binary_metrics(
        targets,
        predictions,
        fault_probabilities=scores,
        early_fault_mask=early,
    )


def choose_threshold(
    strategy: str,
    targets: list[int],
    scores: list[float],
    early: list[bool],
) -> float:
    if strategy == "fixed_0p5":
        return 0.5

    y_true = np.asarray(targets, dtype=np.int64)
    y_scores = np.asarray(scores, dtype=np.float64)
    y_early = np.asarray(early, dtype=bool) & (y_true == 1)
    has_both = len(np.unique(y_true)) == 2
    ranked: list[tuple[float, ...]] = []
    for threshold in candidate_thresholds(y_scores):
        y_pred = (y_scores >= threshold).astype(np.int64)
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        precision = float(
            np.sum((y_pred == 1) & (y_true == 1)) / max(np.sum(y_pred == 1), 1)
        )
        early_recall = (
            float(np.mean(y_pred[y_early] == 1)) if np.any(y_early) else float("nan")
        )
        balanced_acc = (
            float(balanced_accuracy_score(y_true, y_pred)) if has_both else float("nan")
        )
        mcc = float(matthews_corrcoef(y_true, y_pred)) if has_both else float("nan")

        if strategy == "macro_f1":
            ranked.append((macro_f1, precision, mcc, -abs(float(threshold) - 0.5), threshold))
        elif strategy == "mcc":
            ranked.append((mcc, macro_f1, precision, -abs(float(threshold) - 0.5), threshold))
        elif strategy == "balanced_acc":
            ranked.append((balanced_acc, macro_f1, precision, -abs(float(threshold) - 0.5), threshold))
        elif strategy.startswith("precision_at_early_recall_"):
            required = float(strategy.rsplit("_", 1)[-1].replace("p", "."))
            if not np.isnan(early_recall) and early_recall >= required:
                ranked.append((precision, macro_f1, mcc, threshold, threshold))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    if not ranked:
        return choose_threshold("macro_f1", targets, scores, early)
    return float(max(ranked)[-1])


def evaluate_checkpoint(
    row: pd.Series,
    args: argparse.Namespace,
    shared_tensors: dict[str, torch.Tensor] | None,
) -> list[dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(str(row["checkpoint_path"]))
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model = MultimodalMotorModel(**infer_model_config(checkpoint)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    family_to_index = checkpoint["family_to_index"]
    processed_dir = Path(str(row["processed_dir"]))
    index = pd.read_csv(processed_dir / "windows_index.csv")
    _, validation_frame, test_frame = select_index_splits(index, smoke_test=False)
    tensor_dir = processed_dir / "tensors"
    dataset_name = str(row["dataset"])

    validation_dataset = MultimodalDataset(
        validation_frame,
        tensor_dir,
        family_to_index,
        shared_tensor_cache=shared_tensors,
        dataset_name=dataset_name,
    )
    test_dataset = MultimodalDataset(
        test_frame,
        tensor_dir,
        family_to_index,
        shared_tensor_cache=shared_tensors,
        dataset_name=dataset_name,
    )
    validation_loader = make_loader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=int(row["seed"]),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=int(row["seed"]),
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    validation_raw = evaluate_model(
        model,
        validation_loader,
        device,
        amp_enabled=device.type == "cuda",
        decision_threshold=0.5,
        amp_dtype=amp_dtype,
    )
    test_raw = evaluate_model(
        model,
        test_loader,
        device,
        amp_enabled=device.type == "cuda",
        decision_threshold=0.5,
        amp_dtype=amp_dtype,
    )

    validation_targets = validation_raw["_recording_targets"]
    validation_scores = validation_raw["_recording_probabilities"]
    validation_early = [False] * len(validation_targets)
    test_targets = test_raw["_recording_targets"]
    test_scores = test_raw["_recording_probabilities"]
    test_early = [False] * len(test_targets)
    if "_recording_early" in validation_raw:
        validation_early = validation_raw["_recording_early"]
    if "_recording_early" in test_raw:
        test_early = test_raw["_recording_early"]

    # evaluate_model currently keeps recording early masks only through public
    # metrics, so rebuild them from window-level data if private masks are absent.
    if "_recording_early" not in validation_raw:
        from src.evaluation.metrics import aggregate_recording_predictions

        _, _, validation_early = aggregate_recording_predictions(
            validation_dataset.recording_ids,
            validation_dataset.health_labels,
            validation_raw["_probabilities"],
            validation_dataset.early_fault_labels,
        )
        _, _, test_early = aggregate_recording_predictions(
            test_dataset.recording_ids,
            test_dataset.health_labels,
            test_raw["_probabilities"],
            test_dataset.early_fault_labels,
        )

    output = []
    for strategy in args.strategies:
        threshold = choose_threshold(
            strategy,
            validation_targets,
            validation_scores,
            validation_early,
        )
        validation_metrics = recording_metrics_at(
            validation_targets,
            validation_scores,
            validation_early,
            threshold,
        )
        test_metrics = recording_metrics_at(
            test_targets,
            test_scores,
            test_early,
            threshold,
        )
        validation_counts = prediction_count_summary(
            {
                "_probabilities": validation_scores,
                "_targets": validation_targets,
                "_recording_probabilities": validation_scores,
                "_recording_targets": validation_targets,
            },
            threshold,
        )
        test_counts = prediction_count_summary(
            {
                "_probabilities": test_scores,
                "_targets": test_targets,
                "_recording_probabilities": test_scores,
                "_recording_targets": test_targets,
            },
            threshold,
        )
        output.append(
            {
                "fold_id": row["fold_id"],
                "experiment": row["experiment"],
                "loss_name": row["loss_name"],
                "modality_gate": row["modality_gate"],
                "seed": int(row["seed"]),
                "strategy": strategy,
                "threshold": threshold,
                **{f"validation_recording_{key}": value for key, value in validation_metrics.items()},
                **{f"test_recording_{key}": value for key, value in test_metrics.items()},
                **{f"validation_{key}": value for key, value in validation_counts.items()},
                **{f"test_{key}": value for key, value in test_counts.items()},
            }
        )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/tables/nln_validation_pilot_corrected.csv")
    parser.add_argument("--output", default="results/tables/nln_threshold_sweep.csv")
    parser.add_argument("--summary", default="results/tables/nln_threshold_sweep_summary.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--cache-max-gb", type=float, default=48.0)
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[
            "macro_f1",
            "mcc",
            "balanced_acc",
            "precision_at_early_recall_0p90",
            "precision_at_early_recall_0p95",
            "precision_at_early_recall_1p00",
            "fixed_0p5",
        ],
    )
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    all_rows: list[dict[str, Any]] = []
    for (fold_id, processed_dir), group in results.groupby(["fold_id", "processed_dir"], sort=False):
        print(f"Evaluating {fold_id}: {len(group)} checkpoints")
        cache = load_protocol_tensor_cache(
            Path(str(processed_dir)),
            maximum_cache_gb=args.cache_max_gb,
            smoke_test=False,
        )
        try:
            for _, row in group.iterrows():
                print(f"  {row['experiment']}")
                all_rows.extend(evaluate_checkpoint(row, args, cache.tensors))
        finally:
            cache.clear()
            del cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    output = pd.DataFrame(all_rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    group_cols = ["strategy", "experiment"]
    summary = (
        output.groupby(group_cols)
        .agg(
            folds=("fold_id", "nunique"),
            validation_macro_f1_mean=("validation_recording_macro_f1", "mean"),
            validation_early_recall_mean=("validation_recording_early_fault_recall", "mean"),
            validation_precision_mean=("validation_recording_fault_precision", "mean"),
            validation_mcc_mean=("validation_recording_mcc", "mean"),
            test_macro_f1_mean=("test_recording_macro_f1", "mean"),
            test_early_recall_mean=("test_recording_early_fault_recall", "mean"),
            test_precision_mean=("test_recording_fault_precision", "mean"),
            test_mcc_mean=("test_recording_mcc", "mean"),
            test_predicted_healthy_total=("test_recording_predicted_healthy", "sum"),
        )
        .reset_index()
        .sort_values(
            [
                "validation_early_recall_mean",
                "validation_macro_f1_mean",
                "validation_precision_mean",
                "validation_mcc_mean",
            ],
            ascending=False,
        )
    )
    summary.to_csv(args.summary, index=False)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary}")


if __name__ == "__main__":
    main()
