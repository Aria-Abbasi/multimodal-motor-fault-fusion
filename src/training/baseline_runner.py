"""Fair classical and deep baseline training on complete processed folds."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    aggregate_recording_predictions,
    compute_binary_metrics,
    is_early_fault,
    select_decision_threshold,
)
from src.models.classical_baselines import (
    CLASSICAL_BASELINE_NAMES,
    build_classical_baseline,
    extract_tensor_features,
)
from src.models.deep_baselines import (
    DEEP_BASELINE_NAMES,
    HealthyAutoencoder,
    build_deep_baseline,
    select_modality,
)
from src.training.data_selection import recording_column, select_label_budget
from src.training.train_multimodal import (
    MultimodalDataset,
    PIPELINE_VERSION,
    build_family_mapping,
    current_git_revision,
    make_loader,
    select_index_splits,
    set_seed,
)


BASELINE_NAMES = CLASSICAL_BASELINE_NAMES + DEEP_BASELINE_NAMES


def _labels(dataframe: pd.DataFrame) -> np.ndarray:
    return np.asarray(
        [
            1 if "fault" in str(value).lower() else 0
            for value in dataframe["health_label"]
        ],
        dtype=np.int64,
    )


def _estimator_scores(model: Any, features: np.ndarray) -> list[float]:
    """Return monotonic fault scores; validation chooses their threshold."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1].tolist()
    decision = np.asarray(model.decision_function(features), dtype=np.float64)
    decision = np.clip(decision, -30, 30)
    return (1.0 / (1.0 + np.exp(-decision))).tolist()


def _early(dataframe: pd.DataFrame, dataset: str) -> list[bool]:
    severity = dataframe.get(
        "severity", pd.Series([""] * len(dataframe), index=dataframe.index)
    )
    return [
        is_early_fault(value, health, dataset)
        for value, health in zip(severity, dataframe["health_label"])
    ]


def _metric_bundle(
    dataframe: pd.DataFrame,
    probabilities: list[float],
    threshold: float,
    dataset: str,
) -> dict[str, Any]:
    targets = _labels(dataframe).tolist()
    predictions = [int(score >= threshold) for score in probabilities]
    early = _early(dataframe, dataset)
    metrics = compute_binary_metrics(
        targets,
        predictions,
        fault_probabilities=probabilities,
        early_fault_mask=early,
    )
    unit = recording_column(dataframe)
    recording_targets, recording_scores, recording_early = (
        aggregate_recording_predictions(
            dataframe[unit].astype(str),
            targets,
            probabilities,
            early,
        )
    )
    recording_predictions = [
        int(score >= threshold) for score in recording_scores
    ]
    recording_metrics = compute_binary_metrics(
        recording_targets,
        recording_predictions,
        fault_probabilities=recording_scores,
        early_fault_mask=recording_early,
    )
    metrics.update(
        {f"recording_{key}": value for key, value in recording_metrics.items()}
    )
    metrics["decision_threshold"] = threshold
    return metrics


def _recording_threshold(
    dataframe: pd.DataFrame,
    probabilities: list[float],
) -> float:
    targets = _labels(dataframe).tolist()
    unit = recording_column(dataframe)
    recording_targets, recording_probabilities, _ = (
        aggregate_recording_predictions(
            dataframe[unit].astype(str),
            targets,
            probabilities,
        )
    )
    return select_decision_threshold(
        recording_targets, recording_probabilities
    )


def _load_feature_matrix(
    dataframe: pd.DataFrame,
    tensor_dir: Path,
    modality: str,
    shared_tensors: dict[str, torch.Tensor] | None = None,
) -> np.ndarray:
    features = []
    for tensor_id in dataframe["tensor_id"]:
        key = str(tensor_id)
        path = tensor_dir / key
        if shared_tensors is not None and key in shared_tensors:
            tensor = shared_tensors[key]
        else:
            if not path.exists():
                raise FileNotFoundError(f"Missing processed tensor: {path}")
            tensor = torch.load(path, map_location="cpu", weights_only=True)
        features.append(extract_tensor_features(tensor, modality))
    return np.stack(features)


def train_classical_baseline(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    tensor_dir: Path,
    dataset: str,
    seed: int,
    modality: str,
    checkpoint_path: Path,
    shared_tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    train_features = _load_feature_matrix(
        train, tensor_dir, modality, shared_tensors
    )
    validation_features = _load_feature_matrix(
        validation, tensor_dir, modality, shared_tensors
    )
    test_features = _load_feature_matrix(
        test, tensor_dir, modality, shared_tensors
    )
    model = build_classical_baseline(name, seed)
    model.fit(train_features, _labels(train))
    validation_scores = _estimator_scores(model, validation_features)
    threshold = _recording_threshold(validation, validation_scores)
    test_scores = _estimator_scores(model, test_features)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "pipeline_version": PIPELINE_VERSION,
            "decision_threshold": threshold,
            "modality": modality,
        },
        checkpoint_path,
    )
    return {
        **{
            f"validation_{key}": value
            for key, value in _metric_bundle(
                validation, validation_scores, threshold, dataset
            ).items()
        },
        **_metric_bundle(test, test_scores, threshold, dataset),
    }


def _probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality: str,
    autoencoder: bool,
) -> list[float]:
    model.eval()
    output: list[float] = []
    with torch.no_grad():
        for inputs, _, _, _, _ in loader:
            selected = select_modality(inputs.to(device), modality)
            if autoencoder:
                scores = model.anomaly_score(selected)
            else:
                scores = torch.softmax(model(selected), dim=1)[:, 1]
            output.extend(scores.float().cpu().tolist())
    return output


def train_deep_baseline(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    tensor_dir: Path,
    dataset: str,
    seed: int,
    modality: str,
    checkpoint_path: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    smoke_test: bool,
    shared_tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    family_mapping = build_family_mapping(train)
    datasets = [
        MultimodalDataset(
            frame,
            tensor_dir,
            family_mapping,
            shared_tensor_cache=shared_tensors,
            dataset_name=dataset,
        )
        for frame in (train, validation, test)
    ]
    loaders = [
        make_loader(
            data,
            batch_size=min(batch_size, 8) if smoke_test else batch_size,
            shuffle=index == 0,
            seed=seed + index,
            num_workers=0 if smoke_test else num_workers,
            pin_memory=device.type == "cuda",
        )
        for index, data in enumerate(datasets)
    ]
    train_loader, validation_loader, test_loader = loaders
    model = build_deep_baseline(name, modality).to(device)
    autoencoder = isinstance(model, HealthyAutoencoder)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_validation = -math.inf
    stale_epochs = 0

    for _ in range(1 if smoke_test else epochs):
        model.train()
        for inputs, labels, _, _, _ in train_loader:
            if autoencoder and not (labels == 0).any():
                continue
            selected = select_modality(inputs.to(device), modality)
            optimizer.zero_grad(set_to_none=True)
            if autoencoder:
                healthy_mask = (labels == 0).to(device)
                selected = selected[healthy_mask]
                loss = nn.functional.mse_loss(model(selected), selected)
            else:
                loss = nn.functional.cross_entropy(
                    model(selected), labels.to(device)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        validation_scores = _probabilities(
            model, validation_loader, device, modality, autoencoder
        )
        threshold = _recording_threshold(validation, validation_scores)
        validation_f1 = _metric_bundle(
            validation, validation_scores, threshold, dataset
        )["recording_macro_f1"]
        if validation_f1 > best_validation:
            best_validation = validation_f1
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    validation_scores = _probabilities(
        model, validation_loader, device, modality, autoencoder
    )
    threshold = _recording_threshold(validation, validation_scores)
    test_scores = _probabilities(
        model, test_loader, device, modality, autoencoder
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_name": name,
            "modality": modality,
            "decision_threshold": threshold,
            "pipeline_version": PIPELINE_VERSION,
        },
        checkpoint_path,
    )
    return {
        **{
            f"validation_{key}": value
            for key, value in _metric_bundle(
                validation, validation_scores, threshold, dataset
            ).items()
        },
        **_metric_bundle(test, test_scores, threshold, dataset),
    }


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """Train one baseline with no implicit train/validation/test subsampling."""
    if args.model not in BASELINE_NAMES:
        raise ValueError(f"Unknown baseline {args.model}")
    if bool(getattr(args, "require_cuda", False)) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run but is not available")
    set_seed(args.seed)
    processed_dir = Path(args.processed_dir)
    index = pd.read_csv(processed_dir / "windows_index.csv")
    train, validation, test = select_index_splits(index, args.smoke_test)
    train = select_label_budget(train, args.label_budget, args.seed)
    if validation.empty:
        raise ValueError("A validation split is required for threshold calibration")
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_path = checkpoint_dir / (
        f"{args.run_id}.joblib"
        if args.model in CLASSICAL_BASELINE_NAMES
        else f"{args.run_id}.pth"
    )
    shared_cache = getattr(args, "shared_tensor_cache", None)
    shared_tensors = shared_cache.tensors if shared_cache is not None else None
    common = (
        args.model,
        train,
        validation,
        test,
        processed_dir / "tensors",
        args.dataset,
        args.seed,
        args.modality,
        checkpoint_path,
    )
    if args.model in CLASSICAL_BASELINE_NAMES:
        metrics = train_classical_baseline(
            *common, shared_tensors=shared_tensors
        )
    else:
        metrics = train_deep_baseline(
            *common,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            smoke_test=args.smoke_test,
            shared_tensors=shared_tensors,
        )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "code_revision": current_git_revision(),
        "run_id": args.run_id,
        "paper_experiment": args.paper_experiment,
        "protocol": args.protocol,
        "fold_id": args.fold_id,
        "dataset": args.dataset,
        "model": args.model,
        "configuration": args.configuration,
        "seed": args.seed,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "N/A",
        "label_budget": args.label_budget,
        "modality": args.modality,
        "n_train_windows": len(train),
        "n_train_recordings": train[recording_column(train)].nunique(),
        "checkpoint_path": str(checkpoint_path),
        "processed_dir": str(processed_dir),
        "status": "COMPLETED",
        **metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", choices=BASELINE_NAMES, required=True)
    parser.add_argument("--modality", choices=("vibration", "current", "both"), default="vibration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-budget", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--checkpoint-dir", default="artifacts/checkpoints/baselines")
    parser.add_argument("--run-id", default="baseline_run")
    parser.add_argument("--paper-experiment", default="E1")
    parser.add_argument("--protocol", default="protocol")
    parser.add_argument("--fold-id", default="fold")
    parser.add_argument("--configuration", default="default")
    return parser


if __name__ == "__main__":
    print(run_baseline(build_parser().parse_args()))
