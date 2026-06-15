"""Train the multimodal model with severity-aware curriculum learning."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from functools import lru_cache
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (  # noqa: E402
    aggregate_recording_predictions,
    compute_binary_metrics,
    format_optional_metric,
    is_early_fault,
    select_decision_threshold,
)
from src.models.multimodal_cross_attention import MultimodalMotorModel  # noqa: E402
from src.training.losses import LOSS_NAMES, build_health_loss  # noqa: E402
from src.training.data_selection import select_label_budget  # noqa: E402


DEFAULT_MODALITY_DROPOUT = 0.2
PIPELINE_VERSION = "corrected_multimodal_v3"


@lru_cache(maxsize=1)
def current_git_revision() -> str:
    """Return the checked-out source revision for result provenance."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass
class ProtocolTensorCache:
    """Read-only CPU tensor cache shared by every run in one protocol."""

    tensors: dict[str, torch.Tensor]
    cached_splits: tuple[str, ...]
    estimated_gb: float

    def clear(self) -> None:
        """Release references so Python can return the protocol cache memory."""
        self.tensors.clear()


def set_seed(seed: int) -> None:
    """Set all random generators used by a training run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed each DataLoader worker from PyTorch's deterministic worker seed."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class MultimodalDataset(Dataset):
    """Load cached two-channel spectrograms and their paper labels."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tensor_dir: Path,
        family_to_index: dict[str, int],
        preload: bool = False,
        shared_tensor_cache: Optional[Mapping[str, torch.Tensor]] = None,
        dataset_name: Optional[str] = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.tensor_dir = Path(tensor_dir)
        self.family_to_index = family_to_index
        self.shared_tensor_cache = shared_tensor_cache
        self.dataset_name = dataset_name
        aggregation_column = (
            "base_recording_id"
            if "base_recording_id" in self.dataframe.columns
            else "recording_id"
        )
        self.recording_ids = self.dataframe[aggregation_column].astype(str).tolist()
        self.health_labels = [
            1 if "fault" in str(label).lower() else 0
            for label in self.dataframe["health_label"]
        ]

        if "severity" in self.dataframe.columns:
            severities = self.dataframe["severity"].fillna("").tolist()
        else:
            severities = [""] * len(self.dataframe)
        self.early_fault_labels = [
            is_early_fault(severity, health, self.dataset_name)
            for severity, health in zip(
                severities, self.dataframe["health_label"].tolist()
            )
        ]

        families = self.dataframe.get(
            "fault_family", pd.Series(["unknown"] * len(self.dataframe))
        )
        self.family_labels = [
            family_to_index.get(str(family), 0) for family in families
        ]

        self.tensors: Optional[list[torch.Tensor]] = None
        if preload:
            print(f"Loading {len(self.dataframe)} tensors into system RAM...")
            self.tensors = [
                torch.load(
                    self.tensor_dir / tensor_id,
                    map_location="cpu",
                    weights_only=True,
                )
                for tensor_id in tqdm(
                    self.dataframe["tensor_id"], desc="Caching tensors"
                )
            ]

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, int, int, bool, str]:
        tensor_id = str(self.dataframe.iloc[index]["tensor_id"])
        if self.shared_tensor_cache is not None and tensor_id in self.shared_tensor_cache:
            tensor = self.shared_tensor_cache[tensor_id]
        elif self.tensors is None:
            tensor = torch.load(
                self.tensor_dir / tensor_id,
                map_location="cpu",
                weights_only=True,
            )
        else:
            tensor = self.tensors[index]
        return (
            tensor.float(),
            self.health_labels[index],
            self.family_labels[index],
            self.early_fault_labels[index],
            self.recording_ids[index],
        )


def select_index_splits(
    index_dataframe: pd.DataFrame, smoke_test: bool
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select train, validation, and test rows consistently for cache and training."""
    train_dataframe = index_dataframe[index_dataframe["split"] == "train"]
    validation_dataframe = index_dataframe[index_dataframe["split"] == "val"]
    test_dataframe = index_dataframe[index_dataframe["split"] == "test"]
    if min(len(train_dataframe), len(test_dataframe)) == 0:
        raise ValueError("Both train and test splits must contain samples")

    if smoke_test:
        train_dataframe = _select_smoke_rows(train_dataframe)
        validation_dataframe = _select_smoke_rows(validation_dataframe)
        test_dataframe = _select_smoke_rows(test_dataframe)
    return train_dataframe, validation_dataframe, test_dataframe


def _select_smoke_rows(
    dataframe: pd.DataFrame, maximum_rows: int = 64
) -> pd.DataFrame:
    """Select a small deterministic subset spanning classes and recordings."""
    if len(dataframe) <= maximum_rows:
        return dataframe.copy()

    recording_column = (
        "base_recording_id"
        if "base_recording_id" in dataframe.columns
        else "recording_id"
    )
    labels = sorted(dataframe["health_label"].astype(str).unique())
    rows_per_label = max(1, maximum_rows // len(labels))
    selected_indices: list[Any] = []

    for label in labels:
        label_rows = dataframe[
            dataframe["health_label"].astype(str) == label
        ]
        diverse_rows = label_rows.drop_duplicates(recording_column)
        label_indices = list(diverse_rows.head(rows_per_label).index)
        if len(label_indices) < rows_per_label:
            remaining = label_rows[~label_rows.index.isin(label_indices)]
            label_indices.extend(
                remaining.head(rows_per_label - len(label_indices)).index
            )
        selected_indices.extend(label_indices)

    if len(selected_indices) < maximum_rows:
        remaining = dataframe[~dataframe.index.isin(selected_indices)]
        selected_indices.extend(
            remaining.head(maximum_rows - len(selected_indices)).index
        )

    return dataframe.loc[selected_indices[:maximum_rows]].copy()


def load_protocol_tensor_cache(
    processed_dir: Path,
    maximum_cache_gb: float = 36.0,
    smoke_test: bool = False,
) -> ProtocolTensorCache:
    """Cache complete splits in RAM once, prioritizing train then val then test."""
    processed_dir = Path(processed_dir)
    index_path = processed_dir / "windows_index.csv"
    tensor_dir = processed_dir / "tensors"
    if not index_path.exists():
        raise FileNotFoundError(f"Window index not found: {index_path}")
    if not tensor_dir.exists():
        raise FileNotFoundError(f"Tensor directory not found: {tensor_dir}")
    if maximum_cache_gb <= 0:
        return ProtocolTensorCache({}, (), 0.0)

    index_dataframe = pd.read_csv(index_path)
    split_frames = dict(
        zip(
            ("train", "val", "test"),
            select_index_splits(index_dataframe, smoke_test),
        )
    )
    first_non_empty = next(
        (frame for frame in split_frames.values() if len(frame)), None
    )
    if first_non_empty is None:
        raise ValueError("No tensors are available to cache")

    sample_id = str(first_non_empty.iloc[0]["tensor_id"])
    sample = torch.load(
        tensor_dir / sample_id, map_location="cpu", weights_only=True
    )
    bytes_per_tensor = sample.numel() * sample.element_size()

    maximum_bytes = int(maximum_cache_gb * (1024**3))
    selected_splits: list[str] = []
    estimated_bytes = 0
    for split_name in ("train", "val", "test"):
        split_bytes = len(split_frames[split_name]) * bytes_per_tensor
        if estimated_bytes + split_bytes <= maximum_bytes:
            selected_splits.append(split_name)
            estimated_bytes += split_bytes
        else:
            print(
                f"RAM cache: leaving {split_name} disk-backed "
                f"(estimated {split_bytes / (1024**3):.2f} GB)"
            )

    if "train" not in selected_splits:
        raise MemoryError(
            f"Training split needs about "
            f"{len(split_frames['train']) * bytes_per_tensor / (1024**3):.2f} GB, "
            f"above --cache-max-gb={maximum_cache_gb:.2f}"
        )

    rows_to_cache = pd.concat(
        [split_frames[name] for name in selected_splits], ignore_index=True
    )
    tensors: dict[str, torch.Tensor] = {sample_id: sample}
    print(
        f"Caching {len(rows_to_cache)} tensors once for splits "
        f"{selected_splits} (estimated {estimated_bytes / (1024**3):.2f} GB)..."
    )
    for tensor_id in tqdm(rows_to_cache["tensor_id"], desc="Protocol RAM cache"):
        key = str(tensor_id)
        if key not in tensors:
            tensors[key] = torch.load(
                tensor_dir / key, map_location="cpu", weights_only=True
            )

    return ProtocolTensorCache(
        tensors=tensors,
        cached_splits=tuple(selected_splits),
        estimated_gb=estimated_bytes / (1024**3),
    )


def build_family_mapping(train_dataframe: pd.DataFrame) -> dict[str, int]:
    """Create a stable auxiliary-label mapping from training metadata only."""
    if "fault_family" not in train_dataframe.columns:
        return {"unknown": 0}
    families = sorted(
        {
            str(value)
            for value in train_dataframe["fault_family"].dropna().tolist()
            if str(value).strip() and str(value).strip().lower() != "unknown"
        }
    )
    return {"unknown": 0, **{family: index + 1 for index, family in enumerate(families)}}


def select_curriculum_stage_two_rows(
    train_dataframe: pd.DataFrame, dataset_name: str
) -> pd.DataFrame:
    """Keep healthy samples and explicitly annotated lowest-severity faults."""
    if "severity" not in train_dataframe.columns:
        return train_dataframe.iloc[0:0].copy()

    early_mask = pd.Series(
        [
            is_early_fault(severity, health, dataset_name)
            for severity, health in zip(
                train_dataframe["severity"].fillna(""),
                train_dataframe["health_label"],
            )
        ],
        index=train_dataframe.index,
    )
    healthy_mask = (
        train_dataframe["health_label"].astype(str).str.strip().str.lower()
        == "healthy"
    )
    return train_dataframe[healthy_mask | early_mask].copy()


def build_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    minimum_learning_rate_ratio: float,
) -> optim.lr_scheduler.LambdaLR:
    """Create a linear-warmup then cosine-decay learning-rate scheduler."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def learning_rate_multiplier(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps == warmup_steps:
            return minimum_learning_rate_ratio
        progress = min(
            1.0, (step - warmup_steps) / float(total_steps - warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_learning_rate_ratio + (
            (1.0 - minimum_learning_rate_ratio) * cosine
        )

    return optim.lr_scheduler.LambdaLR(optimizer, learning_rate_multiplier)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Build a deterministic DataLoader."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker if num_workers else None,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def evaluate_model(
    model: MultimodalMotorModel,
    data_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    decision_threshold: float = 0.5,
    amp_dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """Evaluate window- and recording-level metrics at a fixed threshold."""
    model.eval()
    targets: list[int] = []
    probabilities: list[float] = []
    early_mask: list[bool] = []
    recording_ids: list[str] = []
    gate_values: list[float] = []

    with torch.no_grad():
        for batch_x, batch_health, _, batch_early, batch_recording_ids in data_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                health_logits, _ = model(batch_x)
            fault_probability = torch.softmax(health_logits.float(), dim=1)[:, 1]

            targets.extend(batch_health.numpy().tolist())
            probabilities.extend(fault_probability.cpu().numpy().tolist())
            early_mask.extend(batch_early.numpy().astype(bool).tolist())
            recording_ids.extend(map(str, batch_recording_ids))
            if model.last_current_gate is not None:
                gate_values.extend(
                    model.last_current_gate.flatten().cpu().numpy().tolist()
                )

    predictions = [
        int(probability >= decision_threshold) for probability in probabilities
    ]
    metrics = compute_binary_metrics(
        targets,
        predictions,
        fault_probabilities=probabilities,
        early_fault_mask=early_mask,
    )
    recording_targets, recording_probabilities, recording_early = (
        aggregate_recording_predictions(
            recording_ids, targets, probabilities, early_mask
        )
    )
    recording_predictions = [
        int(probability >= decision_threshold)
        for probability in recording_probabilities
    ]
    recording_metrics = compute_binary_metrics(
        recording_targets,
        recording_predictions,
        fault_probabilities=recording_probabilities,
        early_fault_mask=recording_early,
    )
    metrics.update(
        {
            f"recording_{name}": value
            for name, value in recording_metrics.items()
        }
    )
    metrics["decision_threshold"] = float(decision_threshold)
    metrics["_targets"] = targets
    metrics["_probabilities"] = probabilities
    metrics["_recording_targets"] = recording_targets
    metrics["_recording_probabilities"] = recording_probabilities
    metrics["current_gate_mean"] = (
        float(np.mean(gate_values)) if gate_values else float("nan")
    )
    metrics["current_gate_std"] = (
        float(np.std(gate_values)) if gate_values else float("nan")
    )
    return metrics


def train_one_epoch(
    model: MultimodalMotorModel,
    data_loader: DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    stage_health_loss: Optional[nn.Module],
    family_loss: nn.Module,
    family_loss_weight: float,
    gradient_clip_norm: float,
    modality_dropout: float,
    scaler: Any,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """Train one epoch and return mean loss and maximum observed gradient norm."""
    model.train()
    running_loss = 0.0
    maximum_gradient_norm = 0.0

    for batch_x, batch_health, batch_family, batch_early, _ in data_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_health = batch_health.to(device, non_blocking=True)
        batch_family = batch_family.to(device, non_blocking=True)
        batch_early = batch_early.to(device, non_blocking=True)
        if model.ablation_mode is None and modality_dropout > 0:
            drop_samples = (
                torch.rand(batch_x.size(0), device=device) < modality_dropout
            )
            dropped_modality = torch.randint(
                0, 2, (batch_x.size(0),), device=device
            )
            if drop_samples.any():
                batch_indices = torch.arange(
                    batch_x.size(0), device=device
                )[drop_samples]
                batch_x[
                    batch_indices,
                    dropped_modality[drop_samples],
                    :,
                    :,
                ] = 0
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            health_logits, family_logits = model(batch_x)
            if stage_health_loss is None:
                health_loss = nn.functional.cross_entropy(
                    health_logits, batch_health
                )
            else:
                health_loss = stage_health_loss(
                    health_logits, batch_health, batch_early
                )
            auxiliary_loss = family_loss(family_logits, batch_family)
            loss = health_loss + family_loss_weight * auxiliary_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm
        )
        maximum_gradient_norm = max(
            maximum_gradient_norm, float(gradient_norm.detach().cpu())
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running_loss += float(loss.detach().cpu())

    return running_loss / max(1, len(data_loader)), maximum_gradient_norm


def append_result_csv(result: dict[str, Any], output_path: Path) -> None:
    """Append a result while safely expanding an older CSV schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([result])
    if output_path.exists():
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        combined = new_row
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    combined.to_csv(temporary_path, index=False, na_rep="N/A")
    temporary_path.replace(output_path)


def train_multimodal(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one configured training run and return its test metrics."""
    seed = int(getattr(args, "seed", 42))
    set_seed(seed)
    if bool(getattr(args, "require_cuda", False)) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run but is not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(getattr(args, "amp", True) and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if amp_enabled and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    processed_dir = Path(args.processed_dir)
    index_path = processed_dir / "windows_index.csv"
    tensor_dir = processed_dir / "tensors"
    if not index_path.exists():
        raise FileNotFoundError(f"Window index not found: {index_path}")
    if not tensor_dir.exists():
        raise FileNotFoundError(f"Tensor directory not found: {tensor_dir}")

    smoke_test = bool(getattr(args, "smoke_test", False))
    index_dataframe = pd.read_csv(index_path)
    train_dataframe, validation_dataframe, test_dataframe = select_index_splits(
        index_dataframe, smoke_test
    )
    label_budget = float(getattr(args, "label_budget", 1.0))
    train_dataframe = select_label_budget(
        train_dataframe, label_budget, seed
    )

    family_to_index = build_family_mapping(train_dataframe)
    use_gate = bool(getattr(args, "use_modality_gate", False))
    model = MultimodalMotorModel(
        num_fault_families=len(family_to_index),
        ablation_mode=getattr(args, "ablation", None),
        use_modality_gate=use_gate,
    ).to(device)

    batch_size = int(getattr(args, "batch_size", 128))
    if smoke_test:
        batch_size = min(batch_size, 16)
    num_workers = int(getattr(args, "num_workers", 2)) if not smoke_test else 0
    preload = bool(getattr(args, "preload", not smoke_test))
    shared_cache = getattr(args, "shared_tensor_cache", None)
    shared_tensors = shared_cache.tensors if shared_cache is not None else None
    dataset_name = str(getattr(args, "dataset", ""))

    train_dataset = MultimodalDataset(
        train_dataframe,
        tensor_dir,
        family_to_index,
        preload=preload and shared_tensors is None,
        shared_tensor_cache=shared_tensors,
        dataset_name=dataset_name,
    )
    stage_two_dataframe = select_curriculum_stage_two_rows(
        train_dataframe, dataset_name
    )
    stage_two_has_early_faults = bool(
        len(stage_two_dataframe)
        and any(
            is_early_fault(severity, health, dataset_name)
            for severity, health in zip(
                stage_two_dataframe.get(
                    "severity", pd.Series(dtype=str)
                ).fillna(""),
                stage_two_dataframe["health_label"],
            )
        )
    )
    stage_two_dataset = (
        MultimodalDataset(
            stage_two_dataframe,
            tensor_dir,
            family_to_index,
            preload=False,
            shared_tensor_cache=shared_tensors,
            dataset_name=dataset_name,
        )
        if stage_two_has_early_faults
        else None
    )
    validation_dataset = MultimodalDataset(
        validation_dataframe,
        tensor_dir,
        family_to_index,
        preload=False,
        shared_tensor_cache=shared_tensors,
        dataset_name=dataset_name,
    )
    test_dataset = MultimodalDataset(
        test_dataframe,
        tensor_dir,
        family_to_index,
        preload=False,
        shared_tensor_cache=shared_tensors,
        dataset_name=dataset_name,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    stage_two_loader = (
        make_loader(
            stage_two_dataset,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + 1,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        if stage_two_dataset is not None
        else None
    )
    validation_loader = (
        make_loader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        if len(validation_dataset)
        else None
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    stage1_epochs = 1 if smoke_test else int(getattr(args, "stage1_epochs", 10))
    stage2_epochs = (
        1
        if smoke_test and getattr(args, "use_curriculum", True)
        else int(getattr(args, "stage2_epochs", 5))
    )
    if not bool(getattr(args, "use_curriculum", True)):
        stage2_epochs = 0
    elif stage_two_loader is None:
        print(
            f"Skipping Stage 2 for {dataset_name}: no valid lowest-severity "
            "fault annotations are available."
        )
        stage2_epochs = 0

    learning_rate = float(getattr(args, "learning_rate", 1e-3))
    minimum_learning_rate = float(getattr(args, "minimum_learning_rate", 1e-5))
    modality_dropout = float(
        getattr(args, "modality_dropout", DEFAULT_MODALITY_DROPOUT)
    )
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 <= minimum_learning_rate <= learning_rate:
        raise ValueError(
            "minimum_learning_rate must be between 0 and learning_rate"
        )
    if not 0 <= modality_dropout < 1:
        raise ValueError("modality_dropout must be in [0, 1)")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(getattr(args, "weight_decay", 1e-4)),
    )
    total_steps = (
        len(train_loader) * stage1_epochs
        + (len(stage_two_loader) * stage2_epochs if stage_two_loader else 0)
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_ratio=float(getattr(args, "warmup_ratio", 0.1)),
        minimum_learning_rate_ratio=minimum_learning_rate / learning_rate,
    )
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(
            "cuda", enabled=amp_enabled and amp_dtype == torch.float16
        )
    else:
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled and amp_dtype == torch.float16
        )
    family_loss = nn.CrossEntropyLoss()
    configured_health_loss = build_health_loss(args.loss_name)

    print(
        f"Training on {device} | seed={seed} | loss={args.loss_name} | "
        f"gate={use_gate} | amp={str(amp_dtype).removeprefix('torch.')} | "
        f"AdamW decay={getattr(args, 'weight_decay', 1e-4)}"
    )

    best_state = copy.deepcopy(model.state_dict())
    best_validation_f1 = -math.inf
    history: list[dict[str, Any]] = []
    total_epochs = stage1_epochs + stage2_epochs
    for epoch in range(total_epochs):
        in_stage_two = epoch >= stage1_epochs
        stage_name = "stage2_early_focus" if in_stage_two else "stage1_general"
        active_health_loss = configured_health_loss if in_stage_two else None
        active_loader = stage_two_loader if in_stage_two else train_loader
        if active_loader is None:
            raise RuntimeError("Stage 2 was scheduled without a valid data loader")
        epoch_loss, maximum_gradient_norm = train_one_epoch(
            model=model,
            data_loader=active_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            stage_health_loss=active_health_loss,
            family_loss=family_loss,
            family_loss_weight=float(getattr(args, "family_loss_weight", 0.5)),
            gradient_clip_norm=float(getattr(args, "gradient_clip_norm", 1.0)),
            modality_dropout=modality_dropout,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )

        validation_f1 = float("nan")
        if validation_loader is not None:
            validation_metrics = evaluate_model(
                model,
                validation_loader,
                device,
                amp_enabled,
                amp_dtype=amp_dtype,
            )
            validation_f1 = validation_metrics["recording_macro_f1"]
            if validation_f1 > best_validation_f1:
                best_validation_f1 = validation_f1
                best_state = copy.deepcopy(model.state_dict())
        else:
            best_state = copy.deepcopy(model.state_dict())

        current_lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch + 1,
                "stage": stage_name,
                "loss": epoch_loss,
                "validation_macro_f1": validation_f1,
                "learning_rate": current_lr,
                "maximum_gradient_norm_before_clip": maximum_gradient_norm,
            }
        )
        print(
            f"{stage_name} epoch {epoch + 1}/{total_epochs} | "
            f"loss={epoch_loss:.4f} | val_f1={format_optional_metric(validation_f1)} "
            f"| lr={current_lr:.7f} | grad_norm={maximum_gradient_norm:.3f}"
        )

    model.load_state_dict(best_state)
    decision_threshold = 0.5
    calibrated_validation_metrics: dict[str, Any] = {}
    if validation_loader is not None:
        calibration = evaluate_model(
            model,
            validation_loader,
            device,
            amp_enabled,
            decision_threshold=0.5,
            amp_dtype=amp_dtype,
        )
        decision_threshold = select_decision_threshold(
            calibration["_recording_targets"],
            calibration["_recording_probabilities"],
        )
        calibrated_validation_metrics = evaluate_model(
            model,
            validation_loader,
            device,
            amp_enabled,
            decision_threshold=decision_threshold,
            amp_dtype=amp_dtype,
        )
        for private_key in (
            "_targets",
            "_probabilities",
            "_recording_targets",
            "_recording_probabilities",
        ):
            calibrated_validation_metrics.pop(private_key, None)
    test_metrics = evaluate_model(
        model,
        test_loader,
        device,
        amp_enabled,
        decision_threshold=decision_threshold,
        amp_dtype=amp_dtype,
    )
    for private_key in (
        "_targets",
        "_probabilities",
        "_recording_targets",
        "_recording_probabilities",
    ):
        test_metrics.pop(private_key, None)
    ablation = getattr(args, "ablation", None) or "fusion"
    run_id = str(
        getattr(
            args,
            "run_id",
            f"{getattr(args, 'dataset', 'dataset')}_{ablation}_{args.loss_name}_"
            f"gate{int(use_gate)}_seed{seed}",
        )
    )
    result = {
        "pipeline_version": PIPELINE_VERSION,
        "code_revision": current_git_revision(),
        "run_id": run_id,
        "paper_experiment": getattr(args, "paper_experiment", "matrix"),
        "protocol": getattr(args, "protocol", dataset_name),
        "model": "proposed",
        "configuration": getattr(
            args, "configuration", getattr(args, "experiment", run_id)
        ),
        "label_budget": label_budget,
        "experiment": getattr(args, "experiment", run_id),
        "dataset": getattr(args, "dataset", "unknown"),
        "fold_id": getattr(args, "fold_id", ""),
        "processed_dir": str(processed_dir),
        "seed": seed,
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "N/A"
        ),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda or "N/A",
        "amp_dtype": (
            str(amp_dtype).removeprefix("torch.") if amp_enabled else "disabled"
        ),
        "ablation": ablation,
        "curriculum_requested": bool(getattr(args, "use_curriculum", True)),
        "curriculum": stage2_epochs > 0,
        "loss_name": args.loss_name,
        "modality_gate": use_gate,
        "stage1_epochs": stage1_epochs,
        "stage2_epochs": stage2_epochs,
        "stage2_samples": len(stage_two_dataframe) if stage2_epochs else 0,
        "n_train_windows": len(train_dataframe),
        "learning_rate": learning_rate,
        "weight_decay": float(getattr(args, "weight_decay", 1e-4)),
        "gradient_clip_norm": float(getattr(args, "gradient_clip_norm", 1.0)),
        "modality_dropout": modality_dropout,
        "warmup_ratio": float(getattr(args, "warmup_ratio", 0.1)),
        "shared_cache_splits": (
            ",".join(shared_cache.cached_splits) if shared_cache is not None else ""
        ),
        "shared_cache_estimated_gb": (
            shared_cache.estimated_gb if shared_cache is not None else 0.0
        ),
        **{
            f"validation_{name}": value
            for name, value in calibrated_validation_metrics.items()
        },
        **test_metrics,
    }

    checkpoint_dir = Path(
        getattr(args, "checkpoint_dir", "artifacts/checkpoints/loss_gate_matrix")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{run_id}.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metrics": test_metrics,
            "history": history,
            "family_to_index": family_to_index,
            "model_config": {
                "embed_dim": model.embed_dim,
                "num_fault_families": len(family_to_index),
                "ablation_mode": getattr(args, "ablation", None),
                "use_modality_gate": use_gate,
                "num_attention_heads": model.num_attention_heads,
                "num_attention_blocks": model.num_attention_blocks,
                "dropout": model.dropout,
            },
            "decision_threshold": decision_threshold,
            "training_config": result,
        },
        checkpoint_path,
    )
    result["checkpoint_path"] = str(checkpoint_path)

    if bool(getattr(args, "write_detailed_metrics", True)):
        append_result_csv(
            result,
            Path(
                getattr(
                    args,
                    "metrics_file",
                    "results/tables/corrected_detailed_metrics.csv",
                )
            ),
        )

    print(
        f"Test Macro F1={result['macro_f1']:.4f} | "
        f"Balanced Acc={result['balanced_acc']:.4f} | "
        f"Early Recall={format_optional_metric(result['early_fault_recall'])}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--dataset", default="nln_emp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loss-name", choices=LOSS_NAMES, default="ce_1.0")
    parser.add_argument("--use-modality-gate", action="store_true")
    parser.add_argument(
        "--use-curriculum", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--ablation", choices=["vibration_only", "current_only"], default=None
    )
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--modality-dropout",
        type=float,
        default=DEFAULT_MODALITY_DROPOUT,
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--family-loss-weight", type=float, default=0.5)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/checkpoints/loss_gate_matrix",
    )
    parser.add_argument(
        "--metrics-file", default="results/tables/corrected_detailed_metrics.csv"
    )
    return parser


if __name__ == "__main__":
    os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    train_multimodal(build_parser().parse_args())
