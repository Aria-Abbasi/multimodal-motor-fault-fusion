"""Run the 6-loss by 2-gate multimodal experiment matrix."""

from __future__ import annotations

import argparse
import gc
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from src.training.experiment_config import LOSS_NAMES
from src.training.train_multimodal import PIPELINE_VERSION


PROTOCOLS = {
    "nln_emp": {
        "dataset": "nln_emp",
        "folder": "nln_emp/nln_emp_leave_one_speed_out",
        "split_file": "nln_emp_leave_one_speed_out.csv",
    },
    "paderborn_artificial_to_natural": {
        "dataset": "paderborn",
        "folder": "paderborn/paderborn_artificial_to_natural",
        "split_file": "paderborn_artificial_to_natural.csv",
    },
    "paderborn_condition_generalization": {
        "dataset": "paderborn",
        "folder": "paderborn/paderborn_condition_generalization",
        "split_file": "paderborn_condition_generalization.csv",
    },
    "cwru": {
        "dataset": "cwru",
        "folder": "cwru/cwru_leave_one_load_out",
        "split_file": "cwru_leave_one_load_out.csv",
    },
}
DEFAULT_SEEDS = (42, 123, 999, 7, 88)
SUMMARY_METRICS = (
    "macro_f1",
    "recording_macro_f1",
    "validation_macro_f1",
    "validation_recording_macro_f1",
    "balanced_acc",
    "recording_balanced_acc",
    "accuracy",
    "early_fault_recall",
    "auroc",
    "auprc",
    "mcc",
    "current_gate_mean",
)


def build_experiment_matrix(
    losses: tuple[str, ...] = LOSS_NAMES,
) -> list[dict[str, Any]]:
    """Return the 12 unique loss/gate configurations."""
    experiments = []
    for use_gate in (False, True):
        gate_name = "gate_on" if use_gate else "gate_off"
        for loss_name in losses:
            experiments.append(
                {
                    "experiment": f"fusion_{gate_name}_{loss_name.replace('.', 'p')}",
                    "loss_name": loss_name,
                    "use_modality_gate": use_gate,
                }
            )
    return experiments


def bank_result(
    result: dict[str, Any],
    output_path: Path,
    key_columns: tuple[str, ...] = (
        "protocol",
        "fold_id",
        "experiment",
        "seed",
    ),
) -> None:
    """Atomically insert or replace one fold-and-seed-level result."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([result])
    if output_path.exists():
        existing = pd.read_csv(output_path)
        if all(column in existing.columns for column in key_columns):
            keep = pd.Series(True, index=existing.index)
            for column in key_columns:
                keep &= existing[column].astype(str) == str(result[column])
            existing = existing[~keep]
        combined = pd.concat([existing, new_row], ignore_index=True, sort=False)
    else:
        combined = new_row

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    combined.to_csv(temporary_path, index=False, na_rep="N/A")
    temporary_path.replace(output_path)


def is_completed(
    output_path: Path,
    protocol: str,
    fold_id: str,
    experiment: str,
    seed: int,
) -> bool:
    """Check whether one exact fold/configuration/seed is already banked."""
    if not output_path.exists():
        return False
    existing = pd.read_csv(output_path)
    required = {"protocol", "fold_id", "experiment", "seed", "status"}
    if not required.issubset(existing.columns):
        return False
    match = existing[
        (existing["protocol"].astype(str) == protocol)
        & (existing["fold_id"].astype(str) == fold_id)
        & (existing["experiment"].astype(str) == experiment)
        & (existing["seed"].astype(str) == str(seed))
    ]
    if "pipeline_version" in match.columns:
        match = match[
            match["pipeline_version"].astype(str) == PIPELINE_VERSION
        ]
    return bool((match["status"] == "COMPLETED").any())


def expected_fold_ids(split_root: Path, protocol: str) -> tuple[str, ...]:
    """Read the authoritative fold list generated at recording level."""
    split_path = split_root / PROTOCOLS[protocol]["split_file"]
    if not split_path.exists():
        raise FileNotFoundError(f"Protocol split file not found: {split_path}")
    split = pd.read_csv(split_path, usecols=["fold_id"])
    folds = tuple(sorted(split["fold_id"].dropna().astype(str).unique()))
    if not folds:
        raise ValueError(f"No fold_id values found in {split_path}")
    return folds


def discover_processed_folds(
    data_root: Path,
    split_root: Path,
    protocol: str,
    requested_folds: tuple[str, ...] = (),
    allow_partial: bool = False,
) -> list[tuple[str, Path]]:
    """Resolve fold directories and reject incomplete cross-validation by default."""
    protocol_root = data_root / PROTOCOLS[protocol]["folder"]
    expected = expected_fold_ids(split_root, protocol)
    selected = requested_folds or expected
    unknown = set(selected) - set(expected)
    if unknown:
        raise ValueError(
            f"Unknown folds for {protocol}: {sorted(unknown)}. "
            f"Expected: {list(expected)}"
        )

    discovered = {
        path.name: path
        for path in protocol_root.iterdir()
        if path.is_dir()
        and (path / "windows_index.csv").exists()
        and (path / "tensors").is_dir()
    } if protocol_root.is_dir() else {}

    if (
        (protocol_root / "windows_index.csv").exists()
        and (protocol_root / "tensors").is_dir()
    ):
        legacy_fold = expected[0]
        index = pd.read_csv(protocol_root / "windows_index.csv", nrows=1)
        if "fold_id" in index.columns and len(index):
            value = str(index["fold_id"].iloc[0])
            if value and value.lower() != "nan":
                legacy_fold = value
        discovered.setdefault(legacy_fold, protocol_root)

    missing = [fold for fold in selected if fold not in discovered]
    if missing and not allow_partial:
        raise FileNotFoundError(
            f"Missing processed folds for {protocol}: {missing}. "
            "Preprocess all folds or pass --allow-partial-folds explicitly."
        )
    resolved = [
        (fold, discovered[fold])
        for fold in selected
        if fold in discovered
    ]
    if not resolved:
        raise FileNotFoundError(
            f"No processed fold tensors found under {protocol_root}"
        )
    return resolved


def default_summary_path(output_path: Path) -> Path:
    """Return a predictable aggregate-results path beside the raw result bank."""
    return output_path.with_name(f"{output_path.stem}_summary.csv")


def write_aggregate_summary(
    output_path: Path,
    summary_path: Path,
) -> pd.DataFrame:
    """Average completed metrics equally across folds and seeds."""
    if not output_path.exists():
        return pd.DataFrame()
    results = pd.read_csv(output_path)
    required = {
        "protocol",
        "fold_id",
        "experiment",
        "seed",
        "status",
    }
    if not required.issubset(results.columns):
        return pd.DataFrame()
    completed = results[results["status"] == "COMPLETED"].copy()
    if completed.empty:
        return pd.DataFrame()

    group_columns = [
        "protocol",
        "dataset",
        "experiment",
        "loss_name",
        "modality_gate",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in completed.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_folds": int(group["fold_id"].nunique()),
                "n_seeds": int(group["seed"].nunique()),
                "n_runs": int(len(group)),
            }
        )
        for metric in SUMMARY_METRICS:
            values = (
                pd.to_numeric(group[metric], errors="coerce").dropna()
                if metric in group.columns
                else pd.Series(dtype=float)
            )
            row[f"{metric}_mean"] = (
                float(values.mean()) if len(values) else float("nan")
            )
            row[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else float("nan")
            )
            if metric in group.columns:
                numeric_group = group.assign(
                    _metric=pd.to_numeric(group[metric], errors="coerce")
                ).dropna(subset=["_metric"])
                fold_means = numeric_group.groupby("fold_id")["_metric"].mean()
                within_fold_stds = (
                    numeric_group.groupby("fold_id")["_metric"]
                    .std(ddof=1)
                    .dropna()
                )
                row[f"{metric}_mean_of_fold_means"] = (
                    float(fold_means.mean())
                    if len(fold_means)
                    else float("nan")
                )
                row[f"{metric}_between_fold_std"] = (
                    float(fold_means.std(ddof=1))
                    if len(fold_means) > 1
                    else float("nan")
                )
                row[f"{metric}_mean_within_fold_seed_std"] = (
                    float(within_fold_stds.mean())
                    if len(within_fold_stds)
                    else float("nan")
                )
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["protocol", "experiment"]
    ).reset_index(drop=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    summary.to_csv(temporary_path, index=False, na_rep="N/A")
    temporary_path.replace(summary_path)
    return summary


def run_protocol_matrix(args: argparse.Namespace, protocol: str) -> None:
    """Execute and bank every matrix cell across all selected protocol folds."""
    import torch

    from src.training.train_multimodal import (
        load_protocol_tensor_cache,
        train_multimodal,
    )

    protocol_config = PROTOCOLS[protocol]
    output_path = Path(args.output_file)
    if output_path.exists():
        existing_columns = pd.read_csv(output_path, nrows=1).columns
        if "pipeline_version" not in existing_columns:
            raise ValueError(
                f"{output_path} contains legacy results. Archive or rename it; "
                "corrected runs must not be mixed with the old pipeline."
            )
    summary_path = Path(
        getattr(args, "summary_file", None) or default_summary_path(output_path)
    )
    seeds = tuple(args.seeds)
    if args.smoke_test:
        seeds = seeds[:1]

    requested_losses = tuple(args.losses) if args.losses else LOSS_NAMES
    if protocol_config["dataset"] == "paderborn":
        unsupported = set(requested_losses) - {"ce_1.0"}
        if args.losses and unsupported:
            raise ValueError(
                "Paderborn has no granular severity labels, so early-weight "
                "losses are not identifiable. Use --losses ce_1.0."
            )
        selected_losses = ("ce_1.0",)
        print(
            "Paderborn uses standard CE only; its bearing IDs are not severity "
            "labels. Gate off/on will still be evaluated."
        )
    else:
        selected_losses = requested_losses
    experiments = build_experiment_matrix(selected_losses)
    folds = discover_processed_folds(
        data_root=Path(args.data_root),
        split_root=Path(getattr(args, "split_root", "data/splits")),
        protocol=protocol,
        requested_folds=tuple(getattr(args, "folds", None) or ()),
        allow_partial=bool(getattr(args, "allow_partial_folds", False)),
    )
    total_jobs = len(folds) * len(experiments) * len(seeds)
    job_number = 0

    print(
        f"Matrix: {len(folds)} folds x {len(experiments)} configurations "
        f"x {len(seeds)} seeds "
        f"= {total_jobs} jobs"
    )
    print(f"Protocol: {protocol} | folds: {[fold for fold, _ in folds]}")

    for fold_id, processed_dir in folds:
        pending_jobs = [
            (experiment, seed)
            for experiment in experiments
            for seed in seeds
            if not is_completed(
                output_path,
                protocol,
                fold_id,
                experiment["experiment"],
                seed,
            )
        ]
        if not pending_jobs:
            job_number += len(experiments) * len(seeds)
            print(f"Fold {fold_id} is already complete; skipping its cache.")
            continue

        shared_cache = None
        if getattr(args, "preload", True):
            shared_cache = load_protocol_tensor_cache(
                processed_dir=processed_dir,
                maximum_cache_gb=getattr(args, "cache_max_gb", 36.0),
                smoke_test=args.smoke_test,
            )
            print(
                f"Fold {fold_id} cache ready: {len(shared_cache.tensors)} tensors, "
                f"splits={shared_cache.cached_splits}"
            )

        try:
            for experiment in experiments:
                for seed in seeds:
                    job_number += 1
                    experiment_name = experiment["experiment"]
                    if is_completed(
                        output_path,
                        protocol,
                        fold_id,
                        experiment_name,
                        seed,
                    ):
                        print(
                            f"[{job_number}/{total_jobs}] skipping completed "
                            f"{fold_id}/{experiment_name}, seed={seed}"
                        )
                        continue

                    run_id = (
                        f"{protocol}_{fold_id}_{experiment_name}_seed{seed}"
                    )
                    print(
                        f"[{job_number}/{total_jobs}] running "
                        f"{fold_id}/{experiment_name}, seed={seed}"
                    )
                    training_args = argparse.Namespace(
                        processed_dir=str(processed_dir),
                        dataset=protocol_config["dataset"],
                        fold_id=fold_id,
                        seed=seed,
                        loss_name=experiment["loss_name"],
                        use_modality_gate=experiment["use_modality_gate"],
                        use_curriculum=True,
                        ablation=None,
                        stage1_epochs=args.stage1_epochs,
                        stage2_epochs=args.stage2_epochs,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        learning_rate=args.learning_rate,
                        minimum_learning_rate=args.minimum_learning_rate,
                        weight_decay=args.weight_decay,
                        gradient_clip_norm=args.gradient_clip_norm,
                        modality_dropout=getattr(args, "modality_dropout", 0.2),
                        warmup_ratio=args.warmup_ratio,
                        family_loss_weight=args.family_loss_weight,
                        preload=False,
                        shared_tensor_cache=shared_cache,
                        amp=args.amp,
                        smoke_test=args.smoke_test,
                        checkpoint_dir=args.checkpoint_dir,
                        write_detailed_metrics=False,
                        run_id=run_id,
                        experiment=experiment_name,
                    )

                    try:
                        result = train_multimodal(training_args)
                        result.update(
                            {
                                "protocol": protocol,
                                "fold_id": fold_id,
                                "status": "COMPLETED",
                            }
                        )
                        bank_result(result, output_path)
                        write_aggregate_summary(output_path, summary_path)
                        print(f"Banked {run_id}")
                    except Exception as error:
                        failure = {
                            "run_id": run_id,
                            "protocol": protocol,
                            "fold_id": fold_id,
                            "experiment": experiment_name,
                            "dataset": protocol_config["dataset"],
                            "seed": seed,
                            "loss_name": experiment["loss_name"],
                            "modality_gate": experiment["use_modality_gate"],
                            "status": "FAILED",
                            "error": f"{type(error).__name__}: {error}",
                        }
                        bank_result(failure, output_path)
                        traceback.print_exc()
                        if args.fail_fast:
                            raise
                    finally:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
        finally:
            if shared_cache is not None:
                cached_count = len(shared_cache.tensors)
                shared_cache.clear()
                del shared_cache
                gc.collect()
                print(
                    f"Released fold {fold_id} cache ({cached_count} tensors)."
                )

    summary = write_aggregate_summary(output_path, summary_path)
    if not summary.empty:
        print(f"Wrote aggregate fold/seed summary: {summary_path}")


def run_experiment_matrix(args: argparse.Namespace) -> None:
    """Run protocols sequentially, releasing each RAM cache before the next."""
    protocols = getattr(args, "protocols", None) or [args.protocol]
    for protocol in protocols:
        run_protocol_matrix(args, protocol)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="nln_emp")
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=PROTOCOLS,
        help="Run several protocols sequentially with separate RAM caches.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--folds",
        nargs="+",
        help="Run only these fold IDs; all expected folds run by default.",
    )
    parser.add_argument(
        "--allow-partial-folds",
        action="store_true",
        help="Allow execution when one or more expected processed folds are absent.",
    )
    parser.add_argument("--losses", nargs="+", choices=LOSS_NAMES)
    parser.add_argument("--stage1-epochs", type=int, default=10)
    parser.add_argument("--stage2-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--modality-dropout", type=float, default=0.2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--family-loss-weight", type=float, default=0.5)
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cache-max-gb",
        type=float,
        default=36.0,
        help="Maximum CPU RAM used by the shared protocol tensor cache.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output-file",
        default="results/tables/corrected_loss_gate_matrix_results.csv",
    )
    parser.add_argument(
        "--summary-file",
        help="Aggregate mean/std CSV; defaults beside --output-file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/checkpoints/loss_gate_matrix",
    )
    return parser


if __name__ == "__main__":
    run_experiment_matrix(build_parser().parse_args())
