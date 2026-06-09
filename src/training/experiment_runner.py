"""Run the 6-loss by 2-gate multimodal experiment matrix."""

from __future__ import annotations

import argparse
import gc
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from src.training.experiment_config import LOSS_NAMES


PROTOCOLS = {
    "nln_emp": {
        "dataset": "nln_emp",
        "folder": "nln_emp/nln_emp_leave_one_speed_out",
    },
    "paderborn_artificial_to_natural": {
        "dataset": "paderborn",
        "folder": "paderborn/paderborn_artificial_to_natural",
    },
    "paderborn_condition_generalization": {
        "dataset": "paderborn",
        "folder": "paderborn/paderborn_condition_generalization",
    },
}
DEFAULT_SEEDS = (42, 123, 999, 7, 88)


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
    key_columns: tuple[str, ...] = ("protocol", "experiment", "seed"),
) -> None:
    """Atomically insert or replace one seed-level result."""
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
    output_path: Path, protocol: str, experiment: str, seed: int
) -> bool:
    """Check whether one exact configuration/seed is already banked."""
    if not output_path.exists():
        return False
    existing = pd.read_csv(output_path)
    required = {"protocol", "experiment", "seed", "status"}
    if not required.issubset(existing.columns):
        return False
    match = existing[
        (existing["protocol"].astype(str) == protocol)
        & (existing["experiment"].astype(str) == experiment)
        & (existing["seed"].astype(str) == str(seed))
    ]
    return bool((match["status"] == "COMPLETED").any())


def run_protocol_matrix(args: argparse.Namespace, protocol: str) -> None:
    """Execute and bank every matrix cell for one cached protocol."""
    import torch

    from src.training.train_multimodal import (
        load_protocol_tensor_cache,
        train_multimodal,
    )

    protocol_config = PROTOCOLS[protocol]
    output_path = Path(args.output_file)
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
    processed_dir = Path(args.data_root) / protocol_config["folder"]
    total_jobs = len(experiments) * len(seeds)
    job_number = 0

    print(
        f"Matrix: {len(experiments)} configurations x {len(seeds)} seeds "
        f"= {total_jobs} jobs"
    )
    print(f"Protocol: {protocol} | data: {processed_dir}")

    pending_jobs = [
        (experiment, seed)
        for experiment in experiments
        for seed in seeds
        if not is_completed(
            output_path, protocol, experiment["experiment"], seed
        )
    ]
    if not pending_jobs:
        print("All requested jobs are already complete; no tensor cache needed.")
        return

    shared_cache = None
    if getattr(args, "preload", True):
        shared_cache = load_protocol_tensor_cache(
            processed_dir=processed_dir,
            maximum_cache_gb=getattr(args, "cache_max_gb", 36.0),
            smoke_test=args.smoke_test,
        )
        print(
            f"Shared cache ready: {len(shared_cache.tensors)} tensors, "
            f"splits={shared_cache.cached_splits}"
        )

    try:
        for experiment in experiments:
            for seed in seeds:
                job_number += 1
                experiment_name = experiment["experiment"]
                if is_completed(output_path, protocol, experiment_name, seed):
                    print(
                        f"[{job_number}/{total_jobs}] skipping completed "
                        f"{experiment_name}, seed={seed}"
                    )
                    continue

                run_id = f"{protocol}_{experiment_name}_seed{seed}"
                print(
                    f"[{job_number}/{total_jobs}] running {experiment_name}, seed={seed}"
                )
                training_args = argparse.Namespace(
                    processed_dir=str(processed_dir),
                    dataset=protocol_config["dataset"],
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
                            "status": "COMPLETED",
                        }
                    )
                    bank_result(result, output_path)
                    print(f"Banked {run_id}")
                except Exception as error:
                    failure = {
                        "run_id": run_id,
                        "protocol": protocol,
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
            print(f"Released shared protocol cache ({cached_count} tensors).")


def run_experiment_matrix(args: argparse.Namespace) -> None:
    """Run protocols sequentially, releasing each RAM cache before the next."""
    protocols = getattr(args, "protocols", None) or [args.protocol]
    for protocol in protocols:
        run_protocol_matrix(args, protocol)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="nln_emp")
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=PROTOCOLS,
        help="Run several protocols sequentially with separate RAM caches.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
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
        default="results/tables/loss_gate_matrix_results.csv",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="artifacts/checkpoints/loss_gate_matrix",
    )
    return parser


if __name__ == "__main__":
    run_experiment_matrix(build_parser().parse_args())
