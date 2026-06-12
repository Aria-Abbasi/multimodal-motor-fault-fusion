"""Execute the complete E1-E7 paper experiment plan."""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from src.models.classical_baselines import CLASSICAL_BASELINE_NAMES
from src.models.deep_baselines import DEEP_BASELINE_NAMES
from src.training.baseline_runner import run_baseline
from src.training.experiment_runner import (
    DEFAULT_SEEDS,
    PROTOCOLS,
    discover_processed_folds,
)
from src.training.result_store import (
    bank_result,
    read_current_results,
    result_completed,
)
from src.training.losses import LOSS_NAMES
from src.training.train_multimodal import train_multimodal
from src.training.train_multimodal import load_protocol_tensor_cache


PAPER_EXPERIMENTS = ("E1", "E2", "E3", "E4", "E5", "E6", "E7")
BASELINE_MODELS = CLASSICAL_BASELINE_NAMES + DEEP_BASELINE_NAMES


@dataclass(frozen=True)
class PaperJob:
    paper_experiment: str
    protocol: str
    fold_id: str
    processed_dir: str
    dataset: str
    model: str
    configuration: str
    seed: int
    label_budget: float = 1.0
    modality: str = "both"
    ablation: str | None = None
    use_gate: bool = False
    use_curriculum: bool = True
    stage1_epochs: int = 10
    stage2_epochs: int = 5

    @property
    def run_id(self) -> str:
        budget = str(self.label_budget).replace(".", "p")
        return (
            f"{self.paper_experiment}_{self.protocol}_{self.fold_id}_"
            f"{self.model}_{self.configuration}_seed{self.seed}_budget{budget}"
        )

    def identity(self) -> dict[str, Any]:
        return {
            "paper_experiment": self.paper_experiment,
            "protocol": self.protocol,
            "fold_id": self.fold_id,
            "model": self.model,
            "configuration": self.configuration,
            "seed": self.seed,
            "label_budget": self.label_budget,
        }


def _protocol_folds(
    data_root: Path,
    split_root: Path,
    protocol: str,
    allow_partial: bool,
) -> list[tuple[str, Path]]:
    return discover_processed_folds(
        data_root=data_root,
        split_root=split_root,
        protocol=protocol,
        allow_partial=allow_partial,
    )


def build_paper_jobs(
    *,
    data_root: Path,
    split_root: Path,
    experiments: tuple[str, ...] = PAPER_EXPERIMENTS[:-1],
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    frozen_gate: bool = True,
    strongest_baseline: str = "cnn",
    allow_partial: bool = False,
) -> list[PaperJob]:
    """Build explicit jobs; E7 is inference-only and therefore omitted."""
    unknown = set(experiments) - set(PAPER_EXPERIMENTS)
    if unknown:
        raise ValueError(f"Unknown paper experiments: {sorted(unknown)}")
    if strongest_baseline not in BASELINE_MODELS:
        raise ValueError(f"Unknown strongest baseline: {strongest_baseline}")

    cache: dict[str, list[tuple[str, Path]]] = {}

    def folds(protocol: str) -> list[tuple[str, Path]]:
        if protocol not in cache:
            cache[protocol] = _protocol_folds(
                data_root, split_root, protocol, allow_partial
            )
        return cache[protocol]

    jobs: list[PaperJob] = []

    if "E1" in experiments:
        for fold_id, directory in folds("nln_emp"):
            for seed in seeds:
                for model in BASELINE_MODELS:
                    jobs.append(
                        PaperJob(
                            "E1",
                            "nln_emp",
                            fold_id,
                            str(directory),
                            "nln_emp",
                            model,
                            "main_comparison",
                            seed,
                            modality="vibration",
                        )
                    )
                jobs.append(
                    PaperJob(
                        "E1",
                        "nln_emp",
                        fold_id,
                        str(directory),
                        "nln_emp",
                        "proposed",
                        "frozen_proposed",
                        seed,
                        use_gate=frozen_gate,
                    )
                )

    if "E2" in experiments:
        variants = (
            ("vibration_only", "vibration_only", False),
            ("current_only", "current_only", False),
            ("fusion_gate_off", None, False),
            ("fusion_gate_on", None, True),
        )
        for fold_id, directory in folds("nln_emp"):
            for seed in seeds:
                for configuration, ablation, gate in variants:
                    jobs.append(
                        PaperJob(
                            "E2",
                            "nln_emp",
                            fold_id,
                            str(directory),
                            "nln_emp",
                            "proposed",
                            configuration,
                            seed,
                            ablation=ablation,
                            use_gate=gate,
                        )
                    )

    if "E3" in experiments:
        for protocol in (
            "nln_emp",
            "paderborn_condition_generalization",
            "cwru",
        ):
            for fold_id, directory in folds(protocol):
                for seed in seeds:
                    is_cwru = protocol == "cwru"
                    jobs.append(
                        PaperJob(
                            "E3",
                            protocol,
                            fold_id,
                            str(directory),
                            PROTOCOLS[protocol]["dataset"],
                            "proposed",
                            "cross_condition",
                            seed,
                            ablation="vibration_only" if is_cwru else None,
                            use_gate=frozen_gate and not is_cwru,
                        )
                    )

    if "E4" in experiments:
        variants = (
            ("standard_15_epochs", False, 15, 0),
            ("stage1_only", False, 10, 0),
            ("severity_curriculum", True, 10, 5),
        )
        for fold_id, directory in folds("nln_emp"):
            for seed in seeds:
                for configuration, curriculum, stage1, stage2 in variants:
                    jobs.append(
                        PaperJob(
                            "E4",
                            "nln_emp",
                            fold_id,
                            str(directory),
                            "nln_emp",
                            "proposed",
                            configuration,
                            seed,
                            use_gate=frozen_gate,
                            use_curriculum=curriculum,
                            stage1_epochs=stage1,
                            stage2_epochs=stage2,
                        )
                    )

    if "E5" in experiments:
        for fold_id, directory in folds("paderborn_artificial_to_natural"):
            for seed in seeds:
                jobs.append(
                    PaperJob(
                        "E5",
                        "paderborn_artificial_to_natural",
                        fold_id,
                        str(directory),
                        "paderborn",
                        "proposed",
                        "artificial_to_natural",
                        seed,
                        use_gate=frozen_gate,
                    )
                )

    if "E6" in experiments:
        for fold_id, directory in folds("nln_emp"):
            for seed in seeds:
                for budget in (0.10, 0.25, 0.50, 1.0):
                    jobs.append(
                        PaperJob(
                            "E6",
                            "nln_emp",
                            fold_id,
                            str(directory),
                            "nln_emp",
                            strongest_baseline,
                            "limited_labels",
                            seed,
                            label_budget=budget,
                            modality="vibration",
                        )
                    )
                    jobs.append(
                        PaperJob(
                            "E6",
                            "nln_emp",
                            fold_id,
                            str(directory),
                            "nln_emp",
                            "proposed",
                            "limited_labels",
                            seed,
                            label_budget=budget,
                            use_gate=frozen_gate,
                        )
                    )
    return jobs


def _baseline_args(
    job: PaperJob,
    args: argparse.Namespace,
    shared_cache: Any = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        processed_dir=job.processed_dir,
        dataset=job.dataset,
        model=job.model,
        modality=job.modality,
        seed=job.seed,
        label_budget=job.label_budget,
        epochs=args.baseline_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        smoke_test=args.smoke_test,
        checkpoint_dir=args.checkpoint_dir,
        run_id=job.run_id,
        paper_experiment=job.paper_experiment,
        protocol=job.protocol,
        fold_id=job.fold_id,
        configuration=job.configuration,
        shared_tensor_cache=shared_cache,
    )


def _proposed_args(
    job: PaperJob,
    args: argparse.Namespace,
    shared_cache: Any = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        processed_dir=job.processed_dir,
        dataset=job.dataset,
        fold_id=job.fold_id,
        seed=job.seed,
        loss_name=args.frozen_loss,
        use_modality_gate=job.use_gate,
        use_curriculum=job.use_curriculum,
        ablation=job.ablation,
        stage1_epochs=job.stage1_epochs,
        stage2_epochs=job.stage2_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        modality_dropout=args.modality_dropout,
        warmup_ratio=args.warmup_ratio,
        family_loss_weight=args.family_loss_weight,
        preload=False,
        shared_tensor_cache=shared_cache,
        amp=args.amp,
        smoke_test=args.smoke_test,
        checkpoint_dir=args.checkpoint_dir,
        write_detailed_metrics=False,
        run_id=job.run_id,
        experiment=job.configuration,
        configuration=job.configuration,
        paper_experiment=job.paper_experiment,
        protocol=job.protocol,
        label_budget=job.label_budget,
    )


def training_signature(job: PaperJob, args: argparse.Namespace) -> str:
    """Describe training behavior independently of the paper table using it."""
    if job.model == "proposed":
        details = (
            args.frozen_loss,
            job.ablation,
            job.use_gate,
            job.use_curriculum,
            job.stage1_epochs,
            job.stage2_epochs,
        )
    else:
        details = (job.modality, args.baseline_epochs)
    return "|".join(
        map(
            str,
            (
                job.protocol,
                job.fold_id,
                job.model,
                job.seed,
                job.label_budget,
                *details,
            ),
        )
    )


def reusable_result(
    output_path: Path,
    job: PaperJob,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    results = read_current_results(output_path)
    if results.empty or "training_signature" not in results:
        return None
    signature = training_signature(job, args)
    match = results[
        (results["status"] == "COMPLETED")
        & (results["training_signature"].astype(str) == signature)
    ]
    if match.empty:
        return None
    source = match.iloc[0].to_dict()
    source.update(
        {
            **job.identity(),
            "run_id": job.run_id,
            "source_run_id": source["run_id"],
            "training_signature": signature,
            "status": "COMPLETED",
        }
    )
    return source


def execute_paper_jobs(
    jobs: list[PaperJob], args: argparse.Namespace
) -> pd.DataFrame:
    output_path = Path(args.output_file)
    grouped: dict[tuple[str, str, str], list[PaperJob]] = {}
    for job in jobs:
        grouped.setdefault(
            (job.protocol, job.fold_id, job.processed_dir), []
        ).append(job)
    number = 0
    for (_, _, processed_dir), fold_jobs in grouped.items():
        pending = [
            job
            for job in fold_jobs
            if not result_completed(output_path, job.identity())
        ]
        shared_cache = None
        if pending and args.preload:
            shared_cache = load_protocol_tensor_cache(
                Path(processed_dir),
                maximum_cache_gb=args.cache_max_gb,
                smoke_test=args.smoke_test,
            )
        try:
            for job in fold_jobs:
                number += 1
                if result_completed(output_path, job.identity()):
                    print(f"[{number}/{len(jobs)}] skip {job.run_id}")
                    continue
                reused = reusable_result(output_path, job, args)
                if reused is not None:
                    print(
                        f"[{number}/{len(jobs)}] reuse "
                        f"{reused['source_run_id']} as {job.run_id}"
                    )
                    bank_result(reused, output_path)
                    continue
                print(f"[{number}/{len(jobs)}] run {job.run_id}")
                try:
                    result = (
                        train_multimodal(
                            _proposed_args(job, args, shared_cache)
                        )
                        if job.model == "proposed"
                        else run_baseline(
                            _baseline_args(job, args, shared_cache)
                        )
                    )
                    result["status"] = "COMPLETED"
                    result["training_signature"] = training_signature(
                        job, args
                    )
                except Exception as error:
                    result = {
                        **job.identity(),
                        "run_id": job.run_id,
                        "dataset": job.dataset,
                        "processed_dir": job.processed_dir,
                        "status": "FAILED",
                        "error": f"{type(error).__name__}: {error}",
                    }
                    bank_result(result, output_path)
                    if args.fail_fast:
                        raise
                else:
                    bank_result(result, output_path)
                finally:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            if shared_cache is not None:
                shared_cache.clear()
                del shared_cache
                gc.collect()
    return pd.read_csv(output_path) if output_path.exists() else pd.DataFrame()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=PAPER_EXPERIMENTS,
        default=list(PAPER_EXPERIMENTS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--frozen-loss", choices=LOSS_NAMES, default="ce_1.0"
    )
    parser.add_argument(
        "--frozen-gate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--strongest-baseline", choices=BASELINE_MODELS, default="cnn"
    )
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--preload", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--cache-max-gb", type=float, default=36.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--modality-dropout", type=float, default=0.2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--family-loss-weight", type=float, default=0.5)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--allow-partial-folds", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output-file",
        default="results/tables/corrected_paper_experiments.csv",
    )
    parser.add_argument(
        "--checkpoint-dir", default="artifacts/checkpoints/paper_experiments"
    )
    parser.add_argument(
        "--plan-file", default="results/tables/paper_experiment_plan.csv"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiments = tuple(args.experiments)
    training_experiments = tuple(
        experiment for experiment in experiments if experiment != "E7"
    )
    seeds = tuple(args.seeds[:1] if args.smoke_test else args.seeds)
    jobs = build_paper_jobs(
        data_root=Path(args.data_root),
        split_root=Path(args.split_root),
        experiments=training_experiments,
        seeds=seeds,
        frozen_gate=args.frozen_gate,
        strongest_baseline=args.strongest_baseline,
        allow_partial=args.allow_partial_folds,
    )
    plan_path = Path(args.plan_file)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(job) | {"run_id": job.run_id} for job in jobs]).to_csv(
        plan_path, index=False
    )
    print(f"Planned {len(jobs)} training jobs in {plan_path}")
    if not args.dry_run:
        execute_paper_jobs(jobs, args)
    if "E7" in experiments and not args.dry_run:
        from src.evaluation.explainability import generate_e7_artifacts

        generate_e7_artifacts(
            results_path=Path(args.output_file),
            output_dir=Path("results/figures"),
        )


if __name__ == "__main__":
    main()
