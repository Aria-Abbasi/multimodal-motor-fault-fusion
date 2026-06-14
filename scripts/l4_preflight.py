"""Verify that an L4 server is ready for the validation pilot and final plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.training.experiment_runner import (
    DEFAULT_SEEDS,
    PROTOCOLS,
    build_experiment_matrix,
    discover_processed_folds,
)
from src.training.paper_experiment_runner import (
    build_paper_jobs,
    training_signature,
)
from src.training.train_multimodal import PIPELINE_VERSION


EXPECTED_PROVENANCE_SHA256 = (
    "aea7de39edef3bf27a9c4bbc1bf64eb9bde3992c9ff1221f3f9b486be24a4933"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(command: str) -> str:
    return subprocess.check_output(
        ["git", *command.split()], text=True
    ).strip()


def _ram_gb() -> float:
    page_size = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return page_size * pages / 1024**3


def run_preflight(
    *,
    data_root: Path,
    split_root: Path,
    provenance_archive: Path,
    minimum_ram_gb: float,
    minimum_free_disk_gb: float,
    minimum_gpu_memory_gb: float,
    require_clean_git: bool,
    full_tensor_count: bool,
    allow_non_l4: bool,
) -> dict[str, Any]:
    """Run all readiness checks and return a machine-readable report."""
    failures: list[str] = []
    report: dict[str, Any] = {
        "pipeline_version": PIPELINE_VERSION,
        "git_revision": _git("rev-parse HEAD"),
    }

    status = _git("status --porcelain")
    report["git_clean"] = not bool(status)
    if require_clean_git and status:
        failures.append("Git worktree is not clean")

    report["cuda_available"] = torch.cuda.is_available()
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
    else:
        properties = torch.cuda.get_device_properties(0)
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = properties.total_memory / 1024**3
        report.update(
            {
                "gpu_name": gpu_name,
                "gpu_memory_gb": gpu_memory_gb,
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            }
        )
        if not allow_non_l4 and "L4" not in gpu_name.upper():
            failures.append(f"Expected an NVIDIA L4, found {gpu_name}")
        if gpu_memory_gb < minimum_gpu_memory_gb:
            failures.append(
                f"GPU memory {gpu_memory_gb:.1f} GiB is below "
                f"{minimum_gpu_memory_gb:.1f} GiB"
            )

    ram_gb = _ram_gb()
    free_disk_gb = shutil.disk_usage(data_root).free / 1024**3
    report["system_ram_gb"] = ram_gb
    report["free_disk_gb"] = free_disk_gb
    if ram_gb < minimum_ram_gb:
        failures.append(
            f"System RAM {ram_gb:.1f} GiB is below {minimum_ram_gb:.1f} GiB"
        )
    if free_disk_gb < minimum_free_disk_gb:
        failures.append(
            f"Free disk {free_disk_gb:.1f} GiB is below "
            f"{minimum_free_disk_gb:.1f} GiB"
        )

    if not provenance_archive.exists():
        failures.append(f"Missing provenance archive: {provenance_archive}")
    else:
        archive_sha256 = _sha256(provenance_archive)
        report["provenance_sha256"] = archive_sha256
        if archive_sha256 != EXPECTED_PROVENANCE_SHA256:
            failures.append("Preprocessing provenance checksum does not match")

    fold_rows: list[dict[str, Any]] = []
    for protocol in PROTOCOLS:
        try:
            folds = discover_processed_folds(
                data_root, split_root, protocol
            )
        except (FileNotFoundError, ValueError) as error:
            failures.append(str(error))
            continue
        for fold_id, fold_dir in folds:
            manifest_path = fold_dir / "preprocessing_manifest.json"
            index_path = fold_dir / "windows_index.csv"
            if not manifest_path.exists():
                failures.append(f"Missing manifest: {manifest_path}")
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            index_rows = sum(1 for _ in index_path.open("rb")) - 1
            tensor_count = None
            if full_tensor_count:
                tensor_count = sum(
                    1
                    for entry in os.scandir(fold_dir / "tensors")
                    if entry.is_file() and entry.name.endswith(".pt")
                )
            saved_windows = int(manifest["saved_windows"])
            if index_rows != saved_windows:
                failures.append(
                    f"{protocol}/{fold_id}: index={index_rows}, "
                    f"manifest={saved_windows}"
                )
            if tensor_count is not None and tensor_count != saved_windows:
                failures.append(
                    f"{protocol}/{fold_id}: tensors={tensor_count}, "
                    f"manifest={saved_windows}"
                )
            fold_rows.append(
                {
                    "protocol": protocol,
                    "fold_id": fold_id,
                    "saved_windows": saved_windows,
                    "index_rows": index_rows,
                    "tensor_files": tensor_count,
                }
            )
    report["processed_folds"] = fold_rows
    report["processed_fold_count"] = len(fold_rows)
    report["processed_window_count"] = sum(
        row["saved_windows"] for row in fold_rows
    )
    if len(fold_rows) != 13:
        failures.append(f"Expected 13 processed folds, found {len(fold_rows)}")

    pilot_jobs = (
        len(discover_processed_folds(data_root, split_root, "nln_emp"))
        * len(build_experiment_matrix())
    )
    final_jobs = build_paper_jobs(
        data_root=data_root,
        split_root=split_root,
        experiments=("E1", "E2", "E3", "E4", "E5", "E6"),
        seeds=DEFAULT_SEEDS,
    )
    signature_args = SimpleNamespace(
        frozen_loss="ce_1.0", baseline_epochs=20
    )
    report["pilot_jobs"] = pilot_jobs
    report["final_planned_rows"] = len(final_jobs)
    report["final_unique_training_signatures"] = len(
        {training_signature(job, signature_args) for job in final_jobs}
    )
    if pilot_jobs != 48:
        failures.append(f"Expected 48 pilot jobs, found {pilot_jobs}")
    if len(final_jobs) != 525:
        failures.append(f"Expected 525 final rows, found {len(final_jobs)}")

    report["failures"] = failures
    report["ready"] = not failures
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--split-root", default="data/splits")
    parser.add_argument(
        "--provenance-archive",
        default="artifacts/provenance/preprocessing_20260614.tar.gz",
    )
    parser.add_argument("--minimum-ram-gb", type=float, default=48.0)
    parser.add_argument("--minimum-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--minimum-gpu-memory-gb", type=float, default=22.0)
    parser.add_argument(
        "--require-clean-git",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--full-tensor-count", action="store_true")
    parser.add_argument("--allow-non-l4", action="store_true")
    parser.add_argument(
        "--output", default="artifacts/logs/l4_preflight.json"
    )
    args = parser.parse_args()
    report = run_preflight(
        data_root=Path(args.data_root),
        split_root=Path(args.split_root),
        provenance_archive=Path(args.provenance_archive),
        minimum_ram_gb=args.minimum_ram_gb,
        minimum_free_disk_gb=args.minimum_free_disk_gb,
        minimum_gpu_memory_gb=args.minimum_gpu_memory_gb,
        require_clean_git=args.require_clean_git,
        full_tensor_count=args.full_tensor_count,
        allow_non_l4=args.allow_non_l4,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
