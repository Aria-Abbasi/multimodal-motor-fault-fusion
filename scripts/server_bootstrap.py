#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


REPO_URL = "https://github.com/Aria-Abbasi/multimodal-motor-fault-fusion"
BRANCH = "master"
WORKDIR = Path.home() / "work"
PROJECT_DIR_NAME = "project"

NLN_EMP_URL = "https://data.4tu.nl/file/2b61183e-c14f-4131-829b-cc4822c369d0/8d84b13f-98f7-4baf-a7ea-60dee1e8876f"
CWRU_KAGGLE_DATASET = "brjapon/cwru-bearing-datasets"
PADERBORN_KAGGLE_DATASET = "dippatel03/paderborn-db"


def header(text: str) -> None:
    print("\n" + "=" * 50)
    print(text)
    print("=" * 50)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[RUN] " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def require_tool(tool: str) -> None:
    if shutil.which(tool) is None:
        raise RuntimeError(f"Missing required command: {tool}")


def clone_or_update_repo(repo_dir: Path) -> None:
    header("Clone or update repository")
    if (repo_dir / ".git").exists():
        run(["git", "fetch", "--all", "--prune"], cwd=repo_dir)
        run(["git", "checkout", BRANCH], cwd=repo_dir)
        run(["git", "pull", "--ff-only", "origin", BRANCH], cwd=repo_dir)
    else:
        run(["git", "clone", "--branch", BRANCH, REPO_URL, str(repo_dir)])


def setup_venv(repo_dir: Path) -> tuple[Path, Path]:
    header("Create virtual environment and install dependencies")
    venv_dir = repo_dir / ".venv"
    venv_python = venv_dir / "bin" / "python"
    venv_pip = venv_dir / "bin" / "pip"

    if not venv_python.exists():
        run(["python3", "-m", "venv", str(venv_dir)])

    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(venv_pip), "install", "-r", "requirements.txt"], cwd=repo_dir)
    return venv_python, venv_pip


def prepare_raw_dirs(repo_dir: Path) -> tuple[Path, Path, Path]:
    header("Prepare dataset folders")
    nln_dir = repo_dir / "data" / "raw" / "nln_emp"
    cwru_dir = repo_dir / "data" / "raw" / "cwru"
    paderborn_dir = repo_dir / "data" / "raw" / "paderborn"

    for directory in (nln_dir, cwru_dir, paderborn_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return nln_dir, cwru_dir, paderborn_dir


def download_nln_emp(nln_dir: Path) -> None:
    header("Download NLN-EMP")
    if any(nln_dir.iterdir()):
        print("[INFO] data/raw/nln_emp is not empty, skipping NLN-EMP download")
        return

    nln_zip = nln_dir / "nln_emp.zip"
    print(f"[INFO] Downloading {NLN_EMP_URL} -> {nln_zip}")
    urllib.request.urlretrieve(NLN_EMP_URL, nln_zip)

    with zipfile.ZipFile(nln_zip, "r") as archive:
        archive.extractall(nln_dir)


def sync_tree(src_dir: Path, dst_dir: Path) -> None:
    for item in src_dir.iterdir():
        dst_item = dst_dir / item.name
        if item.is_dir():
            if dst_item.exists():
                shutil.rmtree(dst_item)
            shutil.copytree(item, dst_item)
        else:
            shutil.copy2(item, dst_item)


def kagglehub_download_to_dir(
    venv_python: Path,
    dataset_id: str,
    target_dir: Path,
    label: str,
) -> None:
    if any(target_dir.iterdir()):
        print(f"[INFO] {target_dir} is not empty, skipping {label} download")
        return

    code = (
        "import kagglehub\n"
        f"path = kagglehub.dataset_download({dataset_id!r})\n"
        "print('KAGGLEHUB_PATH=' + path)\n"
    )
    result = subprocess.run(
        [str(venv_python), "-c", code],
        check=True,
        text=True,
        capture_output=True,
    )

    output_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    path_line = next((line for line in output_lines if line.startswith("KAGGLEHUB_PATH=")), None)
    if path_line is None:
        raise RuntimeError(
            f"Could not parse kagglehub download path for {dataset_id}. Output:\n{result.stdout}"
        )

    downloaded_path = Path(path_line.split("=", 1)[1])
    if not downloaded_path.exists():
        raise RuntimeError(f"kagglehub returned path that does not exist: {downloaded_path}")

    print(f"[INFO] {label} downloaded to cache: {downloaded_path}")
    sync_tree(downloaded_path, target_dir)
    print(f"[INFO] {label} synced to: {target_dir}")


def download_kaggle_datasets(venv_python: Path, cwru_dir: Path, paderborn_dir: Path) -> None:
    header("Download CWRU and Paderborn from KaggleHub")
    kagglehub_download_to_dir(venv_python, CWRU_KAGGLE_DATASET, cwru_dir, "CWRU")
    kagglehub_download_to_dir(venv_python, PADERBORN_KAGGLE_DATASET, paderborn_dir, "Paderborn")


def run_smoke(repo_dir: Path, venv_python: Path) -> None:
    header("Run smoke checks")
    run([str(venv_python), "-m", "pytest"], cwd=repo_dir)
    run([str(venv_python), "-m", "src.training.train", "--help"], cwd=repo_dir)
    run([str(venv_python), "-m", "src.evaluation.evaluate", "--help"], cwd=repo_dir)
    run(
        [
            str(venv_python),
            "-m",
            "src.training.train",
            "--config",
            "configs/base.yaml",
            "--experiment-name",
            "smoke_train",
            "--seed",
            "42",
            "--output-dir",
            "artifacts",
        ],
        cwd=repo_dir,
    )
    run(
        [
            str(venv_python),
            "-m",
            "src.evaluation.evaluate",
            "--config",
            "configs/base.yaml",
            "--experiment-name",
            "smoke_eval",
            "--seed",
            "42",
            "--output-dir",
            "results",
        ],
        cwd=repo_dir,
    )


def main() -> int:
    header("Preflight checks")
    for tool in ("git", "python3"):
        require_tool(tool)

    WORKDIR.mkdir(parents=True, exist_ok=True)
    repo_dir = WORKDIR / PROJECT_DIR_NAME

    clone_or_update_repo(repo_dir)
    venv_python, venv_pip = setup_venv(repo_dir)

    run([str(venv_pip), "install", "kagglehub"])

    nln_dir, cwru_dir, paderborn_dir = prepare_raw_dirs(repo_dir)
    download_nln_emp(nln_dir)
    download_kaggle_datasets(venv_python, cwru_dir, paderborn_dir)

    run_smoke(repo_dir, venv_python)

    header("Bootstrap complete")
    print(f"[OK] Repo: {repo_dir}")
    print(f"[OK] Virtual env: {repo_dir / '.venv'}")
    print(f"[OK] Raw data root: {repo_dir / 'data' / 'raw'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
