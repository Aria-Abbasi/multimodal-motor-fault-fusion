#!/usr/bin/env python3
from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub


def sync_tree(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        destination = dst_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(item, destination)
        else:
            shutil.copy2(item, destination)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data" / "raw"
    paderborn_dir = raw_root / "paderborn"
    cwru_dir = raw_root / "cwru"

    # Download latest version
    paderborn_path = Path(kagglehub.dataset_download("dippatel03/paderborn-db"))
    print("Path to dataset files:", paderborn_path)
    sync_tree(paderborn_path, paderborn_dir)
    print("Copied Paderborn files to:", paderborn_dir)

    # Download latest version
    cwru_path = Path(kagglehub.dataset_download("brjapon/cwru-bearing-datasets"))
    print("Path to dataset files:", cwru_path)
    sync_tree(cwru_path, cwru_dir)
    print("Copied CWRU files to:", cwru_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
