#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

def sync_tree(src_dir: Path, dst_dir: Path) -> None:
    """Move files from the download cache into the raw-data tree."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        destination = dst_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(str(item), str(destination))
        else:
            if destination.exists():
                destination.unlink()
            shutil.move(str(item), str(destination))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("KAGGLEHUB_CACHE", "data/interim/kagglehub_cache"),
    )
    args = parser.parse_args()
    raw_root = Path(args.raw_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)

    import kagglehub

    paderborn_dir = raw_root / "paderborn"
    cwru_dir = raw_root / "cwru"

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading Paderborn dataset...")
    paderborn_path = Path(kagglehub.dataset_download("dippatel03/paderborn-db"))
    print("Downloaded to cache:", paderborn_path)
    sync_tree(paderborn_path, paderborn_dir)
    print("Moved Paderborn files to:", paderborn_dir)

    print("\nDownloading CWRU dataset...")
    cwru_path = Path(kagglehub.dataset_download("brjapon/cwru-bearing-datasets"))
    print("Downloaded to cache:", cwru_path)
    sync_tree(cwru_path, cwru_dir)
    print("Moved CWRU files to:", cwru_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
