#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

import kagglehub

# 1. CRITICAL: Redirect the Kagglehub cache to the big drive 
# This prevents the 28GB root partition from filling up during the download.
os.environ["KAGGLEHUB_CACHE"] = "/home/Aria/data/kagglehub_cache"

def sync_tree(src_dir: Path, dst_dir: Path) -> None:
    """Moves files from source to destination to save disk space."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        destination = dst_dir / item.name
        if item.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            # Use move instead of copy to free up space on the fly
            shutil.move(str(item), str(destination))
        else:
            if destination.exists():
                destination.unlink()
            shutil.move(str(item), str(destination))

def main() -> int:
    # 2. Set the absolute path to your large drive
    raw_root = Path("/home/Aria/data/multimodal-motor-fault-fusion/data/raw")
    paderborn_dir = raw_root / "paderborn"
    cwru_dir = raw_root / "cwru"

    # Ensure the cache directory on the big drive exists
    Path(os.environ["KAGGLEHUB_CACHE"]).mkdir(parents=True, exist_ok=True)

    # Download Paderborn
    print("Downloading Paderborn dataset...")
    paderborn_path = Path(kagglehub.dataset_download("dippatel03/paderborn-db"))
    print("Downloaded to cache:", paderborn_path)
    
    sync_tree(paderborn_path, paderborn_dir)
    print("Moved Paderborn files to:", paderborn_dir)

    # Download CWRU
    print("\nDownloading CWRU dataset...")
    cwru_path = Path(kagglehub.dataset_download("brjapon/cwru-bearing-datasets"))
    print("Downloaded to cache:", cwru_path)
    
    sync_tree(cwru_path, cwru_dir)
    print("Moved CWRU files to:", cwru_dir)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())