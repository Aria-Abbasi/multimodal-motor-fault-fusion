"""Training entry point for the early-fault-fusion-paper project."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.utils.logging import setup_logger
from src.utils.seed import resolve_device, set_global_seed


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for training CLI.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Train models for early fault detection.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config file.")
    parser.add_argument("--experiment-name", type=str, default="train_run", help="Experiment name.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Output directory root for training artifacts.",
    )
    return parser


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML config file.

    Args:
        config_path: Path to config file.

    Returns:
        Parsed config dictionary.
    """
    import yaml

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> int:
    """CLI main function for training scaffold.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    set_global_seed(args.seed)

    output_root = Path(args.output_dir)
    log_path = output_root / "logs" / f"{args.experiment_name}.log"
    logger = setup_logger("train", log_file=log_path)

    configured_device = str(config.get("runtime", {}).get("device", "auto"))
    active_device = resolve_device(configured_device)

    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(config.get("paths", {}).get("runs", "results/runs")) / args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting training scaffold")
    logger.info("Experiment: %s", args.experiment_name)
    logger.info("Seed: %d", args.seed)
    logger.info("Device: %s", active_device)
    logger.info("Leakage rule: %s", config.get("datasets", {}).get("leakage_rule", "unknown"))

    summary = {
        "run_id": run_id,
        "experiment_name": args.experiment_name,
        "seed": args.seed,
        "device": active_device,
        "config": str(config_path),
        "status": "scaffold_complete",
    }
    summary_path = run_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote scaffold summary to %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
