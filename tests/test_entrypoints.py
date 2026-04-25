"""Tests for CLI entry points and parser arguments."""

from __future__ import annotations

from src.evaluation.evaluate import build_parser as build_eval_parser
from src.training.train import build_parser as build_train_parser


def test_train_parser_defaults() -> None:
    parser = build_train_parser()
    args = parser.parse_args([])
    assert args.config == "configs/base.yaml"
    assert args.experiment_name == "train_run"
    assert args.seed == 42
    assert args.output_dir == "artifacts"


def test_train_parser_custom_args() -> None:
    parser = build_train_parser()
    args = parser.parse_args(
        ["--config", "x.yaml", "--experiment-name", "exp1", "--seed", "7", "--output-dir", "tmp"]
    )
    assert args.config == "x.yaml"
    assert args.experiment_name == "exp1"
    assert args.seed == 7
    assert args.output_dir == "tmp"


def test_eval_parser_defaults() -> None:
    parser = build_eval_parser()
    args = parser.parse_args([])
    assert args.config == "configs/base.yaml"
    assert args.experiment_name == "eval_run"
    assert args.seed == 42
    assert args.output_dir == "results"
