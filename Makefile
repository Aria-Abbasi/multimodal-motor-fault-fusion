.PHONY: install test train-help eval-help train-smoke eval-smoke bootstrap

install:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	pytest

train-help:
	python -m src.training.train --help

eval-help:
	python -m src.evaluation.evaluate --help

train-smoke:
	python -m src.training.train --config configs/base.yaml --experiment-name smoke_train --seed 42 --output-dir artifacts

eval-smoke:
	python -m src.evaluation.evaluate --config configs/base.yaml --experiment-name smoke_eval --seed 42 --output-dir results

bootstrap:
	python scripts/server_bootstrap.py
