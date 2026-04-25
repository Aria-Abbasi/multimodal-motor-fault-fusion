# Early Fault Fusion Paper Repository

Reproducible ML research scaffold for the paper:

**Severity-Aware Cross-Condition Early Fault Detection in Electric Motor Systems Using Vibration-Current Cross-Attention Learning**

## Goal

Build, evaluate, and document models for early fault detection under unseen operating conditions using vibration + current modalities.

## Dataset roles (frozen)

- **NLN-EMP**: main dataset for primary evidence and core claims.
- **Paderborn**: external robustness and artificial-to-natural transfer validation.
- **CWRU**: benchmark-only comparability dataset.
- **IMS**: excluded from Paper 1 (reserved for future prognostics/RUL work).

## Leakage rule (critical)

Never split windows from the same original recording across train/validation/test.
All split logic must be recording-level and leakage-safe.

## Tracked metrics

- Macro F1
- Balanced accuracy
- Early-fault recall (central paper metric)
- AUROC
- AUPRC
- MCC

## Repository layout

See the file tree below or run:

```bash
find project -maxdepth 4 -print
```

## Quick start

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Core commands

Training entry point (placeholder scaffold):

```bash
python -m src.training.train --help
python -m src.training.train \
  --config configs/base.yaml \
  --experiment-name smoke_train \
  --seed 42 \
  --output-dir artifacts
```

Evaluation entry point (placeholder scaffold):

```bash
python -m src.evaluation.evaluate --help
python -m src.evaluation.evaluate \
  --config configs/base.yaml \
  --experiment-name smoke_eval \
  --seed 42 \
  --output-dir results
```

Run tests:

```bash
pytest
```

## Notes

- Raw datasets are intentionally excluded from Git.
- Empty directories are preserved with `.gitkeep` placeholders.
- This scaffold focuses on reproducibility primitives and CLI wiring for subsequent steps.
