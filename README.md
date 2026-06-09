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

Run one configured multimodal training job:

```bash
python -m src.training.train_multimodal \
  --processed-dir data/processed/nln_emp/nln_emp_leave_one_speed_out \
  --dataset nln_emp \
  --loss-name ce_2.0 \
  --use-modality-gate \
  --seed 42
```

Run the full loss/gate matrix (12 configurations x 5 seeds):

```bash
python -m src.training.experiment_runner \
  --protocol nln_emp \
  --output-file results/tables/loss_gate_matrix_results.csv
```

Run the same matrix on the Paderborn artificial-to-natural protocol:

```bash
python -m src.training.experiment_runner \
  --protocol paderborn_artificial_to_natural \
  --output-file results/tables/paderborn_loss_gate_matrix_results.csv
```

Run NLN-EMP and then Paderborn in one process. The runner releases the
NLN-EMP cache before loading Paderborn:

```bash
python -m src.training.experiment_runner \
  --protocols nln_emp paderborn_artificial_to_natural \
  --cache-max-gb 36 \
  --output-file results/tables/loss_gate_matrix_results.csv
```

The runner resumes at seed level. Paderborn early-fault recall is reported as
`N/A` when the test metadata has no granular severity labels.

Run tests:

```bash
python -m pytest
```

For a CPU-only local environment, install PyTorch from its CPU wheel index:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Notes

- Raw datasets are intentionally excluded from Git.
- Empty directories are preserved with `.gitkeep` placeholders.
- This scaffold focuses on reproducibility primitives and CLI wiring for subsequent steps.
