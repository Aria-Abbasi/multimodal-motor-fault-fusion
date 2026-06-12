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

Build all four leakage-safe NLN-EMP leave-one-speed-out folds. Each fold is
written to its own directory and receives train-only normalization statistics:

```bash
python -m src.data.build_spectrograms \
  --split-file data/splits/nln_emp_leave_one_speed_out.csv \
  --dataset nln_emp \
  --all-folds
```

Run the full NLN-EMP loss/gate matrix
(`4 folds x 12 configurations x 5 seeds = 240 jobs`):

```bash
python -m src.training.experiment_runner \
  --protocol nln_emp \
  --output-file results/tables/nln_loss_gate_results.csv \
  --summary-file results/tables/nln_loss_gate_summary.csv
```

The raw CSV contains one row per fold/configuration/seed. The summary CSV
reports mean and sample standard deviation across all fold-seed runs for each
configuration. Resume checks also use the fold ID, so completing one held-out
speed never suppresses another.

Build and run the Paderborn artificial-to-natural protocol:

```bash
python -m src.data.build_spectrograms \
  --split-file data/splits/paderborn_artificial_to_natural.csv \
  --dataset paderborn
```

```bash
python -m src.training.experiment_runner \
  --protocol paderborn_artificial_to_natural \
  --output-file results/tables/paderborn_loss_gate_matrix_results.csv
```

Because Paderborn bearing IDs are not severity annotations, Paderborn runs use
standard cross-entropy only. The runner evaluates gate off/on across five
seeds (10 jobs per Paderborn protocol) and reports early-fault recall as `N/A`.

Run complete NLN-EMP and then Paderborn in one process. The runner caches and
releases one fold at a time:

```bash
python -m src.training.experiment_runner \
  --protocols nln_emp paderborn_artificial_to_natural \
  --cache-max-gb 36 \
  --output-file results/tables/loss_gate_matrix_results.csv
```

The runner resumes at seed level. Paderborn early-fault recall is reported as
`N/A` when the test metadata has no granular severity labels. Missing expected
folds stop execution by default; `--allow-partial-folds` is intended only for
explicit debugging.

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
