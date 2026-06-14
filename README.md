# Early Fault Fusion Paper Repository

Reproducible ML research scaffold for the paper:

**Severity-Aware Cross-Condition Early Fault Detection in Electric Motor Systems Using Vibration-Current Cross-Attention Learning**

## Goal

Build, evaluate, and document models for early fault detection under unseen operating conditions using vibration + current modalities.

The current problem register, resolution procedure, execution order, and final
publication checklist are maintained in
[PROJECT_ROADMAP.md](PROJECT_ROADMAP.md).

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
  --all-folds \
  --nln-vibration-channel 2 \
  --nln-current-channels 1 2 3 \
  --tensor-dtype float16
```

Run the full NLN-EMP loss/gate matrix
(`4 folds x 12 configurations x 5 seeds = 240 jobs`):

```bash
python -m src.training.experiment_runner \
  --protocol nln_emp \
  --output-file results/tables/corrected_nln_loss_gate_results.csv \
  --summary-file results/tables/corrected_nln_loss_gate_summary.csv
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
  --protocols nln_emp paderborn_condition_generalization paderborn_artificial_to_natural \
  --cache-max-gb 36 \
  --output-file results/tables/corrected_loss_gate_matrix_results.csv
```

The runner resumes at seed level. Paderborn early-fault recall is reported as
`N/A` when the test metadata has no granular severity labels. Missing expected
folds stop execution by default; `--allow-partial-folds` is intended only for
explicit debugging.

Run tests:

```bash
python -m pytest
```

## Complete paper experiments

Inspect the complete E1-E7 training plan without launching jobs:

```bash
python -m src.training.paper_experiment_runner \
  --dry-run \
  --frozen-loss ce_1.0 \
  --frozen-gate
```

After selecting and freezing the loss/gate configuration using validation
results, execute the full resumable plan:

```bash
python -m src.training.paper_experiment_runner \
  --experiments E1 E2 E3 E4 E5 E6 E7 \
  --frozen-loss FROZEN_LOSS \
  --frozen-gate \
  --output-file results/tables/corrected_paper_experiments.csv \
  --fail-fast
```

The plan includes:

- E1: SVM, Random Forest, CNN, LSTM, CNN-LSTM, Transformer, healthy-only
  autoencoder, and proposed-model comparison.
- E2: vibration/current/fusion/gating ablations.
- E3: NLN-EMP, Paderborn condition, and CWRU generalization.
- E4: standard, stage-one-only, and severity-curriculum training.
- E5: Paderborn artificial-to-natural transfer.
- E6: recording-level 10%, 25%, 50%, and 100% label budgets.
- E7: Grad-CAM, saliency, and cross-attention artifacts from a
  validation-selected checkpoint.

Generate corrected summaries and figures:

```bash
python -m src.evaluation.reporting
python -m src.evaluation.prediction_artifacts
python -m src.evaluation.explainability
```

Download and validate all CWRU early-fault files for loads 0-3:

```bash
python scripts/download_cwru_benchmark.py
python scripts/download_cwru_benchmark.py --validate-only
```

Files in `results/tables` and `results/figures` that predate
`corrected_multimodal_v2` are legacy artifacts and must not be used.

For a CPU-only local environment, install PyTorch from its CPU wheel index:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Notes

- Raw datasets are intentionally excluded from Git.
- Empty directories are preserved with `.gitkeep` placeholders.
- This scaffold focuses on reproducibility primitives and CLI wiring for subsequent steps.
