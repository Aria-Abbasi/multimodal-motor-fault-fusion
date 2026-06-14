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
- Fault precision
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

Run the validation pilot
(`4 folds x 12 configurations x 1 seed = 48 jobs`):

```bash
python -m src.training.experiment_runner \
  --protocol nln_emp \
  --seeds 42 \
  --require-cuda \
  --cache-max-gb 48 \
  --output-file results/tables/nln_validation_pilot.csv \
  --summary-file results/tables/nln_validation_pilot_summary.csv \
  --checkpoint-dir artifacts/checkpoints/nln_validation_pilot \
  --fail-fast
```

Freeze one configuration using validation metrics only:

```bash
python -m src.training.pilot_selection \
  --results results/tables/nln_validation_pilot.csv \
  --output configs/frozen_l4_selection.yaml \
  --minimum-early-recall 0.95
```

The pilot selector requires all 48 rows and never reads test metrics for
configuration selection.

The final runner transfers the validation-frozen NLN loss and gate to the
Paderborn protocols. Fixed early-fault CE multipliers are inactive where
Paderborn has no early-severity annotation; focal loss remains focal if it is
the frozen choice. Paderborn early-fault recall is reported as `N/A`.

Run tests:

```bash
python -m pytest
```

## Complete paper experiments

Inspect the complete E1-E7 training plan without launching jobs:

```bash
python -m src.training.paper_experiment_runner \
  --dry-run \
  --frozen-config configs/frozen_l4_selection.yaml \
  --require-cuda \
  --cache-max-gb 48
```

After selecting and freezing the loss/gate configuration using validation
results, execute the full resumable plan:

```bash
python -m src.training.paper_experiment_runner \
  --experiments E1 E2 E3 E4 E5 E6 E7 \
  --frozen-config configs/frozen_l4_selection.yaml \
  --require-cuda \
  --cache-max-gb 48 \
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
`corrected_multimodal_v3` are legacy artifacts and must not be used for final
GPU results. The audited preprocessing archive separately records the v2 CPU
data-generation provenance.

Before the validation pilot, run:

```bash
python scripts/l4_preflight.py --full-tensor-count
```

See [docs/cloud_runbook.md](docs/cloud_runbook.md) for the disk-transfer,
preflight, pilot, freeze, and final-run sequence.

For a CPU-only local environment, install PyTorch from its CPU wheel index:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
```

## Notes

- Raw datasets are intentionally excluded from Git.
- Empty directories are preserved with `.gitkeep` placeholders.
- This scaffold focuses on reproducibility primitives and CLI wiring for subsequent steps.
