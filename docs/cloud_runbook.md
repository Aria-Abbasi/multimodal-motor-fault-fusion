# Corrected Pipeline Cloud Runbook

The CPU preprocessing server and L4 training server must use the same Git
revision and configuration files. Never copy the old processed tensors or old
result CSVs into the corrected run directories.

## 1. CPU Server

Recommended storage:

- 120 GB for raw NLN-EMP and Paderborn data
- 45-60 GB for FP16 processed tensors, depending on retained windows
- 30 GB for temporary files, environment and safety margin
- 220 GB free disk is a comfortable minimum

Install and validate:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m pytest
```

Download raw datasets, then rebuild metadata and splits:

```bash
python -m src.data.make_raw_manifest
python scripts/inspect_paderborn_channels.py
python -m src.data.build_metadata_master
python -m src.data.generate_splits
python scripts/validate_splits.py
```

NLN vibration channel 2 is frozen from the dataset README: electric-motor
driven-end bearing, vertical. The selection rationale is recorded in
`docs/nln_sensor_selection.md`.

Build corrected tensors:

```bash
python -m src.data.build_spectrograms \
  --split-file data/splits/nln_emp_leave_one_speed_out.csv \
  --dataset nln_emp \
  --all-folds \
  --nln-vibration-channel 2 \
  --nln-current-channels 1 2 3 \
  --tensor-dtype float16

python -m src.data.build_spectrograms \
  --split-file data/splits/paderborn_condition_generalization.csv \
  --dataset paderborn \
  --all-folds \
  --tensor-dtype float16

python -m src.data.build_spectrograms \
  --split-file data/splits/paderborn_artificial_to_natural.csv \
  --dataset paderborn \
  --tensor-dtype float16
```

For every fold, inspect:

- `preprocessing_manifest.json`
- `normalization_stats.json`
- `paired_recordings.csv`
- `preprocessing_exclusions.csv`, when present
- QC plots
- train/validation/test base-recording overlap

Do not use `--skip-bad-recordings` for final data generation. It is only a
diagnostic option.

Run a CPU smoke test against one processed fold:

```bash
python -m src.training.train_multimodal \
  --processed-dir data/processed/nln_emp/nln_emp_leave_one_speed_out/test_speed_50 \
  --dataset nln_emp \
  --loss-name ce_1.0 \
  --use-modality-gate \
  --smoke-test \
  --no-amp
```

Archive the source revision and processed-data manifests before transfer:

```bash
git rev-parse HEAD
find data/processed -name preprocessing_manifest.json -print
find data/processed -name normalization_stats.json -print
```

## 2. L4 Server

Copy the repository and corrected `data/processed/` tree. Raw data is not
required on the L4 server after preprocessing has been audited.

Run another smoke job, then launch the corrected matrix:

```bash
python -m pytest

python -m src.training.paper_experiment_runner \
  --dry-run \
  --frozen-loss FROZEN_LOSS \
  --frozen-gate

python -m src.training.paper_experiment_runner \
  --experiments E1 E2 E3 E4 E5 E6 E7 \
  --frozen-loss FROZEN_LOSS \
  --frozen-gate \
  --output-file results/tables/corrected_paper_experiments.csv \
  --fail-fast
```

On a 64 GB RAM instance, `--cache-max-gb 48` to `50` is reasonable. Monitor
available memory before increasing it.

Select loss and gate settings using
`validation_recording_macro_f1`. Do not select configurations or seeds using
test Macro F1. Use held-out test metrics only after the configuration is
frozen.

Generate paired significance results using identical fold/seed cells:

```bash
python -m src.evaluation.generate_table6_stats \
  --input results/tables/corrected_paper_experiments.csv \
  --experiment-column model \
  --reference proposed \
  --metric recording_macro_f1
```
