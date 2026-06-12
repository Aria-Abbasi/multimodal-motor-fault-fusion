# Project Problems, Resolution Plan, and Final Goal

## Final Goal

Produce one reproducible, publication-ready paper on severity-aware early fault
detection under unseen operating conditions.

- **NLN-EMP** provides the primary evidence.
- **Paderborn** tests condition generalization and artificial-to-natural
  robustness.
- **CWRU** provides a conventional benchmark.
- **IMS is excluded** from this paper.

The intended claim is that leakage-safe, severity-aware vibration-current
fusion improves early-fault detection under unseen operating conditions. That
claim may be made only if the final experiments support it.

## Current State

The code-only corrections are complete. The local test suite currently passes
45 tests, including synthetic CPU execution of every model family and
experiment type.

Implemented safeguards include:

- recording-level, leakage-safe split handling;
- corrected vibration/current pairing;
- train-only normalization;
- SVM, Random Forest, CNN, LSTM, CNN-LSTM, Transformer, autoencoder, and
  proposed-model support;
- six loss configurations and optional dynamic modality gating;
- AdamW, weight decay, warmup/cosine scheduling, and gradient clipping;
- validation-derived recording thresholds and complete metrics;
- resume-safe E1-E7 planning and versioned result banking;
- paired fold/seed statistics and validation-only checkpoint selection;
- Grad-CAM, saliency, and cross-attention generation;
- CWRU load 0-3 download and validation support;
- rejection of unversioned legacy results.

This proves that the software paths work on synthetic data. It does **not**
prove that the raw datasets, sensor selection, generated tensors, or final
scientific results are correct.

## Remaining Problems

### 1. NLN-EMP vibration channel is not verified

**Problem:** `configs/base.yaml` intentionally leaves
`nln_vibration_channel: null`. Selecting a channel by guess could make the
experiment physically invalid even when the code runs correctly.

**Resolution:**

1. Read the NLN-EMP sensor-location appendix.
2. Match channel numbers to the real recording schema.
3. Inspect representative healthy and faulty files.
4. Record the chosen physical sensor and channel in the experiment notes.
5. Set the verified channel in `configs/base.yaml`.

**Acceptance check:** the configuration, appendix, and inspected file schema
all identify the same vibration sensor.

### 2. Complete raw datasets are not locally available

**Problem:** synthetic tests cannot reveal missing, corrupt, duplicated, or
unexpected real files.

**Resolution:** download complete NLN-EMP, Paderborn, and CWRU data on the CPU
server, then create and validate the raw manifest.

```bash
python -m src.data.make_raw_manifest
python scripts/validate_raw_manifest.py
python scripts/download_cwru_benchmark.py --validate-only
```

**Acceptance check:** every expected file is represented, checksums are
available, unreadable files are zero, and all 16 selected CWRU files across
loads 0-3 pass validation.

### 3. Real metadata and protocol splits must be rebuilt

**Problem:** corrected parsing and split logic have not yet been exercised on
the complete real datasets.

**Resolution:**

```bash
python scripts/inspect_paderborn_channels.py
python -m src.data.build_metadata_master
python scripts/validate_metadata_master.py
python -m src.data.generate_splits
python scripts/validate_splits.py
```

Required protocols:

- NLN-EMP: four leave-one-speed-out folds;
- Paderborn: four leave-one-condition-out folds;
- Paderborn: artificial-to-natural transfer;
- CWRU: four leave-one-load-out folds.

**Acceptance check:** every recording has a stable ID and no base recording
appears in more than one of train, validation, and test within a fold.

### 4. Corrected tensors have not been generated from real recordings

**Problem:** all old tensors were produced before the final data corrections
and must not be mixed with the corrected pipeline.

**Resolution:** generate a new FP16 processed-data tree from the validated
splits. Do not use `--skip-bad-recordings` for final generation.

```bash
python -m src.data.build_spectrograms \
  --split-file data/splits/nln_emp_leave_one_speed_out.csv \
  --dataset nln_emp \
  --all-folds \
  --nln-vibration-channel VERIFIED_CHANNEL \
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

python -m src.data.build_spectrograms \
  --split-file data/splits/cwru_leave_one_load_out.csv \
  --dataset cwru \
  --all-folds \
  --tensor-dtype float16
```

For every generated fold, inspect:

- `preprocessing_manifest.json`;
- `normalization_stats.json`;
- `paired_recordings.csv`;
- `preprocessing_exclusions.csv`, if present;
- QC plots and class/condition counts.

**Acceptance check:** tensors are finite, normalization uses training data
only, paired modalities come from the same recording, exclusions are
explained, and split overlap remains zero.

### 5. Real-data execution is not yet verified

**Problem:** synthetic execution does not measure real memory usage, loading
time, class distributions, or metadata edge cases.

**Resolution:** on the CPU server, run the full test suite and one smoke job
for each protocol using the real processed tensors.

```bash
python -m pytest -q

python -m src.training.paper_experiment_runner \
  --experiments E1 E2 E3 E4 E5 E6 \
  --seeds 42 \
  --smoke-test \
  --no-amp \
  --allow-partial-folds \
  --output-file results/tables/real_data_cpu_smoke.csv \
  --fail-fast
```

The partial-fold option is allowed only for this smoke test.

**Acceptance check:** all protocol/model paths finish, metrics contain no
unexpected NaN values, Paderborn early recall is explicitly unavailable when
severity labels do not exist, and peak disk/RAM use is recorded.

### 6. Loss and modality gate are not scientifically frozen

**Problem:** the old fixed 5.0 early-fault weight caused excessive false
positives. The replacement loss and gate must be selected without looking at
test metrics.

**Resolution:** on the L4, run the 12 loss-by-gate configurations on NLN-EMP
using a pilot seed across all four folds.

```bash
python -m src.training.experiment_runner \
  --protocol nln_emp \
  --seeds 42 \
  --cache-max-gb 36 \
  --output-file results/tables/nln_validation_pilot.csv \
  --summary-file results/tables/nln_validation_pilot_summary.csv \
  --fail-fast
```

Choose one configuration using only validation recording metrics:

1. primary: validation recording Macro F1;
2. constraint: acceptable early-fault recall;
3. tie-breakers: precision, MCC, and stability across folds.

Test metrics produced during the pilot must not influence this choice. Record
the frozen loss, gate state, threshold policy, selection rule, and Git revision
before final training.

**Acceptance check:** exactly one loss/gate configuration is documented and no
later choice is made from test performance.

### 7. Final E1-E7 evidence has not been generated

**Problem:** the models and runner exist, but the corrected five-seed,
all-fold experiment results do not.

**Resolution:** inspect the plan first, then run it on the L4 with the frozen
configuration.

```bash
python -m src.training.paper_experiment_runner \
  --experiments E1 E2 E3 E4 E5 E6 E7 \
  --frozen-loss FROZEN_LOSS \
  --frozen-gate \
  --dry-run \
  --plan-file results/tables/final_experiment_plan.csv

python -m src.training.paper_experiment_runner \
  --experiments E1 E2 E3 E4 E5 E6 E7 \
  --frozen-loss FROZEN_LOSS \
  --frozen-gate \
  --cache-max-gb 36 \
  --output-file results/tables/corrected_paper_experiments.csv \
  --checkpoint-dir results/runs/corrected_paper \
  --fail-fast
```

Use `--no-frozen-gate` in both commands if the validation pilot selects gate
off. The runner uses five default seeds, resumes completed jobs, reuses
identical training jobs, caches one fold at a time, and releases that cache
before moving to the next fold or dataset.

Experiment roles:

- **E1:** fair comparison with all baseline families;
- **E2:** modality and gate ablations;
- **E3:** cross-condition and cross-dataset generalization;
- **E4:** curriculum ablation;
- **E5:** Paderborn artificial-to-natural robustness;
- **E6:** 10%, 25%, 50%, and 100% recording-level label budgets;
- **E7:** validation-selected Grad-CAM, saliency, and attention explanations.

**Acceptance check:** every planned fold/seed cell is present, every row uses
the current `pipeline_version`, failed jobs are resolved, and no legacy result
is merged into the corrected table.

### 8. Publication tables, figures, and claims remain unfinished

**Problem:** a working experiment is not yet a paper. Outputs must be generated
directly from corrected results and every claim must match the evidence.

**Resolution:**

```bash
python -m src.evaluation.reporting \
  --results results/tables/corrected_paper_experiments.csv

python -m src.evaluation.prediction_artifacts \
  --results results/tables/corrected_paper_experiments.csv

python -m src.evaluation.explainability \
  --results results/tables/corrected_paper_experiments.csv

python -m src.evaluation.generate_table6_stats \
  --input results/tables/corrected_paper_experiments.csv \
  --experiment-column model \
  --reference proposed \
  --metric recording_macro_f1
```

Report fold variation separately from seed variation and use paired
fold/seed comparisons with Holm correction. A p-value below 0.05 is supporting
evidence, not a requirement that permits selective rerunning or cherry-picking.

**Acceptance check:** all tables and figures regenerate without manual row
editing, checkpoint selection uses validation metrics only, and the manuscript
does not claim superiority where corrected results do not support it.

## Execution Order

### Phase A: Before renting a server

1. Keep the 45 local tests passing.
2. Commit or archive the corrected code revision.
3. Preserve the current pipeline version.
4. Prepare dataset credentials, documentation, and storage.

### Phase B: CPU data server

1. Install the exact repository revision and dependencies.
2. Download and validate raw data.
3. Verify the NLN vibration channel.
4. Build metadata and all recording-level splits.
5. Validate zero leakage.
6. Generate all corrected FP16 tensors.
7. Audit manifests, exclusions, QC, and processed-data size.
8. Run real-data CPU smoke tests.
9. Freeze and archive the processed-data manifests.

Do not proceed to expensive training until every CPU acceptance check passes.

### Phase C: L4 validation pilot

1. Copy the repository revision and audited processed tensors.
2. Run tests and one GPU smoke job.
3. Run the 12-configuration NLN validation pilot.
4. Select and document one loss/gate configuration from validation data only.

### Phase D: L4 final training

1. Generate and audit the E1-E7 dry-run plan.
2. Run E1-E6 across all required folds and five seeds.
3. Resume failed jobs without deleting successful rows.
4. Generate E7 from validation-selected checkpoints.
5. Produce paired statistics, tables, and figures.

### Phase E: Paper and submission

1. Write Methods and Experimental Setup from the frozen pipeline.
2. Write Results from generated tables without inventing values.
3. Discuss false positives, early-fault recall, modality-gate behavior, and
   limitations.
4. Write the Introduction after the evidence and claims are stable.
5. Audit reproducibility, archive the code/configuration/results, and prepare
   the submission package.

## Final Go/No-Go Checklist

The paper is ready only when all items below are true:

- [ ] The physical NLN vibration channel is documented and frozen.
- [ ] All raw files and checksums are validated.
- [ ] All required folds exist and have zero recording leakage.
- [ ] Corrected tensors and train-only normalization are audited.
- [ ] Every real-data smoke test passes.
- [ ] The loss and gate are frozen from validation data only.
- [ ] E1-E6 contain all required folds and five seeds.
- [ ] E7 uses validation-selected checkpoints.
- [ ] Legacy results are absent from all final analyses.
- [ ] Tables, figures, and statistics regenerate from one versioned result set.
- [ ] Every manuscript claim is supported by corrected results.

## Result Interpretation Rule

The final scientific conclusion is not predetermined. A publishable outcome
may show that gating helps, that current is useful only in some conditions, or
that a simpler vibration-only baseline is stronger. Report the corrected
evidence honestly. The fixed contribution is the leakage-safe, severity-aware,
cross-condition evaluation; model-superiority claims depend on the final data.
