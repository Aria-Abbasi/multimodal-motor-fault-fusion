# Project Roadmap

Last verified: June 14, 2026

Preprocessing provenance revision:
`420407a2695779ebf38a0ed7b321a7729498cc4c`

Pipeline version: `corrected_multimodal_v2`

## Final Goal

Produce a reproducible, publication-ready paper on severity-aware early fault
detection under unseen operating conditions.

- NLN-EMP is the primary multimodal and early-fault dataset.
- Paderborn tests condition generalization and artificial-to-natural transfer.
- CWRU provides a conventional vibration-only benchmark.
- IMS is outside the scope of this paper.

The scientific conclusion is not predetermined. Model-superiority claims will
be made only when the corrected experiments support them.

## Verified Current State

### Code and reproducibility

- The server copies of `signal_io.py`, `build_spectrograms.py`, and
  `configs/base.yaml` used for preprocessing have byte-for-byte hashes matching
  revision `420407a`.
- Those files were present on the server before the completed preprocessing
  jobs. Committing and pushing afterward therefore did not invalidate the
  generated tensors.
- The complete local and CPU-server test suites pass: **46 tests**.
- Synthetic CPU tests cover all model families and E1-E7 experiment paths.
- Legacy result files are marked as unusable and are rejected when they lack
  the current `pipeline_version`.
- Generated processed data and transient inspection logs are no longer tracked
  by Git. Corrected compact metadata and split CSVs remain versioned.

### Raw data and protocols

- The CPU server contains **3,961 raw files** across NLN-EMP, Paderborn, and
  CWRU.
- Raw-data validation found no missing, empty, unreadable, or duplicate-checksum
  groups in the final manifest.
- One Paderborn source recording,
  `KA08/N15_M01_F10_KA08_2.mat`, is defective in the source dataset. An
  independent download had the same checksum, so it is intentionally excluded.
- The metadata master contains **2,808 usable recordings**:
  2,560 Paderborn, 232 NLN-EMP, and 16 CWRU.
- All required recording-level splits exist and pass zero-overlap validation:
  four NLN speed folds, four Paderborn condition folds, one Paderborn
  artificial-to-natural protocol, and four CWRU load folds.

### NLN sensor selection

- The physical vibration input is frozen to **channel 2**, the electric-motor
  drive-end bearing vertical sensor.
- Current channels are frozen to **1, 2, and 3**.
- The selection and its physical interpretation are documented in
  `docs/nln_sensor_selection.md`.

### Corrected processed tensors

| Dataset/protocol | Completed folds | Windows per fold | Approx. size | Audit |
| --- | ---: | ---: | ---: | --- |
| CWRU leave-one-load-out | 4/4 | 1,521 | 406 MB total | Passed |
| NLN leave-one-speed-out | 4/4 | 101,732 | 27 GB total | Passed |
| Paderborn condition generalization | 4/4 | 318,035 | 83 GB total | Passed |
| Paderborn artificial-to-natural | 0/1 | Not generated | About 21 GB expected | Pending |

The completed Paderborn condition folds were checked for:

- exit code zero for every fold;
- exactly 318,035 indexed tensors per fold;
- FP16 tensor shape `(2, 128, 128)`;
- finite sampled tensors;
- synchronized vibration/current channels;
- train-only normalization;
- zero base-recording overlap;
- zero preprocessing exclusions.

The completed NLN folds have four documented missing-modality exclusions per
fold:

- `motor_2_100_stator_short_2`
- `motor_2_50_bearing_bpfi_3`
- `motor_2_50_loose_foot_motor`
- `motor_4_70_cavitation_suction_2`

No preprocessing error manifest was produced for the completed NLN folds.

### CPU server capacity

At the latest check:

- raw data: about 105 GB;
- interim data and retained archives: about 20 GB;
- corrected processed data: about 110 GB;
- free disk space: about 102 GB.

This is sufficient for the remaining Paderborn artificial-to-natural tensors
and the CPU smoke outputs.

## Completed Software Work

- Leakage-safe recording-level folds and split validation.
- Corrected NLN vibration/current pairing and train-only normalization.
- SVM, Random Forest, CNN, LSTM, CNN-LSTM, Transformer, autoencoder, and
  proposed-model execution.
- Six loss configurations: CE weights 1.0, 1.5, 2.0, 3.0, 4.0, and focal loss.
- Optional dynamic current-modality gating.
- AdamW weight decay, warmup/cosine scheduling, and gradient clipping.
- Validation-derived recording thresholds and complete recording metrics.
- Safe `N/A` early recall when Paderborn has no granular early-severity labels.
- Five fixed seeds: 42, 123, 999, 7, and 88.
- Resume-safe E1-E7 planning and versioned result banking.
- E6 recording-level label budgets: 10%, 25%, 50%, and 100%.
- E7 Grad-CAM, saliency, and cross-attention artifact generation.
- Paired fold/seed statistics and validation-only checkpoint selection.

## Remaining CPU Work

### 1. Generate Paderborn artificial-to-natural tensors

Run on the CPU server:

```bash
mkdir -p data/interim/preprocess_logs/paderborn_artificial_to_natural

nohup bash -lc '
  cd /home/Aria/data/multimodal-motor-fault-fusion
  source .venv/bin/activate
  python -m src.data.build_spectrograms \
    --split-file data/splits/paderborn_artificial_to_natural.csv \
    --dataset paderborn \
    --tensor-dtype float16
  code=$?
  echo "$code" > \
    data/interim/preprocess_logs/paderborn_artificial_to_natural/run.exit
  exit "$code"
' > data/interim/preprocess_logs/paderborn_artificial_to_natural/run.log 2>&1 &
```

Check progress directly, without a sleep loop:

```bash
tail -c 300 \
  data/interim/preprocess_logs/paderborn_artificial_to_natural/run.log |
  tr '\r' '\n' | tail -1

cat data/interim/preprocess_logs/paderborn_artificial_to_natural/run.exit
```

Acceptance checks:

- exit code is zero;
- manifest, normalization statistics, paired recordings, and index exist;
- tensor count equals index row count;
- sampled tensors are FP16, finite, and shaped `(2, 128, 128)`;
- artificial recordings occur only in train/validation;
- natural recordings occur only in test;
- no base recording crosses splits.

### 2. Run the real-data CPU smoke suite

After the artificial-to-natural audit:

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

The partial-fold option is permitted only for this smoke test. Confirm that all
protocol/model paths complete, metrics contain no unexpected NaN values, and
Paderborn early recall is reported as unavailable when severity labels are
absent.

### 3. Freeze preprocessing provenance

Archive:

- canonical Git revision and pipeline version;
- raw-data manifest and checksums;
- metadata and split validation reports;
- preprocessing manifests and normalization statistics;
- exclusions and preprocessing logs;
- processed fold sizes and tensor/index counts.

Do not move to paid GPU training until these CPU checks are complete.

## L4 Validation Pilot

1. Synchronize the clean canonical repository revision to the L4 server.
2. Transfer or attach the audited processed-data tree.
3. Run all tests and one GPU smoke job.
4. Run the 12 loss-by-gate configurations on NLN-EMP using seed 42 across all
   four folds. This is 12 configurations and 48 fold-level training jobs.
5. Select exactly one loss/gate configuration using validation recording
   metrics only.

Selection rule:

1. primary metric: validation recording Macro F1;
2. constraint: acceptable validation early-fault recall;
3. tie-breakers: precision, MCC, and fold stability.

Pilot test metrics must not influence configuration selection. Record the
frozen loss, gate state, threshold policy, selection rule, Git revision, and
pipeline version before final training.

## L4 Final Experiments

Generate and inspect the dry-run plan before training:

```bash
python -m src.training.paper_experiment_runner \
  --experiments E1 E2 E3 E4 E5 E6 E7 \
  --frozen-loss FROZEN_LOSS \
  --frozen-gate \
  --dry-run \
  --plan-file results/tables/final_experiment_plan.csv
```

Use `--no-frozen-gate` if validation selects gate off.

Then execute E1-E6 across all required folds and five seeds. Generate E7 only
from validation-selected checkpoints.

- **E1:** fair comparison against all baseline families.
- **E2:** vibration, current, fusion, and gate ablations.
- **E3:** NLN, Paderborn condition, and CWRU generalization.
- **E4:** curriculum ablation.
- **E5:** Paderborn artificial-to-natural robustness.
- **E6:** recording-level label-budget experiments.
- **E7:** Grad-CAM, saliency, and cross-attention explanations.

The runner must resume completed jobs, preserve successful rows, reuse
identical training jobs, cache one dataset/fold at a time, and release the
cache before loading the next dataset/fold.

## Reporting and Paper

After all planned result cells are present:

1. generate paired fold/seed statistics with Holm correction;
2. report fold variation separately from seed variation;
3. regenerate tables and figures directly from one versioned result set;
4. write Methods and Experimental Setup from the frozen pipeline;
5. write Results from generated outputs without manual row edits;
6. discuss false positives, early recall, gating behavior, and limitations;
7. archive code, configurations, manifests, checkpoints, and final results.

A p-value below 0.05 is supporting evidence, not a license to selectively
rerun seeds or configurations.

## Go/No-Go Checklist

- [x] Corrected code is committed and synchronized.
- [x] Local and CPU-server test suites pass.
- [x] NLN physical sensor channels are documented and frozen.
- [x] Raw files and checksums are validated.
- [x] Metadata and all required recording-level splits are validated.
- [x] CWRU corrected folds are generated.
- [x] NLN corrected folds are generated and audited.
- [x] Paderborn condition folds are generated and audited.
- [x] Generated data is removed from Git tracking.
- [ ] Paderborn artificial-to-natural tensors are generated and audited.
- [ ] The real-data CPU smoke suite passes.
- [ ] Preprocessing provenance is frozen and archived.
- [ ] The loss and gate are frozen from validation data only.
- [ ] E1-E6 contain every planned fold and seed.
- [ ] E7 uses validation-selected checkpoints.
- [ ] Final analyses contain no legacy results.
- [ ] Tables, figures, and statistics regenerate from one result set.
- [ ] Every manuscript claim is supported by the corrected evidence.

## Immediate Next Action

Generate and audit the single Paderborn artificial-to-natural processed
dataset, then run the real-data CPU smoke suite.
