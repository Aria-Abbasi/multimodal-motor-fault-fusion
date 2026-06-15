# Project Roadmap

Last verified: June 14, 2026

Preprocessing provenance revision:
`420407a2695779ebf38a0ed7b321a7729498cc4c`

GPU training pipeline version: `corrected_multimodal_v3`

Audited preprocessing/smoke provenance version: `corrected_multimodal_v2`

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
- A real-data smoke-selection regression test was added after the CPU smoke
  run exposed contiguous single-class sampling. The current local suite passes
  **52 tests**; the same revision must pass on the L4 before training.
- Synthetic CPU tests cover all model families and E1-E7 experiment paths.
- Legacy result files are marked as unusable and are rejected when they lack
  the current `pipeline_version`.
- Generated processed data and transient inspection logs are no longer tracked
  by Git. Corrected compact metadata and split CSVs remain versioned.
- GPU runs now fail when `--require-cuda` is set and CUDA is unavailable,
  preventing accidental CPU fallback.
- The pilot configuration is frozen by a validation-only selector and the
  final runner rejects missing or incompatible frozen-selection YAML.

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
| Paderborn artificial-to-natural | 1/1 | 318,035 | 21 GB total | Passed |

The completed Paderborn condition folds were checked for:

- exit code zero for every fold;
- exactly 318,035 indexed tensors per fold;
- FP16 tensor shape `(2, 128, 128)`;
- finite sampled tensors;
- synchronized vibration/current channels;
- train-only normalization;
- zero base-recording overlap;
- zero preprocessing exclusions.

The Paderborn artificial-to-natural protocol was checked for:

- exactly 318,035 indexed and physical tensors;
- FP16 tensor shape `(2, 128, 128)` and finite sampled tensors;
- train-only normalization from 1,087 training recordings;
- artificial faults only in train/validation;
- real-damage faults only in test;
- zero base-recording overlap;
- zero preprocessing errors or exclusions.

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
- corrected processed data: about 131 GB;
- free disk space: about 81 GB.

This is sufficient for the remaining provenance and transfer work.

## Completed Software Work

- Leakage-safe recording-level folds and split validation.
- Corrected NLN vibration/current pairing and train-only normalization.
- SVM, Random Forest, CNN, LSTM, CNN-LSTM, Transformer, autoencoder, and
  proposed-model execution.
- Six loss configurations: CE weights 1.0, 1.5, 2.0, 3.0, 4.0, and focal loss.
- Optional dynamic current-modality gating.
- AdamW weight decay, warmup/cosine scheduling, and gradient clipping.
- Validation-derived recording thresholds and complete recording metrics.
- Fault precision at window and recording level, explicitly tracking the
  false-positive failure mode.
- Safe `N/A` early recall when Paderborn has no granular early-severity labels.
- Five fixed seeds: 42, 123, 999, 7, and 88.
- Resume-safe E1-E7 planning and versioned result banking.
- E6 recording-level label budgets: 10%, 25%, 50%, and 100%.
- E7 Grad-CAM, saliency, and cross-attention artifact generation.
- Paired fold/seed statistics and validation-only checkpoint selection.

## CPU Phase Complete

The real-data CPU smoke suite completed all **105 planned jobs** using seed 42:

- E1: 32 jobs;
- E2: 16 jobs;
- E3: 12 jobs;
- E4: 12 jobs;
- E5: 1 job;
- E6: 32 jobs.

All jobs completed, all required metrics were finite, no run IDs were missing
or duplicated, and every row used `corrected_multimodal_v2`. Paderborn and
CWRU early recall were unavailable rather than incorrectly reported as zero.
These smoke metrics verify execution only and must not be used as paper
results.

Preprocessing and smoke provenance is frozen in:

`artifacts/provenance/preprocessing_20260614.tar.gz`

Archive SHA-256:

`aea7de39edef3bf27a9c4bbc1bf64eb9bde3992c9ff1221f3f9b486be24a4933`

The archive contains the raw manifest, metadata, splits, preprocessing
manifests, normalization statistics, pairing/exclusion records, logs, smoke
plan/results, source hashes, and a 13-fold tensor/index inventory. Across all
folds it records **2,003,187 tensors**, with no count mismatches.

## L4 Validation Pilot

Verified infrastructure:

- one standard NVIDIA L4 quota in `europe-west4`;
- `motor-fault-l4` running as `g2-standard-16` in `europe-west4-c`;
- existing 350 GB persistent data disk in `europe-west4-c`;
- 126.4 GiB processed tree and about 81 GB currently free.
- final L4 preflight passed with all 2,003,187 tensors counted;
- both BF16 gate-off/on CUDA smoke jobs completed with finite gradients.

Execution:

1. Push the complete v3 readiness revision to GitHub.
2. Snapshot the 350 GB data disk.
3. Stop the CPU VM, detach its non-boot data disk, and attach it to a
   `g2-standard-16` L4 VM in `europe-west4-c`.
4. Run all tests, `scripts/l4_preflight.py --full-tensor-count`, and one GPU
   smoke job with `--require-cuda`.
5. Run the 12 loss-by-gate configurations on NLN-EMP using seed 42 across all
   four folds. This is 12 configurations and 48 fold-level training jobs.
6. Run `src.training.pilot_selection` with the predeclared 0.95 minimum
   validation recording early-fault recall.
7. Review and commit `configs/frozen_l4_selection.yaml`.

Selection rule:

1. primary metric: validation recording Macro F1;
2. constraint: validation recording early-fault recall >= 0.95;
3. tie-breakers: precision, MCC, and fold stability.

Pilot test metrics must not influence configuration selection. The selector
records the loss, gate state, threshold policy, selection rule, source-results
checksum, Git revision, and pipeline version before final training.

## L4 Final Experiments

Generate and inspect the dry-run plan before training:

```bash
python -m src.training.paper_experiment_runner \
  --experiments E1 E2 E3 E4 E5 E6 E7 \
  --frozen-config configs/frozen_l4_selection.yaml \
  --require-cuda \
  --cache-max-gb 48 \
  --dry-run \
  --plan-file results/tables/final_experiment_plan.csv
```

The expected plan has 525 result rows and 425 unique training signatures after
cross-experiment reuse.

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

Publication work still required after GPU training, beyond the original
`m5.md` checklist:

- report 95% confidence intervals and paired effect sizes alongside corrected
  p-values;
- create the framework and split-protocol diagrams from the frozen methods;
- include both ROC and precision-recall curves;
- report calibrated decision thresholds and gate-value distributions;
- document compute energy/runtime, failure handling, and the defective
  Paderborn recording as reproducibility limitations.

## Go/No-Go Checklist

- [x] The v3 L4-readiness code is committed and synchronized.
- [x] Local and CPU-server test suites pass.
- [x] NLN physical sensor channels are documented and frozen.
- [x] Raw files and checksums are validated.
- [x] Metadata and all required recording-level splits are validated.
- [x] CWRU corrected folds are generated.
- [x] NLN corrected folds are generated and audited.
- [x] Paderborn condition folds are generated and audited.
- [x] Generated data is removed from Git tracking.
- [x] Paderborn artificial-to-natural tensors are generated and audited.
- [x] The real-data CPU smoke suite passes.
- [x] Preprocessing provenance is frozen and archived.
- [x] L4 quota, zone availability, disk, and experiment counts are verified.
- [x] Validation-only pilot freezing and CUDA preflight safeguards are implemented.
- [x] The L4 preflight and BF16 GPU smoke test pass.
- [ ] The loss and gate are frozen from validation data only.
- [ ] E1-E6 contain every planned fold and seed.
- [ ] E7 uses validation-selected checkpoints.
- [ ] Final analyses contain no legacy results.
- [ ] Tables, figures, and statistics regenerate from one result set.
- [ ] Every manuscript claim is supported by the corrected evidence.

## Immediate Next Action

Push the post-deployment BF16 stability commits, then run the 48-job NLN
validation pilot and freeze the loss/gate choice from validation metrics only.
