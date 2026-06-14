# Project Handoff

Last updated: June 15, 2026

## Current Objective

Complete a reproducible, publication-ready study of severity-aware early fault
detection under unseen operating conditions. The next phase is GPU training on
a Google Cloud NVIDIA L4 after a validation-only pilot selects the loss and
modality-gate configuration.

The paper uses:

- NLN-EMP as the primary multimodal early-fault dataset;
- Paderborn for condition generalization and artificial-to-natural transfer;
- CWRU as a vibration-only benchmark;
- IMS is outside this paper's scope.

Do not assume the proposed model must win. Claims must follow the corrected
fold-and-seed results.

## Repository State

- Repository: `Aria-Abbasi/multimodal-motor-fault-fusion`
- Ready revision: `088c8f6`
- Branch: `master`
- Local, CPU server, and GitHub were synchronized at that revision.
- GPU pipeline version: `corrected_multimodal_v3`
- Preprocessing/smoke provenance version: `corrected_multimodal_v2`
- `m5.md` is an intentionally untracked historical plan. Preserve it.
- `PROJECT_ROADMAP.md` is the current scientific and operational roadmap.
- `docs/cloud_runbook.md` contains the exact L4 migration and training commands.

## Verified Completed Work

- All leakage-safe recording-level splits are complete:
  - four NLN leave-one-speed-out folds;
  - four Paderborn condition folds;
  - one Paderborn artificial-to-natural protocol;
  - four CWRU leave-one-load-out folds.
- All 13 processed folds are generated and audited.
- Total processed tensors: `2,003,187`.
- Processed tree: about `126.4 GiB`.
- Raw data, metadata, checksums, tensor counts, normalization, and overlap
  checks passed.
- One defective Paderborn source file is intentionally excluded:
  `KA08/N15_M01_F10_KA08_2.mat`.
- NLN vibration channel is frozen to channel 2.
- NLN current channels are frozen to channels 1, 2, and 3.
- The real-data CPU smoke suite completed all 105 jobs.
- The current test suite passes 52 tests locally and on the CPU server.
- The full real-data plan was generated successfully:
  - 525 result rows;
  - 425 unique training signatures;
  - E1: 160 rows;
  - E2: 80 rows;
  - E3: 60 rows;
  - E4: 60 rows;
  - E5: 5 rows;
  - E6: 160 rows.

## Implemented Corrections

- Six losses: CE early weights 1.0, 1.5, 2.0, 3.0, 4.0, and dynamic focal.
- Optional learnable current-modality gate.
- AdamW weight decay.
- Linear warmup plus cosine learning-rate decay.
- Gradient clipping.
- Validation-calibrated recording-level decision thresholds.
- Window-level and recording-level fault precision.
- Safe `N/A` early recall when severity labels are unavailable.
- CUDA-required execution guard to prevent silent CPU training.
- Validation-only pilot selector.
- Final runner requires a versioned frozen loss/gate YAML.
- Resume-safe result banking and cross-experiment training reuse.
- ROC and precision-recall artifacts.

## Frozen Experimental Procedure

### Validation Pilot

Run the 12 loss-by-gate configurations using seed 42 over all four NLN folds:

- 6 losses x 2 gate states x 4 folds = 48 training jobs.

Select a configuration using validation metrics only:

1. Require mean validation recording early-fault recall >= 0.95.
2. Maximize validation recording Macro F1.
3. Tie-break using fault precision, MCC, lower between-fold variation, and
   experiment name.
4. Never use test metrics to choose the configuration.

The selector creates `configs/frozen_l4_selection.yaml`.

### Final Training

Use five fixed seeds: `42`, `123`, `999`, `7`, and `88`.

- E1: all classical/deep baselines versus the proposed model.
- E2: vibration, current, fusion, and gate ablations.
- E3: NLN, Paderborn condition, and CWRU generalization.
- E4: curriculum ablation.
- E5: Paderborn artificial-to-natural transfer.
- E6: recording-level 10%, 25%, 50%, and 100% label budgets.
- E7: validation-selected Grad-CAM, saliency, and attention explanations.

## Infrastructure

- Google Cloud project: `project-c90a5081-79c6-4ba6-9f2`
- Zone: `europe-west4-c`
- CPU VM: `cpu-20260425-122010`
- CPU repository:
  `/home/Aria/data/multimodal-motor-fault-fusion`
- Persistent data disk: `data-nether-20260525-204259`
- Disk: 350 GB balanced persistent disk, about 81 GB free
- Target GPU VM: `g2-standard-16`
- GPU: one NVIDIA L4 with 24 GB VRAM
- System RAM: 64 GB
- RAM cache limit: 48 GB

The 48 GB limit is safe: tensors remain FP16 in system RAM and are converted
to FP32 per batch. The runner loads one fold at a time and releases it before
the next fold or dataset.

Preprocessing provenance:

`artifacts/provenance/preprocessing_20260614.tar.gz`

SHA-256:

`aea7de39edef3bf27a9c4bbc1bf64eb9bde3992c9ff1221f3f9b486be24a4933`

## Immediate Next Action

Follow `docs/cloud_runbook.md` in order:

1. Confirm local, CPU server, and GitHub still point to revision `088c8f6`.
2. Snapshot `data-nether-20260525-204259`.
3. Stop the CPU VM and detach only its non-boot data disk.
4. Create `motor-fault-l4` as `g2-standard-16` in `europe-west4-c`.
5. Attach and mount the existing data disk at `/home/Aria/data`.
6. Do not reuse the CPU `.venv`; it contains CPU-only PyTorch.
7. Verify CUDA and run all 52 tests.
8. Run `python scripts/l4_preflight.py --full-tensor-count`.
9. Run the single GPU smoke command from the runbook.
10. Start the 48-job validation pilot only if preflight and smoke both pass.

Do not start the final 525-row experiment plan before committing the
validation-selected `configs/frozen_l4_selection.yaml`.

## Remaining Publication Work

After final GPU training:

- generate paired Wilcoxon tests with Holm correction;
- report fold variance separately from seed variance;
- add 95% confidence intervals and paired effect sizes;
- regenerate all tables and figures from the v3 result file;
- create framework and split-protocol diagrams;
- report calibrated thresholds and learned gate distributions;
- generate E7 from validation-selected checkpoints only;
- document runtime, compute environment, failure recovery, exclusions, and
  limitations;
- write manuscript claims only after inspecting corrected evidence.

## Safety Rules

- Do not use legacy result CSVs or figures for the paper.
- Do not modify or regenerate the audited processed tensors unless an audit
  fails.
- Do not lower the pilot's 0.95 early-recall requirement after viewing test
  results.
- Do not tune thresholds or choose checkpoints on test data.
- Do not run final training without `--require-cuda`.
- Do not delete the persistent data disk after deleting or stopping a VM.
- Snapshot the disk before changing ownership or moving it to the L4.
