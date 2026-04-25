# Project Charter (Paper 1)

## Title

Severity-Aware Cross-Condition Early Fault Detection in Electric Motor Systems Using Vibration-Current Cross-Attention Learning

## 1) One-sentence paper claim

A severity-aware multimodal model that fuses vibration and current signals improves early-fault detection, especially early-fault recall, under unseen operating conditions using leakage-safe recording-level evaluation.

## 2) Research question

Can a severity-aware vibration-current fusion model detect incipient faults more reliably than strong classical and deep baselines when tested on unseen operating conditions with strict recording-level, leakage-safe splits?

## 3) Hypotheses

- **H1 (Fusion):** Vibration + current fusion outperforms vibration-only and current-only models on early-fault detection metrics.
- **H2 (Severity-aware learning):** A two-stage severity-aware curriculum improves early-fault recall versus equivalent non-curriculum training.
- **H3 (Generalization):** The proposed model generalizes better than baseline models under unseen speed/load/condition and seeded-to-natural transfer settings.

## 4) Frozen dataset roles (Paper 1)

- **NLN-EMP (main dataset):** Primary evidence source for model development, ablations, and main claims.
- **Paderborn (external robustness + transfer):** External validation for operating-condition robustness and artificial-to-natural transfer.
- **CWRU (benchmark only):** Familiar benchmark for comparability, not the paper centerpiece.
- **IMS:** Explicitly excluded from Paper 1; reserved for a future prognostics/RUL-focused paper.

## 5) Frozen evaluation rules

- Main evaluation must use **recording-level, leakage-safe splits**.
- Windows from the same original recording must never be split across train/validation/test.
- Central paper metric is **early-fault recall under unseen operating conditions**.
- Supporting metrics: Macro F1, balanced accuracy, AUROC, AUPRC, MCC (reported with mean ± std across seeds).

## 6) Non-goals (Paper 1)

- No full prognostics/RUL modeling (IMS deferred to Paper 2).
- No claim of deployment-ready industrial product.
- No mixed-scope paper that combines diagnosis and prognostics in one contribution.
- No dependence on random window-level splits or accuracy-only reporting.

## 7) Experiment package (E1-E7)

- **E1: Main comparison on NLN-EMP** against classical and deep baselines.
- **E2: Modality ablation** (vibration-only, current-only, fusion).
- **E3: Cross-condition generalization**
  - NLN-EMP leave-one-speed-out,
  - Paderborn leave-one-condition-out,
  - CWRU leave-one-load-out.
- **E4: Severity curriculum ablation** (no curriculum vs stage variants vs full two-stage).
- **E5: Paderborn artificial-to-natural transfer** (train on artificial, test on natural faults).
- **E6: Limited-label robustness** (10%, 25%, 50%, 100% label budgets).
- **E7: Explainability** (attention/Grad-CAM style inspection on spectrograms).

## 8) Success criteria for submission readiness

The manuscript is submission-ready when all criteria below are met:

1. **Leakage-safe protocol proof:** split validator reports zero recording overlap across train/val/test.
2. **Competitive performance:** proposed method is competitive with all required baselines.
3. **Core claim support:** early-fault recall improves in unseen-condition evaluations.
4. **External robustness evidence:** method remains stable on Paderborn robustness/transfer tests.
5. **Complete evidence package:** required tables/figures for E1-E7 are generated and reproducible from scripts.
6. **Reproducibility package:** configs, seeds, split definitions, and run logs allow rerunning main results.

## 9) 14-week milestone plan

### Weeks 1-2: Scope + data foundation
- Freeze title, claim, hypotheses, and dataset roles.
- Build raw data manifest and unified recording-level metadata.
- Define and validate leakage-safe split protocols.

### Weeks 3-4: Preprocessing + classical floor
- Implement windowing/spectrogram pipeline with train-only normalization stats.
- Run SVM/Random Forest baselines with fixed metrics.

### Weeks 5-6: Initial deep modeling
- Implement deep baselines and multimodal fusion model v1.
- Run first modality ablations on NLN-EMP.

### Week 7: Severity-aware training
- Add two-stage severity curriculum and modality dropout.
- Execute curriculum ablation and compare early-fault recall.

### Week 8: Main NLN-EMP runs
- Execute leave-one-speed-out experiments with fixed hyperparameters.
- Freeze candidate model configuration for cross-dataset tests.

### Week 9: Paderborn robustness/transfer
- Run condition generalization and artificial-to-natural transfer experiments.

### Week 10: CWRU benchmark
- Run smallest-defect early-fault benchmark under leave-one-load-out.

### Week 11: Statistical validation
- Run 5-seed repeats and significance testing.
- Generate explainability outputs and complexity/inference metrics.

### Week 12: Methods + setup writing
- Draft Methods and Experimental Setup from finalized configs/logs.

### Week 13: Full draft completion
- Draft Introduction, Related Work, Results, Discussion, Conclusion.
- Prepare cover letter and submission package v1.

### Week 14: Final QA + submission
- Reproducibility audit, figure/table quality check, and language polishing.
- Journal fit check and submission (MSSP first target, Sensors fallback).

## 10) Frozen decisions summary

- Paper 1 is diagnosis-focused (not prognostics).
- NLN-EMP is the main dataset.
- Paderborn is the external robustness/transfer dataset.
- CWRU is benchmark-only support.
- IMS is deferred to Paper 2 (prognostics/RUL).
- Main protocol must be recording-level and leakage-safe.
- Core success signal is early-fault recall under unseen operating conditions.