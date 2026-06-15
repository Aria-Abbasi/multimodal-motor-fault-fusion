# L4 Training Runbook

Last verified: June 14, 2026

## Frozen Infrastructure

- Project: `project-c90a5081-79c6-4ba6-9f2`
- Zone: `europe-west4-c`
- GPU VM: `g2-standard-16`
- GPU: one NVIDIA L4 with 24 GB VRAM
- System RAM: 64 GB
- Existing data disk: `data-nether-20260525-204259`
- Data disk size: 350 GB balanced persistent disk
- Recommended RAM cache limit: 48 GB

The project currently has quota for one standard L4 in `europe-west4` and one
GPU globally. L4 and `g2-standard-16` are available in `europe-west4-c`.

The corrected processed tree is about 126.4 GiB with 2,003,187 tensors. Do not
copy those files individually. Stop the CPU VM, detach the non-boot data disk,
and attach it to the L4 VM in the same zone. Take a snapshot before moving the
disk.

## 1. Before Creating the L4

The repository must contain the L4-readiness changes and be pushed to GitHub.
The CPU server and GitHub should point to the same commit.

On the CPU server:

```bash
cd /home/Aria/data/multimodal-motor-fault-fusion
git status --short
git rev-parse HEAD
git rev-parse origin/master
sha256sum artifacts/provenance/preprocessing_20260614.tar.gz
```

Expected provenance SHA-256:

```text
aea7de39edef3bf27a9c4bbc1bf64eb9bde3992c9ff1221f3f9b486be24a4933
```

Create a safety snapshot:

```bash
gcloud compute disks snapshot data-nether-20260525-204259 \
  --zone europe-west4-c \
  --snapshot-names motor-fault-data-pre-l4-20260614 \
  --project project-c90a5081-79c6-4ba6-9f2
```

Stop the CPU VM and detach only the non-boot data disk:

```bash
gcloud compute instances stop cpu-20260425-122010 \
  --zone europe-west4-c \
  --project project-c90a5081-79c6-4ba6-9f2

gcloud compute instances detach-disk cpu-20260425-122010 \
  --disk data-nether-20260525-204259 \
  --zone europe-west4-c \
  --project project-c90a5081-79c6-4ba6-9f2
```

## 2. Create the L4 VM

The current image family verified on June 14, 2026 is:

```text
pytorch-2-9-cu129-ubuntu-2404-nvidia-580
```

Create the VM with a 120 GB boot disk and attach the existing data disk:

```bash
gcloud compute instances create motor-fault-l4 \
  --project project-c90a5081-79c6-4ba6-9f2 \
  --zone europe-west4-c \
  --machine-type g2-standard-16 \
  --maintenance-policy TERMINATE \
  --image-family pytorch-2-9-cu129-ubuntu-2404-nvidia-580 \
  --image-project deeplearning-platform-release \
  --boot-disk-size 120GB \
  --boot-disk-type pd-balanced \
  --disk name=data-nether-20260525-204259,device-name=motor-fault-data,mode=rw,boot=no,auto-delete=no
```

Do not use Spot/Preemptible capacity for the first pilot. The runner is
resume-safe, but an uninterrupted pilot makes validation selection easier to
audit.

## 3. Mount and Validate

Mount the attached disk at the same logical location used during preprocessing:

```bash
sudo mkdir -p /home/Aria/data
sudo mount /dev/disk/by-id/google-motor-fault-data /home/Aria/data

DATA_UUID="$(sudo blkid -s UUID -o value \
  /dev/disk/by-id/google-motor-fault-data)"
echo "UUID=${DATA_UUID} /home/Aria/data ext4 defaults,nofail 0 2" |
  sudo tee -a /etc/fstab

findmnt /home/Aria/data
stat -c '%u:%g %U:%G %n' \
  /home/Aria/data/multimodal-motor-fault-fusion
```

The current data tree is owned numerically by UID 1002 and GID 1003. If the
new `Aria` account has different IDs and cannot write the repository, run the
following once after confirming the snapshot completed:

```bash
sudo chown -R "$(id -u):$(id -g)" /home/Aria/data
```

Do not activate or copy the existing `.venv`; it contains CPU-only PyTorch.
Create a boot-disk virtual environment that inherits the CUDA-enabled PyTorch
supplied by the Deep Learning VM, then install the remaining dependencies:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv
python3 -m venv --system-site-packages /home/Aria/gpu-venv
source /home/Aria/gpu-venv/bin/activate

cd /home/Aria/data/multimodal-motor-fault-fusion
python -m pip install -e . --no-deps
python -m pip install \
  joblib kagglehub matplotlib numpy openpyxl pandas PyYAML \
  scikit-learn scipy seaborn tqdm pytest
```

Verify:

```bash
nvidia-smi

python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_properties(0).total_memory / 1024**3)
PY

python -m pytest -q

python scripts/l4_preflight.py --full-tensor-count
```

The preflight must report:

- `ready: true`;
- NVIDIA L4;
- at least 22 GiB GPU memory;
- at least 48 GiB system RAM;
- 13 processed folds;
- 2,003,187 processed windows;
- 48 pilot jobs;
- 525 planned final result rows.

## 4. GPU Smoke Test

Run one direct proposed-model smoke test before the pilot:

```bash
python -m src.training.experiment_runner \
  --protocol nln_emp \
  --folds test_speed_100 \
  --losses ce_1.0 \
  --seeds 42 \
  --smoke-test \
  --require-cuda \
  --cache-max-gb 1 \
  --output-file results/tables/l4_gpu_smoke.csv \
  --summary-file results/tables/l4_gpu_smoke_summary.csv \
  --checkpoint-dir artifacts/checkpoints/l4_gpu_smoke \
  --fail-fast
```

Confirm the result records `device=cuda`, an L4 `gpu_name`, and pipeline
version `corrected_multimodal_v3`.

## 5. Validation Pilot

Run exactly 12 loss/gate configurations, seed 42, and all four NLN folds:

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

This produces 48 fold-level jobs. Do not use pilot test metrics to select the
configuration.

Freeze the configuration from validation metrics only:

```bash
python -m src.training.pilot_selection \
  --results results/tables/nln_validation_pilot.csv \
  --output configs/frozen_l4_selection.yaml \
  --summary results/tables/nln_validation_pilot_selection.csv \
  --expected-seed 42 \
  --expected-folds 4 \
  --minimum-early-recall 0.95
```

Review and commit:

```bash
git add configs/frozen_l4_selection.yaml \
  results/tables/nln_validation_pilot.csv \
  results/tables/nln_validation_pilot_summary.csv \
  results/tables/nln_validation_pilot_selection.csv
git commit -m "exp: freeze validation-selected loss and gate"
git push origin master
```

If no configuration reaches 0.95 validation recording early-fault recall, the
selector fails intentionally. Do not lower the threshold after viewing test
metrics.

## 6. Final E1-E6 Training

Generate the frozen plan:

```bash
python -m src.training.paper_experiment_runner \
  --experiments E1 E2 E3 E4 E5 E6 \
  --frozen-config configs/frozen_l4_selection.yaml \
  --require-cuda \
  --cache-max-gb 48 \
  --dry-run \
  --plan-file results/tables/final_experiment_plan.csv
```

Expected:

- 525 result rows;
- 425 unique training signatures after cross-experiment reuse;
- 480 NLN rows;
- 20 Paderborn condition rows;
- 20 CWRU rows;
- 5 Paderborn artificial-to-natural rows.

Launch in a detached session:

```bash
tmux new-session -d -s final_e1_e6 \
  "cd /home/Aria/data/multimodal-motor-fault-fusion && \
   source /home/Aria/gpu-venv/bin/activate && \
   python -m src.training.paper_experiment_runner \
     --experiments E1 E2 E3 E4 E5 E6 \
     --frozen-config configs/frozen_l4_selection.yaml \
     --require-cuda \
     --cache-max-gb 48 \
     --output-file results/tables/corrected_paper_experiments.csv \
     --checkpoint-dir artifacts/checkpoints/corrected_paper \
     --plan-file results/tables/final_experiment_plan.csv \
     --fail-fast \
     > artifacts/logs/final_e1_e6.log 2>&1"
```

The runner caches one fold at a time, banks every completed row atomically,
and releases the fold cache before moving to the next fold or dataset.

## 7. E7 and Reporting

After all 525 planned rows are complete:

```bash
python -m src.evaluation.explainability \
  --results results/tables/corrected_paper_experiments.csv

python -m src.evaluation.prediction_artifacts \
  --results results/tables/corrected_paper_experiments.csv

python -m src.evaluation.reporting \
  --results results/tables/corrected_paper_experiments.csv
```

Generate paired statistics only on identical fold/seed cells and apply Holm
correction. Archive results, checkpoints, the frozen selection YAML, plan,
logs, environment versions, and GPU details before stopping the L4.
