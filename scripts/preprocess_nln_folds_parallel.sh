#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
source .venv/bin/activate

log_dir="data/interim/preprocess_logs/nln"
mkdir -p "$log_dir"

if (($#)); then
  folds=("$@")
else
  folds=(
    test_speed_100
    test_speed_50
    test_speed_70
    test_speed_75
  )
fi

pids=()
for fold in "${folds[@]}"; do
  output_dir="data/processed/nln_emp/nln_emp_leave_one_speed_out/$fold"
  if [[ -s "$output_dir/preprocessing_manifest.json" ]]; then
    echo "Refusing to overwrite completed fold: $fold" >&2
    exit 1
  fi
  log_file="$log_dir/$fold.log"
  exit_file="$log_dir/$fold.exit"
  rm -f "$log_file" "$exit_file"
  (
    python -m src.data.build_spectrograms \
      --split-file data/splits/nln_emp_leave_one_speed_out.csv \
      --dataset nln_emp \
      --fold-id "$fold" \
      --nln-vibration-channel 2 \
      --nln-current-channels 1 2 3 \
      --tensor-dtype float16 \
      >"$log_file" 2>&1
    code=$?
    echo "$code" >"$exit_file"
    exit "$code"
  ) &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

exit "$status"
