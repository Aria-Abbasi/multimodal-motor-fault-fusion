#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
source .venv/bin/activate

log_dir="data/interim/preprocess_logs/paderborn_condition"
mkdir -p "$log_dir"

folds=(
  test_condition_n09_m07_f10
  test_condition_n15_m01_f10
  test_condition_n15_m07_f04
  test_condition_n15_m07_f10
)

pids=()
for fold in "${folds[@]}"; do
  output_dir="data/processed/paderborn/paderborn_condition_generalization/$fold"
  if [[ -s "$output_dir/preprocessing_manifest.json" ]]; then
    echo "Refusing to overwrite completed fold: $fold" >&2
    exit 1
  fi
  log_file="$log_dir/$fold.log"
  exit_file="$log_dir/$fold.exit"
  rm -f "$log_file" "$exit_file"
  (
    python -m src.data.build_spectrograms \
      --split-file data/splits/paderborn_condition_generalization.csv \
      --dataset paderborn \
      --fold-id "$fold" \
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
