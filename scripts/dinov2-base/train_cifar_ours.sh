#!/usr/bin/env bash
# Recommended Accelerate launch for ViT-L/16 @384 with standard LoRA on VTAB-1K CIFAR.
# Adjust NUM_PROCS, dataset, and logging as needed.

set -euo pipefail

DATASET="caltech101"
MODEL_NAME="facebook/dinov2-base"

ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-/home/yu/workspace/lora_image/accelerate_config/local_config.yaml}"
export ACCELERATE_CONFIG_FILE
PYTHON_BINARY="${PYTHON_BINARY:-/home/yu/peft_playground/.venv/bin/python}"

SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
CACHE_DIR="${CACHE_DIR:-}"
CSV_PATH_DIR="${CSV_PATH_DIR:-test_experiments}"

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-train_logs}"
mkdir -p "${TRAIN_LOG_DIR}"

# 5 two-digit primes as a deterministic "random seed table" for 5 repeated runs.
SEEDS=(11 23 37 47 53)

n=(2 3 4 5)

for i in "${n[@]}"; do
  for seed in "${SEEDS[@]}"; do
    timestamp="$(date +%Y%m%d%H%M%S)"
    echo "==> Training ${DATASET} with seed=${seed}, timestamp=${timestamp}"

    dataset_safe="${DATASET//\//_}"
    train_log="${TRAIN_LOG_DIR}/train_${dataset_safe}_s${seed}_${timestamp}.log"
    "${PYTHON_BINARY}" -m src.cli train \
      --timestamp "${timestamp}" \
      --output_dir "test_out1" \
      --dataset_name "${DATASET}" \
      --model_name "${MODEL_NAME}" \
      --seed "${seed}" \
      --global_batch_size 64 \
      --per_device_batch_size 64 \
      --max_steps 500 \
      --learning_rate 2e-4 \
      --weight_decay 0.05 \
      --warmup_ratio 0.1 \
      --lora_r 16 \
      --lora_alpha 1 \
      --lora_dropout 0.05 \
      --lora_bias none \
      --target_modules query,key,value,dense \
      --peft_variant lora \
      --init_lora_weights olora \
      --logging_steps 20 \
      --eval_steps 20 \
      --use_cleaned_svd_ref_trainer \
      --repeat_n "${i}" \
      2>&1 | tee "${train_log}"

    model_path="$(awk -F'\t' '/^TRAIN_OUTPUT_DIR\t/ {print $2}' "${train_log}" | tail -n 1)"
    if [[ -z "${model_path}" ]]; then
      echo "Error: failed to parse TRAIN_OUTPUT_DIR from log: ${train_log}" >&2
      exit 1
    fi

    eval_args=(
      -m src.cli evaluate
      --model_path "${model_path}"
      --test_split "${SPLIT}"
      --image_column img
      --label_column label
      --batch_size "${BATCH_SIZE}"
      --mixed_precision "${MIXED_PRECISION}"
      --csv_path_dir "${CSV_PATH_DIR}"
    )
    if [[ -n "${CACHE_DIR}" ]]; then
      eval_args+=(--cache_dir "${CACHE_DIR}")
    fi
    "${PYTHON_BINARY}" "${eval_args[@]}"
  done
done