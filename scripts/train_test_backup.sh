#!/bin/bash
#PBS -q short-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=01:00:00
#PBS -j oe
#PBS -m abe

set -euo pipefail

DATASET="medmnist-v2:bloodmnist"
MODEL_NAME="google/vit-base-patch16-224"

ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-lora_image/accelerate_config/local_config.yaml}"
export ACCELERATE_CONFIG_FILE


CSV_PATH_DIR="${CSV_PATH_DIR:-test_experiments}"

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-train_logs}"
mkdir -p "${TRAIN_LOG_DIR}"

# 5 two-digit primes as a deterministic "random seed table" for 5 repeated runs.
SEEDS=(13)

for seed in "${SEEDS[@]}"; do
  timestamp="$(date +%Y%m%d%H%M%S)"
  echo "==> Training ${DATASET} with seed=${seed}, timestamp=${timestamp}"

  dataset_safe="${DATASET//\//_}"
  train_log="${TRAIN_LOG_DIR}/train_${dataset_safe}_s${seed}_${timestamp}.log"
  python -m src.cli train \
    --timestamp "${timestamp}" \
    --output_dir "output-medmnist-test" \
    --dataset_name "${DATASET}" \
    --model_name "${MODEL_NAME}" \
    --seed "${seed}" \
    --global_batch_size 64 \
    --per_device_batch_size 64 \
    --max_steps 1000 \
    --learning_rate 2e-4 \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --lora_r 16 \
    --lora_alpha 1 \
    --lora_dropout 0.01 \
    --lora_bias none \
    --target_modules query,key,value,dense \
    --peft_variant lora \
    --init_lora_weights true \
    --init_num_samples 512 \
    --eval_split "val" \
    --eval_steps 25 \
    --eval_batch_size 512 \
    --use_wandb \
    --wandb_online \
    --logging_steps 20 \
    2>&1 | tee "${train_log}"

  model_path="$(awk -F'\t' '/^TRAIN_OUTPUT_DIR\t/ {print $2}' "${train_log}" | tail -n 1)"
  if [[ -z "${model_path}" ]]; then
    echo "Error: failed to parse TRAIN_OUTPUT_DIR from log: ${train_log}" >&2
    exit 1
  fi

  eval_args=(
    -m src.cli evaluate
    --model_path "${model_path}"
    --dataset_name "${DATASET}"
    --test_split "test"
    --image_column img
    --label_column label
    --batch_size 256
    --mixed_precision "bf16"
    --csv_path_dir "${CSV_PATH_DIR}"
  )
  python "${eval_args[@]}"
done
