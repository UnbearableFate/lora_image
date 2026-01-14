#!/bin/bash
#PBS -q short-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=02:30:00
#PBS -j oe
#PBS -m abe

set -euo pipefail

cd /work/xg24i002/x10041/lora_image

MODEL_NAME="facebook/dinov2-base"
CSV_PATH_DIR="experiments_lr_search"

ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-lora_image/accelerate_config/local_config.yaml}"
export ACCELERATE_CONFIG_FILE
PYTHON_BINARY="${PYTHON_BINARY:-/work/xg24i002/x10041/lora-ns/.venv/bin/python}"

SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
CACHE_DIR="${CACHE_DIR:-}"

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-train_logs}"
mkdir -p "${TRAIN_LOG_DIR}"

# 5 two-digit primes as a deterministic "random seed table" for 5 repeated runs.
SEEDS=(11 23 37)
lrs=(2.5e-3 1e-3 7.5e-4 5e-4 2.5e-4)

# DATASETS=(
#   "cifar"
#   "caltech101"
#   "dtd"
#   "flowers102"
#   "pets"
#   "svhn"
#   "sun397"
#   "patch_camelyon"
#   "eurosat"
#   "resisc45"
#   "retinopathy"
#   "clevr_count"
#   "clevr_distance"
#   "dsprites_location"
#   "dsprites_orientation"
#   "smallnorb_azimuth"
#   "smallnorb_elevation"
#   "kitti_distance"
# )

dataset="resisc45"

for seed in "${SEEDS[@]}"; do
  for lr in "${lrs[@]}"; do
    timestamp="$(date +%Y%m%d%H%M%S)"
    echo "==> Training ${dataset} with init_lora_weights=${init_lora_weights}, seed=${seed}, timestamp=${timestamp}"

    dataset_safe="${dataset//\//_}"
    train_log="${TRAIN_LOG_DIR}/train_${dataset_safe}_s${seed}_${timestamp}.log"
    "${PYTHON_BINARY}" -m src.cli train \
      --timestamp "${timestamp}" \
      --output_dir "outputs" \
      --dataset_name "${dataset}" \
      --model_name "${MODEL_NAME}" \
      --seed "${seed}" \
      --global_batch_size 64 \
      --per_device_batch_size 64 \
      --max_steps 500 \
      --learning_rate "${lr}" \
      --weight_decay 0.05 \
      --warmup_ratio 0.1 \
      --lora_r 16 \
      --lora_alpha 1 \
      --lora_dropout 0.05 \
      --lora_bias none \
      --target_modules query,key,value,dense \
      --peft_variant lora \
      --init_lora_weights true \
      --skip_eval \
      2>&1 | tee "${train_log}"

    model_path="$(awk -F'\t' '/^TRAIN_OUTPUT_DIR\t/ {print $2}' "${train_log}" | tail -n 1)"
    if [[ -z "${model_path}" ]]; then
      echo "Error: failed to parse TRAIN_OUTPUT_DIR from log: ${train_log}" >&2
      exit 1
    fi

    eval_args=(
      -m src.cli evaluate
      --model_path "${model_path}"
      --dataset_name "${dataset}"
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

  timestamp="$(date +%Y%m%d%H%M%S)"
  echo "==> Training ${dataset} with n=8, seed=${seed}, timestamp=${timestamp}"

  dataset_safe="${dataset//\//_}"
  train_log="${TRAIN_LOG_DIR}/train_${timestamp}_${dataset_safe}_s${seed}_n8.log"
  "${PYTHON_BINARY}" -m src.cli train \
    --timestamp "${timestamp}" \
    --output_dir "outputs_ours" \
    --dataset_name "${dataset}" \
    --model_name "${MODEL_NAME}" \
    --seed "${seed}" \
    --global_batch_size 64 \
    --per_device_batch_size 64 \
    --max_steps 500 \
    --learning_rate "${lr}" \
    --weight_decay 0.05 \
    --warmup_ratio 0.1 \
    --lora_r 16 \
    --lora_alpha 1 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --target_modules query,key,value,dense \
    --peft_variant lora \
    --init_lora_weights "orthogonal" \
    --use_cleaned_svd_ref_trainer \
    --repeat_n 6 \
    --skip_eval \
    2>&1 | tee "${train_log}"

  model_path="$(awk -F'\t' '/^TRAIN_OUTPUT_DIR\t/ {print $2}' "${train_log}" | tail -n 1)"
  if [[ -z "${model_path}" ]]; then
    echo "Error: failed to parse TRAIN_OUTPUT_DIR from log: ${train_log}" >&2
    exit 1
  fi

  eval_args=(
    -m src.cli evaluate
    --model_path "${model_path}"
    --dataset_name "${dataset}"
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