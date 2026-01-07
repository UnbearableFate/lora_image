#!/usr/bin/env bash
# Recommended Accelerate launch for ViT-L/16 @384 with standard LoRA on VTAB-1K CIFAR.
# Adjust NUM_PROCS, dataset, and logging as needed.

set -euo pipefail

MODEL_NAME="facebook/dinov2-base"

ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-/home/yu/workspace/lora_image/accelerate_config/local_config.yaml}"
export ACCELERATE_CONFIG_FILE
PYTHON_BINARY="${PYTHON_BINARY:-/home/yu/peft_playground/.venv/bin/python}"

# VTAB_ALIASES keys (see `src/data.py`)
#"cifar"
DATASETS=(
  "caltech101"
  "dtd"
  "flowers102"
  "pets"
  "svhn"
  "sun397"
  "patch_camelyon"
  "eurosat"
  "resisc45"
  "retinopathy"
  "clevr_count"
  "clevr_distance"
  "dsprites_location"
  "dsprites_orientation"
  "smallnorb_azimuth"
  "smallnorb_elevation"
  "kitti_distance"
)

# 5 two-digit primes as a deterministic "random seed table" for 5 repeated runs.
SEEDS=(11 23 37 47 53)
init_lora_weights_list=("gaussian" "true" "olora" "pissa" "orthogonal" "eva" "corda")

for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for init_lora_weights in "${init_lora_weights_list[@]}"; do
      timestamp="$(date +%Y%m%d%H%M%S)"
      echo "==> Training ${dataset} with init_lora_weights=${init_lora_weights}, seed=${seed}, timestamp=${timestamp}"

      "${PYTHON_BINARY}" -m src.cli train \
        --timestamp "${timestamp}" \
        --dataset_name "${dataset}" \
        --model_name "${MODEL_NAME}" \
        --seed "${seed}" \
        --global_batch_size 64 \
        --per_device_batch_size 64 \
        --max_steps 500 \
        --learning_rate 1e-4 \
        --weight_decay 0.05 \
        --warmup_ratio 0.1 \
        --lora_r 16 \
        --lora_alpha 1 \
        --lora_dropout 0.05 \
        --lora_bias none \
        --target_modules query,key,value,dense \
        --peft_variant lora \
        --init_lora_weights "${init_lora_weights}" \
        --logging_steps 10 \
        --eval_steps 20
    done

    timestamp="$(date +%Y%m%d%H%M%S)"
    echo "==> Training ${dataset} sr-init with init_lora_weights=${init_lora_weights}, seed=${seed}, timestamp=${timestamp}"

    "${PYTHON_BINARY}" -m src.cli train \
      --timestamp "${timestamp}" \
      --dataset_name "${dataset}" \
      --model_name "${MODEL_NAME}" \
      --seed "${seed}" \
      --global_batch_size 64 \
      --per_device_batch_size 64 \
      --max_steps 500 \
      --learning_rate 1e-4 \
      --weight_decay 0.05 \
      --warmup_ratio 0.1 \
      --lora_r 16 \
      --lora_alpha 1 \
      --lora_dropout 0.05 \
      --lora_bias none \
      --target_modules query,key,value,dense \
      --peft_variant lora \
      --init_lora_weights orthogonal \
      --logging_steps 10 \
      --eval_steps 20 \
      --use_cleaned_svd_ref_trainer
  done
done
