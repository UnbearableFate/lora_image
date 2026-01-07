#!/usr/bin/env bash
# Recommended Accelerate launch for ViT-L/16 @384 with standard LoRA on VTAB-1K CIFAR.
# Adjust NUM_PROCS, dataset, and logging as needed.

set -euo pipefail

DATASET="cifar"
MODEL_NAME="facebook/dinov2-base"
USE_WANDB="FALSE"

export ACCELERATE_CONFIG_FILE="/home/yu/workspace/lora_image/accelerate_config/local_config.yaml"
PATHON_BINARY="/home/yu/peft_playground/.venv/bin/python"

timestamp=$(date +%Y%m%d_%H%M%S)

"$PATHON_BINARY" -m src.cli train \
  --timestamp "${timestamp}" \
  --dataset_name "${DATASET}" \
  --model_name "${MODEL_NAME}" \
  --seed 11 \
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
  --init_lora_weights true \
  --logging_steps 10 \
  --eval_steps 20 \
  --use_wandb "${USE_WANDB}" \