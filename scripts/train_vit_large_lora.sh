#!/usr/bin/env bash
# Recommended Accelerate launch for ViT-L/16 @384 with standard LoRA on VTAB-1K CIFAR.
# Adjust NUM_PROCS, dataset, and logging as needed.

set -euo pipefail

DATASET="${DATASET:-fw407/vtab-1k_cifar}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/vit_base_lora}"

export ACCELERATE_CONFIG_FILE="/home/yu/workspace/lora_image/accelerate_config/local_config.yaml"

PATHON_BINARY="/home/yu/peft_playground/.venv/bin/python"

"$PATHON_BINARY" -m src.cli train \
  --dataset_name "${DATASET}" \
  --model_name google/vit-large-patch16-224-in21k \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size 64 \
  --eval_batch_size 64 \
  --num_train_epochs 30 \
  --learning_rate 5e-4 \
  --weight_decay 0.05 \
  --warmup_ratio 0.1 \
  --gradient_accumulation_steps 1 \
  --lora_r 16 \
  --lora_alpha 1 \
  --lora_dropout 0.0 \
  --lora_bias none \
  --target_modules query,key,value,dense \
  --peft_variant lora \
  --init_lora_weights true \
  --logging_steps 50 \
  --eval_steps 100 \
  --use_wandb True \
