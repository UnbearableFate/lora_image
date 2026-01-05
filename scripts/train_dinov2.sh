#!/usr/bin/env bash
# Recommended Accelerate launch for ViT-L/16 @384 with standard LoRA on VTAB-1K CIFAR.
# Adjust NUM_PROCS, dataset, and logging as needed.

set -euo pipefail

DATASET="${DATASET:-fw407/vtab-1k_cifar}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/dinov2_small_lora}"

export ACCELERATE_CONFIG_FILE="/home/yu/workspace/lora_image/accelerate_config/local_config.yaml"

PATHON_BINARY="/home/yu/peft_playground/.venv/bin/python"

"$PATHON_BINARY" -m src.cli train \
  --dataset_name "${DATASET}" \
  --model_name facebook/dinov2-base \
  --output_dir runs/qkv \
  --global_batch_size 64 \
  --per_device_batch_size 64 \
  --num_train_epochs 40 \
  --learning_rate 2.5e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --lora_r 16 \
  --lora_alpha 1 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --target_modules query,key,value,dense \
  --peft_variant lora \
  --init_lora_weights olora \
  --logging_steps 50 \
  --eval_steps 20 \
  --use_wandb True \

"$PATHON_BINARY" -m src.cli train \
  --dataset_name "${DATASET}" \
  --model_name facebook/dinov2-base \
  --output_dir runs/qkv \
  --global_batch_size 64 \
  --per_device_batch_size 64 \
  --num_train_epochs 40 \
  --learning_rate 2.5e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --lora_r 16 \
  --lora_alpha 1 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --target_modules query,key,value,dense \
  --peft_variant lora \
  --init_lora_weights pissa \
  --logging_steps 50 \
  --eval_steps 20 \
  --use_wandb True \

"$PATHON_BINARY" -m src.cli train \
  --dataset_name "${DATASET}" \
  --model_name facebook/dinov2-base \
  --output_dir runs/qkv \
  --global_batch_size 64 \
  --per_device_batch_size 64 \
  --num_train_epochs 40 \
  --learning_rate 2.5e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --lora_r 16 \
  --lora_alpha 1 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --target_modules query,key,value,dense \
  --peft_variant lora \
  --init_lora_weights orthogonal \
  --logging_steps 50 \
  --eval_steps 20 \
  --use_wandb True \

"$PATHON_BINARY" -m src.cli train \
  --dataset_name "${DATASET}" \
  --model_name facebook/dinov2-base \
  --output_dir runs/qkv \
  --global_batch_size 64 \
  --per_device_batch_size 64 \
  --num_train_epochs 40 \
  --learning_rate 2.5e-4 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --lora_r 16 \
  --lora_alpha 1 \
  --lora_dropout 0.05 \
  --lora_bias none \
  --target_modules query,key,value,dense \
  --peft_variant lora \
  --init_lora_weights gaussian \
  --logging_steps 50 \
  --eval_steps 20 \
  --use_wandb True \
