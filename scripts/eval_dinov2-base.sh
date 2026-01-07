#!/usr/bin/env bash
# Evaluation helper for VTAB + ViT LoRA runs.
# Mirrors `scripts/train_vit_large_lora.sh` and calls `src/evaluate.py` via `src.cli`.
#
# Usage:
#   MODEL_PATH=runs/vit_base_lora DATASET=fw407/vtab-1k_cifar bash scripts/evaluate_vit_large_lora.sh
#
# Optional env vars:
#   SPLIT=test|validation, BATCH_SIZE=32, MIXED_PRECISION=bf16|fp16|no, SEED=42, CACHE_DIR=/path

set -euo pipefail

DATASET="${DATASET:-fw407/vtab-1k_cifar}"
MODEL_PATH="runs/vtab-1k_cifar/dinov2-base/r16/lora_r16_a1_lr0.0001_olora_sr#3rp_s42_20260106-071422"
SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEED="${SEED:-42}"
CACHE_DIR="${CACHE_DIR:-}"

export ACCELERATE_CONFIG_FILE="/home/yu/workspace/lora_image/accelerate_config/local_config.yaml"

PYTHON_BINARY="${PYTHON_BINARY:-/home/yu/peft_playground/.venv/bin/python}"

args=(
  -m src.cli evaluate
  --model_path "${MODEL_PATH}"
  --dataset_name "${DATASET}"
  --test_split "${SPLIT}"
  --image_column img
  --label_column label
  --batch_size "${BATCH_SIZE}"
  --seed "${SEED}"
  --mixed_precision "${MIXED_PRECISION}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  args+=(--cache_dir "${CACHE_DIR}")
fi

"${PYTHON_BINARY}" "${args[@]}"
