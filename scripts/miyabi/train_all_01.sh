#!/usr/bin/env bash
#!/bin/bash
#PBS -q short-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -m abe

set -euo pipefail
cd /work/xg24i002/x10041/lora_image

MODEL_NAME="facebook/dinov2-large"

ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-accelerate_config/local_config.yaml}"
export ACCELERATE_CONFIG_FILE
PYTHON_BINARY="/work/xg24i002/x10041/lora-ns/.venv/bin/python"

# VTAB_ALIASES keys (see `src/data.py`)
DATASETS=(
  "cifar"
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
        --logging_steps 20 \
        --eval_steps 20
    done
  done
done
