#!/bin/bash
#PBS -q short-g
#PBS -W group_list=xg24i002
#PBS -l select=1:mpiprocs=1
#PBS -l walltime=02:30:00
#PBS -j oe
#PBS -m abe

set -euo pipefail

MODEL_NAME="google/vit-base-patch16-224"
dataset="medmnist-v2:DermaMNIST"
CSV_PATH_DIR="experiments_medmnist"

ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-lora_image/accelerate_config/local_config.yaml}"
export ACCELERATE_CONFIG_FILE

TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-train_logs}"
mkdir -p "${TRAIN_LOG_DIR}"

# 5 two-digit primes as a deterministic "random seed table" for 5 repeated runs.
SEEDS=(23 37 47 53)
init_lora_weights_list=("eva" "corda" "lora_ga" "gaussian" "true" "olora" "pissa" "orthogonal" )
fit_init_lora_weights_list=("gaussian" "true" "olora" "orthogonal" )

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



for seed in "${SEEDS[@]}"; do
  for init_lora_weights in "${init_lora_weights_list[@]}"; do
    timestamp="$(date +%Y%m%d%H%M%S)"
    echo "==> Training ${dataset} with init_lora_weights=${init_lora_weights}, seed=${seed}, timestamp=${timestamp}"

    dataset_safe="${dataset//\//_}"
    train_log="${TRAIN_LOG_DIR}/train_${dataset_safe}_s${seed}_${timestamp}.log"
    python -m src.cli train \
      --timestamp "${timestamp}" \
      --output_dir "outputs" \
      --dataset_name "${dataset}" \
      --model_name "${MODEL_NAME}" \
      --seed "${seed}" \
      --global_batch_size 64 \
      --per_device_batch_size 64 \
      --num_train_epochs 5 \
      --learning_rate 4e-4 \
      --weight_decay 0.01 \
      --warmup_ratio 0.03 \
      --lora_r 16 \
      --lora_alpha 1 \
      --lora_dropout 0.01 \
      --lora_bias none \
      --target_modules query,key,value,dense \
      --peft_variant lora \
      --init_lora_weights "${init_lora_weights}" \
      --eval_split "val" \
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
      --test_split "test"
      --image_column img
      --label_column label
      --batch_size 512
      --mixed_precision "bf16"
      --csv_path_dir "${CSV_PATH_DIR}"
    )
    python "${eval_args[@]}"
  done

  for init_lora_weights in "${fit_init_lora_weights_list[@]}"; do
    timestamp="$(date +%Y%m%d%H%M%S)"
    echo "==> Training ${dataset} seed=${seed}, timestamp=${timestamp}, init_lora_weights=${init_lora_weights} using cleaned SVD ref trainer"

    dataset_safe="${dataset//\//_}"
    train_log="${TRAIN_LOG_DIR}/train_${timestamp}_${dataset_safe}_s${seed}_n8_${init_lora_weights}.log"
    python -m src.cli train \
      --timestamp "${timestamp}" \
      --output_dir "outputs_ours" \
      --dataset_name "${dataset}" \
      --model_name "${MODEL_NAME}" \
      --seed "${seed}" \
      --global_batch_size 64 \
      --per_device_batch_size 64 \
      --num_train_epochs 5 \
      --learning_rate 4e-4 \
      --weight_decay 0.01 \
      --warmup_ratio 0.03 \
      --lora_r 16 \
      --lora_alpha 1 \
      --lora_dropout 0.01 \
      --lora_bias none \
      --target_modules query,key,value,dense \
      --peft_variant lora \
      --init_lora_weights "${init_lora_weights}" \
      --eval_split "val" \
      --use_cleaned_svd_ref_trainer \
      --repeat_n 3 \
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
      --test_split "test"
      --image_column img
      --label_column label
      --batch_size 512
      --mixed_precision "bf16"
      --csv_path_dir "${CSV_PATH_DIR}"
    )
    python "${eval_args[@]}"

  done
done