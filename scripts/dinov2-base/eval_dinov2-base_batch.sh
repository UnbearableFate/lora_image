#!/usr/bin/env bash

set -euo pipefail

MODELS_ROOT="outputs/cifar/dinov2-base/r16"
CSV_PATH_DIR="final_experiments"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/eval_dinov2-base_batch.sh /path/to/models_root

Env (optional):
  DATASET=fw407/vtab-1k_cifar SPLIT=test BATCH_SIZE=512 MIXED_PRECISION=bf16 SEED=42 CACHE_DIR=/path
  PYTHON_BINARY=/path/to/python ACCELERATE_CONFIG_FILE=/path/to/accelerate.yaml CSV_PATH_DIR=experiments
  REQUIRE_ADAPTER=1
  DRY_RUN=1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi


if [[ ! -d "${MODELS_ROOT}" ]]; then
  echo "Error: MODELS_ROOT is not a directory: ${MODELS_ROOT}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

SPLIT="${SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
CACHE_DIR="${CACHE_DIR:-}"
REQUIRE_ADAPTER="${REQUIRE_ADAPTER:-1}"

DRY_RUN="${DRY_RUN:-0}"

ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${REPO_ROOT}/accelerate_config/local_config.yaml}"
export ACCELERATE_CONFIG_FILE

PYTHON_BINARY="${PYTHON_BINARY:-/home/yu/peft_playground/.venv/bin/python}"

mapfile -d '' -t model_dirs < <(find "${MODELS_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if (( ${#model_dirs[@]} == 0 )); then
  echo "No subdirectories found under: ${MODELS_ROOT}" >&2
  exit 0
fi

run_eval() {
  local model_path="$1"
  local adapter_cfg="${model_path}/adapter_config.json"
  if [[ "${REQUIRE_ADAPTER}" == "1" && ! -f "${adapter_cfg}" ]]; then
    echo "Skipping (no adapter_config.json): ${model_path}"
    return 0
  fi

  local -a args=(
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
    args+=(--cache_dir "${CACHE_DIR}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '%q ' "${PYTHON_BINARY}" "${args[@]}"
    printf '\n'
  else
    "${PYTHON_BINARY}" "${args[@]}"
  fi
}

for model_path in "${model_dirs[@]}"; do
  echo "==> Evaluating model_path: ${model_path}"
  run_eval "${model_path}/checkpoint-500"
done
