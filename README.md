# VTAB-1K ViT + LoRA Playground

This project provides a minimal-yet-practical pipeline to evaluate ViT backbones with LoRA adapters on VTAB-1K tasks. It relies on Hugging Face datasets/transformers, Fire for the CLI, and Weights & Biases for logging.

## Quickstart
- Create an environment and install deps:
  - `python3 -m venv .venv && source .venv/bin/activate`
  - `pip install -r requirements.txt`
- Configure Accelerate (for distributed Trainer) one-time: `accelerate config`
- Train with Hugging Face Trainer (ViT-B/16 + LoRA):
  - `accelerate launch -m src.cli train --output_dir runs/cifar_trainer --batch_size 16 --eval_steps 200`
- Evaluate on test split with Accelerate DDP:
  - `accelerate launch -m src.evaluate evaluate_model --model_path runs/cifar_trainer --dataset_name fw407/vtab-1k_cifar --test_split test`
- MPI launch (requires mpirun installed) still delegates to the same Trainer entrypoint:
  - `NUM_PROCS=4 bash scripts/mpirun_train.sh --output_dir runs/cifar_mpi --batch_size 16 --eval_steps 200`

## CLI
The CLI is backed by Fire; every parameter in `src/training.py::train` is exposed. Common flags:
- `dataset_name`: HF dataset id or short alias (see below). Default `fw407/vtab-1k_cifar`.
- `model_name`: ViT checkpoint, e.g., `google/vit-base-patch16-224-in21k`.
- `peft_variant`: LoRA family variant, e.g. `lora`, `dora`, `rslora`, `qalora`.
- `init_lora_weights`: LoRA weight init for PEFT, e.g. `true` (default), `gaussian`, `olora`, `pissa`, `eva`, `corda`, `lora_ga`.
- `target_modules`: Comma-separated module name fragments to receive LoRA (default `query,value`).
- `use_wandb`: Set false to disable W&B. Without an API key we fall back to offline mode automatically.

Example: run with a different init and dataset
```
python -m src.cli train \
  --dataset_name fw407/vtab-1k_flowers102 \
  --init_lora_weights gaussian \
  --output_dir runs/flowers_gauss \
  --num_train_epochs 15
```

## VTAB-1K aliases
You can pass either a full HF dataset id or these aliases:
- `caltech101`, `cifar`, `dtd`, `flowers102`, `pets`, `svhn`, `sun397`, `patch_camelyon`, `eurosat`, `resisc45`, `retinopathy`, `clevr_count`, `clevr_distance`, `dsprites_location`, `dsprites_orientation`, `smallnorb_azimuth`, `smallnorb_elevation`, `kitti_distance`

## Notes
- Training uses `transformers.Trainer` (distributed via Accelerate) with `evaluate` for validation accuracy.
- LoRA applied via `peft`; base weights are frozen by default.
- W&B logging is enabled by default; runs without a key go to offline mode to avoid failures.
- Test evaluation is provided separately in `src/evaluate.py` (Accelerate DDP).
- The code assumes VTAB datasets expose `image` and `label` columns plus `train`/`validation`/`test` splits; override via CLI if needed.
