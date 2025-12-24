import os
from typing import Dict, Optional, Sequence, Union

import evaluate
import numpy as np
from datasets import DatasetDict
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from accelerate import Accelerator

from src.data import get_label_info, load_vtab_dataset, preprocess_splits, resolve_dataset_id
from src.lora_loader import LoraHyperparameters, VisionDataCollator, attach_lora_adapter, get_lora_config

def _maybe_enable_wandb(use_wandb: bool, project: str, run_name: Optional[str]) -> None:
    if not use_wandb:
        os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "disabled")
        return
    if os.environ.get("WANDB_API_KEY") is None and os.environ.get("WANDB_MODE") is None:
        os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("WANDB_PROJECT", project)
    if run_name:
        os.environ.setdefault("WANDB_NAME", run_name)


def train(
    dataset_name: str = "fw407/vtab-1k_cifar",
    model_name: str = "google/vit-base-patch16-224-in21k",
    output_dir: str = "outputs/vtab_trainer",
    train_split: str = "train",
    eval_split: str = "validation",
    image_column: str = "img",
    label_column: str = "label",
    peft_variant: str = "lora",
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    lora_bias: str = "none",
    target_modules: Sequence[str] = ("query", "value"),
    modules_to_save: Optional[Sequence[str]] = ("classifier",),
    init_lora_weights: Union[bool, str, None] = True,
    init_num_samples: int = 512,
    init_batch_size: int = 8,
    init_seed: Optional[int] = None,
    corda_method: str = "kpm",
    loraga_direction: str = "ArB2r",
    lora_cache_dir: str = "data_cache",
    learning_rate: float = 5e-4,
    weight_decay: float = 0.05,
    warmup_ratio: float = 0.05,
    num_train_epochs: float = 10.0,
    batch_size: int = 32,
    eval_batch_size: Optional[int] = None,
    gradient_accumulation_steps: int = 1,
    logging_steps: int = 50,
    eval_steps: int = 500,
    seed: int = 42,
    use_wandb: bool = True,
    wandb_project: str = "vtab-lora",
    wandb_run_name: Optional[str] = None,
    fp16: bool = False,
    bfloat16: bool = False,
    gradient_checkpointing: bool = False,
    cache_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    push_to_hub: bool = False,
) -> Dict[str, float]:
    print("Starting training with the following parameters:")
    accelerator = Accelerator()
    set_seed(seed)
    _maybe_enable_wandb(use_wandb, wandb_project, wandb_run_name)

    dataset_id = resolve_dataset_id(dataset_name)
    raw_dataset = load_vtab_dataset(dataset_id, cache_dir=cache_dir)
    if train_split not in raw_dataset or eval_split not in raw_dataset:
        raise ValueError(f"Splits {train_split}/{eval_split} not found in dataset {dataset_id}.")

    label_names, id2label, label2id, label_value_to_id = get_label_info(
        raw_dataset[train_split],
        label_column=label_column,
    )
    processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)

    prepared = preprocess_splits(
        DatasetDict({"train": raw_dataset[train_split], "validation": raw_dataset[eval_split]}),
        processor,
        image_column=image_column,
        label_column=label_column,
        label_value_to_id=label_value_to_id,
    )

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        cache_dir=cache_dir,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    print(model)

    def _parse_str_list(value) -> Optional[list[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",")]
            return [p for p in parts if p]
        return [str(v).strip() for v in value if str(v).strip()]

    target_modules_list = _parse_str_list(target_modules) or []
    if not target_modules_list:
        raise ValueError("target_modules must contain at least one module name fragment.")
    modules_to_save_list = _parse_str_list(modules_to_save)

    effective_init_seed = init_seed if init_seed is not None else seed * 2 + 1

    parsed_init_lora_weights = init_lora_weights
    if isinstance(parsed_init_lora_weights, str):
        lowered = parsed_init_lora_weights.strip().lower()
        if lowered in {"true", "1", "yes"}:
            parsed_init_lora_weights = True
        elif lowered in {"false", "0", "no"}:
            parsed_init_lora_weights = False
        elif lowered in {"none", "null"}:
            parsed_init_lora_weights = None
        else:
            parsed_init_lora_weights = lowered

    lora_hparams = LoraHyperparameters(
        variant=peft_variant,
        r=lora_r,
        alpha=lora_alpha,
        dropout=lora_dropout,
        bias=lora_bias,
        target_modules=target_modules_list,
        modules_to_save=modules_to_save_list,
        init_lora_weights=parsed_init_lora_weights,
        init_num_samples=init_num_samples,
        init_batch_size=init_batch_size,
        corda_method=corda_method,
        loraga_direction=loraga_direction,
        cache_dir=lora_cache_dir,
        model_name_or_path=model_name,
        dataset_name=dataset_id,
        init_seed=effective_init_seed,
    )
    lora_cfg = get_lora_config(lora_hparams)

    model.to(accelerator.device)

    model = attach_lora_adapter(
        base_model=model,
        lora_cfg=lora_cfg,
        train_dataset=prepared["train"],
        init_num_samples=init_num_samples,
        batch_size=init_batch_size,
        seed=effective_init_seed,
        accelerator=accelerator,
        data_collator=VisionDataCollator(),
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size or batch_size,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to=["wandb"] if use_wandb else [],
        seed=seed,
        dataloader_num_workers=4,
        fp16=fp16,
        bf16=bfloat16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=push_to_hub,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=prepared["train"],
        eval_dataset=prepared["validation"],
        processing_class=processor,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = trainer.evaluate()
    trainer.save_state()
    trainer.save_model()
    return metrics
