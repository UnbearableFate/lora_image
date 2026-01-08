import datetime
import math
import os
import re
from typing import Any, Dict, Optional, Sequence, Union

import evaluate
import numpy as np
from datasets import DatasetDict
import torch
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

from src.utis import _maybe_enable_wandb, build_wandb_project_run_tags, init_classification_head

def train(
    dataset_name: str = "fw407/vtab-1k_cifar",
    model_name: str = "google/vit-base-patch16-224-in21k",
    output_dir: str = "outputs",
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
    init_batch_size: int = 16,
    init_seed: Optional[int] = None,
    corda_method: str = "kpm",
    loraga_direction: str = "ArB2r",
    lora_cache_dir: str = "data_cache",
    learning_rate: float = 5e-4,
    weight_decay: float = 0.05,
    warmup_ratio: float = 0.05,
    num_train_epochs: float = 10.0,
    max_steps: Optional[int] = None,
    global_batch_size: int = 32,
    per_device_batch_size: int = 4,
    eval_batch_size: Optional[int] = None,
    logging_steps: int = 50,
    eval_steps: int = 500,
    seed: int = 42,
    use_wandb: bool = False,
    wandb_online: bool = False,
    fp16: bool = False,
    bf16: bool = False,
    gradient_checkpointing: bool = False,
    cache_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    push_to_hub: bool = False,
    use_cleaned_svd_ref_trainer: bool = False,
    adjust_lora_alpha_at: Union[str, Sequence[int]] = (2,),
    min_alpha_ratio: float = 0.8,
    max_alpha_ratio: float = 1.25,
    repeat_n: int = 3,
    repeat_warmup_ratio: float = 0.03,
    repeat_decay_ratio: float = 0.03,
    repeat_end_lr_rate: float = 0.97,
    final_warmup_ratio: float = 0.03,
    min_lr_rate: float = 0.001,
    repeat_decay_type: str = "cosine",
    final_decay_type: str = "linear",
    warmup_start_lr_rate: float = 0.1,
    first_warmup_start_lr_rate: float = 0.001,
    last_epoch: int = -1,
    timestamp: Optional[str] = None,
) -> Dict[str, float]:
    print("Starting training with the following parameters:")
    accelerator = Accelerator()
    set_seed(seed)
    timestamp = timestamp or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    dataset_id = resolve_dataset_id(dataset_name)

    def _parse_str_list(value) -> Optional[list[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",")]
            return [p for p in parts if p]
        return [str(v).strip() for v in value if str(v).strip()]

    def _parse_int_list(value) -> Optional[list[int]]:
        parsed = _parse_str_list(value)
        if parsed is None:
            return None
        values: list[int] = []
        for item in parsed:
            values.append(int(item))
        return values

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

    parsed_adjust_lora_alpha_at = _parse_int_list(adjust_lora_alpha_at)

    derived_project, derived_run_name, tags = build_wandb_project_run_tags(
            model_name=model_name,
            dataset_id=dataset_id,
            peft_variant=peft_variant,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_bias=lora_bias,
            target_modules=target_modules_list,
            modules_to_save=modules_to_save_list,
            init_lora_weights=parsed_init_lora_weights,
            init_num_samples=init_num_samples,
            init_batch_size=init_batch_size,
            init_seed=effective_init_seed,
            corda_method=corda_method,
            loraga_direction=loraga_direction,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_train_epochs,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            seed=seed,
            fp16=fp16,
            bfloat16=bf16,
            gradient_checkpointing=gradient_checkpointing,
            use_cleaned_svd_ref_trainer=use_cleaned_svd_ref_trainer,
            repeat_n=repeat_n,
            adjust_lora_alpha_at=parsed_adjust_lora_alpha_at,
            timestamp=timestamp,
        )

    if max_steps is not None:
        if isinstance(max_steps, str):
            max_steps = int(max_steps)
        if max_steps <= 0:
            raise ValueError("max_steps must be a positive integer when provided.")
        tags = [tag for tag in tags if not tag.startswith("epochs=")]
        tags.append(f"steps={max_steps}")
    
    if accelerator.is_main_process:   
        _maybe_enable_wandb(
            use_wandb,
            derived_project,
            derived_run_name,
            tags=tags,
            online=wandb_online,
            config=locals(),
        )

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
        dtype= torch.bfloat16 if bf16 else torch.float32,
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    print(model)

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

    # where we initialize the classification head weights with a normal distribution with σ = 2e-5 and bias with zeros
    init_classification_head(model, weight_std=2e-5)
    
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=preds, references=labels)

    assert global_batch_size % (per_device_batch_size * accelerator.num_processes) == 0, "global_batch_size must be divisible by per_device_batch_size * number of processes"
    gradient_accumulation_steps = global_batch_size // (per_device_batch_size * accelerator.num_processes) 
    effective_max_steps = max_steps
    if effective_max_steps is None:
        effective_max_steps = math.ceil(num_train_epochs * len(prepared["train"]) / global_batch_size)

    args = TrainingArguments(
        output_dir=os.path.join(output_dir,dataset_name.split("/")[-1],model_name.split("/")[-1],f"r{lora_r}", derived_run_name),
        remove_unused_columns=False,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=eval_batch_size or per_device_batch_size,
        max_steps=effective_max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        report_to=["wandb"] if use_wandb else [],
        seed=seed,
        data_seed=seed,
        bf16=bf16,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        push_to_hub=push_to_hub,
        save_total_limit=2,
        dataloader_persistent_workers=True,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
        dataloader_num_workers=8,
    )

    if use_cleaned_svd_ref_trainer:
        from src.cleaned_svd_ref_trainer import get_cleaned_svd_ref_trainer

        adjust_lora_alpha_at_list = parsed_adjust_lora_alpha_at or []
        trainer = get_cleaned_svd_ref_trainer(
            model=model,
            args=args,
            train_dataset=prepared["train"],
            eval_dataset=prepared["validation"],
            processing_class=processor,
            data_collator=default_data_collator,
            compute_metrics=compute_metrics,
            global_batch_size=global_batch_size,
            adjust_lora_alpha_at=adjust_lora_alpha_at_list,
            basic_alpha=lora_alpha,
            min_alpha_ratio=min_alpha_ratio,
            max_alpha_ratio=max_alpha_ratio,
            repeat_n=repeat_n,
            repeat_warmup_ratio=repeat_warmup_ratio,
            repeat_decay_ratio=repeat_decay_ratio,
            repeat_end_lr_rate=repeat_end_lr_rate,
            final_warmup_ratio=final_warmup_ratio,
            min_lr_rate=min_lr_rate,
            repeat_decay_type=repeat_decay_type,
            final_decay_type=final_decay_type,
            warmup_start_lr_rate=warmup_start_lr_rate,
            first_warmup_start_lr_rate=first_warmup_start_lr_rate,
            last_epoch=last_epoch,
        )
    else:
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
    if accelerator.is_main_process:
        print(f"TRAIN_OUTPUT_DIR\t{trainer.args.output_dir}", flush=True)
    return trainer.args.output_dir