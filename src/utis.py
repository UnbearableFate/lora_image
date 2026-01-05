import hashlib
import os
import re
from typing import Any, Dict, Optional, Sequence, Union

import torch

def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip())
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned or "unknown"


def _flatten_hf_id(value: str) -> str:
    return _slugify("-".join(part for part in value.split("/") if part))


def _truncate_with_hash(value: str, max_len: int = 128) -> str:
    if len(value) <= max_len:
        return value
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    keep = max(1, max_len - (1 + len(digest)))
    return f"{value[:keep]}-{digest}"


def _format_float(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 0.001 or abs(value) >= 1000:
        return f"{value:.2e}"
    return f"{value:g}"


def build_wandb_project_run_tags(
    *,
    model_name: str,
    dataset_id: str,
    peft_variant: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_bias: str,
    target_modules: Sequence[str],
    modules_to_save: Optional[Sequence[str]],
    init_lora_weights: Union[bool, str, None],
    init_num_samples: int,
    init_batch_size: int,
    init_seed: int,
    corda_method: str,
    loraga_direction: str,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    num_train_epochs: float,
    global_batch_size: int,
    per_device_batch_size: int,
    eval_steps: int,
    logging_steps: int,
    seed: int,
    fp16: bool,
    bfloat16: bool,
    gradient_checkpointing: bool,
    use_cleaned_svd_ref_trainer: bool,
    repeat_n: int, 
    adjust_lora_alpha_at: Optional[Sequence[int]],
    timestamp: str,
) -> tuple[str, str, list[str], Dict[str, Any]]:
    model_component = _flatten_hf_id(model_name)
    dataset_component = _flatten_hf_id(dataset_id)
    project = _truncate_with_hash(f"{model_component}__{dataset_component}", max_len=128)

    if isinstance(init_lora_weights, bool) and init_lora_weights:
        init_lora_weights_str = "kaiming"
    elif isinstance(init_lora_weights, str):
        init_lora_weights_str = _slugify(init_lora_weights)
    key_parts: list[str] = [
        peft_variant,
        f"r{lora_r}",
        f"a{lora_alpha}",
        f'lr{learning_rate}',
        f"{init_lora_weights_str}",
    ]

    if use_cleaned_svd_ref_trainer:
        sr_info = f'sr#{repeat_n}rp'
        key_parts.append(sr_info)
    
    key_parts.append(timestamp)
    run_name = _truncate_with_hash("_".join(key_parts), max_len=128)

    tags: list[str] = [
        f"model={model_component}",
        f"dataset={dataset_component}",
        f"lr={_format_float(learning_rate)}",
        f"wd={_format_float(weight_decay)}",
        f"warmup={_format_float(warmup_ratio)}",
        f"epochs={_format_float(num_train_epochs)}",
        f"gbs={global_batch_size}",
        f"pbs={per_device_batch_size}",
        f"eval_steps={eval_steps}",
        f"logging_steps={logging_steps}",
        f"seed={seed}",
    ]
    if fp16:
        tags.append("fp16")
    if bfloat16:
        tags.append("bf16")
    if gradient_checkpointing:
        tags.append("grad_ckpt")
    if use_cleaned_svd_ref_trainer:
        tags.append("trainer=cleaned_svd_ref")
        if adjust_lora_alpha_at:
            tags.append(f"adjust_alpha_at={','.join(str(v) for v in adjust_lora_alpha_at)}")

    return project, run_name, tags

def _maybe_enable_wandb(
    use_wandb: bool,
    project: str,
    run_name: Optional[str],
    online: bool = True,
    *,
    tags: Optional[Sequence[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    if not use_wandb:
        os.environ["WANDB_MODE"] = os.environ.get("WANDB_MODE", "disabled")
        return

    try:
        import wandb  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "use_wandb=True but wandb is not importable; install wandb or set use_wandb=False."
        ) from exc

    if wandb.run is None:
        wandb.init(
            project=project,
            name=run_name,
            mode="online" if online else "offline",
            tags=list(tags) if tags else None,
            config=config or None,
        )


def _resolve_classification_head(module: Any) -> Optional[torch.nn.Module]:
    for attr in ("classifier", "head", "score"):
        candidate = getattr(module, attr, None)
        if candidate is not None and hasattr(candidate, "weight"):
            return candidate
    base_model = getattr(module, "base_model", None)
    if base_model is not None:
        return _resolve_classification_head(base_model)
    inner_model = getattr(module, "model", None)
    if inner_model is not None and inner_model is not module:
        return _resolve_classification_head(inner_model)
    return None


def init_classification_head(
    model: Any,
    *,
    weight_std: float = 2e-5,
) -> None:
    classification_head = _resolve_classification_head(model)
    if classification_head is None:
        raise AttributeError("Could not locate classification head (expected attribute like 'classifier').")
    torch.nn.init.normal_(classification_head.weight, mean=0.0, std=weight_std)
    if getattr(classification_head, "bias", None) is not None:
        torch.nn.init.zeros_(classification_head.bias)
    for param in classification_head.parameters():
        param.requires_grad = True
    return
