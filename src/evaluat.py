import os
import contextlib
import csv
import datetime
import json
import logging
import copy
from typing import Dict, Optional

import evaluate
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, set_seed

from src.data import (
    get_label_info,
    is_medmnist_dataset_name,
    load_medmnist_dataset,
    load_vtab_dataset,
    preprocess_splits,
    resolve_dataset_id,
)

class DataCollator:
    def __call__(self, features):
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}


@contextlib.contextmanager
def _temporary_log_level(logger_name: str, level: int):
    logger = logging.getLogger(logger_name)
    old_level = logger.level
    old_disabled = logger.disabled
    try:
        logger.disabled = False
        logger.setLevel(level)
        yield
    finally:
        logger.setLevel(old_level)
        logger.disabled = old_disabled


def _infer_classifier_num_labels_from_adapter(model_path: str) -> Optional[int]:
    """
    Best-effort inference of num_labels from a PEFT adapter directory that saved
    `modules_to_save=["classifier"]`.
    """

    safetensors_path = os.path.join(model_path, "adapter_model.safetensors")
    bin_path = os.path.join(model_path, "adapter_model.bin")

    if os.path.isfile(safetensors_path):
        try:
            from safetensors.torch import safe_open
        except ImportError:
            return None
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for key in (
                "base_model.model.classifier.weight",
                "base_model.model.classifier.modules_to_save.default.weight",
            ):
                if key in f.keys():
                    weight = f.get_tensor(key)
                    if weight.ndim == 2:
                        return int(weight.shape[0])
        return None

    if os.path.isfile(bin_path):
        try:
            state = torch.load(bin_path, map_location="cpu")
        except Exception:
            return None
        for key in (
            "base_model.model.classifier.weight",
            "base_model.model.classifier.modules_to_save.default.weight",
        ):
            weight = state.get(key)
            if isinstance(weight, torch.Tensor) and weight.ndim == 2:
                return int(weight.shape[0])
        return None

    return None


def _maybe_load_classifier_from_adapter(model, model_path: str) -> bool:
    """
    Best-effort: explicitly load saved classifier weights from a PEFT adapter dir
    (when `modules_to_save=["classifier"]`) into the model.

    PEFT usually does this automatically, but this makes evaluation robust and
    avoids accidentally evaluating with a freshly initialized head.
    """

    safetensors_path = os.path.join(model_path, "adapter_model.safetensors")
    bin_path = os.path.join(model_path, "adapter_model.bin")

    keys = (
        "base_model.model.classifier.weight",
        "base_model.model.classifier.bias",
        "base_model.model.classifier.modules_to_save.default.weight",
        "base_model.model.classifier.modules_to_save.default.bias",
    )

    state = {}
    if os.path.isfile(safetensors_path):
        try:
            from safetensors.torch import safe_open
        except ImportError:
            return False
        with safe_open(safetensors_path, framework="pt", device="cpu") as f:
            for k in keys:
                if k in f.keys():
                    state[k] = f.get_tensor(k)
    elif os.path.isfile(bin_path):
        try:
            loaded = torch.load(bin_path, map_location="cpu")
        except Exception:
            return False
        for k in keys:
            v = loaded.get(k)
            if isinstance(v, torch.Tensor):
                state[k] = v

    if not state:
        return False

    model.load_state_dict(state, strict=False)
    return True


def _default_csv_path(model_path: str) -> str:
    if model_path and os.path.isdir(model_path):
        return os.path.join(model_path, "eval_results.csv")
    return "eval_results.csv"


def _rewrite_csv_with_extended_header(csv_path: str, fieldnames) -> None:
    if not os.path.exists(csv_path):
        return

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow(row)


def _append_row_to_csv(csv_path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            header_reader = csv.reader(f)
            existing_header = next(header_reader, [])
        fieldnames = list(dict.fromkeys(existing_header + [k for k in row.keys() if k not in existing_header]))
        if fieldnames != existing_header:
            _rewrite_csv_with_extended_header(csv_path, fieldnames)
    else:
        fieldnames = list(row.keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writerow(row)


def _make_peft_config_safe_for_inference(peft_config, accelerator: Optional[Accelerator] = None):
    """
    PEFT will re-run certain LoRA initialization schemes (e.g. `corda`, `eva`, `pissa`, `loftq`)
    when loading an adapter. This is unnecessary for evaluation and can fail unless extra
    preprocessing was done (e.g. `preprocess_corda`).

    For evaluation we override the config passed to `PeftModel.from_pretrained` to disable those
    init hooks; adapter weights are loaded right after module creation anyway.
    """

    cfg = copy.deepcopy(peft_config)
    init = getattr(cfg, "init_lora_weights", None)
    if isinstance(init, str):
        init_lowered = init.strip().lower()
        heavy_or_preprocess_inits = {"corda" , "lora_ga"}
        if init_lowered in heavy_or_preprocess_inits:
            if accelerator is not None and accelerator.is_main_process:
                accelerator.print(
                    f"Overriding PEFT `init_lora_weights={init!r}` -> True for evaluation-time adapter loading."
                )
            cfg.init_lora_weights = True
            # PEFT's LoRA layer code may unconditionally `kwargs.update(lora_config.loftq_config)`,
            # so ensure it is always a mapping (even if we're not using LoftQ during eval load).
            if hasattr(cfg, "loftq_config") and getattr(cfg, "loftq_config") is None:
                cfg.loftq_config = {}
            for extra in ("corda_config", "eva_config"):
                if hasattr(cfg, extra):
                    setattr(cfg, extra, None)
    return cfg

def get_info_from_model_path(model_path: str):
    info = model_path.split("/")
    if "checkpoint" in info[-1]:
        info = info[-2].split("_")
    else:
        info = info[-1].split("_")
    info_dict = {
        "timestamp": info[-1],
        "seed": int(info[-2][1:]),
    }
    if info[-3].startswith("sr#"):
        info_dict["extra"] = info[-3]
    else:
        info_dict["extra"] = 'none'
    return info_dict

def evaluate_model(
    model_path: str,
    test_split: str = "test",
    image_column: str = "img",
    label_column: str = "label",
    batch_size: int = 32,
    dataset_name: Optional[str] = None,
    mixed_precision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    csv_path_dir: Optional[str] = "experiments",
) -> Dict[str, float]:
    info_dict = get_info_from_model_path(model_path)
    set_seed(info_dict["seed"])

    accelerator = Accelerator(mixed_precision=mixed_precision or "no")
    accelerator.print(f"Running evaluation with {accelerator.num_processes} processes.")
    is_medmnist = dataset_name is not None and is_medmnist_dataset_name(dataset_name)
    dataset_id = dataset_name if is_medmnist else resolve_dataset_id(dataset_name)
    if is_medmnist:
        raw_dataset = load_medmnist_dataset(dataset_name, cache_dir=cache_dir)
        if image_column == "img":
            image_column = "image"
    else:
        raw_dataset = load_vtab_dataset(dataset_id, cache_dir=cache_dir)
    if test_split not in raw_dataset:
        raise ValueError(f"Split {test_split} not found in dataset {dataset_id}.")

    label_names, id2label, label2id, label_value_to_id = get_label_info(
        raw_dataset[test_split],
        label_column=label_column,
    )

    peft_adapter_config_path = os.path.join(model_path, "adapter_config.json")
    is_peft_adapter_dir = os.path.isdir(model_path) and os.path.isfile(peft_adapter_config_path)
    peft_config = None
    if is_peft_adapter_dir:
        try:
            from peft import PeftConfig, PeftModel
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Evaluating a PEFT adapter requires `peft` to be installed."
            ) from exc
        peft_config = PeftConfig.from_pretrained(model_path)
        peft_config_for_load = _make_peft_config_safe_for_inference(peft_config, accelerator)
        base_model_name_or_path = peft_config.base_model_name_or_path

        adapter_num_labels = _infer_classifier_num_labels_from_adapter(model_path)
        if adapter_num_labels is not None and adapter_num_labels != len(label_names):
            raise ValueError(
                f"Adapter classifier head has {adapter_num_labels} labels, but dataset split "
                f"'{test_split}' produced {len(label_names)} labels. Ensure you're evaluating "
                "with the same dataset/label mapping as training."
            )

        try:
            processor = AutoImageProcessor.from_pretrained(model_path)
        except Exception:
            processor = AutoImageProcessor.from_pretrained(base_model_name_or_path)

        # When num_labels differs from the pretraining head, Transformers logs a warning about
        # newly initialized classifier weights. This is expected because we overwrite them with
        # the adapter-saved `modules_to_save` weights right after loading.
        with _temporary_log_level("transformers.modeling_utils", logging.ERROR):
            base_model = AutoModelForImageClassification.from_pretrained(
                base_model_name_or_path,
                num_labels=adapter_num_labels or len(label_names),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
                cache_dir=cache_dir,
            )
        model = PeftModel.from_pretrained(base_model, model_path, config=peft_config_for_load)
        classifier_loaded = _maybe_load_classifier_from_adapter(model, model_path)
        if classifier_loaded and accelerator.is_main_process:
            accelerator.print("Loaded saved classifier weights from adapter checkpoint (modules_to_save).")
    else:
        processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModelForImageClassification.from_pretrained(
            model_path,
            cache_dir=cache_dir,
        )

    prepared = preprocess_splits(
        raw_dataset.select_columns([image_column, label_column]) if hasattr(raw_dataset, "select_columns") else raw_dataset,
        processor,
        image_column=image_column,
        label_column=label_column,
        label_value_to_id=label_value_to_id,
    )

    test_ds = prepared[test_split]
    collator = DataCollator()
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collator,
    )

    model, test_loader = accelerator.prepare(model, test_loader)

    metric = evaluate.load("accuracy")
    model.eval()
    for batch in test_loader:
        with torch.no_grad():
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds = accelerator.gather_for_metrics(preds)
        all_labels = accelerator.gather_for_metrics(batch["labels"])
        metric.add_batch(
            predictions=all_preds.cpu().numpy(),
            references=all_labels.cpu().numpy(),
        )

    results = metric.compute()
    if accelerator.is_main_process:
        accelerator.print(f"Test metrics: {results}")
        row: Dict[str, object] = {
            "timestamp": info_dict["timestamp"],
            "base_model": peft_config.base_model_name_or_path.split("/")[-1],
            "dataset_name": dataset_name,
            "init_lora_weights": getattr(peft_config, "init_lora_weights", None),
            "extra": info_dict["extra"],
            "seed": info_dict["seed"],
        }
        
        resolved_csv_path = os.path.join(csv_path_dir,row["dataset_name"],row["base_model"], "eval_results.csv")

        if str(row["init_lora_weights"]).lower() == "true":
            row["init_lora_weights"] = "kaiming"

        for k, v in results.items():
            row[f"metric_{k}"] = v

        row["model_path"] =  model_path.split("/")[-1]

        if peft_config is not None:
            try:
                peft_dict = peft_config.to_dict()
            except Exception:
                peft_dict = {}

            key_fields = (
                "r",
                "lora_alpha",
                "lora_dropout",
                "target_modules",
                "bias",
                "use_dora",
                "use_rslora",
            )
            for k in key_fields:
                if k in peft_dict:
                    v = peft_dict[k]
                    row[f"{k}"] = (
                        json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v
                    )

        _append_row_to_csv(resolved_csv_path, row)
        accelerator.print(f"Wrote evaluation row to CSV: {resolved_csv_path}")

    return results


def main():
    import fire

    fire.Fire(evaluate_model)


if __name__ == "__main__":
    main()
