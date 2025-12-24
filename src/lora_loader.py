from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
import tqdm

from peft import LoraConfig, get_peft_model

try:  # optional, only needed for init_lora_weights="eva"
    from peft import initialize_lora_eva_weights  # type: ignore
except Exception:  # pragma: no cover
    initialize_lora_eva_weights = None

try:  # optional, only needed for init_lora_weights="corda" / "eva"
    from peft.tuners.lora.corda import preprocess_corda  # type: ignore
    from peft.tuners.lora.config import CordaConfig, EvaConfig  # type: ignore
except Exception:  # pragma: no cover
    preprocess_corda = None
    CordaConfig = None
    EvaConfig = None

from accelerate import Accelerator

import logging
logger = logging.getLogger(__name__)

from my_peft import LoraGAConfig

class VisionDataCollator:
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}

@dataclass
class LoraHyperparameters:
    """Hyperparameters for the LoRA-family adapters."""

    variant: str = "lora"  # lora, dora, qalora, rslora
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(
        default_factory=lambda: ["query", "value"]
    )
    modules_to_save: Optional[List[str]] = field(default_factory=lambda: ["classifier"])
    task_type: Optional[str] = None
    init_lora_weights: Union[bool, str, None] = True # ["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "corda", "loftq", "orthogonal"]
    init_num_samples: int = 512
    init_batch_size: int = 8
    corda_method: str = "kpm"  # kpm or ipm

    loraga_direction : str = "ArB2r"  # ArB2r, A2rB, BrA2r
    loraga_dtype : torch.dtype = torch.float32

    cache_dir: Optional[str] = "data_cache"
    unique_cache_filename : Optional[str] = None
    model_name_or_path: Optional[str] = None
    dataset_name: Optional[str] = None
    subdataset_name: Optional[str] = None
    init_seed: int = 1337

    def __post_init__(self):
        if not self.model_name_or_path or not self.dataset_name:
            return
        unique_cache_filename = f"{self.model_name_or_path.replace('/', '-')}_{self.dataset_name}"
        if self.subdataset_name:
            unique_cache_filename += f"_{self.subdataset_name}"
        self.unique_cache_filename = f"{unique_cache_filename}_r{self.r}_dp{self.init_num_samples}_bs{self.init_batch_size}_{self.init_seed}.pt"
    
    def get_unique_cache_path(self,path_mid_name) -> str:
        parent_path = Path(self.cache_dir, path_mid_name)
        if not parent_path.exists():
            parent_path.mkdir(parents=True, exist_ok=True)
        return parent_path.joinpath(self.unique_cache_filename).as_posix()

_VARIANT_TO_FLAGS = {
    "lora": {"use_dora": False, "use_rslora": False, "use_qalora": False},
    "dora": {"use_dora": True, "use_rslora": False, "use_qalora": False},
    "rslora": {"use_dora": False, "use_rslora": True, "use_qalora": False},
    "qalora": {"use_dora": False, "use_rslora": False, "use_qalora": True},
}

def get_lora_config(lora_cfg: LoraHyperparameters) -> LoraConfig | LoraGAConfig:
    variant = lora_cfg.variant.lower()
    if variant not in _VARIANT_TO_FLAGS:
        raise ValueError(f"Unsupported LoRA variant: {variant}")

    common_kwargs: Dict[str, Any] = {
        "r": lora_cfg.r,
        "lora_alpha": lora_cfg.alpha,
        "lora_dropout": lora_cfg.dropout,
        "bias": lora_cfg.bias,
        "target_modules": list(lora_cfg.target_modules),
        **_VARIANT_TO_FLAGS[variant],
    }
    if lora_cfg.task_type:
        common_kwargs["task_type"] = lora_cfg.task_type
    if lora_cfg.modules_to_save:
        common_kwargs["modules_to_save"] = list(lora_cfg.modules_to_save)

    if lora_cfg.init_lora_weights != "lora_ga":
        corda_config = None
        eva_config = None
        if lora_cfg.init_lora_weights == "corda":
                if CordaConfig is None:
                    raise ImportError("init_lora_weights='corda' requires a PEFT build that ships CordaConfig.")
                corda_config = CordaConfig(
                    corda_method=lora_cfg.corda_method, # kpm or ipm
                    cache_file=lora_cfg.get_unique_cache_path("corda_cache"),
                    covariance_file=lora_cfg.get_unique_cache_path("covariance_file"),
                )
        elif lora_cfg.init_lora_weights == "eva":
            if EvaConfig is None:
                raise ImportError("init_lora_weights='eva' requires a PEFT build that ships EvaConfig.")
            eva_config = EvaConfig()

        peft_config = LoraConfig(
            **common_kwargs,
            init_lora_weights=lora_cfg.init_lora_weights,
            corda_config=corda_config,
            eva_config=eva_config,
        )
        
    else:
        peft_config = LoraGAConfig(
            **common_kwargs,
            bsz=lora_cfg.init_batch_size,
            direction=lora_cfg.loraga_direction,
            dtype= lora_cfg.loraga_dtype,
            gradient_save_path=lora_cfg.get_unique_cache_path("loraga_gradient"),
        )
    
    print(f"lora config: {peft_config}")
    return peft_config

def attach_lora_adapter(
    base_model,
    lora_cfg: LoraConfig | LoraGAConfig,
    train_dataset,
    *,
    init_num_samples: int,
    batch_size: int,
    seed: int,
    accelerator: Optional[Accelerator] = None,
    data_collator: Optional[Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]]] = None,
):
    init_mode = getattr(lora_cfg, "init_lora_weights", None)
    if init_mode is None and lora_cfg.__class__.__name__ == "LoraGAConfig":
        init_mode = "lora_ga"
    if init_mode not in ["corda", "eva", "lora_ga"]:
        return get_peft_model(base_model, lora_cfg)
    sub_dataset = train_dataset.shuffle(seed=seed).select(range(init_num_samples))
    data_collator = data_collator or VisionDataCollator()

    if init_mode == "corda":
        return get_peft_model_with_corda(base_model, lora_cfg, sub_dataset, data_collator)
    if init_mode == "eva":
        return get_peft_model_with_eva(base_model, lora_cfg, sub_dataset, data_collator, batch_size)
    return get_peft_model_with_lora_ga(base_model, lora_cfg, sub_dataset, data_collator, batch_size, accelerator)

def freeze_lora_A_weights(peft_model):
    for name, param in peft_model.named_parameters():
        if "lora_A" in name:
            param.requires_grad = False

def get_peft_model_with_corda(
    base_model,
    lora_cfg: LoraConfig,
    sub_dataset,
    data_collator: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
):
    if preprocess_corda is None:
        raise ImportError("init_lora_weights='corda' requires a PEFT build that includes preprocess_corda.")
    calib_loader = DataLoader(
        sub_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=data_collator,
    )

    device = base_model.device
    print(f"Running Corda preprocessing on device: {device}")
    #calib_loader = accelerator.prepare(calib_loader)

    @torch.no_grad()
    def _run_model():
        was_training = base_model.training
        base_model.eval()
        # for batch in calib_loader:
        #     batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        #     base_model(**batch)
        for batch in tqdm.tqdm(calib_loader, desc="corda preprocessing"):
            batch = {k: v.to(device) for k, v in batch.items()}
            base_model(**batch)
        if was_training:
            base_model.train()

    print(f"Starting Corda preprocessing... with sub-dataset of size {len(sub_dataset)}")
    preprocess_corda(
        base_model,
        lora_cfg,
        run_model=_run_model,
    )
    return get_peft_model(base_model, lora_cfg)

def get_peft_model_with_eva(
        base_model,
        lora_cfg: LoraConfig,
        sub_dataset,
        data_collator: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
        batch_size: int,
    ):
    if initialize_lora_eva_weights is None:
        raise ImportError(
            "init_lora_weights='eva' requires a PEFT build that exports initialize_lora_eva_weights."
        )
    
    device = next(base_model.parameters()).device

    def _collate_to_device(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = data_collator(features)
        return {k: v.to(device) for k, v in batch.items()}

    dataloader = DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_to_device,
    )

    peft_model = get_peft_model(base_model, lora_cfg, low_cpu_mem_usage=True)
    print(f"Initializing Eva LoRA weights... with sub-dataset of size {len(sub_dataset)}")
    initialize_lora_eva_weights(peft_model, dataloader)
    return peft_model

__all__ = [
    "LoraHyperparameters",
    "VisionDataCollator",
    "attach_lora_adapter",
    "get_lora_config",
]

def get_peft_model_with_lora_ga(
        model,
        lora_ga_cfg: LoraGAConfig,
        sub_dataset,
        data_collator: Callable[[List[Dict[str, Any]]], Dict[str, torch.Tensor]],
        batch_size: int,
        accelerator: Optional[Accelerator] = None,
    ):

    from my_peft.utils.lora_ga_utils import (
                LoraGAContext,
                estimate_gradient,
            )
    from my_peft import get_peft_model as my_get_peft_model

    gradient_loader = DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )
    if accelerator is None:
        accelerator = Accelerator()
    named_grad = estimate_gradient(
        model=model,
        dataloader=gradient_loader,
        accelerator=accelerator,
        quant_flag=False,
        origin_type=None,
        quant_type=None,
        no_split_module_classes=None,
        grad_save_path=lora_ga_cfg.gradient_save_path,
    )
    start_time = time()
    with LoraGAContext(model=model, named_grad=named_grad):
        model = my_get_peft_model(model=model, peft_config=lora_ga_cfg)
    logger.info(f"LoRA-GA initialization took {time() - start_time:.2f} seconds")
    
    return model
