import os
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterator, Optional, Tuple

import torch
from torch import nn
from transformers import Trainer
from typing import Union
from datasets import Dataset
from transformers.trainer_utils import PredictionOutput

class EmaTrainer(Trainer):
    """
    A `transformers.Trainer` subclass that maintains an exponential moving average (EMA)
    of model parameters (and optionally buffers) during training.
    """

    EMA_STATE_FILENAME = "ema_state.pt"

    def __init__(
        self,
        *args: Any,
        ema_decay: float = 0.995,
        ema_update_after_step: int = 150,
        ema_update_every: int = 1,
        ema_device: Optional[str] = None,
        ema_fp32: bool = True,
        ema_track_buffers: bool = False,
        use_ema_for_eval: bool = True,
        use_ema_for_save: bool = True,
        save_ema_state: bool = True,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if not (0.0 < float(ema_decay) <= 1.0):
            raise ValueError(f"ema_decay must be in (0,1], got {ema_decay}")
        if ema_update_every < 1:
            raise ValueError(f"ema_update_every must be >= 1, got {ema_update_every}")
        if ema_update_after_step < 0:
            raise ValueError(f"ema_update_after_step must be >= 0, got {ema_update_after_step}")

        self.ema_decay = float(ema_decay)
        self.ema_update_after_step = int(ema_update_after_step)
        self.ema_update_every = int(ema_update_every)
        self.ema_device = ema_device
        self.ema_fp32 = bool(ema_fp32)
        self.ema_track_buffers = bool(ema_track_buffers)
        self.use_ema_for_eval = bool(use_ema_for_eval)
        self.use_ema_for_save = bool(use_ema_for_save)
        self.save_ema_state = bool(save_ema_state)

        self._ema_params: Optional[Dict[str, torch.Tensor]] = None
        self._ema_buffers: Optional[Dict[str, torch.Tensor]] = None
        self._ema_num_updates: int = 0
        self._optimizer_step_count: int = 0
        self._ema_loaded_state: Optional[Dict[str, Any]] = None
        self._ema_device: Optional[torch.device] = None

    def _get_unwrapped_model(self) -> nn.Module:
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            try:
                return accelerator.unwrap_model(self.model)
            except Exception:
                pass
        return self.model

    def _ema_target_device(self, model: nn.Module) -> torch.device:
        if self.ema_device is not None:
            return torch.device(self.ema_device)
        for p in model.parameters():
            return p.device
        return torch.device("cpu")

    def _ema_target_dtype(self, tensor: torch.Tensor) -> torch.dtype:
        if self.ema_fp32 and tensor.dtype in (torch.float16, torch.bfloat16, torch.float32):
            return torch.float32
        return tensor.dtype

    @staticmethod
    def _iter_float_named_params(model: nn.Module) -> Iterator[Tuple[str, nn.Parameter]]:
        for name, p in model.named_parameters():
            if torch.is_floating_point(p.data) and p.requires_grad:
                yield name, p

    @staticmethod
    def _iter_float_named_buffers(model: nn.Module) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, b in model.named_buffers():
            if torch.is_floating_point(b):
                yield name, b

    def _maybe_init_ema(self) -> None:
        if self._ema_params is not None:
            return

        model = self._get_unwrapped_model()
        device = self._ema_target_device(model)
        self._ema_device = device

        ema_params: Dict[str, torch.Tensor] = {}
        for name, p in self._iter_float_named_params(model):
            ema_params[name] = p.detach().to(device=device, dtype=self._ema_target_dtype(p)).clone()

        ema_buffers: Dict[str, torch.Tensor] = {}
        if self.ema_track_buffers:
            for name, b in self._iter_float_named_buffers(model):
                ema_buffers[name] = b.detach().to(device=device, dtype=self._ema_target_dtype(b)).clone()

        self._ema_params = ema_params
        self._ema_buffers = ema_buffers

        if self._ema_loaded_state is not None:
            self.load_ema_state_dict(self._ema_loaded_state)
            self._ema_loaded_state = None

    @torch.no_grad()
    def _update_ema(self) -> None:
        self._maybe_init_ema()
        assert self._ema_params is not None
        assert self._ema_buffers is not None
        assert self._ema_device is not None

        model = self._get_unwrapped_model()
        decay = self.ema_decay
        one_minus_decay = 1.0 - decay

        for name, p in self._iter_float_named_params(model):
            if name not in self._ema_params:
                self._ema_params[name] = (
                    p.detach()
                    .to(device=self._ema_device, dtype=self._ema_target_dtype(p))
                    .clone()
                )
            ema_p = self._ema_params[name]
            src = p.detach().to(device=ema_p.device, dtype=ema_p.dtype)
            ema_p.mul_(decay).add_(src, alpha=one_minus_decay)

        if self.ema_track_buffers:
            for name, b in self._iter_float_named_buffers(model):
                if name not in self._ema_buffers:
                    self._ema_buffers[name] = (
                        b.detach()
                        .to(device=self._ema_device, dtype=self._ema_target_dtype(b))
                        .clone()
                    )
                ema_b = self._ema_buffers[name]
                ema_b.copy_(b.detach().to(device=ema_b.device, dtype=ema_b.dtype))
        self._ema_num_updates += 1

    @contextmanager
    def ema_weights(self) -> Iterator[None]:
        if self._ema_params is None:
            yield
            return

        model = self._get_unwrapped_model()
        assert self._ema_params is not None
        assert self._ema_buffers is not None

        backup_params: Dict[str, torch.Tensor] = {}
        backup_buffers: Dict[str, torch.Tensor] = {}

        try:
            for name, p in self._iter_float_named_params(model):
                ema_p = self._ema_params.get(name)
                if ema_p is None:
                    continue
                backup_params[name] = p.detach().clone()
                p.data.copy_(ema_p.to(device=p.device, dtype=p.dtype))

            if self.ema_track_buffers:
                for name, b in self._iter_float_named_buffers(model):
                    ema_b = self._ema_buffers.get(name)
                    if ema_b is None:
                        continue
                    backup_buffers[name] = b.detach().clone()
                    b.data.copy_(ema_b.to(device=b.device, dtype=b.dtype))

            yield
        finally:
            for name, p in self._iter_float_named_params(model):
                backup = backup_params.get(name)
                if backup is not None:
                    p.data.copy_(backup.to(device=p.device, dtype=p.dtype))
            if self.ema_track_buffers:
                for name, b in self._iter_float_named_buffers(model):
                    backup = backup_buffers.get(name)
                    if backup is not None:
                        b.data.copy_(backup.to(device=b.device, dtype=b.dtype))

    def state_dict_ema(self) -> Dict[str, Any]:
        if self._ema_params is None:
            return {"num_updates": self._ema_num_updates, "params": {}, "buffers": {}}
        assert self._ema_params is not None
        assert self._ema_buffers is not None
        return {
            "num_updates": self._ema_num_updates,
            "decay": self.ema_decay,
            "params": {k: v.detach().cpu() for k, v in self._ema_params.items()},
            "buffers": {k: v.detach().cpu() for k, v in self._ema_buffers.items()},
        }

    def load_ema_state_dict(self, ema_state: Dict[str, Any]) -> None:
        self._maybe_init_ema()
        assert self._ema_params is not None
        assert self._ema_buffers is not None

        params: Dict[str, torch.Tensor] = ema_state.get("params", {}) or {}
        buffers: Dict[str, torch.Tensor] = ema_state.get("buffers", {}) or {}

        device = self._ema_device or (next(iter(self._ema_params.values())).device if self._ema_params else torch.device("cpu"))
        for name, tensor in params.items():
            if not torch.is_tensor(tensor):
                continue
            self._ema_params[name] = tensor.detach().to(device=device, dtype=self._ema_target_dtype(tensor)).clone()

        if self.ema_track_buffers:
            for name, tensor in buffers.items():
                if not torch.is_tensor(tensor):
                    continue
                self._ema_buffers[name] = tensor.detach().to(device=device, dtype=self._ema_target_dtype(tensor)).clone()

        self._ema_num_updates = int(ema_state.get("num_updates", self._ema_num_updates))

    def train(self, *args: Any, **kwargs: Any):
        resume_from_checkpoint = kwargs.get("resume_from_checkpoint", None)
        if resume_from_checkpoint is None and len(args) >= 1:
            resume_from_checkpoint = args[0]

        if isinstance(resume_from_checkpoint, str):
            ema_path = os.path.join(resume_from_checkpoint, self.EMA_STATE_FILENAME)
            if os.path.isfile(ema_path):
                try:
                    self._ema_loaded_state = torch.load(ema_path, map_location="cpu")
                except Exception:
                    self._ema_loaded_state = None

        return super().train(*args, **kwargs)

    def training_step(self, model: nn.Module, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        self._optimizer_step_count += 1
        if (
            self._optimizer_step_count > self.ema_update_after_step
            and (self._optimizer_step_count % self.ema_update_every) == 0
        ):
            self._update_ema()

        return loss

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        ctx = self.ema_weights() if self.use_ema_for_eval else nullcontext()
        with ctx:
            return super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[list[str]] = None, metric_key_prefix: str = "test"
    ) -> PredictionOutput:
        ctx = self.ema_weights() if self.use_ema_for_eval else nullcontext()
        with ctx:
            return super().predict(test_dataset=test_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False) -> None:
        output_dir = output_dir or self.args.output_dir

        if self.save_ema_state and self.is_world_process_zero() and self._ema_params is not None:
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.state_dict_ema(), os.path.join(output_dir, self.EMA_STATE_FILENAME))

        ctx = self.ema_weights() if self.use_ema_for_save else nullcontext()
        with ctx:
            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)