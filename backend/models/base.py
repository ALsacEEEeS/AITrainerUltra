"""Abstract base class for all model trainers with TPU/NPU support."""

from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch

from backend.core.config import TrainingConfig
from backend.core.events import EventBus, EventType
from backend.utils.device import (
    DEVICE_INFO,
    get_torch_device,
    get_device_for_strategy,
    is_tpu_available,
    is_npu_available,
)


class BaseTrainer(ABC):
    """Base class for all model type trainers.

    Supports CUDA, MPS (Apple Silicon), ROCm (AMD), TPU (Google Cloud),
    NPU (Huawei Ascend), XPU (Intel), and CPU training.
    """

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None) -> None:
        self.config = config
        self.event_bus = event_bus
        self._stopped = False

        # Auto-detect the best device
        self.device, self.device_type = get_device_for_strategy(
            getattr(config, "device_strategy", "auto")
        )
        self.device_name = DEVICE_INFO.name

    @abstractmethod
    async def load_model(self) -> Any:
        ...

    @abstractmethod
    async def train(self) -> Dict[str, Any]:
        ...

    def is_tpu(self) -> bool:
        return self.device_type == "tpu"

    def is_npu(self) -> bool:
        return self.device_type == "npu"

    def is_cuda(self) -> bool:
        return self.device_type in ("cuda", "rocm")

    def is_mps(self) -> bool:
        return self.device_type == "mps"

    def get_device_info(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "device_type": self.device_type,
            "device_name": self.device_name,
        }

    async def save_checkpoint(self, path: str, metrics: Dict[str, Any]) -> None:
        if self.event_bus:
            await self.event_bus.emit(EventType.MODEL_SAVED.value, {
                "path": path,
                "metrics": metrics,
                "device": self.device_name,
            })

    async def log_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        if self.event_bus:
            await self.event_bus.emit(EventType.METRICS_UPDATE.value, {
                "step": step,
                "metrics": metrics,
                "device": self.device_type,
            })

    async def log_message(self, message: str) -> None:
        if self.event_bus:
            await self.event_bus.emit(EventType.LOG_MESSAGE.value, {
                "message": f"[{self.device_type}] {message}",
            })

    def stop(self) -> None:
        self._stopped = True

    def to_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move model to the detected device, handling TPU/NPU specifics."""
        if self.is_tpu():
            try:
                import torch_xla.core.xla_model as xm
                return xm.send_cpu_data_to_device(model, self.device)
            except Exception:
                pass
        elif self.is_npu():
            try:
                import torch_npu
                return model.npu()
            except Exception:
                pass
        return model.to(self.device)

    def to_device_batch(self, batch: Any) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return type(batch)(v.to(self.device) if isinstance(v, torch.Tensor) else v for v in batch)
        return batch

    def prepare_config(self) -> Dict[str, Any]:
        """Convert config to a flat dict for the training library."""
        hp = self.config.hyperparameters
        return {
            "output_dir": self.config.output_dir,
            "learning_rate": hp.learning_rate,
            "per_device_train_batch_size": hp.batch_size,
            "num_train_epochs": hp.num_epochs,
            "warmup_steps": hp.warmup_steps,
            "weight_decay": hp.weight_decay,
            "max_grad_norm": hp.max_grad_norm,
            "logging_steps": hp.logging_steps,
            "save_steps": hp.save_steps,
            "eval_steps": hp.eval_steps,
            "gradient_accumulation_steps": hp.gradient_accumulation_steps,
            "fp16": hp.fp16 and self.is_cuda(),
            "bf16": hp.bf16 or self.is_tpu(),
            "device": self.device_type,
        }
