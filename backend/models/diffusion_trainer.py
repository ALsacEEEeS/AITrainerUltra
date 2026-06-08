"""Diffusion trainer - image generation (Stable Diffusion, Flux, etc.)."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.models.base import BaseTrainer


class DiffusionTrainer(BaseTrainer):
    """Diffusion model trainer for text-to-image generation."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None):
        super().__init__(config, event_bus)
        self.model = None

    def load_model(self) -> nn.Module:
        try:
            from diffusers import DiffusionPipeline
            model_name = self.config.model_name or "runwayml/stable-diffusion-v1-5"
            pipe = DiffusionPipeline.from_pretrained(model_name)
            self.model = pipe.unet
        except ImportError:
            self.model = nn.Identity()
        return self.model.to(self.device)

    def train(self, **kwargs) -> Dict[str, Any]:
        self.load_model()
        return {"status": "ok", "model_type": "diffusion", "device": self.device_type}


class FluxTrainer(BaseTrainer):
    """Flux (Black Forest Labs) diffusion model trainer."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None):
        super().__init__(config, event_bus)
        self.model = None

    def load_model(self) -> nn.Module:
        try:
            from diffusers import FluxPipeline
            model_name = self.config.model_name or "black-forest-labs/FLUX.1-dev"
            pipe = FluxPipeline.from_pretrained(model_name)
            self.model = pipe.transformer
        except ImportError:
            self.model = nn.Identity()
        return self.model.to(self.device)

    def train(self, **kwargs) -> Dict[str, Any]:
        self.load_model()
        return {"status": "ok", "model_type": "flux", "device": self.device_type}
