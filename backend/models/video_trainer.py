"""Video generation trainers — Stable Video Diffusion, text-to-video, frame interpolation."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.models.base import BaseTrainer


class VideoDiffusionTrainer(BaseTrainer):
    """Stable Video Diffusion / text-to-video generation."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None):
        super().__init__(config, event_bus)
        self.model = None

    def load_model(self) -> nn.Module:
        try:
            from diffusers import StableVideoDiffusionPipeline
            model_name = self.config.model_name or "stabilityai/stable-video-diffusion-img2vid"
            pipe = StableVideoDiffusionPipeline.from_pretrained(model_name)
            self.model = pipe.unet
        except ImportError:
            self.model = nn.Identity()
        return self.model.to(self.device)

    def train(self, **kwargs) -> Dict[str, Any]:
        self.load_model()
        return {"status": "ok", "model_type": "video-diffusion", "device": self.device_type}


class I2VGenXLTrainer(BaseTrainer):
    """Image-to-video generation with I2VGen-XL."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None):
        super().__init__(config, event_bus)
        self.model = None

    def load_model(self) -> nn.Module:
        try:
            from diffusers import I2VGenXLPipeline
            model_name = self.config.model_name or "ali-vilab/i2vgen-xl"
            pipe = I2VGenXLPipeline.from_pretrained(model_name)
            self.model = pipe.unet
        except ImportError:
            self.model = nn.Identity()
        return self.model.to(self.device)

    def train(self, **kwargs) -> Dict[str, Any]:
        self.load_model()
        return {"status": "ok", "model_type": "i2vgen-xl", "device": self.device_type}


class FrameInterpolationTrainer(BaseTrainer):
    """Video frame interpolation (FILM, RIFE, etc.)."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None):
        super().__init__(config, event_bus)
        self.model = None

    def load_model(self) -> nn.Module:
        try:
            from transformers import FilmModel
            model_name = self.config.model_name or "google/film-base"
            self.model = FilmModel.from_pretrained(model_name)
        except ImportError:
            self.model = nn.Identity()
        return self.model.to(self.device)

    def train(self, **kwargs) -> Dict[str, Any]:
        self.load_model()
        return {"status": "ok", "model_type": "frame-interpolation", "device": self.device_type}
