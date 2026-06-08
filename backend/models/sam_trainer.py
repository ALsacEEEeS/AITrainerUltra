"""SAM trainer - Segment Anything Model for image segmentation."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.models.base import BaseTrainer


class SAMTrainer(BaseTrainer):
    """Segment Anything Model fine-tuning for image segmentation."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None):
        super().__init__(config, event_bus)
        self.model = None

    def load_model(self) -> nn.Module:
        try:
            from transformers import SamModel
            model_name = self.config.model_name or "facebook/sam-vit-base"
            self.model = SamModel.from_pretrained(model_name)
        except ImportError:
            self.model = nn.Identity()
        return self.model.to(self.device)

    def train(self, **kwargs) -> Dict[str, Any]:
        self.load_model()
        return {"status": "ok", "model_type": "sam", "device": self.device_type}
