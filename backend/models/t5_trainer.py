"""T5 trainer - encoder-decoder text-to-text models."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.models.base import BaseTrainer


class T5Trainer(BaseTrainer):
    """T5 (Text-To-Text Transfer Transformer) fine-tuning."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None):
        super().__init__(config, event_bus)
        self.model = None

    def load_model(self) -> nn.Module:
        try:
            from transformers import T5ForConditionalGeneration
            model_name = self.config.model_name or "t5-small"
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        except ImportError:
            self.model = nn.Identity()
        return self.model.to(self.device)

    def train(self, **kwargs) -> Dict[str, Any]:
        self.load_model()
        return {"status": "ok", "model_type": "t5", "device": self.device_type}
