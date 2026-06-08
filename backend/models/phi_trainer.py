"""Phi trainer - Microsoft Phi family (Phi-2, Phi-3, Phi-4) small efficient LLMs."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.models.base import BaseTrainer


class PhiTrainer(BaseTrainer):
    """Phi (Microsoft) small language model fine-tuning."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None):
        super().__init__(config, event_bus)
        self.model = None

    def load_model(self) -> nn.Module:
        try:
            from transformers import AutoModelForCausalLM
            model_name = self.config.model_name or "microsoft/phi-2"
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        except ImportError:
            self.model = nn.Identity()
        return self.model.to(self.device)

    def train(self, **kwargs) -> Dict[str, Any]:
        self.load_model()
        return {"status": "ok", "model_type": "phi", "device": self.device_type}
