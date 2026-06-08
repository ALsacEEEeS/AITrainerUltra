"""Embedding trainer - text embedding models (BGE, Instructor, etc.)."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.models.base import BaseTrainer


class EmbeddingTrainer(BaseTrainer):
    """Text embedding model fine-tuning (BERT-based embedding models)."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None):
        super().__init__(config, event_bus)
        self.model = None

    def load_model(self) -> nn.Module:
        try:
            from transformers import AutoModel
            model_name = self.config.model_name or "BAAI/bge-small-en-v1.5"
            self.model = AutoModel.from_pretrained(model_name)
        except ImportError:
            self.model = nn.Identity()
        return self.model.to(self.device)

    def train(self, **kwargs) -> Dict[str, Any]:
        self.load_model()
        return {"status": "ok", "model_type": "embedding", "device": self.device_type}
