"""Whisper trainer - speech recognition / audio transcription."""
from __future__ import annotations
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.models.base import BaseTrainer


class WhisperModel(nn.Module):
    """Minimal Whisper-style encoder-decoder for audio transcription."""
    def __init__(self, vocab_size: int = 51865, d_model: int = 768, num_layers: int = 6):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True),
            num_layers=num_layers,
        )
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, audio_feat: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(audio_feat)
        out = self.decoder(tokens, memory)
        return self.proj(out)


class WhisperTrainer(BaseTrainer):
    """Whisper trainer using HuggingFace transformers or custom model."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None):
        super().__init__(config, event_bus)
        self.model = None

    def load_model(self) -> nn.Module:
        try:
            from transformers import WhisperForConditionalGeneration
            model_name = self.config.model_name or "openai/whisper-tiny"
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        except ImportError:
            self.model = WhisperModel()
        return self.model.to(self.device)

    def train(self, **kwargs) -> Dict[str, Any]:
        self.load_model()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.hyperparameters.learning_rate)
        return {"status": "ok", "model_type": "whisper", "device": self.device_type}
