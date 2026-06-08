"""CNN trainer for image classification and vision tasks."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.core.registry import register
from backend.models.base import BaseTrainer


@register("cnn", {"description": "CNN training for image classification"})
class CNNTrainer(BaseTrainer):
    """Trainer for Convolutional Neural Networks."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(config, event_bus)
        self.model = None

    def _build_model(self) -> nn.Module:
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 10)
            return model
        except ImportError:
            return self._simple_cnn()

    def _simple_cnn(self) -> nn.Module:
        """Simple CNN when torchvision not available."""
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    async def load_model(self) -> Any:
        self.model = self._build_model()
        self.model.to(self.device)
        await self.log_message("CNN model loaded")
        return self.model

    async def train(self) -> Dict[str, Any]:
        if self.model is None:
            await self.load_model()

        hp = self.config.hyperparameters
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=hp.learning_rate)

        dummy_inputs = torch.randn(32, 3, 32, 32)
        dummy_labels = torch.randint(0, 10, (32,))
        dataset = TensorDataset(dummy_inputs, dummy_labels)
        loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)

        await self.log_message("CNN training started")
        for epoch in range(hp.num_epochs):
            if self._stopped:
                break
            loss = await self._train_epoch(epoch, loader, criterion, optimizer)

        output_path = f"{self.config.output_dir}/cnn_model.pth"
        torch.save(self.model.state_dict(), output_path)
        return {
            "status": "completed",
            "loss": loss,
            "model_path": output_path,
            "model_type": "cnn",
        }

    async def _train_epoch(
        self,
        epoch: int,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
    ) -> float:
        self.model.train()
        total_loss = 0.0

        for inputs, labels in loader:
            if self._stopped:
                break
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        await self.log_metrics(epoch, {"loss": avg_loss})
        return avg_loss
