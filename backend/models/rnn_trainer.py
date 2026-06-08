"""RNN trainer - recurrent neural networks for sequence modeling."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.core.registry import register
from backend.models.base import BaseTrainer


class RNNModel(nn.Module):
    """Multi-layer RNN for sequence classification."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        last_hidden = hidden[-1]
        out = self.dropout(last_hidden)
        return self.fc(out)


@register("rnn", {"description": "RNN training for sequence modeling"})
class RNNTrainer(BaseTrainer):
    """Trainer for Recurrent Neural Networks.

    Uses: text classification, sequence prediction, time series
    """

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(config, event_bus)
        self.model = None

    async def load_model(self) -> Any:
        self.model = RNNModel(
            vocab_size=10000,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            num_classes=10,
            dropout=0.3,
        )
        self.model.to(self.device)
        await self.log_message(f"RNN model created on {self.device}")
        return self.model

    async def train(self) -> Dict[str, Any]:
        if self.model is None:
            await self.load_model()

        hp = self.config.hyperparameters
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=hp.learning_rate,
            weight_decay=hp.weight_decay,
        )

        from backend.utils.training_utils import load_real_data_for_trainer
        result = load_real_data_for_trainer(self.config, self.config.task, self.config.model_type)
        if result:
            loader, num_classes = result
        else:
            seq_len = self.config.dataset.max_seq_length or 64
            dummy_x = torch.randint(0, 1000, (32, seq_len))
            dummy_y = torch.randint(0, 10, (32,))
            dataset = torch.utils.data.TensorDataset(dummy_x, dummy_y)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=hp.batch_size, shuffle=True,
            )

        await self.log_message("RNN training started")
        final_loss = 0.0
        final_acc = 0.0

        for epoch in range(hp.num_epochs):
            if self._stopped:
                break
            epoch_loss, epoch_acc = await self._train_epoch(
                epoch, loader, criterion, optimizer,
            )
            final_loss = epoch_loss
            final_acc = epoch_acc

        output_path = f"{self.config.output_dir}/rnn_model.pth"
        torch.save(self.model.state_dict(), output_path)

        return {
            "status": "completed",
            "final_loss": final_loss,
            "accuracy": final_acc,
            "model_path": output_path,
            "model_type": "rnn",
            "architecture": f"RNN-2x256",
        }

    async def _train_epoch(
        self,
        epoch: int,
        loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in loader:
            if self._stopped:
                break
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            outputs = self.model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(loader)
        accuracy = correct / total if total > 0 else 0
        await self.log_metrics(epoch, {"loss": avg_loss, "accuracy": accuracy})
        return avg_loss, accuracy
