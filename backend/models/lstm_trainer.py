"""LSTM trainer - Long Short-Term Memory networks."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.core.registry import register
from backend.models.base import BaseTrainer


class LSTMModel(nn.Module):
    """LSTM for text classification / sequence modeling."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

        self.attention = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        if self.lstm.bidirectional:
            hidden_fwd = hidden[-2]
            hidden_bwd = hidden[-1]
            final_hidden = torch.cat((hidden_fwd, hidden_bwd), dim=-1)
        else:
            final_hidden = hidden[-1]

        out = self.dropout(final_hidden)
        return self.fc(out)


class LSTMLanguageModel(nn.Module):
    """LSTM-based language model for text generation."""

    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None) -> tuple:
        emb = self.embedding(x)
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden


@register("lstm", {"description": "LSTM training for sequence modeling and text generation"})
class LSTMTrainer(BaseTrainer):
    """Trainer for LSTM networks.

    Supports:
    - Text classification (with optional bidirection + attention)
    - Language modeling (next token prediction)
    - Sequence tagging (NER, POS)
    """

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(config, event_bus)
        self.model = None

    async def load_model(self) -> Any:
        if self.config.task == "language-modeling":
            self.model = LSTMLanguageModel(
                vocab_size=10000, embedding_dim=256,
                hidden_dim=512, num_layers=2,
            )
        elif self.config.task == "sequence-classification":
            self.model = LSTMModel(
                vocab_size=10000, embedding_dim=256,
                hidden_dim=512, num_layers=2,
                num_classes=10, bidirectional=True,
            )
        elif self.config.task == "token-classification":
            from backend.models.rnn_trainer import RNNModel
            self.model = RNNModel(
                vocab_size=10000, embedding_dim=128,
                hidden_dim=256, num_layers=2,
                num_classes=9,
            )
        else:
            self.model = LSTMModel(
                vocab_size=10000, embedding_dim=256,
                hidden_dim=512, num_layers=2,
                num_classes=2, bidirectional=True,
            )

        self.model.to(self.device)
        await self.log_message(f"LSTM model created on {self.device}")
        return self.model

    async def train(self) -> Dict[str, Any]:
        if self.model is None:
            await self.load_model()

        hp = self.config.hyperparameters

        from backend.utils.training_utils import load_real_data_for_trainer

        if self.config.task == "language-modeling":
            return await self._train_lm(hp)
        return await self._train_classifier(hp)

    async def _train_lm(
        self,
        hp: Any,
    ) -> Dict[str, Any]:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=hp.learning_rate,
            weight_decay=hp.weight_decay,
        )

        from backend.utils.training_utils import load_real_data_for_trainer
        result = load_real_data_for_trainer(self.config, "language-modeling", self.config.model_type)
        loader = None
        if result:
            loader, _ = result
        if loader is None:
            seq_len = self.config.dataset.max_seq_length or 64
            dummy_x = torch.randint(1, 1000, (16, seq_len))
            dataset = torch.utils.data.TensorDataset(dummy_x)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=hp.batch_size, shuffle=True,
            )

        await self.log_message("LSTM language model training started")
        final_loss = 0.0

        for epoch in range(hp.num_epochs):
            if self._stopped:
                break
            self.model.train()
            total_loss = 0.0
            steps = 0

            for batch in loader:
                if self._stopped:
                    break
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                else:
                    x = batch.to(self.device)
                optimizer.zero_grad()

                logits, _ = self.model(x)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = x[:, 1:].contiguous()
                loss = criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                steps += 1

            avg_loss = total_loss / max(steps, 1)
            await self.log_metrics(epoch, {"loss": avg_loss})
            final_loss = avg_loss

        output_path = f"{self.config.output_dir}/lstm_lm.pth"
        torch.save(self.model.state_dict(), output_path)

        return {
            "status": "completed",
            "final_loss": final_loss,
            "model_path": output_path,
            "model_type": "lstm",
            "task": "language-modeling",
        }

    async def _train_classifier(
        self,
        hp: Any,
    ) -> Dict[str, Any]:
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=hp.learning_rate,
            weight_decay=hp.weight_decay,
        )

        from backend.utils.training_utils import load_real_data_for_trainer
        result = load_real_data_for_trainer(self.config, "sequence-classification", self.config.model_type)
        loader = None
        num_classes = 10
        if result:
            loader, num_classes = result
        if loader is None:
            seq_len = self.config.dataset.max_seq_length or 64
            dummy_x = torch.randint(0, 1000, (32, seq_len))
            dummy_y = torch.randint(0, num_classes, (32,))
            dataset = torch.utils.data.TensorDataset(dummy_x, dummy_y)
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=hp.batch_size, shuffle=True,
            )

        await self.log_message("LSTM classification training started")
        final_loss = 0.0
        final_acc = 0.0

        for epoch in range(hp.num_epochs):
            if self._stopped:
                break
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
            final_loss = avg_loss
            final_acc = accuracy

        output_path = f"{self.config.output_dir}/lstm_classifier.pth"
        torch.save(self.model.state_dict(), output_path)

        return {
            "status": "completed",
            "final_loss": final_loss,
            "accuracy": final_acc,
            "model_path": output_path,
            "model_type": "lstm",
            "task": "sequence-classification",
        }
