"""LoRA adapter trainer - Parameter-Efficient Fine-Tuning."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.core.registry import register
from backend.models.base import BaseTrainer


@register("lora", {"description": "LoRA parameter-efficient fine-tuning"})
class LoRATrainer(BaseTrainer):
    """Trainer for LoRA adapters on any transformer model."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(config, event_bus)
        self.model = None
        self.tokenizer = None

    async def load_model(self) -> Any:
        """Load base model and apply LoRA configuration."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        model_name = self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = self.device_type if self.device_type != "auto" else "auto"
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=device_map,
        )

        lc = self.config.lora or LoraConfig()
        lora_config = LoraConfig(
            r=lc.r if isinstance(lc.r, int) else 16,
            lora_alpha=lc.alpha if isinstance(lc.alpha, int) else 32,
            lora_dropout=lc.dropout if isinstance(lc.dropout, float) else 0.05,
            task_type=TaskType.CAUSAL_LM,
            target_modules=lc.target_modules or ["q_proj", "v_proj"],
        )
        self.model = get_peft_model(base_model, lora_config)
        self.model.print_trainable_parameters()
        await self.log_message(f"LoRA adapters applied to {model_name}")
        return self.model

    async def train(self) -> Dict[str, Any]:
        from transformers import (
            TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        )
        from peft import LoraConfig
        from datasets import Dataset

        await self.load_model()
        hp = self.config.hyperparameters

        sample_texts = [
            {"text": f"Training LoRA adapter - step {i}"}
            for i in range(100)
        ]
        dataset = Dataset.from_list(sample_texts)

        def tokenize(examples):
            tokens = self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.config.dataset.max_seq_length or 2048,
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens

        tokenized = dataset.map(tokenize, batched=True)

        args = TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=hp.learning_rate,
            per_device_train_batch_size=hp.batch_size,
            num_train_epochs=hp.num_epochs,
            warmup_steps=hp.warmup_steps,
            logging_steps=5,
            save_strategy="epoch",
            fp16=hp.fp16,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        await self.log_message("LoRA training started")
        trainer.train()

        adapter_path = f"{self.config.output_dir}/lora_adapter"
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        return {
            "status": "completed",
            "model_path": adapter_path,
            "model_type": "lora",
            "adapter_name": self.config.lora.task_type if self.config.lora else "default",
        }
