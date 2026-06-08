"""QLoRA trainer - 4-bit quantized LoRA fine-tuning."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from backend.core.config import TrainingConfig
from backend.core.events import EventBus
from backend.core.registry import register
from backend.models.base import BaseTrainer


@register("qlora", {"description": "QLoRA 4-bit quantized fine-tuning"})
class QLoRATrainer(BaseTrainer):
    """Trainer for QLoRA - memory efficient 4-bit fine-tuning."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(config, event_bus)
        self.model = None
        self.tokenizer = None
        self._quant_config = None

    def _build_quant_config(self):
        """Build 4-bit quantization config lazily."""
        from transformers import BitsAndBytesConfig
        qc = self.config.qlora
        compute_dtype = getattr(torch, qc.compute_dtype if qc else "float32")

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=qc.double_quant if qc else True,
            bnb_4bit_quant_type=qc.quant_type if qc else "nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )

    async def load_model(self) -> Any:
        """Load model with 4-bit quantization and apply LoRA."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType

        model_name = self.config.model_name
        await self.log_message(f"Loading quantized model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = self.device_type if self.device_type != "auto" else "auto"
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self._build_quant_config(),
            device_map=device_map,
            torch_dtype=torch.float32,
        )

        lc = self.config.lora
        lora_config = LoraConfig(
            r=lc.r if lc else 16,
            lora_alpha=lc.alpha if lc else 32,
            lora_dropout=lc.dropout if lc else 0.05,
            task_type=TaskType.CAUSAL_LM,
            target_modules=lc.target_modules if lc else ["q_proj", "v_proj"],
            bias="none",
        )

        self.model = get_peft_model(base_model, lora_config)
        await self.log_message(f"QLoRA model loaded ({model_name})")
        return self.model

    async def train(self) -> Dict[str, Any]:
        from transformers import (
            TrainingArguments, Trainer, DataCollatorForLanguageModeling,
        )
        from datasets import Dataset

        await self.load_model()
        hp = self.config.hyperparameters

        dataset = Dataset.from_list([
            {"text": f"QLoRA training sample {i}"}
            for i in range(50)
        ])

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
            optim="paged_adamw_8bit",
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
        )

        await self.log_message("QLoRA training started")
        trainer.train()

        adapter_path = f"{self.config.output_dir}/qlora_adapter"
        self.model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)

        return {
            "status": "completed",
            "model_path": adapter_path,
            "model_type": "qlora",
            "quantization": self.config.qlora.quant_type if self.config.qlora else "nf4",
        }
