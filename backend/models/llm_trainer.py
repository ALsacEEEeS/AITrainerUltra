"""LLM fine-tuning trainer using transformers & PEFT."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset

from backend.core.config import TrainingConfig
from backend.core.events import EventBus, EventType
from backend.core.registry import register
from backend.models.base import BaseTrainer


@register("llm", {"description": "LLM fine-tuning (GPT, LLaMA, Mistral, etc.)"})
class LLMTrainer(BaseTrainer):
    """Trainer for causal language models."""

    def __init__(self, config: TrainingConfig, event_bus: Optional[EventBus] = None) -> None:
        super().__init__(config, event_bus)
        self.model = None
        self.tokenizer = None

    async def load_model(self) -> Any:
        model_name = self.config.model_name
        if not model_name:
            raise ValueError(
                "LLM 模型名称不能为空. "
                "请提供 HuggingFace 模型ID (如 'gpt2', 'meta-llama/Meta-Llama-3-8B') "
                "或本地路径"
            )
        await self.log_message(f"Loading model: {model_name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(
                f"Tokenizer 加载失败: '{model_name}'. "
                f"请确认模型ID正确且可访问. "
                f"错误: {e}"
            ) from e

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
            )
        except ImportError as e:
            raise RuntimeError(
                f"加载 LLM 模型缺少依赖: {e}. "
                f"请安装: pip install transformers accelerate bitsandbytes"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"LLM 模型加载失败: '{model_name}'. "
                f"可能原因: 模型不存在、磁盘空间不足、内存不足. "
                f"错误: {e}"
            ) from e

        await self.log_message(f"Model loaded: {model_name}")
        return self.model

    async def train(self) -> Dict[str, Any]:
        await self.load_model()

        if self.config.lora:
            from peft import LoraConfig, get_peft_model
            lora_cfg = LoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.alpha,
                lora_dropout=self.config.lora.dropout,
                target_modules=self.config.lora.target_modules,
                bias=self.config.lora.bias,
                task_type=self.config.lora.task_type,
            )
            self.model = get_peft_model(self.model, lora_cfg)
            await self.log_message("LoRA adapters applied")

        tokenized = await self._prepare_dataset()
        training_args = self._build_args()

        hf_trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            data_collator=DataCollatorForLanguageModeling(
                self.tokenizer, mlm=False
            ),
        )

        await self.log_message("Training started")
        if self.event_bus:
            await self.event_bus.emit(EventType.TRAINING_START.value, {})

        hf_trainer.train()

        output_path = f"{self.config.output_dir}/final"
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

        return {
            "status": "completed",
            "model_path": output_path,
            "model_type": "llm",
        }

    def _build_args(self) -> TrainingArguments:
        hp = self.config.hyperparameters
        return TrainingArguments(
            output_dir=self.config.output_dir,
            learning_rate=hp.learning_rate,
            per_device_train_batch_size=hp.batch_size,
            num_train_epochs=hp.num_epochs,
            warmup_steps=hp.warmup_steps,
            weight_decay=hp.weight_decay,
            max_grad_norm=hp.max_grad_norm,
            logging_steps=hp.logging_steps,
            save_steps=hp.save_steps,
            evaluation_strategy="steps" if self.config.dataset.split == "eval" else "no",
            eval_steps=hp.eval_steps,
            gradient_accumulation_steps=hp.gradient_accumulation_steps,
            fp16=hp.fp16,
            bf16=hp.bf16,
            report_to=["tensorboard"],
            save_total_limit=2,
            remove_unused_columns=False,
        )

    async def _prepare_dataset(self) -> Dataset:
        dc = self.config.dataset
        from datasets import load_dataset

        if dc.path:
            dataset = load_dataset(dc.path, dc.name, split=dc.split)
        else:
            dataset = Dataset.from_list([
                {"text": "No dataset configured. Using placeholder."}
            ])

        if dc.max_samples and len(dataset) > dc.max_samples:
            dataset = dataset.select(range(dc.max_samples))

        def tokenize_fn(examples):
            result = self.tokenizer(
                examples[dc.text_column],
                truncation=True,
                padding="max_length",
                max_length=dc.max_seq_length,
            )
            result["labels"] = result["input_ids"].copy()
            return result

        return dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
