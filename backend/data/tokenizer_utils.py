"""Tokenization utilities for LLM training."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def load_tokenizer(
    model_name: str,
    use_fast: bool = True,
    add_pad_token: bool = True,
) -> Any:
    """Load a tokenizer with sensible defaults."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)

    if add_pad_token and tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer


def tokenize_dataset(
    dataset: Any,
    tokenizer: Any,
    text_column: str = "text",
    max_length: int = 512,
    truncation: bool = True,
    padding: str = "max_length",
) -> Any:
    """Tokenize a HuggingFace dataset."""
    def tokenize_fn(examples: Dict[str, List]) -> Dict[str, Any]:
        result = tokenizer(
            examples[text_column],
            truncation=truncation,
            padding=padding,
            max_length=max_length,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    return dataset.map(tokenize_fn, batched=True)


def prepare_conversation_dataset(
    conversations: List[Dict[str, str]],
    tokenizer: Any,
    max_length: int = 2048,
) -> Any:
    """Prepare a conversation dataset for fine-tuning."""
    from datasets import Dataset

    formatted = []
    for conv in conversations:
        text = _format_training_conversation(conv)
        formatted.append({"text": text})

    dataset = Dataset.from_list(formatted)
    return tokenize_dataset(dataset, tokenizer, max_length=max_length)


def _format_training_conversation(conv: Dict[str, str]) -> str:
    """Format a single conversation for training."""
    parts = []
    for role, content in conv.items():
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
    return "\n".join(parts)
