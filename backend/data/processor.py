"""Data preprocessing and augmentation pipelines."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union


class DataProcessor:
    """Chainable data preprocessing pipeline."""

    def __init__(self) -> None:
        self._steps: List[Callable] = []

    def add_step(self, fn: Callable, name: Optional[str] = None) -> "DataProcessor":
        """Add a preprocessing step."""
        self._steps.append(fn)
        return self

    def process(self, data: Any) -> Any:
        for step in self._steps:
            data = step(data)
        return data

    def clear(self) -> None:
        """Remove all steps."""
        self._steps.clear()


def build_text_pipeline(
    max_length: int = 512,
    add_special_tokens: bool = True,
) -> DataProcessor:
    """Build standard text preprocessing pipeline."""
    processor = DataProcessor()
    processor.add_step(lambda x: x.strip(), "strip")
    processor.add_step(lambda x: x[:max_length] if len(x) > max_length else x, "truncate")
    return processor


def build_image_pipeline(
    size: tuple = (224, 224),
    normalize: bool = True,
) -> DataProcessor:
    """Build image preprocessing pipeline."""
    processor = DataProcessor()

    try:
        from torchvision import transforms
        transform_list = [transforms.Resize(size), transforms.ToTensor()]
        if normalize:
            transform_list.append(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ))
        compose = transforms.Compose(transform_list)
        processor.add_step(lambda x: compose(x), "torchvision_transform")
    except ImportError:
        processor.add_step(lambda x: x, "identity_transform")

    return processor


def format_conversation(
    messages: List[Dict[str, str]],
    template: str = "chatml",
) -> str:
    """Format a conversation according to a template."""
    if template == "chatml":
        return _format_chatml(messages)
    elif template == "llama":
        return _format_llama(messages)
    else:
        return _format_chatml(messages)


def _format_chatml(messages: List[Dict[str, str]]) -> str:
    """Format using ChatML template."""
    result = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        result += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    result += "<|im_start|>assistant\n"
    return result


def _format_llama(messages: List[Dict[str, str]]) -> str:
    """Format using LLaMA chat template."""
    result = "<s>"
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            result += f"<<SYS>>\n{content}\n<</SYS>>\n"
        elif role == "user":
            result += f"[INST] {content} [/INST]"
        elif role == "assistant":
            result += f" {content}</s>"
    return result
