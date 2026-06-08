"""Dataset loaders for various training tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Dataset for text-based training (LLM, LoRA, QLoRA)."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        max_length: int = 512,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"text": self.texts[idx]}
        if self.labels:
            item["label"] = self.labels[idx]
        return item


class ImageDataset(Dataset):
    """Dataset for image-based training (CNN, LCM)."""

    def __init__(
        self,
        images: List[torch.Tensor],
        labels: Optional[List[int]] = None,
        transform: Optional[Any] = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        item = {"image": image}
        if self.labels:
            item["label"] = self.labels[idx]
        return item


class ConversationDataset(Dataset):
    """Dataset for conversation/chat fine-tuning."""

    def __init__(
        self,
        conversations: List[Dict[str, str]],
        tokenizer: Optional[Any] = None,
        max_length: int = 2048,
    ) -> None:
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        conv = self.conversations[idx]
        formatted = self._format_conversation(conv)
        return {"text": formatted}

    def _format_conversation(self, conv: Dict[str, str]) -> str:
        """Format conversation into training text."""
        parts = []
        for role, content in conv.items():
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "user":
                parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
        return "\n".join(parts)


def create_dataset_from_config(
    source: Union[str, List[str]],
    dataset_type: str = "text",
    **kwargs: Any,
) -> Dataset:
    """Factory function to create datasets from config."""
    if dataset_type == "text":
        texts = source if isinstance(source, list) else [source]
        return TextDataset(texts, **kwargs)
    elif dataset_type == "image":
        return ImageDataset(source, **kwargs)
    elif dataset_type == "conversation":
        convs = source if isinstance(source, list) else [{"user": str(source)}]
        return ConversationDataset(convs, **kwargs)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
