"""Synthetic data generators for training from scratch and pretraining.

Produces random token sequences, image tensors, and paired multimodal data
for use when real datasets are not available.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch


def generate_text_batch(
    vocab_size: int = 10000,
    seq_len: int = 64,
    batch_size: int = 32,
    num_classes: Optional[int] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, None]]:
    """Generate a synthetic batch of tokenized text data.

    For language modeling, targets == input_ids.
    For classification, targets are random class labels.
    """
    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len))
    if num_classes:
        labels = torch.randint(0, num_classes, (batch_size,))
        return input_ids, labels
    return input_ids, input_ids.clone()


def generate_image_batch(
    image_size: Tuple[int, int, int] = (3, 32, 32),
    batch_size: int = 32,
    num_classes: Optional[int] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, None]]:
    """Generate a synthetic batch of random image data."""
    images = torch.randn(batch_size, *image_size)
    if num_classes:
        labels = torch.randint(0, num_classes, (batch_size,))
        return images, labels
    return images, None


def generate_multimodal_batch(
    batch_size: int = 16,
    image_size: Tuple[int, int, int] = (3, 224, 224),
    text_seq_len: int = 64,
    vocab_size: int = 10000,
    num_classes: int = 10,
) -> Dict[str, torch.Tensor]:
    """Generate a synthetic batch of paired image-text data."""
    return {
        "images": torch.randn(batch_size, *image_size),
        "texts": torch.randint(2, vocab_size, (batch_size, text_seq_len)),
        "labels": torch.randint(0, num_classes, (batch_size,)),
    }


def generate_pretraining_corpus(
    vocab_size: int = 10000,
    seq_len: int = 128,
    num_samples: int = 1000,
    seed: int = 42,
) -> torch.Tensor:
    """Generate a larger corpus for pretraining from scratch.

    Uses a simple Markov-like structure so the model can learn patterns.
    """
    rng = random.Random(seed)
    data = []
    for _ in range(num_samples):
        topic = rng.randint(1, min(vocab_size, 100))
        sample = [topic]
        for _ in range(seq_len - 1):
            prev = sample[-1]
            # ~30% chance to stay on topic, otherwise random
            if rng.random() < 0.3:
                next_token = prev
            elif rng.random() < 0.5:
                next_token = prev + rng.randint(-5, 5)
            else:
                next_token = rng.randint(1, vocab_size - 1)
            next_token = max(1, min(next_token, vocab_size - 1))
            sample.append(next_token)
        data.append(sample[:seq_len])

    return torch.tensor(data, dtype=torch.long)


class SyntheticDataset(torch.utils.data.Dataset):
    """A synthetic dataset that generates data on-the-fly.

    Useful for testing training pipelines and scratch training without
    requiring real data downloads.
    """

    def __init__(
        self,
        dataset_type: str = "text",
        num_samples: int = 512,
        vocab_size: int = 10000,
        seq_len: int = 64,
        image_size: Tuple[int, int, int] = (3, 32, 32),
        num_classes: Optional[int] = None,
    ) -> None:
        self.dataset_type = dataset_type
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.dataset_type == "text":
            x = torch.randint(1, self.vocab_size, (self.seq_len,))
            y = x.clone()
            item = {"input_ids": x, "labels": y}
        elif self.dataset_type == "image":
            x = torch.randn(*self.image_size)
            item = {"images": x}
        elif self.dataset_type == "multimodal":
            x_img = torch.randn(*self.image_size)
            x_txt = torch.randint(1, self.vocab_size, (self.seq_len,))
            item = {"images": x_img, "input_ids": x_txt}
        else:
            item = {"data": torch.randn(self.seq_len)}

        if self.num_classes:
            item["labels"] = torch.tensor(random.randint(0, self.num_classes - 1))
        return item


def create_pretraining_iterator(
    vocab_size: int = 10000,
    seq_len: int = 128,
    batch_size: int = 32,
    infinite: bool = True,
) -> Any:
    """Create an infinite iterator that yields batches of synthetic pretraining data."""
    while True:
        x = torch.randint(1, vocab_size, (batch_size, seq_len))
        yield {"input_ids": x, "labels": x.clone()}


# Dataset size estimator
def estimate_dataset_size(
    vocab_size: int,
    seq_len: int,
    num_samples: int,
) -> Dict[str, float]:
    """Estimate the memory footprint of a synthetic dataset."""
    bytes_per_token = 4  # int32
    tokens = num_samples * seq_len
    total_bytes = tokens * bytes_per_token
    return {
        "tokens": tokens,
        "bytes": total_bytes,
        "mb": round(total_bytes / (1024 * 1024), 2),
        "gb": round(total_bytes / (1024 ** 3), 4),
    }
