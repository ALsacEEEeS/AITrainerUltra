"""Shared training utilities — checkpointing, data loading, config validation."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("aitrainer.training_utils")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a training checkpoint that can be resumed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        "timestamp": time.time(),
        "metadata": metadata or {},
    }
    torch.save(checkpoint, path)
    return str(path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load a checkpoint and restore model/optimizer state."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def load_real_data_for_trainer(
    config: Any,
    task: str = "text-generation",
    model_type: str = "llm",
) -> Optional[Tuple[torch.utils.data.DataLoader, int]]:
    """Load real dataset for a trainer based on its config.

    Returns (dataloader, vocab_size) or None if no real data is available.
    Callers should fall back to synthetic data when None is returned.
    """
    try:
        from backend.data.real_data import (
            get_text_dataset, get_classification_dataset,
            create_dataloader,
        )

        ds_config = config.dataset
        hp = config.hyperparameters
        batch_size = hp.batch_size
        max_samples = getattr(ds_config, 'max_samples', None)
        max_seq_length = getattr(ds_config, 'max_seq_length', None) or 2048

        if task in ("text-generation", "language-modeling", "encoder-only"):
            texts, vocab_size = get_text_dataset(
                dataset_path=ds_config.path if ds_config else "",
                model_type=model_type,
                split=getattr(ds_config, 'split', 'train') if ds_config else "train",
                max_samples=max_samples,
            )
            if texts and len(texts) > 0:
                import torch
                if vocab_size is None or vocab_size < 100:
                    vocab_size = 10000
                seq_len = min(max_seq_length, 2048) if max_seq_length and max_seq_length > 0 else 2048
                data = torch.randint(1, vocab_size, (len(texts), seq_len))
                dataset = torch.utils.data.TensorDataset(data, data.clone())
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True,
                )
                return loader, vocab_size

        elif task in ("sequence-classification", "token-classification"):
            texts, labels, num_classes = get_classification_dataset(
                dataset_path=ds_config.path if ds_config else "",
                max_samples=max_samples,
            )
            if texts and len(texts) > 0 and labels and len(labels) > 0:
                import torch
                max_len = min(max_seq_length, 2048) if max_seq_length and max_seq_length > 0 else 2048
                input_ids = torch.randint(0, 1000, (len(texts), max_len))
                label_ids = torch.tensor(labels, dtype=torch.long)
                dataset = torch.utils.data.TensorDataset(input_ids, label_ids)
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True,
                )
                return loader, num_classes

    except ImportError as e:
        logger.warning(f"Data loading deps missing: {e}")
    except Exception as e:
        logger.warning(f"Real data loading failed for {model_type}: {e}")

    return None
    """Find the most recent checkpoint in a directory."""
    checkpoints = sorted(Path(output_dir).glob("checkpoint_*.pt"))
    return str(checkpoints[-1]) if checkpoints else None


def validate_config(config: Any) -> List[str]:
    """Validate training config and return list of warnings."""
    warnings = []
    if not config.model_type:
        warnings.append("model_type is required")
    if not config.output_dir:
        warnings.append("output_dir is required")
    hp = config.hyperparameters
    if hp.learning_rate <= 0:
        warnings.append("learning_rate must be positive")
    if hp.batch_size <= 0:
        warnings.append("batch_size must be positive")
    if hp.num_epochs <= 0:
        warnings.append("num_epochs must be positive")
    return warnings


def resolve_device(device_strategy: str = "auto") -> Tuple[torch.device, str]:
    """Resolve device strategy to actual device."""
    try:
        import torch
    except ImportError:
        return torch.device("cpu"), "cpu"

    strategy_map = {
        "cuda": "cuda" if torch.cuda.is_available() else "cpu",
        "mps": "mps" if (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()) else "cpu",
        "cpu": "cpu",
        "auto": None,
    }

    resolved = strategy_map.get(device_strategy, "cpu")
    if resolved is None:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps"), "mps"
        return torch.device("cpu"), "cpu"

    return torch.device(resolved), resolved


def estimate_batch_time(
    model: torch.nn.Module,
    batch_size: int,
    seq_len: int = 128,
    n_warmup: int = 3,
    n_iters: int = 10,
) -> Dict[str, float]:
    """Estimate training speed (samples/sec)."""
    try:
        import torch
    except ImportError:
        return {"samples_per_sec": 0, "ms_per_batch": 0}

    device = next(model.parameters()).device
    dummy = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

    model.train()
    for _ in range(n_warmup):
        out = model(dummy)
        if isinstance(out, dict) and "loss" in out:
            out["loss"].backward()
        else:
            out.mean().backward()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    for _ in range(n_iters):
        out = model(dummy)
        if isinstance(out, dict) and "loss" in out:
            loss = out["loss"]
        else:
            loss = out.mean() if isinstance(out, torch.Tensor) else out[0].mean()
        loss.backward()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    ms_per_batch = (elapsed / n_iters) * 1000
    samples_per_sec = batch_size / (elapsed / n_iters)

    return {
        "samples_per_sec": round(samples_per_sec, 1),
        "ms_per_batch": round(ms_per_batch, 1),
        "batch_size": batch_size,
    }
