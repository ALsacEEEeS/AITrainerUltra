"""Custom neural network layers with fallbacks for older PyTorch versions."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Fallback for PyTorch < 2.4.0 which doesn't have nn.RMSNorm.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight).to(x.dtype)


def get_rmsnorm(d_model: int) -> nn.Module:
    """Get RMSNorm - uses native if available, otherwise fallback."""
    if hasattr(nn, 'RMSNorm'):
        try:
            return nn.RMSNorm(d_model)
        except Exception:
            pass
    return RMSNorm(d_model)
