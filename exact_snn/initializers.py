"""Optional weight initialization helpers for Exact-SNN layers.

These are thin, optional utilities that write initialization into an existing
weight tensor of shape ``(fan_out, fan_in + 1)`` (the trailing column is the
bias, which these initializers set to a small value). Users who prefer standard
PyTorch initialization can ignore these entirely — they exist for convenience
and parity with the original research codebase.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def xavier_init(weight: torch.Tensor, fan_in: int, fan_out: int,
                seed: int = 0) -> None:
    """Xavier/Glorot uniform init for a weight tensor ``(fan_out, fan_in+1)``.

    Writes in-place into ``weight`` (an nn.Parameter or plain tensor). The last
    column (bias) is set to a small value.
    """
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    rng = np.random.default_rng(seed)
    w_np = rng.uniform(-limit, limit, (fan_out, fan_in + 1)).astype(np.float64)
    w_np[:, -1] = 0.1
    with torch.no_grad():
        weight.copy_(torch.tensor(w_np, dtype=weight.dtype, device=weight.device))


def kaiming_init(weight: torch.Tensor, fan_in: int, fan_out: int,
                 seed: int = 0, leaky_relu_slope: float = 0.01) -> None:
    """He/Kaiming init for a weight tensor ``(fan_out, fan_in+1)``.

    Writes in-place into ``weight``. The last column (bias) is set to a small
    value.
    """
    std = math.sqrt(2.0 / ((1 + leaky_relu_slope ** 2) * fan_in))
    rng = np.random.default_rng(seed)
    w_np = (rng.standard_normal((fan_out, fan_in + 1)) * std).astype(np.float64)
    w_np[:, -1] = 0.1
    with torch.no_grad():
        weight.copy_(torch.tensor(w_np, dtype=weight.dtype, device=weight.device))
