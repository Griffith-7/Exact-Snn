"""Loss functions for Exact-SNN (autograd-compatible scalar losses).

Adds `rate_latency_loss` -- a combined rate (spike-count) + latency (first
spike) cross-entropy. The base timing loss and spike-count loss live in the
core package (see exact_snn/__init__.py and exact_snn/extended.py); this module
provides the additional combined loss as a drop-in, autograd-compatible scalar.

Unlike the original v1 losses (which returned `(float, dL_tensor)` for a manual
backward pass), these losses return a scalar `torch.Tensor` so they integrate
with standard `loss.backward()`.
"""
from __future__ import annotations

import torch


def rate_latency_loss(t_all: torch.Tensor, y: torch.Tensor,
                      t_max: float, beta: float = 1.0,
                      temp: float = 2.0) -> torch.Tensor:
    """Combined rate-latency loss: spike-count CE + latency CE on first spike.

    t_all: (n_out, B, K) multi-spike times.

    Rate term uses a soft, differentiable spike count via sigmoid (so gradient
    flows through every spike time), and latency term uses the first-spike time
    with softmax(-beta * t). Both logits are combined before a single CE.

    Returns a scalar differentiable loss tensor (supports .backward()).
    """
    B = t_all.shape[1]
    # --- rate: soft spike counts (differentiable) ---
    t_soft = torch.where(torch.isfinite(t_all), t_all,
                         torch.full_like(t_all, 2.0 * t_max))
    counts = torch.sigmoid(-(t_soft - t_max) / temp).sum(dim=2)
    counts = counts - counts.mean(dim=0, keepdim=True)
    logits_rate = beta * counts
    # --- latency: first-spike times ---
    t_first = t_all[:, :, 0]
    t_for_lat = torch.where(torch.isfinite(t_first), t_first,
                            torch.full_like(t_first, 2.0 * t_max + 10.0))
    logits_lat = -beta * t_for_lat
    # --- combined softmax CE ---
    logits = logits_rate + logits_lat
    logits = logits - logits.max(dim=0, keepdim=True).values
    p = torch.exp(logits)
    p = p / p.sum(dim=0, keepdim=True)
    loss = -torch.log(p[y, torch.arange(B, device=t_all.device)] + 1e-12).mean()
    return loss
