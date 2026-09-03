"""SpikeNorm: batch normalization adapted for spike times (optional module).

Analogous to batch normalization but operating on spike times:
    t_norm = (t - running_mean) / sqrt(running_var + eps)
    t_out = gamma * t_norm + beta

This is an optional nn.Module. Users are NOT required to use it -- it is a
drop-in, standard-PyTorch-compatible component (gamma/beta are nn.Parameter,
running stats are buffers) that helps stabilize deeper spike networks.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SpikeNorm(nn.Module):
    """Normalize spike times across the batch dimension.

    Args:
        n_features: Number of feature channels (spike-time rows) to normalize.
        momentum: Exponential moving average factor for running stats.
        eps: Small constant added to variance for numerical stability.
    """

    def __init__(self, n_features: int, momentum: float = 0.1, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.gamma = nn.Parameter(torch.ones(int(n_features)))
        self.beta = nn.Parameter(torch.zeros(int(n_features)))
        self.register_buffer("running_mean", torch.zeros(int(n_features)))
        self.register_buffer("running_var", torch.ones(int(n_features)))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize spike times across the first (feature) dimension.

        Args:
            t: Spike times of shape ``(n_features, B)``.

        Returns:
            Normalized spike times of the same shape.
        """
        if self.training:
            mean = t.mean(dim=1)
            var = t.var(dim=1, unbiased=False)
            self.running_mean.mul_(1 - self.momentum).add_(
                mean.detach() * self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(
                var.detach() * self.momentum)
        else:
            mean = self.running_mean
            var = self.running_var
        t_norm = (t - mean.unsqueeze(1)) / torch.sqrt(var.unsqueeze(1) + self.eps)
        return self.gamma.unsqueeze(1) * t_norm + self.beta.unsqueeze(1)
