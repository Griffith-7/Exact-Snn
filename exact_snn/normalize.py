"""SpikeNorm: batch normalization adapted for spike times (optional module).

Analogous to batch normalization but operating on spike times:
    t_norm = (t - running_mean) / sqrt(running_var + eps)
    t_out = gamma * t_norm + beta

This is an optional nn.Module. Users are NOT required to use it -- it is a
drop-in, standard-PyTorch-compatible component (gamma/beta are nn.Parameter,
running stats are buffers) that helps stabilize deeper spike networks.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SpikeNorm(nn.Module):
    """Normalize spike times across the batch dimension.

    Args:
        n_features: Number of feature channels (spike-time rows) to normalize.
        momentum: Exponential moving average factor for running stats.
        eps: Small constant added to variance for numerical stability.
    """

    def __init__(self, n_features: int, momentum: float = 0.1, eps: float = 1e-5,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.gamma = nn.Parameter(
            torch.ones(int(n_features), dtype=dtype, device=device))
        self.beta = nn.Parameter(
            torch.zeros(int(n_features), dtype=dtype, device=device))
        self.register_buffer("running_mean",
                             torch.zeros(int(n_features), dtype=dtype,
                                         device=device))
        self.register_buffer("running_var",
                             torch.ones(int(n_features), dtype=dtype,
                                        device=device))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize spike times across the first (feature) dimension.

        Only the neurons that actually fired in this batch participate: the
        mean/variance are computed over the finite spike times of each feature,
        silent (``inf``) entries are left untouched as ``inf``, and the running
        statistics are updated per-feature only where that feature fired.
        This mirrors how the enclosing ``ExactSpikingFFN`` treats silent hidden
        cells (they key nothing into the next layer), so a hidden layer with a
        valid silent neuron never crashes the block -- and a batch that is silent
        on a feature never poisons ``running_mean``/``running_var`` with NaN.

        Args:
            t: Spike times of shape ``(n_features, B)``.

        Returns:
            Normalized spike times of the same shape.
        """
        if not t.is_floating_point():
            raise ValueError("SpikeNorm requires floating-point input")
        finite = torch.isfinite(t)                        # (n, B)
        count = finite.sum(dim=1)                         # (n,)
        if self.training:
            nz = count > 0
            s = torch.where(finite, t, torch.zeros_like(t)).sum(dim=1)
            mean = s / count.clamp(min=1)
            centered = torch.where(
                finite, t - mean.unsqueeze(1), torch.zeros_like(t))
            var = (centered * centered).sum(dim=1) / count.clamp(min=1)
            mean_d, var_d = mean.detach(), var.detach()
            m = self.momentum
            upd_mean = (1 - m) * self.running_mean[nz] + m * mean_d[nz]
            upd_var = (1 - m) * self.running_var[nz] + m * var_d[nz]
            self.running_mean[nz] = upd_mean.to(self.running_mean.dtype)
            self.running_var[nz] = upd_var.to(self.running_var.dtype)
        else:
            mean = self.running_mean
            var = self.running_var
        # Keep the normalization math finite at silent entries (paper over inf
        # with a 0 placeholder); `where` re-selects the true `inf` afterwards,
        # so no inf arithmetic reaches the autograd graph.
        safe_t = torch.where(finite, t, torch.zeros_like(t))
        t_norm = (safe_t - mean.unsqueeze(1)) / torch.sqrt(
            var.unsqueeze(1) + self.eps)
        out = torch.where(
            finite,
            self.gamma.unsqueeze(1) * t_norm + self.beta.unsqueeze(1),
            t)
        return out
