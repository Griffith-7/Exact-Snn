"""Optional data-utility helpers for Exact-SNN (lazy opt-in).

A thin, dependency-free utility module. Currently provides `spike_time_augment`
(augmentation of spike times with additive Gaussian noise and random time
shifts), ported from the original research codebase as a reusable function.

Optional -- import only if needed:
    from exact_snn.util import spike_time_augment
"""
from __future__ import annotations

import torch


def spike_time_augment(t_in: torch.Tensor, t_max: float = 40.0,
                       noise_std: float = 0.1, time_shift: float = 0.5) -> torch.Tensor:
    """Augment spike times with additive noise and random time shifts.

    Args:
        t_in: Input spike times of arbitrary shape.
        t_max: Maximum allowed spike time (clamp upper bound).
        noise_std: Standard deviation of Gaussian noise added per spike.
        time_shift: Maximum uniform random shift applied per sample.

    Returns:
        Augmented spike times clamped to ``[0, t_max]``.
    """
    noise = torch.randn_like(t_in) * noise_std
    shifted = t_in + (torch.rand(1, device=t_in.device) * 2 - 1) * time_shift
    return torch.clamp(shifted + noise, 0.0, t_max)
