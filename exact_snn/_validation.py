"""Shared input validation for spike-time tensors.

All exact blocks treat spike times as float tensors where ``inf`` marks a
silent neuron (never fired). ``NaN`` is always corrupt input: it cannot be a
valid spike time and silently poisons autograd gradients, so we reject it at
every public entry point rather than propagating it.
"""
import torch


def validate_spike_times(t: torch.Tensor,
                         name: str = "Input") -> None:
    """Reject non-float and NaN spike-time tensors.

    Args:
        t: candidate spike-time tensor.
        name: label used in error messages.

    Raises:
        ValueError: if ``t`` is not floating-point or contains NaN.
    """
    if not t.is_floating_point():
        raise ValueError(f"{name} spike times must use a floating-point dtype")
    if torch.isnan(t).any():
        raise ValueError(f"{name} spike times must not contain NaN "
                         "(use float('inf') for silent neurons)")