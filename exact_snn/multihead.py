"""SP-05b: exact spiking multi-head attention as a drop-in torch.nn module.

A multi-head wrapper around `ExactSpikingAttention`. Spike-time signals cannot
be sliced into per-head feature subspaces the way dense vectors can (a spike
time is a single scalar per neuron), so each head is an *independent* single-
head block with its own Q/K/V `ExactTTFSLinear` projections that attend over
all tokens. The per-head attended spike times are then fused:

    fuse="min"  (default): element-wise earliest-spike-wins across heads.
               Keeps the result a genuine spike time; exact gradient via
               autograd (only the winning head's path receives gradient).
    fuse="mean": element-wise average across heads (still exact closed-form).
    fuse="full": stack -> (n_heads, n_in, B) raw per-head times.

Every inner head is exact (IFT projections + closed-form combine), so the
whole block -- including the fusion -- has exact gradients and is consumable
by the existing exact-gradient layers. See `exact_snn.attention`.

Public API:
    from exact_snn.multihead import ExactSpikingMultiHeadAttention
    from exact_snn.extended import ExactSpikingMultiHeadAttention
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from exact_snn.attention import ExactSpikingAttention, _validate_layer_config

__all__ = ["ExactSpikingMultiHeadAttention"]


class ExactSpikingMultiHeadAttention(nn.Module):
    """Exact spiking multi-head attention.

    Args:
        n: square token/neuron count (n_in == n_out, same as the single-head
            block).
        n_heads: number of independent single-head attention blocks.
        fuse: "min" (default; earliest-spike-wins), "mean", or "full".
        Each additional kwarg is forwarded to every `ExactSpikingAttention`
        head (see its docstring: tm, ts, theta, t_max, w_scale, bias_val,
        grid_pts, temp, combine, dtype, device, peak_tol).

    Output:
        fuse="full": (n_heads, n, B); otherwise (n, B) spike times.
    """

    def __init__(self, n: int, n_heads: int = 2, fuse: str = "min",
                 seed: int = 0, **forwarded: object) -> None:
        super().__init__()
        _validate_layer_config(n, n, forwarded.get("tm", 15.0),
                               forwarded.get("ts", 4.0),
                               forwarded.get("theta", 1.0),
                               forwarded.get("t_max", 40.0),
                               forwarded.get("grid_pts", 2001))
        n_heads = int(n_heads)
        if n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if fuse not in ("min", "mean", "full"):
            raise ValueError("fuse must be 'min', 'mean', or 'full'")
        dtype = forwarded.get("dtype", None)
        if dtype is None:
            forwarded["dtype"] = torch.float32
        elif not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")
        self.n = int(n)
        self.n_heads = n_heads
        self.fuse = str(fuse)
        self.seed = int(seed)
        self.heads = nn.ModuleList(
            ExactSpikingAttention(n, n, seed=self.seed + 17 * h, **forwarded)
            for h in range(n_heads)
        )

    def extra_repr(self) -> str:
        return f"n={self.n}, n_heads={self.n_heads}, fuse={self.fuse!r}"

    def forward(self, t_in: torch.Tensor) -> torch.Tensor:
        """t_in (n, B) -> fused attended spike times (see class docstring)."""
        outs = torch.stack([h(t_in) for h in self.heads], dim=0)
        if self.fuse == "full":
            return outs
        if self.fuse == "mean":
            return outs.mean(dim=0)
        return outs.min(dim=0).values

    def calibrate_init_fire(self, target: float = 0.5, n_probe: int = 32,
                            cal_grid_pts: int = 65, iters: int = 6) -> None:
        """Calibrate every head's Q/K/V projection firing (see
        ``ExactTTFSLinear.calibrate_init_fire``)."""
        for head in self.heads:
            head.calibrate_init_fire(target=target, n_probe=n_probe,
                                     cal_grid_pts=cal_grid_pts, iters=iters)