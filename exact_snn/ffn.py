"""SP-05: exact spiking feed-forward block (g(t) > theta dynamics).

An FFN block is the feed-forward companion to `ExactSpikingAttention`: two
exact TTFS projection steps with an expand--contract shape,

    t_h  = TTFS(n_in  -> n_hidden)(t_in)      # nonlinear spike-time map
    t_out = TTFS(n_hidden -> n_out)(t_h),     # inter-layer spike feed

where every layer is an `ExactTTFSLinear`, i.e. the membrane sum
``g(t) = sum_k W_k K(t - t_k)`` crossing the threshold ``g(t) > theta`` is
solved exactly (first-spike root), and the backward uses the exact IFT rule.
There is no surrogate and no rate coding: spike times are passed between
layers as-is (silent hidden cells carry ``inf`` and contribute ``K(t-inf) = 0``
to the next layer, consistent with the library's silent semantics).

Public API:
    ExactSpikingFFN            - the nn.Module block (layers .in_proj/.out_proj)

Optional companion module (lazy opt-in):
    from exact_snn.ffn import ExactSpikingFFN
    from exact_snn.extended import ExactSpikingFFN
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from exact_snn import ExactTTFSLinear

__all__ = ["ExactSpikingFFN"]


class ExactSpikingFFN(nn.Module):
    """Exact spiking feed-forward block: TTFS -> hidden -> TTFS -> out.

    Args:
        n_in: input spike-time dimension.
        n_hidden: expansion dimension (spike-time feature map).
        n_out: output spike-time dimension.
        tm, ts, theta, t_max: TTFS dynamics params (see ExactTTFSLinear).
        w_scale, bias_val, grid_pts, seed, dtype, device, peak_tol: forwarded
            to both `ExactTTFSLinear` projections.
        residual: if True (requires n_out == n_in), fuse a spike-time skip
            ``min(t_out, t_in)`` (earliest-spike-wins) with the input.
        use_norm: if True, apply a `SpikeNorm` between the two projections.
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 tm: float = 15.0, ts: float = 4.0, theta: float = 1.0,
                 t_max: float = 40.0, w_scale: float = 0.2,
                 bias_val: float = 1.5, grid_pts: int = 2001, seed: int = 0,
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[torch.device] = None,
                 peak_tol: float = 1e-2, residual: bool = False,
                 use_norm: bool = False) -> None:
        super().__init__()
        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_out = int(n_out)
        self.tm = float(tm)
        self.ts = float(ts)
        self.theta = float(theta)
        self.t_max = float(t_max)
        self.residual = bool(residual)
        self.use_norm = bool(use_norm)
        if self.residual and n_out != n_in:
            raise ValueError("residual=True requires n_out == n_in")
        dtype = torch.float32 if dtype is None else dtype
        common = dict(tm=self.tm, ts=self.ts, theta=self.theta,
                      t_max=self.t_max, w_scale=float(w_scale),
                      bias_val=float(bias_val), grid_pts=int(grid_pts),
                      dtype=dtype, device=device, peak_tol=float(peak_tol))
        self.in_proj = ExactTTFSLinear(n_in, n_hidden, seed=int(seed), **common)
        if self.use_norm:
            from exact_snn.normalize import SpikeNorm
            self.norm = SpikeNorm(self.n_hidden, dtype=dtype, device=device)
        self.out_proj = ExactTTFSLinear(n_hidden, n_out,
                                        seed=int(seed) + 1000, **common)

    def extra_repr(self) -> str:
        return (f"n_in={self.n_in}, n_hidden={self.n_hidden}, "
                f"n_out={self.n_out}, tm={self.tm:.1f}, ts={self.ts:.1f}, "
                f"theta={self.theta}")

    def forward(self, t_in: torch.Tensor) -> torch.Tensor:
        """t_in: (n_in, B) -> (n_out, B) output spike times."""
        t_h = self.in_proj(t_in)
        if self.use_norm:
            t_h = self.norm(t_h)
        t_out = self.out_proj(t_h)
        if self.residual:
            t_out = torch.minimum(t_out, t_in)
        return t_out

    def calibrate_init_fire(self, target: float = 0.5, n_probe: int = 32,
                            cal_grid_pts: int = 65, iters: int = 6) -> None:
        """Calibrate each projection's bias in sequence (propagating spike
        times between layers), so roughly `target` fires per layer. See
        ``ExactTTFSLinear.calibrate_init_fire``."""
        from exact_snn import _forward_layer_torch, _K
        from exact_snn.existence import peak_margin_torch
        dev = self.in_proj.weight.device
        dtype = self.in_proj.weight.dtype
        cal_grid = torch.linspace(0.0, self.t_max, int(cal_grid_pts),
                                  dtype=dtype, device=dev)
        n_probe = int(n_probe)
        probe = (torch.rand(self.n_in, n_probe, dtype=dtype, device=dev)
                 * 0.8 * self.t_max + 0.1)
        for layer in (self.in_proj, self.out_proj):
            with torch.no_grad():
                for _ in range(int(iters)):
                    W = layer.weight
                    t_post, _ = _forward_layer_torch(
                        W, probe, layer.t_bias, layer.theta, cal_grid,
                        layer.tm, layer.ts, layer._alpha, layer.k_peak,
                        peak_tol=layer.peak_tol)
                    fired = torch.isfinite(t_post)
                    t_peak, u_peak = peak_margin_torch(
                        W, probe, layer.t_bias, layer.theta, cal_grid,
                        layer.tm, layer.ts, layer._alpha, layer.k_peak)
                    need = torch.where(
                        fired, torch.full_like(u_peak, layer.theta), u_peak)
                    vals, idx = torch.sort(need, dim=1, descending=True)
                    k = max(1, int(round(float(target) * n_probe)))
                    delta = layer.theta - vals[:, k - 1]
                    t_k = t_peak[torch.arange(layer.n_out, dtype=torch.long,
                                              device=dev), idx[:, k - 1]]
                    h = _K(t_k - layer.t_bias, layer.tm, layer.ts,
                           layer._alpha, layer.k_peak).clamp(min=1e-3)
                    layer.weight[:, -1] = (
                        layer.weight[:, -1] + delta / h).clamp(min=0.0)
            t_post2, _ = _forward_layer_torch(
                layer.weight.detach(), probe, layer.t_bias, layer.theta,
                cal_grid, layer.tm, layer.ts, layer._alpha, layer.k_peak,
                peak_tol=layer.peak_tol)
            probe = t_post2
            if self.use_norm and layer is self.in_proj:
                probe = self.norm(probe)