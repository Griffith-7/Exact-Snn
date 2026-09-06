"""Surrogate-gradient twin of the exact TTFS layers, used ONLY by the
exact-vs-surrogate learning comparison (benchmarks/sine_waveform_comparison).

This file deliberately does NOT ship in the exact_snn package. The library's
policy is exact IFT gradients everywhere and no surrogates; this benchmark
twin exists solely so we can MEASURE how the exact rules compare against the
standard "sigmoid-membrane surrogate" on the same network, the same loss and
the same data.

Forward  = the library's exact first-spike solver (`_forward_layer_torch`).
           Outputs are bit-identical to an exact layer, so both networks in
           the comparison evaluate exactly the same function at every epoch.
Loss     = identical (SSE on spike times) in both twins.
Backward = the ONLY difference. Exact uses the IFT adjoint  lam/up  (sharp,
           unbounded near a flat crossing). The surrogate replaces it with a
           smooth sigmoid-membrane weight phi = sigmoid'(beta*margin) on the
           fired neurons, magnitude-matched per batch to the exact signal so
           that learning-rate effects stay comparable.

Silent neurons carry zero gradient in BOTH twins. That is an honest finding
of the benchmark: with a spike-time readout and an SSE loss, the surrogate
gains no silent-rescue advantage -- silent outputs contribute a constant to
the loss, so their upstream gradient is zero either way. (Surrogate silent
rescue only appears with rate/membrane losses, which are out of scope for
this exact first-spike comparison.)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from exact_snn import (
    _K,
    _Kd,
    ExactTTFSLinear,
    _forward_layer_torch,
)
from exact_snn.attention import ExactAttentionCombineFn
from exact_snn.existence import peak_margin_torch

__all__ = ["SurrogateTTFSLinear", "SurrogateSpikingAttention"]


class SurrogateTTFSLinearFn(torch.autograd.Function):
    """One TTFS layer: exact forward solve, smooth sigmoid-margin backward.

    Forward is `_forward_layer_torch` (exact root solve, same output times as
    the library layers). Backward substitutes the standard membrane-margin
    surrogate weighting for the exact 1/up adjoint on fired neurons.
    """

    @staticmethod
    def forward(ctx, W, t_prev, t_bias, theta, tm, ts, alpha, k_peak,
                grid, peak_tol, beta):
        t_post, up = _forward_layer_torch(
            W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak,
            peak_tol=float(peak_tol))

        # Margin channel: fired neurons sit ON threshold -> smooth-weight max;
        # silent neurons sit below it by their (negative) peak margin.
        fired = torch.isfinite(t_post)
        t_peak, u_peak = peak_margin_torch(
            W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak)
        margin = torch.where(fired,
                             torch.zeros_like(u_peak),
                             u_peak - theta)          # <= 0 for silent

        ctx.save_for_backward(W, t_prev, t_post, up, margin)
        ctx.t_bias = float(t_bias)
        ctx.tm = float(tm)
        ctx.ts = float(ts)
        ctx.alpha = bool(alpha)
        ctx.k_peak = float(k_peak)
        ctx.beta = float(beta)
        return t_post

    @staticmethod
    def backward(ctx, grad_output):
        W, t_prev, t_post, up, margin = ctx.saved_tensors
        n_cur, n_inp = W.shape
        n_in = n_inp - 1
        B = t_post.shape[1]
        dev = W.device
        dtype = W.dtype
        tm, ts = ctx.tm, ctx.ts
        alpha, k_peak, beta = ctx.alpha, ctx.k_peak, ctx.beta
        t_bias = ctx.t_bias

        grad = torch.zeros_like(W)
        lam_prev = torch.zeros((n_in, B), dtype=dtype, device=dev)

        fired = torch.isfinite(t_post)
        if not fired.any():
            return grad, lam_prev, None, None, None, None, None, None, None, None, None

        la = torch.where(fired, grad_output, torch.zeros_like(grad_output))
        up_safe = torch.where(up != 0.0, up, torch.ones_like(up))
        adj_exact = torch.where(fired & (up != 0.0), la / up_safe,
                                torch.zeros_like(la))

        # Soft sigmoid-membrane weight: fired -> 0.25 (constant), silent -> ~0.
        s = torch.sigmoid(beta * margin)
        phi = s * (1.0 - s)
        sur = la * phi
        # Magnitude-match to the exact adjoint signal so both rules receive
        # comparable gradient magnitudes (the benchmark isolates the *shape*
        # of the per-neuron weighting, not the overall scale).
        denom = sur.abs().sum() + 1e-12
        scale = adj_exact.abs().sum() / denom
        adj_sur = torch.where(fired, sur * scale, torch.zeros_like(sur))

        grad[:, n_in] = -(adj_sur * _K(t_post - t_bias,
                                       tm, ts, alpha, k_peak)).sum(dim=1)
        t_data = t_prev[:n_in]
        D_back = t_post.unsqueeze(-1) - t_data.T.unsqueeze(0)
        K_back = _K(D_back, tm, ts, alpha, k_peak)
        Kd_back = _Kd(D_back, tm, ts, alpha, k_peak)
        grad[:, :n_in] = -(adj_sur.unsqueeze(-1) * K_back).sum(dim=1)
        lam_prev = (adj_sur.unsqueeze(-1) * W[:, :n_in].unsqueeze(1)
                    * Kd_back).sum(dim=0).T

        # forward(): W, t_prev, t_bias, theta, tm, ts, alpha, k_peak, grid,
        #            peak_tol, beta
        return (grad, lam_prev, None, None, None, None, None, None,
                None, None, None)


class SurrogateTTFSLinear(nn.Module):
    """Drop-in twin of `ExactTTFSLinear` with a surrogate (not exact) gradient.

    Owns an internal `ExactTTFSLinear` purely as the parameter/grid container,
    so construction, weight initialization and `calibrate_init_fire` are
    identical to the exact layer. Only the autograd rule differs.
    """

    def __init__(self, n_in: int, n_out: int, tm: float = 15.0, ts: float = 4.0,
                 theta: float = 1.0, t_max: float = 40.0, w_scale: float = 0.2,
                 bias_val: float = 1.5, grid_pts: int = 2001, seed: int = 0,
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 peak_tol: float = 1e-2, beta: Optional[float] = None) -> None:
        super().__init__()
        self.base = ExactTTFSLinear(
            n_in, n_out, tm=tm, ts=ts, theta=theta, t_max=t_max,
            w_scale=w_scale, bias_val=bias_val, grid_pts=grid_pts, seed=seed,
            dtype=dtype, device=device, peak_tol=peak_tol)
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.tm = float(tm)
        self.ts = float(ts)
        self.theta = float(theta)
        self.t_max = float(t_max)
        self.t_bias = 0.0
        self.peak_tol = float(peak_tol)
        self._alpha = self.base._alpha
        self.k_peak = self.base.k_peak
        self.beta = float(beta) if beta is not None else 8.0 / float(theta)

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def grid(self) -> torch.Tensor:
        return self.base.grid

    def calibrate_init_fire(self, target: float = 0.5, n_probe: int = 32,
                            cal_grid_pts: int = 65) -> None:
        self.base.calibrate_init_fire(target=target, n_probe=n_probe,
                                      cal_grid_pts=cal_grid_pts)

    def forward(self, t_prev: torch.Tensor) -> torch.Tensor:
        if t_prev.dim() != 2:
            raise ValueError(f"Expected (n_in, B) but got shape {tuple(t_prev.shape)}")
        if t_prev.shape[0] != self.n_in:
            raise ValueError(f"Input dim {t_prev.shape[0]} != n_in {self.n_in}")
        W = self.weight
        grid = self.grid.to(dtype=W.dtype, device=W.device)
        return SurrogateTTFSLinearFn.apply(
            W, t_prev, self.t_bias, self.theta, self.tm, self.ts,
            self._alpha, self.k_peak, grid, self.peak_tol, self.beta)


class SurrogateSpikingAttention(nn.Module):
    """Twin of `exact_snn.attention.ExactSpikingAttention`: same connectivity
    and the same EXACT analytic combine step; only the Q/K/V projection layers
    are trained with the surrogate (not IFT) gradient rule."""

    def __init__(self, n_in: int, n_out: int, tm: float = 15.0,
                 ts: float = 4.0, theta: float = 1.0, t_max: float = 40.0,
                 w_scale: float = 0.2, bias_val: float = 1.5,
                 grid_pts: int = 2001, seed: int = 0,
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 temp: float = 1.0, combine: str = "gaussian",
                 peak_tol: float = 1e-2, beta: Optional[float] = None) -> None:
        super().__init__()
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.tm = float(tm)
        self.ts = float(ts)
        self.theta = float(theta)
        self.t_max = float(t_max)
        self.temp = float(temp)
        self.combine = str(combine)
        self._alpha = abs(self.tm - self.ts) < 1e-9
        self.k_peak = ExactTTFSLinear._compute_k_peak(self.tm, self.ts)

        common = dict(tm=self.tm, ts=self.ts, theta=self.theta,
                      t_max=self.t_max, w_scale=float(w_scale),
                      bias_val=float(bias_val), grid_pts=int(grid_pts),
                      dtype=dtype, device=device, peak_tol=float(peak_tol),
                      beta=beta)
        self.WQ = SurrogateTTFSLinear(n_in, n_out, seed=seed, **common)
        self.WK = SurrogateTTFSLinear(n_in, n_out, seed=seed + 1000, **common)
        self.WV = SurrogateTTFSLinear(n_in, n_out, seed=seed + 2000, **common)

    def calibrate_init_fire(self, target: float = 0.5, n_probe: int = 32,
                            cal_grid_pts: int = 65) -> None:
        for p in (self.WQ, self.WK, self.WV):
            p.calibrate_init_fire(target=target, n_probe=n_probe,
                                  cal_grid_pts=cal_grid_pts)

    def forward(self, t_in: torch.Tensor) -> torch.Tensor:
        t_q = self.WQ(t_in)
        t_k = self.WK(t_in)
        t_v = self.WV(t_in)
        return ExactAttentionCombineFn.apply(
            t_q, t_k, t_v, self.tm, self.ts, self._alpha, self.k_peak,
            self.temp, self.t_max, self.combine)