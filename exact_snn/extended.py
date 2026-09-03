"""Extended Exact-SNN layers as drop-in torch.nn.Module components.

Adds the CONV, MULTI-SPIKE and RECURRENT layers on top of the base
`ExactTTFSLinear` (see exact_snn/__init__.py). All use the exact IFT /
saltation gradient math, wired into autograd so they work with
`torch.optim` and `loss.backward()`.

Contents:
    ExactTTFSConv2d   - convolutional TTFS layer (spike-time maps)
    ExactMultiSpike   - multi-spike layer with exact saltation gradients
    ExactRecurrent    - recurrent TTFS layer with eligibility traces
    multispike_latency_loss    - differentiable CE on first-spike times
    spike_count_cross_entropy  - differentiable soft spike-count CE
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from exact_snn import (
    _K,
    _Kd,
    _u_at,
    _du_at,
    _forward_layer_torch,
    _backward_layer_torch,
    _ExactTTFSLayerFn,
    ExactTTFSLinear,
    latency_cross_entropy,
    _validate_layer_config,
)

__all__ = [
    "ExactTTFSConv2d",
    "ExactMultiSpike",
    "ExactRecurrent",
    "multispike_latency_loss",
    "spike_count_cross_entropy",
]


# ===========================================================================
# Multi-spike (rate coding) forward/backward kernels (exact saltation).
# ===========================================================================
def _interp_grid(grid, vals, m):
    """Linear interpolation of a grid-sampled membrane `vals` (n_cur,B,G) at
    arbitrary times `m` (n_cur,B). Returns (n_cur,B).

    Used in the bisection phase of the spike-time solve: `vals` is exactly the
    membrane `u` sampled on the grid, so interpolating it avoids recomputing
    the expensive O(n_in) input-sum for every bisection step. Newton still
    refines the result exactly afterwards.
    """
    G = vals.shape[2]
    t0, t1 = grid[0], grid[-1]
    pos = (m.clamp(t0, t1) - t0) / (t1 - t0) * (G - 1)
    lo = pos.to(torch.long).clamp(0, G - 1)
    hi = (lo + 1).clamp(max=G - 1)
    frac = (pos - lo).clamp(0.0, 1.0)
    vlo = vals.gather(2, lo.unsqueeze(-1)).squeeze(-1)
    vhi = vals.gather(2, hi.unsqueeze(-1)).squeeze(-1)
    return vlo + frac * (vhi - vlo)


def _multispike_forward(W, t_prev, t_bias, tm, ts, theta, k_peak, t_max,
                        grid, max_spikes=20, n_bisect=15, n_newton=8):
    """GPU multi-spike forward (verified engine, autograd-friendly).

    Returns (t_post, up, t_all, up_all):
        t_post: (n_cur, B) first-spike times (inf if silent)
        up:     (n_cur, B) u' at first spike (0 if silent)
        t_all:  (n_cur, B, K) all spike times, padded with inf
        up_all: (n_cur, B, K) u' at each spike, 0 for padding
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    G = grid.numel()
    dev = W.device
    dtype = W.dtype
    b_inv = 1.0 / ts
    alpha = False

    g = grid.view(1, 1, -1)
    if n_in:
        # One matmul instead of looping over n_in inputs (the previous loop was
        # the dominant cost: G=n_in tensor-adds building the (n_cur,B,G) grid).
        K_grid = _K(g - t_prev.unsqueeze(-1), tm, ts, alpha, k_peak)  # (n_in,B,G)
        U_base = (W[:, :n_in] @ K_grid.reshape(n_in, -1)).reshape(n_cur, B, G)
    else:
        K_grid = None
        U_base = torch.zeros((n_cur, B, G), dtype=dtype, device=dev)
    U_base += W[:, n_in].view(n_cur, 1, 1) * _K(g - t_bias, tm, ts, alpha, k_peak)

    t_post = torch.full((n_cur, B), float("inf"), dtype=dtype, device=dev)
    up = torch.zeros((n_cur, B), dtype=dtype, device=dev)
    t_all = torch.full((n_cur, B, max_spikes), float("inf"), dtype=dtype, device=dev)
    up_all = torch.zeros((n_cur, B, max_spikes), dtype=dtype, device=dev)

    active = torch.ones((n_cur, B), dtype=torch.bool, device=dev)
    t_f_prev = torch.zeros((n_cur, B), dtype=dtype, device=dev)
    i_f_prev = torch.zeros((n_cur, B), dtype=dtype, device=dev)
    unconsumed = torch.ones((n_cur, B, n_in), dtype=torch.bool, device=dev) if n_in else None
    U = U_base.clone()

    for k in range(max_spikes):
        if not active.any():
            break

        mask = (U >= theta) & active.unsqueeze(-1)
        any_mask = mask.any(dim=2)
        if not any_mask.any():
            break

        idx = mask.long().argmax(dim=2)
        idxc = idx.clamp(min=1)
        a_br = grid[(idxc - 1).clamp(min=0)]
        b_br = grid[idxc.clamp(max=G - 1)]
        fa = U.gather(2, (idxc - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1) - theta
        fb = U.gather(2, idxc.unsqueeze(-1)).squeeze(-1) - theta
        at_first = (idx == 0) & any_mask

        for _ in range(n_bisect):
            m = 0.5 * (a_br + b_br)
            fm = _interp_grid(grid, U, m) - theta
            left = fa * fm <= 0.0
            b_br = torch.where(left, m, b_br)
            fb = torch.where(left, fm, fb)
            a_br = torch.where(left, a_br, m)
            fa = torch.where(left, fa, fm)

        m = 0.5 * (a_br + b_br)
        for _ in range(n_newton):
            if k == 0:
                um = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m) - theta
                dum = _du_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m)
            else:
                um = _u_at_ms(W, t_prev, tm, ts, k_peak, m,
                              t_f_prev, i_f_prev, unconsumed) - theta
                dum = _du_at_ms(W, t_prev, tm, ts, k_peak, m,
                                t_f_prev, i_f_prev, unconsumed)
            safe = dum > 1e-10
            nm = m - um / torch.where(safe, dum, torch.ones_like(dum))
            nm = nm.clamp(min=a_br, max=b_br)
            m = torch.where(safe, nm, m)

        tf = torch.where(at_first, grid[0], m)
        tf = torch.where(any_mask, tf, torch.full_like(tf, float("inf")))

        if k == 0:
            up_k = _du_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, tf)
        else:
            up_k = _du_at_ms(W, t_prev, tm, ts, k_peak, tf,
                             t_f_prev, i_f_prev, unconsumed)

        fired = any_mask & torch.isfinite(tf)
        t_all[:, :, k] = torch.where(fired, tf, t_all[:, :, k])
        up_all[:, :, k] = torch.where(fired, up_k, up_all[:, :, k])
        if k == 0:
            t_post = torch.where(fired, tf, t_post)
            up = torch.where(fired, up_k, up)

        if not fired.any():
            break

        if n_in:
            t_prev_bt = t_prev.t()
            consumed_now = t_prev_bt.unsqueeze(0) <= tf.unsqueeze(-1)
            dt_f = (tf.unsqueeze(-1) - t_prev_bt.unsqueeze(0)).clamp(min=0.0)
            exp_dt = torch.exp(-b_inv * dt_f)
            i_f_input = (W[:, :n_in].unsqueeze(1) * exp_dt *
                         consumed_now.float()).sum(dim=2)
        else:
            i_f_input = torch.zeros_like(tf)
        i_f_bias = W[:, n_in].unsqueeze(1) * torch.exp(-b_inv * tf)
        i_f_new = i_f_bias + i_f_input

        if n_in:
            new_consumed = t_prev_bt.unsqueeze(0) <= tf.unsqueeze(-1)
            unconsumed = torch.where(fired.unsqueeze(-1),
                                     unconsumed & ~new_consumed, unconsumed)

        if n_in:
            forced = torch.einsum('jbi,ibg->jbg',
                                  W[:, :n_in].unsqueeze(1) * unconsumed.float(),
                                  K_grid)
        else:
            forced = torch.zeros((n_cur, B, G), dtype=dtype, device=dev)
        U_work = i_f_new.unsqueeze(-1) * ts * k_peak * _K(
            g - tf.unsqueeze(-1), tm, ts, alpha, k_peak) + forced
        free = None

        U = torch.where(fired.unsqueeze(-1), U_work, U)
        t_f_prev = torch.where(fired, tf, t_f_prev)
        i_f_prev = torch.where(fired, i_f_new, i_f_prev)

    return t_post, up, t_all, up_all


def _u_at_ms(W, t_prev, tm, ts, k_peak, t, t_f_prev, i_f_prev, unconsumed):
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    alpha = False
    free = i_f_prev * ts * k_peak * _K(t - t_f_prev, tm, ts, alpha, k_peak)
    forced = torch.zeros_like(t)
    if n_in:
        D = t.unsqueeze(-1) - t_prev[:n_in].t().unsqueeze(0)
        K_val = _K(D, tm, ts, alpha, k_peak)
        forced = (W[:, :n_in].unsqueeze(1) * K_val *
                  unconsumed.float()).sum(dim=2)
    return free + forced


def _du_at_ms(W, t_prev, tm, ts, k_peak, t, t_f_prev, i_f_prev, unconsumed):
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    alpha = False
    free = i_f_prev * ts * k_peak * _Kd(t - t_f_prev, tm, ts, alpha, k_peak)
    forced = torch.zeros_like(t)
    if n_in:
        D = t.unsqueeze(-1) - t_prev[:n_in].t().unsqueeze(0)
        Kd_val = _Kd(D, tm, ts, alpha, k_peak)
        forced = (W[:, :n_in].unsqueeze(1) * Kd_val *
                  unconsumed.float()).sum(dim=2)
    return free + forced


def _multispike_backward(W, t_prev, t_bias, t_all, up_all, lam_post, tm, ts,
                         k_peak, t_max, theta, first_spike_only=False):
    """Exact saltation backward through all resets (vectorized).

    lam_post is dL/d(spike time), accumulated per neuron over spikes and
    weighted by the running saltation product inside. Returns (grad, lam_prev).
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_all.shape[1]
    K = t_all.shape[2]
    dev = W.device
    dtype = W.dtype

    grad = torch.zeros_like(W)
    lam_prev = torch.zeros((n_in, B), dtype=dtype, device=dev)

    u_reset = 0.0
    S = torch.ones((n_cur, B), dtype=dtype, device=dev)
    k_range = 1 if first_spike_only else K

    for k in range(k_range):
        t_fk = t_all[:, :, k]
        up_k = up_all[:, :, k]
        fired_k = torch.isfinite(t_fk) & (up_k.abs() > 1e-12)
        if not fired_k.any():
            continue

        i_f_k = up_k * tm + theta
        den_k = i_f_k - theta
        safe_den = den_k.abs() > 1e-12
        Xi_uu_k = torch.where(
            safe_den,
            (i_f_k - u_reset) / torch.where(safe_den, den_k, torch.ones_like(den_k)),
            torch.ones_like(den_k))

        effective_adj = lam_post * S
        up_safe = torch.where(fired_k & (up_k.abs() > 1e-12), up_k,
                              torch.ones_like(up_k))
        adj = torch.where(fired_k, effective_adj / up_safe,
                          torch.zeros_like(effective_adj))

        K_bias = _K(t_fk - t_bias, tm, ts, False, k_peak)
        grad[:, n_in] += -(adj * K_bias).sum(dim=1)

        if n_in:
            t_fk_exp = t_fk.unsqueeze(-1)
            t_in_exp = t_prev[:n_in].T.unsqueeze(0)
            D = t_fk_exp - t_in_exp
            K_all = _K(D, tm, ts, False, k_peak)
            grad[:, :n_in] += -(adj.unsqueeze(-1) * K_all).sum(dim=1)
            Kd_all = _Kd(D, tm, ts, False, k_peak)
            lam_prev += (adj.unsqueeze(-1) * W[:, :n_in].unsqueeze(1) *
                         Kd_all).sum(dim=0).T

        if not first_spike_only:
            S = torch.where(fired_k, S * Xi_uu_k, S)

    return grad, lam_prev


# ===========================================================================
# Multi-spike autograd Function + nn.Module
# ===========================================================================
class _ExactMultiSpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, t_prev, t_bias, theta, tm, ts, k_peak, t_max, grid,
                max_spikes, first_spike_only):
        t_post, up, t_all, up_all = _multispike_forward(
            W, t_prev, t_bias, tm, ts, theta, k_peak, t_max, grid,
            max_spikes=int(max_spikes))
        ctx.save_for_backward(W, t_prev, t_all, up_all)
        ctx.t_bias = float(t_bias)
        ctx.theta = float(theta)
        ctx.tm = float(tm)
        ctx.ts = float(ts)
        ctx.k_peak = float(k_peak)
        ctx.t_max = float(t_max)
        ctx.first_spike_only = bool(first_spike_only)
        return t_all

    @staticmethod
    def backward(ctx, grad_t_all):
        W, t_prev, t_all, up_all = ctx.saved_tensors
        # lam = dL/d(spike time) summed over spikes per neuron.
        lam = grad_t_all.sum(dim=2)
        grad, lam_prev = _multispike_backward(
            W, t_prev, ctx.t_bias, t_all, up_all, lam, ctx.tm, ctx.ts,
            ctx.k_peak, ctx.t_max, ctx.theta,
            first_spike_only=ctx.first_spike_only)
        return (grad, lam_prev, None, None, None, None, None, None,
                None, None, None)


class ExactMultiSpike(nn.Module):
    """A multi-spike (rate-coded) layer with exact saltation gradients.

    Forward returns the full spike train t_all of shape (n_out, B, K) so
    rate-coded losses (spike count) can be applied. Weight is an nn.Parameter
    of shape (n_out, n_in + 1).

    Args:
        first_spike_only: when True only the first spike participates in the
            backward (TTFS-style); when False all spikes contribute, each
            weighted by the running saltation product (rate/count style).
    """

    def __init__(self, n_in: int, n_out: int, tm: float = 15.0, ts: float = 4.0,
                 theta: float = 1.0, t_max: float = 40.0, w_scale: float = 0.2,
                 bias_val: float = 1.2, grid_pts: int = 2001, seed: int = 0,
                 max_spikes: int = 20, dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 first_spike_only: bool = False) -> None:
        super().__init__()
        _validate_layer_config(n_in, n_out, tm, ts, theta, t_max, grid_pts)
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.tm = float(tm)
        self.ts = float(ts)
        self.theta = float(theta)
        self.t_max = float(t_max)
        self.t_bias = 0.0
        self.max_spikes = int(max_spikes)
        self.first_spike_only = bool(first_spike_only)
        self.k_peak = ExactTTFSLinear._compute_k_peak(self.tm, self.ts)

        rng = np.random.default_rng(seed)
        w = (rng.standard_normal((n_out, n_in + 1)) * w_scale).astype(np.float64)
        w[:, -1] = bias_val
        self.weight = nn.Parameter(torch.tensor(w, dtype=dtype, device=device))
        self.register_buffer("grid", torch.linspace(
            0.0, t_max, int(grid_pts), dtype=self.weight.dtype,
            device=self.weight.device))

    def extra_repr(self) -> str:
        return (f"n_in={self.n_in}, n_out={self.n_out}, K={self.max_spikes}, "
                f"first_spike_only={self.first_spike_only}")

    def forward(self, t_prev: torch.Tensor) -> torch.Tensor:
        """t_prev: (n_in, B) -> t_all: (n_out, B, K)."""
        if t_prev.shape[0] != self.n_in:
            raise ValueError(f"Input dim {t_prev.shape[0]} != n_in {self.n_in}")
        if not t_prev.is_floating_point():
            raise ValueError("Input spike times must use a floating-point dtype")
        if t_prev.dtype != self.weight.dtype:
            raise ValueError(f"Input dtype {t_prev.dtype} != layer dtype {self.weight.dtype}")
        if t_prev.device != self.weight.device:
            raise ValueError(f"Input device {t_prev.device} != layer device {self.weight.device}")
        W = self.weight
        grid = self.grid.to(dtype=W.dtype, device=W.device)
        return _ExactMultiSpikeFn.apply(
            W, t_prev, self.t_bias, self.theta, self.tm, self.ts, self.k_peak,
            self.t_max, grid, self.max_spikes, self.first_spike_only)


# ===========================================================================
# Convolutional TTFS layer
# ===========================================================================
class _ExactTTFSConvFn(torch.autograd.Function):
    """2D conv TTFS as an autograd function (unfold -> IFT -> fold)."""

    @staticmethod
    def forward(ctx, W, t_in, t_bias, theta, tm, ts, alpha, k_peak, grid,
                t_max, stride, padding, kernel, out_c, H_in, W_in, peak_tol):
        B, C, H, Wimg = t_in.shape
        kh = kw = kernel
        H_out = (H_in + 2 * padding - kh) // stride + 1
        W_out = (W_in + 2 * padding - kw) // stride + 1
        t_pad = torch.full((B, C, H_in + 2 * padding, W_in + 2 * padding),
                           t_max, dtype=t_in.dtype, device=t_in.device)
        t_pad[:, :, padding:padding + H_in, padding:padding + W_in] = t_in
        pat = F.unfold(t_pad, kernel_size=(kh, kw), stride=stride)
        n_in = C * kh * kw
        L = pat.shape[2]
        pat = pat.permute(0, 2, 1).reshape(B * L, n_in).t()
        bias = torch.zeros(1, B * L, dtype=t_in.dtype, device=t_in.device)
        pat = torch.cat([pat, bias], dim=0)

        t_post, up = _forward_layer_torch(
            W, pat, t_bias, theta, grid, tm, ts, alpha, k_peak,
            peak_tol=float(peak_tol))
        t_feat = t_post.reshape(out_c, B, H_out, W_out).permute(1, 0, 2, 3)

        ctx.save_for_backward(W, pat, t_post, up)
        ctx.t_bias = float(t_bias)
        ctx.theta = float(theta)
        ctx.tm = float(tm)
        ctx.ts = float(ts)
        ctx.alpha = bool(alpha)
        ctx.k_peak = float(k_peak)
        ctx.stride = int(stride)
        ctx.padding = int(padding)
        ctx.kernel = int(kernel)
        ctx.out_c = int(out_c)
        ctx.H_in = int(H_in)
        ctx.W_in = int(W_in)
        ctx.C = int(C)
        ctx.B = int(B)
        ctx.H_out = int(H_out)
        ctx.W_out = int(W_out)
        ctx.peak_tol = float(peak_tol)
        return t_feat

    @staticmethod
    def backward(ctx, grad_input):
        W, pat, t_post, up = ctx.saved_tensors
        gl = grad_input.permute(1, 0, 2, 3).reshape(
            ctx.out_c, ctx.B * ctx.H_out * ctx.W_out)
        grad, lam_prev_patches = _backward_layer_torch(
            W, pat, ctx.t_bias, t_post, gl, up, ctx.tm, ctx.ts, ctx.alpha,
            ctx.k_peak)
        n_in = ctx.C * ctx.kernel * ctx.kernel
        kh = kw = ctx.kernel
        s = ctx.stride
        p = ctx.padding
        lam_p = lam_prev_patches[:n_in]                      # (C*kh*kw, B*L)
        lam_chn = []
        for c in range(ctx.C):
            ch_rows = lam_p[c * kh * kw:(c + 1) * kh * kw]   # (kh*kw, B*L)
            lam_chn.append(F.fold(
                ch_rows.reshape(ctx.B, kh * kw, ctx.H_out * ctx.W_out),
                output_size=(ctx.H_in + 2 * p, ctx.W_in + 2 * p),
                kernel_size=(kh, kw), stride=s).squeeze(1))
        lam_full = torch.stack(lam_chn, dim=1)
        lam_crop = lam_full[:, :, p:p + ctx.H_in, p:p + ctx.W_in]
        return (grad, lam_crop, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None)


class ExactTTFSConv2d(nn.Module):
    """A 2D convolutional TTFS layer as an nn.Module.

    Operates on spike-time maps: input (B, C, H, W) of spike times ->
    output (B, out_channels, H_out, W_out). Weight is an nn.Parameter of shape
    (out_channels, C*kh*kw + 1).
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, tm: float = 15.0,
                 ts: float = 4.0, theta: float = 1.0, t_max: float = 40.0,
                 w_scale: float = 0.2, bias_val: float = 0.2,
                 grid_pts: int = 501, seed: int = 0,
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 peak_tol: float = 1e-2) -> None:
        super().__init__()
        _validate_layer_config(in_channels * kernel_size * kernel_size,
                               out_channels, tm, ts, theta, t_max, grid_pts)
        if kernel_size <= 0 or stride <= 0 or padding < 0:
            raise ValueError("kernel_size and stride must be positive; padding cannot be negative")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.tm = float(tm)
        self.ts = float(ts)
        self.theta = float(theta)
        self.t_max = float(t_max)
        self.t_bias = 0.0
        self.peak_tol = float(peak_tol)
        self._alpha = abs(self.tm - self.ts) < 1e-9
        self.k_peak = ExactTTFSLinear._compute_k_peak(self.tm, self.ts)

        fan_in = in_channels * kernel_size * kernel_size
        rng = np.random.default_rng(seed)
        w = (rng.standard_normal((out_channels, fan_in + 1)) * w_scale).astype(np.float64)
        w[:, -1] = bias_val
        self.weight = nn.Parameter(torch.tensor(w, dtype=dtype, device=device))
        self.register_buffer("grid", torch.linspace(
            0.0, t_max, int(grid_pts), dtype=self.weight.dtype,
            device=self.weight.device))

    def extra_repr(self) -> str:
        return (f"in={self.in_channels}, out={self.out_channels}, "
                f"k={self.kernel_size}, s={self.stride}, p={self.padding}")

    def forward(self, t_in: torch.Tensor) -> torch.Tensor:
        if t_in.dim() != 4:
            raise ValueError(f"Expected (B, C, H, W) but got {tuple(t_in.shape)}")
        B, C, H, Wimg = t_in.shape
        if C != self.in_channels:
            raise ValueError(f"Input channels {C} != {self.in_channels}")
        Wt = self.weight
        if not t_in.is_floating_point():
            raise ValueError("Input spike times must use a floating-point dtype")
        if t_in.dtype != Wt.dtype:
            raise ValueError(f"Input dtype {t_in.dtype} != layer dtype {Wt.dtype}")
        if t_in.device != Wt.device:
            raise ValueError(f"Input device {t_in.device} != layer device {Wt.device}")
        grid = self.grid.to(dtype=Wt.dtype, device=Wt.device)
        return _ExactTTFSConvFn.apply(
            Wt, t_in, self.t_bias, self.theta, self.tm, self.ts, self._alpha,
            self.k_peak, grid, self.t_max, self.stride, self.padding,
            self.kernel_size, self.out_channels, H, Wimg, self.peak_tol)


# ===========================================================================
# Recurrent TTFS layer (shared feedback trace)
# ===========================================================================
class ExactRecurrent(nn.Module):
    """A recurrent TTFS layer with an eligibility trace (shared feedback).

    Each output neuron maintains a per-neuron eligibility trace e_j(t) that
    decays since its last spike. The feedback that re-enters the IFT layer is
    a single SHARED row ``trace.mean(0)`` per sample, giving each neuron a
    per-neuron recurrent weight column while keeping exact IFT gradients.

    Weight is an nn.Parameter of shape (n_out, n_in + 2): n_in synapse
    columns, one recurrent (feedback) column, one bias column.
    """

    def __init__(self, n_in: int, n_out: int, tm: float = 15.0, ts: float = 4.0,
                 theta: float = 1.0, t_max: float = 40.0, w_scale: float = 0.15,
                 bias_val: float = 0.1, grid_pts: int = 501, seed: int = 0,
                 tau_rec: float = 5.0, dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 peak_tol: float = 1e-2) -> None:
        super().__init__()
        _validate_layer_config(n_in, n_out, tm, ts, theta, t_max, grid_pts)
        if tau_rec <= 0:
            raise ValueError("tau_rec must be positive")
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")
        self.n_in = int(n_in)
        self.n_out = int(n_out)
        self.tm = float(tm)
        self.ts = float(ts)
        self.theta = float(theta)
        self.t_max = float(t_max)
        self.tau_rec = float(tau_rec)
        self.t_bias = 0.0
        self.peak_tol = float(peak_tol)
        self._alpha = abs(self.tm - self.ts) < 1e-9
        self.k_peak = ExactTTFSLinear._compute_k_peak(self.tm, self.ts)

        rng = np.random.default_rng(seed)
        w = (rng.standard_normal((n_out, n_in + 2)) * w_scale).astype(np.float64)
        w[:, -2] = 0.1          # recurrent (feedback) weight column
        w[:, -1] = bias_val     # bias column
        self.weight = nn.Parameter(torch.tensor(w, dtype=dtype, device=device))
        self.register_buffer("grid", torch.linspace(
            0.0, t_max, int(grid_pts), dtype=self.weight.dtype,
            device=self.weight.device))
        self.register_buffer("_trace", torch.zeros(n_out, 1, dtype=dtype,
                                                   device=device))
        self.register_buffer("_last_spike", torch.full(
            (n_out, 1), float("inf"), dtype=dtype, device=device))

    def reset_state(self, B: int = 1) -> None:
        self._trace = torch.zeros(self.n_out, B, dtype=self.weight.dtype,
                                  device=self.weight.device)
        self._last_spike = torch.full((self.n_out, B), float("inf"),
                                      dtype=self.weight.dtype,
                                      device=self.weight.device)

    def forward_step(self, t_in: torch.Tensor) -> torch.Tensor:
        """One recurrent step. t_in: (n_in, B) -> (n_out, B) spike times.

        The shared feedback trace is carried as a (detached) extra IFT input
        row so the recurrent weight column and all synapse columns receive
        exact IFT gradients.
        """
        B = t_in.shape[1]
        if t_in.dim() != 2 or t_in.shape[0] != self.n_in:
            raise ValueError(f"Expected ({self.n_in}, B) input but got {tuple(t_in.shape)}")
        if not t_in.is_floating_point():
            raise ValueError("Input spike times must use a floating-point dtype")
        if t_in.dtype != self.weight.dtype:
            raise ValueError(f"Input dtype {t_in.dtype} != layer dtype {self.weight.dtype}")
        if t_in.device != self.weight.device:
            raise ValueError(f"Input device {t_in.device} != layer device {self.weight.device}")
        if self._trace.shape[1] != B:
            self.reset_state(B)
        tr = self._trace.detach()
        trace_row = tr.mean(dim=0, keepdim=True)          # (1, B) shared
        t_with = torch.cat([t_in, trace_row], dim=0)      # (n_in+1, B)
        W = self.weight
        grid = self.grid.to(dtype=W.dtype, device=W.device)
        t_post = _ExactTTFSLayerFn.apply(
            W, t_with, self.t_bias, self.theta, self.tm, self.ts, self._alpha,
            self.k_peak, grid, self.peak_tol)
        fired = torch.isfinite(t_post)
        last = self._last_spike.detach()
        # If this neuron has never fired (last = inf), treat dt as t_max so the
        # old trace fully decays (~0) before we add the new event; otherwise
        # dt = t_post - last would be -inf -> exp(+inf) -> NaN.
        dt_since = torch.where(
            fired & torch.isfinite(last), t_post - last,
            torch.full_like(t_post, self.t_max))
        decay = torch.exp(-dt_since / self.tau_rec)
        self._trace = torch.where(fired, decay * self._trace + 1.0,
                                  decay * self._trace)
        self._last_spike = torch.where(fired, t_post.detach(), self._last_spike)
        return t_post


# ===========================================================================
# Multi-spike / rate losses (autograd-compatible)
# ===========================================================================
def multispike_latency_loss(t_all: torch.Tensor, y: torch.Tensor,
                            t_max: float, beta: float = 1.0) -> torch.Tensor:
    """Latency CE on first-spike times of a multi-spike output.

    t_all: (n_out, B, K). Uses t_all[:, :, 0] which is fully differentiable.
    """
    t_first = t_all[:, :, 0]
    return latency_cross_entropy(t_first, y, t_max, beta)


def spike_count_cross_entropy(t_all: torch.Tensor, y: torch.Tensor,
                              t_max: float, beta: float = 1.0,
                              temp: float = 2.0) -> torch.Tensor:
    """Differentiable spike-count CE.

    t_all: (n_out, B, K). A soft spike count for neuron j is
        count_j = sum_k sigmoid(-(t_jk - t_max)/temp)
    which is 1 for an early spike, ~0.5 at the window edge and 0 for a silent
    (inf) slot, and is smoothly differentiable in each spike time. CE is applied
    on these soft counts, so gradient flows back through t_all exactly.
    """
    B = t_all.shape[1]
    t = torch.where(torch.isfinite(t_all), t_all,
                    torch.full_like(t_all, 2.0 * t_max))
    counts = torch.sigmoid(-(t - t_max) / temp).sum(dim=2)
    counts = counts - counts.mean(dim=0, keepdim=True)
    logits = beta * counts
    logits = logits - logits.max(dim=0, keepdim=True).values
    p = torch.exp(logits)
    p = p / p.sum(dim=0, keepdim=True)
    return -torch.log(p[y, torch.arange(B, device=t_all.device)] + 1e-12).mean()
