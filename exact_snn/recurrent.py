"""SP-04: exact recurrent TTFS cell (NBTT through the spike-time rule).

    t_h[k] = TTFS([ t_in[k] ; t_h[k-1] ] ; W),    t_h[0] = silent (no spikes)

at every step k of a sequence of length T. The forward is the same exact
first-spike root solve the feed-forward layers use; the backward is the same
exact IFT rule, so the recurrence trains with the library's surrogate-free
gradient -- backprop-through-time where every step's output spike times are
differentiable in the previous step's spike times through the IFT Jacobian,
and in W through the IFT weight gradient. No surrogate, no reset
approximation; the recurrent connections are ordinary TTFS synapses that
re-enter the membrane equation on the following step.

Silent hidden neurons carry `inf`, exactly like a feed-forward silent output:
their `K(t - inf) = 0` contribution means a silent cell takes no part in the
next step's membrane (this is why `_K` clamps its argument to d >= 0), and no
autograd path is created for them -- consistent with the library's silent ==
zero-gradient convention.

Public API:
    ExactTTFSRnn            - the nn.Module recurrent cell (unrolls T steps)
    ExactTTFSRnn.forward_step(t_in, t_prev)  - a single recurrent step

Optional companion module (lazy opt-in):
    from exact_snn.recurrent import ExactTTFSRnn      # or re-exported from
    from exact_snn.extended import ExactTTFSRnn       # exact_snn.extended
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from exact_snn import (
    _backward_layer_torch,
    _forward_layer_torch,
    _validate_layer_config,
    ExactTTFSLinear,
)
from exact_snn._validation import validate_spike_times

__all__ = ["ExactTTFSRnn"]


# ---------------------------------------------------------------------------
# One exact recurrent step (autograd.Function).
# ---------------------------------------------------------------------------
class _ExactTTFSRnnStepFn(torch.autograd.Function):
    """One step of the recurrent cell.

    forward: cats the current input times with the previous hidden times and
    runs the exact first-spike solve with the full weight tensor.
    backward: exact IFT: returns the weight gradient, the sensitivity of the
    loss to the current input times, and the sensitivity to the previous
    hidden times (the recurrence Jacobian).
    """

    @staticmethod
    def forward(ctx, W, t_in, t_prev, t_bias, theta, tm, ts, alpha, k_peak,
                grid, peak_tol):
        t_all = torch.cat([t_in, t_prev], dim=0)
        use_cuda = (
            W.is_cuda
            and W.dtype == torch.float32
            and t_all.is_cuda
            and t_all.dtype == torch.float32
        )
        if use_cuda:
            from exact_snn import cuda_ops
            use_cuda = cuda_ops.is_enabled()
        if use_cuda:
            from exact_snn import cuda_ops
            t_post, up = cuda_ops.cuda_forward(
                W, t_all, t_bias, theta, grid, tm, ts, alpha, k_peak,
                n_bisect=15, n_newton=8, peak_tol=float(peak_tol))
        else:
            t_post, up = _forward_layer_torch(
                W, t_all, t_bias, theta, grid, tm, ts, alpha, k_peak,
                peak_tol=float(peak_tol))
        ctx.save_for_backward(W, t_in, t_prev, t_post, up)
        ctx.t_bias = float(t_bias)
        ctx.tm = float(tm)
        ctx.ts = float(ts)
        ctx.alpha = bool(alpha)
        ctx.k_peak = float(k_peak)
        return t_post

    @staticmethod
    def backward(ctx, grad_output):
        W, t_in, t_prev, t_post, up = ctx.saved_tensors
        t_all = torch.cat([t_in, t_prev], dim=0)
        grad, lam_all = _backward_layer_torch(
            W, t_all, ctx.t_bias, t_post, grad_output, up,
            ctx.tm, ctx.ts, ctx.alpha, ctx.k_peak)
        n_in = t_in.shape[0]
        # forward args: W, t_in, t_prev, t_bias, theta, tm, ts, alpha, k_peak, grid, peak_tol
        return (grad, lam_all[:n_in], lam_all[n_in:],
                None, None, None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# Public recurrent cell module.
# ---------------------------------------------------------------------------
class ExactTTFSRnn(nn.Module):
    """A recurrent TTFS cell with FULL per-neuron feedback and exact NBTT.

    Applies, for each step k of an input sequence,

        t_h[k] = TTFS([ t_in[k] ; t_h[k-1] ] ; W)

    where ``TTFS`` is the exact first-spike root solve and ``W`` is a single
    shared weight tensor of shape ``(n_hidden, n_in + n_hidden + 1)``: ``n_in``
    input columns, ``n_hidden`` recurrent columns, and one bias column in the
    last position. The exact IFT gives the layer's weight gradient and, via the
    input-time sensitivity, the Jacobian of t_h[k] in t_h[k-1], so training is
    exact backprop-through-time of the spike-time rule (no surrogate, no
    truncated gradient).

    State convention: the hidden state at step 0 is "silent" -- every hidden
    neuron holds ``inf`` and contributes ``K(t - inf) = 0`` to the next step's
    membrane. A hidden neuron that fails to fire at step k is ``inf`` and
    therefore takes no part in step k+1, exactly mirroring feed-forward silent
    semantics (silent => zero gradient; use ``exact_snn.existence`` if you want
    to revive dead hidden cells).

    Args:
        n_in: number of input channels per step.
        n_hidden: number of recurrent cells (output dimension).
        tm, ts, theta, t_max: TTFS dynamics params (see ExactTTFSLinear).
        w_scale: input-synapse weight scale.
        w_scale_h: recurrent-synapse weight scale. Defaults to ``w_scale``;
            smaller values stabilize the recurrence.
        bias_val: initial value of the bias column.
        grid_pts: resolution of the solve grid.
        seed, dtype, device, peak_tol: as ExactTTFSLinear.
    """

    def __init__(self, n_in: int, n_hidden: int, tm: float = 15.0,
                 ts: float = 4.0, theta: float = 1.0, t_max: float = 40.0,
                 w_scale: float = 0.2, w_scale_h: Optional[float] = None,
                 bias_val: float = 1.5, grid_pts: int = 2001, seed: int = 0,
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 peak_tol: float = 1e-2) -> None:
        super().__init__()
        _validate_layer_config(n_in + n_hidden, n_hidden, tm, ts, theta,
                               t_max, grid_pts)
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")
        w_scale_h = w_scale if w_scale_h is None else float(w_scale_h)
        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.tm = float(tm)
        self.ts = float(ts)
        self.theta = float(theta)
        self.t_max = float(t_max)
        self.t_bias = 0.0
        self.peak_tol = float(peak_tol)
        self._alpha = abs(self.tm - self.ts) < 1e-9
        self.k_peak = ExactTTFSLinear._compute_k_peak(self.tm, self.ts)

        rng = np.random.default_rng(seed)
        w = (rng.standard_normal((n_hidden, n_in + n_hidden + 1))
             * w_scale).astype(np.float64)
        w[:, n_in:n_in + n_hidden] *= (w_scale_h / w_scale)
        w[:, -1] = bias_val
        self.weight = nn.Parameter(torch.tensor(w, dtype=dtype, device=device))

        self.register_buffer("grid", torch.linspace(
            0.0, t_max, int(grid_pts), dtype=self.weight.dtype,
            device=self.weight.device))

    def extra_repr(self) -> str:
        return (f"n_in={self.n_in}, n_hidden={self.n_hidden}, "
                f"tm={self.tm:.1f}, ts={self.ts:.1f}, theta={self.theta}")

    def forward_step(self, t_in: torch.Tensor,
                     t_prev: torch.Tensor) -> torch.Tensor:
        """One recurrent step: (n_in, B) + (n_hidden, B) -> (n_hidden, B).

        ``t_prev`` carries the previous hidden spike times; ``inf`` entries
        (silent cells) contribute nothing.
        """
        if t_in.dim() != 2 or t_in.shape[0] != self.n_in:
            raise ValueError(
                f"Expected ({self.n_in}, B) input but got {tuple(t_in.shape)}")
        if t_prev.dim() != 2 or t_prev.shape[0] != self.n_hidden:
            raise ValueError(
                f"Expected ({self.n_hidden}, B) state but got {tuple(t_prev.shape)}")
        if t_in.shape[1] != t_prev.shape[1]:
            raise ValueError("input and state batch dims differ")
        if not t_in.is_floating_point() or not t_prev.is_floating_point():
            raise ValueError("Input spike times must use a floating-point dtype")
        validate_spike_times(t_in)
        validate_spike_times(t_prev)
        if t_in.dtype != self.weight.dtype or t_prev.dtype != self.weight.dtype:
            raise ValueError(f"Input dtype {t_in.dtype} != layer dtype {self.weight.dtype}")
        if t_in.device != self.weight.device or t_prev.device != self.weight.device:
            raise ValueError(f"Input device {t_in.device} != layer device {self.weight.device}")
        W = self.weight
        grid = self.grid.to(dtype=W.dtype, device=W.device)
        return _ExactTTFSRnnStepFn.apply(
            W, t_in, t_prev, self.t_bias, self.theta, self.tm, self.ts,
            self._alpha, self.k_peak, grid, self.peak_tol)

    def forward(self, t_in: torch.Tensor,
                h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Unroll T recurrent steps.

        Args:
            t_in: input sequence (n_in, B, T).
            h0: optional initial hidden state (n_hidden, B); defaults to an
                all-silent (inf) cold start that contributes nothing.

        Returns:
            Hidden spike-time sequence (n_hidden, B, T); ``inf`` marks a
            cell that did not fire at that step.
        """
        if t_in.dim() != 3 or t_in.shape[0] != self.n_in:
            raise ValueError(
                f"Expected ({self.n_in}, B, T) sequence but got {tuple(t_in.shape)}")
        n_in, B, T = t_in.shape
        if h0 is None:
            h0 = torch.full((self.n_hidden, B), float("inf"),
                            dtype=self.weight.dtype, device=self.weight.device)
        else:
            if h0.shape != (self.n_hidden, B):
                raise ValueError(
                    f"Expected ({self.n_hidden}, {B}) h0 but got {tuple(h0.shape)}")
            if h0.dtype != self.weight.dtype or h0.device != self.weight.device:
                h0 = h0.to(dtype=self.weight.dtype, device=self.weight.device)
        outs = []
        h = h0
        for k in range(int(T)):
            h = self.forward_step(t_in[:, :, k], h)
            outs.append(h)
        return torch.stack(outs, dim=2)