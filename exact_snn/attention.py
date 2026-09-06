"""SP-04: exact spiking attention (single-head) as a drop-in torch.nn module.

Motivation
----------
Classical attention mixes queries, keys and values. In an exact-SNN we keep
every signal as a *spike time* (TTFS coding) so the whole attention block stays
consumable by the existing exact-gradient layers (`ExactTTFSLinear`, etc.) and
the gradients stay exact in the same closed-form + Newton sense the library
already guarantees -- no surrogate gradients anywhere.

The block composes three existing `ExactTTFSLinear` layers (one each for Q, K,
V) whose first-spike maps are IFT-exact, with a *stateless* exact combine step.
The combine step is a pure, analytic function of the Q/K/V spike times:

    score[i, j] = alignment(t_Q[i] - t_K[j])      # temporal-alignment similarity
    a[i, j]     = softmax_t(score[i, :])_j        # time-coded softmax over keys
    out[i]      = sum_j a[i, j] * t_V[j]          # attended value spike time

`out` is a real-valued spike time, so a downstream `ExactTTFSLinear` /
`ExactTTFSConv2d` can feed directly on it, and the whole block is wired into
autograd through a custom `autograd.Function`.

Alignment map (default "gaussian", the recommended similarity):
    score[i,j] = exp( -d_ij^2 / (2 ts^2) ),    d_ij = t_Q[i] - t_K[j]

A Gaussian with the synaptic timescale `ts`. Smooth, symmetric, peaks at zero
(exactly the "aligned" case), decays with temporal distance, and its gradient
is exact and closed-form:
    ds/dt_Q = -s * d / ts^2,     ds/dt_K = +s * d / ts^2.

The "kernel" mode uses the library's raw synaptic kernel symmetrically:
    score[i,j] = K(|d_ij|) / temp
which is the literal "synaptic kernel K(t_Q - t_K)" option. Note K is causal
and peaks at |d| = tm (not at 0), so the gaussian alignment is the more
faithful "same time -> similar" map and is therefore the default. Both are
exact and differentiable.

Gradients through the combine are exact analytic partials that chain into the
Q/K/V layers' IFT `lam_prev` adjoints below in the usual way.

Silent neurons (inf spike times) carry no gradient, exactly like the rest of
the library. They are replaced by a finite placeholder for the softmax *only*
(so the forward value is well-defined and stable), and their gradient is
explicitly zeroed in the backward -- silent q/k/v can be revived later with the
optional existence channel (`exact_snn.existence`).

Public API
----------
    from exact_snn.attention import ExactSpikingAttention

    attn = ExactSpikingAttention(n_in, n_out, tm=15.0, ts=4.0,
                                 theta=1.0, t_max=40.0, w_scale=0.2,
                                 bias_val=1.5, grid_pts=2001, seed=0,
                                 temp=1.0, combine="gaussian", ...)
    t_out = attn(t_in)              # (n_in, B) -> (n_out, B) attended times
    t_out = attn(t_seq)             # (S, n_in, B) -> (S, n_out, B) sequence attention
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from exact_snn import (
    _K,
    _Kd,
    ExactTTFSLinear,
    _validate_layer_config,
)
from exact_snn._validation import validate_spike_times

__all__ = [
    "ExactSpikingAttention",
    "ExactAttentionCombineFn",
    "exact_attention_scores",
]


# ---------------------------------------------------------------------------
# Stateless combine step: scores -> softmax -> attended value time.
# All of it is a pure function of Q/K/V spike times with exact analytic
# partials, so it is wrapped in a single autograd.Function.
# ---------------------------------------------------------------------------
def _align(d, tm, ts, alpha, k_peak, combine):
    """Temporal-alignment similarity for time difference d (nq, nk, B)."""
    if combine == "gaussian":
        return torch.exp(-(d * d) / (2.0 * ts * ts))
    # combine == "kernel": symmetric raw synaptic kernel
    return _K(d.abs(), tm, ts, alpha, k_peak)


def _align_grad(d, tm, ts, alpha, k_peak, combine):
    """Gradient of the alignment score w.r.t. d. Extremum d=0 -> grad 0."""
    if combine == "gaussian":
        s = torch.exp(-(d * d) / (2.0 * ts * ts))
        return -s * (d / (ts * ts))
    kd = _Kd(d.abs(), tm, ts, alpha, k_peak)
    return kd * torch.sign(d)


def _combine_forward(t_q, t_k, t_v, tm, ts, alpha, k_peak, temp, t_max, combine):
    """Forward value spikes `out = sum_j a_ij t_V_j` given Q/K/V times.

    Silent (inf) entries are handled with a finite placeholder for the softmax
    and are masked to zero in the backward.

    Returns (out, score, attn) where
        out:   (n_out, B) attended value spike time (finite)
        score: (n_out, n_out, B) alignment similarity per (query, key, sample)
        attn:  (n_out, n_out, B) time-coded softmax over keys
    """
    d = t_q.unsqueeze(1) - t_k.unsqueeze(0)              # (nq, nk, B)
    finite = torch.isfinite(d)
    d_safe = torch.where(finite, d, torch.zeros_like(d))
    score = _align(d_safe, tm, ts, alpha, k_peak, combine)
    # stable softmax over keys (dim=1)
    score = score / max(float(temp), 1e-12)
    s_shift = score - score.max(dim=1, keepdim=True).values
    p = torch.exp(s_shift)
    p = p / p.sum(dim=1, keepdim=True).clamp(min=1e-12)
    attn = p                                             # (nq, nk, B)
    # attended value: placeholder for silent values so the sum stays finite;
    # silent values carry zero gradient anyway.
    t_v_safe = torch.where(torch.isfinite(t_v), t_v,
                           torch.full_like(t_v, 2.0 * t_max))
    out = (attn * t_v_safe.unsqueeze(0)).sum(dim=1)      # (nq, B)
    return out, score, attn


class ExactAttentionCombineFn(torch.autograd.Function):
    """Exact analytic backward for the spiking-attention combine step."""

    @staticmethod
    def forward(ctx, t_q, t_k, t_v, tm, ts, alpha, k_peak, temp, t_max, combine):
        out, score, attn = _combine_forward(
            t_q, t_k, t_v, tm, ts, alpha, k_peak, temp, t_max, combine)
        ctx.save_for_backward(t_q, t_k, t_v, attn)
        ctx.tm = float(tm)
        ctx.ts = float(ts)
        ctx.alpha = bool(alpha)
        ctx.k_peak = float(k_peak)
        ctx.temp = float(temp)
        ctx.t_max = float(t_max)
        ctx.combine = str(combine)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        t_q, t_k, t_v, attn = ctx.saved_tensors
        tm, ts, alpha, k_peak, temp, t_max, combine = (
            ctx.tm, ctx.ts, ctx.alpha, ctx.k_peak, ctx.temp, ctx.t_max,
            ctx.combine)
        nq, nk, B = attn.shape

        d = t_q.unsqueeze(1) - t_k.unsqueeze(0)          # (nq, nk, B)
        finite = torch.isfinite(d)
        d_safe = torch.where(finite, d, torch.zeros_like(d))
        gd = _align_grad(d_safe, tm, ts, alpha, k_peak, combine)  # ds/d(d)
        # ds/d(t_q) = gd, ds/d(t_k) = -gd
        ds_dtq = gd
        ds_dtk = -gd

        # gradient of out w.r.t. attn and t_v
        t_v_safe = torch.where(torch.isfinite(t_v), t_v,
                               torch.full_like(t_v, 2.0 * t_max))
        dout_dattn = t_v_safe.unsqueeze(0)               # (nq, nk, B)
        dout_dattn_g = grad_out.unsqueeze(1) * dout_dattn  # (nq, nk, B)

        # softmax Jacobian: da_ij = a_ij ( ds_ij - sum_k a_ik ds_ik ) / temp
        sum_a_ds = (attn * dout_dattn_g).sum(dim=1, keepdim=True)
        grad_attn = attn * (dout_dattn_g - sum_a_ds)
        g_ds = grad_attn / max(float(temp), 1e-12)

        g_tq = (g_ds * ds_dtq).sum(dim=1)                # (nq, B)
        g_tk = (g_ds * ds_dtk).sum(dim=0)                # (nk, B)
        g_tv = attn.sum(dim=0) * grad_out                # (nk, B)

        # silent Q/K/V entries carry no gradient
        q_finite = finite.all(dim=1)                     # query row all-finite
        k_finite = finite.all(dim=0)
        v_finite = torch.isfinite(t_v)
        g_tq = torch.where(q_finite, g_tq, torch.zeros_like(g_tq))
        g_tk = torch.where(k_finite, g_tk, torch.zeros_like(g_tk))
        g_tv = torch.where(v_finite, g_tv, torch.zeros_like(g_tv))

        return (g_tq, g_tk, g_tv, None, None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# Public scores helper (for inspection / custom losses).
# ---------------------------------------------------------------------------
def exact_attention_scores(t_q: torch.Tensor, t_k: torch.Tensor,
                           tm: float, ts: float, alpha: bool, k_peak: float,
                           temp: float = 1.0,
                           combine: str = "gaussian") -> torch.Tensor:
    """Alignment attention scores (nq, nk, B)."""
    d = t_q.unsqueeze(1) - t_k.unsqueeze(0)
    d_safe = torch.where(torch.isfinite(d), d, torch.zeros_like(d))
    s = _align(d_safe, tm, ts, alpha, k_peak, combine)
    return s / max(float(temp), 1e-12)


# ---------------------------------------------------------------------------
# nn.Module attention block.
# ---------------------------------------------------------------------------
class ExactSpikingAttention(nn.Module):
    """Spiking self-attention with exact closed-form gradients.

    Input `(n_in, B)` -> output `(n_out, B)` attended spike times (token-dim
    attention), or `(S, n_in, B)` -> `(S, n_out, B)` sequence attention across
    the `S` positions (per-feature exact scores).

    The block owns three `ExactTTFSLinear` child modules:

        WQ: (n_in, B) -> (n_out, B)  query  spike times (IFT-exact)
        WK: (n_in, B) -> (n_out, B)  key    spike times (IFT-exact)
        WV: (n_in, B) -> (n_out, B)  value  spike times (IFT-exact)

    then the stateless alignment-softmax combine:

        score[i,j] = alignment(t_Q[i] - t_K[j]) / temp
        a[i,j]     = softmax_j(score[i,:])
        out[i]     = sum_j a[i,j] * t_V[j]

    All gradients are exact: through WQ/WK/WV via IFT, and through the combine
    via `ExactAttentionCombineFn` analytic partials. Silent neurons carry zero
    gradient (consistent with the rest of the library).

    Args:
        combine: "gaussian" (default; peaks at alignment, recommended) or
            "kernel" (raw symmetric synaptic kernel K(|d|), peaks at |d| = tm).
    """

    def __init__(self, n_in: int, n_out: int, tm: float = 15.0,
                 ts: float = 4.0, theta: float = 1.0, t_max: float = 40.0,
                 w_scale: float = 0.2, bias_val: float = 1.5,
                 grid_pts: int = 2001, seed: int = 0,
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 temp: float = 1.0, combine: str = "gaussian",
                 peak_tol: float = 1e-2) -> None:
        super().__init__()
        _validate_layer_config(n_in, n_out, tm, ts, theta, t_max, grid_pts)
        dtype = torch.float32 if dtype is None else dtype
        if not dtype.is_floating_point:
            raise ValueError("dtype must be a floating-point torch dtype")
        if n_in != n_out:
            raise ValueError(
                "ExactSpikingAttention requires n_in == n_out (square) in v3.0.0")
        if combine not in ("gaussian", "kernel"):
            raise ValueError("combine must be 'gaussian' or 'kernel'")
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
        self.seed = int(seed)

        common = dict(tm=self.tm, ts=self.ts, theta=self.theta,
                      t_max=self.t_max, w_scale=float(w_scale),
                      bias_val=float(bias_val), grid_pts=int(grid_pts),
                      dtype=dtype, device=device, peak_tol=float(peak_tol))
        self.WQ = ExactTTFSLinear(n_in, n_out, seed=seed, **common)
        self.WK = ExactTTFSLinear(n_in, n_out, seed=seed + 1000, **common)
        self.WV = ExactTTFSLinear(n_in, n_out, seed=seed + 2000, **common)

    def extra_repr(self) -> str:
        return (f"n_in={self.n_in}, n_out={self.n_out}, temp={self.temp}, "
                f"combine={self.combine}, tm={self.tm:.1f}, ts={self.ts:.1f}, "
                f"theta={self.theta}")

    def foreground(self, t_in: torch.Tensor):
        """Return (t_q, t_k, t_v) first-spike maps for inspection/tests."""
        t_q = self.WQ(t_in)
        t_k = self.WK(t_in)
        t_v = self.WV(t_in)
        return t_q, t_k, t_v

    def calibrate_init_fire(self, target: float = 0.5, n_probe: int = 32,
                            cal_grid_pts: int = 65, iters: int = 6) -> None:
        """Calibrate the Q/K/V projection layers' initial firing rates.

        Exact IFT gradients only exist through firing neurons, so a projection
        that is fully silent at init cannot be trained (deadlock). This runs
        ``ExactTTFSLinear.calibrate_init_fire`` on each projection with the
        same arguments, adjusting only the bias columns.
        """
        for p in (self.WQ, self.WK, self.WV):
            p.calibrate_init_fire(target=target, n_probe=n_probe,
                                  cal_grid_pts=cal_grid_pts, iters=iters)

    def _forward_2d(self, t_in: torch.Tensor) -> torch.Tensor:
        if t_in.shape[0] != self.n_in:
            raise ValueError(f"Input dim {t_in.shape[0]} != n_in {self.n_in}")
        t_q = self.WQ(t_in)
        t_k = self.WK(t_in)
        t_v = self.WV(t_in)
        return ExactAttentionCombineFn.apply(
            t_q, t_k, t_v, self.tm, self.ts, self._alpha, self.k_peak,
            self.temp, self.t_max, self.combine)

    def _forward_seq(self, t_in: torch.Tensor) -> torch.Tensor:
        """Sequence attention over the leading axis.

        Input `(S, n, B)`: the Q/K/V projections are applied per position
        (same weights across positions, as in a transformer), then attention is
        computed *across* the S positions with per-feature scores,

            out[s, i, b] = sum_{s'} a[s, s', i, b] * t_v[i, s', b]
            a = softmax_{s'}( alignment(t_q[i, s, b] - t_k[i, s', b]) / temp )

        The per-feature softmax is exactly the 2D combine applied to the
        `(S, B*n)` fold of the position axis, so the same exact closed-form
        combine gradients apply (no mean-collapse over features). Shape `S` is
        checked against ``n_in`` (square attention block).
        """
        S, n, B = t_in.shape
        if n != self.n_in:
            raise ValueError(f"Input feature dim {n} != n_in {self.n_in}")
        validate_spike_times(t_in)
        # project each position: flatten (n, S, B) -> (n_in, S*B)
        flat = t_in.permute(1, 0, 2).reshape(n, S * B)
        q = self.WQ(flat).reshape(n, S, B).permute(1, 0, 2)  # (S, n, B)
        k = self.WK(flat).reshape(n, S, B).permute(1, 0, 2)
        v = self.WV(flat).reshape(n, S, B).permute(1, 0, 2)
        # fold (B, n) into the batch axis so the softmax is per (sample, feature)
        qb = q.permute(0, 2, 1).reshape(S, B * n)
        kb = k.permute(0, 2, 1).reshape(S, B * n)
        vb = v.permute(0, 2, 1).reshape(S, B * n)
        out = ExactAttentionCombineFn.apply(
            qb, kb, vb, self.tm, self.ts, self._alpha, self.k_peak,
            self.temp, self.t_max, self.combine)
        return out.reshape(S, B, n).permute(0, 2, 1)  # (S, n, B)

    def forward(self, t_in: torch.Tensor) -> torch.Tensor:
        """t_in (n_in, B) or (S, n_in, B) -> attended spike times (same shape).

        The 2D path attends across the n token/neuron dimensions; the 3D path
        attends across the S sequence positions with per-feature (exact) scores.
        """
        if t_in.dim() == 2:
            if not t_in.is_floating_point():
                raise ValueError("Input spike times must use a floating-point dtype")
            return self._forward_2d(t_in)
        if t_in.dim() == 3:
            if not t_in.is_floating_point():
                raise ValueError("Input spike times must use a floating-point dtype")
            return self._forward_seq(t_in)
        raise ValueError(
            f"Expected (n_in, B) or (S, n_in, B) but got shape {tuple(t_in.shape)}")