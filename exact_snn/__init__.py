"""Exact-SNN as a drop-in PyTorch (torch.nn) library.

This re-implements the Exact-SNN TTFS engine as REAL torch.nn.Module
components so that it integrates with the standard PyTorch ecosystem:

    * exact gradients via Implicit Function Theorem (IFT) - no surrogates
    * weight parameters are torch.nn.Parameter
    * forward/backward are wired into autograd via torch.autograd.Function
        * works with torch.optim.Adam, nn.DataParallel, and standard PyTorch
            model composition.

The mathematical kernels (grid-scan forward + vectorized IFT backward) are
self-contained in this file, matching the verified engine 1:1, so the package
has NO dependency on the original project's layout and can be pip-installed on
its own.

Quick start (standard PyTorch):
    import torch
    from exact_snn import ExactTTFSLinear, latency_encode, latency_cross_entropy

    model = torch.nn.Sequential(
        ExactTTFSLinear(784, 128),
        ExactTTFSLinear(128, 10),
    )
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)

    t_in = latency_encode(images, t_max=40.0)   # (..., N) -> spike times
    loss, t_out = ...
    loss.backward()       # exact IFT gradients
    opt.step()
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

__version__ = "2.0.0"
__author__ = "Sumith Kumar"

__all__ = [
    "ExactTTFSLinear",
    "ExactTTFSNetwork",
    "latency_encode",
    "latency_cross_entropy",
    "train_simple",
]


def _validate_layer_config(n_in: int, n_out: int, tm: float, ts: float,
                           theta: float, t_max: float, grid_pts: int) -> None:
    if n_in <= 0 or n_out <= 0:
        raise ValueError("n_in and n_out must be positive")
    if tm <= 0 or ts <= 0 or theta <= 0 or t_max <= 0:
        raise ValueError("tm, ts, theta, and t_max must be positive")
    if grid_pts < 3:
        raise ValueError("grid_pts must be at least 3")

# ---------------------------------------------------------------------------
# Optional companion modules (lazy opt-in -- NOT imported at package load so
# the core stays minimal and there is no circular-import risk). Import them
# on demand, e.g.:
#     from exact_snn import existence          # silent-neuron gradients
#     from exact_snn import normalize          # SpikeNorm
#     from exact_snn import losses             # rate_latency_loss
#     from exact_snn import initializers       # xavier_init / kaiming_init
#     from exact_snn import util               # spike_time_augment
#     from exact_snn import reset              # ResetLIF (reference solver)
#     from exact_snn.event import ExactEventLinear   # event-driven drop-in
# Each is an independent, optional component that uses only the core kernels
# here; nothing is required to use the base layers.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Math kernels (self-contained, identical to the verified grid engine).
# ---------------------------------------------------------------------------
def _K(d, tm, ts, alpha, k_peak):
    d = torch.clamp(d, min=0.0)
    d = torch.where(torch.isnan(d), torch.zeros_like(d), d)
    if alpha:
        return (d / tm) * torch.exp(1.0 - d / tm) / k_peak
    return (torch.exp(-d / tm) - torch.exp(-d / ts)) / (tm - ts) / k_peak


def _Kd(d, tm, ts, alpha, k_peak):
    m = d > 0
    d = torch.clamp(d, min=0.0)
    if alpha:
        val = (1.0 - d / tm) * torch.exp(1.0 - d / tm) / (tm * k_peak)
    else:
        val = (-torch.exp(-d / tm) / tm + torch.exp(-d / ts) / ts) / (tm - ts) / k_peak
    return torch.where(m, val, torch.zeros_like(d))


def _u_at(W, t_in, t_bias, tm, ts, alpha, k_peak, t):
    n_in = W.shape[1] - 1
    u = W[:, n_in].view(-1, 1) * _K(t - t_bias, tm, ts, alpha, k_peak)
    if n_in:
        D = t.unsqueeze(-1) - t_in[:n_in].t().unsqueeze(0)
        u = u + (_K(D, tm, ts, alpha, k_peak) * W[:, :n_in].unsqueeze(1)).sum(-1)
    return u


def _du_at(W, t_in, t_bias, tm, ts, alpha, k_peak, t):
    n_in = W.shape[1] - 1
    du = W[:, n_in].view(-1, 1) * _Kd(t - t_bias, tm, ts, alpha, k_peak)
    if n_in:
        D = t.unsqueeze(-1) - t_in[:n_in].t().unsqueeze(0)
        du = du + (_Kd(D, tm, ts, alpha, k_peak) * W[:, :n_in].unsqueeze(1)).sum(-1)
    return du


def _interp_grid(grid, vals, m):
    """Linear interpolation of a grid-sampled membrane `vals` (n_cur,B,G) at
    arbitrary times `m` (n_cur,B). Returns (n_cur,B).

    `vals` holds the exact membrane `u` sampled on the grid, so interpolating
    it avoids recomputing the O(n_in) input-sum for every bisection / golden-
    section step. Newton still refines the result exactly afterwards.
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


def _forward_layer_torch(W, t_prev, t_bias, theta, grid, tm, ts, alpha,
                         k_peak, n_bisect=15, n_newton=8, peak_tol=1e-2):
    """Exact first-spike-time solve for one layer. W: (n_cur, n_in+1)."""
    dev = W.device
    dtype = W.dtype
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    G = grid.numel()

    g = grid.view(1, 1, -1)
    t_data = t_prev[:n_in]
    D = g - t_data.unsqueeze(-1)
    K_vals = _K(D, tm, ts, alpha, k_peak)
    U = (W[:, :n_in] @ K_vals.reshape(n_in, -1)).reshape(n_cur, B, G)
    U += W[:, n_in].view(n_cur, 1, 1) * _K(g - t_bias, tm, ts, alpha, k_peak)

    mask = U >= theta
    any_mask = mask.any(dim=2)
    idx = mask.long().argmax(dim=2)

    t_post = torch.full((n_cur, B), float("inf"), dtype=dtype, device=dev)
    up = torch.zeros((n_cur, B), dtype=dtype, device=dev)

    if any_mask.any():
        at_first = (idx == 0) & any_mask
        idxc = idx.clamp(min=1)
        a = grid[idxc - 1]
        b = grid[idxc]
        fa = U.gather(2, (idxc - 1).unsqueeze(-1)).squeeze(-1) - theta
        fb = U.gather(2, idxc.unsqueeze(-1)).squeeze(-1) - theta
        for _ in range(n_bisect):
            m = 0.5 * (a + b)
            fm = _interp_grid(grid, U, m) - theta
            take_left = fa * fm <= 0.0
            b = torch.where(take_left, m, b)
            fb = torch.where(take_left, fm, fb)
            a = torch.where(take_left, a, m)
            fa = torch.where(take_left, fa, fm)
        m = 0.5 * (a + b)
        for _ in range(n_newton):
            um = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m) - theta
            dum = _du_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m)
            safe = dum > 1e-10
            nm = m - um / torch.where(safe, dum, torch.ones_like(dum))
            nm = nm.clamp(min=a, max=b)
            m = torch.where(safe, nm, m)
        tf = m
        tf = torch.where(at_first, grid[0], tf)
        t_post = torch.where(any_mask, tf, t_post)
        up = torch.where(any_mask, _du_at(W, t_prev, t_bias, tm, ts, alpha,
                                          k_peak, tf), up)

    not_fired = ~any_mask
    if not_fired.any():
        u_max = U.max(dim=2).values
        candidates = not_fired & (u_max >= theta - peak_tol)
        if candidates.any():
            imax = U.argmax(dim=2)
            lo = grid[(imax - 1).clamp(min=0)]
            hi = grid[(imax + 1).clamp(max=G - 1)]
            gr = (math.sqrt(5.0) - 1.0) / 2.0
            c = hi - gr * (hi - lo)
            d = lo + gr * (hi - lo)
            for _ in range(12):
                uc = _interp_grid(grid, U, c)
                ud = _interp_grid(grid, U, d)
                go_hi = uc > ud
                hi = torch.where(go_hi, d, hi)
                lo = torch.where(go_hi, lo, c)
                c = hi - gr * (hi - lo)
                d = lo + gr * (hi - lo)
            t_peak = 0.5 * (lo + hi)
            u_peak = _interp_grid(grid, U, t_peak)
            fire2 = (u_peak >= theta) & not_fired
            if fire2.any():
                a2 = torch.zeros_like(t_peak)
                b2 = t_peak
                fa2 = _interp_grid(grid, U, a2) - theta
                fb2 = u_peak - theta
                for _ in range(n_bisect):
                    m2 = 0.5 * (a2 + b2)
                    fm2 = _interp_grid(grid, U, m2) - theta
                    take_left = fa2 * fm2 <= 0.0
                    b2 = torch.where(take_left, m2, b2)
                    fb2 = torch.where(take_left, fm2, fb2)
                    a2 = torch.where(take_left, a2, m2)
                    fa2 = torch.where(take_left, fa2, fm2)
                m2 = 0.5 * (a2 + b2)
                for _ in range(n_newton):
                    um2 = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m2) - theta
                    dum2 = _du_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m2)
                    safe = dum2 > 1e-10
                    nm2 = m2 - um2 / torch.where(safe, dum2, torch.ones_like(dum2))
                    nm2 = nm2.clamp(min=a2, max=b2)
                    m2 = torch.where(safe, nm2, m2)
                t_post = torch.where(fire2, m2, t_post)
                up = torch.where(fire2, _du_at(W, t_prev, t_bias, tm, ts,
                                               alpha, k_peak, m2), up)
    return t_post, up


def _backward_layer_torch(W, t_prev, t_bias, t_post, lam_post, up,
                          tm, ts, alpha, k_peak):
    """Exact IFT backward for one layer. Returns (grad, lam_prev)."""
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_post.shape[1]
    dev = W.device
    dtype = W.dtype
    grad = torch.zeros_like(W)
    lam_prev = torch.zeros((n_in, B), dtype=dtype, device=dev)

    fired = torch.isfinite(t_post)
    if not fired.any():
        return grad, lam_prev

    la = torch.where(fired, lam_post, torch.zeros_like(lam_post))
    up_safe = torch.where(up != 0.0, up, torch.ones_like(up))
    adj = torch.where(fired & (up != 0.0), la / up_safe, torch.zeros_like(la))

    grad[:, n_in] = -(adj * _K(t_post - t_bias, tm, ts, alpha, k_peak)).sum(dim=1)
    t_data = t_prev[:n_in]
    D_back = t_post.unsqueeze(-1) - t_data.T.unsqueeze(0)
    K_back = _K(D_back, tm, ts, alpha, k_peak)
    Kd_back = _Kd(D_back, tm, ts, alpha, k_peak)
    grad[:, :n_in] = -(adj.unsqueeze(-1) * K_back).sum(dim=1)
    lam_prev = (adj.unsqueeze(-1) * W[:, :n_in].unsqueeze(1) * Kd_back).sum(dim=0).T
    return grad, lam_prev


# ---------------------------------------------------------------------------
# autograd.Function wrapping one exact TTFS layer.
# ---------------------------------------------------------------------------
class _ExactTTFSLayerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, t_prev, t_bias, theta, tm, ts, alpha, k_peak, grid, peak_tol):
        t_post, up = _forward_layer_torch(
            W, t_prev, t_bias, theta, grid, tm, ts, alpha, k_peak,
            peak_tol=float(peak_tol))
        ctx.save_for_backward(W, t_prev, t_post, up)
        ctx.t_bias = float(t_bias)
        ctx.tm = float(tm)
        ctx.ts = float(ts)
        ctx.alpha = bool(alpha)
        ctx.k_peak = float(k_peak)
        return t_post

    @staticmethod
    def backward(ctx, grad_output):
        W, t_prev, t_post, up = ctx.saved_tensors
        grad, lam_prev = _backward_layer_torch(
            W, t_prev, ctx.t_bias, t_post, grad_output, up,
            ctx.tm, ctx.ts, ctx.alpha, ctx.k_peak)
        # forward args: W, t_prev, t_bias, theta, tm, ts, alpha, k_peak, grid, peak_tol
        return grad, lam_prev, None, None, None, None, None, None, None, None


# ---------------------------------------------------------------------------
# Public nn.Module layer.
# ---------------------------------------------------------------------------
class ExactTTFSLinear(nn.Module):
    """A single TTFS (Time-to-First-Spike) linear layer as an nn.Module.

    Maps input spike times (n_in, B) -> output spike times (n_out, B) with the
    exact IFT gradient engine. Weight is an `nn.Parameter` shaped
    (n_out, n_in + 1); the last column is the bias.
    """

    def __init__(self, n_in: int, n_out: int, tm: float = 15.0, ts: float = 4.0,
                 theta: float = 1.0, t_max: float = 40.0, w_scale: float = 0.2,
                 bias_val: float = 1.5, grid_pts: int = 2001, seed: int = 0,
                 dtype: torch.dtype = torch.float32,
                 device: Optional[torch.device] = None,
                 peak_tol: float = 1e-2) -> None:
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
        self.peak_tol = float(peak_tol)
        self._alpha = abs(self.tm - self.ts) < 1e-9
        self.k_peak = self._compute_k_peak(self.tm, self.ts)

        rng = np.random.default_rng(seed)
        w = (rng.standard_normal((n_out, n_in + 1)) * w_scale).astype(np.float64)
        w[:, -1] = bias_val
        self.weight = nn.Parameter(torch.tensor(w, dtype=dtype, device=device))

        # Create the grid on the same device as the weight so it tracks it.
        wdev = self.weight.device
        self.register_buffer("grid", torch.linspace(
            0.0, t_max, int(grid_pts), dtype=self.weight.dtype, device=wdev))

    @staticmethod
    def _compute_k_peak(tm: float, ts: float) -> float:
        if abs(tm - ts) < 1e-9:
            return 1.0
        s = (tm * ts / (tm - ts)) * math.log(tm / ts)
        return float((math.exp(-s / tm) - math.exp(-s / ts)) / (tm - ts))

    def extra_repr(self) -> str:
        return (f"n_in={self.n_in}, n_out={self.n_out}, "
                f"tm={self.tm:.1f}, ts={self.ts:.1f}, theta={self.theta}")

    def forward(self, t_prev: torch.Tensor) -> torch.Tensor:
        if t_prev.dim() != 2:
            raise ValueError(f"Expected (n_in, B) but got shape {tuple(t_prev.shape)}")
        if t_prev.shape[0] != self.n_in:
            raise ValueError(f"Input dim {t_prev.shape[0]} != n_in {self.n_in}")
        W = self.weight
        if not t_prev.is_floating_point():
            raise ValueError("Input spike times must use a floating-point dtype")
        if t_prev.dtype != W.dtype:
            raise ValueError(f"Input dtype {t_prev.dtype} != layer dtype {W.dtype}")
        if t_prev.device != W.device:
            raise ValueError(f"Input device {t_prev.device} != layer device {W.device}")
        grid = self.grid.to(dtype=W.dtype, device=W.device)
        return _ExactTTFSLayerFn.apply(
            W, t_prev, self.t_bias, self.theta, self.tm, self.ts,
            self._alpha, self.k_peak, grid, self.peak_tol)

    def calibrate_init_fire(self, target: float = 0.5, n_probe: int = 32,
                            cal_grid_pts: int = 65) -> None:
        """Post-init calibration so roughly `target` of this layer's neurons
        spike on a random reference input.

        Exact IFT gradients only exist through firing neurons, so a layer that is
        fully silent at init cannot be trained (deadlock). This adjusts the bias
        column (last weight column) so that on a random latency-coded probe input
        roughly `target` of the layer's neurons cross threshold, leaving the
        weight structure intact.

        Uses the existence-channel peak margin (SP-02) to estimate each neuron's
        peak potential for the knot, together with a coarse internal grid and a
        compact probe batch so the estimate is cheap.

        Args:
            target: target firing fraction in (0, 1].
            n_probe: number of random probe samples.
            cal_grid_pts: number of points in the calibration grid.
        """
        from exact_snn.existence import peak_margin_torch
        dev = self.weight.device
        dtype = self.weight.dtype
        cal_grid = torch.linspace(0.0, self.t_max, int(cal_grid_pts),
                                  dtype=dtype, device=dev)
        probe = (torch.rand(self.n_in, int(n_probe), dtype=dtype, device=dev)
                 * 0.8 * self.t_max + 0.1)
        W = self.weight.detach()
        t_post, _ = _forward_layer_torch(
            W, probe, self.t_bias, self.theta, cal_grid, self.tm, self.ts,
            self._alpha, self.k_peak, peak_tol=self.peak_tol)
        fired = torch.isfinite(t_post)
        t_peak, u_peak = peak_margin_torch(
            W, probe, self.t_bias, self.theta, cal_grid, self.tm, self.ts,
            self._alpha, self.k_peak)
        u_eff = torch.where(fired, self.theta * torch.ones_like(u_peak), u_peak)
        need = u_eff.reshape(-1)
        target = float(target)
        k = min(max(1, int(round(target * need.numel()))), need.numel())
        quantile = torch.sort(need).values[k - 1]
        bias = (self.theta - quantile).clamp(min=0.0).item()
        with torch.no_grad():
            self.weight[:, -1] = bias


# ---------------------------------------------------------------------------
# Latency encoding (pixels -> spike times) and differentiable loss.
# ---------------------------------------------------------------------------
def latency_encode(images: torch.Tensor, t_max: float = 40.0,
                   min_t: float = 0.01, max_t: float = 0.99) -> torch.Tensor:
    """Encode intensity images to first-spike times: brighter -> earlier.

    Args:
        images: float tensor in [0, 1] of any shape.
        t_max: simulation window.

    Returns:
        Spike times (same shape): bright pixels map to early (small) times.
    """
    x = torch.clamp(images, min_t, max_t)
    return t_max * (1.0 - x) + 0.1


def latency_cross_entropy(t_out: torch.Tensor, y: torch.Tensor,
                          t_max: float, beta: float = 1.0) -> torch.Tensor:
    """Differentiable latency cross-entropy.

    p_k = softmax(-beta * t_out_k);  L = -ln p_y, averaged over batch.

    Silent outputs (inf) get a large finite placeholder for the softmax only;
    the backward already zeroes silent-neuron gradients, so the placeholder
    never leaks any gradient into autograd.

    Args:
        t_out: (n_out, B) output spike times.
        y: (B,) class indices.
        beta: temperature scale.

    Returns:
        A scalar differentiable loss tensor (supports .backward()).
    """
    B = t_out.shape[1]
    t = torch.where(torch.isfinite(t_out), t_out,
                    torch.full_like(t_out, 2.0 * t_max + 10.0))
    logits = -beta * t
    logits = logits - logits.max(dim=0, keepdim=True).values
    p = torch.exp(logits)
    p = p / p.sum(dim=0, keepdim=True)
    loss = -torch.log(p[y, torch.arange(B, device=t_out.device)] + 1e-12).mean()
    return loss


# ---------------------------------------------------------------------------
# Convenience container: build an nn.Module from sizes (like TTFSNet).
# ---------------------------------------------------------------------------
class ExactTTFSNetwork(nn.Module):
    """A full TTFS network built from sizes, as an nn.Module.

    Provides the familiar `TTFSNet(sizes)` interface, but with the
    autograd-connected layers above, so it works with `torch.optim` and
    `loss.backward()` out of the box.

    Args:
        sizes: [n_in, ..., n_out].
        Additional kwargs are forwarded to each ExactTTFSLinear layer.
    """

    def __init__(self, sizes: Sequence[int], **kwargs) -> None:
        super().__init__()
        sizes = list(sizes)
        self.sizes = sizes
        self.t_max = float(kwargs.pop("t_max", 40.0))
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers.append(ExactTTFSLinear(a, b, t_max=self.t_max, **kwargs))
        self.layers = nn.ModuleList(layers)

    def forward(self, t_in: torch.Tensor) -> torch.Tensor:
        x = t_in
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, t_in: torch.Tensor, y: torch.Tensor,
             beta: float = 1.0) -> torch.Tensor:
        t_out = self.forward(t_in)
        return latency_cross_entropy(t_out, y, self.t_max, beta)

    def calibrate_init_fire(self, target: float = 0.5, n_probe: int = 32,
                            cal_grid_pts: int = 65) -> None:
        """Calibrate every layer's bias so roughly `target` fires per layer.

        Propagates a random probe input through the network, calibrating each
        layer in sequence (see ExactTTFSLinear.calibrate_init_fire).
        """
        from exact_snn.existence import peak_margin_torch
        dev = self.layers[0].weight.device
        dtype = self.layers[0].weight.dtype
        cal_grid = torch.linspace(0.0, self.t_max, int(cal_grid_pts),
                                  dtype=dtype, device=dev)
        probe = (torch.rand(self.sizes[0], int(n_probe), dtype=dtype,
                            device=dev) * 0.8 * self.t_max + 0.1)
        cur = probe
        for layer in self.layers:
            W = layer.weight.detach()
            t_post, _ = _forward_layer_torch(
                W, cur, layer.t_bias, layer.theta, cal_grid, layer.tm,
                layer.ts, layer._alpha, layer.k_peak, peak_tol=layer.peak_tol)
            fired = torch.isfinite(t_post)
            t_peak, u_peak = peak_margin_torch(
                W, cur, layer.t_bias, layer.theta, cal_grid, layer.tm,
                layer.ts, layer._alpha, layer.k_peak)
            u_eff = torch.where(fired, layer.theta * torch.ones_like(u_peak),
                                u_peak)
            need = u_eff.reshape(-1)
            k = min(max(1, int(round(float(target) * need.numel()))),
                    need.numel())
            quantile = torch.sort(need).values[k - 1]
            bias = (layer.theta - quantile).clamp(min=0.0).item()
            with torch.no_grad():
                layer.weight[:, -1] = bias
            # propagate the (re-biased) layer output to feed the next layer
            t_post2, _ = _forward_layer_torch(
                W, cur, layer.t_bias, layer.theta, cal_grid, layer.tm,
                layer.ts, layer._alpha, layer.k_peak, peak_tol=layer.peak_tol)
            cur = t_post2


# ---------------------------------------------------------------------------
# One-shot training helper (optional, but keeps it simple to use).
# ---------------------------------------------------------------------------
def train_simple(model: nn.Module, X: torch.Tensor, y: torch.Tensor,
                 epochs: int = 50, lr: float = 2e-3, batch_size: int = 32,
                 t_max: float = 40.0, beta: float = 1.0,
                 device: Optional[torch.device] = None,
                 seed: int = 0) -> None:
    """Train a TTFS network on pixel data with standard Adam.

    Args:
        model: any forward() that takes (n_in, B) spike times.
        X: (N, features) pixel intensities in [0, 1].
        y: (N,) class labels.
        epochs: number of passes over the data.
        lr: Adam learning rate.
        batch_size: mini-batch size.
        t_max: simulation window for latency encoding.
        beta: latency CE temperature.
        device: compute device (defaults to model param device).
        seed: deterministic shuffle seed.
    """
    device = device or next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    X = torch.as_tensor(X, device=device, dtype=dtype)
    y = torch.as_tensor(y, device=device, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    N = X.shape[0]
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = rng.permutation(N)
        total = 0.0
        nb = 0
        for i in range(0, N, batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx]
            t_in = latency_encode(xb.T, t_max=t_max)
            yb = y[idx]
            opt.zero_grad()
            loss = model.loss(t_in, yb, beta=beta)
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        print(f"epoch {ep + 1}/{epochs}  loss = {total / nb:.4f}")
