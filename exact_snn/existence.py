"""SP-02: existence-channel gradients for silent neurons (optional, exact).

Purpose
-------
Exact IFT gradients only exist through neurons that FIRE. A silent neuron
(u(t) < theta for all t) therefore receives zero gradient through the timing
path and can stay dead forever (deadlock). The "existence channel" revives such
neurons by treating the peak membrane potential u_peak as a smooth escape-noise
existence signal:

    p_j = sigmoid((u_peak_j - theta) / T_noise)
    L_exist = -(lam/B) * sum over targeted silent j of log p_j

with the envelope theorem giving an *exact* gradient

    d(u_peak_j)/dW_ji = K(t_peak_j - t_i)

This module provides:

  - `peak_margin_torch`: exact (t_peak, u_peak) extremum for silent neurons
    (golden-section refined, response-window restricted).
  - `edge_peak_guard`: flags degenerate/extremum-boundary cases where the
    envelope theorem is unreliable (deadlock plateau, flippable earliest event).
  - `silent_existence_targets`: the SP-02 target mask per layer.
  - `existence_loss_and_grads`: forward + escape-noise existence loss/weight
    gradients for a list of ExactTTFSLinear layers, to be ADDED to the normal
    autograd timing gradients.

Usage (plugin — fully optional, no mandatory pipeline)
------------------------------------------------------
    model = nn.Sequential(ExactTTFSLinear(784,128), ExactTTFSLinear(128,10))
    t_out = model(t_in)
    loss = latency_cross_entropy(t_out, y, t_max)
    loss.backward()                       # exact timing gradients

    # --- optionally add silent-neuron existence gradients ---
    layers = list(model)
    e_loss, e_grads, stats = existence_loss_and_grads(
        layers, t_in, y, T_noise=1.0, lam=1.0)
    for layer, eg in zip(layers, e_grads):
        layer.weight.grad.add_(eg)
    total = float(loss) + e_loss

The existence gradients are computed from the SAME exact kernels as the timing
path, so they are exact for the peak-margin model. `existence_loss_and_grads`
is verified against finite differences on silent targeted neurons (see
tests/test_existence.py).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch

from exact_snn import (
    _K,
    _Kd,
    _u_at,
    _forward_layer_torch,
    latency_cross_entropy,
)


# ---------------------------------------------------------------------------
# Peak margin + edge guard (standalone, operate on a raw weight tensor).
# ---------------------------------------------------------------------------
def peak_margin_torch(
    W: torch.Tensor, t_prev: torch.Tensor, t_bias: float, theta: float,
    grid: torch.Tensor, tm: float, ts: float, alpha: bool,
    k_peak: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Refined extremum (time, potential) per neuron over the RESPONSE window.

    The peak search is restricted to t >= t_start, where t_start is the time of
    the earliest *contributing* presynaptic event (|w| > 0). Without this the
    argmax of a subthreshold neuron collapses onto the pre-input plateau
    (u = 0, K(t_peak - t_in) = 0), giving a zero existence gradient -- the exact
    far-dead deadlock this channel exists to fix.

    Extremum choice (SP-02):
      - positive response (u_max >= 0): interior MAX, u'(t_peak) = 0;
      - all-negative response (u_max < 0): the max sits at the window boundary
        (u' undefined, envelope fails -> deadlock), so the channel uses the
        interior MIN instead -- still u' = 0, so the envelope theorem
        d(u_peak)/dW_ji = K(t_peak - t_i) holds.

    Fired neurons (u_peak >= theta) are returned with t_peak = inf, u_peak = 0
    as a marker; only SILENT neurons' extrema are meaningful.

    Args:
        W: weight tensor (n_cur, n_in + 1).
        t_prev: presynaptic spike times (n_in, B).
        t_bias, theta, tm, ts, alpha, k_peak: model params.
        grid: time grid (G,).

    Returns:
        (t_peak, u_peak) each of shape (n_cur, B).
    """
    dev = W.device
    dtype = W.dtype
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    G = grid.numel()
    inf = float("inf")

    g = grid.view(1, 1, -1)
    U = torch.zeros((n_cur, B, G), dtype=dtype, device=dev)
    for i in range(n_in):
        d = g - t_prev[i].view(1, -1, 1)
        U += W[:, i].view(n_cur, 1, 1) * _K(d, tm, ts, alpha, k_peak)
    U += W[:, n_in].view(n_cur, 1, 1) * _K(g - t_bias, tm, ts, alpha, k_peak)

    ev_times = torch.full((n_cur, B, n_in + 1), inf, dtype=dtype, device=dev)
    if n_in:
        ev_times[:, :, :n_in] = t_prev.t().unsqueeze(0)
    ev_times[:, :, n_in] = t_bias
    ev_w = torch.cat([W[:, :n_in], W[:, n_in].view(-1, 1)], dim=1).abs()
    contrib = ev_w > 1e-12
    t_start = ev_times.where(contrib.unsqueeze(1),
                             torch.full_like(ev_times, inf)).min(dim=2).values
    t_start = torch.where(torch.isfinite(t_start), t_start, torch.zeros_like(t_start))

    g_win = g >= t_start.unsqueeze(-1)
    U_max = torch.where(g_win, U, torch.full_like(U, -inf))
    U_min = torch.where(g_win, U, torch.full_like(U, inf))
    imax = U_max.argmax(dim=2)
    u_max = U.gather(2, imax.unsqueeze(-1)).squeeze(-1)
    all_neg = u_max <= 0.0
    im = torch.where(all_neg, U_min.argmin(dim=2), imax)

    idx_start = torch.where(
        torch.isfinite(t_start), (t_start / grid[1]).round().long(),
        torch.zeros_like(t_start, dtype=torch.long))
    idx_start = idx_start.clamp(max=G - 1)
    im_cl = im.clamp(min=idx_start, max=torch.full_like(idx_start, G - 1))
    lo = grid[(im_cl - 1).clamp(min=idx_start)]
    hi = grid[(im_cl + 1).clamp(max=G - 1)]
    sgn = torch.where(all_neg, torch.full_like(u_max, -1.0), torch.ones_like(u_max))
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    for _ in range(30):
        uc = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, c)
        ud = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, d)
        go_hi = (sgn * (uc - ud)) > 0.0
        hi = torch.where(go_hi, d, hi)
        lo = torch.where(go_hi, lo, c)
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
    t_peak = 0.5 * (lo + hi)
    u_peak = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, t_peak)

    fired = u_peak >= theta
    t_peak = torch.where(fired, torch.full_like(t_peak, inf), t_peak)
    u_peak = torch.where(fired, torch.zeros_like(u_peak), u_peak)
    return t_peak, u_peak


def edge_peak_guard(
    W: torch.Tensor, t_prev: torch.Tensor, t_bias: float,
    t_peak: torch.Tensor, u_peak: torch.Tensor, grid: torch.Tensor,
    w_cut: float = 1e-9, edge_cells: float = 1.5,
    u_cut: float = 1e-6,
) -> torch.Tensor:
    """SP-02 boundary-extremum guard. Returns (n_cur, B) mask: True => the
    existence channel must NOT target this neuron.

    The envelope theorem d(u_peak)/dW_ji = K(t_peak - t_in_i) is valid at
    interior extrema (u'(t_peak) = 0) and at the fixed right endpoint t_max
    (dt_peak/dW = 0). At the window start t_start it is ALSO valid whenever the
    earliest contributing event's weight is stably nonzero: u(t_start) = 0
    always (causal kernels K(0) = 0), so a max AT t_start has u_peak = 0 and
    falls into the all-negative branch, which selects the interior MIN instead.

    The single failure mode is the DEGENERATE pre-input plateau (u(t) = 0
    identically, e.g. all-near-zero weights): u_peak = 0 at t_start, the
    channel gradient is exactly 0 (deadlock); and if the earliest event's weight
    sits at the |w| > 1e-12 contrib cutoff, t_start itself moves with W and the
    envelope misses a u'(t_start) dt_start/dW term. Both sub-cases are flagged.
    """
    n_cur, n_inp = W.shape
    n_in = n_inp - 1
    B = t_prev.shape[1]
    dev = W.device
    dtype = W.dtype
    inf = float("inf")
    ev_times = torch.full((n_cur, B, n_in + 1), inf, dtype=dtype, device=dev)
    if n_in:
        ev_times[:, :, :n_in] = t_prev.t().unsqueeze(0)
    ev_times[:, :, n_in] = t_bias
    ev_w = torch.cat([W[:, :n_in], W[:, n_in].view(-1, 1)], dim=1).abs()
    contrib = ev_w > 1e-12
    masked = torch.where(contrib.unsqueeze(1), ev_times,
                         torch.full_like(ev_times, inf))
    t_start = masked.min(dim=2).values
    t_start = torch.where(torch.isfinite(t_start), t_start,
                          torch.zeros_like(t_start))
    earliest_idx = masked.argmin(dim=2)             # (n_cur, B)
    earliest_w = ev_w.gather(1, earliest_idx)       # (n_cur, B)
    step = grid[1]
    at_start = t_peak <= t_start + edge_cells * step
    flat = (u_peak.abs() < u_cut) & at_start
    flippable = (earliest_w <= w_cut) & at_start
    return flat | flippable


def silent_existence_targets(
    fired: torch.Tensor, is_output: bool, y: torch.Tensor,
    correct_output_target: bool = True, hidden_target: float = 1.0,
) -> torch.Tensor:
    """SP-02 target mask (n_cur, B) of silent neurons to revive.

    Hidden layers: all silent neurons (scaled by hidden_target).
    Output layer: only silent neurons of the correct class (if correct_output_target).
    """
    if is_output and correct_output_target:
        onehot = torch.zeros_like(fired)
        onehot[y, torch.arange(fired.shape[1], device=fired.device)] = True
        return (~fired) & onehot
    return (~fired) * (hidden_target != 0.0)


# ---------------------------------------------------------------------------
# Existence loss + gradients over a list of ExactTTFSLinear layers.
# ---------------------------------------------------------------------------
def existence_loss_and_grads(
    layers: Sequence[torch.nn.Module],
    t_in: torch.Tensor,
    y: torch.Tensor,
    T_noise: float = 1.0,
    lam: float = 1.0,
    hidden_target: float = 1.0,
    correct_output_target: bool = True,
    exclude: Optional[List[Optional[torch.Tensor]]] = None,
):
    """Forward + escape-noise existence channel over a list of ExactTTFSLinear.

    Args:
        layers: iterable of ExactTTFSLinear (n_in, B) -> (n_out, B).
        t_in: input spike times (n_in, B).
        y: class labels (B,).
        T_noise: escape-noise temperature.
        lam: existence-channel strength (scalar).
        hidden_target, correct_output_target: target selection (see
            silent_existence_targets).
        exclude: optional per-layer (n_cur, B) bool masks to remove from targets.

    Returns:
        (e_loss, e_grads, stats) where
          e_loss: scalar float escape-noise existence loss.
          e_grads: list aligned to `layers` of weight-gradient tensors
                   (n_cur, n_in+1) to ADD to layer.weight.grad.
          stats: dict with per-layer silent/targeted/guarded counts.

    The timing loss is NOT computed here (that is done by the normal autograd
    path); only the existence channel is returned so the user can combine them.
    """
    layers = list(layers)
    n_layers = len(layers)
    B = t_in.shape[1]
    dev = t_in.device
    dtype = t_in.dtype

    def _forward(layer, x):
        return _forward_layer_torch(
            layer.weight, x, layer.t_bias, layer.theta, layer.grid,
            layer.tm, layer.ts, layer._alpha, layer.k_peak,
            peak_tol=layer.peak_tol)[0]

    # Forward, caching per-layer outputs and firing masks.
    acts = []
    x = t_in
    for layer in layers:
        t_post = _forward(layer, x)
        acts.append((x, t_post))
        x = t_post

    lam_l = [float(lam)] * n_layers

    g = []           # dL_exist/d(u_peak) per layer, masked (n_cur, B)
    peaks = []       # (t_peak, u_peak) per layer
    silent_stats = []
    e_loss = 0.0
    for l, layer in enumerate(layers):
        W = layer.weight
        t_prev, t_post = acts[l]
        fired = torch.isfinite(t_post)
        n_cur = t_post.shape[0]
        if fired.all():
            g.append(torch.zeros((n_cur, B), dtype=dtype, device=dev))
            peaks.append((t_post, torch.zeros_like(t_post)))
            silent_stats.append({"n_silent": 0, "n_targeted": 0,
                                 "n_edge_guarded": 0})
            continue
        t_peak, u_peak = peak_margin_torch(
            W, t_prev, layer.t_bias, layer.theta, layer.grid,
            layer.tm, layer.ts, layer._alpha, layer.k_peak)
        is_output = (l == n_layers - 1)
        target = silent_existence_targets(
            fired, is_output, y,
            correct_output_target=correct_output_target,
            hidden_target=hidden_target).to(dtype)
        guard = edge_peak_guard(
            W, t_prev, layer.t_bias, t_peak, u_peak, layer.grid)
        target = target * (~guard).to(dtype)
        if exclude is not None and exclude[l] is not None:
            target = target * (~exclude[l]).to(dtype)
        p = torch.sigmoid((u_peak - layer.theta) / T_noise)
        e_loss += (lam_l[l] / B) * float(
            (-target * torch.log(p.clamp(min=1e-12))).sum().detach())
        g.append(-(lam_l[l] / B) * target * (1.0 - p) / T_noise)
        peaks.append((t_peak, u_peak))
        silent_stats.append({
            "n_silent": int((~fired).sum().item()),
            "n_targeted": int((target > 0).sum().item()),
            "n_edge_guarded": int(guard.sum().item()),
        })

    # Weight gradients via the envelope theorem + existence adjoint into prev.
    e_grads = [None] * n_layers
    lam_exist = None
    for l in reversed(range(n_layers)):
        layer = layers[l]
        W = layer.weight
        t_prev, _t_post = acts[l]
        n_in = W.shape[1] - 1
        e_grad_layer = torch.zeros_like(W)
        g_l = g[l].clone()
        targeted = g_l != 0
        if targeted.any():
            t_peak_l, _ = peaks[l]
            e_grad_layer[:, n_in] = (
                g_l * _K(t_peak_l - layer.t_bias, layer.tm, layer.ts,
                         layer._alpha, layer.k_peak)).sum(dim=1)
            for i in range(n_in):
                d = t_peak_l - t_prev[i].view(1, -1)
                e_grad_layer[:, i] = (
                    g_l * _K(d, layer.tm, layer.ts, layer._alpha,
                             layer.k_peak)).sum(dim=1)
        e_grads[l] = e_grad_layer
        # existence adjoint into previous layer (only needed to flow the
        # existence signal back; kept minimal and exact).
        lam_exist = torch.zeros((n_in, B), dtype=dtype, device=dev)
        if targeted.any():
            t_peak_l, _ = peaks[l]
            for i in range(n_in):
                d = t_peak_l - t_prev[i].view(1, -1)
                lam_exist[i] = (
                    g_l * W[:, i].view(-1, 1)
                    * _Kd(d, layer.tm, layer.ts, layer._alpha,
                          layer.k_peak)).sum(dim=0)

    stats = {
        "loss_exist": e_loss,
        "silent_per_layer": silent_stats,
    }
    return e_loss, e_grads, stats
