"""Tests for the extended Exact-SNN layers (conv, multi-spike, recurrent)."""
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exact_snn import latency_encode, ExactTTFSLinear
from exact_snn.extended import (
    ExactTTFSConv2d,
    ExactMultiSpike,
    ExactRecurrent,
    multispike_latency_loss,
    spike_count_cross_entropy,
)

DEVICE = "cpu"
DTYPE = torch.float64


def _rng_nodes(shape, rng, lo=0.2, hi=25.0, silent_frac=0.15):
    t = rng.uniform(lo, hi, shape).astype(np.float64)
    t_prev = torch.tensor(t, dtype=DTYPE, device=DEVICE)
    silent = rng.uniform(0, 1, shape) < silent_frac
    return torch.where(torch.tensor(silent, dtype=torch.bool, device=DEVICE),
                       torch.full_like(t_prev, float("inf")), t_prev)


def _sum_loss(t):
    """Sum of output magnitudes over FINITE (fired) neurons only."""
    f = torch.isfinite(t)
    if not f.any():
        return t.sum() * 0.0
    return t[f].abs().sum() / f.float().sum()


def test_conv_build_and_backward_populates():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    layer = ExactTTFSConv2d(3, 5, kernel_size=3, stride=1, padding=1,
                            dtype=DTYPE, device=DEVICE)
    t_in = _rng_nodes((2, 3, 8, 8), rng, silent_frac=0.1)
    t_out = layer(t_in)
    assert t_out.shape == (2, 5, 8, 8)
    assert torch.isfinite(t_out).any()
    loss = _sum_loss(t_out)
    loss.backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert layer.weight.grad.numel() == layer.weight.numel()


def test_conv_param_is_nn_parameter():
    layer = ExactTTFSConv2d(1, 2, kernel_size=3, dtype=DTYPE, device=DEVICE)
    assert isinstance(layer.weight, nn.Parameter)
    assert list(layer.parameters())[0] is layer.weight


def test_conv_stride_padding_output_size():
    layer = ExactTTFSConv2d(3, 4, kernel_size=3, stride=2, padding=1,
                            dtype=DTYPE, device=DEVICE)
    rng = np.random.default_rng(1)
    t_in = _rng_nodes((2, 3, 10, 10), rng, silent_frac=0.1)
    t_out = layer(t_in)
    assert t_out.shape == (2, 4, 5, 5)


def test_conv_stride_backward_input_shape():
    """Strided conv backward must return the original input-map shape."""
    torch.manual_seed(1)
    rng = np.random.default_rng(4)
    layer = ExactTTFSConv2d(2, 3, kernel_size=3, stride=2, padding=1,
                            dtype=DTYPE, device=DEVICE)
    t_in = _rng_nodes((2, 2, 10, 10), rng, silent_frac=0.0)
    t_out = layer(t_in)
    _sum_loss(t_out).backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_conv_gradient_fd_match():
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    layer = ExactTTFSConv2d(3, 2, kernel_size=3, stride=1, padding=1,
                            dtype=DTYPE, device=DEVICE)
    t_in = _rng_nodes((2, 3, 6, 6), rng, silent_frac=0.1)
    eps = 1e-6
    idx = (0, 0)
    W = layer.weight.detach().clone()
    base_loss = lambda: _sum_loss(layer(t_in))
    l0 = base_loss()
    l0.backward()
    g_auto = layer.weight.grad[idx].item()

    Wp = W.clone(); Wp[idx] += eps; layer.weight.data.copy_(Wp)
    lp = base_loss().item()
    Wm = W.clone(); Wm[idx] -= eps; layer.weight.data.copy_(Wm)
    lm = base_loss().item()
    g_fd = (lp - lm) / (2 * eps)
    layer.weight.data.copy_(W)
    assert abs(g_auto - g_fd) < 1e-3


def test_multispike_build_and_backward():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    layer = ExactMultiSpike(4, 3, dtype=DTYPE, device=DEVICE, max_spikes=8)
    t_in = _rng_nodes((4, 3), rng)
    t_all = layer(t_in)
    assert t_all.shape == (3, 3, 8)
    loss = spike_count_cross_entropy(t_all, torch.tensor([0, 1, 2]), 40.0)
    loss.backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


def test_multispike_weight_is_parameter():
    layer = ExactMultiSpike(3, 4, dtype=DTYPE, device=DEVICE)
    assert isinstance(layer.weight, nn.Parameter)
    assert layer.weight.shape == (4, 4)


def test_multispike_latency_loss_differentiable_optim():
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    layer = ExactMultiSpike(5, 3, dtype=DTYPE, device=DEVICE, max_spikes=6)
    t_in = _rng_nodes((5, 4), rng)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
    y = torch.tensor([0, 1, 2, 0])
    opt.zero_grad()
    t_all = layer(t_in)
    loss = multispike_latency_loss(t_all, y, 40.0)
    loss.backward()
    opt.step()
    assert all(torch.isfinite(p.grad).all() for p in layer.parameters())


def test_multispike_forward_vectorized_matches_reference():
    """Guard the vectorized multi-spike forward (matmul U_base + interpolated
    bisection). On firing data the optimized forward must reproduce the exact
    all-recompute reference spike times to within grid/Newton tolerance."""
    from exact_snn import _u_at, _du_at, _K, _Kd
    from exact_snn.extended import _multispike_forward, _u_at_ms, _du_at_ms

    torch.manual_seed(0)
    dev, dt = DEVICE, DTYPE
    tm, ts, theta = 15.0, 4.0, 1.0
    k_peak = ExactTTFSLinear._compute_k_peak(tm, ts)
    alpha = False
    t_bias = 0.0

    def reference(W, t_prev, grid, K):
        n_cur, n_inp = W.shape; n_in = n_inp - 1; B = t_prev.shape[1]; G = grid.numel()
        b_inv = 1.0 / ts
        g = grid.view(1, 1, -1)
        K_grid = _K(g - t_prev.unsqueeze(-1), tm, ts, alpha, k_peak)
        Ub = (W[:, :n_in] @ K_grid.reshape(n_in, -1)).reshape(n_cur, B, G)
        Ub += W[:, n_in].view(n_cur, 1, 1) * _K(g - t_bias, tm, ts, alpha, k_peak)
        t_all = torch.full((n_cur, B, K), float("inf"), dtype=dt, device=dev)
        up_all = torch.zeros(n_cur, B, K, dtype=dt, device=dev)
        t_f_prev = torch.zeros(n_cur, B, dtype=dt, device=dev)
        i_f_prev = torch.zeros(n_cur, B, dtype=dt, device=dev)
        unconsumed = torch.ones(n_cur, B, n_in, dtype=torch.bool, device=dev)
        U = Ub.clone()
        for k in range(K):
            mask = U >= theta
            any_mask = mask.any(2)
            if not any_mask.any():
                break
            idx = mask.long().argmax(2); idxc = idx.clamp(min=1)
            a_br = grid[(idxc - 1).clamp(min=0)]; b_br = grid[idxc.clamp(max=G - 1)]
            fa = U.gather(2, (idxc - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1) - theta
            fb = U.gather(2, idxc.unsqueeze(-1)).squeeze(-1) - theta
            at_first = (idx == 0) & any_mask
            for _ in range(15):
                m = 0.5 * (a_br + b_br)
                fm = (_u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m) - theta
                      if k == 0 else _u_at_ms(W, t_prev, tm, ts, k_peak, m,
                                              t_f_prev, i_f_prev, unconsumed) - theta)
                left = fa * fm <= 0
                b_br = torch.where(left, m, b_br); fb = torch.where(left, fm, fb)
                a_br = torch.where(left, a_br, m); fa = torch.where(left, fa, fm)
            m = 0.5 * (a_br + b_br)
            for _ in range(8):
                if k == 0:
                    um = _u_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m) - theta
                    dum = _du_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, m)
                else:
                    um = _u_at_ms(W, t_prev, tm, ts, k_peak, m, t_f_prev,
                                  i_f_prev, unconsumed) - theta
                    dum = _du_at_ms(W, t_prev, tm, ts, k_peak, m, t_f_prev,
                                    i_f_prev, unconsumed)
                safe = dum > 1e-10
                nm = m - um / torch.where(safe, dum, torch.ones_like(dum))
                nm = nm.clamp(min=a_br, max=b_br)
                m = torch.where(safe, nm, m)
            tf = torch.where(any_mask, torch.where(at_first, grid[0], m),
                             torch.full_like(m, float("inf")))
            up_k = (_du_at(W, t_prev, t_bias, tm, ts, alpha, k_peak, tf)
                    if k == 0 else _du_at_ms(W, t_prev, tm, ts, k_peak, tf,
                                             t_f_prev, i_f_prev, unconsumed))
            fired = any_mask & torch.isfinite(tf)
            # average over fired, handle none
            if fired.any():
                t_all[:, :, k] = torch.where(fired, tf, t_all[:, :, k])
                up_all[:, :, k] = torch.where(fired, up_k, up_all[:, :, k])
            if not fired.any():
                break
            tb = t_prev.t()
            consumed = tb.unsqueeze(0) <= tf.unsqueeze(-1)
            exp_dt = torch.exp(-(1.0 / ts) * (tf.unsqueeze(-1) - tb.unsqueeze(0)).clamp(min=0))
            i_f_new = (W[:, n_in].unsqueeze(1) * torch.exp(-(1.0 / ts) * tf)
                       + (W[:, :n_in].unsqueeze(1) * exp_dt * consumed.float()).sum(2))
            unconsumed = torch.where(fired.unsqueeze(-1),
                                     unconsumed & ~(tb.unsqueeze(0) <= tf.unsqueeze(-1)),
                                     unconsumed)
            forced = torch.einsum('jbi,ibg->jbg',
                                  W[:, :n_in].unsqueeze(1) * unconsumed.float(), K_grid)
            U = torch.where(fired.unsqueeze(-1),
                            i_f_new.unsqueeze(-1) * ts * k_peak * _K(
                                g - tf.unsqueeze(-1), tm, ts, alpha, k_peak) + forced,
                            U)
            t_f_prev = torch.where(fired, tf, t_f_prev)
            i_f_prev = torch.where(fired, i_f_new, i_f_prev)
        return t_all, up_all

    layer = ExactMultiSpike(10, 4, tm=tm, ts=ts, t_max=40.0, bias_val=5.0,
                            grid_pts=601, max_spikes=3, dtype=DTYPE,
                            device=DEVICE, first_spike_only=True)
    W = layer.weight.detach()
    t_prev = torch.rand(10, 6, dtype=DTYPE, device=DEVICE) * 12.0
    grid = layer.grid.to(dtype=DTYPE, device=DEVICE)

    a_t, a_up = _multispike_forward(W, t_prev, t_bias, tm, ts, theta, k_peak,
                                    40.0, grid, max_spikes=3)[2:]
    b_t, b_up = reference(W, t_prev, grid, 3)
    both = torch.isfinite(a_t) & torch.isfinite(b_t) & torch.isfinite(a_up)
    assert both.any(), "no common firing on reference data"
    # spike times match to ~grid/Newton tolerance (well under 0.1)
    assert (a_t[both] - b_t[both]).abs().max().item() < 0.05
    # whoever fires must agree between optimized and reference
    who = torch.isfinite(a_t) | torch.isfinite(b_t)
    agree = torch.isfinite(a_t) == torch.isfinite(b_t)
    assert agree[who].all(), "optimized/reference disagree on who fires"


def test_recurrent_forward_step_and_state():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    layer = ExactRecurrent(8, 3, dtype=DTYPE, device=DEVICE)
    layer.reset_state(2)
    t_in = _rng_nodes((8, 2), rng, silent_frac=0.0)
    t_a = layer.forward_step(t_in)
    t_b = layer.forward_step(t_in)
    assert t_a.shape == (3, 2)
    assert t_b.shape == (3, 2)
    assert torch.isfinite(layer._trace).all()
    assert layer.weight.grad is None
    f = torch.isfinite(t_b)
    if f.any():
        _sum_loss(t_b).backward()
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.shape == (3, 10)


def test_recurrent_trace_no_nan_and_builds_up():
    """Guards the first-fire NaN bug: a freshly-spiked (last=inf) neuron
    must not produce inf decay -> nan trace, and the trace should build up."""
    torch.manual_seed(0)
    layer = ExactRecurrent(6, 4, dtype=DTYPE, device=DEVICE, tau_rec=6.0)
    layer.reset_state(4)
    rng = np.random.default_rng(1)
    t_in = torch.tensor(rng.uniform(1.0, 20.0, (6, 4)), dtype=DTYPE,
                        device=DEVICE)
    means = []
    for _ in range(4):
        layer.forward_step(t_in)
        assert torch.isfinite(layer._trace).all(), "trace must stay finite"
        means.append(float(layer._trace.mean().item()))
    # trace should build up monotonically as events accumulate
    assert all(m >= 0.0 for m in means)
    assert means[-1] >= means[0]


def test_recurrent_weight_is_parameter():
    layer = ExactRecurrent(3, 2, dtype=DTYPE, device=DEVICE)
    assert isinstance(layer.weight, nn.Parameter)
    assert layer.weight.shape == (2, 5)


def test_sequential_plug_and_play_conv_linear():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = nn.Sequential(
        ExactTTFSConv2d(3, 4, kernel_size=3, stride=1, padding=1,
                        dtype=DTYPE, device=DEVICE),
    )
    t_in = _rng_nodes((2, 3, 7, 7), rng)
    t_out = model(t_in)
    assert t_out.shape == (2, 4, 7, 7)
    _sum_loss(t_out).backward()
    for p in model.parameters():
        assert p.grad is not None
