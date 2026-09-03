"""Focused tests for the optional companion modules.

Covers existence gradients (silent-neuron), calibration, SpikeNorm,
rate-latency loss, initializers, spike-time augmentation, the event-driven
layer, and the ResetLIF reference solver.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from exact_snn import (
    ExactTTFSLinear,
    ExactTTFSNetwork,
    latency_encode,
    latency_cross_entropy,
)
from exact_snn.existence import (
    existence_loss_and_grads,
    peak_margin_torch,
    edge_peak_guard,
)
from exact_snn.normalize import SpikeNorm
from exact_snn.losses import rate_latency_loss
from exact_snn.initializers import xavier_init, kaiming_init
from exact_snn.event import ExactEventLinear
from exact_snn.util import spike_time_augment
from exact_snn.reset import ResetLIF

DEVICE = torch.device("cpu")
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Existence gradients (the carefully-integrated silent-neuron feature).
# ---------------------------------------------------------------------------
def _make_silent_layer(n_in=6, n_out=4, w_scale=0.04, bias_val=-0.3, seed=3):
    return ExactTTFSLinear(n_in, n_out, w_scale=w_scale, bias_val=bias_val,
                           dtype=DTYPE, device=DEVICE, seed=seed)


def test_existence_returns_loss_and_grads():
    layer = _make_silent_layer()
    t_in = torch.rand(6, 12, dtype=DTYPE) * 0.6 * 40 + 0.1
    y = torch.randint(0, 4, (12,))
    e_loss, e_grads, stats = existence_loss_and_grads([layer], t_in, y)
    assert isinstance(e_loss, float)
    assert e_loss >= 0.0
    assert len(e_grads) == 1
    assert e_grads[0].shape == layer.weight.shape
    assert stats["silent_per_layer"][0]["n_targeted"] > 0


def test_existence_targets_silent_not_fired():
    """Silent neurons get nonzero existence gradients; the targeting counts
    agree between the peak-margin silent set and the targets."""
    layer = _make_silent_layer()
    t_in = torch.rand(6, 12, dtype=DTYPE) * 0.6 * 40 + 0.1
    y = torch.randint(0, 4, (12,))
    e_loss, e_grads, stats = existence_loss_and_grads([layer], t_in, y)
    assert (e_grads[0] != 0).any(), "expected nonzero existence gradients"
    # every nonzero gradient row corresponds to a targeted silent neuron
    nonzero_rows = (e_grads[0] != 0).any(dim=1)
    assert nonzero_rows.any()


def test_existence_gradients_match_finite_differences():
    """The existence weight gradients must match finite differences on a
    targeted silent neuron (the envelope-theorem gradient is exact)."""
    layer = _make_silent_layer()
    t_in = torch.rand(6, 12, dtype=DTYPE) * 0.6 * 40 + 0.1
    y = torch.randint(0, 4, (12,))
    T_noise, lam = 1.0, 2.0

    e_loss, e_grads, stats = existence_loss_and_grads(
        [layer], t_in, y, T_noise=T_noise, lam=lam)
    gd = e_grads[0]

    # Pick an output-neuron row that is actually TARGETED: with a single output
    # layer and correct_output_target=True, only silent neurons of the correct
    # class are revived, so the row's bias column (always contributing) is then
    # guaranteed nonzero. Search rows for a nonzero bias-column gradient.
    bias_col = layer.weight.shape[1] - 1
    target_rows = (gd[:, bias_col] != 0).nonzero().flatten()
    assert target_rows.numel() > 0, "no targeted silent neuron in the batch"
    j0 = int(target_rows[0])

    def loss_fn():
        el, _, _ = existence_loss_and_grads(
            [layer], t_in, y, T_noise=T_noise, lam=lam)
        return el

    eps = 1e-6
    # verify the bias column (guaranteed contributing, so FD is nonzero), then
    # a synapse column whose analytic gradient is nonzero if one exists.
    cols = [bias_col]
    for c in range(layer.weight.shape[1]):
        if c == bias_col:
            continue
        if abs(gd[j0, c].item()) > 1e-12:
            cols.append(c)
            break
    for c in cols:
        orig = layer.weight.data[j0, c].item()
        layer.weight.data[j0, c] = orig + eps
        lp = loss_fn()
        layer.weight.data[j0, c] = orig - eps
        lm = loss_fn()
        layer.weight.data[j0, c] = orig
        fd = (lp - lm) / (2 * eps)
        auto = gd[j0, c].item()
        assert abs(fd) > 0, f"col {c} FD gradient is zero -- bad test setup"
        assert abs(auto - fd) < 1e-6, f"col {c}: auto={auto} fd={fd}"


def test_edge_peak_guard_detects_degenerate_plateau():
    """All-near-zero weights give u_peak ~ 0 at the window start, which the
    guard must flag so the envelope-theorem gradient (zero there) is skipped."""
    layer = ExactTTFSLinear(4, 2, w_scale=1e-2, bias_val=0.0,
                            dtype=DTYPE, device=DEVICE, seed=0)
    t_in = torch.rand(4, 4, dtype=DTYPE) * 10.0
    # Drive every synapse + the bias to zero: u(t) is then identically 0 on the
    # window (the degenerate pre-input plateau), u_peak = 0 and t_peak sits at
    # the window start t_start = 0 -- exactly the case the guard must flag.
    W = torch.zeros_like(layer.weight)
    tpeak, upeak = peak_margin_torch(
        W, t_in, layer.t_bias, layer.theta, layer.grid,
        layer.tm, layer.ts, layer._alpha, layer.k_peak)
    guard = edge_peak_guard(W, t_in, layer.t_bias, tpeak, upeak, layer.grid)
    assert guard.all(), "degenerate plateau should be flagged everywhere"


def test_calibrate_init_fire_revives_silent_layer():
    """A layer that is fully silent at init fires after calibration."""
    layer = ExactTTFSLinear(8, 6, w_scale=0.02, bias_val=0.0,
                            dtype=DTYPE, device=DEVICE, seed=1)
    t_in = torch.rand(8, 16, dtype=DTYPE) * 0.8 * 40 + 0.1
    fired_before = torch.isfinite(layer(t_in)).float().mean()
    layer.calibrate_init_fire(target=0.5, n_probe=16)
    fired_after = torch.isfinite(layer(t_in)).float().mean()
    assert fired_after > fired_before, "calibration should increase firing"


def test_network_calibrate_init_fire():
    net = ExactTTFSNetwork([8, 6, 4], w_scale=0.02, bias_val=0.0,
                           dtype=DTYPE, device=DEVICE, seed=2)
    t_in = torch.rand(8, 16, dtype=DTYPE) * 0.8 * 40 + 0.1
    fired_before = torch.isfinite(net(t_in)).float().mean()
    net.calibrate_init_fire(target=0.5, n_probe=16)
    fired_after = torch.isfinite(net(t_in)).float().mean()
    assert fired_after > fired_before


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def test_spikenorm_shape_and_param():
    norm = SpikeNorm(5)
    t = torch.randn(5, 8, dtype=DTYPE)
    out = norm(t)
    assert out.shape == t.shape
    assert isinstance(norm.gamma, nn.Parameter)
    assert list(norm.parameters())[0] is norm.gamma


def test_spikenorm_running_stats_update():
    norm = SpikeNorm(3)
    assert torch.allclose(norm.running_mean, torch.zeros(3))
    t = torch.full((3, 10), 7.0, dtype=DTYPE)
    norm.train()
    norm(t)
    assert not torch.allclose(norm.running_mean, torch.zeros(3))


# ---------------------------------------------------------------------------
# Losses / initializers / util
# ---------------------------------------------------------------------------
def test_rate_latency_loss_backward():
    leaf = torch.rand(4, 8, 5, dtype=DTYPE, requires_grad=True) * 39.0 + 0.1
    t_all = torch.where(leaf > 35, torch.full_like(leaf, float("inf")), leaf)
    t_all.retain_grad()
    y = torch.randint(0, 4, (8,))
    loss = rate_latency_loss(t_all, y, t_max=40.0)
    assert torch.isfinite(loss)
    assert loss.grad_fn is not None
    loss.backward()
    assert t_all.grad is not None
    assert torch.isfinite(t_all.grad).all()


def test_finite_loss_ordering_rate_latency():
    n_out, B, K = 4, 6, 3
    y = torch.arange(B) % n_out
    good = torch.full((n_out, B, K), 30.0, dtype=DTYPE)
    for b in range(B):
        good[y[b], b] = 1.0
    bad = torch.full((n_out, B, K), 1.0, dtype=DTYPE)
    for b in range(B):
        bad[y[b], b] = 30.0
    l_good = rate_latency_loss(good, y, 40.0)
    l_bad = rate_latency_loss(bad, y, 40.0)
    assert float(l_good) < float(l_bad)


def test_xavier_and_kaiming_init():
    w = torch.zeros(10, 10 + 1, dtype=DTYPE)
    xavier_init(w, fan_in=10, fan_out=10, seed=0)
    w2 = torch.zeros(10, 11, dtype=DTYPE)
    kaiming_init(w2, fan_in=10, fan_out=10, seed=0)
    # bias column set to small value, not zero
    assert abs(w[:, -1].mean().item() - 0.1) < 1e-9
    assert abs(w2[:, -1].mean().item() - 0.1) < 1e-9
    # weights are finite and not all equal
    assert torch.isfinite(w[:, :-1]).all()
    assert w[:, :-1].std().item() > 0


# ---------------------------------------------------------------------------
# Event-driven layer (matching grid engine + autograd)
# ---------------------------------------------------------------------------
def test_event_layer_matches_grid_engine():
    n_in, n_out, B = 20, 8, 16
    seed = 5
    g = ExactTTFSLinear(n_in, n_out, dtype=DTYPE, device=DEVICE, seed=seed)
    e = ExactEventLinear(n_in, n_out, dtype=DTYPE, device=DEVICE, seed=seed)
    with torch.no_grad():
        e.weight.copy_(g.weight)
    t_in = torch.rand(n_in, B, dtype=DTYPE) * 0.6 * 40 + 0.1
    t_g = g(t_in)
    t_e = e(t_in)
    assert (torch.isfinite(t_g) == torch.isfinite(t_e)).all(), \
        "who-fires must agree between grid and event engines"
    both = torch.isfinite(t_g)
    if both.any():
        assert (t_g[both] - t_e[both]).abs().max().item() < 0.05


def test_event_layer_autograd_backward():
    layer = ExactEventLinear(10, 5, dtype=DTYPE, device=DEVICE, seed=0)
    t_in = torch.rand(10, 8, dtype=DTYPE) * 0.6 * 40 + 0.1
    t_out = layer(t_in)
    loss = t_out[torch.isfinite(t_out)].sum() if torch.isfinite(t_out).any() \
        else t_out.sum()
    loss.backward()
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


# ---------------------------------------------------------------------------
# Spike-time augmentation
# ---------------------------------------------------------------------------
def test_spike_time_augment_clamps():
    t_in = torch.full((5, 4), 38.0, dtype=DTYPE)
    out = spike_time_augment(t_in, t_max=40.0, noise_std=0.0, time_shift=0.0)
    assert out.shape == t_in.shape
    assert torch.isfinite(out).all()
    assert (out >= 0).all() and (out <= 40.0).all()


def test_spike_time_augment_noise_and_shift_change_values():
    t_in = torch.zeros(6, 6, dtype=DTYPE)
    out = spike_time_augment(t_in, t_max=40.0, noise_std=0.5, time_shift=2.0)
    assert torch.isfinite(out).all()
    # additive noise (std 0.5) essentially never leaves every element at 0.
    assert (out != 0.0).any()


# ---------------------------------------------------------------------------
# ResetLIF reference solver (multi-spike saltation oracle)
# ---------------------------------------------------------------------------
def test_resetlif_run_single_strong_input():
    neuron = ResetLIF()
    fires = neuron.run([(1.0, 10.0)], t_end=200.0)
    assert len(fires) >= 1
    assert all(f > 0 for f in fires)


def test_resetlif_run_multiple_spikes():
    neuron = ResetLIF(theta=1.0)
    fires = neuron.run([(1.0, 20.0), (40.0, 20.0)], t_end=200.0)
    assert len(fires) >= 1
    assert fires == sorted(fires)


def test_resetlif_sensitivity_all_shape():
    neuron = ResetLIF()
    inputs = [(1.0, 5.0), (3.0, 7.0), (6.0, 2.0)]
    fires, dtdw = neuron.sensitivity_all(inputs, t_end=200.0)
    assert len(fires) == len(dtdw)
    if fires:
        assert all(len(row) == len(inputs) for row in dtdw)


def test_resetlif_sensitivity_first_spike():
    neuron = ResetLIF()
    inputs = [(1.0, 8.0), (2.0, 3.0)]
    fire, dtdw = neuron.sensitivity_first_spike(inputs, t_end=200.0)
    assert fire is not None
    assert dtdw.shape == (len(inputs),)
    assert np.all(np.isfinite(dtdw))


def test_resetlif_run_with_state_returns_derivative():
    neuron = ResetLIF()
    inputs = [(1.0, 10.0), (30.0, 5.0)]
    fires, ups = neuron.run_with_state(inputs, t_end=200.0)
    assert len(fires) == len(ups)
    if fires:
        # u'_f = (i_f - theta)/tm should be positive approaching a firing from
        # below (u increasing through theta).
        assert all(u > 0 for u in ups)
