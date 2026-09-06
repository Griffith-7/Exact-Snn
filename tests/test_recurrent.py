"""Tests for the exact recurrent cell (ExactTTFSRnn, NBTT IFT gradients).

Core checks: the backprop-through-time gradient of the full unrolled rollout
matches central finite differences in the weight tensor, the input sequence,
and the initial hidden state (i.e. the recurrence Jacobian is exact), and
silent hidden cells contribute nothing to the next step.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exact_snn.recurrent import ExactTTFSRnn

DEVICE = "cpu"
DTYPE = torch.float64


def _make_cell(n_in=2, n_hidden=3, w_scale=0.12, bias_val=1.2, seed=0,
               **kw):
    kw.setdefault("dtype", DTYPE)
    kw.setdefault("device", DEVICE)
    return ExactTTFSRnn(n_in, n_hidden, w_scale=w_scale, bias_val=bias_val,
                        seed=seed, **kw)


def _seq(B=4, T=3, n_in=2, seed=1):
    torch.manual_seed(seed)
    t_in = torch.rand(n_in, B, T, dtype=DTYPE, device=DEVICE) * 0.7 * 40 + 0.1
    return t_in


def _finite_mean(o):
    f = torch.isfinite(o)
    if not f.any():
        return torch.tensor(0.0, dtype=DTYPE)
    return o[f].abs().sum() / f.float().sum()


def _rollout_loss(cell, t_in, h0):
    o = cell(t_in, h0)
    return _finite_mean(o)


def _det_cell():
    """Deterministic 1-neuron recurrent cell configured so both steps fire on
    steep crossings (u'~1.5-2.1): the exact IFT gradient is then measurable
    against central finite differences even though the forward solver's spike
    times quantize below ~1e-7 (use a generous FD step)."""
    c = ExactTTFSRnn(1, 1, w_scale=0.1, bias_val=0.2, dtype=DTYPE,
                     device=DEVICE, seed=0, tm=8, ts=2)
    with torch.no_grad():
        c.weight.data = torch.tensor([[2.5, 0.8, 0.2]], dtype=DTYPE)
    return c


def _det_data():
    tin = torch.tensor([[[3.0, 5.0]]], dtype=DTYPE, device=DEVICE)  # (1,1,2)
    h0 = torch.tensor([[1.0]], dtype=DTYPE, device=DEVICE)          # (1,1)
    return tin, h0


def _det_loss(c, tin, h0):
    o = c(tin, h0)
    return o[0, 0, 0] + o[0, 0, 1]


def _fd_grad(ref, fn, eps=1e-3):
    """Central finite-difference gradient by perturbing `ref` storage."""
    flat = ref.detach().flatten()
    g = torch.zeros_like(flat)
    with torch.no_grad():
        for i in range(flat.numel()):
            v0 = flat[i].item()
            flat[i] = v0 + eps
            lp = fn()
            flat[i] = v0 - eps
            lm = fn()
            flat[i] = v0
            g[i] = (lp - lm) / (2 * eps)
    return g.reshape(ref.shape)


def _assert_fd_matches(an, fd):
    err = (an.detach() - fd).abs()
    tol = 0.05 * (1.0 + an.detach().abs())
    assert (err <= tol).all(), f"max FD deviation {err.max().item()}"


def test_rnn_build_and_parameters():
    cell = _make_cell()
    assert isinstance(cell.weight, nn.Parameter)
    assert tuple(cell.weight.shape) == (3, 2 + 3 + 1)
    assert list(cell.parameters())[0] is cell.weight


def test_rnn_rollout_shape_and_cold_start():
    cell = _make_cell()
    t_in = _seq()
    o = cell(t_in)
    assert o.shape == (3, 4, 3)
    # cold start must let some cells fire (no inf bleed into step 0)
    assert torch.isfinite(o).any()


def test_rnn_backward_populates_all_params():
    cell = _make_cell()
    t_in = _seq(seed=3)
    loss = _rollout_loss(cell, t_in, None)
    assert loss.requires_grad
    loss.backward()
    assert cell.weight.grad is not None
    assert torch.isfinite(cell.weight.grad).all()


def test_rnn_weight_grad_matches_finite_differences():
    cell = _det_cell()
    tin, h0 = _det_data()
    tin.requires_grad_(True)
    h0.requires_grad_(True)
    loss = _det_loss(cell, tin, h0)
    loss.backward()
    g_fd = _fd_grad(cell.weight, lambda: _det_loss(cell, tin.detach(), h0.detach()))
    _assert_fd_matches(cell.weight.grad, g_fd)


def test_rnn_input_grad_matches_finite_differences():
    cell = _det_cell()
    tin, h0 = _det_data()
    tin.requires_grad_(True)
    h0.requires_grad_(True)
    loss = _det_loss(cell, tin, h0)
    loss.backward()
    g_fd = _fd_grad(tin, lambda: _det_loss(cell, tin.detach(), h0.detach()))
    _assert_fd_matches(tin.grad, g_fd)


def test_rnn_state_grad_matches_finite_differences():
    """The recurrence Jacobian (d loss / d h0 through T steps) is exact. h0=1
    is an early 'past spike' that genuinely shapes the step-0 output, which in
    turn shapes step 1, so the gradient flows through the recurrence."""
    cell = _det_cell()
    tin, h0 = _det_data()
    tin.requires_grad_(True)
    h0.requires_grad_(True)
    loss = _det_loss(cell, tin, h0)
    loss.backward()
    g_fd = _fd_grad(h0, lambda: _det_loss(cell, tin.detach(), h0.detach()))
    _assert_fd_matches(h0.grad, g_fd)


def test_rnn_silent_hidden_contributes_nothing():
    """A silent hidden cell (inf) receives a zero gradient from the next step:
    K(t - inf) = 0 makes its only contribution to the next membrane vanish."""
    torch.manual_seed(0)
    cell = _make_cell()
    t_in = _seq(seed=11)
    o1 = cell.forward_step(t_in[:, :, 0], torch.full(
        (3, 4), float("inf"), dtype=DTYPE))
    silent = torch.where(torch.isfinite(o1), o1,
                         torch.full_like(o1, float("inf"))).detach()
    silent[2] = float("inf")
    silent.requires_grad_(True)
    o2 = cell.forward_step(t_in[:, :, 1], silent)
    out = o2[1].abs().sum()  # gradient target: row 1 of the output
    out.backward()
    assert torch.isfinite(silent.grad).all()
    assert silent.grad[2].abs().max().item() == 0.0


def test_rnn_guards_nan_inputs():
    """NaN spike times are corrupt and rejected at the public API."""
    cell = _make_cell(seed=0)
    nan_seq = torch.full((2, 4, 3), float("nan"), dtype=DTYPE, device=DEVICE)
    with pytest.raises(ValueError, match="must not contain NaN"):
        cell(nan_seq)
    nan_step = torch.full((2, 4), float("nan"), dtype=DTYPE, device=DEVICE)
    ok_prev = torch.full((3, 4), float("inf"), dtype=DTYPE, device=DEVICE)
    with pytest.raises(ValueError, match="must not contain NaN"):
        cell.forward_step(nan_step, ok_prev)
    ok_in = torch.full((2, 4), float("inf"), dtype=DTYPE, device=DEVICE)
    nan_prev = torch.full((3, 4), float("nan"), dtype=DTYPE, device=DEVICE)
    with pytest.raises(ValueError, match="must not contain NaN"):
        cell.forward_step(ok_in, nan_prev)

def test_rnn_shorter_rollout_trains():
    """A tiny seq-to-seq task engages the exact NBTT gradients: the loss falls
    under Adam from a fixed init."""
    torch.manual_seed(0)
    cell = ExactTTFSRnn(2, 3, w_scale=0.4, bias_val=1.0,
                        dtype=torch.float32, seed=0)
    opt = torch.optim.Adam(cell.parameters(), lr=5e-3)
    t_in = (torch.rand(2, 8, 4, dtype=torch.float32) * 0.7 * 40 + 0.1)
    first = None
    last = None
    for _ in range(25):
        opt.zero_grad()
        loss = _finite_mean(cell(t_in))
        loss.backward()
        if cell.weight.grad is not None and torch.isfinite(
                cell.weight.grad).all():
            opt.step()
        first = first if first is not None else float(loss.item())
        last = float(loss.item())
    assert last < first, "exact recurrent training must reduce the loss"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])