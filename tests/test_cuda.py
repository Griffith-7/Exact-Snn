"""Tests for the optional CUDA root-solve backend (exact_snn.cuda_ops).

The CUDA extension fuses the grid scan + bisection + Newton + peak search of
``_forward_layer_torch`` into a single kernel. These tests pin it to the torch
path: identical firing masks, spike times and membrane slopes within float32
precision, plus end-to-end autograd parity through ``_ExactTTFSLayerFn``.

Skip conditions: no CUDA device, or the JIT extension could not be built
(e.g. no nvcc / no MSVC). The torch path is the always-available fallback and
is tested in the other test modules.

Run:  python -m pytest tests/test_cuda.py -v   (from the project root)
"""
from __future__ import annotations

import pytest
import torch

from exact_snn import (
    ExactTTFSLinear,
    _ExactTTFSLayerFn,
    _backward_layer_torch,
    _forward_layer_torch,
)
from exact_snn import cuda_ops

CUDA = torch.cuda.is_available() and cuda_ops.available()

pytestmark = pytest.mark.skipif(
    not CUDA, reason="CUDA extension not buildable / no CUDA device")


def _config(seed=0, G=2001):
    torch.manual_seed(seed)
    tm, ts, theta = 15.0, 4.0, 1.0
    alpha = abs(tm - ts) < 1e-9
    k_peak = ExactTTFSLinear._compute_k_peak(tm, ts)
    return dict(tm=tm, ts=ts, theta=theta, alpha=alpha, k_peak=k_peak,
                t_bias=0.0, G=G)


def _forward_both(W, t_prev, cfg):
    grid = torch.linspace(0, 40.0, cfg["G"], dtype=W.dtype, device=W.device)
    t_t, up_t = _forward_layer_torch(
        W, t_prev, cfg["t_bias"], cfg["theta"], grid, cfg["tm"], cfg["ts"],
        cfg["alpha"], cfg["k_peak"], n_bisect=15, n_newton=8, peak_tol=1e-2)
    t_c, up_c = cuda_ops.cuda_forward(
        W, t_prev, cfg["t_bias"], cfg["theta"], grid, cfg["tm"], cfg["ts"],
        cfg["alpha"], cfg["k_peak"], n_bisect=15, n_newton=8, peak_tol=1e-2)
    return t_t, up_t, t_c, up_c


class TestForwardParity:
    def test_fp32_dense_firing(self):
        cfg = _config(seed=0)
        W = torch.randn(8, 128 + 1, dtype=torch.float32, device="cuda") * 0.2
        W[:, -1] = torch.tensor([9.0, 8.0, 7.0, 6.0, 5.0, 4.5, 4.0, 3.2],
                                dtype=torch.float32, device="cuda")
        t_prev = torch.rand(128 + 1, 64, dtype=torch.float32, device="cuda") * 40.0
        t_t, up_t, t_c, up_c = _forward_both(W, t_prev, cfg)
        assert torch.equal(torch.isfinite(t_t), torch.isfinite(t_c))
        both = torch.isfinite(t_t)
        assert both.any()
        torch.testing.assert_close(t_c[both], t_t[both], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(up_c[both], up_t[both], atol=1e-5, rtol=1e-4)

    def test_fp32_coarse_grid_with_peak_search(self):
        """G=65 misses most crossings vs the fine grid; exercises the golden
        peak search branch. Firing masks must still match the torch path."""
        for G in (65,):
            cfg = _config(seed=7)
            cfg["G"] = G
            W = torch.randn(16, 64 + 1, dtype=torch.float32, device="cuda") * 0.10
            W[:, -1] = 1.4
            t_prev = (0.05 + torch.rand(64 + 1, 64,
                                        dtype=torch.float32, device="cuda")
                      * 0.9 * 40.0)
            t_t, up_t, t_c, up_c = _forward_both(W, t_prev, cfg)
            assert torch.equal(torch.isfinite(t_t), torch.isfinite(t_c))
            assert int(torch.isfinite(t_t).sum()) > 0
            assert int((~torch.isfinite(t_t)).sum()) > 0, "need silent neurons"
            both = torch.isfinite(t_t)
            torch.testing.assert_close(t_c[both], t_t[both], atol=5e-5, rtol=1e-4)

    def test_fp64_direct_kernel_parity(self):
        """The kernel is templated for float32/float64; float64 must be
        essentially exact (no float32 rounding)."""
        cfg = _config(seed=2)
        W = torch.randn(6, 32 + 1, dtype=torch.float64, device="cuda") * 0.2
        W[:, -1] = 2.0
        t_prev = torch.rand(32 + 1, 16, dtype=torch.float64, device="cuda") * 40.0
        t_t, up_t, t_c, up_c = _forward_both(W, t_prev, cfg)
        assert torch.equal(torch.isfinite(t_t), torch.isfinite(t_c))
        both = torch.isfinite(t_t)
        torch.testing.assert_close(t_c[both], t_t[both], atol=1e-9, rtol=1e-8)
        torch.testing.assert_close(up_c[both], up_t[both], atol=1e-9, rtol=1e-8)

    def test_alpha_exp_kernel_parity(self):
        cfg = _config(seed=5)
        cfg["tm"] = cfg["ts"] = 6.0          # tm == ts -> alpha kernel
        cfg["alpha"] = True
        cfg["k_peak"] = 1.0
        W = torch.randn(6, 32 + 1, dtype=torch.float64, device="cuda") * 0.2
        W[:, -1] = 2.5
        t_prev = torch.rand(32 + 1, 16, dtype=torch.float64, device="cuda") * 40.0
        t_t, up_t, t_c, up_c = _forward_both(W, t_prev, cfg)
        assert torch.equal(torch.isfinite(t_t), torch.isfinite(t_c))
        both = torch.isfinite(t_t)
        torch.testing.assert_close(t_c[both], t_t[both], atol=1e-9, rtol=1e-8)

    def test_silent_neurons_zero_gradient_source(self):
        cfg = _config(seed=11, G=65)
        W = torch.zeros(8, 32 + 1, dtype=torch.float64, device="cuda")
        W[:, -1] = 0.5                      # peak bias*K -- well below theta
        t_prev = torch.rand(32 + 1, 8, dtype=torch.float64, device="cuda") * 40.0
        t_t, up_t, t_c, up_c = _forward_both(W, t_prev, cfg)
        assert not torch.isfinite(t_t).any()
        assert not torch.isfinite(t_c).any()
        assert (up_c == 0).all() and (up_t == 0).all()


class TestBackwardParity:
    def test_grad_matches_torch_componentwise(self):
        cfg = _config(seed=3, G=2001)
        W = torch.randn(8, 64 + 1, dtype=torch.float32, device="cuda") * 0.2
        W[:, -1] = 3.0
        t_prev = torch.rand(64 + 1, 32, dtype=torch.float32, device="cuda") * 40.0
        lam = torch.randn(8, 32, dtype=torch.float32, device="cuda")
        t_t, up_t, t_c, up_c = _forward_both(W, t_prev, cfg)
        g_t, lp_t = _backward_layer_torch(
            W, t_prev, cfg["t_bias"], t_t, lam, up_t, cfg["tm"], cfg["ts"],
            cfg["alpha"], cfg["k_peak"])
        g_c, lp_c = _backward_layer_torch(
            W, t_prev, cfg["t_bias"], t_c, lam, up_c, cfg["tm"], cfg["ts"],
            cfg["alpha"], cfg["k_peak"])
        torch.testing.assert_close(g_c, g_t, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(lp_c, lp_t, atol=1e-5, rtol=1e-4)


class TestAutogradDispatch:
    def test_layerfn_fp32_cuda_dispatch_parity(self):
        cfg = _config(seed=17)
        W = torch.randn(8, 64 + 1, dtype=torch.float32, device="cuda") * 0.2
        W[:, -1] = 3.0
        t_prev = torch.rand(64 + 1, 16, dtype=torch.float32, device="cuda") * 40.0
        grid = torch.linspace(0, 40.0, cfg["G"], dtype=torch.float32, device="cuda")

        was = cuda_ops.is_enabled()
        try:
            t_cuda = _ExactTTFSLayerFn.apply(
                W, t_prev, cfg["t_bias"], cfg["theta"], cfg["tm"], cfg["ts"],
                cfg["alpha"], cfg["k_peak"], grid, 1e-2)
            cuda_ops.set_enabled(False)
            t_torch = _ExactTTFSLayerFn.apply(
                W, t_prev, cfg["t_bias"], cfg["theta"], cfg["tm"], cfg["ts"],
                cfg["alpha"], cfg["k_peak"], grid, 1e-2)
        finally:
            cuda_ops.set_enabled(was)
        torch.testing.assert_close(t_cuda, t_torch, atol=1e-5, rtol=1e-4)

    def test_grad_backward_dispatched_forward(self):
        cfg = _config(seed=29)
        W = torch.randn(6, 32 + 1, dtype=torch.float32, device="cuda") * 0.2
        W[:, -1] = 2.5
        t_prev = torch.rand(32 + 1, 12, dtype=torch.float32, device="cuda") * 40.0
        grid = torch.linspace(0, 40.0, cfg["G"], dtype=torch.float32, device="cuda")
        W.requires_grad_(True)

        out = _ExactTTFSLayerFn.apply(
            W, t_prev, cfg["t_bias"], cfg["theta"], cfg["tm"], cfg["ts"],
            cfg["alpha"], cfg["k_peak"], grid, 1e-2)
        loss = (out[torch.isfinite(out)] / 40.0).sum()
        loss.backward()
        g_cuda = W.grad.clone()

        W.grad = None
        was = cuda_ops.is_enabled()
        try:
            cuda_ops.set_enabled(False)
            out = _ExactTTFSLayerFn.apply(
                W, t_prev, cfg["t_bias"], cfg["theta"], cfg["tm"], cfg["ts"],
                cfg["alpha"], cfg["k_peak"], grid, 1e-2)
            loss = (out[torch.isfinite(out)] / 40.0).sum()
            loss.backward()
            g_torch = W.grad.clone()
        finally:
            cuda_ops.set_enabled(was)
        torch.testing.assert_close(g_cuda, g_torch, atol=1e-5, rtol=1e-5)

    def test_module_forward_backward_on_cuda(self):
        torch.manual_seed(0)
        layer = ExactTTFSLinear(
            64, 16, tm=15.0, ts=4.0, theta=1.0, t_max=40.0, grid_pts=2001,
            dtype=torch.float32, device="cuda")
        t_in = (0.05 + torch.rand(64, 8, dtype=torch.float32, device="cuda")
                * 0.9 * 40.0)
        out = layer(t_in)
        loss = out[torch.isfinite(out)].mean()
        if torch.isfinite(out).any():
            loss.backward()
            assert layer.weight.grad is not None
            assert torch.isfinite(layer.weight.grad).all()
        assert torch.isfinite(out[torch.isfinite(out)]).all()