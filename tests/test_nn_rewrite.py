"""Tests for the nn.Module (autograd) rewrite of Exact-SNN.

Verifies the key plug-and-play property: everything works with the STANDARD
PyTorch API — torch.optim, loss.backward(), model.parameters() — and that the
autograd gradients match finite differences.

Run:  python -m pytest tests/ -v    (from the project root)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from exact_snn import (
    ExactTTFSLinear,
    ExactTTFSNetwork,
    latency_encode,
    latency_cross_entropy,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def _make_net(sizes=(784, 64, 10), **kwargs):
    defaults = dict(tm=15.0, ts=4.0, theta=1.0, t_max=40.0, w_scale=0.2,
                    bias_val=1.5, seed=42, dtype=DTYPE, device=DEVICE, grid_pts=2001)
    defaults.update(kwargs)
    return ExactTTFSNetwork(list(sizes), **defaults)


class TestAutogradIntegration:
    def test_invalid_configuration_is_rejected(self):
        with pytest.raises(ValueError, match="n_in and n_out"):
            ExactTTFSLinear(0, 5)
        with pytest.raises(ValueError, match="must be positive"):
            ExactTTFSLinear(5, 5, tm=0.0)
        with pytest.raises(ValueError, match="at least 3"):
            ExactTTFSLinear(5, 5, grid_pts=2)

    def test_input_dtype_is_checked(self):
        layer = ExactTTFSLinear(3, 2, dtype=DTYPE, device=DEVICE)
        integer_input = torch.ones(3, 2, dtype=torch.int64, device=DEVICE)
        with pytest.raises(ValueError, match="floating-point dtype"):
            layer(integer_input)
        wrong_dtype = torch.ones(3, 2, dtype=torch.float32, device=DEVICE)
        if DTYPE != torch.float32:
            with pytest.raises(ValueError, match="Input dtype"):
                layer(wrong_dtype)

    def test_weights_are_parameters(self):
        layer = ExactTTFSLinear(10, 5, dtype=DTYPE, device=DEVICE)
        assert isinstance(layer.weight, torch.nn.Parameter)
        # bias is the last column, not a separate parameter
        assert layer.weight.shape == (5, 11)
        # total params exposed through .parameters()
        nparams = sum(p.numel() for p in layer.parameters())
        assert nparams == 5 * 11

    def test_backward_populates_grad(self):
        net = _make_net(sizes=(10, 5, 3))
        t_in = torch.rand(10, 8, dtype=DTYPE, device=DEVICE) * 0.8 * 40.0 + 0.1
        y = torch.randint(0, 3, (8,), device=DEVICE)
        loss = net.loss(t_in, y)
        loss.backward()
        for layer in net.layers:
            assert layer.weight.grad is not None
            assert torch.isfinite(layer.weight.grad).all()

    def test_torch_optim_adam_works(self):
        """The whole point: standard torch.optim works with loss.backward()."""
        net = _make_net(sizes=(10, 5, 3))
        t_in = torch.rand(10, 8, dtype=DTYPE, device=DEVICE) * 0.8 * 40.0 + 0.1
        y = torch.randint(0, 3, (8,), device=DEVICE)

        opt = torch.optim.Adam(net.parameters(), lr=0.01)
        before = [p.clone() for p in net.parameters()]

        loss = net.loss(t_in, y)
        loss.backward()
        opt.step()

        for b, a in zip(before, net.parameters()):
            assert not torch.equal(b, a), "Adam did not update weights"

    def test_gradient_cosine_vs_fd(self):
        """Autograd IFT gradient should match finite differences ~1.0 on smooth weights."""
        net = _make_net(sizes=(10, 5, 3))
        g = torch.Generator().manual_seed(42)
        t_in = (torch.rand(10, 8, generator=g, dtype=DTYPE) * 0.8 * net.t_max + 0.1).to(DEVICE)
        y = torch.randint(0, 3, (8,), generator=g).to(DEVICE)

        def loss_fn():
            return net.loss(t_in, y)

        # Autograd gradient of layer 0
        loss = loss_fn()
        loss.backward()
        g_auto = net.layers[0].weight.grad.clone()

        # Finite difference (central) on layer 0 weight
        W = net.layers[0].weight
        eps = 1e-5
        g_fd = torch.zeros_like(W)
        smooth = torch.ones_like(W, dtype=torch.bool)
        loss0 = float(loss_fn().detach())
        rows, cols = W.shape
        for i in range(rows):
            for j in range(cols):
                orig = W[i, j].item()
                with torch.no_grad():
                    W[i, j] = orig + eps
                lp, _ = loss_fn(), None
                lp = float(loss_fn().detach())
                with torch.no_grad():
                    W[i, j] = orig - eps
                lm = float(loss_fn().detach())
                with torch.no_grad():
                    W[i, j] = orig
                d_r = (lp - loss0) / eps
                d_l = (loss0 - lm) / eps
                g_fd[i, j] = (lp - lm) / (2 * eps)
                if abs(g_fd[i, j]) > 1e-12 and abs(d_r - d_l) > 0.05 * abs(g_fd[i, j]):
                    smooth[i, j] = False
                elif abs(g_fd[i, j]) <= 1e-12 and abs(d_r - d_l) > 1e-6:
                    smooth[i, j] = False

        assert smooth.any(), "No smooth weights found"
        a = g_auto[smooth].float().detach().cpu().numpy().ravel()
        b = g_fd[smooth].float().detach().cpu().numpy().ravel()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cos > 0.99, f"Cosine too low: {cos:.6f}"


class TestFunctional:
    def test_latency_encode(self):
        x = torch.tensor([[1.0, 0.0], [0.5, 0.5]], dtype=DTYPE)
        t = latency_encode(x, t_max=40.0)
        # bright -> early
        assert t[0, 0] < t[0, 1]
        assert 0 < float(t[0, 0]) < float(t[0, 1])

    def test_loss_decreases_with_earlier_correct_class(self):
        n_out, B = 5, 8
        y = torch.arange(B, device=DEVICE) % n_out
        t_good = torch.ones(n_out, B, dtype=DTYPE, device=DEVICE) * 30.0
        for b in range(B):
            t_good[y[b], b] = 1.0
        t_bad = torch.ones(n_out, B, dtype=DTYPE, device=DEVICE) * 1.0
        for b in range(B):
            t_bad[y[b], b] = 30.0
        l_good = latency_cross_entropy(t_good, y, 40.0)
        l_bad = latency_cross_entropy(t_bad, y, 40.0)
        assert float(l_good) < float(l_bad)


class TestPlugAndPlayPattern:
    def test_sequential_model_like_standard_pytorch(self):
        """Show the exact usage a user would expect from a PyTorch library."""
        model = torch.nn.Sequential(
            ExactTTFSLinear(784, 128, dtype=DTYPE, device=DEVICE),
            ExactTTFSLinear(128, 10, dtype=DTYPE, device=DEVICE),
        )
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        t_in = torch.rand(784, 16, dtype=DTYPE, device=DEVICE) * 0.8 * 40.0 + 0.1
        y = torch.randint(0, 10, (16,), device=DEVICE)
        t_out = model(t_in)
        loss = latency_cross_entropy(t_out, y, 40.0)
        loss.backward()
        opt.step()
        # All parameters got gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) == 2
        assert t_out.shape == (10, 16)
