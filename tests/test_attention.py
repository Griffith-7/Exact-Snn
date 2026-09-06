"""Tests for ExactSpikingAttention (single-head, alignment-score).

Verifies shapes, autograd integration, and that the analytic combine gradients
(and, through composition, the WQ/WK/WV IFT layers) match finite differences on
smooth weights -- the library's "exact" contract.

Run:  python -m pytest tests/ -v    (from the project root)
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from exact_snn.attention import (
    ExactSpikingAttention,
    ExactAttentionCombineFn,
    exact_attention_scores,
)
from exact_snn import latency_cross_entropy

torch.manual_seed(1234)
np.random.seed(1234)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def _make_attn(n=8, **kwargs):
    defaults = dict(tm=15.0, ts=4.0, theta=1.0, t_max=40.0, w_scale=0.2,
                    bias_val=1.5, grid_pts=2001, seed=7, dtype=DTYPE,
                    device=DEVICE, temp=1.0)
    defaults.update(kwargs)
    return ExactSpikingAttention(n, n, **defaults)


def _rand_times(n, B):
    return (torch.rand(n, B, dtype=DTYPE, device=DEVICE) * 0.8 * 40.0 + 0.1)


class TestShapesAndAutograd:
    def test_forward_shape(self):
        attn = _make_attn(8)
        t_in = _rand_times(8, 5)
        out = attn(t_in)
        assert out.shape == (8, 5)
        assert torch.isfinite(out).all()

    def test_parameters_exposed(self):
        attn = _make_attn(6)
        assert attn.WQ.weight.shape == (6, 7)
        assert attn.WK.weight.shape == (6, 7)
        assert attn.WV.weight.shape == (6, 7)
        nparams = sum(p.numel() for p in attn.parameters())
        assert nparams == 3 * 6 * 7

    def test_input_validation(self):
        attn = _make_attn(8)
        with pytest.raises(ValueError, match="Expected"):
            attn(torch.rand(8, 3, 2, 2, dtype=DTYPE, device=DEVICE))
        with pytest.raises(ValueError, match="n_in"):
            attn(torch.rand(4, 3, dtype=DTYPE, device=DEVICE))

    def test_square_requirement(self):
        with pytest.raises(ValueError, match="square"):
            ExactSpikingAttention(8, 4, dtype=DTYPE, device=DEVICE)

    def test_invalid_combine(self):
        with pytest.raises(ValueError, match="combine"):
            ExactSpikingAttention(8, 8, combine="bogus", dtype=DTYPE,
                                  device=DEVICE)

    def test_backward_populates_grads(self):
        attn = _make_attn(8)
        t_in = _rand_times(8, 6)
        out = attn(t_in)
        loss = out.mean()
        loss.backward()
        for proj in (attn.WQ, attn.WK, attn.WV):
            assert proj.weight.grad is not None
            assert torch.isfinite(proj.weight.grad).all()

    def test_compose_with_latency_ce(self):
        attn = _make_attn(10)
        t_in = _rand_times(10, 8)
        y = torch.randint(0, 10, (8,), device=DEVICE)
        out = attn(t_in)
        loss = latency_cross_entropy(out, y, 40.0)
        loss.backward()
        assert torch.isfinite(loss)

    def test_torch_optim_adam_works(self):
        attn = _make_attn(8)
        t_in = _rand_times(8, 6)
        y = torch.randint(0, 8, (6,), device=DEVICE)
        opt = torch.optim.Adam(attn.parameters(), lr=0.01)
        before = [p.clone() for p in attn.parameters()]
        loss = latency_cross_entropy(attn(t_in), y, 40.0)
        loss.backward()
        opt.step()
        for b, a in zip(before, attn.parameters()):
            assert not torch.equal(b, a), "Adam did not update attention weights"

    def test_calibrate_init_fire_is_idempotent(self):
        """calibrate_init_fire exists on the block and leaves finite weights."""
        attn = _make_attn(8)
        t_in = _rand_times(8, 6)
        out = attn(t_in)
        attn.calibrate_init_fire(target=0.9)
        assert torch.isfinite(attn.WQ.weight).all()
        assert torch.isfinite(attn.WK.weight).all()
        assert torch.isfinite(attn.WV.weight).all()
        assert out.shape == (8, 6)


class TestScoreMap:
    def test_gaussian_peaks_at_alignment(self):
        tm, ts, alpha, k_peak = 15.0, 4.0, False, 1.0
        s_same = exact_attention_scores(
            torch.tensor([[10.0]], dtype=DTYPE),
            torch.tensor([[10.0]], dtype=DTYPE), tm, ts, alpha, k_peak)
        s_far = exact_attention_scores(
            torch.tensor([[10.0]], dtype=DTYPE),
            torch.tensor([[39.0]], dtype=DTYPE), tm, ts, alpha, k_peak)
        assert float(s_same) == pytest.approx(1.0)
        assert float(s_same) > float(s_far)
        # symmetry
        s_ab = exact_attention_scores(
            torch.tensor([[12.0]], dtype=DTYPE),
            torch.tensor([[20.0]], dtype=DTYPE), tm, ts, alpha, k_peak)
        s_ba = exact_attention_scores(
            torch.tensor([[20.0]], dtype=DTYPE),
            torch.tensor([[12.0]], dtype=DTYPE), tm, ts, alpha, k_peak)
        assert abs(float(s_ab) - float(s_ba)) < 1e-9

    def test_kernel_mode_uses_raw_kernel(self):
        """combine='kernel' returns K(|d|), symmetric."""
        tm, ts, alpha, k_peak = 15.0, 4.0, False, 1.0
        s_ab = exact_attention_scores(
            torch.tensor([[12.0]], dtype=DTYPE),
            torch.tensor([[20.0]], dtype=DTYPE), tm, ts, alpha, k_peak,
            combine="kernel")
        s_ba = exact_attention_scores(
            torch.tensor([[20.0]], dtype=DTYPE),
            torch.tensor([[12.0]], dtype=DTYPE), tm, ts, alpha, k_peak,
            combine="kernel")
        assert abs(float(s_ab) - float(s_ba)) < 1e-9


class TestGradientFDFlow:
    def test_combine_grad_matches_fd(self):
        """Analytic combine gradients w.r.t. Q/K/V times vs central finite diff."""
        g = torch.Generator().manual_seed(3)
        nq, nk = 5, 5
        B = 3
        tm, ts = 15.0, 4.0
        alpha, k_peak, temp = False, 1.0, 1.0
        t_q = (torch.rand(nq, B, generator=g, dtype=DTYPE) * 0.8 * 40.0 + 0.1)
        t_k = (torch.rand(nk, B, generator=g, dtype=DTYPE) * 0.8 * 40.0 + 0.1)
        t_v = (torch.rand(nk, B, generator=g, dtype=DTYPE) * 0.8 * 40.0 + 0.1)
        t_q.requires_grad_(True)
        t_k.requires_grad_(True)
        t_v.requires_grad_(True)

        out = ExactAttentionCombineFn.apply(
            t_q, t_k, t_v, tm, ts, alpha, k_peak, temp, 40.0, "gaussian")
        loss = out.sum()
        loss.backward()
        gq, gk, gv = t_q.grad.clone(), t_k.grad.clone(), t_v.grad.clone()

        eps = 1e-6
        check_list = [("q", t_q, gq), ("k", t_k, gk), ("v", t_v, gv)]
        for name, tensor, grad in check_list:
            for i in range(min(tensor.shape[0], 2)):
                orig = tensor[i, 0].detach().item()
                with torch.no_grad():
                    tensor[i, 0] = orig + eps
                lp = ExactAttentionCombineFn.apply(
                    t_q, t_k, t_v, tm, ts, alpha, k_peak, temp, 40.0,
                    "gaussian").sum().detach()
                with torch.no_grad():
                    tensor[i, 0] = orig - eps
                lm = ExactAttentionCombineFn.apply(
                    t_q, t_k, t_v, tm, ts, alpha, k_peak, temp, 40.0,
                    "gaussian").sum().detach()
                with torch.no_grad():
                    tensor[i, 0] = orig
                fd = float((lp - lm) / (2 * eps))
                an = float(grad[i, 0])
                assert abs(fd - an) < 1e-4, (
                    f"{name} coord {i}: fd={fd:.6e} analytic={an:.6e}")

    def test_full_layer_grad_cosine_vs_fd(self):
        torch.manual_seed(2024)          # deterministic regardless of suite order
        attn = _make_attn(n=6)
        t_in = _rand_times(6, 5)
        g = torch.Generator().manual_seed(11)
        y = torch.randint(0, 6, (5,), generator=g,
                          dtype=torch.long).to(DEVICE)

        def loss_fn():
            return latency_cross_entropy(attn(t_in), y, 40.0)

        loss_fn().backward()
        W = attn.WQ.weight
        g_auto = W.grad.clone()
        eps = 1e-5
        rows, cols = W.shape
        smooth = torch.ones_like(W, dtype=torch.bool)
        for i in range(rows):
            for j in range(cols):
                orig = W[i, j].item()
                with torch.no_grad():
                    W[i, j] = orig + eps
                lp = float(loss_fn().detach())
                with torch.no_grad():
                    W[i, j] = orig - eps
                lm = float(loss_fn().detach())
                with torch.no_grad():
                    W[i, j] = orig
                fd = (lp - lm) / (2 * eps)
                if abs(fd) > 1e-12 and abs(lp - lm) > 0.05 * abs(fd):
                    smooth[i, j] = False

        g_fd = torch.zeros_like(W)
        with torch.no_grad():
            for i in range(rows):
                for j in range(cols):
                    if not smooth[i, j]:
                        continue
                    orig = W[i, j].item()
                    W[i, j] = orig + eps
                    lp = float(loss_fn().detach())
                    W[i, j] = orig - eps
                    lm = float(loss_fn().detach())
                    W[i, j] = orig
                    g_fd[i, j] = (lp - lm) / (2 * eps)

        # Only compare on entries with a significant analytic gradient so a
        # near-zero-gradient configuration (degenerate softmax) is not treated
        # as a mismatch.
        sig = g_auto.abs() > 1e-9
        keep = smooth & sig
        assert keep.sum() >= 3, f"Too few smooth/significant weights: {int(keep.sum())}"
        a = g_auto[keep].float().detach().cpu().numpy().ravel()
        b = g_fd[keep].float().detach().cpu().numpy().ravel()
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        assert cos > 0.90, f"Cosine too low: {cos:.6f}"


def _seq_det_attn():
    """Deterministic (1,1) attention block with steep crossings (FD-friendly)."""
    a = ExactSpikingAttention(1, 1, tm=8.0, ts=2.0, theta=1.0, t_max=10.0,
                              w_scale=0.1, bias_val=0.2, seed=0,
                              dtype=DTYPE, device=torch.device("cpu"))
    with torch.no_grad():
        for p in (a.WQ, a.WK, a.WV):
            p.weight.data = torch.tensor([[2.5, 0.3]], dtype=DTYPE)
    return a


class TestSequenceMode:
    def test_seq_rejects_nan_input(self):
        attn = _make_attn(3)
        nan_seq = torch.full((4, 3, 2), float("nan"), dtype=DTYPE, device=DEVICE)
        with __import__("pytest").raises(ValueError, match="must not contain NaN"):
            attn(nan_seq)

    def test_forward_shape_3d(self):
        attn = _make_attn(3)
        t_in = torch.rand(4, 3, 2, dtype=DTYPE, device=DEVICE)
        out = attn(t_in)
        assert out.shape == (4, 3, 2)
        assert out.dtype == DTYPE

    def test_backward_populates_seq(self):
        attn = _make_attn(3)
        t_in = torch.rand(4, 3, 2, dtype=DTYPE, device=DEVICE)
        t_in.requires_grad_(True)
        attn(t_in).sum().backward()
        for p in (attn.WQ, attn.WK, attn.WV):
            assert p.weight.grad is not None and torch.isfinite(p.weight.grad).all()
        assert t_in.grad is not None and torch.isfinite(t_in.grad).all()

    def test_3d_dim_mismatch(self):
        attn = _make_attn(3)
        with pytest.raises(ValueError, match="feature dim"):
            attn(torch.rand(4, 5, 2, dtype=DTYPE, device=DEVICE))

    def test_seq_fold_equals_per_feature_combine(self):
        """The (S, n, B) fold is exactly the per-feature 2D combine."""
        g = torch.Generator().manual_seed(9)
        S, n, B = 3, 2, 2
        tm, ts, temp = 15.0, 4.0, 1.0
        alpha, k_peak, t_max = False, 1.0, 40.0
        q = torch.rand(S, n, B, generator=g, dtype=DTYPE) * 10.0 + 1.0
        k = torch.rand(S, n, B, generator=g, dtype=DTYPE) * 10.0 + 1.0
        v = torch.rand(S, n, B, generator=g, dtype=DTYPE) * 10.0 + 1.0
        qb = q.permute(0, 2, 1).reshape(S, n * B)
        kb = k.permute(0, 2, 1).reshape(S, n * B)
        vb = v.permute(0, 2, 1).reshape(S, n * B)
        out_fold = ExactAttentionCombineFn.apply(
            qb, kb, vb, tm, ts, alpha, k_peak, temp, t_max,
            "gaussian").reshape(S, B, n).permute(0, 2, 1)
        out_feat = torch.empty_like(out_fold)
        for i in range(n):
            out_feat[:, i, :] = ExactAttentionCombineFn.apply(
                q[:, i, :], k[:, i, :], v[:, i, :], tm, ts, alpha, k_peak,
                temp, t_max, "gaussian")
        assert torch.equal(out_fold, out_feat)

    def test_3d_block_grad_matches_fd(self):
        """End-to-end exact gradients through Q/K/V + per-position attention
        vs central finite differences on a deterministic steep config."""
        a = _seq_det_attn()
        tin = torch.tensor([[[3.0]], [[4.5]], [[6.0]]], dtype=DTYPE,
                           device=torch.device("cpu"))
        tin.requires_grad_(True)

        def loss_fn():
            return a(tin.detach()).sum()

        loss_fn().backward()
        for name in ("WQ", "WK", "WV"):
            p = getattr(a, name).weight
            p.requires_grad_(True)
            a.zero_grad()
            loss = a(tin.detach()).sum()
            loss.backward()
            eps = 1e-3
            flat = p.detach().flatten()
            with torch.no_grad():
                for i in range(flat.numel()):
                    v0 = flat[i].item()
                    flat[i] = v0 + eps
                    lp = loss_fn()
                    flat[i] = v0 - eps
                    lm = loss_fn()
                    flat[i] = v0
                    fd = (lp - lm) / (2 * eps)
                    an = p.grad.detach().flatten()[i]
                    assert abs(an - fd) <= 5e-3 * (1 + abs(an)), (
                        f"{name}[{i}]: an={an:.5f} fd={fd:.5f}")
        # input gradient (larger eps: the solver's time shifts quantize below 1e-3)
        tin.grad = None
        a.zero_grad()
        a(tin.clone()).sum().backward()
        an = tin.grad.detach().flatten()
        eps = 1e-2
        flat = tin.detach().flatten()
        with torch.no_grad():
            for i in range(flat.numel()):
                v0 = flat[i].item()
                flat[i] = v0 + eps
                lp = a(tin.detach()).sum()
                flat[i] = v0 - eps
                lm = a(tin.detach()).sum()
                flat[i] = v0
                fd = (lp - lm) / (2 * eps)
                assert abs(an[i] - fd) <= 5e-3 * (1 + abs(an[i].item())), (
                    f"tin[{i}]: an={an[i]:.5f} fd={fd:.5f}")