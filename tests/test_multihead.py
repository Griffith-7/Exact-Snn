"""Tests for ExactSpikingMultiHeadAttention.

The wrapper is a composition of already-FD-verified exact single-head blocks,
so exactness is pinned by *equivalence*: a 1-head multi-head block must give
identical outputs and identical parameter gradients to an equivalent plain
`ExactSpikingAttention` (bit-for-bit honoring the surrogate-free contract),
plus shape / autograd / calibration / training coverage for the fused modes.

Run:  python -m pytest tests/ -v    (from the project root)
"""
from __future__ import annotations

import torch

from exact_snn import latency_cross_entropy
from exact_snn.attention import ExactSpikingAttention
from exact_snn.multihead import ExactSpikingMultiHeadAttention
from exact_snn.extended import ExactSpikingMultiHeadAttention as ExtendedMH

torch.manual_seed(1234)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
T_MAX = 40.0


def _make_mh(n=8, n_heads=2, **kwargs):
    defaults = dict(tm=15.0, ts=4.0, theta=1.0, t_max=T_MAX, w_scale=0.2,
                    bias_val=1.5, grid_pts=2001, seed=7, dtype=DTYPE,
                    device=DEVICE, temp=1.0)
    defaults.update(kwargs)
    return ExactSpikingMultiHeadAttention(n, n_heads, **defaults)


def _rand_times(n, B):
    return (torch.rand(n, B, dtype=DTYPE, device=DEVICE) * 0.8 * T_MAX + 0.1)


def _copy_into(ref, dst):
    """Copy every parameter value from `ref` (a plain attention block) into the
    given head of a multi-head block."""
    with torch.no_grad():
        for p_r, p_d in zip(ref.parameters(), dst.parameters()):
            p_d.copy_(p_r)
    return dst


class TestBuildAndShapes:
    def test_is_exported(self):
        assert ExtendedMH is ExactSpikingMultiHeadAttention

    def test_parameter_count(self):
        mh = _make_mh(8, n_heads=4)
        nparams = sum(p.numel() for p in mh.parameters())
        assert nparams == 4 * 3 * 8 * 9

    def test_forward_shape(self):
        mh = _make_mh(8, n_heads=3)
        t_in = _rand_times(8, 5)
        out = mh(t_in)
        assert out.shape == (8, 5)
        assert out.dtype == DTYPE

    def test_forward_fuse_full(self):
        mh = _make_mh(6, n_heads=3, fuse="full")
        t_in = _rand_times(6, 4)
        out = mh(t_in)
        assert out.shape == (3, 6, 4)

    def test_forward_fuse_mean(self):
        mh = _make_mh(6, n_heads=3, fuse="mean")
        t_in = _rand_times(6, 4)
        out = mh(t_in)
        assert out.shape == (6, 4)

    def test_input_validation(self):
        import pytest
        with pytest.raises(ValueError, match="n_heads"):
            _make_mh(8, n_heads=0)
        with pytest.raises(ValueError, match="fuse"):
            _make_mh(8, n_heads=2, fuse="bogus")

    def test_default_dtype_is_float32(self):
        m = ExactSpikingMultiHeadAttention(4, n_heads=2, device=DEVICE)
        assert m.heads[0].WQ.weight.dtype == torch.float32


class TestExactnessEquivalence:
    def test_one_head_equals_single_head_gradients(self):
        """Exactness contract: a 1-head wrapper is the identity composition of
        the exact single-head block (same outputs, same grads, choice-blind)."""
        ref = ExactSpikingAttention(6, 6, tm=15.0, ts=4.0, theta=1.0,
                                    t_max=T_MAX, w_scale=0.2, bias_val=1.5,
                                    grid_pts=2001, seed=7, dtype=DTYPE,
                                    device=DEVICE)
        mh = ExactSpikingMultiHeadAttention(6, n_heads=1, tm=15.0, ts=4.0,
                                            theta=1.0, t_max=T_MAX,
                                            w_scale=0.2, bias_val=1.5,
                                            grid_pts=2001, seed=7,
                                            dtype=DTYPE, device=DEVICE)
        mh = _copy_into(ref, mh)
        t_in = _rand_times(6, 5).requires_grad_(True)
        o_ref = ref(t_in)
        o_mh = mh(t_in)
        assert torch.equal(o_ref, o_mh)
        y = torch.zeros(5, dtype=torch.long, device=DEVICE)
        latency_cross_entropy(o_ref, y, T_MAX).backward()
        g_ref = [p.grad.detach().clone() for p in ref.parameters()]
        latency_cross_entropy(o_mh, y, T_MAX).backward()
        for g_r, p_m in zip(g_ref, mh.parameters()):
            assert torch.equal(g_r, p_m.grad)

    def test_min_fuse_gradient_flows(self):
        """fuse='min' routes the gradient through the earliest-spiking head."""
        mh = _make_mh(6, n_heads=2)
        t_in = _rand_times(6, 4)
        loss = mh(t_in).sum()
        loss.backward()
        for p in mh.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()


class TestAutogradAndTraining:
    def test_backward_populates(self):
        mh = _make_mh(8, n_heads=3)
        t_in = _rand_times(8, 6).requires_grad_(True)
        p = mh(t_in)
        y = torch.zeros(6, dtype=torch.long, device=DEVICE)
        loss = latency_cross_entropy(p, y, T_MAX)
        loss.backward()
        for p in mh.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()
        assert t_in.grad is not None

    def test_sequence_input_passthrough(self):
        """3D (S, n, B) input flows through every head and fuses along S."""
        mh = _make_mh(3, n_heads=2)
        t_in = _rand_times(3, 2).unsqueeze(0).expand(4, 3, 2).contiguous()
        out_min = mh(t_in)
        assert out_min.shape == (4, 3, 2)
        mh_full = _make_mh(3, n_heads=2, fuse="full")
        out_full = mh_full(t_in)
        assert out_full.shape == (2, 4, 3, 2)
        assert torch.isfinite(out_min).any()

    def test_calibrate_init_fire_is_finite(self):
        mh = _make_mh(8, n_heads=2)
        mh.calibrate_init_fire(target=0.9)
        for head in mh.heads:
            assert torch.isfinite(head.WQ.weight).all()
            assert torch.isfinite(head.WK.weight).all()
            assert torch.isfinite(head.WV.weight).all()

    def test_trains_small_task(self):
        mh = _make_mh(n=2, n_heads=2, seed=3)
        g = torch.Generator(device=DEVICE).manual_seed(991)
        t_in = (torch.rand(2, 8, generator=g, dtype=DTYPE, device=DEVICE)
                * 0.8 * T_MAX + 0.1)
        y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.long,
                         device=DEVICE)
        opt = torch.optim.Adam(mh.parameters(), lr=1e-2)
        mh.calibrate_init_fire(target=0.5)
        initial = float(latency_cross_entropy(mh(t_in), y, T_MAX))
        for _ in range(150):
            opt.zero_grad()
            loss = latency_cross_entropy(mh(t_in), y, T_MAX)
            loss.backward()
            opt.step()
        final = float(latency_cross_entropy(mh(t_in), y, T_MAX))
        assert final < initial