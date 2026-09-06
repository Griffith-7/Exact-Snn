"""Tests for ExactSpikingFFN (exact spiking feed-forward block).

Verifies shapes, autograd integration, calibration, and that the analytic
gradients match finite differences on a deterministic steep configuration --
the library's "exact" contract.

Run:  python -m pytest tests/ -v    (from the project root)
"""
from __future__ import annotations

import torch

from exact_snn import ExactTTFSLinear, latency_cross_entropy
from exact_snn.ffn import ExactSpikingFFN
from exact_snn.extended import ExactSpikingFFN as ExtendedExactSpikingFFN

torch.manual_seed(1234)

DEVICE = torch.device("cpu")
DTYPE = torch.float64
TM, TS, THETA = 15.0, 4.0, 1.0
T_MAX = 40.0


def _make_ffn(n_in=3, n_hidden=5, n_out=3, **kwargs):
    defaults = dict(tm=TM, ts=TS, theta=THETA, t_max=T_MAX, w_scale=0.2,
                    bias_val=1.5, seed=7, dtype=DTYPE, device=DEVICE)
    defaults.update(kwargs)
    return ExactSpikingFFN(n_in, n_hidden, n_out, **defaults)


def _rand_times(n, B):
    return torch.rand(n, B, dtype=DTYPE, device=DEVICE) * 0.8 * T_MAX + 0.1


def _det_cell():
    """Deterministic FFN with one expand/one contract projection configured so
    both crossings are smooth; the exact saltation gradient is then measurable
    against central finite differences at a generous FD step (the forward
    solver's spike times quantize below ~1e-7)."""
    m = ExactSpikingFFN(1, 1, 1, tm=8.0, ts=2.0, theta=1.0, t_max=10.0,
                        w_scale=0.1, bias_val=0.2, seed=0, dtype=DTYPE,
                        device=DEVICE)
    with torch.no_grad():
        m.in_proj.weight.data = torch.tensor([[2.5, 0.2]], dtype=DTYPE,
                                             device=DEVICE)
        m.out_proj.weight.data = torch.tensor([[1.5, 0.8]], dtype=DTYPE,
                                              device=DEVICE)
    return m


def _det_data():
    return torch.tensor([[3.5]], dtype=DTYPE, device=DEVICE)


def _det_loss(m, tin):
    return m.out_proj(m.in_proj(tin))[0, 0]


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
    tol = 5e-3 * (1.0 + an.detach().abs())
    assert (err <= tol).all(), f"max FD deviation {err.max().item()}"


class TestBuildAndShapes:
    def test_is_exported(self):
        assert ExtendedExactSpikingFFN is ExactSpikingFFN

    def test_forward_shape_and_parameters(self):
        m = _make_ffn()
        assert m.in_proj.weight.shape == (5, 4)
        assert m.out_proj.weight.shape == (3, 6)
        nparams = sum(p.numel() for p in m.parameters())
        assert nparams == 5 * 4 + 3 * 6
        t_in = _rand_times(3, 5)
        out = m(t_in)
        assert out.shape == (3, 5)
        assert out.dtype == DTYPE
        assert out.device == DEVICE

    def test_some_neurons_fire(self):
        m = _make_ffn(w_scale=0.8, bias_val=1.2)
        t_in = _rand_times(3, 6)
        out = m(t_in)
        assert torch.isfinite(out).any()

    def test_silent_hidden_propagates(self):
        """inf (silent) hidden cells contribute K(t-inf)=0 upstream (inf out)."""
        m = _make_ffn(w_scale=0.0, bias_val=0.0)
        t_in = _rand_times(3, 5)
        out = m(t_in)
        assert (out == float("inf")).all()

    def test_dtype_casting(self):
        m = _make_ffn(dtype=torch.float32)
        t_in = torch.rand(3, 4, dtype=torch.float32)
        out = m(t_in)
        assert out.dtype == torch.float32

    def test_default_dtype_is_float32(self):
        """dtype=None (omitted) must not crash the projections."""
        m = ExactSpikingFFN(3, 5, 3, device=DEVICE)
        assert m.in_proj.weight.dtype == torch.float32
        assert m.out_proj.weight.dtype == torch.float32


class TestAutograd:
    def test_backward_populates(self):
        m = _make_ffn()
        t_in = _rand_times(3, 6).requires_grad_(True)
        p = m(t_in)
        y = torch.zeros(6, dtype=torch.long, device=DEVICE)
        loss = latency_cross_entropy(p, y, T_MAX)
        loss.backward()
        assert m.in_proj.weight.grad is not None
        assert m.out_proj.weight.grad is not None
        assert torch.isfinite(m.in_proj.weight.grad).all()
        assert torch.isfinite(m.out_proj.weight.grad).all()
        assert t_in.grad is not None
        assert torch.isfinite(t_in.grad).all()

    def test_input_grad_matches_finite_differences(self):
        m = _det_cell()
        tin = _det_data()
        tin.requires_grad_(True)
        loss = _det_loss(m, tin)
        loss.backward()
        g_fd = _fd_grad(tin, lambda: _det_loss(m, tin.detach()))
        _assert_fd_matches(tin.grad, g_fd)

    def test_in_proj_grad_matches_finite_differences(self):
        m = _det_cell()
        tin = _det_data()
        loss = _det_loss(m, tin)
        loss.backward()
        g_fd = _fd_grad(m.in_proj.weight,
                        lambda: _det_loss(m, tin.detach()))
        _assert_fd_matches(m.in_proj.weight.grad, g_fd)

    def test_out_proj_grad_matches_finite_differences(self):
        m = _det_cell()
        tin = _det_data()
        loss = _det_loss(m, tin)
        loss.backward()
        g_fd = _fd_grad(m.out_proj.weight,
                        lambda: _det_loss(m, tin.detach()))
        _assert_fd_matches(m.out_proj.weight.grad, g_fd)


class TestCalibration:
    def test_calibrate_init_fire_is_idempotent(self):
        m = _make_ffn(w_scale=0.05, bias_val=0.2, seed=7)
        m.calibrate_init_fire(target=0.9)
        assert torch.isfinite(m.in_proj.weight).all()
        assert torch.isfinite(m.out_proj.weight).all()
        # re-calibrating an already-calibrated block changes little
        for proj in (m.in_proj, m.out_proj):
            b0 = proj.weight[:, -1].clone()
            m.calibrate_init_fire(target=0.9)
            assert (proj.weight[:, -1] - b0).abs().max() < 1e-1


class TestResidualAndNorm:
    def test_residual_requires_square(self):
        with __import__("pytest").raises(ValueError, match="residual"):
            ExactSpikingFFN(3, 5, 4, residual=True, device=DEVICE)

    def test_residual_fuses_earliest(self):
        m = _make_ffn(3, 5, 3, residual=True, seed=5)
        t_in = _rand_times(3, 5)
        t_in.requires_grad_(True)
        out = m(t_in)
        with torch.no_grad():
            plain = m.out_proj(m.in_proj(t_in.detach()))
        assert torch.equal(out, torch.minimum(plain, t_in.detach()))
        out.sum().backward()
        assert m.in_proj.weight.grad is not None
        assert m.out_proj.weight.grad is not None
        assert t_in.grad is not None

    def test_use_norm_applies_spikenorm(self):
        m = _make_ffn(3, 6, 3, use_norm=True, seed=5)
        assert hasattr(m, "norm")
        t_in = _rand_times(3, 5).requires_grad_(True)
        out = m(t_in)
        assert out.shape == (3, 5)
        out.sum().backward()
        for p in m.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()

    def test_spikenorm_masked_silent_batch(self):
        """SpikeNorm normalizes only fired neurons; a silent batch is returned
        as-is (all inf) and must not poison running stats with NaN."""
        from exact_snn.normalize import SpikeNorm
        norm = SpikeNorm(3, device=DEVICE).train()
        silent = torch.full((3, 4), float("inf"), dtype=DTYPE, device=DEVICE)
        out = norm(silent)
        assert torch.isinf(out).all(), "fully silent batch must stay silent"
        assert torch.isfinite(norm.running_mean).all(), "state must stay clean"
        assert torch.isfinite(norm.running_var).all()

    def test_spikenorm_masks_silent_entries(self):
        """Silent entries survive spike normalization; fired entries are
        normalized; per-feature stats update only where anything fired."""
        from exact_snn.normalize import SpikeNorm
        norm = SpikeNorm(3, device=DEVICE).train()
        t = torch.tensor(
            [[3.0, 5.0, 7.0],                # fully fired feature
             [float("inf"), float("inf"), float("inf")],   # silent feature
             [2.0, float("inf"), 4.0]],      # mixed feature
            dtype=DTYPE, device=DEVICE)
        out = norm(t)
        assert torch.isinf(out[1]).all(), "silent feature stays silent"
        assert torch.isfinite(out[0]).all(), "fired feature normalized"
        assert torch.isfinite(out[2, 0]) and torch.isfinite(out[2, 2])
        assert torch.isinf(out[2, 1]), "mixed-feature silent entry stays inf"
        assert torch.isfinite(norm.running_mean).all()
        assert torch.isfinite(norm.running_var).all()
        # feature 1 never fired: its running stats must remain the init values
        assert norm.running_mean[1] == 0.0
        assert norm.running_var[1] == 1.0

    def test_ffn_norm_with_silent_hidden_does_not_crash(self):
        """Forward with use_norm=True must survive a valid silent hidden cell
        (masked normalization), not raise, and keep gradients finite."""
        m = _make_ffn(3, 6, 3, use_norm=True, seed=5)
        with torch.no_grad():
            m.in_proj.weight[0, -1] = -1e3  # force hidden neuron 0 silent
        t_in = _rand_times(3, 5).requires_grad_(True)
        out = m(t_in)
        assert out.shape == (3, 5)
        out.sum().backward()
        for p in m.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()
        assert torch.isfinite(m.norm.running_mean).all()


class TestTraining:
    def test_trains_small_task(self):
        m = _make_ffn(2, 4, 2, seed=3)
        g = torch.Generator().manual_seed(991)
        t_in = (torch.rand(2, 8, generator=g, dtype=DTYPE, device=DEVICE)
                * 0.8 * T_MAX + 0.1)
        y = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], dtype=torch.long,
                         device=DEVICE)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        m.calibrate_init_fire(target=0.5)
        initial = float(latency_cross_entropy(m(t_in), y, T_MAX))
        for _ in range(40):
            opt.zero_grad()
            loss = latency_cross_entropy(m(t_in), y, T_MAX)
            loss.backward()
            opt.step()
        final = float(latency_cross_entropy(m(t_in), y, T_MAX))
        assert final < initial
        assert all(ExactTTFSLinear is type(t) for t in (m.in_proj, m.out_proj))