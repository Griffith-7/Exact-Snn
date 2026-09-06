"""Unit tests for ExactSpikingRecurrentMemory module."""

import pytest
import torch

from exact_snn.recurrent_memory import ExactSpikingRecurrentMemory
from exact_snn import latency_cross_entropy


def test_recurrent_memory_step_shape():
    rnn = ExactSpikingRecurrentMemory(n_in=8, n_hidden=16, n_out=8, bias_val=2.0)
    x_t = torch.rand((8, 4)) * 2.0
    h_prev = rnn.init_hidden(4, x_t.device)

    y_t, h_t = rnn.forward_step(x_t, h_prev)
    assert y_t.shape == (8, 4)
    assert h_t.shape == (16, 4)


def test_recurrent_memory_sequence_shape():
    rnn = ExactSpikingRecurrentMemory(n_in=6, n_hidden=12, n_out=4, bias_val=2.0)
    seq_in = torch.rand((5, 6, 3)) * 2.0  # (seq_len=5, n_in=6, B=3)

    seq_out, h_final = rnn(seq_in)
    assert seq_out.shape == (5, 4, 3)
    assert h_final.shape == (12, 3)


def test_recurrent_memory_autograd_backward():
    torch.manual_seed(42)
    rnn = ExactSpikingRecurrentMemory(n_in=4, n_hidden=8, n_out=4, bias_val=2.0)
    seq_in = torch.rand((3, 4, 2), requires_grad=True) * 2.0

    seq_out, _ = rnn(seq_in)
    loss = seq_out[torch.isfinite(seq_out)].sum()
    loss.backward()

    assert rnn.cell.weight.grad is not None
    assert rnn.out_proj.weight.grad is not None
    assert rnn.cell.weight.grad.abs().sum() > 0


def test_recurrent_memory_optimizer_step():
    torch.manual_seed(42)
    rnn = ExactSpikingRecurrentMemory(n_in=8, n_hidden=16, n_out=8, bias_val=2.0)
    opt = torch.optim.Adam(rnn.parameters(), lr=1e-2)

    seq_in = torch.rand((3, 8, 4)) * 2.0
    targets = torch.tensor([0, 1, 2, 3])

    seq_out0, _ = rnn(seq_in)
    loss0 = latency_cross_entropy(seq_out0[-1], targets, t_max=40.0)
    loss0.backward()
    opt.step()

    seq_out1, _ = rnn(seq_in)
    loss1 = latency_cross_entropy(seq_out1[-1], targets, t_max=40.0)

    assert loss1.item() < loss0.item()


def test_recurrent_memory_finite_difference_gradient():
    torch.manual_seed(77)
    rnn = ExactSpikingRecurrentMemory(n_in=3, n_hidden=4, n_out=3, bias_val=2.0)
    seq_in = torch.rand((2, 3, 2)) * 1.5

    seq_out, _ = rnn(seq_in)
    loss = seq_out[torch.isfinite(seq_out)].sum()
    loss.backward()

    analytic_grad = rnn.cell.weight.grad.clone()

    eps = 1e-4
    numeric_grad = torch.zeros_like(rnn.cell.weight)
    with torch.no_grad():
        for i in range(rnn.cell.weight.shape[0]):
            for j in range(rnn.cell.weight.shape[1]):
                w_orig = rnn.cell.weight[i, j].item()

                rnn.cell.weight[i, j] = w_orig + eps
                seq_pos, _ = rnn(seq_in)
                l_pos = seq_pos[torch.isfinite(seq_pos)].sum().item()

                rnn.cell.weight[i, j] = w_orig - eps
                seq_neg, _ = rnn(seq_in)
                l_neg = seq_neg[torch.isfinite(seq_neg)].sum().item()

                rnn.cell.weight[i, j] = w_orig
                numeric_grad[i, j] = (l_pos - l_neg) / (2 * eps)

    cos_sim = torch.nn.functional.cosine_similarity(
        analytic_grad.view(-1), numeric_grad.view(-1), dim=0
    )
    assert cos_sim.item() > 0.90
