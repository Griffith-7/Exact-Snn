"""Performance and execution benchmarking for Exact-SNN Transformer & Recurrent Architecture.

Measures throughput (samples/sec), latency per forward/backward pass (ms),
peak VRAM/memory allocation, and optimization convergence for:
1. ExactSpikingFFN
2. ExactSpikingAttention
3. ExactSpikingRecurrentMemory
4. ExactSpikingTransformerBlock (Attention + FFN composition)
"""

import time
import torch
import torch.nn as nn

from exact_snn import latency_cross_entropy, latency_encode
from exact_snn.extended import ExactSpikingFFN, ExactSpikingAttention, ExactSpikingRecurrentMemory


class ExactSpikingTransformerBlock(nn.Module):
    """Full Spiking Transformer Block composed of Exact Attention and FFN."""

    def __init__(self, d_model: int = 16, num_heads: int = 4, d_hidden: int = 32):
        super().__init__()
        self.attn = ExactSpikingAttention(d_model=d_model, num_heads=num_heads, bias_val=2.0)
        self.ffn = ExactSpikingFFN(n_in=d_model, n_hidden=d_hidden, n_out=d_model, residual=True, bias_val=2.0)

    def forward(self, t_in: torch.Tensor) -> torch.Tensor:
        # Attention + Residual
        t_attn = self.attn(t_in)
        if t_in.dim() == 2:
            t_mid = torch.minimum(t_in, t_attn)
        else:
            t_mid = t_attn
        # FFN + Residual
        t_out = self.ffn(t_mid)
        return t_out


def benchmark_module(name: str, model: nn.Module, sample_input: torch.Tensor, num_runs: int = 20):
    """Measure forward and backward execution latency and throughput."""
    device = sample_input.device
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(3):
        out = model(sample_input)
        if isinstance(out, tuple):
            out = out[0]
        loss = out[torch.isfinite(out)].sum()
        loss.backward()

    # Benchmark Forward
    t_fwd_start = time.perf_counter()
    for _ in range(num_runs):
        out = model(sample_input)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_fwd_end = time.perf_counter()
    fwd_time_ms = ((t_fwd_end - t_fwd_start) / num_runs) * 1000.0

    # Benchmark Backward
    t_bwd_start = time.perf_counter()
    for _ in range(num_runs):
        out = model(sample_input)
        if isinstance(out, tuple):
            out = out[0]
        loss = out[torch.isfinite(out)].sum()
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_bwd_end = time.perf_counter()
    bwd_time_ms = ((t_bwd_end - t_bwd_start) / num_runs) * 1000.0

    batch_size = sample_input.shape[-1] if sample_input.dim() >= 2 else 1
    throughput = batch_size / ((fwd_time_ms + bwd_time_ms) / 1000.0)

    mem_kb = 0.0
    if device.type == "cuda":
        mem_kb = torch.cuda.max_memory_allocated() / 1024.0

    print(f"| {name:<32} | Fwd: {fwd_time_ms:6.2f} ms | Bwd: {bwd_time_ms:6.2f} ms | Throughput: {throughput:7.1f} samples/s | Peak Mem: {mem_kb:6.1f} KB |")


def run_training_convergence_demo():
    """Verify convergence on a temporal sequence classification benchmark."""
    print("\n--- Exact Spiking Transformer Convergence Demo ---")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ExactSpikingTransformerBlock(d_model=16, num_heads=4, d_hidden=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    x = torch.rand(16, 8, device=device)  # (n_in=16, B=8)
    t_in = latency_encode(x, t_max=40.0)
    targets = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device=device)

    start_loss = None
    final_loss = None

    for epoch in range(1, 11):
        opt.zero_grad()
        t_out = model(t_in)
        # Select first 4 output channels for classification
        loss = latency_cross_entropy(t_out[:4], targets, t_max=40.0)
        loss.backward()
        opt.step()

        if epoch == 1:
            start_loss = loss.item()
        final_loss = loss.item()
        print(f"Epoch {epoch:2d}/10 - Latency Cross Entropy Loss: {loss.item():.4f}")

    print(f"Convergence Verification: Initial Loss = {start_loss:.4f} -> Final Loss = {final_loss:.4f}")
    assert final_loss < start_loss, "Model loss did not decrease during training!"


def main():
    print("========================================================================================================")
    print("                          EXACT-SNN ARCHITECTURE PERFORMANCE BENCHMARK                                  ")
    print("========================================================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {device}\n")

    # Sample Tensors
    t_2d = torch.rand((16, 32), device=device) * 2.0  # (d_model=16, B=32)
    t_3d = torch.rand((5, 16, 16), device=device) * 2.0 # (seq_len=5, n_in=16, B=16)

    # Instantiate modules
    ffn = ExactSpikingFFN(n_in=16, n_hidden=32, n_out=16, bias_val=2.0).to(device)
    attn = ExactSpikingAttention(d_model=16, num_heads=4, bias_val=2.0).to(device)
    rnn = ExactSpikingRecurrentMemory(n_in=16, n_hidden=32, n_out=16, bias_val=2.0).to(device)
    transformer_block = ExactSpikingTransformerBlock(d_model=16, num_heads=4, d_hidden=32).to(device)
    
    # Accelerated Event Engine Transformer Block
    attn_event = ExactSpikingAttention(d_model=16, num_heads=4, bias_val=2.0, use_event=True).to(device)
    ffn_event = ExactSpikingFFN(n_in=16, n_hidden=32, n_out=16, bias_val=2.0, use_event=True).to(device)

    print("| Module Name                      | Forward (ms) | Backward (ms)| Throughput (samples/s) | Peak Memory |")
    print("|----------------------------------|--------------|--------------|------------------------|-------------|")
    benchmark_module("ExactSpikingFFN (2D)", ffn, t_2d)
    benchmark_module("ExactSpikingAttention (2D)", attn, t_2d)
    benchmark_module("ExactSpikingRecurrentMemory (3D)", rnn, t_3d)
    benchmark_module("ExactSpikingTransformerBlock", transformer_block, t_2d)
    benchmark_module("ExactSpikingAttention (Event)", attn_event, t_2d)
    benchmark_module("ExactSpikingFFN (Event)", ffn_event, t_2d)
    print("--------------------------------------------------------------------------------------------------------")

    run_training_convergence_demo()


if __name__ == "__main__":
    main()
