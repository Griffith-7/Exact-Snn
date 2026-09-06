"""Exact Spiking Recurrent Memory module using exact spike-time gradients.

Provides `ExactSpikingRecurrentMemory`, an unrolled multi-step temporal recurrent
block that combines current input spikes and persistent recurrent hidden spikes
into an exact IFT spike linear layer, maintaining exact autograd gradient flow
across arbitrary sequence lengths.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import torch
import torch.nn as nn

from exact_snn import ExactTTFSLinear


class ExactSpikingRecurrentMemory(nn.Module):
    """Multi-Step Exact Spiking Recurrent Memory Cell.

    Args:
        n_in: Number of input features.
        n_hidden: Recurrent hidden state dimension.
        n_out: Optional output feature dimension (defaults to n_hidden if None).
        tm: Membrane time constant (ms).
        ts: Synaptic time constant (ms).
        theta: Threshold voltage.
        t_max: Maximum temporal window limit.
        w_scale: Initial weight scaling parameter.
        bias_val: Initial bias scale.
        grid_pts: Number of grid points for grid-scan root search.
        dtype: PyTorch tensor data type.
        device: Target compute device.
    """

    def __init__(
        self,
        n_in: int,
        n_hidden: int,
        n_out: Optional[int] = None,
        tm: float = 10.0,
        ts: float = 2.5,
        theta: float = 1.0,
        t_max: float = 40.0,
        w_scale: float = 0.3,
        bias_val: float = 0.5,
        grid_pts: int = 101,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_hidden if n_out is None else n_out

        # Recurrent transition layer: takes [x_t; h_{t-1}] -> h_t
        self.cell = ExactTTFSLinear(
            n_in=n_in + n_hidden,
            n_out=n_hidden,
            tm=tm,
            ts=ts,
            theta=theta,
            t_max=t_max,
            w_scale=w_scale,
            bias_val=bias_val,
            grid_pts=grid_pts,
            dtype=dtype,
            device=device,
        )

        # Output projection layer: takes h_t -> y_t
        self.out_proj = ExactTTFSLinear(
            n_in=n_hidden,
            n_out=self.n_out,
            tm=tm,
            ts=ts,
            theta=theta,
            t_max=t_max,
            w_scale=w_scale,
            bias_val=bias_val,
            grid_pts=grid_pts,
            dtype=dtype,
            device=device,
        )

    def init_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Initialize initial hidden spike times with zeros (earliest baseline spike times)."""
        return torch.zeros((self.n_hidden, batch_size), dtype=dtype, device=device)

    def forward_step(
        self, x_t: torch.Tensor, h_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process a single sequence step.

        Args:
            x_t: Input spike times tensor `(n_in, B)`.
            h_prev: Previous hidden spike times tensor `(n_hidden, B)`.

        Returns:
            Tuple `(y_t, h_t)` of output spike times and new hidden state spike times.
        """
        # Concatenate along feature dimension
        xh = torch.cat([x_t, h_prev], dim=0)  # (n_in + n_hidden, B)
        h_t = self.cell(xh)  # (n_hidden, B)
        y_t = self.out_proj(h_t)  # (n_out, B)
        return y_t, h_t

    def forward(
        self, seq_in: torch.Tensor, h_0: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass over an unrolled temporal sequence.

        Args:
            seq_in: Sequence tensor of shape `(seq_len, n_in, B)`.
            h_0: Optional initial hidden state tensor `(n_hidden, B)`.

        Returns:
            Tuple `(seq_out, h_final)`:
            - `seq_out`: Output sequence tensor `(seq_len, n_out, B)`.
            - `h_final`: Final hidden state tensor `(n_hidden, B)`.
        """
        S, n_in, B = seq_in.shape
        if h_0 is None:
            h_curr = self.init_hidden(B, seq_in.device, seq_in.dtype)
        else:
            h_curr = h_0

        outputs: List[torch.Tensor] = []
        for s in range(S):
            x_s = seq_in[s]  # (n_in, B)
            y_s, h_curr = self.forward_step(x_s, h_curr)
            outputs.append(y_s)

        seq_out = torch.stack(outputs, dim=0)  # (seq_len, n_out, B)
        return seq_out, h_curr
