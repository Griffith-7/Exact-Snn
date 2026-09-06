# Exact-SNN API Reference

`ExactSNN-nn` is a PyTorch plug-in. It supplies layers and mathematical helpers;
it does not own a dataset, application model, optimizer, scheduler, or training loop.

## Core API

Import from `exact_snn`:

### `ExactTTFSLinear(n_in, n_out, ...)`

A time-to-first-spike layer. Input shape is `(n_in, B)` and output shape is
`(n_out, B)`. Weights are stored as the public `nn.Parameter` `weight` with
shape `(n_out, n_in + 1)`; the final column is the bias channel.

`calibrate_init_fire(target=0.5, n_probe=32, cal_grid_pts=65, iters=6)`
optionally adjusts the bias channel so a target fraction of a probe population
fires (per-neuron Newton solve, see CHANGELOG 3.0.0).

### `ExactTTFSNetwork(sizes, ...)`

An `nn.ModuleList` stack of `ExactTTFSLinear` layers. `sizes` is
`[input_features, ..., output_features]`. `forward()` returns output spike times;
`loss()` applies latency cross-entropy. `calibrate_init_fire()` calibrates its
layers in sequence.

### `latency_encode(images, t_max=40.0, min_t=0.01, max_t=0.99)`

Maps intensities to latency spike times. The output has the same shape as the input;
brighter values produce earlier times.

### `latency_cross_entropy(t_out, y, t_max, beta=1.0)`

Differentiable cross-entropy over first-spike times. `t_out` has shape `(classes, B)`
and `y` has shape `(B,)`.

### `train_simple(model, X, y, ...)`

Optional small convenience helper for users who want a basic Adam loop. It is not
required by any layer and does not constrain application training.

## Extended layers

Import from `exact_snn.extended`:

- `ExactTTFSConv2d`: TTFS convolution. Input `(B, C, H, W)` and output
  `(B, out_channels, H_out, W_out)`.
- `ExactMultiSpike`: multi-spike layer. Input `(n_in, B)` and output
  `(n_out, B, K)`, where `K=max_spikes`.
- `ExactRecurrent`: recurrent TTFS layer. Call `reset_state(B)` between independent
  sequences, then call `forward_step(t_in)` for each step.
- `ExactTTFSRnn`: exact recurrent cell with full per-neuron feedback. Weight
  `(n_hidden, n_in + n_hidden + 1)` (input columns, recurrent columns, bias
  last). `forward_step(t_in, t_prev)` advances one step; `forward(t_in, h0=None)`
  unrolls `(n_in, B, T)` -> `(n_hidden, B, T)`. Silent cells carry `inf` and
  contribute (and back-propagate) zero. The backward passes the IFT recurrence
  Jacobian `dt / dh_{t-1}` through the unrolled chain, so gradients are exact
  end-to-end (no BPTT truncation).
- `ExactSpikingFFN`: exact spiking feed-forward block, the companion to
  `ExactSpikingAttention`. Two `ExactTTFSLinear` projections
  (`in_proj: n_in -> n_hidden`, `out_proj: n_hidden -> n_out`) drive the
  `g(t) > theta` membrane crossing with exact IFT gradients and inter-layer
  spike-time feed (silent hidden cells contribute `K(t - inf) = 0`).
  Options: `residual=True` fuses the skip `min(t_out, t_in)` (earliest-spike-
  wins, requires `n_out == n_in`); `use_norm=True` inserts a `SpikeNorm`
  between projections.
  `calibrate_init_fire(target=...)` calibrates both projections in sequence.
- `ExactSpikingAttention`: spiking self-attention. Input `(n_in, B)` -> output
  `(n_out, B)` attended spike times, **or** `(S, n_in, B)` -> `(S, n_out, B)`
  sequence attention across the `S` positions (per-feature exact scores,
  Q/K/V projections shared across positions). Internally it is three
  `ExactTTFSLinear` layers (`WQ`, `WK`, `WV`) plus a stateless exact combine
  step: `score = alignment(t_Q - t_K)`, softmax over keys, attended value time
  `sum_j a_ij t_V_j`. `combine="gaussian"` (default; peaks at alignment) or
  `"kernel"` (raw symmetric synaptic kernel). Requires `n_in == n_out` in v3.0.0.
- `ExactSpikingMultiHeadAttention`: multi-head spiking self-attention. `n_heads`
  independent single-head blocks (own Q/K/V projections) over the same `(n, B)`
  spikes, fused by `fuse="min"` (earliest-spike-wins, default; exact gradient
  routes through the winning head), `"mean"`, or `"full"` (`(n_heads, n, B)`).
- `multispike_latency_loss`: latency loss using the first spike from a multi-spike output.
- `spike_count_cross_entropy`: differentiable soft spike-count cross-entropy.

## Optional companion modules

- `exact_snn.event.ExactEventLinear`: event-driven drop-in TTFS layer. Compare it
  with the grid layer on the target workload before assuming a speed improvement.
- `exact_snn.existence.existence_loss_and_grads`: optional silent-neuron existence
  channel. It returns explicit weight gradients to add after normal autograd.
- `exact_snn.existence.peak_margin_torch`: peak time and membrane potential for
  existence calculations.
- `exact_snn.existence.edge_peak_guard`: mask for degenerate peak cases.
- `exact_snn.normalize.SpikeNorm`: batch normalization for `(n_features, B)` spike times.
- `exact_snn.losses.rate_latency_loss`: combined rate and first-spike latency loss.
- `exact_snn.reset.ResetLIF`: scalar reference/reset solver for verification.
- `exact_snn.initializers.xavier_init`, `kaiming_init`: optional initializers for
  `(fan_out, fan_in + 1)` weight tensors.
- `exact_snn.util.spike_time_augment`: optional bounded noise and time-shift augmentation.
- `exact_snn.attention.ExactSpikingAttention`: pure-spiking self-attention as an
  `nn.Module` (imported from `exact_snn.attention` or re-exported from
  `exact_snn.extended`).
- `exact_snn.recurrent.ExactTTFSRnn`: exact recurrent cell as an `nn.Module`
  (imported from `exact_snn.recurrent` or re-exported from `exact_snn.extended`).
- `exact_snn.ffn.ExactSpikingFFN`: exact spiking feed-forward block as an
  `nn.Module` (imported from `exact_snn.ffn` or re-exported from
  `exact_snn.extended`).
- `exact_snn.multihead.ExactSpikingMultiHeadAttention`: exact multi-head
  spiking attention as an `nn.Module` (imported from `exact_snn.multihead` or
  re-exported from `exact_snn.extended`).

## Optional CUDA root-solve backend

`exact_snn.cuda_ops` provides an optional CUDA backend that fuses the grid scan,
bisection, Newton refinement, and peak search of `_forward_layer_torch` into a
single kernel. It is JIT-compiled and cached by torch's `cpp_extension`; if it
cannot be built (no CUDA device, no nvcc, no MSVC) the library silently falls
back to the torch path.

- Enabled automatically for `ExactTTFSLinear` (and layers built on it: conv,
  recurrent, and attention projections) when input weights are CUDA `float32`.
- `cuda_ops.available()`: whether the extension built; `cuda_ops.status()`: a
  one-line human-readable backend state; `cuda_ops.set_enabled(False)`: force
  the torch path. `EXACT_SNN_CUDA_VERBOSE=1` prints the build log.
- The kernel builds the membrane grid in a transposed `(n, grid, B)` layout for
  coalesced scans and tiles a weight row plus a batch slice of input spikes into
  shared memory, so the exact `u_at`/`du_at` Newton evaluations stay off global
  memory. The math is identical to the torch path (verified to float32 parity
  in `tests/test_cuda.py`), including the near-threshold peak search.
- Resource notes: the membrane grid `(n, grid, B)` is materialized in memory in
  both paths, so very large layers still need GPU memory proportional to
  `n * grid * B`; on a 4 GB GPU a fine 1001-point grid fits roughly a
  `n ~ few hundred, B ~ 1000` layer.

## Benchmark: exact vs surrogate gradients (sine waveform)

A measured answer to "how good is exact-SNN training vs standard surrogate
training?" lives in the benchmarks:

- `benchmarks/surrogate_ttfs.py` -- a surrogate-gradient twin of the exact TTFS
  layers (kept out of the `exact_snn` package by design). Forward is the exact
  root solver (identical outputs); only the spike-time Jacobian is replaced by
  a sigmoid-membrane surrogate, magnitude-matched per batch to the exact signal.
- `benchmarks/sine_waveform_exact_vs_surrogate.py` -- the comparison harness.
  Task: given the first 8 latency-coded samples of a random-phase sine
  (period 1000 ms), predict the remaining 16 samples. Exact and surrogate nets
  are built with the same seed, train on the same data and the identical SSE
  loss; only the gradient rule differs.

Run: `python benchmarks/sine_waveform_exact_vs_surrogate.py --seeds 3`

Measured result (default config, 3 seeds, 40 epochs):

- exact:     final val RMSE 0.462 +/- 0.012  (final silent ~15%)
- surrogate: final val RMSE 0.427 +/- 0.009  (final silent ~18%)

Interpretation (honest):

1. On a healthy, input-driven init both rules train the task and reduce the
   decoded RMSE well below its chance level (0.354 is the mean-predictor RMSE).
   The exact IFT gradient is competitive but slightly worse on this hard
   regression: the sharp `lam/up` adjoint near flat crossings is erratic, while
   the smooth surrogate weighting earns a small, consistent advantage here.
2. Both rules share the spike-time deadlock: an output that goes silent
   contributes a constant to the SSE loss, so its upstream gradient is zero
   under either rule, and it can never be revived. `--dead-init` demonstrates
   this: both nets stall at ~0.53 (only the ~30% already-firing outputs can
   train). Reviving silent outputs needs a membrane/rate loss or the existence
   channel, not a surrogate.
3. The surrogate twin costs ~3x the exact wall-clock here (it solves the
   membrane margin for every layer), a cost not present in the exact library.

Keep default sizes small: the harness is CPU-bound by design (a 60-epoch run is
~1 minute).

## Compatibility contract
The package does not promise ONNX export or universal `torch.compile` compatibility.
All layers use standard parameters and can be composed with user-defined PyTorch
modules, optimizers, schedulers, datasets, and training code.
