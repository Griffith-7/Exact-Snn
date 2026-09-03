# Exact-SNN API Reference

`ExactSNN-nn` is a PyTorch plug-in. It supplies layers and mathematical helpers;
it does not own a dataset, application model, optimizer, scheduler, or training loop.

## Core API

Import from `exact_snn`:

### `ExactTTFSLinear(n_in, n_out, ...)`

A time-to-first-spike layer. Input shape is `(n_in, B)` and output shape is
`(n_out, B)`. Weights are stored as the public `nn.Parameter` `weight` with
shape `(n_out, n_in + 1)`; the final column is the bias channel.

`calibrate_init_fire(target=0.5, n_probe=32, cal_grid_pts=65)` optionally adjusts
the bias channel so a target fraction of a probe population fires.

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

## Compatibility contract

The supported path is eager PyTorch execution with Python 3.10+ and PyTorch 2.0+.
The package does not promise ONNX export or universal `torch.compile` compatibility.
All layers use standard parameters and can be composed with user-defined PyTorch
modules, optimizers, schedulers, datasets, and training code.
