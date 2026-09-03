# ExactSNN-nn

Exact, closed-form gradients for spiking neural networks — packaged as a
drop-in `torch.nn` library. No surrogate gradients for firing neurons, no
membrane ODE simulation, and no custom CUDA kernels: every `loss.backward()`
flows through **implicit-function-theorem (IFT)** and **saltation** matrix
gradients of the spike-time maps.

The layered models (full-featured exact conv, multi-spike, recurrent) are
faster and more memory-efficient than the original [Exact-SNN]
implementation: vectorized in pure PyTorch/torch.autograd (no C++), with the
bottlenecks (the bisection root-finds and the peak searches) accelerated by
grid interpolation of the sampled membrane voltage.

[Exact-SNN]: https://github.com/Noxsios/Exact-SNN

## Features

- **Four layer types**, all exact-gradient `nn.Module`s:
  - `ExactTTFSLinear` — time-to-first-spike (TTFS) fully-connected
  - `ExactTTFSConv2d` — TTFS 2-D convolution (unfold → IFT → fold)
  - `ExactMultiSpike` — multi-spike neuron; full spike train `(n_out, B, K)`
  - `ExactRecurrent` — recurrent TTFS layer with a shared feedback eligibility trace
- **Exact backward** through spike times (IFT for the first spike, saltation
  chain for subsequent resets) — the gradient is mathematical, not a proxy.
- **Big-batch**, pure-`torch` vectorized forward (no Python per-kernel loops):
  `U_base` built with one `matmul`, bisection done via `_interp_grid`
  (linear interpolation of the grid-sampled membrane), Newton refined only
  for the final sub-grid precision.
- **Memory-conscious**: the multi-spike K-loop reuses a single membrane
  buffer in place instead of allocating fresh `(n_cur, B, G)` grids each
  spike, cutting peak memory sharply at large `(B, G)`.
- **Autograd-integrated** — plug into any `torch.optim`, `nn.Sequential`, etc.

## Installation

Requires Python 3.10+, PyTorch ≥ 2.0.

```bash
pip install -e .
```

The full public API is documented in [docs/API.md](docs/API.md). Release notes
and the versioning policy are in [CHANGELOG.md](CHANGELOG.md).

The package is a single importable module:

```python
from exact_snn import ExactTTFSLinear, latency_encode, latency_cross_entropy
from exact_snn.extended import (
    ExactTTFSConv2d, ExactMultiSpike, ExactRecurrent,
    spike_count_cross_entropy, multispike_latency_loss,
)
```

## Quick start

### TTFS MLP (latency-coded MNIST)

```python
import torch
from exact_snn import ExactTTFSLinear, latency_encode, latency_cross_entropy

t_in  = latency_encode(X.T, t_max=40.0)   # (784, N) spike-time input
model = torch.nn.Sequential(
    ExactTTFSLinear(784, 128, t_max=40.0, bias_val=1.5),
    ExactTTFSLinear(128,   10, t_max=40.0, bias_val=1.5),
)

for xb, yb in batches:                    # xb: (784, B), yb: (B,)
    loss = latency_cross_entropy(model(xb), yb)
    loss.backward()                       # exact IFT gradients
    optimizer.step()
```

### Conv SNN

```python
from exact_snn.extended import ExactTTFSConv2d

conv = ExactTTFSConv2d(1, 8, kernel_size=3, stride=1, padding=1,
                       t_max=40.0, w_scale=0.35, bias_val=0.5, grid_pts=301)
t_conv = conv(xb)                         # (B, 8, H, W) first-spike map
```

### Multi-spike network

```python
from exact_snn.extended import ExactMultiSpike, spike_count_cross_entropy

layer = ExactMultiSpike(196, 64, t_max=40.0, max_spikes=4, first_spike_only=True)
t_all = layer(t_in)                       # (64, B, K) full spike train
loss  = spike_count_cross_entropy(t_all, y, t_max=40.0)
```

## Public imports

```python
# Core exact-SNN layers, encoding and the timing loss
from exact_snn import (
    ExactTTFSLinear,      # (n_in, B) -> (n_out, B) first-spike layer
    ExactTTFSNetwork,     # small nn.Sequential convenience wrapper
    latency_encode,       # input -> latency-coded spike times
    latency_cross_entropy,
    train_simple,         # optional convenience helper (not mandatory)
)

# Extended layers + multi-spike losses (feedforward, conv, multi-spike, recurrent)
from exact_snn.extended import (
    ExactTTFSConv2d,
    ExactMultiSpike,
    ExactRecurrent,
    spike_count_cross_entropy,
    multispike_latency_loss,
)
```

All modules are pure `torch.nn` components: bring your own `nn.Module`
composition, dataset, `torch.optim` optimizer, and training loop.

## Layer reference

### `ExactTTFSLinear(n_in, n_out, ...)`
- Input `(n_in, B)` spike times → output `(n_out, B)` first-spike times.
- `tm, ts, theta, t_max, w_scale, bias_val, grid_pts, seed, dtype, device`.

### `ExactTTFSConv2d(in_channels, out_channels, kernel_size, ...)`
- Input `(B, C, H, W)` spike-time map → output `(B, C_out, H_out, W_out)`.
- `stride, padding, tm, ts, theta, t_max, w_scale, bias_val, grid_pts, peak_tol`.
- Internally unfolds patches, solves the first-spike IFT per patch, folds back.

### `ExactMultiSpike(n_in, n_out, ...)`
- Input `(n_in, B)` → output `(n_out, B, K)` full spike train (K = `max_spikes`).
- `first_spike_only=True` limits the backward to the reset slot 0 (TTFS-style,
  stable); `False` runs the full saltation chain (rate/count-style).
- Backward needs `t_all`; to feed a multi-spike output into a single-spike
  layer, use the first spike `t_all[:, :, 0]`.

### `ExactRecurrent(n_in, n_out, ...)`
- Recurrent TTFS layer with shared-feedback eligibility trace; see
  `forward_step` and `reset_state`.

### Losses
- `latency_cross_entropy(t_out, y)` — cross-entropy on first-spike latencies.
- `spike_count_cross_entropy(t_all, y, t_max)` — soft spike-count cross-entropy
  over the full spike train, differentiable w.r.t. every spike time.
- `multispike_latency_loss(t_all, y, t_max)` — latency loss on the full train.

## Optional companion modules

The core package stays small and framework-free. These are **optional,
independent, lazy opt-in** modules (import them only if you need them):

```python
from exact_snn import existence      # silent-neuron existence gradients
from exact_snn import normalize      # SpikeNorm
from exact_snn import losses         # rate_latency_loss
from exact_snn import initializers   # xavier_init / kaiming_init
from exact_snn import util           # spike_time_augment
from exact_snn import reset          # ResetLIF (reference solver)
from exact_snn.event import ExactEventLinear   # event-driven drop-in layer
```

- **`existence`** — `peak_margin_torch`, `edge_peak_guard`,
  `existence_loss_and_grads`. Revives silent neurons (escape-noise peak-margin
  model) whose exact IFT timing gradient is otherwise zero. Add the returned
  weight gradients to `layer.weight.grad` after the normal `loss.backward()`.
  Verified against finite differences on targeted silent neurons.
- **`normalize.SpikeNorm`** — batch normalization adapted for spike times
  (`(n_features, B)` tensors), with `gamma`/`beta` as `nn.Parameter`.
- **`losses.rate_latency_loss`** — combined spike-count + first-spike latency CE.
- **`initializers`** — `xavier_init` / `kaiming_init` that write into an
  existing layer weight tensor `(fan_out, fan_in+1)`.
- **`util.spike_time_augment`** — additive Gaussian noise + random time shift,
  clamped to `[0, t_max]`.
- **`reset.ResetLIF`** — a standalone, dependency-free multi-spike LIF reference
  solver with hard reset + saltation jump map (`run`, `sensitivity`,
  `sensitivity_all`, `sensitivity_first_spike`, `state_at`). A scalar-level
  oracle for the saltation math; not an `nn.Module`.
- **`event.ExactEventLinear`** — a drop-in alternative to `ExactTTFSLinear` that
  solves the spike-time forward from the inter-event closed form of the kernel
  (no dense grid scan). Same weight shape and interface; benchmark it on your
  workload before relying on a speedup claim.

**Calibration** — `ExactTTFSLinear.calibrate_init_fire()` (and the network-level
`.calibrate_init_fire()` on `ExactTTFSNetwork`) adjusts each layer's bias so a
target fraction of neurons fire on random input at init, preventing a silent
"dead-on-arrival" network. Call it once after constructing a model.

These components add mathematical capability as plug-in pieces; they do **not**
impose a training loop, dataset, optimizer, scheduler, or model pipeline.
The custom autograd functions are tested with the supported PyTorch eager
execution path; `torch.compile` and ONNX export are not promised by this
package.

## How the gradients are exact

For a single neuron with membrane `u(t)`, its spike time `t*` satisfies
`u(t*) = theta`. Differentiating gives the IFT relation used in the backward
pass (exact up to the tolerance of the Newton root-find):

```
dt*/dW = -(du/dW) / (du/dt)   at t = t*
```

For multi-spike reset dynamics, the backward chains saltation matrices across
each reset so that the *total* spike-time map gradient is exact:

```
dU|resets  = S . dU|pre          (saltation matrix across the discontinuity)
```

The forward evaluates spike times directly (no membrane ODE integration), and
`U` on the grid is reused via interpolation — this is what makes the layers
fast in torch while keeping gradients exact.

## Example / demo

```bash
python examples/test_all_layers.py            # all four layers on small MNIST
python examples/test_all_layers.py --cuda     # force GPU (default: CPU)
```

Result summary (small MNIST subsets, short schedules — the aim is exact
gradients that **work end-to-end past chance**, not SOTA accuracy):

| Layer | Result on MNIST |
|-------|-----------------|
| TTFS MLP        | ~72% (985 samples, 8 epochs) |
| Conv SNN        | ~36% on 14×14 (1000 samples, 35 epochs, chance 10%) |
| Multi-spike     | >50% on 3-class (chance 33%) |
| Recurrent       | eligibility trace builds up; exact single-step gradients |

The demo defaults to CPU for a robust, reproducible run — the conv and
multi-spike layers build large internal `(batch × grid)` saltation grids that
can trip a small (≤4 GB) GPU's WDDM driver during training.

**On the conv ceiling.** The conv climbs each epoch (30% → 36% by 35 epochs)
and its plateau is a property of the task setup, not a training/capacity bug:
with every other setting fixed, doubling the conv channels (8 → 16) makes
accuracy *worse* (31.7% vs 36.2%). This matches the original Exact-SNN
project's own observation that training rate/latency-coded SNNs on tiny MNIST
subsets is genuinely hard — the library's purpose is to provide **exact**
gradients (which it does), not to chase SOTA on an under-data regime.

## Tests

```bash
pytest tests/
```

Tests cover the core IFT/conv backward (FD cosine comparison against

## Benchmarking

The optional event-driven layer can be compared with the grid layer on local
hardware:

```bash
python benchmarks/benchmark_event.py
```

The command reports measured milliseconds and speedup for the selected workload;
performance is hardware- and batch-size-dependent.
numerical gradients on smooth weights), layer forward/backward shapes, the
multi-spike saltation backward, and a regression test that the **vectorized**
multi-spike forward matches the exact all-recompute reference. The optional
companion modules (`existence`, `normalize`, `losses`, `initializers`, `util`,
`reset`, `event`) each have focused tests including a finite-difference check
of the silent-neuron existence gradients.
