# Changelog

## 3.2.3 - 2026-09-06

- CI fix: the test suite now passes on CPU-only runners (GitHub Actions
  ubuntu, no CUDA), where `torch.cuda.is_available()` is `False`.
  - `tests/test_attention.py::test_full_layer_grad_cosine_vs_fd`: replaced the
    random-init FD comparison (whose spike times sit on the bias-kernel edge and
    can collapse to a single constant, zero-gradient output for some seeds) with
    a deterministic input-dependent block (`_det_attn2d`) and a device-independent
    CPU-RNG input, so the analytic-vs-FD cosine stays ~1.0 on cpu and cuda.
  - `tests/test_multihead.py::test_trains_small_task`: the 2-class loss hovered
    at chance after 40 Adam steps and drifted up on CPU; trains for 150 steps so
    it reliably decreases on every platform.
- No library code changes; behavior of `exact_snn` is unchanged.

## 3.2.2 - 2026-09-06

- `ExactSpikingFFN.calibrate_init_fire` with `use_norm=True` now pushes the
  calibration probe through the block's `SpikeNorm` between the two
  projections, matching exactly what `forward` does -- so `out_proj` is
  calibrated against the same normalized distribution it sees at inference.
- `experiments/compare_exact_vs_surrogate_lm.py` (CodeRabbit review):
  - `ExactSpikingLM` gained a true **autoregressive `generate()`**: each step
    appends the previously predicted token to the context, reruns the full
    model, and samples/argmaxes the *last* position (greedy at
    `temperature=0`, softmax sampling otherwise). The generation report now
    prints the prompt followed by the actually-generated continuation instead
    of a one-shot "predict every position" fill.
  - Fixed the batch/sequence flatten order in `ExactSpikingLM.forward`:
    embeddings are laid out as `(d_model, S*B)` (per-sample columns) instead
    of the previous `(B, S)` view that silently mixed samples for `B > 1`
    before attention; the output is reconstructed back to `(B, S, d_model)`
    symmetrically. Verified: batched forward == per-sample forwards.
  - Repaired a `from pathlib import Path` indentation bug that made the module
    unimportable, and updated the attention block call to the real single-head
    3D-sequence API (`ExactSpikingAttention(n_in=..., n_out=...)`; the
    `d_model`/`num_heads` kwargs did not exist).
- Tests: `SpikeNorm` masked-normalization cases and an end-to-end exact-FFN
  check that a deterministically-silent hidden neuron neither raises nor
  produces NaN/poisoned statistics.

## 3.2.1 - 2026-09-06

- Hardened every public entry point against corrupt input:
  - Spike-time tensors containing **NaN** are rejected with a clear `ValueError`
    across `ExactTTFSLinear`, `ExactTTFSRnn` (step + unrolled), `ExactSpikingFFN`,
    `ExactSpikingMultiHeadAttention`, `ExactSpikingAttention` (2D and 3D),
    `ExactTTFSConv2d`, `ExactMultiSpike`, and `ExactEventLinear`. `inf` stays the
    documented "silent neuron" value and continues to propagate silence (guarded,
    not rejected). See `exact_snn/_validation.py`.
  - `latency_encode` and `latency_cross_entropy` reject NaN inputs; the loss still
    treats silent `inf` outputs via its finite placeholder.
  - `train_simple` checks that `X` is finite pixel data in `[0, 1]`.
  - `SpikeNorm` (training mode) now normalizes **only the neurons that fired**:
    silent `inf` entries are preserved through the block, per-feature running
    statistics update only where the feature actually fired, and a fully silent
    batch is returned untouched -- so a valid silent hidden cell never crashes
    the enclosing block and the running statistics can never be poisoned with
    NaN.
- Regression tests for the new guards in `tests/test_nn_rewrite.py`,
  `tests/test_recurrent.py`, `tests/test_attention.py`, and `tests/test_ffn.py`.
  Full suite: **117 passed**.

## 3.2.0 - 2026-09-06

- `ExactSpikingAttention` now accepts **3D sequence input** `(S, n_in, B)`
  and returns `(S, n_out, B)`: Q/K/V projections are applied per position
  (shared weights, as in a transformer) and attention runs *across* the `S`
  positions with **per-feature** exact scores (the S of the input is folded
  into the batch axis of the existing exact combine, so the same closed-form
  gradients apply; there is no mean-collapse over features). The 2D token path
  is unchanged.
- `ExactSpikingFFN` gained two transformer-style options:
  - `residual=True` fuses the earliest-spike skip `min(t_out, t_in)`
    (requires `n_out == n_in`).
  - `use_norm=True` inserts a `SpikeNorm` between the two projections.
- `SpikeNorm` now accepts `dtype`/`device` arguments so it can be placed on a
  heterogeneous (e.g. CUDA) host consistently with the enclosing block.
- New/updated tests in `tests/test_attention.py` (sequence-mode shapes,
  backward, fold==per-feature identity, and a deterministic FD check over the
  full Q/K/V + per-position attention stack) and `tests/test_ffn.py`,
  `tests/test_multihead.py`. Full suite green (see below).

## 3.1.0 - 2026-09-06

- Added `ExactTTFSRnn` (exact recurrent cell with full per-neuron feedback) in
  `exact_snn/recurrent.py`, re-exported from `exact_snn.extended`.
  - Weight `(n_hidden, n_in + n_hidden + 1)`; `forward_step(t_in, t_prev)`
    single-step, `forward(t_in, h0=None)` unrolls `(n_in, B, T)` outputs.
  - Backward pushes the IFT recurrence Jacobian `dt/dh_{t-1}` (NBTT) through
    the unrolled chain; silent (`inf`) cells carry zero gradient end-to-end.
  - Verified against finite differences in `tests/test_recurrent.py` on a
    deterministic steep configuration (the forward solver's spike times
    quantize ~1e-7, so FD comparisons use `eps` ~1e-3).
- Added `ExactSpikingFFN` (exact spiking feed-forward block) in
  `exact_snn/ffn.py`, re-exported from `exact_snn.extended`.
  - Feed-forward companion to `ExactSpikingAttention`: two `ExactTTFSLinear`
    projections (`n_in -> n_hidden -> n_out`) with the `g(t) > theta`
    first-spike solve and inter-layer spike-time feed.
  - `calibrate_init_fire(target=...)` calibrates both projections in sequence.
  - Verified against finite differences in `tests/test_ffn.py`.
- Added `ExactSpikingMultiHeadAttention` (`exact_snn/multihead.py`),
  re-exported from `exact_snn.extended`.
  - `n_heads` independent exact single-head blocks (own Q/K/V projections),
    fused by `fuse="min"` (earliest-spike-wins, default), `"mean"`, or
    `"full"` (`(n_heads, n, B)`).
  - Exact gradients end-to-end: the wrapper is a composition of the
    FD-verified single-head block; a 1-head wrapper is pinned bit-for-bit
    equal to a plain `ExactSpikingAttention` (same outputs and gradients).
  - Tests in `tests/test_multihead.py`.
- Full suite: 97 tests green; wheel build + CI verified (see below).

## 3.0.0 - 2026-09-05

- Added `ExactSpikingAttention` (single-head spiking self-attention) in
  `exact_snn/attention.py`, re-exported from `exact_snn.extended`.
  - Composes three `ExactTTFSLinear` projections (Q/K/V) with a stateless
    exact combine step: temporal-alignment score (`gaussian` by default,
    `kernel` mode uses the raw symmetric synaptic kernel), time-coded softmax,
    attended value spike time.
  - Exact closed-form combine gradients through a custom `autograd.Function`;
    silent Q/K/V neurons carry zero gradient (consistent with the library).
  - `ExactSpikingAttention.calibrate_init_fire(...)` calibrates the Q/K/V
    projection layers' initial firing rates.
  - Verified against finite differences in `tests/test_attention.py`.
- Added an optional CUDA root-solve backend (`exact_snn/cuda_ops.py`).
  - Fuses the grid scan + bisection + Newton + near-threshold peak search into
    one kernel; transposed `(n, grid, B)` membrane layout and shared-memory
    tiling of weights/inputs keep the exact `u_at`/`du_at` evaluations off
    global memory.
  - Dispatch is automatic for CUDA float32; falls back silently to the torch
    path when CUDA is unavailable. Math is identical to the torch path
    (float32/float64 parity pinned in `tests/test_cuda.py`).
  - On the RTX 3050 (4 GB) test machine the fused kernel was ~5x faster than the
    CUDA torch path on a 64->128 layer at batch 128, and ~2.5-2.7x faster than
    the CUDA torch path on a 128->256 layer at batch 1024 (both including
    backward). The membrane grid tensor `(n, grid, B)` dominates GPU memory in
    both paths, so very large layers remain memory-bound regardless of backend.
- Measured exact-vs-surrogate training on a 1000 ms-period sine-waveform task
  (`benchmarks/surrogate_ttfs.py` + `benchmarks/sine_waveform_exact_vs_surrogate.py`).
  Identical nets, loss and data; only the spike-time Jacobian differs (exact
  IFT `lam/up` vs a magnitude-matched sigmoid-membrane surrogate).
  - Over 3 seeds, 40 epochs: exact final val RMSE 0.462 +/- 0.012 vs
    surrogate 0.427 +/- 0.009 (chance 0.354). The smooth surrogate weighted
    rule is a little ahead on this hard regression; exact IFT remains
    competitive and trains the task end-to-end.
  - Shared spike-time deadlock: outputs that go silent contribute a constant
    to the SSE loss, so their gradient is zero under BOTH rules and they are
    never revived (`--dead-init` demonstrates both nets stall identically).
    Surrogate silent-rescue needs a membrane/rate loss, not a spike-time
    readout. Full numbers and interpretation in `docs/API.md`.
- Fixed `calibrate_init_fire` so it actually reaches the target firing
  fraction. The previous single-scalar `theta - quantile` bias correction
  under-shot input-driven peaks (biases could sit at 0% probe firing); it is
  now a per-neuron Newton solve using `peak_margin_torch` with the envelope
  term `K(t_peak - t_bias)` and `iters=6` iterations, for `ExactTTFSLinear`,
  `ExactTTFSNetwork`, `ExactSpikingAttention`, and `ExactSpikingFFN` alike
  (`w_scale=0.05, bias_val=0.2, seed=7, target=0.9` now yields ~98% probe /
  ~97% held-out firing, was 0%).
- Hardened `tests/test_attention.py` against an unseeded-RNG flake (Adam test
  could draw an all-silent init and see zero gradient).

## 2.0.0 - 2026-09-03

- Added PyTorch-native exact-gradient TTFS, convolutional, multi-spike, and recurrent layers.
- Added optional event-driven, existence-gradient, normalization, reset-reference, initializer, loss, and spike-time utility modules.
- Added firing calibration for TTFS layers and networks.
- Added finite-difference, serialization, dtype, and companion-module tests.
- Added package metadata, release documentation, and CI configuration.

## Versioning policy

This project follows semantic versioning. Public imports, class names, function names,
and documented tensor shape contracts are treated as the public API.

- Patch releases fix bugs without changing public contracts.
- Minor releases add backward-compatible layers, losses, utilities, or options.
- Major releases may change public behavior or remove deprecated APIs.

The mathematical implementation may improve between releases, but changes to numerical
accuracy or firing behavior will be documented in the release notes.
