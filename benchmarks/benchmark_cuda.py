"""Benchmark the CUDA root-solve backend against the torch grid path.

Compares per-call wall time of a single ExactTTFSLinear forward (+ backward)
across:
    * cpu + torch path
    * cuda + torch path   (cuda_ops disabled)
    * cuda + native kernel (cuda_ops enabled)

Run from the project root:
    python benchmarks/benchmark_cuda.py [--repeats 20] [--grid-pts 1001]

This reports measurements for the local machine. It intentionally does not make
a universal speed claim because results depend on hardware and workload. The
native kernel is JIT-compiled on first use (takes a few minutes) and then
cached on disk by torch's cpp_extension.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from exact_snn import ExactTTFSLinear
from exact_snn import cuda_ops

SIZES = [(64, 128, 128), (128, 256, 1024), (256, 1024, 4096)]


def _measure(layer: ExactTTFSLinear, t_in: torch.Tensor, repeats: int,
             with_backward: bool) -> float:
    dev = t_in.device
    with torch.no_grad():
        for _ in range(3):
            layer(t_in)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        out = layer(t_in)
    if with_backward:
        t_out = out
        mask = torch.isfinite(t_out)
        if mask.any():
            loss = (t_out[mask] / 40.0).sum()
            layer.zero_grad(set_to_none=True)
            loss.backward()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1000.0


def _run_grid(device: str, dtype: torch.dtype, n_in: int, n_out: int,
              batch: int, grid_pts: int, repeats: int, with_backward: bool,
              use_cuda_kernel: bool) -> float:
    layer = ExactTTFSLinear(n_in, n_out, tm=15.0, ts=4.0, theta=1.0,
                            t_max=40.0, grid_pts=grid_pts, dtype=dtype,
                            device=device)
    layer.calibrate_init_fire(target=0.5)
    t_in = (0.05 + torch.rand(n_in, batch, dtype=dtype, device=device)
            * 0.9 * 40.0)
    return _measure(layer, t_in, repeats, with_backward)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--grid-pts", type=int, default=1001)
    parser.add_argument("--fwd-only", action="store_true")
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    if args.verbose:
        print("CUDA extension status:", cuda_ops.status())

    with_backward = not args.fwd_only
    cpu_col = not args.skip_cpu
    print(f"repeats={args.repeats} grid_pts={args.grid_pts} "
          f"fwd_only={args.fwd_only}")
    hdr = "n_in n_out batch | "
    if cpu_col:
        hdr += "cpu_torch | "
    hdr += "cuda_torch | cuda_kernel | kernel-vs-cuda_torch"
    if cpu_col:
        hdr += " | kernel-vs-cpu"
    print(hdr, flush=True)

    was = cuda_ops.is_enabled()
    for n_in, n_out, batch in SIZES:
        rows = []
        if cpu_col:
            rows.append(("cpu_torch", _run_grid(
                "cpu", torch.float32, n_in, n_out, batch, args.grid_pts,
                args.repeats, with_backward, False)))
        cuda_ops.set_enabled(False)
        rows.append(("cuda_torch", _run_grid(
            "cuda", torch.float32, n_in, n_out, batch, args.grid_pts,
            args.repeats, with_backward, False)))
        cuda_ops.set_enabled(True)
        rows.append(("cuda_kernel", _run_grid(
            "cuda", torch.float32, n_in, n_out, batch, args.grid_pts,
            args.repeats, with_backward, True)))
        cuda_ops.set_enabled(was)

        d = dict(rows)
        out = f"{n_in:4d} {n_out:4d} {batch:5d} | "
        if cpu_col:
            out += f"{d['cpu_torch']:8.3f} | "
        out += f"{d['cuda_torch']:8.3f} | {d['cuda_kernel']:8.3f} | " \
               f"{d['cuda_torch'] / d['cuda_kernel']:6.2f}x"
        if cpu_col:
            out += f" | {d['cpu_torch'] / d['cuda_kernel']:6.2f}x"
        print(out, flush=True)


if __name__ == "__main__":
    main()