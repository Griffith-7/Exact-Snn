"""Benchmark the optional event-driven layer against the grid layer.

Run from the project root:
    python benchmarks/benchmark_event.py

This reports measurements for the local machine. It intentionally does not
make a universal speed claim because results depend on hardware and workload.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from exact_snn import ExactTTFSLinear
from exact_snn.event import ExactEventLinear


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--outputs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--grid-pts", type=int, default=501)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    device = torch.device(args.device)
    grid = ExactTTFSLinear(args.features, args.outputs, grid_pts=args.grid_pts,
                           device=device)
    event = ExactEventLinear(args.features, args.outputs, device=device)
    event.load_state_dict(grid.state_dict(), strict=False)
    t_in = torch.rand(args.features, args.batch_size, device=device) * 20.0 + 0.1

    def measure(layer: torch.nn.Module) -> float:
        with torch.no_grad():
            for _ in range(3):
                layer(t_in)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.repeats):
                layer(t_in)
            if device.type == "cuda":
                torch.cuda.synchronize()
        return (time.perf_counter() - start) / args.repeats * 1000.0

    grid_ms = measure(grid)
    event_ms = measure(event)
    print(f"device={device} features={args.features} outputs={args.outputs} "
          f"batch={args.batch_size} grid_pts={args.grid_pts}")
    print(f"grid_ms={grid_ms:.3f} event_ms={event_ms:.3f} "
          f"speedup={grid_ms / event_ms:.2f}x")


if __name__ == "__main__":
    main()
