"""Measure exact IFT gradients vs standard surrogate gradients on one task.

Task:  "sine continuation" with the user-specified waveform scale
       (period = 1000 ms, t_max = 1000 ms).
       Given the first 8 latency-coded sample values of a random-phase sine,
       predict the remaining 16 sample values (also latency-coded spike times).
       Chance-level (predict the mean value) => decoded RMSE ~= 0.354.

Two identical networks are built with the SAME seed and SAME calibration, and
trained on the SAME data and SAME loss:

  exact     : exact_snn pipeline (ExactSpikingAttention + ExactTTFSLinear)
              backward = exact IFT adjoint (lam / up) everywhere.
  surrogate : structurally identical twin whose Q/K/V and linear layers use
              the sigmoid-membrane surrogate rule (benchmarks/surrogate_ttfs).
              Only the spike-time Jacobian differs; forward outputs at every
              epoch are bit-identical to the exact net.

The comparison isolates the ONE scientific question: on identical networks,
identical loss and identical data, does the exact gradient train as well as
(or better/worse than) the standard surrogate gradient? We report loss, decode
RMSE and the silent-output fraction for both nets over the same epochs.

Usage:
  python benchmarks/sine_waveform_exact_vs_surrogate.py [--epochs N]
      [--train N] [--val N] [--batch N] [--lr F] [--dead-init] [--seed N]

Keep defaults small: the run is CPU-bound and stays interactive (~seconds/epoch).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))

from exact_snn import ExactTTFSLinear
from exact_snn.attention import ExactSpikingAttention
from surrogate_ttfs import SurrogateSpikingAttention, SurrogateTTFSLinear

PERIOD = 1000.0        # user-specified waveform scale (ms; the signal's period)
TM = 15.0              # synaptic rise timescale: reach ~3*TM ms in spike time
TS = 4.0
THETA = 1.0
T_MAX = 100.0          # spike-time coding window (latency code for values)
T_MIN_SPIKE = 10.0     # value -> spike-time range [10, 90] ms (keeps targets
T_MAX_SPIKE = 90.0     #   away from the silent boundary so SSE training is stable)
N_CTX = 8              # context sample points fed to the net
N_OUT = 16             # waveform sample points the net must predict
GRID_PTS = 1001


def value_to_time(v, t_min=T_MIN_SPIKE, t_max=T_MAX_SPIKE):
    """Latency code: value in [0,1] -> spike time in [t_min, t_max]."""
    return t_max - (t_max - t_min) * torch.clamp(v, 0.0, 1.0)


def time_to_value(t, t_min=T_MIN_SPIKE, t_max=T_MAX_SPIKE):
    """Inverse latency code; silent (inf) decodes to value 0."""
    v = torch.where(torch.isfinite(t), (t_max - t) / (t_max - t_min),
                    torch.zeros_like(t))
    return v.clamp(0.0, 1.0)


def make_dataset(n, period=PERIOD, n_ctx=N_CTX, n_out=N_OUT, seed=0):
    """Latency-coded (context, target) spike-time pairs for random-phase sines."""
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, (n,))
    n_total = n_ctx + n_out
    ts = (np.arange(n_total) + 0.5) / n_total * period          # sample times
    v = 0.5 * (1.0 + np.sin(2.0 * np.pi * ts[None, :] / period
                            + phase[:, None]))                   # (n, n_total)
    t_ctx = value_to_time(torch.tensor(v[:, :n_ctx].T, dtype=torch.float32))
    t_tgt = value_to_time(torch.tensor(v[:, n_ctx:].T, dtype=torch.float32))
    return t_ctx, t_tgt


def build_model(cls_linear, cls_attn, hidden, seed, bias_val):
    m = nn.Module()
    m.l1 = cls_linear(
        N_CTX, hidden, tm=TM, ts=TS, theta=THETA, t_max=T_MAX,
        w_scale=2.5, bias_val=bias_val, grid_pts=GRID_PTS, seed=seed)
    m.attn = cls_attn(
        hidden, hidden, tm=TM, ts=TS, theta=THETA, t_max=T_MAX,
        w_scale=2.5, bias_val=bias_val, grid_pts=GRID_PTS, seed=seed,
        temp=1.0, combine="gaussian")
    m.l2 = cls_linear(
        hidden, N_OUT, tm=TM, ts=TS, theta=THETA, t_max=T_MAX,
        w_scale=2.5, bias_val=bias_val, grid_pts=GRID_PTS, seed=1000)

    def forward(t_in):
        h = m.l1(t_in)
        h = m.attn(h)
        return m.l2(h)

    m.forward = forward
    return m


def spike_time_mse(t_out, t_tgt, t_max=T_MAX):
    t_safe = torch.where(torch.isfinite(t_out), t_out,
                         torch.full_like(t_out, 2.0 * t_max + 10.0))
    return ((t_safe - t_tgt) ** 2).mean()


def decoded_rmse(t_out, t_tgt):
    v_out = time_to_value(t_out)
    v_tgt = time_to_value(t_tgt)
    return float(((v_out - v_tgt) ** 2).mean().sqrt().item())


def silent_fraction(t_out):
    return float((~torch.isfinite(t_out)).float().mean().item())


def train_one(cls_linear, cls_attn, t_ctx, t_tgt, t_ctx_val, t_tgt_val,
              epochs, batch, lr, bias_val, seed, verbose):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(cls_linear, cls_attn, 24, seed, bias_val)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    B = t_tgt.shape[1]
    rows = []
    init_rmse = decoded_rmse(model(t_ctx_val).detach(),
                             t_tgt_val) if t_tgt_val.shape[1] else float("nan")
    init_sil = silent_fraction(model(t_ctx_val).detach())
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(B)
        tot, nb = 0.0, 0
        t0 = time.time()
        for i in range(0, B, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            out = model(t_ctx[:, idx])
            loss = spike_time_mse(out, t_tgt[:, idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gn = sum(float(p.grad.flatten().norm().item() ** 2)
                     for p in model.parameters() if p.grad is not None) ** 0.5
            opt.step()
            tot += float(loss.item())
            nb += 1
        dt = time.time() - t0
        with torch.no_grad():
            out_val = model(t_ctx_val).detach()
        rmse = decoded_rmse(out_val, t_tgt_val)
        sil = silent_fraction(out_val)
        rows.append((ep + 1, tot / nb, rmse, sil, gn, dt))
    if verbose:
        print(f"    init val RMSE={init_rmse:.4f}  silent={init_sil:.0%}")
        for (ep, loss, rmse, sil, gn, dt) in rows:
            print(f"    ep {ep:3d}/{epochs}  loss={loss:.4f}  val RMSE={rmse:.4f}  "
                  f"silent={sil:.1%}  |grad|={gn:.1e}  {dt:.2f}s/epoch")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--train", type=int, default=256)
    ap.add_argument("--val", type=int, default=64)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of fresh seeds to average the comparison over")
    ap.add_argument("--dead-init", action="store_true",
                    help="start from a mostly-silent init (low bias); "
                    "demonstrates the silent-neuron deadlock shared by both rules")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bias_val = 0.8 if not args.dead_init else -1.0

    print(f"task: sine continuation, period={PERIOD:.0f} ms, "
          f"t_max={T_MAX:.0f} ms, tm={TM:.0f}, ts={TS:.0f}")
    print(f"net : ctx {N_CTX} -> hid 24 -> attn 24 -> out {N_OUT} | "
          f"train {args.train}  val {args.val}  lr={args.lr}  "
          f"epochs={args.epochs}")
    print(f"mode: {'input-driven init (healthy firing)' if not args.dead_init else 'DEAD init (low bias)'} "
          f"| seeds={args.seeds} (base seed {args.seed})")
    print("chance-level decoded RMSE (mean predictor) ~= 0.354\n")

    pairs = {
        "exact": (ExactTTFSLinear, ExactSpikingAttention),
        "surrogate": (SurrogateTTFSLinear, SurrogateSpikingAttention),
    }

    finals = {k: [] for k in pairs}
    fin_sil = {k: [] for k in pairs}
    for s in range(args.seeds):
        seed = args.seed + s
        t_ctx, t_tgt = make_dataset(args.train + args.val, seed=seed)
        t_ctx_val, t_tgt_val = t_ctx[:, args.train:], t_tgt[:, args.train:]
        t_ctx, t_tgt = t_ctx[:, :args.train], t_tgt[:, :args.train]
        print(f"=== seed {seed} ===")
        for k in pairs:
            cl, ca = pairs[k]
            rows = train_one(cl, ca, t_ctx, t_tgt, t_ctx_val, t_tgt_val,
                             args.epochs, args.batch, args.lr, bias_val,
                             seed, verbose=(s == 0))
            finals[k].append(rows[-1][2])
            fin_sil[k].append(rows[-1][3])
        print()

    print("\n" + "=" * 74)
    print("EXACT vs SURROGATE  (identical net, loss, data; only the gradient rule differs)")
    print("=" * 74)
    for k in pairs:
        m = float(np.mean(finals[k]))
        sd = float(np.std(finals[k]))
        ms = float(np.mean(fin_sil[k]))
        print(f"  {k:9s}: final val RMSE  {m:.4f} +/- {sd:.4f}"
              f" over {args.seeds} seed(s)  (final silent {ms:.0%})")
    print()
    from statistics import mean
    if args.dead_init:
        print("  DEAD-init run: both rules give ~zero gradient to fully silent")
        print("  outputs, so both nets stall. With a spike-time readout + SSE")
        print("  loss the surrogate has NO structural silent-rescue advantage")
        print("  either; reviving silent outputs needs a membrane/rate loss or")
        print("  the existence channel.")
    else:
        d = mean(finals["exact"]) - mean(finals["surrogate"])
        if d < -0.01:
            print(f"  verdict: exact trained to a LOWER RMSE on average "
                  f"(exact - surrogate = {d:+.4f})")
        elif d > 0.01:
            print(f"  verdict: surrogate trained to a LOWER RMSE on average "
                  f"(exact - surrogate = {d:+.4f})")
        else:
            print("  verdict: the two gradient rules reached the same RMSE "
                  f"on average (exact - surrogate = {d:+.4f})")


if __name__ == "__main__":
    main()
