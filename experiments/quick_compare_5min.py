"""Quick 5-minute backend comparison for single-monomial partial derivatives.

Compares the three implemented derivative backends of Section 4.2 of the
paper on a representative sweep of multi-indices. Designed to fit in well
under 5 minutes of wall-clock so the user can sanity-check accuracy and timing
without launching the full benchmark suite.

Outputs:
  - per-pattern wall-clock per single deriv call (median over 60 timed reps)
  - relative-error against direct nested autograd (the exact reference)
  - direction count for each backend

Run with:
    bash scripts/run_quick_compare.sh
"""
from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from apolarity import (
    monomial_waring_directions,
    polarization_directions,
    single_monomial_partial,
)


class SinhActivation(nn.Module):
    def forward(self, x):
        return torch.sinh(x)


def build_mlp(d: int, hidden: int, depth: int, *, dtype: torch.dtype, device) -> nn.Sequential:
    layers = []
    in_dim = d
    for _ in range(depth):
        layers += [nn.Linear(in_dim, hidden), SinhActivation()]
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers).to(device=device, dtype=dtype)


@dataclass
class Result:
    backend: str
    pattern: tuple
    alpha: tuple
    p: int
    R_C: int
    R_pol: int
    median_ms: float
    rel_err: float


def time_call(fn, n_warm: int = 5, n_rep: int = 60):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def reference_direct(model, x, alpha):
    """Tensor reference computed by direct nested autograd (B1)."""
    return single_monomial_partial(model, x, alpha, backend="direct_autodiff")


def relerr(a, b):
    a = a.detach()
    b = b.detach()
    return (a - b).abs().max().item() / (b.abs().max().item() + 1e-30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=60)
    parser.add_argument("--complex-net", action="store_true",
                        help="If set, use complex-parameter network natively")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    real_dtype = torch.float64 if args.dtype == "float64" else torch.float32
    net_dtype = (torch.complex128 if args.dtype == "float64" else torch.complex64) if args.complex_net else real_dtype

    model = build_mlp(args.d, args.hidden, args.depth, dtype=net_dtype, device=device)
    if args.complex_net:
        x_real = torch.randn(args.batch, args.d, device=device, dtype=real_dtype)
        x = x_real.to(net_dtype)
    else:
        x = torch.randn(args.batch, args.d, device=device, dtype=real_dtype)

    # Patterns chosen to span (a) pure power, (b) min-exp=1, (c) repeated balanced,
    # (d) square-free. Each entry is the expanded zero-based multi-index alpha.
    patterns = [
        ("(3) pure",       (0, 0, 0)),
        ("(2,1) min=1",    (0, 0, 1)),
        ("(1,1,1) sqf",    (0, 1, 2)),
        ("(4) pure",       (0, 0, 0, 0)),
        ("(2,2) binary",   (0, 0, 1, 1)),
        ("(3,1) min=1",    (0, 0, 0, 1)),
        ("(6) pure",       (0,)*6),
        ("(4,2) repeat",   (0,0,0,0,1,1)),
        ("(2,2,2) repeat", (0,0,1,1,2,2)),
        ("(1^6) sqf",      (0,1,2,3,4,5)),
    ]

    print(f"# device   : {torch.cuda.get_device_name(0) if device.type=='cuda' else 'cpu'}")
    print(f"# d={args.d}  batch={args.batch}  hidden={args.hidden}  depth={args.depth}  dtype={args.dtype}  complex_net={args.complex_net}  warmup={args.warmup}  reps={args.reps}")
    print(f"#")
    header = f"{'pattern':<18s} {'p':>2} {'R_C':>4} {'R_pol':>5}  " \
             f"{'B1 direct':>10s} {'B2 polariz':>10s} {'B3 wC':>10s}  " \
             f"{'errB2':>8s} {'errB3':>8s}"
    print(header)
    print("-" * len(header))

    results = []
    for label, alpha in patterns:
        p = len(alpha)
        # Direction counts
        Vc, _, info = monomial_waring_directions(alpha, args.d, device=device,
                                                 dtype=torch.complex128 if real_dtype==torch.float64 else torch.complex64)
        R_C = info.rank
        Vp, _, infop = polarization_directions(alpha, args.d, device=device, dtype=real_dtype)
        R_pol = Vp.shape[0]

        # Reference
        try:
            ref = reference_direct(model, x, alpha)
        except Exception as e:
            print(f"{label:<18s} skipped (reference failed): {e}")
            continue

        # Time each backend
        t_b1 = time_call(lambda: reference_direct(model, x, alpha), args.warmup, args.reps)
        t_b2 = time_call(lambda: single_monomial_partial(model, x, alpha, backend="polarization_jet"), args.warmup, args.reps)
        try:
            t_b3 = time_call(lambda: single_monomial_partial(model, x, alpha, backend="waring_complex_jet"), args.warmup, args.reps)
            err_b3 = relerr(single_monomial_partial(model, x, alpha, backend="waring_complex_jet"), ref)
        except Exception as e:
            t_b3, err_b3 = float("nan"), float("nan")

        err_b2 = relerr(single_monomial_partial(model, x, alpha, backend="polarization_jet"), ref)

        print(f"{label:<18s} {p:>2d} {R_C:>4d} {R_pol:>5d}  "
              f"{t_b1:>10.3f} {t_b2:>10.3f} {t_b3:>10.3f}  "
              f"{err_b2:>8.1e} {err_b3:>8.1e}")

        results.append(Result(
            backend="all", pattern=tuple(label.split()[0]), alpha=alpha, p=p,
            R_C=R_C, R_pol=R_pol,
            median_ms=t_b3, rel_err=err_b3,
        ))

    print("\n# Notes")
    print(f"# - Times are MEDIAN ms over {args.reps} timed reps after {args.warmup} warm-ups (single deriv call).")
    print("# - errBk = max relative error of backend Bk vs direct nested autograd reference.")
    print("# - R_C < R_pol indicates a direction-count advantage, not a guaranteed wall-clock win.")


if __name__ == "__main__":
    main()
