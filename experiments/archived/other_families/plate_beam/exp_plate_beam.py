#!/usr/bin/env python3
"""4th-order real oscillatory eigenmodes -- the real baselines' home turf.

Kirchhoff plate vibration (2D biharmonic eigenmode):
    Delta^2 w = S^2 w,   w = sin(m*pi*x) sin(n*pi*y),   S = (m^2+n^2) pi^2,
    simply-supported (Navier) BC: w = 0, Delta w = 0 on the boundary.

Euler-Bernoulli beam vibration (1D, 4th order):
    w'''' = (m*pi)^4 w,   w = sin(m*pi*x),
    simply-supported BC: w = 0, w'' = 0 at the ends.

Sweep the mode number m (= n) to raise the oscillation while keeping the order
fixed at 4.  Real "high-frequency SOTA" baselines (Fourier, SIREN, MscaleDNN)
are frequency-matched to the complex sinh net (literal width; see width study).
Acceptance: complex advantage grows with the mode number.

Run:
  python experiments/exp_plate_beam.py --out results/plate_beam.csv
"""
from __future__ import annotations

import math

import torch

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
from osc_common import LinearProblem, laplacian_power_terms, run_linear_suite, default_argparser


def _plate_u(m, n):
    def u(x):
        return torch.sin(m * math.pi * x[..., 0]) * torch.sin(n * math.pi * x[..., 1])
    return u


def _beam_u(m):
    def u(x):
        return torch.sin(m * math.pi * x[..., 0])
    return u


def _plate_problem(m, n, tag):
    S = (m * m + n * n) * math.pi ** 2              # -Delta eigenvalue
    up = _plate_u(m, n)
    fmax = max(m, n)
    return LinearProblem(
        name=tag, d=2, order=4,
        terms=laplacian_power_terms(2, 2), zeroth=0.0,
        u_exact=up, source_f=(lambda u=up, S=S: (lambda x: (S ** 2) * u(x)))(),
        res_scale=S ** 2, S=S, bc_lap_powers=(1,), sweep=float(fmax),
        extra={"omega0": max(10.0, 2.0 * math.pi * fmax),
               "fourier_sigma": max(2.0, math.pi * fmax)},
    )


def make_mixed_problems(modes=(2, 3, 4)):
    """Anisotropic (m, n=m+1) plate modes: oscillation differs per axis at fixed
    order 4 -- a harder, non-separable-frequency variant of the plate sweep."""
    return [_plate_problem(m, m + 1, f"platemix_m{m}") for m in modes]


def make_problems(modes=(1, 2, 3)):
    probs = []
    for m in modes:
        # --- Kirchhoff plate (2D biharmonic), isotropic mode (m, m) ---
        probs.append(_plate_problem(m, m, f"plate_m{m}"))
        # --- Euler-Bernoulli beam (1D, 4th order) ---
        Sb = (m * math.pi) ** 2
        ub = _beam_u(m)
        probs.append(LinearProblem(
            name=f"beam_m{m}", d=1, order=4,
            terms=[(1.0, (0, 0, 0, 0))], zeroth=0.0,
            u_exact=ub, source_f=(lambda u=ub, Sb=Sb: (lambda x: (Sb ** 2) * u(x)))(),
            res_scale=Sb ** 2, S=Sb, bc_lap_powers=(1,), sweep=float(m),
            extra={"omega0": max(10.0, 2.0 * math.pi * m),
                   "fourier_sigma": max(2.0, math.pi * m)},
        ))
    return probs


if __name__ == "__main__":
    ap = default_argparser(seconds=90.0)
    ap.add_argument("--modes", default="1,2,3")
    ap.add_argument("--kind", default="both", choices=["both", "plate", "beam", "mix"])
    args = ap.parse_args()
    modes = [int(s) for s in args.modes.split(",") if s]
    variants = [v for v in args.variants.split(",") if v]
    if args.kind == "mix":
        probs = make_mixed_problems(modes)
    else:
        probs = make_problems(modes)
        if args.kind == "plate":
            probs = [p for p in probs if p.name.startswith("plate")]
        elif args.kind == "beam":
            probs = [p for p in probs if p.name.startswith("beam")]
    run_linear_suite(probs, variants, args, args.out or "results/plate_beam.csv")
