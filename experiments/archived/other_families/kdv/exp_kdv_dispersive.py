#!/usr/bin/env python3
"""Linearized KdV / dispersive wave (3rd order, real, oscillatory + dispersive).

    L[u] = u_t + delta * u_xxx = f   on (x,t) in (-1,1)^2,
    u = sin(k*pi*x) cos(k*pi*t),  Dirichlet u = u_exact on the boundary.

Coordinate 0 = x, coordinate 1 = t.  Sweep the wavenumber k in {2,3,4,5}; the
odd 3rd-order dispersion term is where the real Taylor-jet operator and complex
sinh shine.  Real baselines are parameter/frequency-matched.  Acceptance: the
complex advantage grows with the wavenumber k.

Run:
  python experiments/exp_kdv_dispersive.py --out results/kdv_dispersive.csv
"""
from __future__ import annotations

import math

import torch

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
from osc_common import LinearProblem, run_linear_suite, default_argparser

DELTA = 1.0  # dispersion coefficient


def _u(k):
    def u(x):
        return torch.sin(k * math.pi * x[..., 0]) * torch.cos(k * math.pi * x[..., 1])
    return u


def _f(k):
    kp = k * math.pi

    def f(x):
        s_x = torch.sin(kp * x[..., 0])
        c_x = torch.cos(kp * x[..., 0])
        s_t = torch.sin(kp * x[..., 1])
        c_t = torch.cos(kp * x[..., 1])
        u_t = -kp * s_x * s_t
        u_xxx = -(kp ** 3) * c_x * c_t
        return u_t + DELTA * u_xxx
    return f


def make_problems(sweeps=(2, 3, 4, 5)):
    probs = []
    for k in sweeps:
        kp = k * math.pi
        probs.append(LinearProblem(
            name=f"kdv_k{k}", d=2, order=3,
            terms=[(1.0, (1,)), (DELTA, (0, 0, 0))], zeroth=0.0,
            u_exact=_u(k), source_f=_f(k),
            res_scale=DELTA * kp ** 3 + kp, S=0.0, bc_lap_powers=(), sweep=float(k),
            extra={"omega0": max(10.0, 2.0 * math.pi * k),
                   "fourier_sigma": max(2.0, math.pi * k)},
        ))
    return probs


if __name__ == "__main__":
    ap = default_argparser(seconds=80.0)
    ap.add_argument("--sweeps", default="2,3,4,5")
    args = ap.parse_args()
    sweeps = [int(s) for s in args.sweeps.split(",") if s]
    variants = [v for v in args.variants.split(",") if v]
    run_linear_suite(make_problems(sweeps), variants, args,
                     args.out or "results/kdv_dispersive.csv")
