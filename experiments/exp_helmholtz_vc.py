#!/usr/bin/env python3
"""Variable-coefficient ("scattering") 2D Helmholtz -- a HETEROGENEOUS medium.

    Delta u + kappa^2(x) u = f   on (-1,1)^2,   kappa^2(x) = (a pi)^2 (1 + C g(x)),
    g(x) = sin(pi x) sin(pi y)   (a smooth lens / scatterer),   C = 0.5,
    manufactured  u = sin(a pi x) sin(a pi y),  Dirichlet u = 0 on the boundary.

Unlike the constant-coefficient Helmholtz benchmark this exercises a spatially
varying zeroth-order coefficient (the harness now accepts a callable c(x)), so it
probes robustness to medium heterogeneity -- the realistic scattering regime --
rather than a single clean eigenmode.  Sweep the background wavenumber a; real
SOTA baselines (Fourier, SIREN, MscaleDNN) are parameter- and frequency-matched.
Acceptance: the complex sinh advantage grows with a.

Run:
  python experiments/exp_helmholtz_vc.py --out results/helmholtz_vc.csv
"""
from __future__ import annotations

import math

import torch

from osc_common import LinearProblem, laplacian_power_terms, run_linear_suite, default_argparser

CONTRAST = 0.5  # medium contrast: kappa^2 varies by +-50% across the domain


def _g(x):
    return torch.sin(math.pi * x[..., 0]) * torch.sin(math.pi * x[..., 1])


def _u(a):
    def u(x):
        return torch.sin(a * math.pi * x[..., 0]) * torch.sin(a * math.pi * x[..., 1])
    return u


def _kappa2(a):
    k0 = (a * math.pi) ** 2

    def k2(x):
        return k0 * (1.0 + CONTRAST * _g(x))
    return k2


def _source(a):
    u = _u(a)
    k2 = _kappa2(a)

    def f(x):
        return (-2.0 * (a * math.pi) ** 2) * u(x) + k2(x) * u(x)  # Delta u + k^2(x) u
    return f


def make_problems(sweeps=(2, 4, 6, 8)):
    probs = []
    for a in sweeps:
        probs.append(LinearProblem(
            name=f"helmvc_a{a}", d=2, order=2,
            terms=laplacian_power_terms(2, 1), zeroth=_kappa2(a),
            u_exact=_u(a), source_f=_source(a),
            res_scale=2.0 * (a * math.pi) ** 2, S=2.0 * (a * math.pi) ** 2,
            bc_lap_powers=(), sweep=float(a),
            extra={"omega0": max(10.0, 2.0 * math.pi * a),
                   "fourier_sigma": max(2.0, math.pi * a)},
        ))
    return probs


if __name__ == "__main__":
    ap = default_argparser(seconds=80.0)
    ap.add_argument("--sweeps", default="2,4,6,8")
    args = ap.parse_args()
    sweeps = [int(s) for s in args.sweeps.split(",") if s]
    variants = [v for v in args.variants.split(",") if v]
    run_linear_suite(make_problems(sweeps), variants, args,
                     args.out or "results/helmholtz_vc.csv")
