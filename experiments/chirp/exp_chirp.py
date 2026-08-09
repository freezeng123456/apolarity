#!/usr/bin/env python3
"""Non-separable oscillatory target -- a fair EXPRESSIVITY test.

    -Delta u + u = f   on (-1,1)^2,   u = sin(a pi (x^2 + y^2) / 2),
    a radial CHIRP whose local frequency  |grad phi| = a pi r  grows with radius
    -- so u is NOT a single Fourier mode.  Dirichlet u = u_exact on the boundary.

Why this benchmark: the rest of the suite uses pure separable sines
u = sin(k x) sin(k y), which sine-based real nets (SIREN / Fourier / MscaleDNN)
represent almost natively, masking any expressivity gap.  A space-varying-frequency
chirp removes that confound: every architecture must actually BUILD the varying
oscillation.  Sweep a; real baselines are parameter- and frequency-matched.

Run:
  python experiments/exp_chirp.py --out results/chirp.csv
"""
from __future__ import annotations

import math

import torch

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
from osc_common import LinearProblem, run_linear_suite, default_argparser
from boundary_weights import parse_weights, weights_for


def _phi(a, x):
    return 0.5 * a * math.pi * (x[..., 0] ** 2 + x[..., 1] ** 2)


def _u(a):
    def u(x):
        return torch.sin(_phi(a, x))
    return u


def _source(a):
    ap = a * math.pi

    def f(x):
        r2 = x[..., 0] ** 2 + x[..., 1] ** 2
        phi = _phi(a, x)
        s, c = torch.sin(phi), torch.cos(phi)
        lap = -(ap ** 2) * r2 * s + 2.0 * ap * c          # Delta u
        return -lap + s                                    # -Delta u + u
    return f


def make_problems(sweeps=(2, 4, 6, 8), bc_weight=None):
    probs = []
    for a in sweeps:
        ap = a * math.pi
        if bc_weight is None:
            try:
                weight = weights_for(f"chirp_a{a}")[0]
            except ValueError:
                # Keep the historical diagnostic sweeps outside jsc_v3 usable.
                weight = 100.0
        else:
            weight = float(bc_weight)
        probs.append(LinearProblem(
            name=f"chirp_a{a}", d=2, order=2,
            terms=[(-1.0, (0, 0)), (-1.0, (1, 1))], zeroth=1.0,   # -Delta u + u
            u_exact=_u(a), source_f=_source(a),
            res_scale=2.0 * ap ** 2, S=2.0 * ap ** 2, bc_lap_powers=(), sweep=float(a),
            bc_weight=weight,
            extra={"omega0": max(10.0, 2.0 * math.pi * a),
                   "fourier_sigma": max(2.0, math.pi * a)},
        ))
    return probs


if __name__ == "__main__":
    ap = default_argparser(seconds=1000.0)
    ap.add_argument("--sweeps", default="1,2,3")
    ap.add_argument("--bc-weight", default=None,
                    help="scalar powers-of-ten Dirichlet weight")
    args = ap.parse_args()
    sweeps = [int(s) for s in args.sweeps.split(",") if s]
    variants = [v for v in args.variants.split(",") if v]
    bc_weight = None if args.bc_weight is None else parse_weights(args.bc_weight)[0]
    run_linear_suite(make_problems(sweeps, bc_weight), variants, args,
                     args.out or "results/chirp.csv")
