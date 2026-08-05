#!/usr/bin/env python3
"""Polyharmonic eigenmode -- a CONTROLLED ORDER SWEEP at FIXED frequency.

  dD:  Delta^m u = (-S)^m u  on (-1,1)^d,
       u = product_i sin(pi x_i), S = d pi^2.
       Delta^m expands into C(d+m-1,m) high-order jet terms.
  1D (--dim 1):  d^(2m)/dx^(2m) u = (-pi^2)^m u  on (-1,1),  u = sin(pi x),
                 S = pi^2.  The operator is a SINGLE monomial partial, so it is
                 cheap enough to push the order axis to 8, 10, 12.

Navier (simply-supported) BCs:  the lower even derivatives vanish on the boundary
(Delta^j u = 0, j = 0..m-1).  The exact solution, frequency, and domain are held
FIXED across the sweep; only the differential ORDER of the operator changes -- so
this isolates the core claim (complex sinh's advantage over frequency-matched real
baselines grows with the derivative order) with no frequency confound.

Frequency-matched init: the order-m operator amplifies an init frequency w by
~w^order, so omega0 must sit at (not above) the target |grad| -- otherwise an
over-large omega0 is amplified by omega0^order and buries the signal (this is why
omega0=10 stalled the high orders).  Defaults: 1D omega0=pi, 2D omega0=2pi.

Run:
  python experiments/exp_polyharmonic.py --out results/polyharmonic.csv          # 2D
  python experiments/exp_polyharmonic.py --dim 1 --orders 2,4,6,8,10 --out ...   # 1D
  python experiments/exp_polyharmonic.py --dim 3 --orders 2,4,6 --out ...         # 3D
"""
from __future__ import annotations

import math

import torch

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
from osc_common import LinearProblem, laplacian_power_terms, run_linear_suite, default_argparser
from boundary_weights import parse_weights, weights_for


def _u_product(x):
    return torch.sin(math.pi * x).prod(dim=-1)


def make_problems(orders=(2, 4, 6, 8), dim=2, omega0=None, sigma=None,
                  bc_weights=None):
    if dim < 1:
        raise ValueError("dim must be positive")
    S, u_exact = dim * math.pi ** 2, _u_product
    default_omega0 = math.pi if dim == 1 else 2.0 * math.pi
    om = omega0 if omega0 is not None else default_omega0
    fs = sigma if sigma is not None else math.pi
    probs = []
    for order in orders:
        if order < 2 or order % 2:
            raise ValueError("polyharmonic orders must be positive even integers")
        m = order // 2
        lam = (-S) ** m                      # Delta^m u = (-S)^m u
        if bc_weights is None:
            try:
                component_weights = weights_for(f"poly_d{dim}_o{order}")
            except ValueError:
                # Keep the historical diagnostic orders (e.g. order 8) usable;
                # the frozen active profile covers only jsc_v3's order 2/4/6 grid.
                component_weights = None
        else:
            component_weights = tuple(bc_weights)
        if component_weights is not None and len(component_weights) != m:
            raise ValueError(
                f"Poly order {order} expects {m} boundary weights "
                f"[u, Delta u, ...], got {len(component_weights)}"
            )
        probs.append(LinearProblem(
            name=f"polyharm{dim}d_o{order}", d=dim, order=order,
            terms=laplacian_power_terms(dim, m), zeroth=0.0,
            u_exact=u_exact, source_f=(lambda lam=lam, u=u_exact: (lambda x: lam * u(x)))(),
            res_scale=S ** m, S=S, bc_lap_powers=tuple(range(1, m)),
            sweep=float(order),
            bc_weights=component_weights,
            extra={"omega0": om, "fourier_sigma": fs},
        ))
    return probs


if __name__ == "__main__":
    ap = default_argparser(seconds=120.0)
    ap.add_argument("--orders", default="2,4,6,8")
    ap.add_argument("--dim", type=int, default=2)
    ap.add_argument("--omega0", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=None)
    ap.add_argument("--bc-weights", default="",
                    help="comma-separated powers-of-ten weights [u, Delta u, ...]")
    args = ap.parse_args()
    orders = [int(s) for s in args.orders.split(",") if s]
    variants = [v for v in args.variants.split(",") if v]
    bc_weights = parse_weights(args.bc_weights) if args.bc_weights else None
    default_out = f"results/polyharmonic{args.dim}d.csv"
    run_linear_suite(make_problems(orders, args.dim, args.omega0, args.sigma, bc_weights),
                     variants, args, args.out or default_out)
