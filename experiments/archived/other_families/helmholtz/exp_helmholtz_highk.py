#!/usr/bin/env python3
"""High-wavenumber 2D Helmholtz (real solution) -- the spectral-bias battlefield.

    Delta u + kappa^2 u = f   on (-1,1)^2,   u = sin(a*pi*x) sin(a*pi*y),
    kappa = a*pi  (wavenumber),  f = -(a*pi)^2 u,  Dirichlet u=0 on boundary.

Sweep the wavenumber a in {2,4,6,8}.  Real "high-frequency SOTA" baselines
(Fourier-features, SIREN, MscaleDNN) are PARAMETER-MATCHED to the complex sinh
net and given frequency-matched inits (omega0, sigma ~ a*pi) so every method
gets its best shot at each wavenumber.  Acceptance: the complex advantage grows
with the wavenumber a.

Run:
  python experiments/exp_helmholtz_highk.py --out results/helmholtz_highk.csv
"""
from __future__ import annotations

import math

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
from osc_common import LinearProblem, laplacian_power_terms, run_linear_suite, default_argparser


def _u(a):
    def u(x):
        import torch
        return torch.sin(a * math.pi * x[..., 0]) * torch.sin(a * math.pi * x[..., 1])
    return u


def _u_aniso(a1, a2):
    def u(x):
        import torch
        return torch.sin(a1 * math.pi * x[..., 0]) * torch.sin(a2 * math.pi * x[..., 1])
    return u


def make_aniso(a1=1, a2=4, k=1.0):
    """Anisotropic Helmholtz of Wang-Teng-Perdikaris (2021): Delta u + k^2 u = q,
    u = sin(a1 pi x) sin(a2 pi y), q = (k^2 - (a1 pi)^2 - (a2 pi)^2) u."""
    return make_wang_aniso_pairs([(a1, a2)], k=k)


def make_wang_aniso_pairs(pairs, k=1.0):
    """Wang (2021) Eq. (8)--(10): one problem per (a1, a2) pair."""
    probs = []
    for a1, a2 in pairs:
        kappa2 = k ** 2
        S = (a1 ** 2 + a2 ** 2) * math.pi ** 2
        lam = kappa2 - S
        u = _u_aniso(a1, a2)
        fmax = max(a1, a2)
        probs.append(LinearProblem(
            name=f"helm_wang_{a1}_{a2}", d=2, order=2,
            terms=laplacian_power_terms(2, 1), zeroth=kappa2,
            u_exact=u, source_f=(lambda u=u, lam=lam: (lambda x: lam * u(x)))(),
            res_scale=max(abs(lam), 1.0), S=S, bc_lap_powers=(), sweep=float(fmax),
            extra={"omega0": max(10.0, 2.0 * math.pi * fmax),
                   "fourier_sigma": max(2.0, math.pi * fmax)},
        ))
    return probs


def parse_aniso_pairs(s: str):
    """Parse '1,1,1,2,1,4' or '1:1,1:2,1:4' into [(1,1), (1,2), (1,4)]."""
    s = s.strip()
    if not s:
        return []
    if ":" in s:
        return [tuple(int(x) for x in p.split(":")) for p in s.split(",") if p]
    nums = [int(x) for x in s.split(",") if x]
    if len(nums) % 2:
        raise ValueError(f"aniso pairs need even count, got {s!r}")
    return list(zip(nums[0::2], nums[1::2]))


def make_problems(sweeps=(2, 4, 6, 8)):
    probs = []
    for a in sweeps:
        kappa2 = (a * math.pi) ** 2
        S = 2.0 * (a * math.pi) ** 2          # -Delta eigenvalue
        lam = kappa2 - S                      # = -(a pi)^2
        u = _u(a)
        probs.append(LinearProblem(
            name=f"helmholtz_a{a}", d=2, order=2,
            terms=laplacian_power_terms(2, 1), zeroth=kappa2,
            u_exact=u, source_f=(lambda u=u, lam=lam: (lambda x: lam * u(x)))(),
            res_scale=abs(lam), S=S, bc_lap_powers=(), sweep=float(a),
            extra={"omega0": max(10.0, 2.0 * math.pi * a),
                   "fourier_sigma": max(2.0, math.pi * a)},
        ))
    return probs


if __name__ == "__main__":
    ap = default_argparser(seconds=80.0)
    ap.add_argument("--sweeps", default="2,4,6,8")
    ap.add_argument("--aniso", action="store_true",
                    help="run the anisotropic (a1,a2)=(1,4) Wang-2021 case instead")
    ap.add_argument("--wang-aniso", action="store_true",
                    help="Wang (2021) Eq.(8) sweep: (1,1),(1,2),(1,4) by default")
    ap.add_argument("--aniso-pairs", default="1,1,1,2,1,4",
                    help="comma pairs a1,a2,... or a1:a2,... for --wang-aniso")
    args = ap.parse_args()
    variants = [v for v in args.variants.split(",") if v]
    if args.wang_aniso:
        pairs = parse_aniso_pairs(args.aniso_pairs)
        probs = make_wang_aniso_pairs(pairs)
        default_out = "results/helmholtz_wang2021.csv"
    elif args.aniso:
        probs = make_aniso(1, 4)
        default_out = "results/helmholtz_aniso.csv"
    else:
        sweeps = [int(s) for s in args.sweeps.split(",") if s]
        probs = make_problems(sweeps)
        default_out = "results/helmholtz_highk.csv"
    run_linear_suite(probs, variants, args, args.out or default_out)
