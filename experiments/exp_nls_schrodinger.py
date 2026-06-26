#!/usr/bin/env python3
"""Cubic NLS / Schrodinger -- the genuinely COMPLEX-valued problem (supplementary).

    i u_t + 0.5 u_xx + |u|^2 u = f,   u : R^2 -> C,
    manufactured  u = sech(x) exp(i k t)  =>  f = (0.5 - k) u   (k=0.5 is the
    exact bright soliton, f=0).  Domain x in [-5,5], t in [0, pi].

This is the "obvious win" upper-bound evidence: a single complex sinh network
represents complex-valued u natively, while the standard real baselines must use
a SPLIT-REAL (two-output Re/Im) representation (RVPINN).  Sweep the temporal
frequency k in {1,2,4}.  Demoted to supplementary because complex networks
winning on a complex-valued target is expected; the main battlefield is the
real-valued high-order suite.

Run:
  python experiments/exp_nls_schrodinger.py --out results/nls.csv
"""
from __future__ import annotations

import math

import torch

from osc_common import (OMEGA0, n_params, train_eval, write_rows, JET_VARIANTS,
                        default_argparser, sched_kwargs, make_complex_field)

# Physical domain x in [-LX, LX], t in [0, T].  Networks take NORMALISED inputs
# xhat in [-1,1]^2 (keeps sinh/cosh preactivations bounded); physical derivatives
# pick up chain-rule factors 1/LX, 1/LT below.
# Domain aligned to the canonical Raissi-Perdikaris-Karniadakis (2019) NLS window
# x in [-5, 5], t in [0, pi/2]; we use the manufactured standing bright soliton
# u = sech(x) exp(i k t), for which the source is f = (1/2 - k) u (f=0 at k=1/2).
LX, T = 5.0, math.pi / 2.0
LT = T / 2.0  # t = LT * (xhat1 + 1)


def _phys(xh):
    x = LX * xh[..., 0]
    t = LT * (xh[..., 1] + 1.0)
    return x, t


def make_u_exact(k):
    def u(xh):
        x, t = _phys(xh)
        sech = (1.0 / torch.cosh(x)).to(torch.complex128)
        return sech * torch.exp(1j * k * t.to(torch.complex128))
    return u


def _sample(B, device, gen=None):
    return torch.empty(B, 2, device=device, dtype=torch.float64).uniform_(-1, 1, generator=gen)


def _sample_bc(B, device):
    x = _sample(B, device)
    face = torch.randint(0, 2, (B,), device=device)
    xmask = face == 0  # x-faces -> xhat0 = +-1 ; t-face -> initial xhat1 = -1
    nx = int(xmask.sum())
    lo = torch.tensor(-1.0, dtype=torch.float64, device=device)
    hi = torch.tensor(1.0, dtype=torch.float64, device=device)
    x[xmask, 0] = torch.where(torch.rand(nx, device=device) < 0.5, lo, hi)
    x[~xmask, 1] = lo
    return x


def run(sweeps, variants, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sk = sched_kwargs(args)
    print(f"device={device} hidden={args.hidden} depth={args.depth} "
          f"budget={args.seconds}s seeds={args.seeds} lr_schedule={sk['lr_schedule']}", flush=True)
    rows = []
    for k in sweeps:
        name = f"nls_k{k}"
        u_exact = make_u_exact(k)
        res_scale = max(1.0, float(k))
        # temporal phase advances by k*T over xhat1 in [-1,1] -> freq ~ k*T/2
        omega0 = max(OMEGA0, 2.0 * k * LT)
        sigma = max(2.0, float(k) * LT)
        g = torch.Generator(device=device).manual_seed(2024)
        eval_r = _sample(8192, device, g)
        print(f"\n=== {name} (k={k}) ===", flush=True)
        print(f"{'variant':<16}{'rep':>6}{'params':>8}{'steps':>7}{'ms/step':>9}"
              f"{'L_int':>11}{'L2_err':>12}", flush=True)
        for seed in range(args.seeds):
            for v in variants:
                torch.manual_seed(seed)
                field, is_complex = make_complex_field(
                    v, 2, args.hidden, args.depth, device, omega0=omega0, sigma=sigma)
                module = field.module
                x_int = _sample(args.n_int, device)
                x_bc = _sample_bc(args.n_bc, device)
                f_int = (0.5 - k) * u_exact(x_int)
                bc_t = u_exact(x_bc)

                def loss_fn():
                    u = field.pred(x_int)
                    u_t = field.deriv(x_int, (1,)) / LT       # chain rule: d/dt
                    u_xx = field.deriv(x_int, (0, 0)) / LX ** 2  # chain rule: d^2/dx^2
                    r = 1j * u_t + 0.5 * u_xx + (u.abs() ** 2) * u - f_int
                    L_int = ((r.abs() / res_scale) ** 2).mean()
                    u_b = field.pred(x_bc)
                    L_bc = ((u_b - bc_t).abs() ** 2).mean()
                    loss = L_int + 100.0 * L_bc
                    if is_complex:
                        loss = loss + 1e-6 * sum((p.imag ** 2).mean()
                                                 for p in module.parameters() if p.requires_grad)
                    return loss, L_int.item()

                def eval_fn():
                    with torch.no_grad():
                        pred = field.pred(eval_r)
                        tgt = u_exact(eval_r)
                        return ((pred - tgt).abs() ** 2).mean().sqrt().item() / \
                               ((tgt.abs() ** 2).mean().sqrt().item() + 1e-30)

                m = train_eval(module, None, loss_fn, eval_fn,
                               seconds=args.seconds, lr=args.lr, device=device, **sk)
                rep = "complex" if is_complex else "split2"
                rows.append({"problem": name, "order": 2, "sweep": float(k),
                             "variant": v, "rep": rep, "seed": seed,
                             "params": n_params(module),
                             "backend": "jet" if v in JET_VARIANTS else "autograd", **m})
                print(f"{v:<16}{rep:>6}{n_params(module):>8}{m['steps']:>7}{m['ms_per_step']:>9.2f}"
                      f"{m['L_int_last']:>11.2e}{m['L2_err']:>12.3e}  (seed {seed})", flush=True)
    write_rows(rows, args.out or "results/nls.csv")
    return rows


if __name__ == "__main__":
    ap = default_argparser(seconds=80.0)
    ap.add_argument("--sweeps", default="1,2,4")
    args = ap.parse_args()
    sweeps = [int(s) for s in args.sweeps.split(",") if s]
    # complex sinh vs split-real RVPINN baselines
    variants = [v for v in (args.variants.split(",") if args.variants else [])]
    if variants == ["complex_sinh", "fourier", "siren", "mscale", "tanh", "real_sinh"]:
        variants = ["complex_sinh", "tanh", "siren", "fourier", "mscale"]
    run(sweeps, variants, args)
