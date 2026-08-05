#!/usr/bin/env python3
"""Cahn-Hilliard (4th & 6th order, real, NONLINEAR) in (x,t).

Coordinate 0 = x, coordinate 1 = t,  Delta = d^2/dx^2.

  4th order:  u_t = M [ Delta(u^3) - Delta u - gamma Delta^2 u ]
  6th order:  u_t = M [ Delta(u^3) - Delta u - gamma Delta^2 u + kappa Delta^3 u ]

Key point: the nonlinear flux Delta(u^3) = 3 u^2 u_xx + 6 u (u_x)^2 is written
purely from SINGLE-MONOMIAL partials of the network (u, u_x, u_xx), so the
entire residual -- including the 4th/6th-order linear terms -- is evaluated with
the fast complex-Waring Taylor-jet operator (no nested autograd over the net).

Manufactured solution u = sin(a*pi*x) cos(a*pi*t); the source f and the boundary
targets are obtained by autograd on the exact (analytic) solution.

Run:
  python experiments/exp_cahn_hilliard.py --out results/cahn_hilliard.csv
"""
from __future__ import annotations

import math

import torch

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "common"))
from osc_common import (OMEGA0, build_model, deriv_alpha, n_params, predict,
                        sample_boundary, sample_interior, train_eval, write_rows,
                        JET_VARIANTS, default_argparser, sched_kwargs)

M = 1.0
GAMMA = 1.0
KAPPA = 1.0


def make_u_exact(a):
    def u(x):
        return torch.sin(a * math.pi * x[..., 0]) * torch.cos(a * math.pi * x[..., 1])
    return u


def autograd_partial(func, x, alpha):
    """d^alpha func at x via nested autograd on the (analytic) exact solution.
    alpha is a tuple of coordinate indices, e.g. (0,0,0,0) = d^4/dx0^4."""
    x = x.clone().requires_grad_(True)
    y = func(x)
    for var in alpha:
        y = torch.autograd.grad(y.sum(), x, create_graph=True)[0][..., var]
    return y.detach()


def ch_terms_net(model, x, order):
    """Return CH residual operator R[u_net] using network single-partials."""
    u = predict(model, x).real.squeeze(-1)
    u_x = deriv_alpha(model, x, (0,)).real.squeeze(-1)
    u_xx = deriv_alpha(model, x, (0, 0)).real.squeeze(-1)
    u_t = deriv_alpha(model, x, (1,)).real.squeeze(-1)
    u_xxxx = deriv_alpha(model, x, (0, 0, 0, 0)).real.squeeze(-1)
    lap_u3 = 3.0 * u * u * u_xx + 6.0 * u * u_x * u_x          # Delta(u^3) in 1D
    flux = lap_u3 - u_xx - GAMMA * u_xxxx
    if order >= 6:
        u_xxxxxx = deriv_alpha(model, x, (0, 0, 0, 0, 0, 0)).real.squeeze(-1)
        flux = flux + KAPPA * u_xxxxxx
    return u_t - M * flux


def ch_source(u_exact, x, order):
    u = u_exact(x)
    u_x = autograd_partial(u_exact, x, (0,))
    u_xx = autograd_partial(u_exact, x, (0, 0))
    u_t = autograd_partial(u_exact, x, (1,))
    u_xxxx = autograd_partial(u_exact, x, (0, 0, 0, 0))
    lap_u3 = 3.0 * u * u * u_xx + 6.0 * u * u_x * u_x
    flux = lap_u3 - u_xx - GAMMA * u_xxxx
    if order >= 6:
        flux = flux + KAPPA * autograd_partial(u_exact, x, (0, 0, 0, 0, 0, 0))
    return u_t - M * flux


def run(cases, variants, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sk = sched_kwargs(args)
    seed_start = getattr(args, "seed_start", 0)
    seed_ids = list(range(seed_start, seed_start + args.seeds))
    print(f"device={device} hidden={args.hidden} depth={args.depth} "
          f"budget={args.seconds}s seeds={seed_ids} lr_schedule={sk['lr_schedule']}", flush=True)
    rows = []
    for a, order in cases:
        name = f"ch{order}_a{a}"
        ap = a * math.pi
        res_scale = (GAMMA if order == 4 else KAPPA) * ap ** order
        u_exact = make_u_exact(a)
        bc_alphas = [(), (0, 0)] + ([(0, 0, 0, 0)] if order >= 6 else [])
        omega0 = max(OMEGA0, 2.0 * math.pi * a)
        sigma = max(2.0, math.pi * a)
        g = torch.Generator(device=device).manual_seed(777)
        eval_r = torch.empty(8192, 2, device=device, dtype=torch.float64).uniform_(-1, 1, generator=g)
        print(f"\n=== {name} (order={order}, a={a}) ===", flush=True)
        print(f"{'variant':<16}{'params':>8}{'steps':>7}{'ms/step':>9}{'L_int':>11}{'L2_err':>12}", flush=True)
        for seed in seed_ids:
            train_gen = torch.Generator(device=device).manual_seed(seed)
            x_int = sample_interior(args.n_int, 2, device=device, generator=train_gen)
            x_bc = sample_boundary(args.n_bc, 2, device=device, generator=train_gen)
            f_int = ch_source(u_exact, x_int, order).unsqueeze(-1).to(torch.float64)
            bc_targets = [u_exact(x_bc) if not al
                          else autograd_partial(u_exact, x_bc, al) for al in bc_alphas]
            for v in variants:
                torch.manual_seed(seed)
                model, mdt = build_model(v, 2, args.hidden, args.depth,
                                         omega0=omega0, fourier_sigma=sigma)
                model = model.to(device)

                def loss_fn():
                    R = ch_terms_net(model, x_int.to(mdt), order).unsqueeze(-1)
                    L_int = (((R - f_int) / res_scale) ** 2).mean()
                    L_bc = 0.0
                    for al, tgt in zip(bc_alphas, bc_targets):
                        if not al:
                            pred = predict(model, x_bc.to(mdt)).real.squeeze(-1)
                        else:
                            pred = deriv_alpha(model, x_bc.to(mdt), al).real.squeeze(-1)
                            tgt = tgt / (ap ** len(al))
                            pred = pred / (ap ** len(al))
                        L_bc = L_bc + ((pred - tgt) ** 2).mean()
                    loss = L_int + 100.0 * L_bc
                    if mdt.is_complex:
                        loss = loss + 1e-6 * sum((p.imag ** 2).mean()
                                                 for p in model.parameters() if p.requires_grad)
                    return loss, L_int.item()

                def eval_fn():
                    with torch.no_grad():
                        pred = predict(model, eval_r.to(mdt)).real.squeeze(-1)
                        tgt = u_exact(eval_r)
                        return (((pred - tgt) ** 2).mean().sqrt() / (tgt ** 2).mean().sqrt()).item()

                m = train_eval(model, mdt, loss_fn, eval_fn,
                               seconds=args.seconds, lr=args.lr, device=device, **sk)
                rows.append({"problem": name, "order": order, "sweep": float(a),
                             "variant": v, "seed": seed, "params": n_params(model),
                             "backend": "jet" if v in JET_VARIANTS else "autograd",
                             "hidden": args.hidden, "depth": args.depth,
                             "budget_seconds": args.seconds, "n_int": args.n_int,
                             "n_bc": args.n_bc, "lr": args.lr,
                             "lr_schedule": sk["lr_schedule"], "omega0": omega0,
                             "fourier_sigma": sigma,
                             "collocation": "paired_seed_v1", **m})
                print(f"{v:<16}{n_params(model):>8}{m['steps']:>7}{m['ms_per_step']:>9.2f}"
                      f"{m['L_int_last']:>11.2e}{m['L2_err']:>12.3e}  (seed {seed})", flush=True)
                del model, loss_fn, eval_fn
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del x_int, x_bc, f_int, bc_targets
    write_rows(rows, args.out or "results/cahn_hilliard.csv")
    return rows


if __name__ == "__main__":
    ap = default_argparser(seconds=90.0)
    ap.add_argument("--a", default="2,3")
    ap.add_argument("--orders", default="4,6")
    args = ap.parse_args()
    a_vals = [int(s) for s in args.a.split(",") if s]
    orders = [int(s) for s in args.orders.split(",") if s]
    cases = [(a, o) for o in orders for a in a_vals]
    variants = [v for v in args.variants.split(",") if v]
    run(cases, variants, args)
