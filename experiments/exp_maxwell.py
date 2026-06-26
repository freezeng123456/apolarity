#!/usr/bin/env python3
"""Time-harmonic Maxwell (2D TM mode) in a LOSSY medium -- genuinely complex.

The transverse-magnetic time-harmonic Maxwell system reduces to a complex
Helmholtz equation for the out-of-plane field E_z:

    Delta E + kappa^2 E = f   on (-1,1)^2,   kappa^2 = (a pi)^2 (1 + i beta),
    manufactured  E = exp(i a pi (x + y))   (a traveling plane wave),
    Dirichlet E = E_exact on the boundary,   loss tangent beta = 0.2.

A complex permittivity (the i beta loss term) makes E genuinely complex-valued, so
this is a LINEAR complex-valued companion to the cubic NLS upper-bound test: a
single complex sinh net carries E in C natively, whereas the real baselines must
use a split-real (Re/Im) RVPINN -- parameter-matched as a pair.  Sweep the
wavenumber a; acceptance: the complex advantage grows with a.

Run:
  python experiments/exp_maxwell.py --out results/maxwell.csv
"""
from __future__ import annotations

import math

import torch

from osc_common import (OMEGA0, n_params, train_eval, write_rows, JET_VARIANTS,
                        default_argparser, sched_kwargs, make_complex_field)

BETA = 0.2  # loss tangent -> complex permittivity -> complex-valued field


def make_e_exact(a):
    ap = a * math.pi

    def E(x):
        phase = ap * (x[..., 0] + x[..., 1])
        return torch.exp(1j * phase.to(torch.complex128))
    return E


def _sample(B, device, gen=None):
    return torch.empty(B, 2, device=device, dtype=torch.float64).uniform_(-1, 1, generator=gen)


def _sample_bc(B, device):
    x = _sample(B, device)
    face = torch.randint(0, 2, (B,), device=device)
    lo = torch.tensor(-1.0, dtype=torch.float64, device=device)
    hi = torch.tensor(1.0, dtype=torch.float64, device=device)
    pick = torch.where(torch.rand(B, device=device) < 0.5, lo, hi)
    idx = torch.arange(B, device=device)
    x[idx, face] = pick
    return x


def run(sweeps, variants, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sk = sched_kwargs(args)
    print(f"device={device} hidden={args.hidden} depth={args.depth} "
          f"budget={args.seconds}s seeds={args.seeds} lr_schedule={sk['lr_schedule']}", flush=True)
    rows = []
    for a in sweeps:
        name = f"maxwell_a{a}"
        ap = a * math.pi
        kappa2 = (ap ** 2) * (1.0 + 1j * BETA)
        lam = (-2.0 * ap ** 2) + kappa2          # f = (Delta + kappa^2) E = lam * E
        res_scale = 2.0 * ap ** 2
        E_exact = make_e_exact(a)
        omega0 = max(OMEGA0, 2.0 * math.pi * a)
        sigma = max(2.0, math.pi * a)
        g = torch.Generator(device=device).manual_seed(2025)
        eval_r = _sample(8192, device, g)
        print(f"\n=== {name} (a={a}, |kappa^2|={abs(kappa2):.1f}) ===", flush=True)
        print(f"{'variant':<16}{'rep':>7}{'params':>8}{'steps':>7}{'ms/step':>9}"
              f"{'L_int':>11}{'L2_err':>12}", flush=True)
        for seed in range(args.seeds):
            for v in variants:
                torch.manual_seed(seed)
                field, is_complex = make_complex_field(
                    v, 2, args.hidden, args.depth, device, omega0=omega0, sigma=sigma)
                module = field.module
                x_int = _sample(args.n_int, device)
                x_bc = _sample_bc(args.n_bc, device)
                f_int = lam * E_exact(x_int)
                bc_t = E_exact(x_bc)

                def loss_fn():
                    u = field.pred(x_int)
                    lap = field.deriv(x_int, (0, 0)) + field.deriv(x_int, (1, 1))
                    r = lap + kappa2 * u - f_int
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
                        tgt = E_exact(eval_r)
                        return ((pred - tgt).abs() ** 2).mean().sqrt().item() / \
                               ((tgt.abs() ** 2).mean().sqrt().item() + 1e-30)

                m = train_eval(module, None, loss_fn, eval_fn,
                               seconds=args.seconds, lr=args.lr, device=device, **sk)
                rep = "complex" if is_complex else "split2"
                rows.append({"problem": name, "order": 2, "sweep": float(a),
                             "variant": v, "rep": rep, "seed": seed,
                             "params": n_params(module),
                             "backend": "jet" if v in JET_VARIANTS else "autograd", **m})
                print(f"{v:<16}{rep:>7}{n_params(module):>8}{m['steps']:>7}{m['ms_per_step']:>9.2f}"
                      f"{m['L_int_last']:>11.2e}{m['L2_err']:>12.3e}  (seed {seed})", flush=True)
    write_rows(rows, args.out or "results/maxwell.csv")
    return rows


if __name__ == "__main__":
    ap = default_argparser(seconds=80.0)
    ap.add_argument("--sweeps", default="2,4,6")
    args = ap.parse_args()
    sweeps = [int(s) for s in args.sweeps.split(",") if s]
    variants = [v for v in (args.variants.split(",") if args.variants else [])]
    if variants == ["complex_sinh", "fourier", "siren", "mscale", "tanh", "real_sinh"]:
        variants = ["complex_sinh", "tanh", "siren", "fourier", "mscale"]
    run(sweeps, variants, args)
