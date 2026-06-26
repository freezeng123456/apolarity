#!/usr/bin/env python3
"""Should we use a COMPLEX-parameter network? Same-network-size PINN comparison.

For each manufactured PDE (single dominant monomial operator d^alpha u = f) we
train, under an equal wall-clock budget and identical optimizer/sampling/seed:

  variant "complex" : complex128 params, physical solution Re(u), + lam||Im W||^2
                      (this is the paper's B3 model)
  variant "real"    : float64 params (SAME width/depth), complex Waring
                      directions cast inside the jet, no Im regularizer
  variant "real_pm" : float64 params, width ~ sqrt(2)*H  (~ parameter-matched
                      to the complex net, for an honest accuracy comparison)

All three use the SAME derivative backend (waring_complex_jet), so per-step cost
is governed by the same R_C complex directions; the comparison isolates the
effect of the parameter TYPE (and, for real_pm, of the parameter COUNT).

Metrics per (problem, variant, seed): steps in budget, ms/step, peak MB,
final interior loss (avg last 20), relative L2 error on a FIXED held-out grid.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn
from torch import Tensor

from apolarity import single_monomial_partial
from apolarity.waring import monomial_waring_directions


class SinhActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sinh(x)


class Sin(nn.Module):  # name recognised by taylor_jet._is_sin_module
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


def activation_module(name: str) -> nn.Module:
    if name == "sinh":
        return SinhActivation()
    if name == "sin":
        return Sin()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(name)


# variant -> (complex params?, activation, width multiplier vs hidden)
VARIANT_SPEC = {
    "complex":      (True,  "sinh", 1.0),
    "real":         (False, "sinh", 1.0),
    "real_pm":      (False, "sinh", math.sqrt(2)),
    "real_sin":     (False, "sin",  1.0),
    "real_sin_pm":  (False, "sin",  math.sqrt(2)),
    "real_tanh_pm": (False, "tanh", math.sqrt(2)),
}


def build_mlp(d: int, hidden: int, depth: int, dtype: torch.dtype, act: str = "sinh") -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = d
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(activation_module(act))
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers).to(dtype=dtype)


def n_params(model: nn.Module) -> int:
    # count real degrees of freedom (complex weight = 2 reals)
    tot = 0
    for p in model.parameters():
        tot += p.numel() * (2 if p.dtype.is_complex else 1)
    return tot


# ---------------------------------------------------------------------------
# Manufactured problems:  d^alpha u_exact = f  on (-1,1)^d
# ---------------------------------------------------------------------------
@dataclass
class Problem:
    name: str
    d: int
    alpha: tuple[int, ...]
    u_exact: Callable[[Tensor], Tensor]
    source_f: Callable[[Tensor], Tensor]
    oscillatory: bool


def _gauss(x, idxs):
    s = sum(x[..., i] ** 2 for i in idxs)
    return torch.exp(-s / 4.0)


def make_problems() -> dict[str, Problem]:
    P: dict[str, Problem] = {}

    # (4,2), d=4, oscillatory (cos x2) -- the paper's benchmark
    def u_ch42_osc(x):
        return torch.sinh(x[..., 0]) * torch.cos(x[..., 1]) * _gauss(x, (2, 3))
    P["ch42_osc"] = Problem("ch42_osc", 4, (0, 0, 0, 0, 1, 1),
                            u_ch42_osc, lambda x: -u_ch42_osc(x), True)

    # (4,2), d=4, non-oscillatory (sinh x2)
    def u_ch42_mono(x):
        return torch.sinh(x[..., 0]) * torch.sinh(0.7 * x[..., 1]) * _gauss(x, (2, 3))
    P["ch42_mono"] = Problem("ch42_mono", 4, (0, 0, 0, 0, 1, 1),
                             u_ch42_mono, lambda x: 0.49 * u_ch42_mono(x), False)

    # (2,2,2), d=3, oscillatory (cos cos cos)
    def u_tri_osc(x):
        return torch.cos(x[..., 0]) * torch.cos(x[..., 1]) * torch.cos(x[..., 2])
    P["tri222_osc"] = Problem("tri222_osc", 3, (0, 0, 1, 1, 2, 2),
                              u_tri_osc, lambda x: -u_tri_osc(x), True)

    # (2,2,2), d=3, non-oscillatory (sinh sinh sinh)
    def u_tri_mono(x):
        return torch.sinh(x[..., 0]) * torch.sinh(x[..., 1]) * torch.sinh(x[..., 2])
    P["tri222_mono"] = Problem("tri222_mono", 3, (0, 0, 1, 1, 2, 2),
                               u_tri_mono, lambda x: u_tri_mono(x), False)

    # (2,2), d=2, oscillatory
    def u_bi_osc(x):
        return torch.sinh(x[..., 0]) * torch.cos(x[..., 1])
    P["bi22_osc"] = Problem("bi22_osc", 2, (0, 0, 1, 1),
                            u_bi_osc, lambda x: -u_bi_osc(x), True)

    # ---- FAIR targets: Gaussian x Hermite, NOT of the form Re(holomorphic) ----
    # u = exp(-|x|^2/2);   d^n_{x} exp(-x^2/2) = (-1)^n He_n(x) exp(-x^2/2)
    def he2(t): return t ** 2 - 1.0
    def he4(t): return t ** 4 - 6.0 * t ** 2 + 3.0

    def gauss(x):
        return torch.exp(-(x ** 2).sum(dim=-1) / 2.0)

    # (4,2), d=4, fair
    P["gauss42"] = Problem(
        "gauss42", 4, (0, 0, 0, 0, 1, 1),
        gauss,
        lambda x: he4(x[..., 0]) * he2(x[..., 1]) * gauss(x),  # (-1)^4 (-1)^2 = +1
        False,
    )
    # (2,2,2), d=3, fair
    P["gauss222"] = Problem(
        "gauss222", 3, (0, 0, 1, 1, 2, 2),
        gauss,
        lambda x: he2(x[..., 0]) * he2(x[..., 1]) * he2(x[..., 2]) * gauss(x),
        False,
    )
    return P


# ---------------------------------------------------------------------------
def sample_interior(B, d, *, device, dtype):
    x = torch.empty(B, d, device=device, dtype=torch.float64).uniform_(-1.0, 1.0)
    return x.to(dtype=dtype)


def sample_boundary(B, d, *, device, dtype):
    x = torch.empty(B, d, device=device, dtype=torch.float64).uniform_(-1.0, 1.0)
    face = torch.randint(0, d, (B,), device=device)
    sign = torch.where(torch.rand(B, device=device) < 0.5,
                       torch.tensor(-1.0, dtype=torch.float64, device=device),
                       torch.tensor(1.0, dtype=torch.float64, device=device))
    x[torch.arange(B, device=device), face] = sign
    return x.to(dtype=dtype)


BC_W, IM_W = 100.0, 1.0e-6


def pinn_loss(model, prob, x_int, x_bc, complex_params: bool):
    deriv = single_monomial_partial(model, x_int, prob.alpha, backend="waring_complex_jet")
    f = prob.source_f(x_int.real).unsqueeze(-1)
    L_int = ((deriv.real - f) ** 2).mean()
    u_bc = model(x_bc)
    bc_t = prob.u_exact(x_bc.real).unsqueeze(-1)
    L_bc = ((u_bc.real - bc_t) ** 2).mean()
    loss = L_int + BC_W * L_bc
    if complex_params:
        loss = loss + IM_W * sum((p.imag ** 2).mean() for p in model.parameters())
    return loss, L_int.item()


@torch.no_grad()
def l2_error(model, prob, eval_x):
    pred = model(eval_x).real.squeeze(-1)
    target = prob.u_exact(eval_x.real)
    return (((pred - target) ** 2).mean().sqrt() / (target ** 2).mean().sqrt()).item()


def run_variant(prob, variant, *, seconds, hidden, depth, n_int, n_bc, lr, seed, device):
    torch.manual_seed(seed)
    complex_params, act, wfac = VARIANT_SPEC[variant]
    dtype = torch.complex128 if complex_params else torch.float64
    H = int(round(hidden * wfac))
    model = build_mlp(prob.d, H, depth, dtype, act=act).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # fixed eval grid (same points for every variant/seed of this problem)
    g = torch.Generator(device=device).manual_seed(12345)
    eval_real = torch.empty(8192, prob.d, device=device, dtype=torch.float64).uniform_(-1, 1, generator=g)
    eval_x = eval_real.to(dtype=dtype)

    for _ in range(5):  # warmup (not timed)
        x_int = sample_interior(n_int, prob.d, device=device, dtype=dtype)
        x_bc = sample_boundary(n_bc, prob.d, device=device, dtype=dtype)
        loss, _ = pinn_loss(model, prob, x_int, x_bc, complex_params)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize()

    t0 = time.perf_counter(); steps = 0; losses = []
    while time.perf_counter() - t0 < seconds:
        x_int = sample_interior(n_int, prob.d, device=device, dtype=dtype)
        x_bc = sample_boundary(n_bc, prob.d, device=device, dtype=dtype)
        loss, L_int = pinn_loss(model, prob, x_int, x_bc, complex_params)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        steps += 1; losses.append(L_int)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    err = l2_error(model, prob, eval_x)
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else float("nan")
    _, _, info = monomial_waring_directions(prob.alpha, prob.d, dtype=torch.complex128)
    return {
        "problem": prob.name, "oscillatory": prob.oscillatory, "variant": variant,
        "activation": act, "seed": seed, "width": H, "params": n_params(model), "R_C": info.rank,
        "steps": steps, "ms_per_step": 1000.0 * elapsed / max(1, steps),
        "peak_mb": peak, "L_int_last": sum(losses[-20:]) / max(1, min(20, len(losses))),
        "L2_err": err,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-int", type=int, default=128)
    ap.add_argument("--n-bc", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--variants", default="complex,real,real_pm")
    ap.add_argument("--problems", default="ch42_osc,ch42_mono,tri222_osc,tri222_mono,bi22_osc")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allp = make_problems()
    problems = [allp[n] for n in args.problems.split(",") if n]
    variants = [v for v in args.variants.split(",") if v]

    print(f"device={device} hidden={args.hidden} depth={args.depth} "
          f"budget={args.seconds}s/run seeds={args.seeds}")
    rows = []
    for prob in problems:
        print(f"\n=== {prob.name}  (alpha={prob.alpha}, d={prob.d}, "
              f"{'oscillatory' if prob.oscillatory else 'non-oscillatory'}) ===")
        print(f"{'variant':<9} {'width':>5} {'params':>8} {'steps':>7} {'ms/step':>8} "
              f"{'peakMB':>7} {'L_int':>10} {'L2_err':>10}")
        for seed in range(args.seeds):
            for v in variants:
                r = run_variant(prob, v, seconds=args.seconds, hidden=args.hidden,
                                depth=args.depth, n_int=args.n_int, n_bc=args.n_bc,
                                lr=args.lr, seed=seed, device=device)
                rows.append(r)
                print(f"{v:<9} {r['width']:>5} {r['params']:>8} {r['steps']:>7} "
                      f"{r['ms_per_step']:>8.2f} {r['peak_mb']:>7.0f} "
                      f"{r['L_int_last']:>10.2e} {r['L2_err']:>10.3e}  (seed {seed})")

    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
        with out.with_suffix(".json").open("w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n[ok] wrote {out}")


if __name__ == "__main__":
    main()
