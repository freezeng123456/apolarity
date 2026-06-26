#!/usr/bin/env python3
"""Comprehensive oscillatory high-order PINN comparison.

Goal: on OSCILLATORY, HIGH-ORDER PDEs, compare a complex-parameter sinh network
(the focus of this project) against the standard real-valued "spectral-bias"
toolkit and a complex-analysis-inspired baseline, under an equal wall-clock
budget, identical optimizer / sampling / seed.

Each PDE is a general linear differential operator
    L[u](x) = sum_t c_t * d^{alpha_t} u (x)  + c0 * u(x)  = f(x)
with a manufactured oscillatory solution u_exact (so f is known analytically).
The residual is normalised by a per-problem scale so the interior loss is O(1)
regardless of the operator order (otherwise S^3 ~ 1e5 factors dominate).

Architectures (variants)
------------------------
  complex_sinh : complex128 params, entire sinh activation  (THIS PROJECT)
                 physical solution Re(u); derivatives via complex Waring jet.
  real_sinh    : float64 params, sinh activation (ablation of param type).
  tanh         : vanilla float64 tanh MLP.
  siren        : float64 sine MLP with SIREN init (periodic activations).
  fourier      : random Fourier-feature embedding + tanh MLP.
  mscale       : MscaleDNN (sum of K input-scaled sine subnetworks).
  cauchy       : compleX-PINN-style learnable Cauchy activation
                 Phi(x)=(m1 x + m2)/(x^2 + d^2); derivatives via nested autograd
                 (the rational activation is not a Taylor-jet primitive).

All jet-supported nets share the SAME exact derivative backend
(waring_complex_jet); cauchy uses nested autograd.  Accuracy (relative L2) is
the primary metric; ms/step and peak memory are reported too (cauchy ms/step is
not directly comparable because of its different derivative backend).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
class SinhActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sinh(x)


class Sin(nn.Module):  # name recognised by taylor_jet._is_sin_module
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


class CauchyActivation(nn.Module):
    """compleX-PINN activation  Phi(x; m1, m2, d) = (m1 x + m2)/(x^2 + d^2).

    Per-feature trainable (m1, m2, d).  Smooth on the real axis (poles at +-i*d).
    Not a Taylor-jet primitive, so models using it must differentiate via autograd.
    """

    def __init__(self, width: int):
        super().__init__()
        self.m1 = nn.Parameter(torch.ones(width))
        self.m2 = nn.Parameter(torch.zeros(width))
        # d initialised away from 0 to keep the activation well-conditioned.
        self.d = nn.Parameter(torch.full((width,), 1.0))

    def forward(self, x: Tensor) -> Tensor:
        denom = x * x + self.d * self.d
        return (self.m1 * x + self.m2) / denom


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
def _seq(layers: list[nn.Module], dtype: torch.dtype) -> nn.Sequential:
    return nn.Sequential(*layers).to(dtype=dtype)


def build_plain(d: int, H: int, depth: int, dtype: torch.dtype, act: str) -> nn.Sequential:
    def a():
        return {"sinh": SinhActivation, "tanh": nn.Tanh, "sin": Sin}[act]()
    layers: list[nn.Module] = []
    in_dim = d
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, H))
        layers.append(a())
        in_dim = H
    layers.append(nn.Linear(in_dim, 1))
    return _seq(layers, dtype)


def siren_init_(net: nn.Sequential, d: int, omega0: float) -> None:
    """SIREN initialisation, with omega0 folded into the first-layer weights so
    the plain Sin activation reproduces sin(omega0 * (W x + b))."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for i, lin in enumerate(linears):
            fan_in = lin.weight.shape[1]
            if i == 0:
                bound = omega0 / fan_in
            elif i == len(linears) - 1:
                bound = math.sqrt(6.0 / fan_in) * 1e-1  # small last layer
            else:
                bound = math.sqrt(6.0 / fan_in)
            lin.weight.uniform_(-bound, bound)
            if lin.bias is not None:
                lin.bias.uniform_(-bound, bound)


def build_siren(d: int, H: int, depth: int, omega0: float) -> nn.Sequential:
    net = build_plain(d, H, depth, torch.float64, "sin")
    siren_init_(net, d, omega0)
    return net


def complex_freq_init_(net: nn.Sequential, omega0: float) -> None:
    """Frequency-rich ('complex-SIREN' / holomorphic) init for a complex sinh
    MLP.  The oscillation of Re(sinh(z)) lives in Im(z), so we inject frequency
    via the imaginary part of the first-layer weights (small real part keeps the
    cosh growth bounded).  Without this, spectral bias pins a complex net to the
    trivial solution on oscillatory targets."""
    linears = [m for m in net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        first = linears[0]
        fan_in = first.weight.shape[1]
        w = torch.zeros_like(first.weight)
        w.real.uniform_(-1.0 / fan_in, 1.0 / fan_in)
        w.imag.uniform_(-omega0 / fan_in, omega0 / fan_in)
        first.weight.copy_(w)
        if first.bias is not None:
            b = torch.zeros_like(first.bias)
            b.imag.uniform_(-math.pi, math.pi)
            first.bias.copy_(b)


def build_fourier(d: int, H: int, depth: int, m_feat: int, sigma: float) -> nn.Sequential:
    """Random Fourier features gamma(x)=[sin(Bx), cos(Bx)] as a frozen
    Linear(d->2m) + Sin, where row block [B; B] and bias [0; pi/2] turn the
    second half into cos.  Followed by a tanh MLP.  Jet-compatible."""
    B = torch.randn(m_feat, d, dtype=torch.float64) * sigma
    W0 = torch.cat([B, B], dim=0)                          # (2m, d)
    b0 = torch.cat([torch.zeros(m_feat), torch.full((m_feat,), math.pi / 2)])
    first = nn.Linear(d, 2 * m_feat)
    with torch.no_grad():
        first.weight.copy_(W0)
        first.bias.copy_(b0)
    first.weight.requires_grad_(False)
    first.bias.requires_grad_(False)

    layers: list[nn.Module] = [first, Sin()]
    in_dim = 2 * m_feat
    for _ in range(depth - 1):
        layers.append(nn.Linear(in_dim, H))
        layers.append(nn.Tanh())
        in_dim = H
    layers.append(nn.Linear(in_dim, 1))
    return _seq(layers, torch.float64)


class MultiScaleNet(nn.Module):
    """MscaleDNN: output = sum_k subnet_k(a_k * x), each subnet a sine MLP.

    The scale a_k is folded into the first linear layer of each subnet, so each
    subnet is a plain jet-compatible Sequential and its derivative can be taken
    independently (derivative is linear, so d^alpha output = sum_k d^alpha sub_k)."""

    def __init__(self, d: int, H: int, depth: int, scales: tuple[float, ...]):
        super().__init__()
        self.subnets = nn.ModuleList()
        for a in scales:
            sub = build_plain(d, H, depth, torch.float64, "sin")
            with torch.no_grad():  # fold scale into first layer
                first = next(m for m in sub if isinstance(m, nn.Linear))
                first.weight.mul_(a)
            self.subnets.append(sub)

    def forward(self, x: Tensor) -> Tensor:
        return sum(sub(x) for sub in self.subnets)


class CauchyNet(nn.Module):
    """compleX-PINN: MLP with the learnable Cauchy activation (real-valued)."""

    def __init__(self, d: int, H: int, depth: int):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = d
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, H))
            layers.append(CauchyActivation(H))
            in_dim = H
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers).to(dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


# shared frequency scale for the periodic / complex-SIREN initialisations
OMEGA0 = 10.0


# variant -> builder spec
def build_model(variant: str, d: int, H: int, depth: int):
    if variant == "complex_sinh":
        net = build_plain(d, H, depth, torch.complex128, "sinh")
        complex_freq_init_(net, OMEGA0)
        return net, torch.complex128
    if variant == "complex_sinh_noinit":  # ablation: default init (no freq init)
        return build_plain(d, H, depth, torch.complex128, "sinh"), torch.complex128
    if variant == "real_sinh":
        return build_plain(d, H, depth, torch.float64, "sinh"), torch.float64
    if variant == "tanh":
        return build_plain(d, H, depth, torch.float64, "tanh"), torch.float64
    if variant == "siren":
        return build_siren(d, H, depth, omega0=OMEGA0), torch.float64
    if variant == "fourier":
        return build_fourier(d, H, depth, m_feat=H, sigma=2.0), torch.float64
    if variant == "mscale":
        w = max(8, int(round(H / math.sqrt(3))))
        return MultiScaleNet(d, w, depth, scales=(1.0, 2.0, 4.0)), torch.float64
    if variant == "cauchy":
        return CauchyNet(d, H, depth), torch.float64
    raise ValueError(variant)


JET_VARIANTS = {"complex_sinh", "complex_sinh_noinit", "real_sinh", "tanh",
                "siren", "fourier", "mscale"}


def n_params(model: nn.Module) -> int:
    tot = 0
    for p in model.parameters():
        if p.requires_grad:
            tot += p.numel() * (2 if p.dtype.is_complex else 1)
    return tot


# ---------------------------------------------------------------------------
# Derivatives / predictions dispatched per architecture
# ---------------------------------------------------------------------------
def predict(model: nn.Module, x: Tensor) -> Tensor:
    out = model(x)
    return out.real if out.is_complex else out


def deriv_alpha(model: nn.Module, x: Tensor, alpha: tuple[int, ...]) -> Tensor:
    if isinstance(model, MultiScaleNet):
        s = None
        for sub in model.subnets:
            t = single_monomial_partial(sub, x, alpha, backend="waring_complex_jet")
            s = t if s is None else s + t
        return s.real
    if isinstance(model, CauchyNet):
        return single_monomial_partial(model.net, x, alpha, backend="direct_autodiff").real
    out = single_monomial_partial(model, x, alpha, backend="waring_complex_jet")
    return out.real if out.is_complex else out


# ---------------------------------------------------------------------------
# Oscillatory high-order PDEs:  sum_t c_t d^alpha_t u + c0 u = f   on (-1,1)^d
# ---------------------------------------------------------------------------
@dataclass
class Problem:
    name: str
    d: int
    order: int
    terms: list[tuple[float, tuple[int, ...]]]      # (coeff, expanded alpha)
    zeroth: float                                    # coeff of u itself
    u_exact: Callable[[Tensor], Tensor]
    source_f: Callable[[Tensor], Tensor]
    res_scale: float
    S: float = 0.0                                   # -Laplacian eigenvalue
    # Navier (simply-supported) BCs: enforce Delta^j u = (-S)^j u_exact on the
    # boundary for j in bc_lap_powers, in addition to the Dirichlet u match.
    # These make the polyharmonic problems well-posed; the sin-product
    # manufactured solutions satisfy them exactly.
    bc_lap_powers: tuple[int, ...] = ()


def _prod_sin(a: tuple[int, ...]):
    def u(x: Tensor) -> Tensor:
        out = torch.ones(x.shape[:-1], dtype=x.dtype, device=x.device)
        for i, ai in enumerate(a):
            out = out * torch.sin(ai * math.pi * x[..., i])
        return out
    return u


def laplacian_power_terms(d: int, j: int) -> list[tuple[float, tuple[int, ...]]]:
    """Expanded terms of Delta^j in d=2 dimensions (binomial expansion)."""
    if d != 2:
        raise NotImplementedError("laplacian_power_terms only implements d=2")
    terms = []
    for i in range(j + 1):
        alpha = (0,) * (2 * i) + (1,) * (2 * (j - i))
        terms.append((float(math.comb(j, i)), alpha))
    return terms


def make_problems() -> dict[str, Problem]:
    P: dict[str, Problem] = {}
    pi = math.pi

    # 1) High-frequency Helmholtz (order 2):  Delta u + k^2 u = f,
    #    u = sin(a pi x) sin(b pi y),  a=b=3 (oscillatory).  Dirichlet u=0 BC.
    a = (3, 3)
    S = pi * pi * (a[0] ** 2 + a[1] ** 2)            # -Delta eigenvalue
    k2 = 1.0
    lamH = -S + k2
    u_h = _prod_sin(a)
    P["helmholtz_hf"] = Problem(
        "helmholtz_hf", 2, 2,
        terms=laplacian_power_terms(2, 1), zeroth=k2,
        u_exact=u_h, source_f=lambda x: lamH * u_h(x), res_scale=abs(lamH),
        S=S, bc_lap_powers=(),
    )

    # 2) Linear KdV / dispersive wave (order 3):  u_t + u_xxx + c u_x = f,
    #    u = sin(k x - w t),  x=coord0, t=coord1.  Full-box Dirichlet match.
    kx, w, c = 3.0, 2.0, 1.0
    coef = (-w - kx ** 3 + c * kx)

    def u_kdv(x):
        return torch.sin(kx * x[..., 0] - w * x[..., 1])

    def f_kdv(x):
        return coef * torch.cos(kx * x[..., 0] - w * x[..., 1])

    P["kdv"] = Problem(
        "kdv", 2, 3,
        terms=[(1.0, (1,)), (1.0, (0, 0, 0)), (c, (0,))], zeroth=0.0,
        u_exact=u_kdv, source_f=f_kdv, res_scale=kx ** 3, S=0.0, bc_lap_powers=(),
    )

    # 3) Biharmonic (order 4):  Delta^2 u = f,  u = sin(pi x) sin(pi y).
    #    Navier BC: u = Delta u = 0 on the boundary.
    a = (1, 1)
    S = pi * pi * (a[0] ** 2 + a[1] ** 2)
    u_b = _prod_sin(a)
    P["biharmonic"] = Problem(
        "biharmonic", 2, 4,
        terms=laplacian_power_terms(2, 2),
        zeroth=0.0, u_exact=u_b, source_f=lambda x: (S ** 2) * u_b(x),
        res_scale=S ** 2, S=S, bc_lap_powers=(1,),
    )

    # 4) Triharmonic / Delta^3 (order 6):  Delta^3 u = f, u = sin(pi x) sin(pi y).
    #    Navier BC: u = Delta u = Delta^2 u = 0 on the boundary.
    a = (1, 1)
    S = pi * pi * (a[0] ** 2 + a[1] ** 2)
    u_t = _prod_sin(a)
    P["triharmonic"] = Problem(
        "triharmonic", 2, 6,
        terms=laplacian_power_terms(2, 3),
        zeroth=0.0, u_exact=u_t, source_f=lambda x: -(S ** 3) * u_t(x),
        res_scale=S ** 3, S=S, bc_lap_powers=(1, 2),
    )
    return P


# ---------------------------------------------------------------------------
# Sampling / loss / training
# ---------------------------------------------------------------------------
def sample_interior(B, d, *, device):
    return torch.empty(B, d, device=device, dtype=torch.float64).uniform_(-1.0, 1.0)


def sample_boundary(B, d, *, device):
    x = torch.empty(B, d, device=device, dtype=torch.float64).uniform_(-1.0, 1.0)
    face = torch.randint(0, d, (B,), device=device)
    sign = torch.where(torch.rand(B, device=device) < 0.5,
                       torch.tensor(-1.0, dtype=torch.float64, device=device),
                       torch.tensor(1.0, dtype=torch.float64, device=device))
    x[torch.arange(B, device=device), face] = sign
    return x


BC_W, IM_W = 100.0, 1.0e-6


def pinn_loss(model, prob, x_int_r, x_bc_r, model_dtype, complex_params):
    x_int = x_int_r.to(model_dtype)
    x_bc = x_bc_r.to(model_dtype)

    res = None
    for coeff, alpha in prob.terms:
        t = deriv_alpha(model, x_int, alpha)
        res = coeff * t if res is None else res + coeff * t
    if prob.zeroth != 0.0:
        res = res + prob.zeroth * predict(model, x_int)
    f = prob.source_f(x_int_r).unsqueeze(-1)
    L_int = (((res - f) / prob.res_scale) ** 2).mean()

    u_bc = predict(model, x_bc)
    bc_t = prob.u_exact(x_bc_r).unsqueeze(-1)
    L_bc = ((u_bc - bc_t) ** 2).mean()
    # Navier (simply-supported) higher-order BCs: Delta^j u = (-S)^j u_exact.
    for j in prob.bc_lap_powers:
        lap = None
        for coeff, alpha in laplacian_power_terms(prob.d, j):
            t = deriv_alpha(model, x_bc, alpha)
            lap = coeff * t if lap is None else lap + coeff * t
        tgt = ((-prob.S) ** j) * prob.u_exact(x_bc_r).unsqueeze(-1)
        L_bc = L_bc + (((lap - tgt) / (prob.S ** j)) ** 2).mean()

    loss = L_int + BC_W * L_bc
    if complex_params:
        loss = loss + IM_W * sum((p.imag ** 2).mean() for p in model.parameters() if p.requires_grad)
    return loss, L_int.item()


@torch.no_grad()
def l2_error(model, prob, eval_r, model_dtype):
    pred = predict(model, eval_r.to(model_dtype)).squeeze(-1)
    target = prob.u_exact(eval_r)
    return (((pred - target) ** 2).mean().sqrt() / (target ** 2).mean().sqrt()).item()


def run_variant(prob, variant, *, seconds, hidden, depth, n_int, n_bc, lr, seed, device):
    torch.manual_seed(seed)
    model, model_dtype = build_model(variant, prob.d, hidden, depth)
    model = model.to(device)
    complex_params = model_dtype.is_complex
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr)

    g = torch.Generator(device=device).manual_seed(12345)
    eval_r = torch.empty(8192, prob.d, device=device, dtype=torch.float64).uniform_(-1, 1, generator=g)

    # Fixed dense collocation set (sampled once): high-order residuals need many
    # points, and a fixed set conditions the loss far better than per-step
    # resampling (which lets the net cheat with high-frequency wiggles).
    x_int = sample_interior(n_int, prob.d, device=device)
    x_bc = sample_boundary(n_bc, prob.d, device=device)

    def step():
        loss, L_int = pinn_loss(model, prob, x_int, x_bc, model_dtype, complex_params)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return L_int

    try:
        for _ in range(5):  # warmup (not timed)
            step()
    except Exception as e:  # noqa: BLE001
        return _fail_row(prob, variant, model, seed, repr(e))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    steps = 0
    losses: list[float] = []
    nan_hit = False
    while time.perf_counter() - t0 < seconds:
        L_int = step()
        steps += 1
        losses.append(L_int)
        if not math.isfinite(L_int):
            nan_hit = True
            break
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    err = l2_error(model, prob, eval_r, model_dtype)
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else float("nan")
    return {
        "problem": prob.name, "order": prob.order, "variant": variant, "seed": seed,
        "n_terms": len(prob.terms), "params": n_params(model),
        "backend": "jet" if variant in JET_VARIANTS else "autograd",
        "steps": steps, "ms_per_step": 1000.0 * elapsed / max(1, steps),
        "peak_mb": peak,
        "L_int_last": sum(losses[-20:]) / max(1, min(20, len(losses))) if losses else float("nan"),
        "L2_err": err, "nan": nan_hit,
    }


def _fail_row(prob, variant, model, seed, msg):
    return {
        "problem": prob.name, "order": prob.order, "variant": variant, "seed": seed,
        "n_terms": len(prob.terms), "params": n_params(model),
        "backend": "jet" if variant in JET_VARIANTS else "autograd",
        "steps": 0, "ms_per_step": float("nan"), "peak_mb": float("nan"),
        "L_int_last": float("nan"), "L2_err": float("inf"), "nan": True, "error": msg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=100.0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-int", type=int, default=4096)
    ap.add_argument("--n-bc", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--variants",
                    default="complex_sinh,real_sinh,tanh,siren,fourier,mscale,cauchy")
    ap.add_argument("--problems", default="helmholtz_hf,kdv,biharmonic,triharmonic")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    allp = make_problems()
    problems = [allp[n] for n in args.problems.split(",") if n]
    variants = [v for v in args.variants.split(",") if v]

    print(f"device={device} hidden={args.hidden} depth={args.depth} "
          f"budget={args.seconds}s/run seeds={args.seeds}", flush=True)
    rows = []
    for prob in problems:
        print(f"\n=== {prob.name}  (d={prob.d}, order={prob.order}, "
              f"terms={len(prob.terms)}) ===", flush=True)
        print(f"{'variant':<13} {'params':>8} {'be':>8} {'steps':>7} {'ms/step':>8} "
              f"{'peakMB':>7} {'L_int':>10} {'L2_err':>11}", flush=True)
        for seed in range(args.seeds):
            for v in variants:
                r = run_variant(prob, v, seconds=args.seconds, hidden=args.hidden,
                                depth=args.depth, n_int=args.n_int, n_bc=args.n_bc,
                                lr=args.lr, seed=seed, device=device)
                rows.append(r)
                print(f"{v:<13} {r['params']:>8} {r['backend']:>8} {r['steps']:>7} "
                      f"{r['ms_per_step']:>8.2f} {r['peak_mb']:>7.0f} "
                      f"{r['L_int_last']:>10.2e} {r['L2_err']:>11.3e}  (seed {seed})",
                      flush=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in rows for k in r})
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        with out.with_suffix(".json").open("w") as f:
            json.dump(rows, f, indent=2)
        print(f"\n[ok] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
