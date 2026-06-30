#!/usr/bin/env python3
"""Shared machinery for the oscillatory high-order PINN benchmark suite.

Provides:
  * architectures (complex sinh w/ holomorphic init, real sinh, tanh, SIREN,
    Fourier-features, MscaleDNN, compleX-PINN/Cauchy).  All variants use the
    same literal hidden width H passed on the command line (no sqrt(2)
    rescaling).  The width study runs real baselines at H=128 and complex sinh
    at both H=64 and H=128 so the two complex widths bracket the reals in
    parameter count (each complex weight counts as 2 real DOF in n_params).
  * a single exact derivative backend (complex Waring + Taylor jet) for all
    jet-compatible nets, with nested autograd as the fallback (Cauchy);
  * a generic train/eval loop (fixed dense collocation, equal wall-clock budget)
  * a LinearProblem dataclass + run_linear_suite for linear operator PDEs.

Each benchmark lives in its own experiments/exp_*.py and imports from here.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]  # repo root (experiments/common/osc_common.py)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn
from torch import Tensor

from apolarity import single_monomial_partial

OMEGA0 = 10.0  # frequency scale for SIREN / complex-SIREN (holomorphic) inits


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
    """compleX-PINN activation Phi(x; m1, m2, d) = (m1 x + m2)/(x^2 + d^2)."""

    def __init__(self, width: int):
        super().__init__()
        self.m1 = nn.Parameter(torch.ones(width))
        self.m2 = nn.Parameter(torch.zeros(width))
        self.d = nn.Parameter(torch.full((width,), 1.0))

    def forward(self, x: Tensor) -> Tensor:
        return (self.m1 * x + self.m2) / (x * x + self.d * self.d)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _seq(layers: list[nn.Module], dtype: torch.dtype) -> nn.Sequential:
    return nn.Sequential(*layers).to(dtype=dtype)


def build_plain(d: int, H: int, depth: int, dtype: torch.dtype, act: str,
                out: int = 1) -> nn.Sequential:
    def a():
        return {"sinh": SinhActivation, "tanh": nn.Tanh, "sin": Sin}[act]()
    layers: list[nn.Module] = []
    in_dim = d
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, H))
        layers.append(a())
        in_dim = H
    layers.append(nn.Linear(in_dim, out))
    return _seq(layers, dtype)


def siren_init_(net: nn.Sequential, omega0: float) -> None:
    linears = [m for m in net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for i, lin in enumerate(linears):
            fan_in = lin.weight.shape[1]
            if i == 0:
                bound = omega0 / fan_in
            elif i == len(linears) - 1:
                bound = math.sqrt(6.0 / fan_in) * 1e-1
            else:
                bound = math.sqrt(6.0 / fan_in)
            lin.weight.uniform_(-bound, bound)
            if lin.bias is not None:
                lin.bias.uniform_(-bound, bound)


def build_siren(d: int, H: int, depth: int, omega0: float, out: int = 1) -> nn.Sequential:
    net = build_plain(d, H, depth, torch.float64, "sin", out=out)
    siren_init_(net, omega0)
    return net


def complex_freq_init_(net: nn.Sequential, omega0: float) -> None:
    """Frequency-rich ('complex-SIREN' / holomorphic) init: oscillation of
    Re(sinh(z)) lives in Im(z), so inject frequency via the imaginary part of
    the first-layer weights (small real part keeps cosh growth bounded)."""
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


def build_fourier(d: int, H: int, depth: int, m_feat: int, sigma: float,
                  out: int = 1) -> nn.Sequential:
    """Random Fourier features [sin(Bx), cos(Bx)] as frozen Linear(d->2m)+Sin,
    then a tanh MLP.  Jet-compatible."""
    B = torch.randn(m_feat, d, dtype=torch.float64) * sigma
    W0 = torch.cat([B, B], dim=0)
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
    layers.append(nn.Linear(in_dim, out))
    return _seq(layers, torch.float64)


class MultiScaleNet(nn.Module):
    """MscaleDNN: output = sum_k subnet_k(a_k * x), each a sine MLP."""

    def __init__(self, d: int, H: int, depth: int, scales: tuple[float, ...], out: int = 1):
        super().__init__()
        self.subnets = nn.ModuleList()
        for a in scales:
            sub = build_plain(d, H, depth, torch.float64, "sin", out=out)
            with torch.no_grad():
                first = next(m for m in sub if isinstance(m, nn.Linear))
                first.weight.mul_(a)
            self.subnets.append(sub)

    def forward(self, x: Tensor) -> Tensor:
        return sum(sub(x) for sub in self.subnets)


class CauchyNet(nn.Module):
    """compleX-PINN: MLP with learnable Cauchy activation (real-valued)."""

    def __init__(self, d: int, H: int, depth: int, out: int = 1):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = d
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, H))
            layers.append(CauchyActivation(H))
            in_dim = H
        layers.append(nn.Linear(in_dim, out))
        self.net = nn.Sequential(*layers).to(dtype=torch.float64)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


JET_VARIANTS = {"complex_sinh", "complex_sinh_noinit", "tanh",
                "siren", "fourier", "mscale"}
COMPLEX_VARIANTS = {"complex_sinh", "complex_sinh_noinit"}


def variant_width(variant: str, H: int, mult: float = 1.0) -> int:
    """Every architecture uses the LITERAL hidden width H -- depth and width are
    matched across methods (no parameter-count rescaling).  The complex net's
    comparability (its complex weights carry ~2x the real DOF) is handled by
    EVALUATING IT AT TWO WIDTHS that bracket the real baselines, e.g. complex at
    64 and 128 against the real baselines at 128.  The `mult` argument is kept for
    call-site compatibility but ignored."""
    return H


def build_model(variant: str, d: int, H: int, depth: int, *, out: int = 1,
                omega0: float = OMEGA0, fourier_sigma: float = 2.0,
                real_width_mult: float = math.sqrt(2.0)):
    """Frequency-aware inits (omega0 / fourier_sigma) let the frequency-rich
    architectures (SIREN, complex-sinh, Fourier) be matched to the problem's
    wavenumber so every method gets its best shot at each frequency."""
    He = variant_width(variant, H, real_width_mult)
    if variant == "complex_sinh":
        net = build_plain(d, H, depth, torch.complex128, "sinh", out=out)
        complex_freq_init_(net, omega0)
        return net, torch.complex128
    if variant == "complex_sinh_noinit":
        return build_plain(d, H, depth, torch.complex128, "sinh", out=out), torch.complex128
    if variant == "tanh":
        return build_plain(d, He, depth, torch.float64, "tanh", out=out), torch.float64
    if variant == "siren":
        return build_siren(d, He, depth, omega0, out=out), torch.float64
    if variant == "fourier":
        return build_fourier(d, He, depth, He, sigma=fourier_sigma, out=out), torch.float64
    if variant == "mscale":
        return MultiScaleNet(d, He, depth, (1.0, 2.0, 4.0), out=out), torch.float64
    if variant == "cauchy":
        return CauchyNet(d, He, depth, out=out), torch.float64
    raise ValueError(variant)


def n_params(model: nn.Module) -> int:
    tot = 0
    for p in model.parameters():
        if p.requires_grad:
            tot += p.numel() * (2 if p.dtype.is_complex else 1)
    return tot


# ---------------------------------------------------------------------------
# Predictions / derivatives (dispatched per architecture)
# ---------------------------------------------------------------------------
def predict(model: nn.Module, x: Tensor) -> Tensor:
    out = model(x)
    return out


def deriv_alpha(model: nn.Module, x: Tensor, alpha: tuple[int, ...]) -> Tensor:
    """Single-monomial partial d^alpha of the model output.  Returns same out_dim
    as the model.  Uses the complex Waring + Taylor-jet backend where supported,
    nested autograd for the Cauchy net."""
    if isinstance(model, MultiScaleNet):
        s = None
        for sub in model.subnets:
            t = single_monomial_partial(sub, x, alpha, backend="waring_complex_jet")
            s = t if s is None else s + t
        return s
    if isinstance(model, CauchyNet):
        return single_monomial_partial(model.net, x, alpha, backend="direct_autodiff")
    return single_monomial_partial(model, x, alpha, backend="waring_complex_jet")


# ---------------------------------------------------------------------------
# Complex-valued field wrappers: native complex net vs split-real RVPINN.
# A single complex net represents u: R^d -> C directly; the real baselines need
# two independent real nets (Re, Im), parameter-matched as a PAIR (mult=1).
# ---------------------------------------------------------------------------
class ComplexField:
    def __init__(self, model, mdt):
        self.model, self.mdt = model, mdt
        self.module = model

    def pred(self, x):
        return predict(self.model, x.to(self.mdt)).squeeze(-1)

    def deriv(self, x, alpha):
        return deriv_alpha(self.model, x.to(self.mdt), alpha).squeeze(-1)


class SplitRealField:
    def __init__(self, re, im):
        self.re, self.im = re, im
        self.module = nn.ModuleList([re, im])

    def pred(self, x):
        return predict(self.re, x).squeeze(-1) + 1j * predict(self.im, x).squeeze(-1)

    def deriv(self, x, alpha):
        return (deriv_alpha(self.re, x, alpha).squeeze(-1)
                + 1j * deriv_alpha(self.im, x, alpha).squeeze(-1))


def make_complex_field(variant, d, H, depth, device, *, omega0=OMEGA0, sigma=2.0):
    if variant.startswith("complex"):
        m, mdt = build_model(variant, d, H, depth, out=1, omega0=omega0, fourier_sigma=sigma)
        return ComplexField(m.to(device), mdt), True
    re, _ = build_model(variant, d, H, depth, out=1, omega0=omega0,
                        fourier_sigma=sigma, real_width_mult=1.0)
    im, _ = build_model(variant, d, H, depth, out=1, omega0=omega0,
                        fourier_sigma=sigma, real_width_mult=1.0)
    return SplitRealField(re.to(device), im.to(device)), False


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_interior(B: int, d: int, *, device, lo: float = -1.0, hi: float = 1.0) -> Tensor:
    return torch.empty(B, d, device=device, dtype=torch.float64).uniform_(lo, hi)


def sample_boundary(B: int, d: int, *, device, lo: float = -1.0, hi: float = 1.0) -> Tensor:
    x = torch.empty(B, d, device=device, dtype=torch.float64).uniform_(lo, hi)
    face = torch.randint(0, d, (B,), device=device)
    sign = torch.where(torch.rand(B, device=device) < 0.5,
                       torch.tensor(lo, dtype=torch.float64, device=device),
                       torch.tensor(hi, dtype=torch.float64, device=device))
    x[torch.arange(B, device=device), face] = sign
    return x


def laplacian_power_terms(d: int, j: int) -> list[tuple[float, tuple[int, ...]]]:
    """Expanded terms of Delta^j.  d=1: d^{2j}/dx^{2j}.  d=2: binomial expansion."""
    if d == 1:
        return [(1.0, (0,) * (2 * j))]
    if d == 2:
        return [(float(math.comb(j, i)), (0,) * (2 * i) + (1,) * (2 * (j - i)))
                for i in range(j + 1)]
    raise NotImplementedError("laplacian_power_terms implements d in {1,2}")


# ---------------------------------------------------------------------------
# Generic train / eval (fixed dense collocation, equal wall-clock budget)
# ---------------------------------------------------------------------------
def train_eval(model, model_dtype, loss_fn, eval_fn, *, seconds, lr, device,
               lr_schedule="constant", lr_final=None, record_history=False,
               history_n=40):
    """loss_fn() -> (loss_tensor, L_int_float).  eval_fn() -> float (rel L2).

    lr_schedule: "constant" or "cosine".  Cosine uses a TIME fraction (elapsed /
    seconds) since the loop is wall-clock bounded, decaying lr -> lr_final.  The
    same schedule is applied to every architecture, so the comparison stays fair.

    record_history: if True, sample (training-elapsed, rel L2, interior loss)
    ~history_n times across the run for convergence figures.  The periodic eval
    cost is excluded from the wall-clock budget so the step count stays
    comparable to a non-history run.
    """
    if lr_final is None:
        lr_final = lr * 0.1  # gentle floor: cosine helps high-order but a too-low
        #                      tail starves slower problems (e.g. KdV) at short budgets
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    try:
        for _ in range(5):  # warmup, not timed
            loss, _ = loss_fn()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    except Exception as e:  # noqa: BLE001
        return {"steps": 0, "ms_per_step": float("nan"), "peak_mb": float("nan"),
                "L_int_last": float("nan"), "L2_err": float("inf"), "nan": True,
                "error": repr(e)[:200]}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps, losses, nan_hit = 0, [], False
    history, eval_accum = [], 0.0
    hist_dt = seconds / max(1, history_n)
    next_hist = 0.0
    while True:
        elapsed = time.perf_counter() - t0 - eval_accum
        if elapsed >= seconds:
            break
        if lr_schedule == "cosine":
            frac = min(1.0, elapsed / seconds)
            cur_lr = lr_final + 0.5 * (lr - lr_final) * (1.0 + math.cos(math.pi * frac))
            for pg in opt.param_groups:
                pg["lr"] = cur_lr
        loss, L_int = loss_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        steps += 1
        losses.append(L_int)
        if not math.isfinite(L_int):
            nan_hit = True
            break
        if record_history and elapsed >= next_hist:
            te = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()
            history.append([round(elapsed, 3), float(eval_fn()), float(L_int)])
            next_hist += hist_dt
            eval_accum += time.perf_counter() - te
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0 - eval_accum
    err = eval_fn()
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else float("nan")
    out = {"steps": steps, "ms_per_step": 1000.0 * elapsed / max(1, steps),
           "peak_mb": peak,
           "L_int_last": sum(losses[-20:]) / max(1, min(20, len(losses))) if losses else float("nan"),
           "L2_err": err, "nan": nan_hit}
    if record_history:
        history.append([round(elapsed, 3), float(err),
                        losses[-1] if losses else float("nan")])
        out["history"] = history
    return out


# ---------------------------------------------------------------------------
# Linear operator problems:  sum_t c_t d^alpha_t u + c0 u = f  on a box
# ---------------------------------------------------------------------------
@dataclass
class LinearProblem:
    name: str
    d: int
    order: int
    terms: list[tuple[float, tuple[int, ...]]]
    zeroth: float | Callable[[Tensor], Tensor]
    u_exact: Callable[[Tensor], Tensor]
    source_f: Callable[[Tensor], Tensor]
    res_scale: float
    S: float = 0.0
    bc_lap_powers: tuple[int, ...] = ()
    box: tuple[float, float] = (-1.0, 1.0)
    sweep: float = 0.0          # the swept parameter value (k, mode, ...), for plots
    bc_weight: float = 100.0
    extra: dict = field(default_factory=dict)


def linear_loss_factory(prob: LinearProblem, x_int, x_bc, model_dtype):
    complex_params = model_dtype.is_complex
    xi = x_int.to(model_dtype)
    xb = x_bc.to(model_dtype)
    f = prob.source_f(x_int).unsqueeze(-1)
    bc_t = prob.u_exact(x_bc).unsqueeze(-1)
    # zeroth-order coefficient may be a constant OR a callable c(x) (variable
    # coefficient / scattering): precompute the spatial field once.
    zeroth_call = callable(prob.zeroth)
    z_int = (prob.zeroth(x_int).reshape(-1, 1) if zeroth_call else prob.zeroth)

    def loss_fn(model):
        res = None
        for coeff, alpha in prob.terms:
            t = deriv_alpha(model, xi, alpha).real
            res = coeff * t if res is None else res + coeff * t
        if zeroth_call:
            res = res + z_int * predict(model, xi).real
        elif prob.zeroth != 0.0:
            res = res + prob.zeroth * predict(model, xi).real
        L_int = (((res - f) / prob.res_scale) ** 2).mean()

        u_bc = predict(model, xb).real
        L_bc = ((u_bc - bc_t) ** 2).mean()
        for j in prob.bc_lap_powers:
            lap = None
            for coeff, alpha in laplacian_power_terms(prob.d, j):
                t = deriv_alpha(model, xb, alpha).real
                lap = coeff * t if lap is None else lap + coeff * t
            tgt = ((-prob.S) ** j) * prob.u_exact(x_bc).unsqueeze(-1)
            L_bc = L_bc + (((lap - tgt) / (prob.S ** j)) ** 2).mean()

        loss = L_int + prob.bc_weight * L_bc
        if complex_params:
            loss = loss + 1e-6 * sum((p.imag ** 2).mean()
                                     for p in model.parameters() if p.requires_grad)
        return loss, L_int.item()

    return loss_fn


def run_linear_suite(problems, variants, args, out_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sk = sched_kwargs(args)
    print(f"device={device} hidden={args.hidden} depth={args.depth} "
          f"budget={args.seconds}s/run seeds={args.seeds} lr_schedule={sk['lr_schedule']}",
          flush=True)
    rows = []
    for prob in problems:
        print(f"\n=== {prob.name}  (d={prob.d}, order={prob.order}, "
              f"sweep={prob.sweep}, terms={len(prob.terms)}) ===", flush=True)
        print(f"{'variant':<20} {'params':>8} {'be':>8} {'steps':>7} {'ms/step':>8} "
              f"{'peakMB':>7} {'L_int':>10} {'L2_err':>11}", flush=True)
        g = torch.Generator(device=device).manual_seed(12345)
        lo, hi = prob.box
        eval_r = torch.empty(8192, prob.d, device=device, dtype=torch.float64).uniform_(lo, hi, generator=g)
        om = prob.extra.get("omega0", OMEGA0)
        fs = prob.extra.get("fourier_sigma", 2.0)
        for seed in range(args.seeds):
            for v in variants:
                torch.manual_seed(seed)
                model, mdt = build_model(v, prob.d, args.hidden, args.depth,
                                         omega0=om, fourier_sigma=fs)
                model = model.to(device)
                x_int = sample_interior(args.n_int, prob.d, device=device, lo=lo, hi=hi)
                x_bc = sample_boundary(args.n_bc, prob.d, device=device, lo=lo, hi=hi)
                lf = linear_loss_factory(prob, x_int, x_bc, mdt)

                def eval_fn():
                    with torch.no_grad():
                        pred = predict(model, eval_r.to(mdt)).real.squeeze(-1)
                        tgt = prob.u_exact(eval_r)
                        return (((pred - tgt) ** 2).mean().sqrt()
                                / (tgt ** 2).mean().sqrt()).item()

                m = train_eval(model, mdt, lambda: lf(model), eval_fn,
                               seconds=args.seconds, lr=args.lr, device=device, **sk)
                row = {"problem": prob.name, "order": prob.order, "sweep": prob.sweep,
                       "variant": v, "seed": seed, "params": n_params(model),
                       "backend": "jet" if v in JET_VARIANTS else "autograd", **m}
                rows.append(row)
                print(f"{v:<20} {row['params']:>8} {row['backend']:>8} {m['steps']:>7} "
                      f"{m['ms_per_step']:>8.2f} {m['peak_mb']:>7.0f} "
                      f"{m['L_int_last']:>10.2e} {m['L2_err']:>11.3e}  (seed {seed})",
                      flush=True)
    write_rows(rows, out_csv)
    return rows


def write_rows(rows, out_csv: str):
    if not out_csv:
        return
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Per-step convergence traces are lists; split them into a sidecar JSON so the
    # CSV/JSON result rows stay flat and human-readable.
    histories, clean = [], []
    id_keys = ("problem", "order", "sweep", "variant", "seed", "rep")
    for r in rows:
        if "history" in r:
            r = dict(r)
            h = r.pop("history")
            histories.append({**{k: r.get(k) for k in id_keys if k in r},
                              "history": h})
        clean.append(r)
    rows = clean
    keys = sorted({k for r in rows for k in r})
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with out.with_suffix(".json").open("w") as f:
        json.dump(rows, f, indent=2)
    if histories:
        hpath = out.with_name(out.stem + "_history.json")
        with hpath.open("w") as f:
            json.dump(histories, f)
        print(f"[ok] wrote {hpath}  ({len(histories)} traces)", flush=True)
    print(f"\n[ok] wrote {out}", flush=True)


def default_argparser(seconds=80.0, n_int=4096, n_bc=512):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=seconds)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-int", type=int, default=n_int)
    ap.add_argument("--n-bc", type=int, default=n_bc)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-schedule", default="cosine", choices=["constant", "cosine"])
    ap.add_argument("--lr-final", type=float, default=None)
    ap.add_argument("--history", action="store_true",
                    help="record rel-L2 & loss vs time traces for convergence figures")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--variants",
                    default="complex_sinh,fourier,siren,mscale")
    ap.add_argument("--out", default="")
    return ap


def sched_kwargs(args) -> dict:
    return {"lr_schedule": getattr(args, "lr_schedule", "constant"),
            "lr_final": getattr(args, "lr_final", None),
            "record_history": getattr(args, "history", False)}
