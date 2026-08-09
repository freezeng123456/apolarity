#!/usr/bin/env python3
"""Shared machinery for the oscillatory high-order PINN benchmark suite.

Provides:
  * the four formal architectures (complex sinh, upstream-faithful SIREN,
    mFF-PINN, and MscaleDNN-2-sin), plus explicitly auxiliary tanh/Cauchy
    implementations;
  * real-trainable-parameter accounting and integer width matching against the
    complex-sinh H=128 capacity reference;
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
from apolarity.taylor_jet import TaylorJet, jet_forward_sequential

OMEGA0 = 10.0  # legacy call-site default for the complex-sinh frequency init
SIREN_FIRST_OMEGA0 = 30.0
SIREN_HIDDEN_OMEGA0 = 30.0
MSCALE_SCALES = (1.0, 2.0, 4.0)
MSCALE5_SCALES = (1.0, 2.0, 4.0, 8.0, 16.0)
FORMAL_VARIANTS = ("complex_sinh", "siren", "fourier", "mscale")


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------
class SinhActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sinh(x)


class Sin(nn.Module):  # name recognised by taylor_jet._is_sin_module
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


class ScaledSin(nn.Module):
    """Official SIREN activation ``sin(omega0 * linear(x))``."""

    def __init__(self, omega0: float):
        super().__init__()
        if omega0 <= 0:
            raise ValueError("omega0 must be positive")
        self.omega0 = float(omega0)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega0 * x)


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


def siren_init_(
    net: nn.Sequential,
    first_omega0: float = SIREN_FIRST_OMEGA0,
    hidden_omega0: float = SIREN_HIDDEN_OMEGA0,
) -> None:
    """Reproduce ``vsitzmann/siren`` SineLayer initialization.

    Only weights are overwritten. Biases intentionally retain the
    ``nn.Linear`` default, matching the upstream PyTorch implementation.
    """
    linears = [m for m in net if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for i, lin in enumerate(linears):
            fan_in = lin.weight.shape[1]
            if i == 0:
                bound = 1.0 / fan_in
            else:
                bound = math.sqrt(6.0 / fan_in) / hidden_omega0
            lin.weight.uniform_(-bound, bound)


def build_siren(
    d: int,
    H: int,
    depth: int,
    first_omega0: float = SIREN_FIRST_OMEGA0,
    hidden_omega0: float = SIREN_HIDDEN_OMEGA0,
    out: int = 1,
) -> nn.Sequential:
    """Official SIREN parameterization with ``depth`` sine hidden layers."""
    if depth < 1:
        raise ValueError("SIREN depth must be at least one")
    layers: list[nn.Module] = [nn.Linear(d, H), ScaledSin(first_omega0)]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(H, H), ScaledSin(hidden_omega0)])
    layers.append(nn.Linear(H, out))
    net = nn.Sequential(*layers)
    siren_init_(net, first_omega0, hidden_omega0)
    return net.to(dtype=torch.float64)


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


def _fourier_feature_map(d: int, frequencies: int, sigma: float) -> nn.Sequential:
    """Frozen ``[sin(Bx), cos(Bx)]`` map used by MultiscalePINNs.

    ``sigma`` is the angular-frequency standard deviation; no extra ``2*pi``
    factor is applied.
    """
    if frequencies < 1:
        raise ValueError("frequencies must be positive")
    if sigma <= 0:
        raise ValueError("Fourier sigma must be positive")
    B = torch.randn(frequencies, d, dtype=torch.float64) * sigma
    first = nn.Linear(d, 2 * frequencies, bias=True, dtype=torch.float64)
    with torch.no_grad():
        first.weight.copy_(torch.cat([B, B], dim=0))
        first.bias.copy_(
            torch.cat([
                torch.zeros(frequencies, dtype=torch.float64),
                torch.full((frequencies,), math.pi / 2, dtype=torch.float64),
            ])
        )
    first.requires_grad_(False)
    return nn.Sequential(first, Sin())


class FourierPINN(nn.Module):
    """Multiscale Fourier-feature PINN with a shared tanh trunk.

    This is the ``NN_mFF`` contract from MultiscalePINNs: two frozen Fourier
    maps, shared trainable hidden layers, branch concatenation, linear output.
    """

    def __init__(
        self,
        d: int,
        H: int,
        depth: int,
        sigma: float,
        out: int = 1,
        *,
        input_mean: tuple[float, ...] | None = None,
        input_std: tuple[float, ...] | None = None,
    ):
        super().__init__()
        if H < 2 or H % 2:
            raise ValueError("FourierPINN width must be an even integer >= 2")
        if depth < 1:
            raise ValueError("FourierPINN depth must be at least one")
        mean = input_mean if input_mean is not None else (0.0,) * d
        # All formal domains are [-1, 1]^d. Use their exact uniform moments
        # instead of estimating normalization from a method-dependent sample.
        std = input_std if input_std is not None else (1.0 / math.sqrt(3.0),) * d
        if len(mean) != d or len(std) != d or any(s <= 0 for s in std):
            raise ValueError("input_mean/input_std must match d and std must be positive")
        self.register_buffer("input_mean", torch.tensor(mean, dtype=torch.float64))
        self.register_buffer("input_std", torch.tensor(std, dtype=torch.float64))
        frequencies = H // 2
        self.branch_sigmas = (1.0, float(sigma))
        self.feature_maps = nn.ModuleList(
            [_fourier_feature_map(d, frequencies, s) for s in self.branch_sigmas]
        )
        trunk_layers: list[nn.Module] = []
        for _ in range(depth):
            trunk_layers.extend([nn.Linear(H, H), nn.Tanh()])
        self.trunk = nn.Sequential(*trunk_layers).to(dtype=torch.float64)
        self.output = nn.Linear(2 * H, out, dtype=torch.float64)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for module in [*self.trunk, self.output]:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_normal_(module.weight)
                    if module.bias is not None:
                        nn.init.normal_(module.bias)

    def _standardize(self, x: Tensor) -> Tensor:
        return (x - self.input_mean) / self.input_std

    def forward(self, x: Tensor) -> Tensor:
        xbar = self._standardize(x)
        branches = [self.trunk(feature(xbar)) for feature in self.feature_maps]
        return self.output(torch.cat(branches, dim=-1))

    def jet_forward(self, jet: TaylorJet) -> TaylorJet:
        standardized = TaylorJet([
            (jet.terms[0] - self.input_mean) / self.input_std,
            *[term / self.input_std for term in jet.terms[1:]],
        ])
        branch_jets = [
            jet_forward_sequential(
                self.trunk,
                jet_forward_sequential(feature, standardized),
            )
            for feature in self.feature_maps
        ]
        merged = TaylorJet([
            torch.cat([branch.terms[k] for branch in branch_jets], dim=-1)
            for k in range(jet.order + 1)
        ])
        return jet_forward_sequential(nn.Sequential(self.output), merged)


def build_fourier(
    d: int,
    H: int,
    depth: int,
    m_feat: int | None,
    sigma: float,
    out: int = 1,
) -> FourierPINN:
    """Build the formal mFF-PINN.

    ``m_feat`` is accepted for backward call-site compatibility. The upstream
    contract fixes each branch to ``H/2`` frequencies so its mapped width is H.
    """
    if m_feat not in (None, H // 2, H):
        raise ValueError("formal Fourier mapping uses H/2 frequencies per branch")
    return FourierPINN(d, H, depth, sigma, out=out)


class MultiScaleNet(nn.Module):
    """MscaleDNN-2-sin: independent ``F_k(a_k*x)`` subnets, summed output."""

    def __init__(self, d: int, H: int, depth: int, scales: tuple[float, ...], out: int = 1):
        super().__init__()
        if depth < 1:
            raise ValueError("MscaleDNN depth must be at least one")
        if not scales or any(a <= 0 for a in scales):
            raise ValueError("MscaleDNN scales must be positive")
        self.scales = tuple(float(a) for a in scales)
        self.subnets = nn.ModuleList([
            build_plain(d, H, depth, torch.float64, "sin", out=out)
            for _ in self.scales
        ])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Original-author Gaussian rule, corrected for each layer's true fan."""
        with torch.no_grad():
            for subnet in self.subnets:
                for layer in subnet:
                    if isinstance(layer, nn.Linear):
                        fan_in, fan_out = layer.weight.shape[1], layer.weight.shape[0]
                        std = 2.0 / math.sqrt(fan_in + fan_out)
                        nn.init.normal_(layer.weight, mean=0.0, std=std)
                        if layer.bias is not None:
                            nn.init.normal_(layer.bias, mean=0.0, std=std)

    def forward(self, x: Tensor) -> Tensor:
        return sum(
            subnet(scale * x)
            for scale, subnet in zip(self.scales, self.subnets, strict=True)
        )

    def jet_forward(self, jet: TaylorJet) -> TaylorJet:
        outputs = []
        for scale, subnet in zip(self.scales, self.subnets, strict=True):
            scaled = TaylorJet([term * scale for term in jet.terms])
            outputs.append(jet_forward_sequential(subnet, scaled))
        return TaylorJet([
            sum(output.terms[k] for output in outputs)
            for k in range(jet.order + 1)
        ])


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


JET_VARIANTS = {
    "complex_sinh", "complex_sinh_noinit", "tanh",
    "siren", "fourier", "mscale", "mscale5",
}
# ``complex_sinh_autodiff`` intentionally shares the proposed model and
# initialization with ``complex_sinh``; only its coordinate-derivative backend
# changes to direct nested autodiff.
AUTODIFF_VARIANTS = {"complex_sinh_autodiff"}
COMPLEX_VARIANTS = {"complex_sinh", "complex_sinh_noinit", *AUTODIFF_VARIANTS}


def variant_width(variant: str, H: int, mult: float = 1.0) -> int:
    """Return an explicitly requested literal width.

    The formal protocol passes H=128 to every architecture. This compatibility
    helper never performs hidden rescaling at model-construction time.
    """
    return H


def build_model(variant: str, d: int, H: int, depth: int, *, out: int = 1,
                omega0: float = OMEGA0, fourier_sigma: float = 2.0,
                real_width_mult: float = 1.0):
    """Construct a formal method or an explicitly auxiliary architecture.

    ``omega0`` applies only to the proposed complex-sinh initialization.
    Formal SIREN keeps the upstream default omega values of 30; Fourier sigma is
    interpreted in standardized coordinates without an extra ``2*pi``.
    """
    He = variant_width(variant, H, real_width_mult)
    if variant in {"complex_sinh", "complex_sinh_autodiff"}:
        net = build_plain(d, H, depth, torch.complex128, "sinh", out=out)
        complex_freq_init_(net, omega0)
        return net, torch.complex128
    if variant == "complex_sinh_noinit":
        return build_plain(d, H, depth, torch.complex128, "sinh", out=out), torch.complex128
    if variant == "tanh":
        return build_plain(d, He, depth, torch.float64, "tanh", out=out), torch.float64
    if variant == "siren":
        return build_siren(d, He, depth, out=out), torch.float64
    if variant == "fourier":
        return build_fourier(
            d, He, depth, He // 2, sigma=fourier_sigma, out=out
        ), torch.float64
    if variant == "mscale":
        return MultiScaleNet(d, He, depth, MSCALE_SCALES, out=out), torch.float64
    if variant == "mscale5":
        return MultiScaleNet(d, He, depth, MSCALE5_SCALES, out=out), torch.float64
    if variant == "cauchy":
        return CauchyNet(d, He, depth, out=out), torch.float64
    raise ValueError(variant)


def n_params(model: nn.Module) -> int:
    """Count trainable real degrees of freedom (complex scalar = two reals)."""
    tot = 0
    for p in model.parameters():
        if p.requires_grad:
            tot += p.numel() * (2 if p.dtype.is_complex else 1)
    return tot


@dataclass(frozen=True)
class ArchitectureSpec:
    method: str
    width: int
    real_dof: int
    reference_real_dof: int
    relative_dof_difference: float
    representation: str


def architecture_real_dof(
    variant: str,
    d: int,
    width: int,
    depth: int,
    *,
    representation: str = "real",
    out: int = 1,
    omega0: float = OMEGA0,
    fourier_sigma: float = 2.0,
) -> int:
    """Return total trainable real DOF for one physical field."""
    if representation not in {"real", "native_complex", "split_real"}:
        raise ValueError(f"unknown representation {representation!r}")
    if variant.startswith("complex") and representation != "native_complex":
        raise ValueError("complex variants require representation='native_complex'")
    if not variant.startswith("complex") and representation == "native_complex":
        raise ValueError("real variants cannot use native_complex representation")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model, _ = build_model(
            variant,
            d,
            width,
            depth,
            out=out,
            omega0=omega0,
            fourier_sigma=fourier_sigma,
        )
    multiplier = 2 if representation == "split_real" else 1
    return multiplier * n_params(model)


def formal_architecture_specs(
    d: int,
    depth: int = 4,
    *,
    literal_width: int = 128,
    split_real_baselines: bool = False,
    omega0: float = OMEGA0,
    fourier_sigma: float = 2.0,
    variants: tuple[str, ...] | None = None,
) -> dict[str, ArchitectureSpec]:
    """Build the fixed-literal-width four-method architecture table.

    Real trainable DOF is reported for transparency but is not used to change
    any method's width.
    """
    reference = architecture_real_dof(
        "complex_sinh",
        d,
        literal_width,
        depth,
        representation="native_complex",
        omega0=omega0,
        fourier_sigma=fourier_sigma,
    )
    specs: dict[str, ArchitectureSpec] = {}
    selected_variants = FORMAL_VARIANTS if variants is None else variants
    for variant in selected_variants:
        representation = (
            "native_complex"
            if variant in COMPLEX_VARIANTS
            else ("split_real" if split_real_baselines else "real")
        )
        dof = architecture_real_dof(
            variant,
            d,
            literal_width,
            depth,
            representation=representation,
            omega0=omega0,
            fourier_sigma=fourier_sigma,
        )
        specs[variant] = ArchitectureSpec(
            variant,
            literal_width,
            dof,
            reference,
            abs(dof - reference) / reference,
            representation,
        )
    return specs


# ---------------------------------------------------------------------------
# Predictions / derivatives (dispatched per architecture)
# ---------------------------------------------------------------------------
def predict(model: nn.Module, x: Tensor) -> Tensor:
    out = model(x)
    return out


def deriv_alpha(
    model: nn.Module,
    x: Tensor,
    alpha: tuple[int, ...],
    *,
    backend: str | None = None,
) -> Tensor:
    """Single-monomial partial d^alpha for a scalar-output model.

    Uses the complex Waring + Taylor-jet backend where supported and nested
    coordinate autodiff for the Cauchy net.  A caller can request the direct
    autodiff backend explicitly for a same-architecture control.
    """
    if backend is None:
        backend = "direct_autodiff" if isinstance(model, CauchyNet) else "waring_complex_jet"
    target = model.net if isinstance(model, CauchyNet) else model
    return single_monomial_partial(target, x, alpha, backend=backend)


# ---------------------------------------------------------------------------
# Complex-valued field wrappers: native complex net vs split-real RVPINN.
# A single complex net represents u: R^d -> C directly; the real baselines need
# two independent real nets (Re, Im), each at the same literal width H.
# ---------------------------------------------------------------------------
class ComplexField:
    def __init__(self, model, mdt, derivative_backend: str = "waring_complex_jet"):
        self.model, self.mdt = model, mdt
        self.derivative_backend = derivative_backend
        self.module = model

    def pred(self, x):
        return predict(self.model, x.to(self.mdt)).squeeze(-1)

    def deriv(self, x, alpha):
        return deriv_alpha(
            self.model, x.to(self.mdt), alpha, backend=self.derivative_backend
        ).squeeze(-1)


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
        backend = "direct_autodiff" if variant in AUTODIFF_VARIANTS else "waring_complex_jet"
        return ComplexField(m.to(device), mdt, backend), True
    re, _ = build_model(variant, d, H, depth, out=1, omega0=omega0,
                        fourier_sigma=sigma, real_width_mult=1.0)
    im, _ = build_model(variant, d, H, depth, out=1, omega0=omega0,
                        fourier_sigma=sigma, real_width_mult=1.0)
    return SplitRealField(re.to(device), im.to(device)), False


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def sample_interior(
    B: int,
    d: int,
    *,
    device,
    lo: float = -1.0,
    hi: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    return torch.empty(B, d, device=device, dtype=torch.float64).uniform_(
        lo, hi, generator=generator
    )


def sample_boundary(
    B: int,
    d: int,
    *,
    device,
    lo: float = -1.0,
    hi: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    x = torch.empty(B, d, device=device, dtype=torch.float64).uniform_(
        lo, hi, generator=generator
    )
    face = torch.randint(0, d, (B,), device=device, generator=generator)
    sign = torch.where(torch.rand(B, device=device, generator=generator) < 0.5,
                       torch.tensor(lo, dtype=torch.float64, device=device),
                       torch.tensor(hi, dtype=torch.float64, device=device))
    x[torch.arange(B, device=device), face] = sign
    return x


def laplacian_power_terms(d: int, j: int) -> list[tuple[float, tuple[int, ...]]]:
    """Expand ``Delta^j`` into monomial partials in any positive dimension.

    For a weak composition ``k_0 + ... + k_{d-1} = j``, the corresponding
    term has multinomial coefficient ``j! / prod_i k_i!`` and differentiates
    coordinate ``i`` exactly ``2 k_i`` times.
    """
    if d < 1:
        raise ValueError("d must be positive")
    if j < 0:
        raise ValueError("j must be non-negative")

    compositions: list[tuple[int, ...]] = []

    def visit(remaining: int, parts: int, prefix: tuple[int, ...]) -> None:
        if parts == 1:
            compositions.append(prefix + (remaining,))
            return
        for value in range(remaining + 1):
            visit(remaining - value, parts - 1, prefix + (value,))

    visit(j, d, ())
    numerator = math.factorial(j)
    terms = []
    for powers in compositions:
        coeff = numerator
        alpha = []
        for coordinate, power in enumerate(powers):
            coeff //= math.factorial(power)
            alpha.extend([coordinate] * (2 * power))
        terms.append((float(coeff), tuple(alpha)))
    return terms


# ---------------------------------------------------------------------------
# Generic train / eval (fixed dense collocation, equal wall-clock budget)
# ---------------------------------------------------------------------------
def train_eval(model, model_dtype, loss_fn, eval_fn, *, seconds, lr, device,
               lr_schedule="constant", lr_final=None, record_history=False,
               history_every_steps=20, history_eval_fn=None):
    """loss_fn() -> (loss_tensor, L_int_float).  eval_fn() -> float (rel L2).

    lr_schedule: "constant" or "cosine".  Cosine uses a TIME fraction (elapsed /
    seconds) since the loop is wall-clock bounded, decaying lr -> lr_final.  The
    same schedule is applied to every architecture, so the comparison stays fair.

    record_history: if True, every history_every_steps training steps record
    (training-elapsed, rel L2, total loss, interior loss).  Periodic and final
    eval wall time is tracked in eval_accum and excluded from the training budget
    (elapsed and ms/step), so every run receives the same training-time budget.
    history_eval_fn, if given, is used for periodic snapshots; eval_fn is always
    used for the final reported rel L2.
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
                "L_int_last": float("nan"), "loss_last": float("nan"),
                "L2_err": float("inf"), "rel_error": float("inf"), "nan": True,
                "error": repr(e)[:200]}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps, losses, nan_hit = 0, [], False
    total_losses: list[float] = []
    history, eval_accum = [], 0.0
    snap_eval = history_eval_fn if history_eval_fn is not None else eval_fn
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
        total_loss = float(loss.detach().item())
        total_losses.append(total_loss)
        if not math.isfinite(L_int):
            nan_hit = True
            break
        if record_history and history_every_steps > 0 and steps % history_every_steps == 0:
            te = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize()
            history.append([
                round(elapsed, 3), float(snap_eval()), total_loss, float(L_int)
            ])
            eval_accum += time.perf_counter() - te
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0 - eval_accum
    te = time.perf_counter()
    err = eval_fn()
    if record_history:
        eval_accum += time.perf_counter() - te
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else float("nan")
    rel_error = err
    loss_last = total_losses[-1] if total_losses else float("nan")
    out = {"steps": steps, "ms_per_step": 1000.0 * elapsed / max(1, steps),
           "peak_mb": peak,
           "L_int_last": sum(losses[-20:]) / max(1, min(20, len(losses))) if losses else float("nan"),
           "loss_last": loss_last,
           "L2_err": err, "rel_error": rel_error, "nan": nan_hit}
    if record_history:
        history.append([
            round(elapsed, 3), float(err), loss_last,
            losses[-1] if losses else float("nan")
        ])
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
    # When set, entries are ordered as [u, Delta u, Delta^2 u, ...] and are
    # applied component-wise.  ``bc_weight`` remains the backwards-compatible
    # scalar path for legacy experiments.
    bc_weights: tuple[float, ...] | None = None
    extra: dict = field(default_factory=dict)


def linear_loss_factory(
    prob: LinearProblem,
    x_int,
    x_bc,
    model_dtype,
    *,
    derivative_backend: str | None = None,
):
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
            t = deriv_alpha(model, xi, alpha, backend=derivative_backend).real
            res = coeff * t if res is None else res + coeff * t
        if zeroth_call:
            res = res + z_int * predict(model, xi).real
        elif prob.zeroth != 0.0:
            res = res + prob.zeroth * predict(model, xi).real
        L_int = (((res - f) / prob.res_scale) ** 2).mean()

        u_bc = predict(model, xb).real
        bc_terms = [((u_bc - bc_t) ** 2).mean()]
        for j in prob.bc_lap_powers:
            lap = None
            for coeff, alpha in laplacian_power_terms(prob.d, j):
                t = deriv_alpha(model, xb, alpha, backend=derivative_backend).real
                lap = coeff * t if lap is None else lap + coeff * t
            tgt = ((-prob.S) ** j) * prob.u_exact(x_bc).unsqueeze(-1)
            bc_terms.append((((lap - tgt) / (prob.S ** j)) ** 2).mean())

        if prob.bc_weights is None:
            loss = L_int + prob.bc_weight * sum(bc_terms)
        else:
            if len(prob.bc_weights) != len(bc_terms):
                raise ValueError(
                    f"{prob.name}: expected {len(bc_terms)} boundary weights, "
                    f"got {len(prob.bc_weights)}"
                )
            loss = L_int + sum(weight * term for weight, term in zip(prob.bc_weights, bc_terms))
        if complex_params:
            loss = loss + 1e-6 * sum((p.imag ** 2).mean()
                                     for p in model.parameters() if p.requires_grad)
        return loss, L_int.item()

    return loss_fn


def run_linear_suite(problems, variants, args, out_csv: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sk = sched_kwargs(args)
    seed_start = getattr(args, "seed_start", 0)
    seed_ids = list(range(seed_start, seed_start + args.seeds))
    print(f"device={device} hidden={args.hidden} depth={args.depth} "
          f"budget={args.seconds}s/run seeds={seed_ids} lr_schedule={sk['lr_schedule']}",
          flush=True)
    rows = []
    for prob in problems:
        print(f"\n=== {prob.name}  (d={prob.d}, order={prob.order}, "
              f"sweep={prob.sweep}, terms={len(prob.terms)}) ===", flush=True)
        print(f"{'variant':<24} {'params':>8} {'be':>8} {'steps':>7} {'ms/step':>8} "
              f"{'loss':>11} {'L_int':>11} {'rel_error':>12}", flush=True)
        g = torch.Generator(device=device).manual_seed(12345)
        lo, hi = prob.box
        eval_r = torch.empty(8192, prob.d, device=device, dtype=torch.float64).uniform_(lo, hi, generator=g)
        hist_n = min(getattr(args, "history_eval_n", 4096), eval_r.shape[0])
        eval_r_hist = eval_r[:hist_n]
        om = prob.extra.get("omega0", OMEGA0)
        fs = prob.extra.get("fourier_sigma", 2.0)
        for seed in seed_ids:
            train_gen = torch.Generator(device=device).manual_seed(seed)
            x_int = sample_interior(
                args.n_int, prob.d, device=device, lo=lo, hi=hi, generator=train_gen
            )
            x_bc = sample_boundary(
                args.n_bc, prob.d, device=device, lo=lo, hi=hi, generator=train_gen
            )
            for v in variants:
                torch.manual_seed(seed)
                model, mdt = build_model(v, prob.d, args.hidden, args.depth,
                                         omega0=om, fourier_sigma=fs)
                model = model.to(device)
                derivative_backend = (
                    "direct_autodiff"
                    if v in AUTODIFF_VARIANTS else None
                )
                lf = linear_loss_factory(
                    prob, x_int, x_bc, mdt,
                    derivative_backend=derivative_backend,
                )

                def eval_fn():
                    with torch.no_grad():
                        pred = predict(model, eval_r.to(mdt)).real.squeeze(-1)
                        tgt = prob.u_exact(eval_r)
                        return (((pred - tgt) ** 2).mean().sqrt()
                                / (tgt ** 2).mean().sqrt()).item()

                def history_eval_fn():
                    with torch.no_grad():
                        pred = predict(model, eval_r_hist.to(mdt)).real.squeeze(-1)
                        tgt = prob.u_exact(eval_r_hist)
                        return (((pred - tgt) ** 2).mean().sqrt()
                                / (tgt ** 2).mean().sqrt()).item()

                m = train_eval(model, mdt, lambda: lf(model), eval_fn,
                               seconds=args.seconds, lr=args.lr, device=device,
                               history_eval_fn=history_eval_fn, **sk)
                weights = prob.bc_weights
                if weights is None:
                    weights = (prob.bc_weight,) * (1 + len(prob.bc_lap_powers))
                row = {"problem": prob.name, "order": prob.order, "sweep": prob.sweep,
                       "variant": v, "seed": seed, "params": n_params(model),
                       "backend": "jet" if v in JET_VARIANTS else "autograd",
                       "hidden": args.hidden, "depth": args.depth,
                       "budget_seconds": args.seconds, "n_int": args.n_int,
                       "n_bc": args.n_bc, "lr": args.lr,
                       "boundary_weights": json.dumps(list(weights)),
                       "lr_schedule": sk["lr_schedule"], "omega0": om,
                       "fourier_sigma": fs, "collocation": "paired_seed_v1", **m}
                row["rel_error"] = row["L2_err"]
                rows.append(row)
                print(f"{v:<24} {row['params']:>8} {row['backend']:>8} {m['steps']:>7} "
                      f"{m['ms_per_step']:>8.2f} {m['loss_last']:>11.3e} "
                      f"{m['L_int_last']:>11.3e} {m['rel_error']:>12.3e}  (seed {seed})",
                      flush=True)
                # High-dimensional operator sums can use most of GPU memory.
                # Release per-run closures and cached blocks before constructing
                # the next architecture so an OOM in one variant does not
                # contaminate subsequent runs through allocator fragmentation.
                del model, lf, eval_fn, history_eval_fn
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del x_int, x_bc
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
    ap.add_argument("--history-every-steps", type=int, default=20,
                    help="with --history: rel-L2 snapshot every N training steps "
                         "(eval wall time excluded from training budget)")
    ap.add_argument("--history-eval-n", type=int, default=4096,
                    help="collocation points for periodic history rel-L2 (final uses 8192)")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed index (inclusive); runs seed-start .. seed-start+seeds-1")
    ap.add_argument("--variants",
                    default="complex_sinh,complex_sinh_autodiff")
    ap.add_argument("--out", default="")
    return ap


def sched_kwargs(args) -> dict:
    return {"lr_schedule": getattr(args, "lr_schedule", "constant"),
            "lr_final": getattr(args, "lr_final", None),
            "record_history": getattr(args, "history", False),
            "history_every_steps": getattr(args, "history_every_steps", 20)}
