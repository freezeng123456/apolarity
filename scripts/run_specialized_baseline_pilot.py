#!/usr/bin/env python3
"""Three-minute diagnostic pilots for problem-structured PDE baselines.

This is deliberately separate from the frozen ``jsc_v2`` protocol.  It runs one
representative setting per formal family with seed 0 and the shared collocation,
optimizer, and held-out evaluation conventions.

Baselines:
* Poly d=2, order=4: one-network mixed residual formulation (MIM-p style).
* Chirp a=2: WIRE complex Gabor INR adapted to a PINN residual.
* Maxwell a=4: one-hidden-layer plane-wave neural network (PWNN).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
sys.path.insert(0, str(COMMON))

from osc_common import (  # noqa: E402
    n_params,
    sample_boundary,
    sample_interior,
    train_eval,
    write_rows,
)


def direct_laplacian(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Batchwise Laplacian of one scalar output per input row."""
    if y.ndim != 1 or x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("expected y=(batch,) and x=(batch,d)")
    grad = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
    lap = torch.zeros_like(y)
    for coordinate in range(x.shape[1]):
        second = torch.autograd.grad(
            grad[:, coordinate].sum(), x, create_graph=True
        )[0][:, coordinate]
        lap = lap + second
    return lap


class TanhMLP(nn.Module):
    def __init__(self, d: int, width: int, depth: int, out: int):
        super().__init__()
        layers: list[nn.Module] = []
        fan_in = d
        for _ in range(depth):
            layers.extend([nn.Linear(fan_in, width), nn.Tanh()])
            fan_in = width
        layers.append(nn.Linear(fan_in, out))
        self.net = nn.Sequential(*layers).to(dtype=torch.float64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ComplexGaborLayer(nn.Module):
    """Float64/complex128 adaptation of vishwa91/wire ``ComplexGaborLayer``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        first: bool,
        omega0: float,
        sigma0: float,
    ):
        super().__init__()
        dtype = torch.float64 if first else torch.complex128
        self.linear = nn.Linear(in_features, out_features, dtype=dtype)
        self.register_buffer("omega0", torch.tensor(omega0, dtype=torch.float64))
        self.register_buffer("sigma0", torch.tensor(sigma0, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.linear(x)
        return torch.exp(1j * self.omega0 * z - (self.sigma0 * z).abs().square())


class WirePINN(nn.Module):
    """Source-faithful WIRE width accounting with a real scalar output."""

    def __init__(self, d: int, width: int, depth: int, omega0: float, sigma0: float):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be positive")
        # Official WIRE reduces complex hidden width by sqrt(2) for capacity.
        hidden = int(width / math.sqrt(2.0))
        layers: list[nn.Module] = [
            ComplexGaborLayer(
                d, hidden, first=True, omega0=omega0, sigma0=sigma0
            )
        ]
        for _ in range(depth - 1):
            layers.append(
                ComplexGaborLayer(
                    hidden, hidden, first=False, omega0=omega0, sigma0=sigma0
                )
            )
        layers.append(nn.Linear(hidden, 1, dtype=torch.complex128))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).real


class PlaneWaveNet(nn.Module):
    """PWNN: trainable complex amplitudes and real wave vectors."""

    def __init__(self, d: int, width: int, init_wavenumber: float):
        super().__init__()
        angles = 2.0 * math.pi * torch.rand(width, dtype=torch.float64)
        directions = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        if d != 2:
            raise ValueError("pilot PWNN currently supports d=2")
        self.wave_vectors = nn.Parameter(init_wavenumber * directions)
        scale = 1.0 / math.sqrt(width)
        amplitudes = scale * (
            torch.randn(width, dtype=torch.float64)
            + 1j * torch.randn(width, dtype=torch.float64)
        )
        self.amplitudes = nn.Parameter(amplitudes.to(torch.complex128))

    def pred_and_laplacian(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase = x @ self.wave_vectors.T
        waves = torch.exp(1j * phase.to(torch.complex128))
        pred = waves @ self.amplitudes
        norm2 = self.wave_vectors.square().sum(dim=1).to(torch.complex128)
        lap = waves @ (-norm2 * self.amplitudes)
        return pred, lap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pred_and_laplacian(x)[0]


def relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (
        (pred - target).abs().square().mean().sqrt()
        / (target.abs().square().mean().sqrt() + 1e-30)
    ).item()


def run_poly(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x_int = sample_interior(args.n_int, 2, device=device, generator=generator)
    x_bc = sample_boundary(args.n_bc, 2, device=device, generator=generator)
    x_int.requires_grad_(True)
    eval_generator = torch.Generator(device=device).manual_seed(args.eval_seed)
    x_eval = sample_interior(8192, 2, device=device, generator=eval_generator)

    model = TanhMLP(2, args.hidden, args.depth, 2).to(device)
    S = 2.0 * math.pi**2
    f_scaled = torch.sin(math.pi * x_int).prod(dim=1).detach()  # f / S^2

    bc_weight = args.bc_weight_poly

    def components():
        uv = model(x_int)
        u, v = uv[:, 0], uv[:, 1]  # v approximates Delta(u)/(-S)
        r1 = direct_laplacian(u, x_int) / S + v
        r2 = direct_laplacian(v, x_int) / S + f_scaled
        L_int = r1.square().mean() + r2.square().mean()
        uv_bc = model(x_bc)
        L_bc = uv_bc.square().mean()
        return L_int, L_bc

    def loss_fn():
        L_int, L_bc = components()
        return L_int + bc_weight * L_bc, L_int.item()

    def eval_fn():
        with torch.no_grad():
            pred = model(x_eval)[:, 0]
            target = torch.sin(math.pi * x_eval).prod(dim=1)
            return relative_l2(pred, target)

    metrics = train_eval(
        model,
        None,
        loss_fn,
        eval_fn,
        seconds=args.seconds,
        lr=args.lr,
        lr_schedule="cosine",
        lr_final=args.lr_final,
        device=device,
        record_history=True,
        history_every_steps=args.history_every_steps,
    )
    L_int, L_bc = components()
    return {
        "problem": "poly_d2_o4",
        "variant": "mim_p_shared",
        "seed": args.seed,
        "params": n_params(model),
        "representation": "real_two_output",
        "bc_weight": bc_weight,
        "L_int_final": L_int.item(),
        "L_bc_final": L_bc.item(),
        "loss_final": (L_int + bc_weight * L_bc).item(),
        **metrics,
    }


def chirp_exact(a: int, x: torch.Tensor) -> torch.Tensor:
    phi = 0.5 * a * math.pi * x.square().sum(dim=1)
    return torch.sin(phi)


def chirp_source(a: int, x: torch.Tensor) -> torch.Tensor:
    ap = a * math.pi
    r2 = x.square().sum(dim=1)
    phi = 0.5 * ap * r2
    lap = -(ap**2) * r2 * torch.sin(phi) + 2.0 * ap * torch.cos(phi)
    return -lap + torch.sin(phi)


def run_chirp(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x_int = sample_interior(args.n_int, 2, device=device, generator=generator)
    x_bc = sample_boundary(args.n_bc, 2, device=device, generator=generator)
    x_int.requires_grad_(True)
    eval_generator = torch.Generator(device=device).manual_seed(args.eval_seed)
    x_eval = sample_interior(8192, 2, device=device, generator=eval_generator)

    a = 2
    model = WirePINN(2, args.hidden, args.depth, omega0=2.0 * math.pi * a,
                     sigma0=args.wire_sigma).to(device)
    f = chirp_source(a, x_int).detach()
    bc = chirp_exact(a, x_bc)
    residual_scale = 2.0 * (a * math.pi) ** 2

    bc_weight = args.bc_weight_chirp

    def components():
        u = model(x_int).squeeze(1)
        residual = -direct_laplacian(u, x_int) + u - f
        L_int = (residual / residual_scale).square().mean()
        L_bc = (model(x_bc).squeeze(1) - bc).square().mean()
        return L_int, L_bc

    def loss_fn():
        L_int, L_bc = components()
        return L_int + bc_weight * L_bc, L_int.item()

    def eval_fn():
        with torch.no_grad():
            return relative_l2(model(x_eval).squeeze(1), chirp_exact(a, x_eval))

    metrics = train_eval(
        model,
        None,
        loss_fn,
        eval_fn,
        seconds=args.seconds,
        lr=args.lr,
        lr_schedule="cosine",
        lr_final=args.lr_final,
        device=device,
        record_history=True,
        history_every_steps=args.history_every_steps,
    )
    L_int, L_bc = components()
    return {
        "problem": "chirp_a2",
        "variant": "wire",
        "seed": args.seed,
        "params": n_params(model),
        "representation": "complex_gabor_real_output",
        "wire_sigma": args.wire_sigma,
        "wire_upstream_commit": "bf95232e0f60434bcbd9b4398ef4c11490832526",
        "bc_weight": bc_weight,
        "L_int_final": L_int.item(),
        "L_bc_final": L_bc.item(),
        "loss_final": (L_int + bc_weight * L_bc).item(),
        **metrics,
    }


def maxwell_exact(a: int, x: torch.Tensor) -> torch.Tensor:
    phase = a * math.pi * (x[:, 0] + x[:, 1])
    return torch.exp(1j * phase.to(torch.complex128))


def run_maxwell(args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x_int = sample_interior(args.n_int, 2, device=device, generator=generator)
    x_bc = sample_boundary(args.n_bc, 2, device=device, generator=generator)
    eval_generator = torch.Generator(device=device).manual_seed(args.eval_seed)
    x_eval = sample_interior(8192, 2, device=device, generator=eval_generator)

    a = 4
    ap = a * math.pi
    beta = 0.2
    kappa2 = (ap**2) * (1.0 + 1j * beta)
    forcing_multiplier = -2.0 * ap**2 + kappa2
    f = forcing_multiplier * maxwell_exact(a, x_int)
    bc = maxwell_exact(a, x_bc)
    model = PlaneWaveNet(2, args.hidden, init_wavenumber=ap).to(device)

    bc_weight = args.bc_weight_maxwell

    def components():
        pred, lap = model.pred_and_laplacian(x_int)
        residual = lap + kappa2 * pred - f
        L_int = (residual.abs() / (2.0 * ap**2)).square().mean()
        L_bc = (model(x_bc) - bc).abs().square().mean()
        return L_int, L_bc

    def loss_fn():
        L_int, L_bc = components()
        return L_int + bc_weight * L_bc, L_int.item()

    def eval_fn():
        with torch.no_grad():
            return relative_l2(model(x_eval), maxwell_exact(a, x_eval))

    metrics = train_eval(
        model,
        None,
        loss_fn,
        eval_fn,
        seconds=args.seconds,
        lr=args.lr,
        lr_schedule="cosine",
        lr_final=args.lr_final,
        device=device,
        record_history=True,
        history_every_steps=args.history_every_steps,
    )
    L_int, L_bc = components()
    return {
        "problem": "maxwell_a4",
        "variant": "pwnn",
        "seed": args.seed,
        "params": n_params(model),
        "representation": "native_complex_plane_wave",
        "bc_weight": bc_weight,
        "L_int_final": L_int.item(),
        "L_bc_final": L_bc.item(),
        "loss_final": (L_int + bc_weight * L_bc).item(),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=("poly", "chirp", "maxwell", "all"),
                        default="all")
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-int", type=int, default=4096)
    parser.add_argument("--n-bc", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-final", type=float, default=1e-4)
    parser.add_argument("--history-every-steps", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--bc-weight-poly", type=float, default=100.0)
    parser.add_argument("--bc-weight-chirp", type=float, default=100.0)
    parser.add_argument("--bc-weight-maxwell", type=float, default=100.0)
    parser.add_argument("--wire-sigma", type=float, default=10.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runners = {"poly": run_poly, "chirp": run_chirp, "maxwell": run_maxwell}
    selected = tuple(runners) if args.problem == "all" else (args.problem,)
    rows = []
    print(f"device={device} selected={selected} seconds={args.seconds}", flush=True)
    for name in selected:
        print(f"[run] {name}", flush=True)
        row = runners[name](args, device)
        rows.append(row)
        print(
            f"[done] {row['problem']} {row['variant']} steps={row['steps']} "
            f"ms/step={row['ms_per_step']:.2f} L2={row['L2_err']:.6g}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_rows(rows, str(args.out))


if __name__ == "__main__":
    main()
