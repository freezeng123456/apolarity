#!/usr/bin/env python3
"""Strict vanilla-PINN pilots: tanh MLPs with nested coordinate autodiff."""

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
sys.path.insert(0, str(ROOT / "scripts"))

from osc_common import (  # noqa: E402
    n_params,
    sample_boundary,
    sample_interior,
    train_eval,
    write_rows,
)
from run_specialized_baseline_pilot import (  # noqa: E402
    TanhMLP,
    chirp_exact,
    chirp_source,
    direct_laplacian,
    maxwell_exact,
    relative_l2,
)


def common_points(args, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x_int = sample_interior(args.n_int, 2, device=device, generator=generator)
    x_bc = sample_boundary(args.n_bc, 2, device=device, generator=generator)
    eval_generator = torch.Generator(device=device).manual_seed(12345)
    x_eval = sample_interior(8192, 2, device=device, generator=eval_generator)
    return x_int, x_bc, x_eval


def train(model, loss_fn, eval_fn, args, device):
    return train_eval(
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


def run_poly(args, device):
    torch.manual_seed(args.seed)
    x_int, x_bc, x_eval = common_points(args, device)
    x_int.requires_grad_(True)
    x_bc.requires_grad_(True)
    model = TanhMLP(2, args.hidden, args.depth, 1).to(device)
    S = 2.0 * math.pi**2
    f = (S**2 * torch.sin(math.pi * x_int).prod(dim=1)).detach()

    def loss_fn():
        u = model(x_int).squeeze(1)
        lap = direct_laplacian(u, x_int)
        bilap = direct_laplacian(lap, x_int)
        L_int = ((bilap - f) / S**2).square().mean()
        u_bc = model(x_bc).squeeze(1)
        lap_bc = direct_laplacian(u_bc, x_bc)
        L_bc = u_bc.square().mean() + (lap_bc / S).square().mean()
        return L_int + 100.0 * L_bc, L_int.item()

    def eval_fn():
        with torch.no_grad():
            target = torch.sin(math.pi * x_eval).prod(dim=1)
            return relative_l2(model(x_eval).squeeze(1), target)

    metrics = train(model, loss_fn, eval_fn, args, device)
    return {"problem": "poly_d2_o4", "variant": "vanilla_tanh_direct_ad",
            "seed": args.seed, "params": n_params(model), **metrics}


def run_chirp(args, device):
    torch.manual_seed(args.seed)
    x_int, x_bc, x_eval = common_points(args, device)
    x_int.requires_grad_(True)
    model = TanhMLP(2, args.hidden, args.depth, 1).to(device)
    a = 2
    f = chirp_source(a, x_int).detach()
    bc = chirp_exact(a, x_bc)
    scale = 2.0 * (a * math.pi) ** 2

    def loss_fn():
        u = model(x_int).squeeze(1)
        residual = -direct_laplacian(u, x_int) + u - f
        L_int = (residual / scale).square().mean()
        L_bc = (model(x_bc).squeeze(1) - bc).square().mean()
        return L_int + 100.0 * L_bc, L_int.item()

    def eval_fn():
        with torch.no_grad():
            return relative_l2(model(x_eval).squeeze(1), chirp_exact(a, x_eval))

    metrics = train(model, loss_fn, eval_fn, args, device)
    return {"problem": "chirp_a2", "variant": "vanilla_tanh_direct_ad",
            "seed": args.seed, "params": n_params(model), **metrics}


def run_maxwell(args, device):
    torch.manual_seed(args.seed)
    x_int, x_bc, x_eval = common_points(args, device)
    x_int.requires_grad_(True)
    re = TanhMLP(2, args.hidden, args.depth, 1).to(device)
    im = TanhMLP(2, args.hidden, args.depth, 1).to(device)
    model = nn.ModuleList([re, im])
    a = 4
    ap = a * math.pi
    kappa2 = ap**2 * (1.0 + 0.2j)
    multiplier = -2.0 * ap**2 + kappa2
    f = (multiplier * maxwell_exact(a, x_int)).detach()
    bc = maxwell_exact(a, x_bc)

    def pred(x):
        return re(x).squeeze(1) + 1j * im(x).squeeze(1)

    def loss_fn():
        ur = re(x_int).squeeze(1)
        ui = im(x_int).squeeze(1)
        u = ur + 1j * ui
        lap = direct_laplacian(ur, x_int) + 1j * direct_laplacian(ui, x_int)
        residual = lap + kappa2 * u - f
        L_int = (residual.abs() / (2.0 * ap**2)).square().mean()
        L_bc = (pred(x_bc) - bc).abs().square().mean()
        return L_int + 100.0 * L_bc, L_int.item()

    def eval_fn():
        with torch.no_grad():
            return relative_l2(pred(x_eval), maxwell_exact(a, x_eval))

    metrics = train(model, loss_fn, eval_fn, args, device)
    return {"problem": "maxwell_a4", "variant": "vanilla_tanh_direct_ad",
            "seed": args.seed, "params": n_params(model),
            "representation": "split_real", **metrics}


def main():
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
        print(f"[done] {row['problem']} steps={row['steps']} "
              f"ms/step={row['ms_per_step']:.2f} L2={row['L2_err']:.6g}",
              flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_rows(rows, str(args.out))


if __name__ == "__main__":
    main()
