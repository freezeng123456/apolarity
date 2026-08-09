#!/usr/bin/env python3
"""Complex-Sinh controls with configurable PDE/boundary loss weights."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
COMMON = ROOT / "experiments" / "common"
ARCHIVED_SCRIPTS = ROOT / "experiments" / "archived" / "scripts"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(ARCHIVED_SCRIPTS))

from osc_common import (  # noqa: E402
    build_model,
    deriv_alpha,
    make_complex_field,
    n_params,
    sample_boundary,
    sample_interior,
    train_eval,
    write_rows,
)
from run_specialized_baseline_pilot import (  # noqa: E402
    chirp_exact,
    chirp_source,
    maxwell_exact,
    relative_l2,
)


def common_points(args, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x_int = sample_interior(args.n_int, 2, device=device, generator=generator)
    x_bc = sample_boundary(args.n_bc, 2, device=device, generator=generator)
    eval_generator = torch.Generator(device=device).manual_seed(args.eval_seed)
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


def finalize(metrics, components, bc_weight: float) -> None:
    L_int, L_bc = components()
    metrics.update({
        "L_int_final": float(L_int.item()),
        "L_bc_final": float(L_bc.item()),
        "loss_final": float((L_int + bc_weight * L_bc).item()),
        "bc_weight": float(bc_weight),
    })


def complex_regularizer(model) -> torch.Tensor:
    return 1e-6 * sum(
        parameter.imag.square().mean()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def run_poly(args, device):
    torch.manual_seed(args.seed)
    x_int, x_bc, x_eval = common_points(args, device)
    model, model_dtype = build_model(
        "complex_sinh", 2, args.hidden, args.depth, omega0=2.0 * math.pi
    )
    model = model.to(device)
    xi, xb = x_int.to(model_dtype), x_bc.to(model_dtype)
    S = 2.0 * math.pi**2
    f = S**2 * torch.sin(math.pi * x_int).prod(dim=1)
    bc_weight = args.bc_weight_poly

    def components():
        bilap = (
            deriv_alpha(model, xi, (0, 0, 0, 0))
            + 2.0 * deriv_alpha(model, xi, (0, 0, 1, 1))
            + deriv_alpha(model, xi, (1, 1, 1, 1))
        ).real.squeeze(1)
        L_int = ((bilap - f) / S**2).square().mean()
        u_bc = model(xb).real.squeeze(1)
        lap_bc = (
            deriv_alpha(model, xb, (0, 0))
            + deriv_alpha(model, xb, (1, 1))
        ).real.squeeze(1)
        L_bc = u_bc.square().mean() + (lap_bc / S).square().mean()
        return L_int, L_bc

    def loss_fn():
        L_int, L_bc = components()
        return L_int + bc_weight * L_bc + complex_regularizer(model), L_int.item()

    def eval_fn():
        with torch.no_grad():
            target = torch.sin(math.pi * x_eval).prod(dim=1)
            return relative_l2(model(x_eval.to(model_dtype)).real.squeeze(1), target)

    metrics = train(model, loss_fn, eval_fn, args, device)
    finalize(metrics, components, bc_weight)
    return {"problem": "poly_d2_o4", "variant": args.variant_label,
            "seed": args.seed, "params": n_params(model), **metrics}


def run_chirp(args, device):
    torch.manual_seed(args.seed)
    x_int, x_bc, x_eval = common_points(args, device)
    a = 2
    model, model_dtype = build_model(
        "complex_sinh", 2, args.hidden, args.depth, omega0=2.0 * math.pi * a
    )
    model = model.to(device)
    xi, xb = x_int.to(model_dtype), x_bc.to(model_dtype)
    f = chirp_source(a, x_int)
    bc = chirp_exact(a, x_bc)
    scale = 2.0 * (a * math.pi) ** 2
    bc_weight = args.bc_weight_chirp

    def components():
        u = model(xi).real.squeeze(1)
        lap = (
            deriv_alpha(model, xi, (0, 0))
            + deriv_alpha(model, xi, (1, 1))
        ).real.squeeze(1)
        L_int = ((-lap + u - f) / scale).square().mean()
        L_bc = (model(xb).real.squeeze(1) - bc).square().mean()
        return L_int, L_bc

    def loss_fn():
        L_int, L_bc = components()
        return L_int + bc_weight * L_bc + complex_regularizer(model), L_int.item()

    def eval_fn():
        with torch.no_grad():
            pred = model(x_eval.to(model_dtype)).real.squeeze(1)
            return relative_l2(pred, chirp_exact(a, x_eval))

    metrics = train(model, loss_fn, eval_fn, args, device)
    finalize(metrics, components, bc_weight)
    return {"problem": "chirp_a2", "variant": args.variant_label,
            "seed": args.seed, "params": n_params(model), **metrics}


def run_maxwell(args, device):
    torch.manual_seed(args.seed)
    x_int, x_bc, x_eval = common_points(args, device)
    a = 4
    ap = a * math.pi
    field, _ = make_complex_field(
        "complex_sinh", 2, args.hidden, args.depth, device,
        omega0=2.0 * math.pi * a, sigma=math.pi * a,
    )
    model = field.module
    kappa2 = ap**2 * (1.0 + 0.2j)
    f = (-2.0 * ap**2 + kappa2) * maxwell_exact(a, x_int)
    bc = maxwell_exact(a, x_bc)
    scale = 2.0 * ap**2
    bc_weight = args.bc_weight_maxwell

    def components():
        u = field.pred(x_int)
        lap = field.deriv(x_int, (0, 0)) + field.deriv(x_int, (1, 1))
        L_int = ((lap + kappa2 * u - f).abs() / scale).square().mean()
        L_bc = (field.pred(x_bc) - bc).abs().square().mean()
        return L_int, L_bc

    def loss_fn():
        L_int, L_bc = components()
        return L_int + bc_weight * L_bc + complex_regularizer(model), L_int.item()

    def eval_fn():
        with torch.no_grad():
            return relative_l2(field.pred(x_eval), maxwell_exact(a, x_eval))

    metrics = train(model, loss_fn, eval_fn, args, device)
    finalize(metrics, components, bc_weight)
    return {"problem": "maxwell_a4", "variant": args.variant_label,
            "seed": args.seed, "params": n_params(model),
            "representation": "native_complex", **metrics}


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
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--bc-weight-poly", type=float, default=100.0)
    parser.add_argument("--bc-weight-chirp", type=float, default=100.0)
    parser.add_argument("--bc-weight-maxwell", type=float, default=100.0)
    parser.add_argument("--variant-label", default="complex_sinh_weight_control")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runners = {"poly": run_poly, "chirp": run_chirp, "maxwell": run_maxwell}
    selected = tuple(runners) if args.problem == "all" else (args.problem,)
    rows = []
    print(f"device={device} selected={selected} seconds={args.seconds}", flush=True)
    for name in selected:
        row = runners[name](args, device)
        rows.append(row)
        print(f"[done] {row['problem']} steps={row['steps']} "
              f"ms/step={row['ms_per_step']:.2f} L2={row['L2_err']:.6g}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_rows(rows, str(args.out))


if __name__ == "__main__":
    main()
