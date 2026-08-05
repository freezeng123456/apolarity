#!/usr/bin/env python3
"""Independent-seed stress tests for Chirp a=2 and Maxwell a=4 baselines."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
SRC = ROOT / "src"
for path in (COMMON, SRC, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from apolarity import single_monomial_partial
from osc_common import (
    build_model,
    make_complex_field,
    n_params,
    sample_boundary,
    sample_interior,
    train_eval,
    write_rows,
)
from run_specialized_baseline_pilot import PlaneWaveNet


def sobol_interior(n: int, seed: int, device: torch.device) -> Tensor:
    engine = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed)
    return (2.0 * engine.draw(n).to(torch.float64) - 1.0).to(device)


def chirp_exact(x: Tensor) -> Tensor:
    a = 2
    return torch.sin(0.5 * a * math.pi * x.square().sum(dim=1))


def chirp_source(x: Tensor) -> Tensor:
    a = 2
    ap = a * math.pi
    r2 = x.square().sum(dim=1)
    phase = 0.5 * ap * r2
    return ap ** 2 * r2 * torch.sin(phase) - 2.0 * ap * torch.cos(phase) + torch.sin(phase)


def maxwell_exact(x: Tensor) -> Tensor:
    a = 4
    return torch.exp(1j * (a * math.pi * x.sum(dim=1)).to(torch.complex128))


def direct_pred_laplacian(model: nn.Module, x: Tensor) -> tuple[Tensor, Tensor]:
    xr = x.detach().clone().requires_grad_(True)
    pred = model(xr).squeeze(1)
    grad = torch.autograd.grad(pred.sum(), xr, create_graph=True)[0]
    lap = torch.zeros_like(pred)
    for coordinate in range(xr.shape[1]):
        second = torch.autograd.grad(
            grad[:, coordinate].sum(), xr, create_graph=True, retain_graph=True
        )[0][:, coordinate]
        lap = lap + second
    return pred, lap


def relative_metrics(predict, points: Tensor, target_fn, chunk: int = 8192):
    sum_sq = 0.0
    target_sq = 0.0
    max_abs = 0.0
    total = 0
    with torch.no_grad():
        for start in range(0, points.shape[0], chunk):
            x = points[start:start + chunk]
            pred = predict(x)
            target = target_fn(x)
            diff = pred - target
            sum_sq += float(diff.abs().square().sum().item())
            target_sq += float(target.abs().square().sum().item())
            max_abs = max(max_abs, float(diff.abs().max().item()))
            total += x.shape[0]
    return math.sqrt(sum_sq / (target_sq + 1.0e-300)), max_abs, total


def complex_regularizer(model: nn.Module) -> Tensor:
    return 1.0e-6 * sum(parameter.imag.square().mean() for parameter in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=("chirp_a2", "maxwell_a4"), required=True)
    parser.add_argument("--method", choices=("vanilla", "complex_sinh", "pwnn"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-int", type=int, default=4096)
    parser.add_argument("--n-bc", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lr-final", type=float, default=1.0e-4)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--eval-n", type=int, default=2 ** 16)
    parser.add_argument("--history-eval-n", type=int, default=4096)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    valid = {
        "chirp_a2": {"vanilla", "complex_sinh"},
        "maxwell_a4": {"pwnn", "complex_sinh"},
    }
    if args.method not in valid[args.problem]:
        parser.error(f"{args.method} is not valid for {args.problem}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x_int = sample_interior(args.n_int, 2, device=device, generator=generator)
    x_bc = sample_boundary(args.n_bc, 2, device=device, generator=generator)
    eval_points = sobol_interior(args.eval_n, args.eval_seed, device)
    history_points = eval_points[:args.history_eval_n]

    if args.problem == "chirp_a2":
        bc_weight = 0.1
        scale = 2.0 * (2.0 * math.pi) ** 2
        source = chirp_source(x_int).detach()
        bc_target = chirp_exact(x_bc)
        if args.method == "vanilla":
            model, model_dtype = build_model("tanh", 2, args.hidden, args.depth)
            model = model.to(device)

            def components():
                pred, lap = direct_pred_laplacian(model, x_int)
                lint = ((-lap + pred - source) / scale).square().mean()
                lbc = (model(x_bc).squeeze(1) - bc_target).square().mean()
                return lint, lbc

            def predict(x):
                return model(x).squeeze(1)

            history_every = 250
        else:
            model, model_dtype = build_model(
                "complex_sinh", 2, args.hidden, args.depth, omega0=4.0 * math.pi
            )
            model = model.to(device)
            xi = x_int.to(model_dtype)
            xb = x_bc.to(model_dtype)

            def components():
                pred = model(xi).real.squeeze(1)
                lap = (
                    single_monomial_partial(model, xi, (0, 0), backend="waring_complex_jet")
                    + single_monomial_partial(model, xi, (1, 1), backend="waring_complex_jet")
                ).real.squeeze(1)
                lint = ((-lap + pred - source) / scale).square().mean()
                lbc = (model(xb).real.squeeze(1) - bc_target).square().mean()
                return lint, lbc

            def predict(x):
                return model(x.to(model_dtype)).real.squeeze(1)

            history_every = 100

        def target(x):
            return chirp_exact(x)

    else:
        bc_weight = 0.03
        a = 4
        ap = a * math.pi
        kappa2 = ap ** 2 * (1.0 + 0.2j)
        source = ((-2.0 * ap ** 2 + kappa2) * maxwell_exact(x_int)).detach()
        bc_target = maxwell_exact(x_bc)
        scale = 2.0 * ap ** 2
        if args.method == "pwnn":
            model = PlaneWaveNet(2, args.hidden, init_wavenumber=ap).to(device)
            model_dtype = None

            def components():
                pred, lap = model.pred_and_laplacian(x_int)
                lint = ((lap + kappa2 * pred - source).abs() / scale).square().mean()
                lbc = (model(x_bc) - bc_target).abs().square().mean()
                return lint, lbc

            def predict(x):
                return model(x)

            history_every = 2000
        else:
            field, _ = make_complex_field(
                "complex_sinh", 2, args.hidden, args.depth, device,
                omega0=8.0 * math.pi, sigma=4.0 * math.pi,
            )
            model = field.module
            model_dtype = torch.complex128

            def components():
                pred = field.pred(x_int)
                lap = field.deriv(x_int, (0, 0)) + field.deriv(x_int, (1, 1))
                lint = ((lap + kappa2 * pred - source).abs() / scale).square().mean()
                lbc = (field.pred(x_bc) - bc_target).abs().square().mean()
                return lint, lbc

            def predict(x):
                return field.pred(x)

            history_every = 100

        def target(x):
            return maxwell_exact(x)

    def loss_fn():
        lint, lbc = components()
        loss = lint + bc_weight * lbc
        if args.method == "complex_sinh":
            loss = loss + complex_regularizer(model)
        return loss, float(lint.item())

    def full_eval():
        return relative_metrics(predict, eval_points, target)[0]

    def history_eval():
        return relative_metrics(predict, history_points, target)[0]

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    metrics = train_eval(
        model, model_dtype, loss_fn, full_eval,
        seconds=args.seconds, lr=args.lr, lr_schedule="cosine",
        lr_final=args.lr_final, device=device, record_history=True,
        history_every_steps=history_every, history_eval_fn=history_eval,
    )
    lint, lbc = components()
    final_l2, final_linf, _ = relative_metrics(predict, eval_points, target)
    row = {
        "problem": args.problem,
        "variant": args.method,
        "seed": args.seed,
        "final_seed_role": "independent_after_seed0_tuning",
        "eval_seed": args.eval_seed,
        "eval_design": "scrambled_sobol",
        "eval_n": args.eval_n,
        "params_real_dof": n_params(model),
        "hidden": args.hidden,
        "depth": args.depth if args.method != "pwnn" else 1,
        "n_int": args.n_int,
        "n_bc": args.n_bc,
        "collocation": "fixed_paired_by_seed",
        "budget_seconds": args.seconds,
        "lr": args.lr,
        "lr_final": args.lr_final,
        "lr_schedule": "cosine_by_training_time",
        "bc_weight": bc_weight,
        "L_int_final": float(lint.item()),
        "L_bc_final": float(lbc.item()),
        "loss_final_without_im_regularizer": float((lint + bc_weight * lbc).item()),
        "L2_err": final_l2,
        "Linf_abs": final_linf,
        "started_at": started_at,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "source_parent_sha": "e1bbf7e36892bd4efa317e95b250c540812fe00e",
        "runner_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
        **metrics,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_rows([row], str(args.out))
    manifest_path = args.out.with_name(args.out.stem + "_manifest.json")
    manifest_path.write_text(json.dumps({
        "argv": sys.argv,
        "row_file": str(args.out),
        "source_parent_sha": row["source_parent_sha"],
        "runner_sha256": row["runner_sha256"],
    }, indent=2))
    print(json.dumps({
        "problem": args.problem,
        "method": args.method,
        "seed": args.seed,
        "steps": metrics.get("steps"),
        "ms_per_step": metrics.get("ms_per_step"),
        "L2_err": final_l2,
        "Linf_abs": final_linf,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
