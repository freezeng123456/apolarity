#!/usr/bin/env python3
"""Polyharmonic Vanilla/Sinh comparison with one shared boundary-weight vector."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(ROOT / "scripts"))

from osc_common import (  # noqa: E402
    build_model,
    deriv_alpha,
    laplacian_power_terms,
    n_params,
    sample_boundary,
    sample_interior,
    train_eval,
    write_rows,
)
from run_specialized_baseline_pilot import (  # noqa: E402
    TanhMLP,
    direct_laplacian,
    relative_l2,
)


def parse_bc_weights(text: str, order: int) -> tuple[float, ...]:
    weights = tuple(float(value) for value in text.split(",") if value.strip())
    expected = order // 2
    if len(weights) != expected:
        raise ValueError(
            f"order {order} requires {expected} boundary weights "
            f"for Delta^j u, j=0..{expected - 1}; got {len(weights)}"
        )
    if any(weight < 0.0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("boundary weights must be finite and non-negative")
    return weights


def exact_solution(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(math.pi * x).prod(dim=1)


def common_points(args, device: torch.device):
    train_generator = torch.Generator(device=device).manual_seed(args.seed)
    x_int = sample_interior(
        args.n_int, 2, device=device, generator=train_generator
    )
    x_bc = sample_boundary(
        args.n_bc, 2, device=device, generator=train_generator
    )
    eval_generator = torch.Generator(device=device).manual_seed(args.eval_seed)
    x_eval = sample_interior(
        args.n_eval, 2, device=device, generator=eval_generator
    )
    return x_int, x_bc, x_eval


def repeated_laplacians(
    value: torch.Tensor, x: torch.Tensor, maximum_power: int
) -> list[torch.Tensor]:
    powers = [value]
    for _ in range(maximum_power):
        powers.append(direct_laplacian(powers[-1], x))
    return powers


def jet_laplacian_power(
    model: torch.nn.Module, x: torch.Tensor, power: int
) -> torch.Tensor:
    if power == 0:
        return model(x).real.squeeze(1)
    value = None
    for coefficient, alpha in laplacian_power_terms(2, power):
        term = deriv_alpha(model, x, alpha).real.squeeze(1)
        value = coefficient * term if value is None else value + coefficient * term
    assert value is not None
    return value


def complex_regularizer(model: torch.nn.Module) -> torch.Tensor:
    return 1e-6 * sum(
        parameter.imag.square().mean()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def train_one(args, method: str, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    order = args.order
    m = order // 2
    weights = parse_bc_weights(args.bc_weights, order)
    S = 2.0 * math.pi**2
    x_int, x_bc, x_eval = common_points(args, device)
    source = ((-S) ** m * exact_solution(x_int)).detach()
    bc_targets = [
        ((-S) ** j * exact_solution(x_bc)).detach() for j in range(m)
    ]

    if method == "vanilla":
        x_int.requires_grad_(True)
        x_bc.requires_grad_(True)
        model = TanhMLP(2, args.hidden, args.depth, 1).to(device)

        def components():
            interior_powers = repeated_laplacians(
                model(x_int).squeeze(1), x_int, m
            )
            boundary_powers = repeated_laplacians(
                model(x_bc).squeeze(1), x_bc, m - 1
            )
            L_int = (
                (interior_powers[m] - source) / (S**m)
            ).square().mean()
            L_bc = [
                ((boundary_powers[j] - bc_targets[j]) / (S**j))
                .square()
                .mean()
                for j in range(m)
            ]
            return L_int, L_bc

        def regularizer():
            return torch.zeros((), dtype=torch.float64, device=device)

        def eval_fn():
            with torch.no_grad():
                return relative_l2(
                    model(x_eval).squeeze(1), exact_solution(x_eval)
                )

        representation = "real"
        variant = "vanilla_tanh_direct_ad"
    else:
        model, model_dtype = build_model(
            "complex_sinh",
            2,
            args.hidden,
            args.depth,
            omega0=2.0 * math.pi,
        )
        model = model.to(device)
        xi = x_int.to(model_dtype)
        xb = x_bc.to(model_dtype)

        def components():
            interior = jet_laplacian_power(model, xi, m)
            L_int = ((interior - source) / (S**m)).square().mean()
            L_bc = [
                ((jet_laplacian_power(model, xb, j) - bc_targets[j]) / (S**j))
                .square()
                .mean()
                for j in range(m)
            ]
            return L_int, L_bc

        def regularizer():
            return complex_regularizer(model)

        def eval_fn():
            with torch.no_grad():
                prediction = model(x_eval.to(model_dtype)).real.squeeze(1)
                return relative_l2(prediction, exact_solution(x_eval))

        representation = "native_complex"
        variant = "complex_sinh"

    def loss_fn():
        L_int, L_bc = components()
        weighted_bc = sum(weight * term for weight, term in zip(weights, L_bc))
        return L_int + weighted_bc + regularizer(), L_int.item()

    metrics = train_eval(
        model,
        None,
        loss_fn,
        eval_fn,
        seconds=args.seconds,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        lr_final=args.lr_final,
        device=device,
        record_history=True,
        history_every_steps=args.history_every_steps,
    )
    L_int, L_bc = components()
    weighted_bc = sum(weight * term for weight, term in zip(weights, L_bc))
    row = {
        "problem": f"poly_d2_o{order}",
        "variant": variant,
        "representation": representation,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "params": n_params(model),
        "bc_weights": list(weights),
        "L_int_final": L_int.item(),
        "L_bc_final": sum(term.item() for term in L_bc),
        "L_bc_weighted_final": weighted_bc.item(),
        "loss_final": (L_int + weighted_bc).item(),
        **metrics,
    }
    for j, (weight, term) in enumerate(zip(weights, L_bc)):
        row[f"bc_weight_j{j}"] = weight
        row[f"L_bc_j{j}_final"] = term.item()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, choices=(2, 4, 6), required=True)
    parser.add_argument(
        "--method", choices=("vanilla", "sinh", "both"), default="both"
    )
    parser.add_argument(
        "--bc-weights",
        required=True,
        help="comma-separated weights for normalized Delta^j boundary losses",
    )
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-seed", type=int, default=54321)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-int", type=int, default=4096)
    parser.add_argument("--n-bc", type=int, default=512)
    parser.add_argument("--n-eval", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-final", type=float, default=1e-4)
    parser.add_argument(
        "--lr-schedule", choices=("constant", "cosine"), default="cosine"
    )
    parser.add_argument("--history-every-steps", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    parse_bc_weights(args.bc_weights, args.order)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    methods = ("vanilla", "sinh") if args.method == "both" else (args.method,)
    print(
        f"device={device} order={args.order} methods={methods} "
        f"bc_weights={args.bc_weights} seconds={args.seconds}",
        flush=True,
    )
    rows = []
    for method in methods:
        print(f"[run] {method}", flush=True)
        row = train_one(args, method, device)
        rows.append(row)
        print(
            f"[done] {method} steps={row['steps']} "
            f"ms/step={row['ms_per_step']:.2f} L2={row['L2_err']:.6g}",
            flush=True,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_rows(rows, str(args.out))


if __name__ == "__main__":
    main()
