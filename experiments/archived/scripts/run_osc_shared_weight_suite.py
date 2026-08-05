#!/usr/bin/env python3
"""Search shared Dirichlet weights and run long Chirp/Maxwell comparisons."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
from apolarity import single_monomial_partial

ROOT = Path(__file__).resolve().parents[3]
COMMON = ROOT / "experiments" / "common"
sys.path.insert(0, str(COMMON))

from osc_common import (  # noqa: E402
    build_model,
    deriv_alpha,
    n_params,
    sample_boundary,
    sample_interior,
    train_eval,
)


GRID_DEFAULT = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
SETTINGS = {"chirp": (1, 2, 3), "maxwell": (2, 4, 6)}


class ScalarOutput(nn.Module):
    def __init__(self, base: nn.Module, index: int):
        super().__init__()
        self.base = base
        self.index = index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)[:, self.index : self.index + 1]


def rel_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float((pred - target).abs().square().mean().sqrt() / (target.abs().square().mean().sqrt() + 1e-30))


def chirp_exact(a: int, x: torch.Tensor) -> torch.Tensor:
    return torch.sin(0.5 * a * math.pi * x.square().sum(dim=1))


def chirp_source(a: int, x: torch.Tensor) -> torch.Tensor:
    ap = a * math.pi
    r2 = x.square().sum(dim=1)
    phi = 0.5 * ap * r2
    return ap**2 * r2 * torch.sin(phi) - 2.0 * ap * torch.cos(phi) + torch.sin(phi)


def maxwell_exact(a: int, x: torch.Tensor) -> torch.Tensor:
    return torch.exp(1j * (a * math.pi * x.sum(dim=1)).to(torch.complex128))


def partial(model: nn.Module, x: torch.Tensor, alpha: tuple[int, ...], method: str):
    backend = "direct_autodiff" if method == "vanilla" else "waring_complex_jet"
    if method == "vanilla":
        # Direct nested autodiff retains an input graph for each derivative.
        # Reusing the same requires-grad tensor across loss calls would make
        # the next optimizer step backpropagate through a freed graph.
        x = x.detach().clone().requires_grad_(True)
    return single_monomial_partial(model, x, alpha, backend=backend)


def build_problem(family: str, a: int, method: str, hidden: int, depth: int, device):
    omega0 = max(10.0, 2.0 * math.pi * a)
    if method == "vanilla":
        out = 2 if family == "maxwell" else 1
        model, dtype = build_model("tanh", 2, hidden, depth, out=out)
    elif method == "sinh":
        model, dtype = build_model("complex_sinh", 2, hidden, depth, omega0=omega0)
    else:
        raise ValueError(method)
    return model.to(device), dtype


def run_one(family: str, a: int, method: str, weight: float, seconds: float,
            seed: int, eval_seed: int, hidden: int, depth: int, n_int: int,
            n_bc: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    x_int = sample_interior(n_int, 2, device=device, generator=gen)
    x_bc = sample_boundary(n_bc, 2, device=device, generator=gen)
    eval_gen = torch.Generator(device=device).manual_seed(eval_seed)
    x_eval = sample_interior(8192, 2, device=device, generator=eval_gen)
    model, dtype = build_problem(family, a, method, hidden, depth, device)
    xi, xb = x_int.to(dtype), x_bc.to(dtype)
    if family == "chirp":
        source = chirp_source(a, x_int).detach()
        bc_target = chirp_exact(a, x_bc)
        scale = 2.0 * (a * math.pi) ** 2

        def components():
            u = model(xi).real.squeeze(1)
            lap = partial(model, xi, (0, 0), method).real.squeeze(1) + partial(model, xi, (1, 1), method).real.squeeze(1)
            lint = ((-lap + u - source) / scale).square().mean()
            lbc = (model(xb).real.squeeze(1) - bc_target).square().mean()
            return lint, lbc

        def evaluate():
            with torch.no_grad():
                return rel_l2(model(x_eval.to(dtype)).real.squeeze(1), chirp_exact(a, x_eval))

    else:
        ap = a * math.pi
        kappa2 = ap**2 * (1.0 + 0.2j)
        source = ((-2.0 * ap**2 + kappa2) * maxwell_exact(a, x_int)).detach()
        bc_target = maxwell_exact(a, x_bc)
        scale = 2.0 * ap**2
        if method == "vanilla":
            re_model, im_model = ScalarOutput(model, 0), ScalarOutput(model, 1)

        def components():
            if method == "vanilla":
                ur, ui = model(xi)[:, 0], model(xi)[:, 1]
                lap = partial(re_model, xi, (0, 0), method).squeeze(1) + partial(re_model, xi, (1, 1), method).squeeze(1)
                lap = lap + 1j * (partial(im_model, xi, (0, 0), method).squeeze(1) + partial(im_model, xi, (1, 1), method).squeeze(1))
                pred = ur + 1j * ui
                bc_pred = model(xb)[:, 0] + 1j * model(xb)[:, 1]
            else:
                pred = model(xi).squeeze(1)
                lap = partial(model, xi, (0, 0), method).squeeze(1) + partial(model, xi, (1, 1), method).squeeze(1)
                bc_pred = model(xb).squeeze(1)
            lint = ((lap + kappa2 * pred - source).abs() / scale).square().mean()
            lbc = (bc_pred - bc_target).abs().square().mean()
            return lint, lbc

        def evaluate():
            with torch.no_grad():
                if method == "vanilla":
                    pred = model(x_eval)[:, 0] + 1j * model(x_eval)[:, 1]
                else:
                    pred = model(x_eval.to(dtype)).squeeze(1)
                return rel_l2(pred, maxwell_exact(a, x_eval))

    def loss_fn():
        lint, lbc = components()
        return lint + weight * lbc, float(lint.item())

    metrics = train_eval(model, dtype, loss_fn, evaluate, seconds=seconds, lr=1e-3,
                         lr_schedule="cosine", lr_final=1e-4, device=device,
                         record_history=True, history_every_steps=20)
    lint, lbc = components()
    return {
        "problem": f"{family}_a{a}", "family": family, "a": a,
        "variant": method, "seed": seed, "eval_seed": eval_seed,
        "bc_weight": weight, "params": n_params(model),
        "L_int_final": float(lint.item()), "L_bc_final": float(lbc.item()),
        "L_bc_weighted_final": float(weight * lbc.item()),
        "loss_final": float((lint + weight * lbc).item()), **metrics,
    }


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_suite(args):
    grid = tuple(float(x) for x in args.grid.split(","))
    settings = [(family, a) for family in args.families for a in SETTINGS[family]]
    selected = {}
    for family, a in settings:
        key = f"{family}_a{a}"
        grid_dir = args.grid_root / key
        ranking_path = grid_dir / "ranking.json"
        if args.phase in ("grid", "all"):
            records = []
            for index, weight in enumerate(grid):
                point = grid_dir / f"point_{index:03d}.json"
                if args.resume and point.exists():
                    rows = json.loads(point.read_text())
                else:
                    rows = [run_one(family, a, method, weight, args.grid_seconds,
                                    args.seed, args.grid_eval_seed, args.hidden,
                                    args.depth, args.n_int, args.n_bc)
                             for method in ("vanilla", "sinh")]
                    write(point, rows)
                by_method = {r["variant"]: r for r in rows}
                gm = math.sqrt(by_method["vanilla"]["L2_err"] * by_method["sinh"]["L2_err"])
                records.append({"bc_weight": weight,
                                "vanilla_L2_err": by_method["vanilla"]["L2_err"],
                                "sinh_L2_err": by_method["sinh"]["L2_err"],
                                "geometric_mean": gm,
                                "max_error": max(by_method[m]["L2_err"] for m in by_method)})
                print(f"[grid] {key} {index+1}/{len(grid)} w={weight:g} gm={gm:.6g}", flush=True)
            records.sort(key=lambda r: (r["geometric_mean"], r["max_error"], r["bc_weight"]))
            write(ranking_path, records)
        weights = json.loads(ranking_path.read_text())[0]["bc_weight"]
        selected[key] = weights
        if args.phase in ("formal", "all"):
            formal_path = args.formal_root / f"{key}.json"
            if not (args.resume and formal_path.exists()):
                rows = [run_one(family, a, method, weights, args.formal_seconds,
                                args.seed, args.formal_eval_seed, args.hidden,
                                args.depth, args.n_int, args.n_bc)
                         for method in ("vanilla", "sinh")]
                write(formal_path, rows)
            print(f"[formal] {key} weight={weights:g}", flush=True)
    write(args.formal_root / "selected_weights.json", selected)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="chirp,maxwell")
    ap.add_argument("--phase", choices=("grid", "formal", "all"), default="all")
    ap.add_argument("--grid", default=",".join(str(x) for x in GRID_DEFAULT))
    ap.add_argument("--grid-seconds", type=float, default=30.0)
    ap.add_argument("--formal-seconds", type=float, default=1200.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-eval-seed", type=int, default=54321)
    ap.add_argument("--formal-eval-seed", type=int, default=12345)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--n-int", type=int, default=4096)
    ap.add_argument("--n-bc", type=int, default=512)
    ap.add_argument("--grid-root", type=Path, required=True)
    ap.add_argument("--formal-root", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    args.families = tuple(x.strip() for x in args.families.split(",") if x.strip())
    if any(x not in SETTINGS for x in args.families):
        ap.error("families must be chirp,maxwell")
    run_suite(args)


if __name__ == "__main__":
    main()
