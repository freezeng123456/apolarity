#!/usr/bin/env python3
"""Paired fixed-time training audit for the 4D (4,2) sixth-order PDE."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import signal
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apolarity import single_monomial_partial


ALPHA = (0, 0, 0, 0, 1, 1)
STOP_REQUESTED = False


def request_stop(_signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


class SinhActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sinh(x)


def build_model(hidden: int, depth: int, device: torch.device, seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    fan_in = 4
    for _ in range(depth):
        layers.extend([nn.Linear(fan_in, hidden), SinhActivation()])
        fan_in = hidden
    layers.append(nn.Linear(fan_in, 1))
    model = nn.Sequential(*layers).to(device=device, dtype=torch.complex128)
    with torch.no_grad():
        for layer in model:
            if not isinstance(layer, nn.Linear):
                continue
            real = torch.empty_like(layer.weight.real)
            imag = torch.empty_like(layer.weight.real)
            nn.init.xavier_uniform_(real)
            nn.init.xavier_uniform_(imag)
            layer.weight.copy_((real + 1j * imag) / math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
    return model


def u_exact(x: Tensor) -> Tensor:
    xr = x.real if x.dtype.is_complex else x
    x1, x2, x3, x4 = xr.unbind(dim=-1)
    return torch.sinh(x1) * torch.cos(x2) * torch.exp(-(x3.square() + x4.square()) / 4.0)


def sample_interior(n: int, generator: torch.Generator, device: torch.device) -> Tensor:
    x = torch.empty(n, 4, dtype=torch.float64, device=device)
    x.uniform_(-1.0, 1.0, generator=generator)
    return x.to(torch.complex128)


def sample_boundary(n: int, generator: torch.Generator, device: torch.device) -> Tensor:
    x = torch.empty(n, 4, dtype=torch.float64, device=device)
    x.uniform_(-1.0, 1.0, generator=generator)
    face = torch.randint(0, 4, (n,), generator=generator, device=device)
    side = torch.randint(0, 2, (n,), generator=generator, device=device, dtype=torch.int64)
    x[torch.arange(n, device=device), face] = side.to(torch.float64).mul_(2.0).sub_(1.0)
    return x.to(torch.complex128)


def sobol_interior(n: int, seed: int) -> Tensor:
    engine = torch.quasirandom.SobolEngine(4, scramble=True, seed=seed)
    return 2.0 * engine.draw(n).to(torch.float64) - 1.0


def sobol_boundary(n: int, seed: int) -> Tensor:
    engine = torch.quasirandom.SobolEngine(6, scramble=True, seed=seed)
    raw = engine.draw(n).to(torch.float64)
    x = 2.0 * raw[:, :4] - 1.0
    face = torch.clamp((4.0 * raw[:, 4]).floor().to(torch.int64), max=3)
    side = torch.where(raw[:, 5] < 0.5, -torch.ones(n), torch.ones(n)).to(torch.float64)
    x[torch.arange(n), face] = side
    return x


def relative_errors(model: nn.Module, points: Tensor, device: torch.device,
                    chunk: int = 8192) -> tuple[float, float, float]:
    sum_sq = 0.0
    target_sq = 0.0
    max_abs = 0.0
    imag_sq = 0.0
    total = 0
    with torch.no_grad():
        for start in range(0, points.shape[0], chunk):
            xr = points[start:start + chunk].to(device=device)
            pred_complex = model(xr.to(torch.complex128)).squeeze(1)
            pred = pred_complex.real
            target = u_exact(xr)
            diff = pred - target
            sum_sq += float(diff.square().sum().item())
            target_sq += float(target.square().sum().item())
            max_abs = max(max_abs, float(diff.abs().max().item()))
            imag_sq += float(pred_complex.imag.square().sum().item())
            total += xr.shape[0]
    return (
        math.sqrt(sum_sq / (target_sq + 1.0e-300)),
        max_abs,
        math.sqrt(imag_sq / max(total, 1)),
    )


def residual_rms(model: nn.Module, points: Tensor, backend: str,
                 device: torch.device, chunk: int = 1024) -> float:
    total_sq = 0.0
    total = 0
    for start in range(0, points.shape[0], chunk):
        xr = points[start:start + chunk].to(device=device)
        x = xr.to(torch.complex128)
        deriv = single_monomial_partial(model, x, ALPHA, backend=backend,
                                        create_graph=False).real.squeeze(1)
        residual = deriv + u_exact(xr)
        total_sq += float(residual.detach().square().sum().item())
        total += xr.shape[0]
        for parameter in model.parameters():
            parameter.grad = None
    return math.sqrt(total_sq / max(total, 1))


def boundary_rms(model: nn.Module, points: Tensor, device: torch.device,
                 chunk: int = 8192) -> float:
    total_sq = 0.0
    total = 0
    with torch.no_grad():
        for start in range(0, points.shape[0], chunk):
            xr = points[start:start + chunk].to(device=device)
            pred = model(xr.to(torch.complex128)).real.squeeze(1)
            diff = pred - u_exact(xr)
            total_sq += float(diff.square().sum().item())
            total += xr.shape[0]
    return math.sqrt(total_sq / max(total, 1))


def state_digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(path)


def learning_rate(initial: float, final: float, elapsed: float, budget: float) -> float:
    fraction = min(max(elapsed / max(budget, 1.0e-12), 0.0), 1.0)
    return final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * fraction))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("direct_autodiff", "polarization_jet", "waring_complex_jet"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=1200.0)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="optional fixed-step stop; zero disables it")
    parser.add_argument("--lr-basis", choices=("time", "steps"), default="time")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-int", type=int, default=128)
    parser.add_argument("--n-bc", type=int, default=64)
    parser.add_argument("--bc-weight", type=float, default=100.0)
    parser.add_argument("--im-weight", type=float, default=1.0e-6)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--lr-final", type=float, default=1.0e-4)
    parser.add_argument("--probe-seconds", type=float, default=10.0)
    parser.add_argument("--eval-seed", type=int, default=12345)
    parser.add_argument("--probe-eval-n", type=int, default=8192)
    parser.add_argument("--final-eval-n", type=int, default=2 ** 16)
    parser.add_argument("--boundary-eval-n", type=int, default=2 ** 14)
    parser.add_argument("--residual-chunk", type=int, default=1024)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    model = build_model(args.hidden, args.depth, device, args.seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator(device=device).manual_seed(700000 + args.seed)
    probe_points = sobol_interior(args.probe_eval_n, args.eval_seed).to(device)
    final_points = sobol_interior(args.final_eval_n, args.eval_seed)
    boundary_points = sobol_boundary(args.boundary_eval_n, args.eval_seed + 1)
    snapshots: dict[int, dict[str, Tensor]] = {}
    snapshot_steps = {0, 1, 10, 100, 500}
    snapshots[0] = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "git_sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "git_dirty": bool(os.popen(f"git -C {ROOT} status --porcelain").read().strip()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "backend": args.backend,
        "seed": args.seed,
        "alpha": list(ALPHA),
        "hidden": args.hidden,
        "depth": args.depth,
        "n_int": args.n_int,
        "n_bc": args.n_bc,
        "bc_weight": args.bc_weight,
        "im_weight": args.im_weight,
        "lr": args.lr,
        "lr_final": args.lr_final,
        "budget_seconds": args.seconds,
        "max_steps": args.max_steps,
        "lr_basis": args.lr_basis,
        "batch_stream_seed": 700000 + args.seed,
        "eval_seed": args.eval_seed,
        "probe_eval_n": args.probe_eval_n,
        "final_eval_n": args.final_eval_n,
        "boundary_eval_n": args.boundary_eval_n,
    }
    history: list[dict[str, Any]] = []
    status: dict[str, Any] = {
        "manifest": manifest,
        "status": "running",
        "history": history,
        "steps": 0,
    }
    atomic_json(args.out, status)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    training_seconds = 0.0
    step = 0
    next_probe = 0.0

    try:
        while not STOP_REQUESTED:
            wall_elapsed = time.perf_counter() - wall_start
            if wall_elapsed >= args.seconds or (args.max_steps > 0 and step >= args.max_steps):
                break
            if args.lr_basis == "steps":
                if args.max_steps <= 0:
                    raise ValueError("--lr-basis steps requires --max-steps > 0")
                lr = learning_rate(args.lr, args.lr_final, step, args.max_steps)
            else:
                lr = learning_rate(args.lr, args.lr_final, wall_elapsed, args.seconds)
            for group in optimizer.param_groups:
                group["lr"] = lr
            x_int = sample_interior(args.n_int, generator, device)
            x_bc = sample_boundary(args.n_bc, generator, device)
            torch.cuda.synchronize(device)
            train_start = time.perf_counter()
            deriv = single_monomial_partial(
                model, x_int, ALPHA, backend=args.backend, create_graph=True
            ).real.squeeze(1)
            residual = deriv + u_exact(x_int)
            l_int = residual.square().mean()
            pred_bc = model(x_bc).real.squeeze(1)
            l_bc = (pred_bc - u_exact(x_bc)).square().mean()
            l_im = sum(parameter.imag.square().mean() for parameter in model.parameters())
            loss = l_int + args.bc_weight * l_bc + args.im_weight * l_im
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            training_seconds += time.perf_counter() - train_start
            step += 1

            if step in snapshot_steps:
                snapshots[step] = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

            wall_elapsed = time.perf_counter() - wall_start
            if step <= 10 or step % 20 == 0 or wall_elapsed >= next_probe:
                record: dict[str, Any] = {
                    "step": step,
                    "wall_seconds": wall_elapsed,
                    "training_seconds": training_seconds,
                    "lr": lr,
                    "loss": float(loss.detach().item()),
                    "L_int": float(l_int.detach().item()),
                    "L_bc": float(l_bc.detach().item()),
                    "L_im": float(l_im.detach().item()),
                }
                if wall_elapsed >= next_probe:
                    probe_l2, probe_linf, probe_imag = relative_errors(
                        model, probe_points, device
                    )
                    record.update({
                        "probe_L2": probe_l2,
                        "probe_Linf": probe_linf,
                        "probe_imag_rms": probe_imag,
                    })
                    next_probe = wall_elapsed + args.probe_seconds
                history.append(record)
                if step <= 10 or wall_elapsed >= next_probe - args.probe_seconds:
                    status.update({
                        "steps": step,
                        "wall_seconds": wall_elapsed,
                        "training_seconds": training_seconds,
                        "state_digest": state_digest(model),
                    })
                    atomic_json(args.out, status)
                    print(json.dumps(record, sort_keys=True), flush=True)
    except Exception as exc:
        status.update({
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "steps": step,
            "wall_seconds": time.perf_counter() - wall_start,
            "training_seconds": training_seconds,
        })
        atomic_json(args.out, status)
        torch.save({"snapshots": snapshots, "final": model.state_dict()}, args.checkpoint)
        raise

    wall_seconds = time.perf_counter() - wall_start
    final_l2, final_linf, final_imag = relative_errors(model, final_points, device)
    pde_rms = residual_rms(
        model, final_points, args.backend, device, chunk=args.residual_chunk
    )
    bc_rms = boundary_rms(model, boundary_points, device)
    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
    status.update({
        "status": "stopped" if STOP_REQUESTED else "complete",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "steps": step,
        "wall_seconds": wall_seconds,
        "training_seconds": training_seconds,
        "ms_per_train_step": 1000.0 * training_seconds / max(step, 1),
        "peak_alloc_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
        "final_L2": final_l2,
        "final_Linf": final_linf,
        "final_imag_rms": final_imag,
        "final_pde_rms": pde_rms,
        "final_boundary_rms": bc_rms,
        "state_digest": state_digest(model),
    })
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"snapshots": snapshots, "final": model.state_dict(), "status": status}, args.checkpoint)
    atomic_json(args.out, status)
    print(json.dumps({key: status[key] for key in (
        "status", "steps", "wall_seconds", "ms_per_train_step", "final_L2",
        "final_pde_rms", "final_boundary_rms", "peak_alloc_mb"
    )}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
