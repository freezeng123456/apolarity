#!/usr/bin/env python3
"""Small PINN case study with one high-order monomial partial in the residual.

Manufactured PDE on [-1, 1]^d:

    partial^alpha u(x) = f_alpha(x),

with a light data/anchor loss to remove the nullspace.  The goal is not to
solve a hard PDE, but to compare derivative backends inside a training loop.
"""
from __future__ import annotations

import argparse
import csv
import gc
import math
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root (experiments/core_method/<file>)
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn

from apolarity.operators import single_monomial_partial


def parse_alpha(text: str) -> tuple[int, ...]:
    vals = [int(ch) for ch in text] if "," not in text else [int(x) for x in text.split(",") if x]
    if any(v <= 0 for v in vals):
        raise ValueError("alpha is one-based, e.g. 111122")
    return tuple(v - 1 for v in vals)


def build_mlp(d: int, hidden: int, depth: int, dtype: torch.dtype, device: torch.device) -> nn.Module:
    layers: list[nn.Module] = [nn.Linear(d, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers.append(nn.Linear(hidden, 1))
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net.to(device=device, dtype=dtype)


def sample_box(n: int, d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return 2.0 * torch.rand(n, d, device=device, dtype=dtype) - 1.0


def frequencies(d: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(0.7, 1.3, d, device=device, dtype=dtype)


def exact_u(x: torch.Tensor) -> torch.Tensor:
    freq = frequencies(x.shape[1], x.device, x.dtype)
    return torch.sin(freq * x).prod(dim=1, keepdim=True)


def exact_partial(x: torch.Tensor, alpha: tuple[int, ...]) -> torch.Tensor:
    d = x.shape[1]
    freq = frequencies(d, x.device, x.dtype)
    counts = Counter(alpha)
    vals = []
    for j in range(d):
        k = counts.get(j, 0)
        vals.append((freq[j] ** k) * torch.sin(freq[j] * x[:, j:j + 1] + k * math.pi / 2.0))
    out = vals[0]
    for v in vals[1:]:
        out = out * v
    return out


def rel_l2(pred: torch.Tensor, truth: torch.Tensor) -> float:
    return ((pred - truth).norm() / (truth.norm() + 1e-30)).item()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="waring_complex_jet", choices=["direct_autodiff", "polarization_jet", "waring_complex_jet", "auto"])
    parser.add_argument("--alpha", default="111122")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch_res", type=int, default=64)
    parser.add_argument("--batch_data", type=int, default=64)
    parser.add_argument("--n_val", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_weight", type=float, default=1.0)
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/pinn_monomial.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    alpha = parse_alpha(args.alpha)
    if max(alpha) >= args.d:
        raise ValueError(f"d={args.d} is too small for alpha={args.alpha}")

    torch.manual_seed(args.seed)
    model = build_mlp(args.d, args.hidden, args.depth, dtype, device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    x_val = sample_box(args.n_val, args.d, device, dtype)
    u_val = exact_u(x_val)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t_start = time.perf_counter()
    for step in range(1, args.steps + 1):
        x_res = sample_box(args.batch_res, args.d, device, dtype)
        x_data = sample_box(args.batch_data, args.d, device, dtype)
        f_res = exact_partial(x_res, alpha)
        u_data = exact_u(x_data)

        opt.zero_grad(set_to_none=True)
        deriv = single_monomial_partial(model, x_res, alpha, backend=args.method).real
        residual_loss = (deriv - f_res).square().mean()
        data_loss = (model(x_data) - u_data).square().mean()
        loss = residual_loss + args.data_weight * data_loss
        sync(device)
        t0 = time.perf_counter()
        loss.backward()
        opt.step()
        sync(device)
        step_ms = (time.perf_counter() - t0) * 1000.0

        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            with torch.no_grad():
                err = rel_l2(model(x_val), u_val)
            rows.append({
                "step": step,
                "method": args.method,
                "alpha": args.alpha,
                "loss": loss.item(),
                "residual_loss": residual_loss.item(),
                "data_loss": data_loss.item(),
                "val_rel_l2": err,
                "step_ms_backward_only": step_ms,
                "wall_time": time.perf_counter() - t_start,
                "peak_alloc_mb": peak_memory_mb(device),
            })
            print(rows[-1])

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
