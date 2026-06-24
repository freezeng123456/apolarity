#!/usr/bin/env python3
"""Benchmark exact single-monomial derivative backends.

This benchmark intentionally evaluates one expanded multi-index at a time.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn
from torch import Tensor

from apolarity.operators import direct_monomial_autodiff, single_monomial_partial
from apolarity.polarization import polarization_directions as _polarization_directions
from apolarity.waring import monomial_waring_directions


def parse_alpha_token(token: str) -> tuple[int, ...]:
    token = token.strip()
    vals = [int(ch) for ch in token] if "," not in token else [int(x) for x in token.split(",") if x]
    if any(v <= 0 for v in vals):
        raise ValueError(f"alpha indices are 1-based positive integers: {token}")
    return tuple(v - 1 for v in vals)


def parse_alpha_list(text: str) -> list[tuple[int, ...]]:
    return [parse_alpha_token(tok) for tok in text.replace(" ", "").split(";") if tok]


def alpha_label(alpha: tuple[int, ...]) -> str:
    return "u_" + "".join(str(i + 1) for i in alpha)


def active_exponents(alpha: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(Counter(alpha).values(), reverse=True))


class Sin(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


class Sinh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sinh(x)


def activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name in {"sin", "sine"}:
        return Sin()
    if name == "sinh":
        return Sinh()
    raise ValueError(f"unknown activation: {name}")


def build_mlp(d_in: int, hidden: int, depth: int, dtype: torch.dtype, device: torch.device, activation: str) -> nn.Module:
    layers: list[nn.Module] = [nn.Linear(d_in, hidden), activation_module(activation)]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), activation_module(activation)]
    layers.append(nn.Linear(hidden, 1))
    net = nn.Sequential(*layers)
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net.to(device=device, dtype=dtype)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_grads(model: nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


def reset_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_alloc_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)


def run_timed(fn: Callable[[], Tensor], model_for_grad: nn.Module, device: torch.device, repeats: int, warmup: int, measure: str) -> tuple[Tensor, float, float | None]:
    last: Tensor | None = None
    for _ in range(warmup):
        clear_grads(model_for_grad)
        y = fn()
        if measure == "backward":
            y.real.square().mean().backward()
        sync(device)
        last = y.detach()
    times: list[float] = []
    mem: float | None = None
    for _ in range(repeats):
        clear_grads(model_for_grad)
        reset_memory(device)
        sync(device)
        t0 = time.perf_counter()
        y = fn()
        if measure == "backward":
            y.real.square().mean().backward()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
        pa = peak_alloc_mb(device)
        mem = pa if mem is None else max(mem, pa or 0.0)
        last = y.detach()
    assert last is not None
    return last, statistics.median(times), mem


def rel_error(ref: Tensor, y: Tensor) -> tuple[float, float]:
    yy = y.real if y.dtype.is_complex else y
    rr = ref.real if ref.dtype.is_complex else ref
    yy = yy.to(dtype=rr.dtype)
    abs_err = (rr - yy).abs().max().item()
    return abs_err, abs_err / (rr.abs().max().item() + 1e-30)


def probabilists_hermite(z: Tensor, n: int) -> Tensor:
    if n == 0:
        return torch.ones_like(z)
    if n == 1:
        return z
    hm2 = torch.ones_like(z)
    hm1 = z
    for k in range(1, n):
        h = z * hm1 - float(k) * hm2
        hm2, hm1 = hm1, h
    return hm1


def hermite_weight(Z: Tensor, alpha: tuple[int, ...]) -> Tensor:
    counts = Counter(alpha)
    w = torch.ones(Z.shape[:-1] + (1,), device=Z.device, dtype=Z.dtype)
    for idx, count in counts.items():
        w = w * probabilists_hermite(Z[..., idx:idx + 1], count)
    return w


def gaussian_hermite_mc(model: nn.Module, x: Tensor, alpha: tuple[int, ...], Z: Tensor, sigma: float) -> Tensor:
    B, K, d = Z.shape
    flat = (x.unsqueeze(1) + sigma * Z).reshape(B * K, d)
    u = model(flat).reshape(B, K, 1)
    w = hermite_weight(Z, alpha)
    return (w * u).mean(dim=1) / (sigma ** len(alpha))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", default="sinh", choices=["tanh", "sigmoid", "sin", "sinh"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--measure", default="value", choices=["value", "backward"])
    parser.add_argument("--alphas", default="111111;111122;112233;123456;11111111;11112222;11223344;12345678")
    parser.add_argument("--methods", default="direct_autodiff,polarization_jet,waring_complex_jet,auto")
    parser.add_argument("--mc_K", type=int, default=256)
    parser.add_argument("--mc_sigma", type=float, default=0.1)
    parser.add_argument("--out", default="results/single_monomial_benchmark.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    alphas = parse_alpha_list(args.alphas)
    methods = [m for m in args.methods.split(",") if m]

    torch.manual_seed(args.seed)
    model = build_mlp(args.d, args.hidden, args.depth, dtype, device, args.activation)
    x = torch.randn(args.batch, args.d, device=device, dtype=dtype) * 0.5
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64

    rows: list[dict[str, Any]] = []
    print(f"device={device} dtype={dtype} d={args.d} B={args.batch} hidden={args.hidden} depth={args.depth} activation={args.activation} measure={args.measure}")
    print("alpha\tpattern\tmethod\tdirs\trel_err\tmedian_ms\tpeak_mb\tstatus")

    for alpha in alphas:
        ref = direct_monomial_autodiff(model, x, alpha, create_graph=False).detach()
        _, _, cinfo = monomial_waring_directions(alpha, args.d, device=device, dtype=complex_dtype)
        _, _, rinfo = _polarization_directions(alpha, args.d, device=device, dtype=dtype)
        for method in methods:
            row = {
                "alpha": alpha_label(alpha),
                "active_exponents": str(active_exponents(alpha)),
                "order": len(alpha),
                "method": method,
                "complex_rank": cinfo.rank,
                "polarization_dirs": rinfo.rank,
                "d": args.d,
                "batch": args.batch,
                "hidden": args.hidden,
                "depth": args.depth,
                "activation": args.activation,
                "dtype": args.dtype,
                "measure": args.measure,
            }
            try:
                if method == "direct_autodiff":
                    fn = lambda alpha=alpha: direct_monomial_autodiff(model, x, alpha, create_graph=(args.measure == "backward"))
                    grad_model = model
                    dirs = None
                elif method == "waring_complex_jet":
                    fn = lambda alpha=alpha: single_monomial_partial(model, x, alpha, backend="waring_complex_jet")
                    grad_model = model
                    dirs = cinfo.rank
                elif method == "polarization_jet":
                    fn = lambda alpha=alpha: single_monomial_partial(model, x, alpha, backend="polarization_jet")
                    grad_model = model
                    dirs = rinfo.rank
                elif method == "auto":
                    selected = "polarization_jet" if args.measure == "backward" else (
                        "waring_complex_jet" if cinfo.rank <= 0.8 * rinfo.rank else "polarization_jet"
                    )
                    fn = lambda alpha=alpha, selected=selected: single_monomial_partial(model, x, alpha, backend=selected)
                    grad_model = model
                    dirs = cinfo.rank if selected == "waring_complex_jet" else rinfo.rank
                    row["selected_backend"] = selected
                elif method == "gaussian_hermite_mc":
                    Z_mc = torch.randn(args.batch, args.mc_K, args.d, device=device, dtype=dtype)
                    fn = lambda alpha=alpha, Z_mc=Z_mc: gaussian_hermite_mc(model, x, alpha, Z_mc, args.mc_sigma)
                    grad_model = model
                    dirs = args.mc_K
                    row["mc_K"] = args.mc_K
                    row["mc_sigma"] = args.mc_sigma
                else:
                    raise ValueError(f"unknown method: {method}")
                y, ms, mem = run_timed(fn, grad_model, device, args.repeats, args.warmup, args.measure)
                ae, re = rel_error(ref, y)
                row.update({"dirs": dirs, "abs_err": ae, "rel_err": re, "median_ms": ms, "peak_alloc_mb": mem, "status": "ok"})
            except Exception as exc:
                row.update({"dirs": None, "abs_err": None, "rel_err": None, "median_ms": None, "peak_alloc_mb": None, "status": f"ERROR: {type(exc).__name__}: {exc}"})
            rows.append(row)
            print(f"{row['alpha']}\t{row['active_exponents']}\t{method}\t{row.get('dirs')}\t{row.get('rel_err')}\t{row.get('median_ms')}\t{row.get('peak_alloc_mb')}\t{row['status']}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    with out.with_suffix(".json").open("w") as f:
        json.dump(rows, f, indent=2)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
