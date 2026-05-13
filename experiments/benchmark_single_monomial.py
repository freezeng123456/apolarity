#!/usr/bin/env python3
"""Benchmark exact single-monomial derivative backends.

This benchmark intentionally evaluates one expanded multi-index at a time.
"""
from __future__ import annotations

import argparse
import copy
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

from aploarity.operators import direct_monomial_autodiff, single_monomial_partial
from aploarity.real_waring import monomial_real_waring_directions
from aploarity.waring import monomial_waring_directions


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


def build_mlp(d_in: int, hidden: int, depth: int, dtype: torch.dtype, device: torch.device) -> nn.Module:
    layers: list[nn.Module] = [nn.Linear(d_in, hidden), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--measure", default="value", choices=["value", "backward"])
    parser.add_argument("--alphas", default="111111;111122;112233;123456;11111111;11112222;11223344;12345678")
    parser.add_argument("--methods", default="direct_autodiff,polarization_jet,waring_complex_jet,auto")
    parser.add_argument("--out", default="results/single_monomial_benchmark.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    alphas = parse_alpha_list(args.alphas)
    methods = [m for m in args.methods.split(",") if m]

    torch.manual_seed(args.seed)
    model = build_mlp(args.d, args.hidden, args.depth, dtype, device)
    x = torch.randn(args.batch, args.d, device=device, dtype=dtype) * 0.5
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    complex_model = copy.deepcopy(model).to(dtype=complex_dtype)

    rows: list[dict[str, Any]] = []
    print(f"device={device} dtype={dtype} d={args.d} B={args.batch} hidden={args.hidden} depth={args.depth} measure={args.measure}")
    print("alpha\tpattern\tmethod\tdirs\trel_err\tmedian_ms\tpeak_mb\tstatus")

    for alpha in alphas:
        ref = direct_monomial_autodiff(model, x, alpha, create_graph=False).detach()
        _, _, cinfo = monomial_waring_directions(alpha, args.d, device=device, dtype=complex_dtype)
        _, _, rinfo = monomial_real_waring_directions(alpha, args.d, device=device, dtype=dtype, strategy="polarization")
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
                "dtype": args.dtype,
                "measure": args.measure,
            }
            try:
                if method == "direct_autodiff":
                    fn = lambda alpha=alpha: direct_monomial_autodiff(model, x, alpha, create_graph=(args.measure == "backward"))
                    grad_model = model
                    dirs = None
                elif method == "waring_complex_jet":
                    fn = lambda alpha=alpha: single_monomial_partial(model, x, alpha, backend="waring_complex_jet", complex_model=complex_model)
                    grad_model = complex_model
                    dirs = cinfo.rank
                elif method == "polarization_jet":
                    fn = lambda alpha=alpha: single_monomial_partial(model, x, alpha, backend="polarization_jet")
                    grad_model = model
                    dirs = rinfo.rank
                elif method == "auto":
                    selected = "waring_complex_jet" if cinfo.rank <= 0.7 * rinfo.rank else "polarization_jet"
                    fn = lambda alpha=alpha, selected=selected: single_monomial_partial(model, x, alpha, backend=selected, complex_model=complex_model)
                    grad_model = complex_model if selected == "waring_complex_jet" else model
                    dirs = cinfo.rank if selected == "waring_complex_jet" else rinfo.rank
                    row["selected_backend"] = selected
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
