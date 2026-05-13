#!/usr/bin/env python3
"""Step-by-step profiler for complex Waring + Taylor-jet single partials.

This script prints wall-clock time for:
  - complex direction generation
  - input/direction preparation
  - each Linear / activation layer in the Taylor-jet pass
  - weighted summation
  - optional backward pass

It is intentionally diagnostic rather than a clean benchmark.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn
from torch import Tensor

from apolarity.waring import monomial_waring_directions


class Sin(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


def activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name in {"sin", "sine"}:
        return Sin()
    raise ValueError(f"unknown activation: {name}")


def build_mlp(d_in: int, hidden: int, depth: int, dtype: torch.dtype, device: torch.device, activation: str) -> nn.Sequential:
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


def parse_alpha(text: str) -> tuple[int, ...]:
    vals = [int(ch) for ch in text] if "," not in text else [int(x) for x in text.split(",") if x]
    if any(v <= 0 for v in vals):
        raise ValueError("alpha is one-based, e.g. 111122 or 1,2,3")
    return tuple(v - 1 for v in vals)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(name: str, out: dict[str, float], device: torch.device, fn):
    sync(device)
    t0 = time.perf_counter()
    ret = fn()
    sync(device)
    out[name] = out.get(name, 0.0) + (time.perf_counter() - t0) * 1000.0
    return ret


def is_sin_module(layer: nn.Module) -> bool:
    return layer.__class__.__name__.lower() in {"sin", "sine", "sinactivation"}


def cast_linear(weight: Tensor, bias: Tensor | None, dtype: torch.dtype) -> tuple[Tensor, Tensor | None]:
    if dtype.is_complex and not weight.dtype.is_complex:
        return weight.to(dtype=dtype), None if bias is None else bias.to(dtype=dtype)
    return weight, bias


def linear_jet_terms(terms: list[Tensor], layer: nn.Linear) -> list[Tensor]:
    out: list[Tensor] = []
    for k, xk in enumerate(terms):
        w, b = cast_linear(layer.weight, layer.bias, xk.dtype)
        yk = xk @ w.T
        if k == 0 and b is not None:
            yk = yk + b
        out.append(yk)
    return out


def tanh_jet_terms(terms: list[Tensor]) -> list[Tensor]:
    p = len(terms) - 1
    x = terms
    y: list[Tensor] = [torch.tanh(x[0])]
    z: list[Tensor] = [1.0 - y[0] * y[0]]
    for k in range(1, p + 1):
        acc = k * z[0] * x[k]
        for j in range(1, k):
            acc = acc + (k - j) * z[j] * x[k - j]
        yk = acc / float(k)
        y.append(yk)
        zk = -y[0] * y[k]
        for j in range(1, k + 1):
            zk = zk - y[j] * y[k - j]
        z.append(zk)
    return y


def sigmoid_jet_terms(terms: list[Tensor]) -> list[Tensor]:
    p = len(terms) - 1
    x = terms
    y: list[Tensor] = [torch.sigmoid(x[0])]
    z: list[Tensor] = [y[0] - y[0] * y[0]]
    for k in range(1, p + 1):
        acc = k * z[0] * x[k]
        for j in range(1, k):
            acc = acc + (k - j) * z[j] * x[k - j]
        yk = acc / float(k)
        y.append(yk)
        yy = y[0] * y[k]
        for j in range(1, k + 1):
            yy = yy + y[j] * y[k - j]
        z.append(y[k] - yy)
    return y


def sin_jet_terms(terms: list[Tensor]) -> list[Tensor]:
    p = len(terms) - 1
    x = terms
    y: list[Tensor] = [torch.sin(x[0])]
    c: list[Tensor] = [torch.cos(x[0])]
    for k in range(1, p + 1):
        acc_y = k * c[0] * x[k]
        acc_c = -k * y[0] * x[k]
        for j in range(1, k):
            acc_y = acc_y + (k - j) * c[j] * x[k - j]
            acc_c = acc_c - (k - j) * y[j] * x[k - j]
        y.append(acc_y / float(k))
        c.append(acc_c / float(k))
    return y


def profile_once(model: nn.Sequential, x: Tensor, alpha: tuple[int, ...], measure: str) -> tuple[dict[str, float], dict[str, Any]]:
    device = x.device
    p = len(alpha)
    complex_dtype = torch.complex128 if x.dtype == torch.float64 else torch.complex64
    times: dict[str, float] = {}

    V, coeff, info = timed(
        "direction_generation",
        times,
        device,
        lambda: monomial_waring_directions(alpha, x.shape[1], device=device, dtype=complex_dtype),
    )

    def prep():
        B, d = x.shape
        cx = x.to(dtype=complex_dtype)
        Z = V.unsqueeze(0).expand(B, V.shape[0], d).contiguous()
        x_exp = cx.unsqueeze(1).expand(B, V.shape[0], d).reshape(B * V.shape[0], d)
        z_flat = Z.reshape(B * V.shape[0], d)
        zero = torch.zeros_like(x_exp)
        terms = [x_exp, z_flat] + [zero for _ in range(p - 1)]
        return terms

    terms = timed("prepare_terms", times, device, prep)

    for li, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            key = f"layer_{li:02d}_linear_{layer.in_features}x{layer.out_features}"
            terms = timed(key, times, device, lambda layer=layer, terms=terms: linear_jet_terms(terms, layer))
        elif isinstance(layer, nn.Tanh):
            terms = timed(f"layer_{li:02d}_tanh", times, device, lambda terms=terms: tanh_jet_terms(terms))
        elif isinstance(layer, nn.Sigmoid):
            terms = timed(f"layer_{li:02d}_sigmoid", times, device, lambda terms=terms: sigmoid_jet_terms(terms))
        elif is_sin_module(layer):
            terms = timed(f"layer_{li:02d}_sin", times, device, lambda terms=terms: sin_jet_terms(terms))
        else:
            raise NotImplementedError(type(layer).__name__)

    def reduce_sum():
        B = x.shape[0]
        Tp = terms[p].reshape(B, V.shape[0], 1)
        return (Tp * coeff.view(1, -1, 1)).sum(dim=1)

    y = timed("weighted_sum", times, device, reduce_sum)

    if measure == "backward":
        for prm in model.parameters():
            prm.grad = None
        timed("backward", times, device, lambda: y.real.square().mean().backward())

    total = sum(times.values())
    meta = {
        "order": p,
        "rank": info.rank,
        "dtype": str(x.dtype).replace("torch.", ""),
        "complex_dtype": str(complex_dtype).replace("torch.", ""),
        "total_ms": total,
    }
    return times, meta


def summarize(records: list[tuple[dict[str, float], dict[str, Any]]]) -> tuple[dict[str, float], dict[str, Any]]:
    keys = sorted({k for times, _ in records for k in times})
    med: dict[str, float] = {}
    for k in keys:
        vals = sorted(times.get(k, 0.0) for times, _ in records)
        med[k] = vals[len(vals) // 2]
    meta = dict(records[-1][1])
    meta["total_ms"] = sum(med.values())
    return med, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--activation", default="tanh", choices=["tanh", "sigmoid", "sin"])
    parser.add_argument("--alpha", default="11223344")
    parser.add_argument("--measure", default="backward", choices=["value", "backward"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    alpha = parse_alpha(args.alpha)
    torch.manual_seed(args.seed)
    model = build_mlp(args.d, args.hidden, args.depth, dtype, device, args.activation)
    x = torch.randn(args.batch, args.d, device=device, dtype=dtype) * 0.5

    for _ in range(args.warmup):
        profile_once(model, x, alpha, args.measure)
    records = [profile_once(model, x, alpha, args.measure) for _ in range(args.repeats)]
    times, meta = summarize(records)

    print("=== complex Waring step profile ===")
    print({**meta, "alpha": args.alpha, "activation": args.activation, "batch": args.batch, "hidden": args.hidden, "depth": args.depth, "measure": args.measure})
    for k, v in sorted(times.items(), key=lambda kv: kv[1], reverse=True):
        print(f"{k:40s} {v:10.3f} ms  {100.0 * v / meta['total_ms']:6.2f}%")
    print(f"{'TOTAL':40s} {meta['total_ms']:10.3f} ms")

    if args.out:
        out = ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["step", "ms", "pct"])
            writer.writeheader()
            for k, v in sorted(times.items(), key=lambda kv: kv[1], reverse=True):
                writer.writerow({"step": k, "ms": v, "pct": 100.0 * v / meta["total_ms"]})
        print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
