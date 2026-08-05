#!/usr/bin/env python3
"""Overnight correctness and performance audit for exact monomial backends.

The runner compares value evaluation and parameter-backward cost on identical
models, and separates the current end-to-end API from cached direction
schedules.  It checkpoints JSON/CSV after every cell so a hard deadline does
not discard completed measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from apolarity.operators import direct_monomial_autodiff, single_monomial_partial
from apolarity.polarization import polarization_directions
from apolarity.taylor_jet import tp_directional_via_jet
from apolarity.waring import monomial_waring_directions


VALUE_RTOL = 2.0e-10
VALUE_ATOL = 2.0e-11
GRAD_RTOL = 2.0e-9
GRAD_ATOL = 2.0e-11


class SinhActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sinh(x)


def build_model(d: int, hidden: int, depth: int, dtype: torch.dtype,
                device: torch.device, seed: int) -> nn.Sequential:
    torch.manual_seed(seed)
    layers: list[nn.Module] = []
    fan_in = d
    for _ in range(depth):
        layers.extend([nn.Linear(fan_in, hidden), SinhActivation()])
        fan_in = hidden
    layers.append(nn.Linear(fan_in, 1))
    model = nn.Sequential(*layers).to(device=device, dtype=dtype)
    with torch.no_grad():
        for layer in model:
            if not isinstance(layer, nn.Linear):
                continue
            if dtype.is_complex:
                real = torch.empty_like(layer.weight.real)
                imag = torch.empty_like(layer.weight.real)
                nn.init.xavier_uniform_(real)
                nn.init.xavier_uniform_(imag)
                layer.weight.copy_((real + 1j * imag) / math.sqrt(2.0))
            else:
                nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return model


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_grads(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def alpha_pattern(alpha: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(Counter(alpha).values(), reverse=True))


def cached_formula(model: nn.Module, x: Tensor, directions: Tensor,
                   coefficients: Tensor, order: int) -> Tensor:
    batch, d = x.shape
    z = directions.unsqueeze(0).expand(batch, directions.shape[0], d).contiguous()
    terms = tp_directional_via_jet(model, x, z, order)
    return (terms * coefficients.view(1, -1, 1)).sum(dim=1)


def build_schedules(alpha: tuple[int, ...], d: int, x: Tensor):
    real_dtype = torch.float64 if x.dtype in (torch.float64, torch.complex128) else torch.float32
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    vp, cp, pinfo = polarization_directions(
        alpha, d, device=x.device, dtype=real_dtype, antipodal=True
    )
    if x.dtype.is_complex:
        vp = vp.to(dtype=x.dtype)
        cp = cp.to(dtype=x.dtype)
    vw, cw, winfo = monomial_waring_directions(
        alpha, d, device=x.device, dtype=complex_dtype
    )
    return vp, cp, pinfo, vw, cw, winfo


def output_error(reference: Tensor, candidate: Tensor) -> dict[str, Any]:
    ref = reference.detach()
    cand = candidate.detach().to(dtype=ref.dtype)
    diff = (ref - cand).abs()
    max_abs = float(diff.max().item())
    denom = float(ref.abs().max().item()) + 1.0e-300
    rel_l2 = float(torch.linalg.vector_norm((ref - cand).reshape(-1)).item()) / (
        float(torch.linalg.vector_norm(ref.reshape(-1)).item()) + 1.0e-300
    )
    return {
        "value_max_abs": max_abs,
        "value_max_rel": max_abs / denom,
        "value_rel_l2": rel_l2,
        "value_allclose": bool(torch.allclose(ref, cand, rtol=VALUE_RTOL, atol=VALUE_ATOL)),
    }


def gradient_snapshot(model: nn.Module) -> list[Tensor]:
    return [
        torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in model.parameters()
    ]


def gradient_error(reference: list[Tensor], candidate: list[Tensor]) -> dict[str, Any]:
    max_abs = 0.0
    ref_norm_sq = 0.0
    diff_norm_sq = 0.0
    allclose = True
    for ref, cand in zip(reference, candidate):
        diff = (ref - cand).abs()
        max_abs = max(max_abs, float(diff.max().item()))
        ref_norm_sq += float(ref.abs().square().sum().item())
        diff_norm_sq += float(diff.square().sum().item())
        allclose = allclose and bool(torch.allclose(ref, cand, rtol=GRAD_RTOL, atol=GRAD_ATOL))
    return {
        "grad_max_abs": max_abs,
        "grad_rel_l2": math.sqrt(diff_norm_sq) / (math.sqrt(ref_norm_sq) + 1.0e-300),
        "grad_allclose": allclose,
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def timed(fn: Callable[[], Tensor], model: nn.Module, measure: str,
          device: torch.device, warmup: int, max_repeats: int,
          target_seconds: float) -> tuple[Tensor, dict[str, Any]]:
    for _ in range(warmup):
        clear_grads(model)
        y = fn()
        if measure == "backward":
            y.real.square().mean().backward()
        sync(device)

    clear_grads(model)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)

    times: list[float] = []
    block_start = time.perf_counter()
    last: Tensor | None = None
    while len(times) < max_repeats:
        clear_grads(model)
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            y = fn()
            if measure == "backward":
                y.real.square().mean().backward()
            end.record()
            end.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            y = fn()
            if measure == "backward":
                y.real.square().mean().backward()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times.append(elapsed_ms)
        last = y.detach()
        if len(times) >= 10 and time.perf_counter() - block_start >= target_seconds:
            break

    assert last is not None
    peak_alloc = None
    peak_reserved = None
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024.0 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024.0 ** 2)
    return last, {
        "repeats": len(times),
        "median_ms": statistics.median(times),
        "q25_ms": percentile(times, 0.25),
        "q75_ms": percentile(times, 0.75),
        "mean_ms": statistics.fmean(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_alloc_mb": peak_alloc,
        "peak_reserved_mb": peak_reserved,
    }


def save_rows(rows: list[dict[str, Any]], out: Path, manifest: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"manifest": manifest, "rows": rows}
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(out)
    csv_path = out.with_suffix(".csv")
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    tmp_csv = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    tmp_csv.replace(csv_path)


def configs(profile: str):
    all_alphas = [
        (0, 0, 0),
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 1),
        (0, 0, 1, 1, 2, 2),
        (0, 1, 2, 3, 4, 5),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 1, 1, 1, 1),
        (0, 0, 1, 1, 2, 2, 3, 3),
        (0, 1, 2, 3, 4, 5, 6, 7),
    ]
    representative = [
        (0, 0, 0, 0),
        (0, 0, 0, 0, 1, 1),
        (0, 1, 2, 3, 4, 5),
        (0, 0, 0, 0, 1, 1, 1, 1),
    ]
    if profile == "smoke":
        return [(torch.complex128, 2, alpha) for alpha in all_alphas[:5]]
    result = [(torch.complex128, 8, alpha) for alpha in all_alphas]
    result += [(torch.complex128, batch, alpha) for batch in (1, 64) for alpha in representative]
    result += [(torch.float64, 8, alpha) for alpha in all_alphas]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-repeats", type=int, default=100)
    parser.add_argument("--target-seconds", type=float, default=3.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seeds = [int(token) for token in args.seeds.split(",") if token]
    variants = (
        "direct_autodiff",
        "polarization_uncached",
        "polarization_cached",
        "waring_uncached",
        "waring_cached",
        "auto_uncached",
        "auto_cached_selected",
    )
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profile": args.profile,
        "argv": sys.argv,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "value_rtol": VALUE_RTOL,
        "value_atol": VALUE_ATOL,
        "grad_rtol": GRAD_RTOL,
        "grad_atol": GRAD_ATOL,
        "git_sha": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
        "git_dirty": bool(os.popen(f"git -C {ROOT} status --porcelain").read().strip()),
    }
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        for dtype, batch, alpha in configs(args.profile):
            dtype_name = str(dtype).replace("torch.", "")
            base = {
                "seed": seed,
                "dtype": dtype_name,
                "d": args.d,
                "batch": batch,
                "hidden": args.hidden,
                "depth": args.depth,
                "alpha": list(alpha),
                "pattern": list(alpha_pattern(alpha)),
                "order": len(alpha),
            }
            try:
                model = build_model(args.d, args.hidden, args.depth, dtype, device, seed)
                generator = torch.Generator(device=device).manual_seed(100000 + seed * 1000 + batch + len(alpha))
                xr = 0.35 * torch.randn(batch, args.d, generator=generator, device=device, dtype=torch.float64)
                x = xr.to(dtype=dtype)
                vp, cp, pinfo, vw, cw, winfo = build_schedules(alpha, args.d, x)
                selected = "waring_complex_jet" if winfo.rank <= 0.8 * pinfo.rank else "polarization_jet"

                def evaluate(variant: str, *, create_graph: bool) -> Tensor:
                    if variant == "direct_autodiff":
                        return direct_monomial_autodiff(model, x, alpha, create_graph=create_graph)
                    if variant == "polarization_uncached":
                        return single_monomial_partial(model, x, alpha, backend="polarization_jet")
                    if variant == "polarization_cached":
                        return cached_formula(model, x, vp, cp, len(alpha))
                    if variant == "waring_uncached":
                        return single_monomial_partial(model, x, alpha, backend="waring_complex_jet")
                    if variant == "waring_cached":
                        return cached_formula(model, x.to(dtype=vw.dtype), vw, cw, len(alpha))
                    if variant == "auto_uncached":
                        return single_monomial_partial(model, x, alpha, backend="auto")
                    if variant == "auto_cached_selected":
                        if selected == "waring_complex_jet":
                            return cached_formula(model, x.to(dtype=vw.dtype), vw, cw, len(alpha))
                        return cached_formula(model, x, vp, cp, len(alpha))
                    raise ValueError(variant)

                clear_grads(model)
                reference_value = direct_monomial_autodiff(model, x, alpha, create_graph=False).detach()
                clear_grads(model)
                reference_for_grad = direct_monomial_autodiff(model, x, alpha, create_graph=True)
                reference_for_grad.real.square().mean().backward()
                reference_grad = gradient_snapshot(model)
                clear_grads(model)

                for variant in variants:
                    for measure in ("value", "backward"):
                        row = {
                            **base,
                            "variant": variant,
                            "measure": measure,
                            "polarization_dirs": pinfo.rank,
                            "waring_dirs": winfo.rank,
                            "auto_selected": selected,
                        }
                        try:
                            clear_grads(model)
                            candidate = evaluate(variant, create_graph=(measure == "backward"))
                            row.update(output_error(reference_value, candidate))
                            if measure == "backward":
                                candidate.real.square().mean().backward()
                                row.update(gradient_error(reference_grad, gradient_snapshot(model)))
                            clear_grads(model)
                            _last, timing = timed(
                                lambda variant=variant, measure=measure: evaluate(
                                    variant, create_graph=(measure == "backward")
                                ), model, measure,
                                device, args.warmup, args.max_repeats, args.target_seconds,
                            )
                            row.update(timing)
                            row["status"] = "ok"
                        except Exception as exc:
                            row["status"] = f"ERROR: {type(exc).__name__}: {exc}"
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                        rows.append(row)
                        save_rows(rows, args.out, manifest)
                        print(json.dumps(row, sort_keys=True), flush=True)
                del model, x, xr
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:
                row = {**base, "variant": "config_setup", "measure": "setup",
                       "status": f"ERROR: {type(exc).__name__}: {exc}"}
                rows.append(row)
                save_rows(rows, args.out, manifest)
                print(json.dumps(row, sort_keys=True), flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    save_rows(rows, args.out, manifest)
    print(f"[complete] rows={len(rows)} out={args.out}", flush=True)


if __name__ == "__main__":
    main()
