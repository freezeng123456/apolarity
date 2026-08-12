#!/usr/bin/env python3
"""Build a convergence-audited Fourier reference for the HO-02 MPFC task.

The reference uses a Fourier pseudospectral spatial discretisation with a
two-thirds dealiased cubic term and a linearly implicit first-order update for
the damped-wave pair ``(phi, phi_t)``.  It is intentionally independent from
the PINN residual implementation.  Nested grids and time steps are evaluated
at the same grid-aligned sample points; only the finest level is written as
the reference dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROTOCOL_ID = "mpfc_2d_imex_fourier_reference_v1"
DOMAIN_MAX = 2.0 * math.pi
T_MAX = 1.0
MOBILITY = 1.0
BETA = 0.1
EPSILON = 0.25
STABILIZATION = 1.0
DEFAULT_LEVELS = "32:0.0005,64:0.00025,128:0.000125"
DEFAULT_TOLERANCE = 2e-3
DIAGNOSTIC_TIMES = (0.0, 0.25, 0.5, 0.75, 1.0)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class Level:
    side: int
    dt: float

    @property
    def steps(self) -> int:
        return round(T_MAX / self.dt)


def parse_levels(raw: str) -> tuple[Level, ...]:
    levels: list[Level] = []
    for item in raw.split(","):
        side_raw, dt_raw = item.split(":", 1)
        level = Level(int(side_raw), float(dt_raw))
        if level.side < 16 or level.side % 2:
            raise ValueError("reference sides must be even and at least 16")
        if level.dt <= 0 or not math.isclose(level.steps * level.dt, T_MAX, abs_tol=1e-12):
            raise ValueError("each dt must divide T_MAX exactly")
        levels.append(level)
    if len(levels) != 3:
        raise ValueError("reference requires exactly three nested levels")
    if any(
        later.side <= earlier.side or later.dt >= earlier.dt
        for earlier, later in zip(levels[:-1], levels[1:], strict=True)
    ):
        raise ValueError("levels must refine space and time monotonically")
    base_side, base_dt = levels[0].side, levels[0].dt
    for level in levels:
        if level.side % base_side:
            raise ValueError("spatial grids must be nested")
        ratio = base_dt / level.dt
        if not math.isclose(ratio, round(ratio), abs_tol=1e-12):
            raise ValueError("time grids must be nested")
    return tuple(levels)


def initial_phi(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        0.1
        + 0.15 * torch.cos(x) * torch.cos(y)
        + 0.05 * torch.cos(2.0 * x) * torch.cos(y)
    )


class IMEXMPFC:
    """Linearly implicit Fourier solver for one nested reference level."""

    def __init__(self, level: Level, device: torch.device) -> None:
        self.level = level
        self.device = device
        self.real_dtype = torch.float64
        self.complex_dtype = torch.complex128
        n = level.side
        frequencies = torch.fft.fftfreq(
            n, d=1.0 / n, device=device, dtype=self.real_dtype
        )
        self.kx, self.ky = torch.meshgrid(frequencies, frequencies, indexing="ij")
        self.k2 = self.kx.square() + self.ky.square()
        q = self.k2.square() - 2.0 * self.k2 + (1.0 - EPSILON)
        self.linear = MOBILITY * self.k2 * q
        cutoff = n / 3.0
        self.dealias = (
            (self.kx.abs() <= cutoff) & (self.ky.abs() <= cutoff)
        )
        grid = torch.arange(n, device=device, dtype=self.real_dtype) * (DOMAIN_MAX / n)
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        self.phi_hat = torch.fft.fft2(initial_phi(x, y)).to(self.complex_dtype)
        self.velocity_hat = torch.zeros_like(self.phi_hat)
        self._refresh_coefficients()

    def _refresh_coefficients(self) -> None:
        dt = self.level.dt
        beta_over_dt = BETA / dt
        self.denominator = beta_over_dt + 1.0 + dt * (
            self.linear + MOBILITY * self.k2 * STABILIZATION
        )
        self.k2_complex = self.k2.to(self.complex_dtype)
        self.linear_complex = self.linear.to(self.complex_dtype)

    def step(self) -> None:
        phi = torch.fft.ifft2(self.phi_hat).real
        nonlinear_hat = torch.fft.fft2(phi.square() * phi).to(self.complex_dtype)
        nonlinear_hat = nonlinear_hat * self.dealias
        rhs = (
            (BETA / self.level.dt) * self.velocity_hat
            - self.linear_complex * self.phi_hat
            - MOBILITY * self.k2_complex * nonlinear_hat
        )
        velocity_new = rhs / self.denominator
        phi_new = self.phi_hat + self.level.dt * velocity_new
        # The zero Fourier mode is constant for zero-mean initial velocity.
        velocity_new[0, 0] = 0.0
        phi_new[0, 0] = self.phi_hat[0, 0]
        self.velocity_hat = velocity_new
        self.phi_hat = phi_new

    def physical(self) -> torch.Tensor:
        return torch.fft.ifft2(self.phi_hat).real

    def diagnostics(self) -> dict[str, float]:
        phi = self.physical()
        lap = torch.fft.ifft2(-self.k2_complex * self.phi_hat).real
        vx = torch.fft.ifft2(1j * self.kx.to(self.complex_dtype) * self.phi_hat).real
        vy = torch.fft.ifft2(1j * self.ky.to(self.complex_dtype) * self.phi_hat).real
        # E(phi)=1/2 ||Delta phi||^2 - ||grad phi||^2
        energy = torch.mean(0.5 * lap.square() - 0.5 * (vx.square() + vy.square()) + 0.25 * phi.pow(4) + 0.5 * (1.0 - EPSILON) * phi.square())
        velocity_hminus1 = (self.velocity_hat.abs().square() / self.k2.clamp_min(1.0)).sum() / (self.level.side**4)
        pseudo_energy = energy + 0.5 * BETA * velocity_hminus1
        return {
            "mass": float(phi.mean().item()),
            "mass_error": float((phi.mean() - 0.1).abs().item()),
            "energy": float(energy.item()),
            "pseudo_energy": float(pseudo_energy.item()),
            "phi_rms": float(phi.square().mean().sqrt().item()),
            "phi_t_rms": float(torch.fft.ifft2(self.velocity_hat).real.square().mean().sqrt().item()),
        }


def fixed_sample_indices(count: int, base_side: int, base_steps: int, seed: int) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    return {
        "x": generator.integers(0, base_side, size=count, dtype=np.int64),
        "y": generator.integers(0, base_side, size=count, dtype=np.int64),
        "t": generator.integers(0, base_steps + 1, size=count, dtype=np.int64),
    }


def solve_level(
    level: Level,
    base_level: Level,
    indices: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, dict[str, float]], float]:
    solver = IMEXMPFC(level, device)
    spatial_ratio = level.side // base_level.side
    temporal_ratio = round(base_level.dt / level.dt)
    requested_steps = indices["t"] * temporal_ratio
    grouped = {int(step): np.flatnonzero(requested_steps == step) for step in np.unique(requested_steps)}
    values = np.empty(indices["t"].shape[0], dtype=np.float64)
    diagnostic_steps = {round(value / level.dt): value for value in DIAGNOSTIC_TIMES}
    diagnostics: dict[str, dict[str, float]] = {}

    def capture(step: int) -> None:
        selected = grouped.get(step)
        if selected is not None:
            phi = solver.physical()
            x_index = torch.as_tensor(indices["x"][selected] * spatial_ratio, device=device, dtype=torch.long)
            y_index = torch.as_tensor(indices["y"][selected] * spatial_ratio, device=device, dtype=torch.long)
            values[selected] = phi[x_index, y_index].detach().cpu().numpy()
        if step in diagnostic_steps:
            diagnostics[f"{diagnostic_steps[step]:.2f}"] = solver.diagnostics()

    started = time.perf_counter()
    capture(0)
    for step in range(1, level.steps + 1):
        solver.step()
        capture(step)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not np.isfinite(values).all():
        raise FloatingPointError(f"non-finite reference at side={level.side}")
    return values, diagnostics, elapsed


def relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    numerator = np.sqrt(np.mean(np.square(left - right)))
    denominator = max(np.sqrt(np.mean(np.square(right))), 1e-14)
    return float(numerator / denominator)


def build_reference(args: argparse.Namespace) -> dict[str, Any]:
    levels = parse_levels(args.levels)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reference requested but unavailable")
    indices = fixed_sample_indices(args.n_eval, levels[0].side, levels[0].steps, args.eval_seed)
    values_by_level: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for level in levels:
        values, diagnostics, elapsed = solve_level(level, levels[0], indices, device)
        values_by_level.append(values)
        records.append({
            "side": level.side,
            "dt": level.dt,
            "steps": level.steps,
            "elapsed_seconds": elapsed,
            "mass_max_abs": max(abs(item["mass"]) for item in diagnostics.values()),
            "mass_error_max_abs": max(item["mass_error"] for item in diagnostics.values()),
            "energy_by_time": {key: value["energy"] for key, value in diagnostics.items()},
            "pseudo_energy_by_time": {key: value["pseudo_energy"] for key, value in diagnostics.items()},
            "phi_rms_by_time": {key: value["phi_rms"] for key, value in diagnostics.items()},
            "phi_t_rms_by_time": {key: value["phi_t_rms"] for key, value in diagnostics.items()},
        })
    coarse_medium = relative_difference(values_by_level[0], values_by_level[1])
    medium_fine = relative_difference(values_by_level[1], values_by_level[2])
    convergence_passed = bool(medium_fine <= args.tolerance and medium_fine <= 1.1 * max(coarse_medium, 1e-14))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    eval_points = np.stack([
        indices["x"] * (DOMAIN_MAX / levels[0].side),
        indices["y"] * (DOMAIN_MAX / levels[0].side),
        indices["t"] * levels[0].dt,
    ], axis=1)
    np.savez_compressed(output / "reference.npz", points=eval_points, phi=values_by_level[-1])
    report = {
        "protocol_id": PROTOCOL_ID,
        "created_at": utc_now(),
        "levels": records,
        "sample_count": int(args.n_eval),
        "eval_seed": args.eval_seed,
        "coarse_medium_rel_error": coarse_medium,
        "medium_fine_rel_error": medium_fine,
        "tolerance": args.tolerance,
        "convergence_passed": convergence_passed,
        "pseudo_energy_nonincreasing": all(
            later <= earlier + 5e-8
            for record in records
            for earlier, later in zip(
                [record["pseudo_energy_by_time"][key] for key in sorted(record["pseudo_energy_by_time"])][:-1],
                [record["pseudo_energy_by_time"][key] for key in sorted(record["pseudo_energy_by_time"])][1:],
            )
        ),
        "device": str(device),
        "parameters": {
            "M": MOBILITY,
            "beta": BETA,
            "epsilon": EPSILON,
            "linear_stabilization": STABILIZATION,
        },
        "initial_condition": "0.1 + 0.15*cos(x)*cos(y) + 0.05*cos(2*x)*cos(y)",
    }
    atomic_write_json(output / "reference_report.json", report)
    digest = hashlib.sha256((output / "reference.npz").read_bytes()).hexdigest()
    atomic_write_text(output / "REFERENCE_SHA256", digest + "  reference.npz\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", default=DEFAULT_LEVELS)
    parser.add_argument("--n-eval", type=int, default=4096)
    parser.add_argument("--eval-seed", type=int, default=68421)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_reference(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["convergence_passed"]:
        raise SystemExit("MPFC reference convergence gate failed")


if __name__ == "__main__":
    main()
