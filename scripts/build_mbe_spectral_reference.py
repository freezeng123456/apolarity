#!/usr/bin/env python3
"""Build and convergence-audit the unforced 2D MBE reference dataset.

The solver is an independently implemented Fourier pseudospectral ETDRK4
scheme with two-thirds dealiasing.  Three nested space/time resolutions are
evaluated at the same fixed grid-aligned points.  Only the finest values enter
the PINN accuracy metric; lower levels are retained in the audit report.
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


REFERENCE_PROTOCOL_ID = "mbe_2d_etdrk4_reference_v1"
DOMAIN_MAX = 2.0 * math.pi
T_MAX = 1.0
NU = 0.05
DEFAULT_LEVELS = "32:0.002,64:0.001,128:0.0005"
DEFAULT_TOLERANCE = 5e-4
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
        if level.side < 8 or level.side % 2:
            raise ValueError("reference grid sides must be even and at least 8")
        if level.dt <= 0 or not math.isclose(
            level.steps * level.dt, T_MAX, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("each reference dt must divide T_MAX exactly")
        levels.append(level)
    if len(levels) != 3:
        raise ValueError("the MBE reference requires exactly three levels")
    if any(
        later.side <= earlier.side or later.dt >= earlier.dt
        for earlier, later in zip(levels[:-1], levels[1:], strict=True)
    ):
        raise ValueError("levels must refine space and time monotonically")
    base_side = levels[0].side
    coarse_dt = levels[0].dt
    for level in levels:
        if level.side % base_side:
            raise ValueError("reference grids must be spatially nested")
        ratio = coarse_dt / level.dt
        if not math.isclose(ratio, round(ratio), abs_tol=1e-12):
            raise ValueError("reference time grids must be nested")
    return tuple(levels)


def initial_height(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (
        0.2 * torch.cos(x) * torch.cos(y)
        + 0.1 * torch.cos(2.0 * x) * torch.cos(y)
    )


class ETDRK4MBE:
    def __init__(self, level: Level, device: torch.device) -> None:
        self.level = level
        self.device = device
        self.real_dtype = torch.float64
        self.complex_dtype = torch.complex128
        n = level.side
        frequency = torch.fft.fftfreq(
            n, d=1.0 / n, device=device, dtype=self.real_dtype
        )
        self.kx, self.ky = torch.meshgrid(frequency, frequency, indexing="ij")
        self.k_squared = self.kx.square() + self.ky.square()
        self.linear = self.k_squared - NU * self.k_squared.square()
        cutoff = n / 3.0
        self.dealias = (
            (self.kx.abs() <= cutoff) & (self.ky.abs() <= cutoff)
        )
        self._build_coefficients()
        grid = torch.arange(n, device=device, dtype=self.real_dtype) * (
            DOMAIN_MAX / n
        )
        x, y = torch.meshgrid(grid, grid, indexing="ij")
        initial = initial_height(x, y)
        self.state = torch.fft.fft2(initial).to(self.complex_dtype)
        self.state[0, 0] = 0.0

    def _build_coefficients(self) -> None:
        dt = self.level.dt
        linear = self.linear
        self.exp_full = torch.exp(dt * linear)
        self.exp_half = torch.exp(0.5 * dt * linear)
        roots = torch.exp(
            1j
            * math.pi
            * (
                torch.arange(1, 33, device=self.device, dtype=self.real_dtype)
                - 0.5
            )
            / 32.0
        )
        lr = dt * linear[..., None].to(self.complex_dtype) + roots
        exp_lr = torch.exp(lr)
        self.q = dt * torch.real(
            torch.mean((torch.exp(lr / 2.0) - 1.0) / lr, dim=-1)
        )
        self.f1 = dt * torch.real(torch.mean(
            (-4.0 - lr + exp_lr * (4.0 - 3.0 * lr + lr.square()))
            / lr.pow(3),
            dim=-1,
        ))
        self.f2 = dt * torch.real(torch.mean(
            (2.0 + lr + exp_lr * (-2.0 + lr)) / lr.pow(3),
            dim=-1,
        ))
        self.f3 = dt * torch.real(torch.mean(
            (-4.0 - 3.0 * lr - lr.square() + exp_lr * (4.0 - lr))
            / lr.pow(3),
            dim=-1,
        ))

    def nonlinear(self, state: torch.Tensor) -> torch.Tensor:
        h_x = torch.fft.ifft2(1j * self.kx * state).real
        h_y = torch.fft.ifft2(1j * self.ky * state).real
        slope_squared = h_x.square() + h_y.square()
        value = (
            1j * self.kx * torch.fft.fft2(slope_squared * h_x)
            + 1j * self.ky * torch.fft.fft2(slope_squared * h_y)
        )
        return value * self.dealias

    def step(self) -> None:
        value = self.state
        nv = self.nonlinear(value)
        a = self.exp_half * value + self.q * nv
        na = self.nonlinear(a)
        b = self.exp_half * value + self.q * na
        nb = self.nonlinear(b)
        c = self.exp_half * a + self.q * (2.0 * nb - nv)
        nc = self.nonlinear(c)
        self.state = (
            self.exp_full * value
            + self.f1 * nv
            + 2.0 * self.f2 * (na + nb)
            + self.f3 * nc
        )
        self.state[0, 0] = 0.0

    def physical(self) -> torch.Tensor:
        return torch.fft.ifft2(self.state).real

    def diagnostics(self) -> dict[str, float]:
        height = self.physical()
        h_x = torch.fft.ifft2(1j * self.kx * self.state).real
        h_y = torch.fft.ifft2(1j * self.ky * self.state).real
        laplacian = torch.fft.ifft2(-self.k_squared * self.state).real
        energy = torch.mean(
            0.25 * (h_x.square() + h_y.square() - 1.0).square()
            + 0.5 * NU * laplacian.square()
        )
        return {
            "mass": float(height.mean().item()),
            "energy": float(energy.item()),
            "slope_rms": float(
                (h_x.square() + h_y.square()).mean().sqrt().item()
            ),
        }


def fixed_sample_indices(
    count: int,
    base_side: int,
    coarse_steps: int,
    seed: int,
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    return {
        "x": generator.integers(0, base_side, size=count, dtype=np.int64),
        "y": generator.integers(0, base_side, size=count, dtype=np.int64),
        "t": generator.integers(0, coarse_steps + 1, size=count, dtype=np.int64),
    }


def solve_level(
    level: Level,
    base_level: Level,
    indices: dict[str, np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, dict[str, dict[str, float]], float]:
    solver = ETDRK4MBE(level, device)
    spatial_ratio = level.side // base_level.side
    temporal_ratio = round(base_level.dt / level.dt)
    requested_steps = indices["t"] * temporal_ratio
    grouped = {
        int(step): np.flatnonzero(requested_steps == step)
        for step in np.unique(requested_steps)
    }
    values = np.empty(indices["t"].shape[0], dtype=np.float64)
    diagnostic_steps = {
        round(value / level.dt): value for value in DIAGNOSTIC_TIMES
    }
    diagnostics: dict[str, dict[str, float]] = {}

    def capture(step: int) -> None:
        selected = grouped.get(step)
        if selected is not None:
            height = solver.physical()
            x_index = torch.as_tensor(
                indices["x"][selected] * spatial_ratio,
                device=device,
                dtype=torch.long,
            )
            y_index = torch.as_tensor(
                indices["y"][selected] * spatial_ratio,
                device=device,
                dtype=torch.long,
            )
            values[selected] = height[x_index, y_index].detach().cpu().numpy()
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
        raise FloatingPointError(f"non-finite reference at N={level.side}")
    return values, diagnostics, elapsed


def relative_difference(left: np.ndarray, right: np.ndarray) -> float:
    numerator = np.sqrt(np.mean(np.square(left - right)))
    denominator = max(np.sqrt(np.mean(np.square(right))), 1e-14)
    return float(numerator / denominator)


def build_reference(args: argparse.Namespace) -> dict[str, Any]:
    levels = parse_levels(args.levels)
    if args.n_eval <= 0:
        raise ValueError("n_eval must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reference requested but unavailable")
    indices = fixed_sample_indices(
        args.n_eval, levels[0].side, levels[0].steps, args.eval_seed
    )
    level_values: list[np.ndarray] = []
    level_records: list[dict[str, Any]] = []
    fine_diagnostics: dict[str, dict[str, float]] = {}
    for level in levels:
        values, diagnostics, elapsed = solve_level(
            level, levels[0], indices, device
        )
        level_values.append(values)
        level_records.append({
            "side": level.side,
            "dt": level.dt,
            "steps": level.steps,
            "elapsed_seconds": elapsed,
            "mass_max_abs": max(
                abs(item["mass"]) for item in diagnostics.values()
            ),
            "energy_by_time": {
                key: value["energy"] for key, value in diagnostics.items()
            },
            "slope_rms_by_time": {
                key: value["slope_rms"] for key, value in diagnostics.items()
            },
        })
        fine_diagnostics = diagnostics

    coarse_medium = relative_difference(level_values[0], level_values[1])
    medium_fine = relative_difference(level_values[1], level_values[2])
    convergence_passed = bool(
        medium_fine <= args.tolerance
        and medium_fine <= 1.1 * max(coarse_medium, 1e-14)
    )
    fine_energies = np.array(
        [fine_diagnostics[f"{value:.2f}"]["energy"] for value in DIAGNOSTIC_TIMES]
    )
    energy_increase_max = float(np.maximum(np.diff(fine_energies), 0.0).max())
    convergence_passed = convergence_passed and energy_increase_max <= 1e-9

    base = levels[0]
    points = np.stack([
        indices["x"] * (DOMAIN_MAX / base.side),
        indices["y"] * (DOMAIN_MAX / base.side),
        indices["t"] * base.dt,
    ], axis=-1).astype(np.float32)
    metadata = {
        "protocol_id": REFERENCE_PROTOCOL_ID,
        "created_at": utc_now(),
        "equation": (
            "h_t = div((|grad h|^2-1)grad h) - nu*Delta^2 h"
        ),
        "nu": NU,
        "domain": "[0,2*pi]^2 x [0,1]",
        "initial_condition": (
            "0.2*cos(x)*cos(y) + 0.1*cos(2*x)*cos(y)"
        ),
        "method": "Fourier pseudospectral ETDRK4 with 2/3 dealiasing",
        "solver_dtype": "float64/complex128",
        "training_dtype_note": (
            "reference precision is independent; neural runs remain "
            "complex64 WAR and float32 real AD"
        ),
        "eval_seed": args.eval_seed,
        "n_eval": args.n_eval,
        "levels": level_records,
        "coarse_medium_relative_difference": coarse_medium,
        "medium_fine_relative_difference": medium_fine,
        "tolerance": args.tolerance,
        "energy_increase_max": energy_increase_max,
        "fine_diagnostics": fine_diagnostics,
        "convergence_passed": convergence_passed,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        points=points,
        values=level_values[-1].astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    os.replace(temporary, output)
    reference_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        **metadata,
        "reference_file": output.name,
        "reference_sha256": reference_sha256,
        "reference_bytes": output.stat().st_size,
        "device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
    }
    report_path = (
        args.report.resolve()
        if args.report is not None
        else output.with_suffix(".json")
    )
    atomic_write_json(report_path, report)
    marker = output.parent / "REFERENCE_COMPLETE"
    if convergence_passed:
        atomic_write_text(
            marker,
            f"completed_at={utc_now()} sha256={reference_sha256}\n",
        )
    else:
        atomic_write_text(
            output.parent / "REFERENCE_FAILED",
            f"completed_at={utc_now()} medium_fine={medium_fine}\n",
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--levels", default=DEFAULT_LEVELS)
    parser.add_argument("--n-eval", type=int, default=16384)
    parser.add_argument("--eval-seed", type=int, default=68421)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--device", default="auto")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_reference(args)
    print(json.dumps({
        "protocol_id": report["protocol_id"],
        "convergence_passed": report["convergence_passed"],
        "coarse_medium_relative_difference": report[
            "coarse_medium_relative_difference"
        ],
        "medium_fine_relative_difference": report[
            "medium_fine_relative_difference"
        ],
        "energy_increase_max": report["energy_increase_max"],
        "reference_sha256": report["reference_sha256"],
    }, sort_keys=True), flush=True)
    return 0 if report["convergence_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
