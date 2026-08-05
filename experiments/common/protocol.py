"""Single source of truth for preregistered JSC experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from boundary_weights import BOUNDARY_WEIGHTS, PROFILE_ID, weights_for
from osc_common import ArchitectureSpec, FORMAL_VARIANTS, formal_architecture_specs


# jsc_v2 is the completed fixed-bc_weight=100 bundle.  The next run changes the
# loss weighting, so it gets a new protocol id and result root rather than
# silently mixing incompatible rows with the old results.
PROTOCOL_ID = "jsc_v3"
LEGACY_PROTOCOL_ID = "jsc_v2"
BOUNDARY_PROFILE_ID = PROFILE_ID
FORMAL_WIDTH = 128
FORMAL_METHODS = FORMAL_VARIANTS
SEEDS = (0, 1, 2, 3, 4)
BUDGET_SECONDS = 1200.0
DEPTH = 4
N_INTERIOR = 4096
N_BOUNDARY = 512
LEARNING_RATE = 1e-3
LEARNING_RATE_FINAL = 1e-4
LEARNING_RATE_SCHEDULE = "cosine"
COLLOCATION_PROTOCOL = "paired_seed_v1"
EVALUATION_PROTOCOL = "fixed_seed_12345_n8192_v1"
HISTORY_EVERY_STEPS = 20
HISTORY_EVAL_N = 4096

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "experiments" / "results" / PROTOCOL_ID


@dataclass(frozen=True)
class AtomicTask:
    family: str
    task_id: str
    dimension: int
    order: int
    sweep: int
    omega0: float
    fourier_sigma: float
    split_real_baselines: bool
    boundary_weights: tuple[float, ...]

    @property
    def setting(self) -> str:
        if self.family == "poly":
            return f"d{self.dimension}_o{self.order}"
        return f"a{self.sweep}"

    def specs(self) -> dict[str, ArchitectureSpec]:
        return formal_architecture_specs(
            self.dimension,
            depth=DEPTH,
            literal_width=FORMAL_WIDTH,
            split_real_baselines=self.split_real_baselines,
            omega0=self.omega0,
            fourier_sigma=self.fourier_sigma,
        )


def poly_task(dimension: int, order: int) -> AtomicTask:
    if dimension not in (2, 3):
        raise ValueError("jsc_v3 Poly dimension must be 2 or 3")
    if order not in (2, 4, 6):
        raise ValueError("jsc_v3 Poly order must be 2, 4, or 6")
    return AtomicTask(
        family="poly",
        task_id=f"poly_d{dimension}_o{order}",
        dimension=dimension,
        order=order,
        sweep=order,
        omega0=2.0 * math.pi,
        fourier_sigma=math.pi,
        split_real_baselines=False,
        boundary_weights=weights_for(f"poly_d{dimension}_o{order}"),
    )


def chirp_task(a: int) -> AtomicTask:
    if a not in (1, 2, 3):
        raise ValueError("jsc_v3 Chirp a must be 1, 2, or 3")
    return AtomicTask(
        family="chirp",
        task_id=f"chirp_a{a}",
        dimension=2,
        order=2,
        sweep=a,
        omega0=max(10.0, 2.0 * math.pi * a),
        fourier_sigma=max(2.0, math.pi * a),
        split_real_baselines=False,
        boundary_weights=weights_for(f"chirp_a{a}"),
    )


def maxwell_task(a: int) -> AtomicTask:
    if a not in (2, 4, 6):
        raise ValueError("jsc_v3 Maxwell a must be 2, 4, or 6")
    return AtomicTask(
        family="maxwell",
        task_id=f"maxwell_a{a}",
        dimension=2,
        order=2,
        sweep=a,
        omega0=max(10.0, 2.0 * math.pi * a),
        fourier_sigma=max(2.0, math.pi * a),
        split_real_baselines=True,
        boundary_weights=weights_for(f"maxwell_a{a}"),
    )


def get_task(
    family: str,
    *,
    dimension: int | None = None,
    order: int | None = None,
    sweep: int | None = None,
) -> AtomicTask:
    if family == "poly":
        if dimension is None or order is None or sweep is not None:
            raise ValueError("Poly requires exactly --dim and --order")
        return poly_task(dimension, order)
    if family == "chirp":
        if sweep is None or dimension is not None or order is not None:
            raise ValueError("Chirp requires exactly --sweep")
        return chirp_task(sweep)
    if family == "maxwell":
        if sweep is None or dimension is not None or order is not None:
            raise ValueError("Maxwell requires exactly --sweep")
        return maxwell_task(sweep)
    raise ValueError("family must be one of: poly, chirp, maxwell")


def all_tasks() -> tuple[AtomicTask, ...]:
    return (
        *(poly_task(d, order) for d in (2, 3) for order in (2, 4, 6)),
        *(chirp_task(a) for a in (1, 2, 3)),
        *(maxwell_task(a) for a in (2, 4, 6)),
    )


def validate_architecture_table(task: AtomicTask) -> dict[str, ArchitectureSpec]:
    specs = task.specs()
    if set(specs) != set(FORMAL_METHODS):
        raise ValueError("architecture table does not contain the four formal methods")
    for spec in specs.values():
        if spec.width != FORMAL_WIDTH:
            raise ValueError(
                f"{PROTOCOL_ID} requires literal H={FORMAL_WIDTH}; "
                f"{spec.method} has H={spec.width}"
            )
    return specs
