"""Frozen power-of-ten boundary-weight profile for the next formal run.

The values are selected from the archived Poly/Chirp/Maxwell searches.  They
are deliberately rounded to powers of ten so the next protocol documents a
stable, interpretable choice rather than claiming a finely tuned optimum.
"""

from __future__ import annotations

import math


PROFILE_ID = "pow10_reasonable_v1"

# Poly entries are ordered as [u, Delta u, Delta^2 u, ...].  Chirp and Maxwell
# have one Dirichlet term, so their entries contain one scalar.
BOUNDARY_WEIGHTS: dict[str, tuple[float, ...]] = {
    "poly_d2_o2": (0.1,),
    "poly_d2_o4": (0.1, 10.0),
    "poly_d2_o6": (0.01, 1.0, 10.0),
    "poly_d3_o2": (0.1,),
    "poly_d3_o4": (0.1, 1.0),
    "poly_d3_o6": (0.1, 0.1, 1.0),
    "chirp_a1": (1.0,),
    "chirp_a2": (0.1,),
    "chirp_a3": (0.01,),
    "maxwell_a2": (0.1,),
    "maxwell_a4": (0.1,),
    "maxwell_a6": (0.01,),
}


def weights_for(task_id: str) -> tuple[float, ...]:
    try:
        return BOUNDARY_WEIGHTS[task_id]
    except KeyError as exc:
        raise ValueError(f"no {PROFILE_ID} boundary weights for {task_id}") from exc


def parse_weights(text: str) -> tuple[float, ...]:
    """Parse a comma-separated weight vector and require powers of ten."""
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("at least one boundary weight is required")
    for value in values:
        if value <= 0.0 or not math.isfinite(value):
            raise ValueError("boundary weights must be finite and positive")
        exponent = math.log10(value)
        if not math.isclose(exponent, round(exponent), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"boundary weight {value:g} is not an exact 10^k value")
    return values


def format_weights(values: tuple[float, ...]) -> str:
    return ",".join(f"{value:g}" for value in values)
