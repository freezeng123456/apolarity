"""Directional schedules for the polyharmonic operator in any dimension.

:mod:`apolarity.symbol` schedules ``Delta^m`` optimally in two variables, where
the symbol ``(z_1^2 + z_2^2)^m`` factors into linear forms and is therefore a
monomial in disguise.  That argument fails for ``d >= 3``: the quadric is
irreducible, so no change of variables turns the symbol into a monomial and the
closed-form rank formula does not apply.

A different equivalence covers every dimension.  Writing ``q(z) = z . z``, a
family of unit directions ``v_r`` and real weights ``w_r`` satisfies

    sum_r w_r (v_r . z)^{2m} = kappa * q(z)^m   for all z

if and only if the rule ``(v_r, w_r)`` integrates every polynomial of even
degree at most ``2m`` exactly over the unit sphere.  The reason is that the
functions ``v -> (v . z)^{2m}`` span the whole space of forms of degree ``2m``
as ``z`` varies, so exactness on that family is exactness on their span.  A real
directional schedule for ``Delta^m`` is therefore precisely a spherical cubature
rule, which buys three things the roots-of-unity construction does not offer:
the nodes are real, the weights are usually positive, and classical rules are
far shorter than either the term-by-term expansion or a coordinate grid.

No schedule can be shorter than ``binom(d + m - 1, m)``.  That bound comes from
the middle catalecticant of ``q^m``, which is positive definite and hence of
full rank, so the image of the decomposition must already span a space of that
dimension.  The bound is attained here for ``m = 1`` in every dimension and for
``(d, m) = (3, 2)`` by the icosahedral axes.

Three rules are used, in order of preference:

* ``m = 1``: an orthonormal frame, of length ``d``, which meets the bound.
* ``(d, m) = (3, 2)``: the six icosahedral axes with equal weights.  A single
  orbit suffices because the icosahedral group has a one-dimensional space of
  invariant quartics, and its length ``6`` meets the bound.
* general ``(d, m)``: orbits of the group of signed permutations.  An orbit sum
  is automatically invariant under that group, so matching ``q^m`` reduces to
  one linear condition per invariant of degree ``2m``, and the weights follow
  from a linear solve rather than a search.

Every schedule is verified against the defining polynomial identity before it is
returned, so a rule that does not reproduce the operator exactly raises instead
of silently returning an approximation.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import Tensor

from .symbol import (
    laplacian_power_directions,
    laplacian_power_termwise_rank,
    quadric_power_coefficients,
)

Exponent = tuple[int, ...]


@dataclass(frozen=True)
class CubatureInfo:
    """Metadata for a polyharmonic directional schedule."""

    operator: str
    order: int
    dimension: int
    nodes: int
    lower_bound: int
    termwise_rank: int
    rule: str
    weights_positive: bool

    @property
    def meets_lower_bound(self) -> bool:
        return self.nodes == self.lower_bound


def laplacian_power_lower_bound(m: int, d: int) -> int:
    """Least possible number of directions for ``Delta^m`` on ``R^d``.

    Equal to ``dim R[y]_m``, the rank of the middle catalecticant of ``q^m``.
    The bound holds over the complex numbers and so a fortiori over the reals.
    """
    if m < 1 or d < 1:
        raise ValueError("m and d must be positive")
    return math.comb(d + m - 1, m)


# --------------------------------------------------------------------------
# symbol bookkeeping
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _monomials(p: int, d: int) -> tuple[Exponent, ...]:
    return tuple(
        e for e in itertools.product(range(p + 1), repeat=d) if sum(e) == p
    )


def _multinomial(p: int, e: Exponent) -> float:
    return math.factorial(p) / math.prod(math.factorial(x) for x in e)


def _expand(nodes: torch.Tensor, coeff: torch.Tensor, p: int) -> torch.Tensor:
    """Coefficients of ``sum_r coeff_r (nodes_r . z)^p`` in the monomial basis."""
    d = nodes.shape[1]
    basis = _monomials(p, d)
    exps = torch.tensor(basis, dtype=torch.float64)
    # Signs are tracked separately so that zero components raised to zero powers
    # do not go through a logarithm.
    magnitude = torch.exp(torch.log(nodes.abs().clamp_min(1e-300)) @ exps.T)
    sign = torch.prod(torch.sign(nodes).unsqueeze(1) ** exps.unsqueeze(0), dim=2)
    mult = torch.tensor([_multinomial(p, e) for e in basis], dtype=torch.float64)
    return (coeff.unsqueeze(1) * mult * magnitude * sign).sum(0)


def _target(m: int, d: int) -> torch.Tensor:
    """The vector ``p! * sigma`` that a schedule for ``Delta^m`` must reproduce."""
    p = 2 * m
    coeffs = quadric_power_coefficients(m, d)
    return torch.tensor(
        [math.factorial(p) * coeffs.get(e, 0.0) for e in _monomials(p, d)],
        dtype=torch.float64,
    )


def _relative_error(nodes: torch.Tensor, coeff: torch.Tensor, m: int, d: int) -> float:
    target = _target(m, d)
    got = _expand(nodes, coeff, 2 * m)
    return float((got - target).abs().max() / target.abs().max())


# --------------------------------------------------------------------------
# the individual rules
# --------------------------------------------------------------------------
def _orthonormal_frame(d: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``Delta = sum_i partial_i^2`` needs the ``d`` coordinate axes."""
    return torch.eye(d, dtype=torch.float64), torch.full((d,), 2.0, dtype=torch.float64)


def _icosahedral_axes() -> torch.Tensor:
    """Six axes of a regular icosahedron, one per antipodal pair of vertices."""
    phi = (1 + math.sqrt(5)) / 2
    raw = torch.tensor(
        [
            [0.0, 1.0, phi],
            [0.0, 1.0, -phi],
            [1.0, phi, 0.0],
            [1.0, -phi, 0.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    return raw / raw.norm(dim=1, keepdim=True)


#: Largest entry allowed in an orbit-generating integer pattern.
_MAX_PATTERN_ENTRY = 3
#: Orbit families considered, smallest first.  Caps the subset enumeration.
_MAX_FAMILIES = 12
#: Largest number of orbits combined into one rule.
_MAX_ORBITS = 5


@lru_cache(maxsize=None)
def _orbit_patterns(d: int) -> tuple[tuple[int, ...], ...]:
    """Nonincreasing positive integer patterns of length at most ``d``.

    A pattern is padded with zeros, normalized and used as an orbit
    representative.  Patterns with a common factor are dropped because they
    generate the same orbit as the reduced pattern.
    """
    out: list[tuple[int, ...]] = []
    for length in range(1, d + 1):
        for pattern in itertools.combinations_with_replacement(
            range(1, _MAX_PATTERN_ENTRY + 1), length
        ):
            if math.gcd(*pattern, 0) != 1:
                continue
            out.append(tuple(sorted(pattern, reverse=True)))
    return tuple(out)


@lru_cache(maxsize=None)
def _orbit(pattern: tuple[int, ...], d: int) -> tuple[tuple[float, ...], ...]:
    """Orbit of a normalized pattern under signed permutations of coordinates.

    Antipodal members are identified, since the order ``2m`` is even and
    ``(-v . z)^{2m} = (v . z)^{2m}``.
    """
    padded = list(pattern) + [0] * (d - len(pattern))
    norm = math.sqrt(sum(x * x for x in padded))
    seen: set[tuple[float, ...]] = set()
    out: list[tuple[float, ...]] = []
    for permuted in set(itertools.permutations(padded)):
        support = [i for i, x in enumerate(permuted) if x != 0]
        for signs in itertools.product((1.0, -1.0), repeat=len(support)):
            vec = [0.0] * d
            for i, sign in zip(support, signs):
                vec[i] = sign * permuted[i] / norm
            key = tuple(vec)
            if key in seen or tuple(-x for x in vec) in seen:
                continue
            seen.add(key)
            out.append(key)
    return tuple(out)


@lru_cache(maxsize=None)
def _orbit_rule(m: int, d: int) -> tuple[tuple[tuple[int, ...], ...], tuple[float, ...]]:
    """Choose orbits of signed permutations and solve linearly for their weights.

    An orbit sum is invariant under the group, so reproducing ``q^m`` imposes one
    condition per invariant of degree ``2m`` and the weights follow from a linear
    solve.  Only the choice of orbits is combinatorial, and it is resolved by
    taking the fewest nodes, with positive weights breaking ties because they
    keep the assembled sum free of cancellation.
    """
    p = 2 * m
    target = _target(m, d)
    patterns = sorted(_orbit_patterns(d), key=lambda pat: len(_orbit(pat, d)))
    patterns = patterns[:_MAX_FAMILIES]
    orbits = {pat: torch.tensor(_orbit(pat, d), dtype=torch.float64) for pat in patterns}
    columns = {
        pat: _expand(orbit, torch.ones(len(orbit), dtype=torch.float64), p)
        for pat, orbit in orbits.items()
    }

    best: tuple[int, bool, tuple[tuple[int, ...], ...], torch.Tensor] | None = None
    for size in range(1, min(_MAX_ORBITS, len(patterns)) + 1):
        for pick in itertools.combinations(patterns, size):
            nodes = sum(len(orbits[pat]) for pat in pick)
            if best is not None and nodes > best[0]:
                continue
            mat = torch.stack([columns[pat] for pat in pick], dim=1)
            weights = torch.linalg.lstsq(mat, target).solution
            residual = float((mat @ weights - target).abs().max() / target.abs().max())
            if residual > 1e-11:
                continue
            positive = bool((weights > -1e-12).all())
            if best is None or (nodes, not positive) < (best[0], not best[1]):
                best = (nodes, positive, pick, weights)

    if best is None:
        raise NotImplementedError(
            f"no signed-permutation rule of degree {p} was found in dimension {d}; "
            f"degree {p} imposes more invariant conditions than the orbits "
            "considered here can satisfy, so this case needs a rule from a "
            "larger family"
        )
    _nodes, _positive, pick, weights = best
    return pick, tuple(float(w) for w in weights)


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _build(m: int, d: int) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Nodes, coefficients and rule name for ``Delta^m`` on ``R^d``, in float64."""
    if m == 1:
        nodes, coeff = _orthonormal_frame(d)
        return nodes, coeff, "orthonormal frame"

    if d == 2:
        # The two-variable symbol factors into linear forms, so the closed-form
        # schedule of Corollary 3.7 applies and is already minimal at m+1.
        nodes, coeff, _info = laplacian_power_directions(m, 2, dtype=torch.float64)
        return nodes, coeff, "equally spaced axes"

    if (d, m) == (3, 2):
        nodes = _icosahedral_axes()
        # One orbit spans the invariant quartics, so a single scalar is free.
        ones = torch.ones(len(nodes), dtype=torch.float64)
        scale = _target(m, d)[0] / _expand(nodes, ones, 2 * m)[0]
        return nodes, ones * scale, "icosahedral orbit"

    pick, weights = _orbit_rule(m, d)
    node_blocks, coeff_blocks = [], []
    for pattern, weight in zip(pick, weights):
        orbit = torch.tensor(_orbit(pattern, d), dtype=torch.float64)
        node_blocks.append(orbit)
        coeff_blocks.append(torch.full((len(orbit),), weight, dtype=torch.float64))
    shown = ", ".join("".join(str(x) for x in pat) for pat in pick)
    return torch.cat(node_blocks), torch.cat(coeff_blocks), f"orbits [{shown}]"


def laplacian_power_cubature_directions(
    m: int,
    d: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor, CubatureInfo]:
    """Real directional schedule for ``Delta^m`` on ``R^d``.

    Args:
        m: Power of the Laplacian; the operator order is ``p = 2m``.
        d: Ambient dimension.  Any ``d >= 1`` is accepted, subject to a rule
            being available for the requested order.
        device, dtype: Tensor placement.  The nodes are real, so a real dtype is
            admissible; a complex dtype is accepted for models whose parameters
            are complex.

    Returns:
        ``(V, coeff, info)`` with ``V`` of shape ``(R, d)`` and ``coeff`` of
        shape ``(R,)``, satisfying

            Delta^m u(x) = sum_r coeff[r] * T_{2m}(x; V[r]).

    Raises:
        NotImplementedError: if no rule of the required degree is available in
            this dimension.
        RuntimeError: if the assembled rule fails to reproduce the symbol, which
            would indicate a defect in the construction rather than in the call.
    """
    if m < 1:
        raise ValueError("m must be at least one")
    if d < 1:
        raise ValueError("d must be at least one")

    nodes, coeff, rule = _build(m, d)

    error = _relative_error(nodes, coeff, m, d)
    if error > 1e-10:
        raise RuntimeError(
            f"the {rule} rule for Delta^{m} on R^{d} reproduces the symbol only "
            f"to relative error {error:.2e}; refusing to return it"
        )

    info = CubatureInfo(
        operator=f"Delta^{m}",
        order=2 * m,
        dimension=d,
        nodes=int(nodes.shape[0]),
        lower_bound=laplacian_power_lower_bound(m, d),
        termwise_rank=laplacian_power_termwise_rank(m, d),
        rule=rule,
        weights_positive=bool((coeff > 0).all()),
    )
    return (
        nodes.to(device=device, dtype=dtype),
        coeff.to(device=device, dtype=dtype),
        info,
    )


__all__ = [
    "CubatureInfo",
    "laplacian_power_cubature_directions",
    "laplacian_power_lower_bound",
    "laplacian_power_termwise_rank",
]
