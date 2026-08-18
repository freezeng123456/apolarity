#!/usr/bin/env python3
"""Numerical checks for the claims of ``docs/paper/theory_directions.tex``.

Each section below corresponds to a numbered statement of the note and prints
the numbers that appear there.  Run with no arguments::

    python experiments/tools/verify_theory_directions.py

The real-schedule search of Theorem 5.3 / Remark 5.5 is a nonconvex fit and is
the only slow part; pass ``--fast`` to skip it and reuse the recorded results.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from math import comb, factorial, prod
from pathlib import Path

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

Multi = tuple[int, ...]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def monomials(p: int, n_vars: int) -> list[Multi]:
    return [e for e in itertools.product(range(p + 1), repeat=n_vars) if sum(e) == p]


def multinomial(p: int, e: Multi) -> float:
    return factorial(p) / prod(factorial(x) for x in e)


def quadric_power(m: int, n_vars: int) -> dict[Multi, float]:
    """Coefficients of ``(z_1^2 + ... + z_N^2)^m``."""
    out: dict[Multi, float] = {}
    for ks in itertools.product(range(m + 1), repeat=n_vars):
        if sum(ks) != m:
            continue
        e = tuple(2 * k for k in ks)
        out[e] = out.get(e, 0.0) + factorial(m) / prod(factorial(k) for k in ks)
    return out


def expand_schedule(
    dirs: np.ndarray, coef: np.ndarray, p: int, n_vars: int
) -> dict[Multi, complex]:
    """Coefficients of ``sum_r coef_r (dirs_r . z)^p``."""
    out = {}
    for e in monomials(p, n_vars):
        mult = multinomial(p, e)
        out[e] = sum(
            coef[r] * mult * prod(dirs[r, j] ** e[j] for j in range(n_vars))
            for r in range(len(coef))
        )
    return out


def schedule_error(
    dirs: np.ndarray, coef: np.ndarray, target: dict[Multi, float], p: int, n_vars: int
) -> float:
    """Max deviation from ``p! * target``, the identity of Proposition 1.1."""
    got = expand_schedule(dirs, coef, p, n_vars)
    return max(
        abs(got[e] - factorial(p) * target.get(e, 0.0)) for e in monomials(p, n_vars)
    )


def monomial_rank(e: Multi) -> int:
    """Complex Waring rank of a monomial; drops the least exponent."""
    active = sorted(x for x in e if x > 0)
    return prod(x + 1 for x in active[1:]) if len(active) > 1 else 1


# --------------------------------------------------------------------------
# Theorem 2.1: the Fourier grid serves an arbitrary symbol
# --------------------------------------------------------------------------
def fourier_grid(
    target: dict[Multi, float], p: int, n_vars: int, base: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Directions and coefficients of Theorem 2.1.

    ``base`` overrides the least-degree coordinate, which Remark 2.2 needs in
    order to exhibit the aliasing that the correct choice rules out.
    """
    deg = [max(e[j] for e in target) for j in range(n_vars)]
    m = [d + 1 for d in deg]
    if base is None:
        base = int(np.argmin(m))
    others = [j for j in range(n_vars) if j != base]
    size = prod(m[j] for j in others)

    dirs, coef = [], []
    for ks in itertools.product(*[range(m[j]) for j in others]):
        zeta = {j: np.exp(2j * np.pi * k / m[j]) for j, k in zip(others, ks)}
        v = np.ones(n_vars, dtype=complex)
        for j in others:
            v[j] = zeta[j]
        c = 0j
        for g, b in target.items():
            if b == 0.0:
                continue
            term = prod(factorial(x) for x in g) * b
            for j in others:
                term *= zeta[j] ** (-g[j])
            c += term
        dirs.append(v)
        coef.append(c / size)
    return np.array(dirs), np.array(coef)


def check_theorem_grid(rng: np.random.Generator) -> None:
    print("Theorem 2.1  the Fourier grid is exact for an arbitrary symbol")
    print(
        f"  {'symbol':28s} {'p':>2} {'N':>2} {'grid':>6} {'generic min':>12} {'error':>10}"
    )
    cases: list[tuple[str, dict[Multi, float], int, int]] = [
        ("monomial z1^2 z2^2", {(2, 2): 1.0}, 4, 2),
        ("(z1^2+z2^2)^2", quadric_power(2, 2), 4, 2),
        ("random quartic", {e: float(rng.standard_normal()) for e in monomials(4, 2)}, 4, 2),
        ("random sextic", {e: float(rng.standard_normal()) for e in monomials(6, 2)}, 6, 2),
        ("random quartic", {e: float(rng.standard_normal()) for e in monomials(4, 3)}, 4, 3),
        ("random sextic", {e: float(rng.standard_normal()) for e in monomials(6, 3)}, 6, 3),
        ("random cubic", {e: float(rng.standard_normal()) for e in monomials(3, 4)}, 3, 4),
    ]
    for name, target, p, n_vars in cases:
        dirs, coef = fourier_grid(target, p, n_vars)
        err = schedule_error(dirs, coef, target, p, n_vars)
        gen = -(-comb(p + n_vars - 1, n_vars - 1) // n_vars)
        assert err < 1e-9, f"{name}: grid is not exact, error {err:.2e}"
        print(f"  {name:28s} {p:>2} {n_vars:>2} {len(coef):>6} {gen:>12} {err:>10.1e}")
    print()


def check_remark_base() -> None:
    """Remark 2.2: basing the grid on the wrong coordinate aliases."""
    print("Remark 2.2  the base coordinate must be one of least degree")
    target = {(4, 0): 1.0, (1, 3): 1.0}  # z0^4 + z0 z1^3
    p, n_vars = 4, 2
    for base, label in ((1, "least degree (correct)"), (0, "greatest degree (wrong)")):
        dirs, coef = fourier_grid(target, p, n_vars, base=base)
        err = schedule_error(dirs, coef, target, p, n_vars)
        got = expand_schedule(dirs, coef, p, n_vars)
        alias = abs(got[(0, 4)]) / factorial(p)
        print(f"  base = z{base}  {label:24s} error {err:>9.1e}   spurious z1^4 {alias:>5.2f}")
    print()


# --------------------------------------------------------------------------
# Section 3: what grid structure costs
# --------------------------------------------------------------------------
def catalecticant(target: dict[Multi, float], p: int, n_vars: int, q: int) -> np.ndarray:
    rows, cols = monomials(q, n_vars), monomials(p - q, n_vars)
    mat = np.zeros((len(rows), len(cols)))
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            mat[i, j] = target.get(tuple(a[k] + b[k] for k in range(n_vars)), 0.0)
    return mat


def initial_degree(target: dict[Multi, float], p: int, n_vars: int) -> int:
    for q in range(1, p + 1):
        mat = catalecticant(target, p, n_vars, q)
        if np.linalg.matrix_rank(mat, tol=1e-8) < mat.shape[0]:
            return q
    return p


def catalecticant_bound(target: dict[Multi, float], p: int, n_vars: int) -> int:
    return max(
        np.linalg.matrix_rank(catalecticant(target, p, n_vars, q), tol=1e-8)
        for q in range(p + 1)
    )


def check_price(rng: np.random.Generator) -> None:
    print("Lemma 3.3 and Corollary 3.4  the price of complete-intersection structure")
    print(
        f"  {'p':>2} {'N':>2} {'delta(sigma)':>13} {'floor(p/2)+1':>13} "
        f"{'CI bound':>9} {'generic rank':>13} {'ratio':>7} {'N!/2^(N-1)':>11}"
    )
    for n_vars in (2, 3, 4):
        for p in (4, 6, 8):
            target = {e: float(rng.standard_normal()) for e in monomials(p, n_vars)}
            delta = initial_degree(target, p, n_vars)
            predicted = p // 2 + 1
            assert delta == predicted, f"initial degree {delta} != {predicted}"
            ci = delta ** (n_vars - 1)
            gen = -(-comb(p + n_vars - 1, n_vars - 1) // n_vars)
            limit = factorial(n_vars) / 2 ** (n_vars - 1)
            print(
                f"  {p:>2} {n_vars:>2} {delta:>13} {predicted:>13} "
                f"{ci:>9} {gen:>13} {ci / gen:>7.2f} {limit:>11.2f}"
            )
    print()


# --------------------------------------------------------------------------
# Example 4.5: the icosahedral orbit
# --------------------------------------------------------------------------
def icosahedral_axes() -> np.ndarray:
    phi = (1 + np.sqrt(5)) / 2
    axes = np.array(
        [[0, 1, phi], [0, 1, -phi], [1, phi, 0], [1, -phi, 0], [phi, 0, 1], [-phi, 0, 1]],
        dtype=float,
    )
    return axes / np.linalg.norm(axes, axis=1, keepdims=True)


def icosahedral_group() -> list[np.ndarray]:
    """The 60 rotations, built from the 120 unit icosians to avoid drift."""
    phi = (1 + np.sqrt(5)) / 2
    quats: list[list[float]] = []
    for i in range(4):
        for s in (1.0, -1.0):
            q = [0.0] * 4
            q[i] = s
            quats.append(q)
    for signs in itertools.product((0.5, -0.5), repeat=4):
        quats.append(list(signs))
    base = [0.0, 0.5, 1 / (2 * phi), phi / 2]
    even = [
        p
        for p in itertools.permutations(range(4))
        if sum(p[a] > p[b] for a in range(4) for b in range(a + 1, 4)) % 2 == 0
    ]
    for perm in even:
        vals = [base[k] for k in perm]
        nz = [k for k in range(4) if abs(vals[k]) > 1e-12]
        for ss in itertools.product((1, -1), repeat=3):
            w = list(vals)
            for t, k in enumerate(nz):
                w[k] = vals[k] * ss[t]
            quats.append(w)

    rots: list[np.ndarray] = []
    for q in np.array(quats) / np.linalg.norm(np.array(quats), axis=1, keepdims=True):
        w, x, y, z = q
        rot = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )
        if not any(np.abs(rot - s).max() < 1e-8 for s in rots):
            rots.append(rot)
    return rots


def action_matrix(g: np.ndarray, p: int) -> np.ndarray:
    basis = monomials(p, 3)
    index = {e: i for i, e in enumerate(basis)}
    mat = np.zeros((len(basis), len(basis)))
    for j, e in enumerate(basis):
        cur = {(0, 0, 0): 1.0}
        for k in range(3):
            for _ in range(e[k]):
                nxt: dict[Multi, float] = {}
                for mono, co in cur.items():
                    for i in range(3):
                        m2 = list(mono)
                        m2[i] += 1
                        nxt[tuple(m2)] = nxt.get(tuple(m2), 0.0) + co * g[i, k]
                cur = nxt
        for mono, co in cur.items():
            mat[index[mono], j] += co
    return mat


def check_abelian_orbit() -> None:
    """Example 4.4: the character-weighted grid orbit average is a monomial.

    Checks ``A^{chi_a}_v = binom(p, a) z^a``, which is what makes the isotypic
    component one-dimensional and so recovers Theorem 2.1 from Theorem 4.3.
    """
    print("Example 4.4  the Fourier grid as a character-weighted abelian orbit")
    print(f"  {'exponent a':16s} {'p':>2} {'orbit':>6} {'binom(p,a)':>11} {'error':>10}")
    for a in [(1, 2), (2, 2), (1, 1, 2), (2, 1, 3), (1, 2, 2)]:
        p = sum(a)
        n_vars = len(a)
        m = [x + 1 for x in a]
        base = int(np.argmin(m))
        others = [j for j in range(n_vars) if j != base]
        size = prod(m[j] for j in others)

        avg: dict[Multi, complex] = {e: 0j for e in monomials(p, n_vars)}
        for ks in itertools.product(*[range(m[j]) for j in others]):
            zeta = {j: np.exp(2j * np.pi * k / m[j]) for j, k in zip(others, ks)}
            v = np.ones(n_vars, dtype=complex)
            for j in others:
                v[j] = zeta[j]
            char = prod(zeta[j] ** a[j] for j in others)
            for e in avg:
                avg[e] += (
                    np.conj(char)
                    * multinomial(p, e)
                    * prod(v[j] ** e[j] for j in range(n_vars))
                    / size
                )
        want = {e: (multinomial(p, a) if e == a else 0.0) for e in avg}
        err = max(abs(avg[e] - want[e]) for e in avg)
        assert err < 1e-9, f"a={a}: orbit average is not a monomial, error {err:.2e}"
        print(
            f"  {str(a):16s} {p:>2} {size:>6} {multinomial(p, a):>11.0f} {err:>10.1e}"
        )
    print()


def check_orbit_filter() -> None:
    print("Theorem 4.3 and Example 4.5  a nonabelian orbit filter")
    group = icosahedral_group()
    assert len(group) == 60, f"group order {len(group)} != 60"
    print(f"  icosahedral rotation group order: {len(group)}")
    print(f"  {'degree p':>9} {'dim R[z]_p':>11} {'dim invariants':>15}")
    expected = {2: 1, 4: 1, 6: 2, 8: 2, 10: 3}
    for p in (2, 4, 6, 8, 10):
        reynolds = sum(action_matrix(g, p) for g in group) / len(group)
        dim = int(round(np.trace(reynolds)))
        assert dim == expected[p], f"degree {p}: invariant dim {dim} != {expected[p]}"
        print(f"  {p:>9} {len(monomials(p, 3)):>11} {dim:>15}")

    axes = icosahedral_axes()
    for m in (1, 2):
        p = 2 * m
        target = quadric_power(m, 3)
        coef = np.ones(len(axes))
        got = expand_schedule(axes, coef, p, 3)
        scale = factorial(p) * target[(2 * m, 0, 0)] / got[(2 * m, 0, 0)]
        err = schedule_error(axes, coef * scale, target, p, 3)
        assert err < 1e-10, f"icosahedral schedule for Delta^{m}: error {err:.2e}"
        print(
            f"  Delta^{m} on R^3 from 6 icosahedral axes, equal weight "
            f"{scale:.6f}: error {err:.1e}"
        )
    print()


# --------------------------------------------------------------------------
# Theorem 5.3 and Remark 5.5: minimal real schedules
# --------------------------------------------------------------------------
def fit_real_schedule(
    target: dict[Multi, float],
    p: int,
    n_vars: int,
    size: int,
    restarts: int = 8,
    steps: int = 2500,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    basis = monomials(p, n_vars)
    mult = torch.tensor([multinomial(p, e) for e in basis])
    tgt = torch.tensor([target.get(e, 0.0) for e in basis])
    # Fit the normalized symbol so that the tolerance means the same thing at
    # every order; the discarded scale is put back into the weights at the end.
    scale = float(tgt.abs().max())
    tgt = tgt / scale
    exps = torch.tensor(basis, dtype=torch.float64)

    def design(dirs: torch.Tensor) -> torch.Tensor:
        """Rows are ``mult * (v_r . z)^p`` in the monomial basis."""
        unit = dirs / dirs.norm(dim=1, keepdim=True)
        mag = torch.exp(torch.log(unit.abs().clamp_min(1e-300)) @ exps.T)
        sign = torch.prod(torch.sign(unit).unsqueeze(1) ** exps.unsqueeze(0), dim=2)
        return mult * mag * sign

    def residual(dirs: torch.Tensor) -> torch.Tensor:
        """Distance from the target to the span of the current directions.

        The weights enter the fit linearly, so they are eliminated by an
        orthogonal projection rather than optimized.  What remains is a much
        better conditioned problem in the directions alone.
        """
        mat = design(dirs).T
        basis_q, _ = torch.linalg.qr(mat, mode="reduced")
        return tgt - basis_q @ (basis_q.T @ tgt)

    best, best_dirs = float("inf"), None
    for seed in range(restarts):
        gen = torch.Generator().manual_seed(1000 * size + seed)
        dirs = torch.randn(size, n_vars, generator=gen, requires_grad=True)

        adam = torch.optim.Adam([dirs], lr=5e-2)
        for step in range(steps):
            adam.zero_grad()
            (residual(dirs) ** 2).sum().backward()
            adam.step()
            if step == steps // 2:
                for group in adam.param_groups:
                    group["lr"] = 5e-3

        lbfgs = torch.optim.LBFGS(
            [dirs], max_iter=500, tolerance_grad=1e-16, line_search_fn="strong_wolfe"
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad()
            loss = (residual(dirs) ** 2).sum()
            loss.backward()
            return loss

        lbfgs.step(closure)
        with torch.no_grad():
            res = float(residual(dirs).abs().max())
        if res < best:
            best = res
            best_dirs = (dirs / dirs.norm(dim=1, keepdim=True)).detach().clone()
        if best < 1e-12:
            break

    assert best_dirs is not None
    with torch.no_grad():
        mat = design(best_dirs).T
        wts = torch.linalg.lstsq(mat, tgt).solution * scale * factorial(p)
    return best, best_dirs, wts


def termwise_total(target: dict[Multi, float]) -> int:
    return sum(monomial_rank(e) for e, b in target.items() if b != 0.0)


def check_real_search(fast: bool) -> None:
    print("Theorem 5.3, Corollary 5.4 and Remark 5.5  shortest real schedules")
    print(
        f"  {'target':10s} {'N':>2} {'grid':>6} {'term-by-term':>13} "
        f"{'bound':>6} {'search':>7} {'weights':>12}"
    )
    recorded = {(3, 1): 3, (3, 2): 6, (3, 3): 11, (4, 1): 4, (4, 2): 11}
    for n_vars in (3, 4):
        for m in (1, 2, 3):
            if n_vars == 4 and m == 3:
                continue
            p = 2 * m
            target = quadric_power(m, n_vars)
            grid = (2 * m + 1) ** (n_vars - 1)
            term = termwise_total(target)
            bound = comb(n_vars + m - 1, m)
            if fast:
                size, note = recorded[(n_vars, m)], "recorded"
            else:
                size, note = None, ""
                for cand in range(bound, bound + 4):
                    res, dirs, wts = fit_real_schedule(target, p, n_vars, cand)
                    if res < 1e-9:
                        # Confirm end to end against the identity of Proposition 1.1.
                        err = schedule_error(
                            dirs.numpy(), wts.numpy(), target, p, n_vars
                        )
                        assert err < 1e-6, f"Delta^{m}, N={n_vars}: error {err:.2e}"
                        size = cand
                        note = "positive" if float(wts.min()) > 0 else "mixed sign"
                        break
                assert size is not None, f"no real schedule found for Delta^{m}, N={n_vars}"
                assert size >= bound, "search beat the lower bound of Theorem 5.3"
                assert size == recorded[(n_vars, m)], (
                    f"Delta^{m}, N={n_vars}: found {size}, note records "
                    f"{recorded[(n_vars, m)]}"
                )
            print(
                f"  {'Delta^' + str(m):10s} {n_vars:>2} {grid:>6} {term:>13} "
                f"{bound:>6} {size:>7} {note:>12}"
            )
    print()


# --------------------------------------------------------------------------
# Corollary 6.3 and Remark 6.4: shared sets and conditioning
# --------------------------------------------------------------------------
def check_implemented_rules() -> None:
    """Table 7.2: the rules that ``apolarity.cubature`` actually builds."""
    root = Path(__file__).resolve().parents[2] / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from apolarity.cubature import laplacian_power_cubature_directions
    except ImportError:  # pragma: no cover - only when run outside the repo
        print("Table 7.2  skipped: apolarity is not importable\n")
        return

    print("Table 7.2  rules built by apolarity.cubature")
    print(
        f"  {'d':>2} {'m':>2} {'nodes':>6} {'bound':>6} {'term-by-term':>13} "
        f"{'positive':>9}  rule"
    )
    recorded = {
        (3, 1): 3, (3, 2): 6, (3, 3): 13, (3, 4): 25,
        (4, 1): 4, (4, 2): 12, (4, 3): 24, (4, 4): 64,
        (5, 1): 5, (5, 2): 21, (5, 3): 41, (5, 4): 121,
        (6, 1): 6, (6, 2): 36, (6, 3): 68, (6, 4): 208,
    }
    for (d, m), want in sorted(recorded.items()):
        _nodes, _coeff, info = laplacian_power_cubature_directions(m, d)
        assert info.nodes == want, f"d={d}, m={m}: {info.nodes} nodes, note records {want}"
        assert info.nodes >= info.lower_bound, "a rule came in below the lower bound"
        star = " *" if info.meets_lower_bound else ""
        print(
            f"  {d:>2} {m:>2} {info.nodes:>6} {info.lower_bound:>6} "
            f"{info.termwise_rank:>13} {str(info.weights_positive):>9}  {info.rule}{star}"
        )
    print()


def veronese(dirs: np.ndarray, p: int, n_vars: int) -> np.ndarray:
    basis = monomials(p, n_vars)
    mat = np.zeros((len(dirs), len(basis)), dtype=dirs.dtype)
    for r, v in enumerate(dirs):
        for j, e in enumerate(basis):
            mat[r, j] = multinomial(p, e) * prod(v[k] ** e[k] for k in range(n_vars))
    return mat


def check_shared(rng: np.random.Generator) -> None:
    print("Corollary 6.3  one shared set for the whole order-p derivative tensor")
    print(
        f"  {'p':>2} {'N':>2} {'dim':>5} {'term-by-term':>13} "
        f"{'shared':>7} {'ratio':>6} {'recon error':>12}"
    )
    for n_vars in (2, 3, 4):
        for p in (4, 6):
            basis = monomials(p, n_vars)
            dim = len(basis)
            term = sum(monomial_rank(e) for e in basis)
            dirs = rng.standard_normal((dim, n_vars))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            mat = veronese(dirs, p, n_vars)
            err = 0.0
            for _ in range(200):
                rhs = rng.standard_normal(dim)
                sol = np.linalg.solve(mat.T, rhs)
                err = max(err, float(np.abs(mat.T @ sol - rhs).max()))
            assert err < 1e-6, f"shared set failed to reconstruct, error {err:.2e}"
            print(
                f"  {p:>2} {n_vars:>2} {dim:>5} {term:>13} {dim:>7} "
                f"{term / dim:>6.1f} {err:>12.1e}"
            )
    print()

    print("Remark 6.4  length against conditioning")
    print(f"  {'p':>2} {'N':>2} {'minimal':>8} {'cond':>10} {'grid':>6} {'cond':>10}")
    for n_vars in (2, 3):
        for p in (4, 6):
            basis = monomials(p, n_vars)
            dim = len(basis)
            dirs = rng.standard_normal((dim, n_vars))
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            cond_min = np.linalg.cond(veronese(dirs, p, n_vars))
            uniform = {e: 1.0 for e in basis}
            grid, _ = fourier_grid(uniform, p, n_vars)
            svals = np.linalg.svd(veronese(grid, p, n_vars), compute_uv=False)
            cond_grid = svals[0] / svals[dim - 1]
            print(
                f"  {p:>2} {n_vars:>2} {dim:>8} {cond_min:>10.1e} "
                f"{len(grid):>6} {cond_grid:>10.1e}"
            )
    print()


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip the nonconvex real-schedule search and print recorded sizes",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    check_theorem_grid(rng)
    check_remark_base()
    check_price(np.random.default_rng(1))
    check_abelian_orbit()
    check_orbit_filter()
    check_real_search(args.fast)
    check_implemented_rules()
    check_shared(np.random.default_rng(7))
    print("all checks passed")


if __name__ == "__main__":
    main()
