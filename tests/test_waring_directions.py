from __future__ import annotations

import itertools
import math
from collections import Counter

import torch

from apolarity.waring import monomial_waring_directions


def _basis(d: int, p: int):
    return itertools.combinations_with_replacement(range(d), p)


def _factorial(expanded):
    out = 1
    for count in Counter(expanded).values():
        out *= math.factorial(count)
    return out


def test_complex_waring_coefficient_filter_fp64():
    d = 5
    alpha = (0, 0, 1, 2)
    p = len(alpha)
    V, coeff, info = monomial_waring_directions(alpha, d, dtype=torch.complex128)
    assert info.rank == 6
    alpha_sorted = tuple(sorted(alpha))
    for beta in _basis(d, p):
        vals = torch.ones(V.shape[0], dtype=V.dtype)
        for idx, power in Counter(beta).items():
            vals = vals * (V[:, idx] ** power)
        got = (coeff * vals).sum() / _factorial(beta)
        expected = 1.0 if beta == alpha_sorted else 0.0
        assert abs(complex(got.item()) - expected) < 1e-12
