from __future__ import annotations

import torch

from apolarity.polarization import polarization_directions
from apolarity.waring import monomial_waring_directions


PAPER_PATTERNS = [
    ((0, 0, 0), 1, 2),
    ((0, 0, 1), 3, 3),
    ((0, 1, 2), 4, 4),
    ((0, 0, 0, 1), 4, 4),
    ((0, 0, 1, 1), 3, 4),
    ((0, 0, 1, 2), 6, 6),
    ((0, 1, 2, 3), 8, 8),
    ((0, 0, 0, 0, 1, 1), 5, 7),
    ((0, 0, 1, 1, 2, 2), 9, 13),
    ((0, 1, 2, 3, 4, 5), 32, 32),
]


def test_paper_pattern_direction_counts() -> None:
    for alpha, expected_complex, expected_polarization in PAPER_PATTERNS:
        d = max(alpha) + 1
        _, _, cinfo = monomial_waring_directions(alpha, d, dtype=torch.complex128)
        _, _, pinfo = polarization_directions(alpha, d, dtype=torch.float64)
        assert cinfo.rank == expected_complex
        assert pinfo.rank == expected_polarization
