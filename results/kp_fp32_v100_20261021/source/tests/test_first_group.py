import inspect

import numpy as np
import pytest
import torch

import first_group
from torch_pinn.model import sample_normal
from torch_pinn.monomials import schedule


@pytest.mark.parametrize('pattern,rank,dtype', [
    ('111111', 1, torch.float32), ('111222', 4, torch.complex64),
    ('112233', 9, torch.complex64), ('123456', 32, torch.float32),
    ('11223344', 27, torch.complex64), ('111222333', 16, torch.complex64),
])
def test_first_group_schedule_metadata(pattern, rank, dtype):
    alpha = first_group.parse_one_based_pattern(pattern)
    directions, weights = schedule(alpha, 16, dtype=torch.float32)
    assert directions.shape == (rank, 16)
    assert directions.dtype == weights.dtype == dtype


def test_first_group_normal_sampler_preserves_historical_numpy_convention():
    actual = sample_normal(31, 3, 4)
    expected = (np.random.default_rng(31).standard_normal((3, 4))*.35).astype(np.float32)
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=0, atol=0)


def test_formal_runner_excludes_compile_and_parameter_backward_from_timed_cells():
    source = inspect.getsource(first_group)
    assert 'torch.compile(' not in source
    assert 'parameter_backward' in source
    assert 'with torch.no_grad()' in source
