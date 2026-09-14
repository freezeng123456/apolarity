"""Check the source distribution and our own backends on its domains."""
import numpy as np
import pytest
import torch

from torch_pinn.model import MLP, sample_points
from torch_pinn.stde_sampling import sample_interior, sample_lateral_boundary
from torch_pinn.operators import evaluate
from torch_pinn.reference import reference


@pytest.mark.parametrize('spatial_dim', [1, 2])
def test_source_distribution(spatial_dim):
    points = sample_interior(np.random.default_rng(714), 100000, spatial_dim)
    radius = np.linalg.norm(points[:, :-1], axis=1)
    assert np.all(radius < 1)
    assert np.all((points[:, -1] >= 0) & (points[:, -1] < 1))
    assert abs(radius.mean() - .5) < .005
    # An area-uniform disk would instead have E[r^2] = 1/2.
    assert abs(np.mean(radius**2) - 1/3) < .005
    assert np.max(np.abs(points[:, :-1].mean(axis=0))) < .005
    assert abs(points[:, -1].mean() - .5) < .005
    if spatial_dim == 2:
        assert abs(np.mean(points[:, 0] * points[:, 1])) < .005
        assert np.max(np.abs((points[:, :2]**2).mean(axis=0) - 1/6)) < .005


@pytest.mark.parametrize('spatial_dim', [1, 2])
def test_lateral_boundary_and_fresh_paired_samples(spatial_dim):
    a, b = np.random.default_rng(913), np.random.default_rng(913)
    first = sample_lateral_boundary(a, 400, spatial_dim)
    np.testing.assert_array_equal(first, sample_lateral_boundary(b, 400, spatial_dim))
    second = sample_lateral_boundary(a, 400, spatial_dim)
    np.testing.assert_array_equal(second, sample_lateral_boundary(b, 400, spatial_dim))
    assert not np.array_equal(first, second)
    np.testing.assert_allclose(np.linalg.norm(first[:, :-1], axis=1), 1, atol=1e-14)
    assert np.all((first[:, -1] >= 0) & (first[:, -1] < 1))


@pytest.mark.parametrize('case,dim', [('gkdv1d', 2), ('kdv2d', 3)])
def test_residual_and_parameter_gradient_on_source_domain(case, dim):
    model = MLP(dim, 5, 3, 17, dtype=torch.float64)
    points = sample_points(81, 7, dim, dtype=torch.float64)
    expected, _ = reference(model, points, case)
    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()), allow_unused=True)
    for method in ('nested_jvp', 'shared_jet_linear'):
        actual = evaluate(model, points, case, method)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
        grads = torch.autograd.grad(actual, tuple(model.parameters()), allow_unused=True)
        for g, ref in zip(grads, expected_grads):
            assert (g is None) == (ref is None)
            if g is not None:
                torch.testing.assert_close(g, ref, rtol=1e-9, atol=1e-11)
