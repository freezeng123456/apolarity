"""Geometry in sail-sg/stde at fae88663b1d2f1b0666d4d250c38a3c34b852ac0.

Source: stde/equations.py, unit_ball_sample_domain_fn, lines 369--396.
Defaults in stde/config.py: max_radius=1 and T=1. Radius, rather than
volume, is uniform. These functions reproduce the distribution, not JAX's
random-number sequence. Geometry does not specify a PDE boundary condition.
"""
import numpy as np


def _directions(rng, n, spatial_dim):
    if n < 0 or spatial_dim < 1:
        raise ValueError('nonnegative count and positive spatial dimension required')
    direction = rng.standard_normal((n, spatial_dim))
    return direction / np.maximum(np.linalg.norm(direction, axis=-1, keepdims=True), 1e-12)


def sample_interior(rng, n, spatial_dim):
    """Rows (x_1, ..., x_d, t); unit ball, uniform radius, t in [0,1)."""
    radius = rng.uniform(0, 1, (n, 1))
    space = _directions(rng, n, spatial_dim) * radius
    return np.concatenate([space, rng.uniform(0, 1, (n, 1))], axis=-1)


def sample_lateral_boundary(rng, n, spatial_dim):
    """Unit sphere times [0,1); these are not initial or terminal points."""
    space = _directions(rng, n, spatial_dim)
    return np.concatenate([space, rng.uniform(0, 1, (n, 1))], axis=-1)
