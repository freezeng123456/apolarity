from __future__ import annotations

import math
import numpy as np
import jax.numpy as jnp


def init_mlp(seed: int, input_dim: int, width: int, depth: int, dtype=jnp.float32):
    """STDE ModelConfig.depth semantics: depth Linear layers including output."""
    if depth < 2:
        raise ValueError("depth must be >=2")
    rng = np.random.default_rng(seed)
    dims = [input_dim] + [width] * (depth - 1) + [1]
    params = []
    for fan_in, fan_out in zip(dims[:-1], dims[1:]):
        bound = 1.0 / math.sqrt(fan_in)
        w = rng.uniform(-bound, bound, size=(fan_out, fan_in)).astype(
            np.float32 if dtype == jnp.float32 else np.float64
        )
        b = rng.uniform(-bound, bound, size=(fan_out,)).astype(
            np.float32 if dtype == jnp.float32 else np.float64
        )
        params.append((jnp.asarray(w, dtype=dtype), jnp.asarray(b, dtype=dtype)))
    return tuple(params)


def mlp_one(params, x):
    y = x
    for i, (w, b) in enumerate(params):
        y = w @ y + b
        if i + 1 < len(params):
            y = jnp.tanh(y)
    return y[0]


def sample_normal(seed: int, n: int, d: int, dtype=jnp.float32, std: float = 0.35):
    rng = np.random.default_rng(seed)
    arr = (rng.standard_normal((n, d)) * std).astype(
        np.float32 if dtype == jnp.float32 else np.float64
    )
    return jnp.asarray(arr, dtype=dtype)


def sample_upstream_unit_ball_space_time(
    seed: int, n: int, spatial_dim: int, dtype=jnp.float32, max_radius=1.0, T=1.0
):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.0, max_radius, size=(n, 1))
    x = rng.standard_normal((n, spatial_dim))
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    x = x / np.maximum(norm, 1e-12) * r
    t = rng.uniform(0.0, T, size=(n, 1))
    arr = np.concatenate([x, t], axis=-1).astype(
        np.float32 if dtype == jnp.float32 else np.float64
    )
    return jnp.asarray(arr, dtype=dtype)
