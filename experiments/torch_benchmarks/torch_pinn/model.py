"""One scalar Linear/tanh MLP shared by all derivative implementations."""
import math
import numpy as np
import torch
from torch import nn
from .stde_sampling import sample_interior


class MLP(nn.Module):
    def __init__(self, dim=2, width=128, depth=4, seed=20260910, *, dtype=torch.float32, device='cpu'):
        super().__init__()
        if dim < 1 or width < 1 or depth < 2:
            raise ValueError('positive dimensions and depth >= 2 required')
        dims = [dim] + [width]*(depth-1) + [1]
        self.layers = nn.ModuleList([nn.Linear(a, b, dtype=dtype, device=device) for a, b in zip(dims[:-1], dims[1:])])
        rng = np.random.default_rng(seed)
        npdtype = np.float64 if dtype == torch.float64 else np.float32
        with torch.no_grad():
            for layer in self.layers:
                bound = 1/math.sqrt(layer.in_features)
                w = rng.uniform(-bound, bound, tuple(layer.weight.shape)).astype(npdtype)
                b = rng.uniform(-bound, bound, tuple(layer.bias.shape)).astype(npdtype)
                layer.weight.copy_(torch.as_tensor(w, device=device))
                layer.bias.copy_(torch.as_tensor(b, device=device))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1:
                x = torch.tanh(x)
        return x[..., 0]


def sample_points(seed, n, dim, *, dtype=torch.float32, device='cpu'):
    """STDE geometry used by the historical PDE compute benchmark.

    In two spatial dimensions this is a disk with uniform radius, not a
    square or an area-uniform disk. Full PINN training has its own sampler.
    """
    rng = np.random.default_rng(seed)
    points = sample_interior(rng, n, dim-1)
    return torch.tensor(points, dtype=dtype, device=device)


def sample_normal(seed, n, dim, *, dtype=torch.float32, device='cpu', std=.35):
    """Historical first-group input convention: N(0, 0.35^2) in every axis.

    This deliberately differs from ``sample_points``: the latter is the
    space-time sampler for the PDE operator benchmark, while the fixed mixed-partial
    matrix used an unconstrained 16-dimensional normal input.  Keeping both
    named samplers prevents the first-group reproduction from silently using
    the PDE distribution.
    """
    rng = np.random.default_rng(seed)
    npdtype = np.float64 if dtype == torch.float64 else np.float32
    values = (rng.standard_normal((n, dim))*std).astype(npdtype)
    return torch.as_tensor(values, dtype=dtype, device=device)
