"""KP-II line soliton, Pu and Chen (2024), Section 3.1, Case 3.

The PDE/region/initial and lateral Dirichlet data follow Eqs. (3.7)-(3.9).
Training follows the existing apolarity equal-update protocol, not the source
paper's L-BFGS experiment. Both derivative constructions use this same problem.
"""
import numpy as np
import torch
from torch.func import jvp, vmap

from .jets import mlp_jet

TERMINAL_TIME = .5
SOURCE_URL = 'https://arxiv.org/html/2401.04982v1'


def exact(points):
    phase = 3*points[..., 0]+6*points[..., 1]-63*points[..., 2]
    return 18*torch.sigmoid(phase)*torch.sigmoid(-phase)


def sample_numpy(rng, n):
    points = rng.random((n, 3)).astype(np.float32)
    points[:, :2] = 6*points[:, :2]-3
    points[:, 2] -= .5
    return points


def constraints(n=16, *, dtype=torch.float32, device='cpu'):
    from .constraint_sampling import ConstraintSampler
    return {k: torch.as_tensor(v, dtype=dtype, device=device)
            for k, v in ConstraintSampler('kp2d', n, 20260002).sample_numpy().items()}


def constraint_losses(model, data):
    return {name: (model(points)-exact(points)).square().mean()
            for name, points in data.items()}


def independent_components(model, points):
    """Coordinate AD oracle, independent of either Taylor construction."""
    def partial(axes):
        fn = model
        for axis in axes:
            previous = fn
            direction = torch.eye(3, dtype=points.dtype, device=points.device)[axis]
            fn = lambda z, fn=previous, v=direction: jvp(fn, (z,), (v,))[1]
        return vmap(fn)(points)
    return dict(u=model(points), u_x=partial((0,)), u_xx=partial((0, 0)),
                u_xxxx=partial((0, 0, 0, 0)), u_tx=partial((0, 2)),
                u_yy=partial((1, 1)))


def components(model, points, method):
    if method == 'stdepp':
        from .stdepp import components as stde_components
        return stde_components('kp2d', model, points)
    if method == 'nested_jvp':
        return independent_components(model, points)
    if method != 'shared_jet_linear':
        raise ValueError(f'unknown KP derivative method: {method}')
    pure = mlp_jet(model, points, points.new_tensor(((1, 0, 0),)), 4)
    lower = mlp_jet(model, points, points.new_tensor(((1, 0, 1), (1, 0, -1), (0, 1, 0))), 2)
    return dict(u=pure[0][0], u_x=pure[1][0], u_xx=2*pure[2][0],
                u_xxxx=24*pure[4][0], u_tx=(lower[2][0]-lower[2][1])/2,
                u_yy=2*lower[2][2])


def residual(d):
    return d['u_tx']+6*(d['u_x'].square()+d['u']*d['u_xx'])+d['u_xxxx']+3*d['u_yy']


def loss(model, points, data, method, *, independent=False):
    d = independent_components(model, points) if independent else components(model, points, method)
    terms = dict(pde=residual(d).square().mean(), **constraint_losses(model, data))
    return terms['pde']+10*sum(v for k, v in terms.items() if k != 'pde'), terms
