"""Unforced 2+1D KdV-type equation with compatible oblique-kink traces.

This is a prescribed-trace computational benchmark, not a well-posedness
theorem. No interior exact-solution values enter the training loss.
"""
import numpy as np
import torch
from torch.func import jvp, vmap
from .operators import evaluate

PROTOCOL = 'kdv2d_oblique_kink_v1'


def exact(points):
    return torch.tanh(.5*points[..., 0] + .5*points[..., 1] - points[..., 2])


def exact_first(points, axis):
    coefficient = (.5, .5, -1.)[axis]
    return coefficient*(1-exact(points).square())


def sample_numpy(rng, n):
    p = rng.random((n, 3)).astype(np.float32)
    p[:, :2] = 2*p[:, :2]-1
    return p


def constraints(n=16, *, dtype=torch.float32, device='cpu'):
    """n x n points on each surface; there is no y=+1 outflow penalty."""
    if n < 2:
        raise ValueError('constraint side must be >= 2')
    s = torch.linspace(-1, 1, n, dtype=dtype, device=device)
    t = torch.linspace(0, 1, n, dtype=dtype, device=device)
    xx, yy = torch.meshgrid(s, s, indexing='ij')
    ss, tt = torch.meshgrid(s, t, indexing='ij')
    make = lambda a, b, c: torch.stack((a.flatten(), b.flatten(), c.flatten()), dim=1)
    return dict(initial=make(xx, yy, torch.zeros_like(xx)),
                x_left=make(-torch.ones_like(ss), ss, tt),
                x_right=make(torch.ones_like(ss), ss, tt),
                y_bottom=make(ss, -torch.ones_like(ss), tt))


def first(model, points, axis):
    direction = torch.eye(3, dtype=points.dtype, device=points.device)[axis]
    return vmap(lambda p: jvp(model, (p,), (direction,))[1])(points)


def constraint_losses(model, data):
    out = {key: (model(p)-exact(p)).square().mean() for key, p in data.items()}
    out['x_right_dx'] = (first(model, data['x_right'], 0)-exact_first(data['x_right'], 0)).square().mean()
    out['y_bottom_dy'] = (first(model, data['y_bottom'], 1)-exact_first(data['y_bottom'], 1)).square().mean()
    return out


def loss(model, points, data, method, *, independent=False):
    if independent:
        from .reference import reference
        pde, _ = reference(model, points, 'kdv2d')
    else:
        pde = evaluate(model, points, 'kdv2d', method)
    terms = dict(pde=pde, **constraint_losses(model, data))
    total = pde + 10*sum(v for k, v in terms.items() if k != 'pde')
    return total, terms


def grid(nxy=33, nt=11, *, dtype=torch.float32, device='cpu'):
    if min(nxy, nt) < 2:
        raise ValueError('grid sides must be >= 2')
    s = torch.linspace(-1, 1, nxy, dtype=dtype, device=device)
    t = torch.linspace(0, 1, nt, dtype=dtype, device=device)
    x, y, time = torch.meshgrid(s, s, t, indexing='ij')
    return torch.stack((x.flatten(), y.flatten(), time.flatten()), dim=1)


def predictions(model, points, chunk=4096):
    with torch.no_grad():
        pred = torch.cat([model(p) for p in points.split(chunk)])
        target = exact(points)
        delta = pred-target
        norm = torch.linalg.vector_norm
        result = dict(relative_l2=float(norm(delta)/norm(target)),
                      linf=float(delta.abs().max()))
        end = points[:, 2] == 1
        result['final_relative_l2'] = float(norm(delta[end])/norm(target[end]))
        result['final_linf'] = float(delta[end].abs().max())
    return result, pred, target
