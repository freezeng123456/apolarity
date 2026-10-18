"""Exact deterministic Taylor constructions described in STDE++.

The individual-partial baseline is the multiple-linear-jet construction of
Section 6.1, evaluated with centered polarization, repeated-index grouping,
global-sign pairing, and collinear-direction merging.  It is NOT the new
random-prime polynomial single-jet construction of Section 6.2.4.

PDE curves reproduce Appendix I.4 (and the official JAX implementation) using
ordinary differentiable PyTorch Taylor arithmetic.  Coefficient r is the
coefficient of s**r, i.e. the raw input/output tangent divided by r!.
"""
from collections import Counter, defaultdict
from functools import lru_cache, reduce
from itertools import product
from math import comb, factorial, gcd

import torch
from torch.nn import functional as F
from torch.func import jvp, vmap

from .jets import mlp_jet, tanh_series
from .operators import assemble, require_real_pde

PAPER_URL = 'https://jmlr.org/papers/volume27/25-1474/25-1474.pdf'
UPSTREAM_URL = 'https://github.com/sail-sg/stde'
UPSTREAM_COMMIT = 'fae88663b1d2f1b0666d4d250c38a3c34b852ac0'
VARIANT = 'STDE++ deterministic multi-jet (Section 6.1)'


@lru_cache(maxsize=None)
def _grouped_schedule(alpha):
    """Integer directions and exact integer weights with denominator 2**k.

    alpha lists repeated zero-based coordinate indices, not exponent counts.
    Centered polarization says D_alpha u = 2**(-k) sum_eps prod(eps)
    q_k(sum_j eps_j e_alpha[j]), with q_k = D_v**k u/k!.
    """
    if not alpha or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in alpha):
        raise ValueError('alpha must be a nonempty tuple of nonnegative coordinate indices')
    counts = Counter(alpha)
    axes = tuple(sorted(counts))
    k = len(alpha)
    weights = defaultdict(int)
    for plus in product(*(range(counts[i]+1) for i in axes)):
        direction = tuple(2*j-counts[i] for i, j in zip(axes, plus))
        if not any(direction):
            continue
        coefficient = (-1)**(k-sum(plus))
        for i, j in zip(axes, plus):
            coefficient *= comb(counts[i], j)
        divisor = reduce(gcd, (abs(v) for v in direction))
        primitive = tuple(v//divisor for v in direction)
        coefficient *= divisor**k
        if next(v for v in primitive if v) < 0:
            primitive = tuple(-v for v in primitive)
            coefficient *= (-1)**k
        weights[primitive] += coefficient
    pairs = tuple((direction, numerator) for direction, numerator in sorted(weights.items()) if numerator)
    return axes, pairs, 2**k


def schedule(alpha, dim, *, dtype=torch.float32, device='cpu'):
    """Return real linear-jet directions and weights for a prescribed partial."""
    alpha = tuple(alpha)
    axes, pairs, denominator = _grouped_schedule(alpha)
    if max(axes) >= dim:
        raise ValueError('coordinate index exceeds the input dimension')
    directions = torch.zeros((len(pairs), dim), dtype=dtype, device=device)
    for row, (direction, _) in enumerate(pairs):
        directions[row, list(axes)] = directions.new_tensor(direction)
    weights = torch.tensor([numerator/denominator for _, numerator in pairs], dtype=dtype, device=device)
    return directions, weights


def partial(model, points, alpha):
    """Return a batch of exact mixed partials; alpha contains coordinate indices."""
    require_real_pde(model, points)
    if points.ndim != 2:
        raise ValueError('points must have shape [batch, dimension]')
    directions, weights = schedule(tuple(alpha), points.shape[-1], dtype=points.dtype, device=points.device)
    return weights @ mlp_jet(model, points, directions, len(alpha))[-1]


def curved_jet(model, points, coefficients):
    """Propagate normalized curved inputs [jet, positive order, input dimension].

    Output q[r] has shape [jet,batch] and equals the coefficient of s**r in
    u(points + sum_{r>=1} coefficients[:,r-1,:] * s**r).  Ordinary PyTorch
    operations retain all parameter gradients, with no custom derivative rule.
    """
    if points.ndim != 2 or coefficients.ndim != 3 or coefficients.shape[-1] != points.shape[-1]:
        raise ValueError('expected points [batch,dim] and coefficients [jet,order,dim]')
    if coefficients.shape[0] < 1 or coefficients.shape[1] < 1 or coefficients.device != points.device:
        raise ValueError('positive jet/order counts on the same device are required')
    dtype = torch.promote_types(points.dtype, coefficients.dtype)
    a = (points.to(dtype).unsqueeze(0).expand(len(coefficients), -1, -1),)
    a += tuple(c.unsqueeze(1).expand(-1, len(points), -1) for c in coefficients.to(dtype).unbind(1))
    for i, layer in enumerate(model.layers):
        linear = F.linear(torch.stack(a), layer.weight.to(dtype)).unbind(0)
        a = (linear[0]+layer.bias.to(dtype),)+linear[1:]
        if i < len(model.layers)-1:
            a = tanh_series(a)
    return tuple(c[..., 0] for c in a)


def _raw_curves(points, order, curves):
    """Convert sparse raw-tangent maps {order: coordinate-vector} to coefficients."""
    out = points.new_zeros((len(curves), order, points.shape[-1]))
    for j, curve in enumerate(curves):
        for r, direction in curve.items():
            out[j, r-1] = points.new_tensor(direction)/factorial(r)
    return out


def components(case, model, points):
    """Interior PDE quantities, including common unchanged time-JVP for CH."""
    require_real_pde(model, points)
    if case in ('kdv1d', 'gkdv1d'):
        # Appendix I.4, Eq.110: x-only 7-jet, x/t curved 7-jet, t-only 2-jet.
        q = curved_jet(model, points, _raw_curves(points, 7, (
            {1: (1, 0)}, {1: (1, 0), 4: (0, 1)})))
        time = mlp_jet(model, points, points.new_tensor(((0, 1),)), 2)
        return dict(u=q[0][0], u_x=q[1][0], u_xx=2*q[2][0],
                    u_xxx=6*q[3][0], u_xxxx=24*q[4][0],
                    u_t=24*(q[4][1]-q[4][0]),
                    u_tx=24*(q[5][1]-q[5][0]),
                    u_txxx=144*(q[7][1]-q[7][0]), u_tt=2*time[2][0])
    if case == 'kdv2d':
        # Appendix I.4, Eq.100-101: 9-jet + two 3-jets (the lower-order variant).
        spatial = curved_jet(model, points, _raw_curves(points, 9, (
            {2: (1, 0, 0), 3: (0, 1, 0)},)))
        lower = curved_jet(model, points, _raw_curves(points, 3, (
            {1: (0, 1, 0), 2: (0, 0, 1)}, {1: (0, 1, 0)})))
        uyyy = 6*lower[3][1]
        return dict(u_x=2*spatial[2][0], u_y=6*spatial[3][0],
                    u_xx=8*spatial[4][0], u_xy=12*spatial[5][0],
                    u_yy=2*lower[2][1],
                    u_ty=(6*lower[3][0]-uyyy)/3,
                    # Eq.101/upstream print 840, but the stated raw tangents
                    # give 9!/(3!*(2!)**3*3!)=1260.  Exact expansion and the
                    # independent coordinate-JVP tests require this correction.
                    u_xxxy=(factorial(9)*spatial[9][0]-280*uyyy)/1260)
    if case == 'ch2d':
        # Adaptation of Sec.6.1 to the biharmonic operator.  Four 4-jets share
        # all lower spatial orders; this PDE is not in the published suite.
        q = mlp_jet(model, points, points.new_tensor(((1, 0, 0), (0, 1, 0),
                                                    (1, 1, 0), (1, -1, 0))), 4)
        pure = 24*(q[4][0]+q[4][1])
        mixed = 2*(q[4][2]+q[4][3]-2*q[4][0]-2*q[4][1])
        time_dir = points.new_tensor((0, 0, 1))
        ut = vmap(lambda z: jvp(model, (z,), (time_dir,))[1])(points)
        return dict(u=q[0][0], ut=ut, lap=2*(q[2][0]+q[2][1]),
                    bih=pure+2*mixed, grad2=q[1][0].square()+q[1][1].square())
    raise ValueError(f'unsupported case {case!r}')


def pde_loss(case, model, points, data):
    """Identical full training objective with STDE++ interior derivatives."""
    d = components(case, model, points)
    if case in ('kdv1d', 'gkdv1d'):
        from .problem import boundary_losses
        interior, residuals = assemble('gkdv1d', d)
        terms = dict(f=residuals['f'].square().mean(), fx=residuals['f_x'].square().mean(),
                     ft=residuals['f_t'].square().mean(), **boundary_losses(model, data))
        return interior+10*sum(terms[k] for k in ('ic', 'bc_left', 'bc_right', 'bc_dx')), terms
    if case == 'kdv2d':
        from .problem_kdv2d import constraint_losses
        interior, _ = assemble(case, d)
        terms = dict(pde=interior, **constraint_losses(model, data))
        return interior+10*sum(v for k, v in terms.items() if k != 'pde'), terms
    from .problem_ch2d import analytic, boundary_losses
    r = (d['ut']-(3*d['u'].square()-1)*d['lap']-6*d['u']*d['grad2']
         +.01*d['bih']-analytic(points)['source'])
    terms = dict(pde=r.square().mean(), **boundary_losses(model, data))
    return terms['pde']+10*sum(terms[k] for k in ('ic', 'bc1', 'bc3')), terms


def metadata(alpha=None):
    """JSON-serializable method attribution; no polynomial-algorithm claim."""
    result = dict(variant=VARIANT, paper=PAPER_URL, upstream=UPSTREAM_URL,
                  upstream_commit=UPSTREAM_COMMIT, backend='eager PyTorch',
                  stochastic_sampling=False, polynomial_prime_construction=False,
                  kdv1d='Appendix I.4 Eq.110: 7+7+2 raw-tangent jets',
                  kdv2d='Appendix I.4 Eq.100: 9+3+3 raw-tangent jets',
                  ch2d='Sec.6.1 adaptation: four spatial 4-jets and common time JVP')
    if alpha is not None:
        axes, pairs, denominator = _grouped_schedule(tuple(alpha))
        result.update(alpha=list(alpha), order=len(alpha), axes=list(axes),
                      direction_count=len(pairs), integer_directions=[list(v) for v, _ in pairs],
                      weight_numerators=[n for _, n in pairs], weight_denominator=denominator)
    return result
