"""Analytic, gradient, state and evaluation-isolation tests for the 2D IBVP."""
import copy
import numpy as np
import pytest
import torch
from torch.func import vmap
from torch_pinn.model import MLP
from torch_pinn.operators import partial, KDV_TERMS, KDV_DIRS, reconstruct
from torch_pinn.problem_kdv2d import exact, exact_first, constraints, loss, sample_numpy, grid


def test_analytic_unforced_residual_and_traces():
    points = torch.tensor(sample_numpy(np.random.default_rng(52), 17), dtype=torch.float64)
    d = {k:vmap(lambda p:partial(exact, p, a))(points) for k,a in KDV_TERMS.items()}
    # Independent assembly, not operators.assemble.
    r = d['u_ty']+d['u_xxxy']+3*(d['u_xy']*d['u_x']+d['u_y']*d['u_xx'])-d['u_xx']+2*d['u_yy']
    assert float(r.abs().max()) < 1e-12
    for axis in range(3):
        actual = vmap(lambda p:partial(exact, p, (axis,)))(points)
        torch.testing.assert_close(actual, exact_first(points, axis), rtol=1e-13, atol=1e-13)
    data = constraints(4, dtype=torch.float64)
    value, terms = loss(exact, points, data, 'nested_jvp')
    assert float(value) < 1e-24
    assert set(terms) == {'pde','initial','x_left','x_right','y_bottom','x_right_dx','y_bottom_dy'}
    assert all(v.shape == (16, 3) for v in data.values())
    assert torch.all(data['initial'][:,2] == 0)
    assert torch.all(data['x_left'][:,0] == -1)
    assert torch.all(data['x_right'][:,0] == 1)
    assert torch.all(data['y_bottom'][:,1] == -1)


def test_full_loss_parameter_gradient_and_adam():
    base = MLP(3, 5, 3, 73, dtype=torch.float64)
    points = torch.tensor(sample_numpy(np.random.default_rng(74), 4), dtype=torch.float64)
    data = constraints(3, dtype=torch.float64)
    records = []
    for method, independent in [('nested_jvp', True), ('nested_jvp', False), ('shared_jet_linear', False)]:
        model = copy.deepcopy(base)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, foreach=False)
        value, terms = loss(model, points, data, method, independent=independent)
        value.backward()
        grads = [p.grad.clone() for p in model.parameters()]
        opt.step()
        records.append((value, terms, grads, model, opt))
    reference = records[0]
    for row in records[1:]:
        torch.testing.assert_close(row[0], reference[0], rtol=1e-10, atol=1e-12)
        for key in row[1]:
            torch.testing.assert_close(row[1][key], reference[1][key], rtol=1e-10, atol=1e-12)
        for g, r in zip(row[2], reference[2]):
            torch.testing.assert_close(g, r, rtol=1e-9, atol=1e-11)
        for p,r in zip(row[3].parameters(), reference[3].parameters()):
            torch.testing.assert_close(p,r,rtol=1e-10,atol=1e-12)
            for key,v in row[4].state[p].items():
                torch.testing.assert_close(v,reference[4].state[r][key],rtol=1e-9,atol=1e-12)
    # Five independent central finite-difference directional parameter checks.
    rng = np.random.default_rng(75)
    for _ in range(5):
        direction = [torch.tensor(rng.standard_normal(tuple(p.shape)), dtype=p.dtype) for p in base.parameters()]
        length = torch.sqrt(sum((v*v).sum() for v in direction))
        direction = [v/length for v in direction]
        values = []
        for sign in (-1, 1):
            model = copy.deepcopy(base)
            with torch.no_grad():
                for p,v in zip(model.parameters(), direction): p.add_(sign*1e-5*v)
            values.append(loss(model,points,data,'shared_jet_linear')[0].detach())
        fd = (values[1]-values[0])/2e-5
        expected = sum((g*v).sum() for g,v in zip(reference[2],direction))
        torch.testing.assert_close(fd,expected,rtol=2e-6,atol=2e-8)


def test_sampling_geometry():
    p = sample_numpy(np.random.default_rng(5), 1000)
    assert p.dtype == np.float32 and p.shape == (1000,3)
    assert np.all((p[:,:2] >= -1)&(p[:,:2] <= 1)) and np.all((p[:,2] >= 0)&(p[:,2] <= 1))
    assert grid(5,3).shape == (75,3)
