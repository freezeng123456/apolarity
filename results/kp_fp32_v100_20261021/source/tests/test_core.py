from itertools import product
from math import factorial,prod
import copy
import inspect

import pytest
import torch
from torch.func import jvp,vmap

from torch_pinn.model import MLP,sample_points
from torch_pinn.jets import mlp_jet
from torch_pinn import jets,operators
from torch_pinn.operators import evaluate,partial,assemble,reconstruct,KDV_DIRS,GKDV_DIRS,KDV_TERMS,GKDV_TERMS
from torch_pinn.problem import exact,exact_x,constraints,loss,boundary_losses
from torch_pinn.reference import reference as independent_reference


def test_reference_keeps_float32_through_unexpanded_residual_jvp():
    model=MLP(2,4,2,73)
    points=sample_points(72,3,2)
    value,d=independent_reference(model,points,'gkdv1d')
    assert value.dtype==torch.float32
    assert all(t.dtype==torch.float32 for t in d.values())


def reference(model,points,case):
    terms=KDV_TERMS if case=='kdv2d' else GKDV_TERMS
    d={k:vmap(lambda x:partial(model,x,alpha))(points) for k,alpha in terms.items()}
    if case=='gkdv1d':
        d['u']=model(points)
        dispersion=points.new_tensor(.0025)
        def f(p):
            return partial(model,p,(1,))+model(p)*partial(model,p,(0,))+dispersion*partial(model,p,(0,0,0))
        fv=vmap(f)(points)
        basis=torch.eye(2,dtype=points.dtype,device=points.device)
        fx=vmap(lambda x:jvp(f,(x,),(basis[0],))[1])(points)
        ft=vmap(lambda x:jvp(f,(x,),(basis[1],))[1])(points)
        return fv.square().mean()+.001*(fx.square().mean()+ft.square().mean()),{**d,'f':fv,'f_x':fx,'f_t':ft}
    val,r=assemble(case,d)
    return val,{**d,**r}


@pytest.mark.parametrize('case',['kdv2d','gkdv1d'])
def test_polynomial_reconstruction(case):
    dirs=KDV_DIRS if case=='kdv2d' else GKDV_DIRS
    dim=len(dirs[0]); terms=KDV_TERMS if dim==3 else {'u':(),**GKDV_TERMS}
    for alpha in product(range(5),repeat=dim):
        if sum(alpha)>4: continue
        q=tuple(torch.tensor([[prod(v**a for v,a in zip(direction,alpha)) if sum(alpha)==k else 0] for direction in dirs],dtype=torch.float64) for k in range(5))
        result=reconstruct(case,q)
        for name,indices in terms.items():
            target=tuple(indices.count(i) for i in range(dim))
            expected=prod(factorial(a) for a in alpha) if alpha==target else 0
            torch.testing.assert_close(result[name],torch.tensor([expected],dtype=torch.float64),rtol=0,atol=1e-13)


@pytest.mark.parametrize('order',[1,2,3,4,6])
def test_taylor_orders(order):
    model=MLP(2,3,2,71,dtype=torch.float64)
    points=sample_points(72,2,2,dtype=torch.float64)
    directions=points.new_tensor(((1.,.5),(-.3,1.)))
    series=mlp_jet(model,points,directions,order)
    for i,v in enumerate(directions):
        fun=model
        for k in range(order+1):
            torch.testing.assert_close(series[k][i],vmap(fun)(points)/factorial(k),rtol=1e-9,atol=1e-11)
            previous=fun
            fun=lambda x,previous=previous:jvp(previous,(x,),(v,))[1]


@pytest.mark.parametrize('case',['kdv2d','gkdv1d'])
@pytest.mark.parametrize('method',operators.METHODS)
def test_components_parameter_grad_and_adam(case,method):
    for seed in (71,72):
        model=MLP(3 if case=='kdv2d' else 2,4,2,seed,dtype=torch.float64)
        expected_model=copy.deepcopy(model)
        points=sample_points(seed+1,3,3 if case=='kdv2d' else 2,dtype=torch.float64)
        actual,d=evaluate(model,points,case,method,components=True)
        expected,r=reference(expected_model,points,case)
        assert set(d)==set(r)
        for k in r: torch.testing.assert_close(d[k],r[k],rtol=1e-9,atol=1e-11)
        torch.testing.assert_close(actual,expected,rtol=1e-9,atol=1e-11)
        actual.backward(); expected.backward()
        for p,q in zip(model.parameters(),expected_model.parameters()):
            a=torch.zeros_like(p) if p.grad is None else p.grad
            b=torch.zeros_like(q) if q.grad is None else q.grad
            torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-10)
        # Unused parameters (KdV constant offset) get no artificial update.
        torch.optim.Adam(model.parameters(),lr=1e-3).step()
        torch.optim.Adam(expected_model.parameters(),lr=1e-3).step()
        for p,q in zip(model.parameters(),expected_model.parameters()):
            torch.testing.assert_close(p,q,rtol=1e-7,atol=1e-8)


def test_full_pinn_loss_gradient_and_parameter_direction_fd():
    model=MLP(2,4,2,71,dtype=torch.float64)
    points=sample_points(72,3,2,dtype=torch.float64)
    data=constraints(4,dtype=torch.float64)
    expected=reference(model,points,'gkdv1d')[0]
    bc=boundary_losses(model,data)
    expected=expected+10*sum(bc.values())
    ref_grad=torch.autograd.grad(expected,tuple(model.parameters()))
    actual,_=loss(model,points,data,'shared_jet_linear')
    gradient=torch.autograd.grad(actual,tuple(model.parameters()))
    for a,b in zip(gradient,ref_grad):torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)
    generator=torch.Generator().manual_seed(991)
    initial=[p.detach().clone() for p in model.parameters()]
    for _ in range(5):
        direction=[torch.randn(p.shape,generator=generator,dtype=p.dtype) for p in model.parameters()]
        norm=sum(v.square().sum() for v in direction).sqrt()
        direction=[v/norm for v in direction]
        target=sum((g*v).sum() for g,v in zip(gradient,direction))
        for h in (1e-4,1e-5):
            vals=[]
            for sign in (1,-1):
                with torch.no_grad():
                    for p,p0,v in zip(model.parameters(),initial,direction):p.copy_(p0+sign*h*v)
                vals.append(loss(model,points,data,'shared_jet_linear')[0].detach())
            torch.testing.assert_close((vals[0]-vals[1])/(2*h),target,rtol=1e-5,atol=1e-7)
        with torch.no_grad():
            for p,p0 in zip(model.parameters(),initial):p.copy_(p0)


def test_exact_soliton_and_no_compile():
    points=sample_points(99,10,2,dtype=torch.float64)
    basis=points.new_tensor((1.,0.))
    torch.testing.assert_close(vmap(lambda p:jvp(exact,(p,),(basis,))[1])(points),exact_x(points))
    f=vmap(lambda p:partial(exact,p,(1,))+exact(p)*partial(exact,p,(0,))+.0025*partial(exact,p,(0,0,0)))(points)
    torch.testing.assert_close(f,torch.zeros_like(f),atol=1e-13,rtol=0)
    for module in (jets,operators):
        source=inspect.getsource(module)
        assert 'torch.compile(' not in source and 'import jax' not in source
