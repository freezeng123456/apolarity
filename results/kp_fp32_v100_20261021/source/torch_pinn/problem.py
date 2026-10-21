"""Unforced 1D KdV soliton IBVP; identical training loss for every backend."""
import torch
from torch.func import jvp, vmap
from .operators import evaluate


def exact(points):
    z = 5*(points[...,0]+.5-.25*points[...,1])
    return .75/torch.cosh(z).square()


def exact_x(points):
    z = 5*(points[...,0]+.5-.25*points[...,1])
    return -10*exact(points)*torch.tanh(z)


def constraints(n=128, *, dtype=torch.float32, device='cpu'):
    if n < 2:
        raise ValueError('at least two constraint points required')
    x = torch.linspace(-1,1,n,dtype=dtype,device=device)
    t = torch.linspace(0,1,n,dtype=dtype,device=device)
    initial = torch.stack((x,torch.zeros_like(x)),dim=1)
    left = torch.stack((-torch.ones_like(t),t),dim=1)
    right = torch.stack((torch.ones_like(t),t),dim=1)
    return initial,left,right


def boundary_losses(model, data):
    initial,left,right = data
    basis = right.new_tensor((1.,0.))
    right_dx = vmap(lambda p: jvp(model,(p,),(basis,))[1])(right)
    return dict(ic=(model(initial)-exact(initial)).square().mean(),
                bc_left=(model(left)-exact(left)).square().mean(),
                bc_right=(model(right)-exact(right)).square().mean(),
                bc_dx=(right_dx-exact_x(right)).square().mean())


def loss(model, points, data, method):
    pde, d = evaluate(model,points,'gkdv1d',method,components=True)
    terms = dict(f=d['f'].square().mean(),fx=d['f_x'].square().mean(),ft=d['f_t'].square().mean(),
                 **boundary_losses(model,data))
    total = pde+10*terms['ic']+10*(terms['bc_left']+terms['bc_right']+terms['bc_dx'])
    return total,terms


def grid(nx=128, nt=51, *, dtype=torch.float32,device='cpu'):
    x=torch.linspace(-1,1,nx,dtype=dtype,device=device)
    t=torch.linspace(0,1,nt,dtype=dtype,device=device)
    xx,tt=torch.meshgrid(x,t,indexing='ij')
    return torch.stack((xx.flatten(),tt.flatten()),dim=1)


def validate(model, points, data, residual_points):
    # No reverse parameter graph for validation; input JVP remains available.
    with torch.no_grad():
        prediction = model(points)
        target=exact(points)
        final = points[:,1] == 1
        final_delta=prediction[final]-target[final]
        _,d=evaluate(model,residual_points,'gkdv1d','nested_jvp',components=True)
        bc=boundary_losses(model,data)
        metrics=dict(relative_l2=float(torch.linalg.vector_norm(prediction-target)/torch.linalg.vector_norm(target)),
                     final_relative_l2=float(torch.linalg.vector_norm(final_delta)/torch.linalg.vector_norm(target[final])),
                     final_linf=float(final_delta.abs().max()),
                     residual_rms=float(d['f'].square().mean().sqrt()),
                     gradient_residual_rms=float((d['f_x'].square().mean()+d['f_t'].square().mean()).sqrt()),
                     **{k+'_rms':float(v.sqrt()) for k,v in bc.items()})
    return metrics,prediction.detach(),target.detach()
