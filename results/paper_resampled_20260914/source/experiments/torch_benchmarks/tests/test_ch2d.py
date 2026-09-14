import torch
from torch.func import vmap
from torch_pinn.model import MLP
from torch_pinn import problem_ch2d as ch
from torch_pinn.operators import partial


def test_manufactured_solution_and_natural_traces():
    torch.manual_seed(7)
    p=torch.rand(9,3,dtype=torch.float64);p[:,:2]*=torch.pi
    ds=ch.analytic(p)
    lap=vmap(lambda z:partial(ch.exact,z,(0,0))+partial(ch.exact,z,(1,1)))(p)
    bih=vmap(lambda z:partial(ch.exact,z,(0,0,0,0))+2*partial(ch.exact,z,(0,0,1,1))+partial(ch.exact,z,(1,1,1,1)))(p)
    torch.testing.assert_close(lap,ds['lap'],atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(bih,ds['bih'],atol=1e-10,rtol=1e-10)
    data=ch.constraints(3,dtype=torch.float64)
    value,terms=ch.loss(ch.exact,p,data,'independent')
    assert float(value)<1e-20


def test_shared_values_full_gradient_and_adam():
    torch.manual_seed(8)
    p=torch.rand(5,3,dtype=torch.float64);p[:,:2]*=torch.pi
    data=ch.constraints(3,dtype=torch.float64)
    models=[MLP(3,8,3,19,dtype=torch.float64) for _ in range(2)]
    ca=ch.components(models[0],p,'shared_jet_linear')
    cb=ch.components(models[1],p,'independent')
    for k in ca:torch.testing.assert_close(ca[k],cb[k],rtol=1e-9,atol=1e-11)
    opts=[torch.optim.Adam(m.parameters(),lr=1e-4,foreach=False) for m in models]
    vals=[]
    for m,o,method in zip(models,opts,('shared_jet_linear','nested_jvp')):
        v,_=ch.loss(m,p,data,method);v.backward();vals.append(v)
    torch.testing.assert_close(vals[0],vals[1],rtol=1e-10,atol=1e-11)
    for a,b in zip(models[0].parameters(),models[1].parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=1e-8,atol=1e-10)
    for o in opts:o.step()
    for a,b in zip(models[0].parameters(),models[1].parameters()):
        torch.testing.assert_close(a,b,rtol=1e-9,atol=1e-11)
