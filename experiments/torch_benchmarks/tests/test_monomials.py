import pytest
import torch
from torch_pinn.model import MLP,sample_points
from torch_pinn.monomials import monomial,schedule


@pytest.mark.parametrize('alpha',[(0,),(0,1),(0,0,1),(0,0,1,1),(0,0,0,1,1,1)])
def test_single_partial_and_parameter_gradient(alpha):
    model=MLP(2,3,2,71,dtype=torch.float64)
    points=sample_points(72,2,2,dtype=torch.float64)
    direct=monomial(model,points,alpha,'nested_jvp')
    raw=monomial(model,points,alpha,'waring_batched',return_complex=True)
    torch.testing.assert_close(raw.real,direct,rtol=1e-8,atol=1e-10)
    if raw.is_complex():torch.testing.assert_close(raw.imag,torch.zeros_like(raw.imag),rtol=0,atol=1e-10)
    a=torch.autograd.grad(direct.square().mean(),tuple(model.parameters()),allow_unused=True)
    b=torch.autograd.grad(raw.real.square().mean(),tuple(model.parameters()),allow_unused=True)
    for p,x,y in zip(model.parameters(),a,b):
        x=torch.zeros_like(p) if x is None else x;y=torch.zeros_like(p) if y is None else y
        torch.testing.assert_close(x,y,rtol=1e-7,atol=1e-10)


@pytest.mark.parametrize('pattern,count',[('111111',1),('111222',4),('112233',9),('123456',32),('11223344',27),('111222333',16)])
def test_original_first_group_direction_counts(pattern,count):
    alpha=tuple(int(c)-1 for c in pattern)
    directions,weights=schedule(alpha,16,dtype=torch.float64)
    assert len(directions)==len(weights)==count
