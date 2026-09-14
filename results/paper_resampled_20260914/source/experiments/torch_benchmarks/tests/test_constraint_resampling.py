import json
import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch_pinn.constraint_sampling import ConstraintSampler, as_tensors, batch_hash
from torch_pinn.model import MLP
from train_fixed_wall import PROBLEMS, full_loss, train


@pytest.mark.parametrize('case,count,dim', [('kdv1d',16,2),('kdv2d',256,3),('ch2d',256,3)])
def test_uniform_surfaces_counts_and_replay(case,count,dim):
    a,b=ConstraintSampler(case,16,317),ConstraintSampler(case,16,317)
    x=a.sample_numpy(); assert batch_hash(x)==batch_hash(b.sample_numpy())
    y=a.sample_numpy(); assert batch_hash(y)==batch_hash(b.sample_numpy())
    assert batch_hash(x)!=batch_hash(y)
    assert all(not np.array_equal(x[k],y[k]) for k in x)
    lo,hi=(0.,math.pi) if case=='ch2d' else (-1.,1.)
    for name,p in x.items():
        assert p.shape==(count,dim) and p.dtype==np.float32
        assert np.all(p[:,-1]>=0) and np.all(p[:,-1]<=1)
        assert np.all(p[:,:-1]>=lo) and np.all(p[:,:-1]<=hi+1e-6)
        if name=='initial': assert np.all(p[:,-1]==0)
        elif 'left' in name: assert np.all(p[:,0]==np.float32(lo))
        elif 'right' in name: assert np.all(p[:,0]==np.float32(hi))
        elif 'bottom' in name: assert np.all(p[:,1]==np.float32(lo))
        elif name=='top': assert np.all(p[:,1]==np.float32(hi))
    saved=a.state
    z=a.sample_numpy()
    b.rng.bit_generator.state=saved
    assert batch_hash(z)==batch_hash(b.sample_numpy())


@pytest.mark.parametrize('case', ['kdv1d','kdv2d','ch2d'])
def test_backend_losses_and_gradients_agree_on_random_constraints(case):
    torch.set_num_threads(1)
    dim=2 if case=='kdv1d' else 3
    model=MLP(dim,8,3,17,dtype=torch.float64)
    arrays=ConstraintSampler(case,3,19).sample_numpy()
    data=as_tensors(case,{k:v.astype(np.float64) for k,v in arrays.items()},'cpu')
    points=torch.tensor(np.random.default_rng(37).random((4,dim)),dtype=torch.float64)
    reference=full_loss(PROBLEMS[case],model,points,data,'nested_jvp')[0]
    refgrad=torch.autograd.grad(reference,tuple(model.parameters()))
    value=full_loss(PROBLEMS[case],model,points,data,'shared_jet_linear')[0]
    grad=torch.autograd.grad(value,tuple(model.parameters()))
    torch.testing.assert_close(value,reference,rtol=1e-9,atol=1e-10)
    for a,b in zip(grad,refgrad):torch.testing.assert_close(a,b,rtol=1e-8,atol=1e-9)


def test_training_resamples_without_changing_interior_or_test_streams(tmp_path):
    torch.set_num_threads(1)
    outputs={}
    for sampling in ['fixed','resampled']:
        for method in ['nested_jvp','shared_jet_linear']:
            out=tmp_path/(sampling+'_'+method);out.mkdir()
            args=SimpleNamespace(out=out,case='kdv1d',method=method,device='cpu',
                width=8,depth=3,seed=51,lr=1e-3,eval_seed_base=20261400,eval_points=32,
                constraint_side=4,batch=4,seconds=30,checkpoint_seconds=100,
                trajectory_seconds=0,max_steps=3,constraint_sampling=sampling)
            train(args);outputs[sampling,method]=out
    read=lambda key,name:json.loads((outputs[key]/name).read_text())
    all_batches=[read(key,'batch_hashes.json') for key in outputs]
    assert all(x==all_batches[0] for x in all_batches)
    keys=[('resampled',m) for m in ['nested_jvp','shared_jet_linear']]
    hashes=[read(key,'constraint_batch_hashes.json') for key in keys]
    assert hashes[0]==hashes[1] and len(set(hashes[0]))==3
    with np.load(outputs['fixed','nested_jvp']/'evaluation_points.npz') as ref:
        for path in outputs.values():
            with np.load(path/'evaluation_points.npz') as got:
                for name in ref.files:np.testing.assert_array_equal(ref[name],got[name])
