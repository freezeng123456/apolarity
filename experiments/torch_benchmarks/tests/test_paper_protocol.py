"""Behavioral checks for the two explicit paper routes and resampled training."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from torch_pinn import problem, problem_kdv2d, problem_ch2d, operators
from torch_pinn.constraint_sampling import ConstraintSampler, as_tensors, batch_hash
from torch_pinn.model import MLP
from torch_pinn.monomials import monomial
from torch_pinn.records import configure
from train_fixed_wall import parse_args, train, sample
from run_fixed_wall_matrix import matrix_cells, verify_pair
from run_first_group_matrix import matrix_cells as partial_cells

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT/'results/paper_resampled_20260914/source/experiments/torch_benchmarks/torch_pinn'


@pytest.fixture(scope='module')
def frozen():
    spec = importlib.util.spec_from_file_location('_paper_frozen', FROZEN/'__init__.py',
                                                  submodule_search_locations=[str(FROZEN)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    return {case:__import__(f'_paper_frozen.{name}',fromlist=[name]) for case,name in
            [('kdv1d','problem'),('kdv2d','problem_kdv2d'),('ch2d','problem_ch2d')]}


@pytest.mark.parametrize('case', ['kdv1d','kdv2d','ch2d'])
@pytest.mark.parametrize('method', operators.METHODS)
def test_loss_gradients_and_adam_match_frozen_source(case, method, frozen):
    module = {'kdv1d':problem,'kdv2d':problem_kdv2d,'ch2d':problem_ch2d}[case]
    dim = 2 if case == 'kdv1d' else 3
    models = [MLP(dim,4,3,73,dtype=torch.float64) for _ in range(2)]
    points = torch.tensor(sample(np.random.default_rng(81),3,case),dtype=torch.float64)
    arrays = ConstraintSampler(case,2,91).sample_numpy()
    data = as_tensors(case,{k:v.astype(np.float64) for k,v in arrays.items()},'cpu')
    records = []
    for mod, model in zip((module,frozen[case]),models):
        opt = torch.optim.Adam(model.parameters(),lr=1e-5,foreach=False)
        value, parts = mod.loss(model,points,data,method)
        assert not value.is_complex()
        value.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        assert all(not g.is_complex() for g in gradients)
        opt.step()
        records.append((value.detach(),parts,gradients,model.state_dict(),opt.state_dict()))
    for a,b in zip(records[0][:4],records[1][:4]):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    for key,state in records[0][4]['state'].items():
        torch.testing.assert_close(state,records[1][4]['state'][key],rtol=0,atol=0)


@pytest.mark.parametrize('case', ['kdv1d','kdv2d','ch2d'])
def test_resampled_training_and_offline_diagnostics(tmp_path, case):
    configure('cpu')
    outputs=[]
    for method in operators.METHODS:
        out=tmp_path/method;out.mkdir()
        args=parse_args(['--out',str(out),'--case',case,'--method',method,'--device','cpu',
                         '--width','4','--depth','2','--batch','3','--constraint-side','2',
                         '--max-steps','3','--eval-points','16','--trajectory-seconds','0'])
        train(args);outputs.append(out)
        summary=json.loads((out/'summary.json').read_text())
        assert summary['steps']==3 and summary['constraint_sampling']=='resampled'
        assert summary['initial_hash'] != summary['final_hash']
    assert verify_pair(*outputs)['matched_updates']==3


@pytest.mark.parametrize('method', operators.METHODS)
def test_checkpoint_cadence_does_not_change_updates(tmp_path,method):
    configure('cpu');states=[]
    for cadence in (0,1e-9):
        out=tmp_path/str(cadence);out.mkdir()
        args=parse_args(['--out',str(out),'--case','kdv1d','--method',method,'--device','cpu',
                         '--width','3','--depth','2','--batch','3','--constraint-side','2',
                         '--max-steps','3','--eval-points','8','--trajectory-seconds',str(cadence)])
        train(args)
        states.append(torch.load(out/'checkpoint_0000003.pt',weights_only=False))
    torch.testing.assert_close(states[0]['model'],states[1]['model'],rtol=0,atol=0)
    assert states[0]['sampler_state']==states[1]['sampler_state']
    assert states[0]['constraint_sampler_state']==states[1]['constraint_sampler_state']
    for key,state in states[0]['optimizer']['state'].items():
        torch.testing.assert_close(state,states[1]['optimizer']['state'][key],rtol=0,atol=0)


@pytest.mark.parametrize('case', ['kdv1d','kdv2d','ch2d'])
def test_paper_defaults(case):
    args=parse_args(['--out','unused','--case',case,'--method','shared_jet_linear'])
    assert args.depth==5 and args.seconds==1200 and args.batch==400
    assert args.constraint_sampling=='resampled' and args.eval_seed_base==20261400
    assert args.lr==(1e-4 if case=='ch2d' else 1e-5)


@pytest.mark.parametrize('option,value',[('--method','auto'),('--method','shared_jet'),
                                        ('--constraint-sampling','fixed'),('--seconds','nan')])
def test_unsupported_cli_options_are_rejected(option,value):
    with pytest.raises(SystemExit):
        parse_args(['--out','unused','--case','kdv1d','--method','nested_jvp',option,value])


def test_only_paper_methods_are_scheduled():
    assert len(matrix_cells())==30
    assert {m for _,_,m in matrix_cells()}=={'nested_jvp','shared_jet_linear'}
    assert len(partial_cells())==20
    assert {m for _,m,_ in partial_cells()}=={'nested_jvp','waring_batched'}


@pytest.mark.parametrize('method',['auto','waring_serial','polarization_jet'])
def test_single_partial_rejects_retired_methods(method):
    with pytest.raises(ValueError):monomial(MLP(2,3,2),torch.zeros(2,2),(0,1),method)


@pytest.mark.parametrize('case',['kdv1d','kdv2d','ch2d'])
def test_pde_does_not_accept_complex_inputs(case):
    dim=2 if case=='kdv1d' else 3
    points=torch.zeros((2,dim),dtype=torch.complex64)
    with pytest.raises(ValueError,match='real'):
        if case=='ch2d':problem_ch2d.components(MLP(dim,3,2),points,'shared_jet_linear')
        else:operators.evaluate(MLP(dim,3,2),points,'gkdv1d' if case=='kdv1d' else case,'shared_jet_linear')
