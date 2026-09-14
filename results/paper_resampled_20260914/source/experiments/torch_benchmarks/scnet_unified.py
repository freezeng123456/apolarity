"""Frozen, bounded SCNet paired runs with dense offline trajectory capture."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch
from bscc_sweep import verify_seal
from torch_pinn.records import configure, save, seal, source_identity
from torch_pinn.model import MLP
from train_fixed_wall import PROBLEMS, full_loss, sample

HERE = Path(__file__).resolve().parent
PLAN_FILE = HERE / 'scnet_unified_plan.json'
PLAN = json.loads(PLAN_FILE.read_text())


def config(case, method, seed, smoke=False):
    return dict(case=case, method=method, seed=seed, lr=1e-4, width=128,
                depth=5, batch=400, seconds=1200, expected_device='RTX 3080',
                eval_seed_base=20261400, eval_points=32 if smoke else 10000,
                checkpoint_seconds=10, trajectory_seconds=.001 if smoke else 2,
                max_steps=3 if smoke else 0)


def runtime():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Slurm allocation required')
    commit, kind = source_identity()
    configure('cuda', 'RTX 3080')
    versions = dict(torch=torch.__version__, numpy=np.__version__,
                    python_major_minor=list(sys.version_info[:2]))
    if versions != PLAN['runtime']:
        raise RuntimeError('unexpected runtime: '+str(versions))
    if torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one allocated visible GPU required')
    return dict(source_commit=commit, source_kind=kind, python=sys.executable,
                torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
                hostname=platform.node(), gpu=torch.cuda.get_device_name(0),
                environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES',
                'SLURM_JOB_ID','SLURM_ARRAY_TASK_ID','SLURM_CPUS_PER_TASK']})


def validate(out, cfg, commit):
    verify_seal(out)
    got=json.loads((out/'config.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    assert (out/'status.txt').read_text().strip()=='COMPLETE'
    assert got['source_commit']==commit
    assert all(got[k]==v for k,v in cfg.items())
    assert summary['initial_hash']!=summary['final_hash']
    assert summary['steps']>=3
    if not cfg['max_steps']:
        assert summary['terminal']=='wall_time_budget' and summary['wall_seconds']>=1200
    trajectory=json.loads((out/'trajectory_checkpoints.json').read_text())
    assert len(trajectory)>=4
    assert trajectory[0]['step']==0 and trajectory[-1]['step']==summary['steps']
    assert all(np.isfinite(summary['errors'][k]['relative_l2']) for k in ['spacetime','terminal','extra'])
    return summary


def child(out, cfg, commit):
    cmd=[sys.executable,str(HERE/'train_fixed_wall.py'),'--out',str(out)]
    for key,value in cfg.items():cmd+=['--'+key.replace('_','-'),str(value)]
    with out.with_suffix('.log').open('w') as log:
        subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1450)
    return validate(out,cfg,commit)


def preflight(root):
    out=root/'preflight';out.mkdir(exist_ok=False)
    env=runtime();save(out/'environment.json',env)
    with (out/'tests.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pytest',str(HERE/'tests'),'-q'],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
    gates=[]
    for case in PLAN['cases']:
        module=PROBLEMS[case];dim=2 if case=='kdv1d' else 3
        model=MLP(dim,128,5,20260921,device='cuda')
        data=module.constraints(128 if dim==2 else 16,device='cuda')
        points=torch.as_tensor(sample(np.random.default_rng(20260920),400,case),device='cuda')
        vals=[];grads=[]
        for method in PLAN['methods']:
            model.zero_grad(set_to_none=True)
            value,_=full_loss(module,model,points,data,method);value.backward()
            vals.append(value.detach().clone())
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]).detach().clone())
        torch.testing.assert_close(vals[0],vals[1],rtol=1e-3,atol=1e-5)
        torch.testing.assert_close(grads[0],grads[1],rtol=1e-3,atol=1e-5)
        gates.append(dict(case=case,loss_abs=float(abs(vals[0]-vals[1])),
                          gradient_max_abs=float((grads[0]-grads[1]).abs().max())))
        del model,data,points,vals,grads
        for method in PLAN['methods']:
            child(out/(case+'_'+method),config(case,method,20260919,True),env['source_commit'])
    save(out/'PASS.json',dict(environment=env,gates=gates));seal(out)


def pair(root,index):
    if not 0<=index<15:raise ValueError('pair index must be 0..14')
    verify_seal(root/'preflight')
    env=runtime();passed=json.loads((root/'preflight'/'PASS.json').read_text())['environment']
    for key in ['source_commit','python','torch','numpy','cuda','gpu']:
        if env[key]!=passed[key]:raise RuntimeError('preflight runtime mismatch: '+key)
    case=PLAN['cases'][index%3];seed=PLAN['seeds'][index//3]
    out=root/'pairs'/f'{case}_seed{seed}';out.mkdir(parents=True,exist_ok=False)
    save(out/'environment.json',env)
    order=PLAN['methods'][::1 if index//3%2==0 else -1]
    summaries={}
    for method in order:
        save(out/'state.json',dict(status='TRAINING',case=case,seed=seed,method=method))
        summaries[method]=child(out/method,config(case,method,seed),env['source_commit'])
    a,b=[out/m for m in PLAN['methods']]
    ah=json.loads((a/'batch_hashes.json').read_text());bh=json.loads((b/'batch_hashes.json').read_text())
    n=min(len(ah),len(bh));assert ah[:n]==bh[:n]
    assert summaries[PLAN['methods'][0]]['initial_hash']==summaries[PLAN['methods'][1]]['initial_hash']
    save(out/'summary.json',summaries)
    save(out/'state.json',dict(status='COMPLETE',case=case,seed=seed,matched_batch_prefix=n))
    (out/'done').write_text('COMPLETE\n');seal(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','pair'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int)
    args=p.parse_args()
    if args.mode=='preflight':preflight(args.root)
    else:pair(args.root,args.index)
