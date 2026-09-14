"""Bounded, restart-safe SCNet campaign for resampled IC/BC training."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
METHODS=('nested_jvp','shared_jet_linear')
SEEDS=list(range(20260919,20260924))
CELLS=[dict(protocol='wall',case=c,seed=s,method=m) for c in ('kdv1d','kdv2d','ch2d') for s in SEEDS for m in METHODS]
CELLS += [dict(protocol='steps',case=c,seed=s,method=m) for c,seeds in [('kdv1d',[20260918]),('kdv2d',list(range(20260921,20260924)))] for s in seeds for m in METHODS]


def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');q.replace(p)


def runtime():
    import numpy as np
    import torch
    from torch_pinn.records import configure,source_identity
    assert os.environ.get('SLURM_JOB_ID'),'Slurm GPU allocation required'
    configure('cuda','RTX 3080')
    assert torch.cuda.device_count()==1
    assert torch.__version__=='2.5.1+cu118'
    assert np.__version__=='1.26.4'
    commit,kind=source_identity()
    return dict(source_commit=commit,source_kind=kind,python=sys.executable,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),hostname=platform.node(),environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','SLURM_JOB_ID','SLURM_ARRAY_TASK_ID','SLURM_CPUS_PER_TASK']})


def name(c):return '{protocol}_{case}_seed{seed}_{method}'.format(**c)


def command(c,out,smoke=False):
    common=['--out',str(out),'--method',c['method'],'--seed',str(c['seed']),'--device','cuda','--expected-device','RTX 3080','--constraint-sampling','resampled','--width','128','--batch','400']
    if c['protocol']=='wall':
        cmd=[sys.executable,str(HERE/'train_fixed_wall.py'),*common,'--case',c['case'],'--depth','5','--lr',str(1e-4 if c['case']=='ch2d' else 1e-5),'--seconds','1200','--eval-seed-base','20261400','--eval-points','32' if smoke else '10000','--trajectory-seconds','.001' if smoke else '2','--checkpoint-seconds','10']
        if smoke:cmd+=['--max-steps','3']
    else:
        entry='train_equal_time.py' if c['case']=='kdv1d' else 'train_kdv2d.py'
        cmd=[sys.executable,str(HERE/entry),*common,'--depth','4','--steps','3' if smoke else '10000','--eval-every','1' if smoke else '50','--lr','1e-3']
        if smoke:
            if c['case']=='kdv1d':cmd+=['--val-nx','3','--val-nt','3']
            else:cmd+=['--val-nxy','3','--val-nt','3','--test-nxy','3','--test-nt','3']
    return cmd


def validate(out,c,smoke=False):
    from bscc_sweep import verify_seal
    verify_seal(out)
    read=lambda f:json.loads((out/f).read_text())
    s=read('summary.json');cfg=read('config.json')
    assert (out/'status.txt').read_text().strip()=='COMPLETE'
    assert s['constraint_sampling']==cfg['constraint_sampling']=='resampled'
    for k in ('case','seed','method'):
        if k in cfg:assert cfg[k]==c[k]
    hashes=read('constraint_batch_hashes.json')
    assert len(hashes)==s['steps']==len(read('batch_hashes.json'))
    assert len(set(hashes))==len(hashes)
    if smoke:assert s['steps']==3
    elif c['protocol']=='wall':assert s['terminal']=='wall_time_budget' and s['wall_seconds']>=1200
    else:assert s['steps']==10000 and s['final_scheduled_lr']==0
    return s


def child(c,out,smoke=False):
    assert not out.exists(),out
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.with_suffix('.log').open('w') as log:
        subprocess.run(command(c,out,smoke),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=500 if smoke else (1480 if c['protocol']=='wall' else 3550))
    return validate(out,c,smoke)


def paired(a,b):
    import numpy as np
    read=lambda p,f:json.loads((p/f).read_text())
    sa,sb=read(a,'summary.json'),read(b,'summary.json')
    assert sa.get('initial_hash',sa.get('initial_parameter_hash'))==sb.get('initial_hash',sb.get('initial_parameter_hash'))
    n=min(sa['steps'],sb['steps'])
    for f in ('batch_hashes.json','constraint_batch_hashes.json'):
        assert read(a,f)[:n]==read(b,f)[:n],f
    with np.load(a/'evaluation_points.npz') as x,np.load(b/'evaluation_points.npz') as y:
        assert x.files==y.files
        for k in x.files:np.testing.assert_array_equal(x[k],y[k])
    return n


def smoke(root):
    import numpy as np
    import torch
    from torch_pinn.model import MLP
    from torch_pinn.constraint_sampling import ConstraintSampler,as_tensors
    from train_fixed_wall import PROBLEMS,full_loss,sample
    out=root/'preflight';out.mkdir(parents=True,exist_ok=False)
    env=runtime();save(out/'environment.json',env)
    with (out/'tests.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pytest',str(HERE/'tests'),'-q'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
    gates=[]
    for c in ('kdv1d','kdv2d','ch2d'):
        dim=2 if c=='kdv1d' else 3
        model=MLP(dim,128,5,20260921,device='cuda')
        data=as_tensors(c,ConstraintSampler(c,128 if dim==2 else 16,20260919).sample_numpy(),'cuda')
        points=torch.as_tensor(sample(np.random.default_rng(20260920),400,c),device='cuda')
        vals=[];grads=[]
        for method in METHODS:
            model.zero_grad(set_to_none=True)
            v,_=full_loss(PROBLEMS[c],model,points,data,method);v.backward()
            vals.append(v.detach());grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]).detach().clone())
        torch.testing.assert_close(vals[0],vals[1],rtol=1e-3,atol=1e-5)
        torch.testing.assert_close(grads[0],grads[1],rtol=1e-3,atol=1e-5)
        gates.append(dict(case=c,loss_abs=float(abs(vals[0]-vals[1])),gradient_max_abs=float((grads[0]-grads[1]).abs().max())))
        del model,data,points,vals,grads
        torch.cuda.empty_cache()
    for protocol,cases in [('wall',('kdv1d','kdv2d','ch2d')),('steps',('kdv1d','kdv2d'))]:
        for case in cases:
            paths=[]
            for method in METHODS:
                c=dict(protocol=protocol,case=case,seed=20260919,method=method)
                p=out/name(c);child(c,p,True);paths.append(p)
            paired(*paths)
    save(out/'PASS.json',dict(status='PASS',environment=env,gates=gates,finished_at=time.time()))
    print('GPU_PREFLIGHT_PASS',flush=True)


def cell(root,index):
    env=runtime();passed=json.loads((root/'preflight/PASS.json').read_text())
    for k in ('source_commit','python','torch','numpy','gpu'):assert env[k]==passed['environment'][k]
    c=CELLS[index];out=root/'cells'/name(c)
    save(root/'environments'/f'{index:02}.json',env)
    s=child(c,out)
    save(root/'markers'/f'{index:02}.json',dict(index=index,cell=c,summary=s,finished_at=time.time()))


def aggregate(root):
    rows=[];pairs=[]
    for i,c in enumerate(CELLS):
        marker=json.loads((root/'markers'/f'{i:02}.json').read_text());assert marker['cell']==c
        out=root/'cells'/name(c);s=validate(out,c)
        row={**c,'steps':s['steps']}
        if c['protocol']=='wall':row.update(wall_seconds=s['wall_seconds'],spacetime=s['errors']['spacetime']['relative_l2'],terminal=s['errors']['terminal']['relative_l2'])
        else:row.update(wall_seconds=s['training_wall_seconds'],test=s['test']['relative_l2'])
        rows.append(row)
    for i in range(0,len(CELLS),2):
        c=CELLS[i];a,b=[root/'cells'/name(x) for x in CELLS[i:i+2]]
        n=paired(a,b)
        speedup=rows[i+1]['steps']/rows[i]['steps'] if c['protocol']=='wall' else rows[i]['wall_seconds']/rows[i+1]['wall_seconds']
        pairs.append(dict(protocol=c['protocol'],case=c['case'],seed=c['seed'],matched_batch_prefix=n,speedup=speedup))
    save(root/'per_seed_metrics.json',rows);save(root/'paired_comparisons.json',pairs)
    save(root/'COMPLETE.json',dict(status='COMPLETE',cells=len(rows),pairs=len(pairs),source_commit=json.loads((root/'preflight/PASS.json').read_text())['environment']['source_commit'],finished_at=time.time()))
    with (root/'per_seed_metrics.csv').open('w') as f:
        fields=['protocol','case','seed','method','steps','wall_seconds','spacetime','terminal','test']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print('COMPLETE_38_CELLS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('smoke','cell','aggregate'));p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int);a=p.parse_args()
    try:
        if a.mode=='smoke':smoke(a.root)
        elif a.mode=='cell':cell(a.root,a.index)
        else:aggregate(a.root)
    except Exception as e:
        save(a.root/'failures'/f'{a.mode}_{a.index}.json',dict(error=repr(e),time=time.time()))
        raise
