"""One T4, selected sequential pairs, with preflight gates and final audit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True)
p.add_argument('--hidden-layers',type=int,default=3)
p.add_argument('--lr',type=float,default=1e-4)
p.add_argument('--ch-lr',type=float,default=None)
p.add_argument('--seconds',type=float,default=600)
p.add_argument('--cases',nargs='+',choices=('kdv1d','kdv2d','ch2d'),default=['kdv1d','kdv2d','ch2d'])
a=p.parse_args()
if len(a.cases)!=len(set(a.cases)):p.error('duplicate cases are not allowed')
if not np.isfinite(a.seconds) or a.seconds<=0:p.error('seconds must be positive and finite')
if a.ch_lr is not None and (not np.isfinite(a.ch_lr) or a.ch_lr<=0):p.error('CH learning rate must be positive and finite')
if not np.isfinite(a.lr) or a.lr<=0:p.error('learning rate must be positive and finite')
if a.hidden_layers<1:p.error('hidden layers must be positive')
a.out.mkdir(parents=True,exist_ok=False)
root=a.out.resolve()
here=Path(__file__).resolve().parent
cases=tuple(a.cases)
methods=('nested_jvp','shared_jet_linear')
lrs={c:(a.ch_lr if c=='ch2d' and a.ch_lr is not None else a.lr) for c in cases}

def save(name,obj):
    (root/name).write_text(json.dumps(obj,indent=2)+'\n')

def launch(case,method,smoke=False):
    cell=('smoke_' if smoke else '')+case+'_'+method
    cmd=[sys.executable,str(here/'train_fixed_wall.py'),'--case',case,'--method',method,'--out',str(root/cell),'--depth',str(a.hidden_layers+1),'--lr',str(lrs[case]),'--seconds',str(a.seconds)]
    if smoke:cmd+=['--max-steps','2','--eval-points','32','--checkpoint-seconds','600']
    save('state.json',dict(stage='SMOKE' if smoke else 'TRAINING',cell=cell,time=time.time()))
    with (root/(cell+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)

try:
    save('provenance.json',dict(run_id=root.name,root=str(root),expected_cells=2*len(cases),commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=sys.executable,seed=20260918,seconds_per_cell=a.seconds,lr_by_case=lrs,batch=400,hidden_layers=[128]*a.hidden_layers,device='one exclusive T4',dtype='float32',eval_points=10000,order=[f'{c}/{m}' for c in cases for m in methods]))
    with (root/'tests.log').open('w') as log:subprocess.run([sys.executable,'-m','pytest',str(here/'tests'),'-q'],stdout=log,stderr=subprocess.STDOUT,check=True)
    for case in cases:
        for method in methods:launch(case,method,True)
        s=[json.loads((root/f'smoke_{case}_{m}'/'summary.json').read_text()) for m in methods]
        assert s[0]['initial_hash']==s[1]['initial_hash']
        assert abs(s[0]['final_fixed_loss']-s[1]['final_fixed_loss'])<=1e-4*max(1,abs(s[0]['final_fixed_loss']))
    for case in cases:
        for method in methods:launch(case,method)
    comparisons=[]
    for case in cases:
        dirs=[root/f'{case}_{m}' for m in methods]
        s=[json.loads((d/'summary.json').read_text()) for d in dirs]
        assert all((d/'status.txt').read_text().strip()=='COMPLETE' for d in dirs)
        assert s[0]['initial_hash']==s[1]['initial_hash']
        assert all(v['steps']>0 and v['initial_hash']!=v['final_hash'] and v['wall_seconds']>=a.seconds and v['lr']==lrs[case] for v in s)
        batches=[json.loads((d/'batch_hashes.json').read_text()) for d in dirs]
        n=min(map(len,batches));assert batches[0][:n]==batches[1][:n]
        with np.load(dirs[0]/'evaluation_points.npz') as x, np.load(dirs[1]/'evaluation_points.npz') as y:
            assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
        for d in dirs:
            for line in (d/'SHA256SUMS').read_text().splitlines():
                digest,name=line.split('  ',1);assert hashlib.sha256((d/name).read_bytes()).hexdigest()==digest
        comparisons.append(dict(case=case,step_ratio_shared_over_nested=s[1]['steps']/s[0]['steps'],matched_batch_prefix=n,methods=s))
    save('paired_summary.json',comparisons)
    save('state.json',dict(stage='COMPLETE',completed_cells=2*len(cases),time=time.time()))
    (root/'exit_code.txt').write_text('0\n')
except BaseException as exc:
    save('failure.json',dict(error=repr(exc),time=time.time()))
    save('state.json',dict(stage='FAILED',time=time.time()))
    (root/'exit_code.txt').write_text('1\n')
    raise
