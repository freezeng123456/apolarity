"""Bounded sequential T4 pilot/formal matrix; formal requires an accepted pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from torch_pinn.records import save,seal
from torch_pinn.problem_kdv2d import PROTOCOL

HERE=Path(__file__).resolve().parent
METHODS=('nested_jvp','shared_jet_linear')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','formal'],required=True)
    p.add_argument('--pilot-root',type=Path)
    a=p.parse_args()
    commit=subprocess.check_output(['git','-C',str(HERE),'rev-parse','HEAD'],text=True).strip()
    source_hash={str(f.relative_to(HERE)):hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted(HERE.rglob('*.py'))}
    if a.phase=='formal':
        if not a.pilot_root: p.error('--pilot-root required for formal phase')
        accepted=json.loads((a.pilot_root/'acceptance.json').read_text())
        if accepted['gate']!='PASS' or accepted['source_commit']!=commit or accepted['source_sha256']!=source_hash:
            raise RuntimeError('pilot not accepted or source changed')
    a.out.mkdir(parents=True,exist_ok=False)
    cells=[]
    if a.phase=='pilot':
        cells.append(('tests',['-m','pytest','-q',str(HERE/'tests')],600))
        cells.append(('verify',[str(HERE/'verify_kdv2d.py'),'--device','cuda','--out',str(a.out/'verify')],600))
        cells.append(('cost',[str(HERE/'verify_kdv2d.py'),'--device','cuda','--mode','cost','--out',str(a.out/'cost')],900))
    seeds=[20260920] if a.phase=='pilot' else [20260921,20260922,20260923]
    steps=1000 if a.phase=='pilot' else 10000
    pairs=[]
    for index,seed in enumerate(seeds):
        # Counterbalance order where possible; each pair uses the same physical GPU.
        methods=METHODS if index%2==0 else METHODS[::-1]
        for method in methods:
            name=f'{method}_{seed}'
            cells.append((name,[str(HERE/'train_kdv2d.py'),'--device','cuda','--method',method,
                         '--seed',str(seed),'--steps',str(steps),'--out',str(a.out/name)],1200 if a.phase=='pilot' else 7200))
        pairs.append((seed,[f'{m}_{seed}' for m in METHODS]))
    save(a.out/'manifest.json',dict(protocol=PROTOCOL,phase=a.phase,source_commit=commit,
         source_sha256=source_hash,cells=[c[0] for c in cells],seeds=seeds,steps=steps,
         decay_steps=10000,eval_every=50,gpu_budget=1,concurrency=1,
         acceptance='Both pilots finite, final fixed loss <= 0.1 initial, final validation relative L2 <= 0.1; matched initialization, constraints and every collocation batch; no speed gate.'))
    env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    progress=[]
    try:
        (a.out/'status.txt').write_text('RUNNING\n')
        for name,command,timeout in cells:
            print('START '+name,flush=True)
            start=time.time()
            with (a.out/f'{name}.log').open('w') as f:
                try:
                    code=subprocess.run([sys.executable,*command],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
                except subprocess.TimeoutExpired: code=124
            progress.append(dict(cell=name,exit_code=code,elapsed_s=time.time()-start,command=[sys.executable,*command]))
            save(a.out/'progress.json',progress)
            print(json.dumps(progress[-1]),flush=True)
            if code: raise RuntimeError(f'{name}: exit {code}')
        pairing=[]
        for seed,names in pairs:
            summaries=[json.loads((a.out/n/'summary.json').read_text()) for n in names]
            hashes=[json.loads((a.out/n/'batch_hashes.json').read_text()) for n in names]
            assert len(hashes[0])==len(hashes[1])==steps and hashes[0]==hashes[1]
            for key in ('initial_parameter_hash','constraint_hash','evaluation_point_hash'):
                assert summaries[0][key]==summaries[1][key]
            assert all(s['steps']==steps and s['termination']=='fixed_steps' for s in summaries)
            if a.phase=='pilot':
                assert all(s['final_validation']['fixed_loss'] <= .1*s['initial_validation']['fixed_loss'] for s in summaries), 'pilot loss reduction gate'
                assert all(s['final_validation']['relative_l2'] <= .1 for s in summaries), 'pilot solution-error gate'
            pairing.append(dict(seed=seed,matched_batches=steps,summaries=summaries))
        save(a.out/'acceptance.json',dict(gate='PASS',source_commit=commit,source_sha256=source_hash,pairs=pairing))
        (a.out/'status.txt').write_text('COMPLETE\n')
    except Exception as exc:
        save(a.out/'failure.json',dict(error=repr(exc)))
        (a.out/'status.txt').write_text('FAILED\n')
        raise
    finally: seal(a.out)


if __name__=='__main__': main()
