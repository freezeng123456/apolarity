"""Bounded migration acceptance: tests, GPU gate, B=400 steps, matched short training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from torch_pinn.records import save,seal


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    here=Path(__file__).resolve().parent
    cells=[('tests',['-m','pytest','-q',str(here/'tests')],300),
           ('verification',[str(here/'verify.py'),'--device','cuda','--out',str(args.out/'verification')],600)]
    for case in ('kdv2d','gkdv1d'):
        for method in ('nested_jvp','shared_jet_linear'):
            name=f'benchmark_{case}_{method}'
            cells.append((name,[str(here/'benchmark.py'),'--case',case,'--method',method,'--device','cuda','--out',str(args.out/name)],600))
    for method in ('nested_jvp','shared_jet_linear'):
        name=f'pilot_{method}'
        cells.append((name,[str(here/'train.py'),'--method',method,'--device','cuda','--out',str(args.out/name)],900))
    save(args.out/'manifest.json',dict(protocol='torch_eager_migration_v1',
        commit=subprocess.check_output(['git','-C',str(here),'rev-parse','HEAD'],text=True).strip(),
        cells=[name for name,_,_ in cells],expected_gpu_verifications=18,
        expected_update_records=9,updates_per_record=20,expected_benchmark_records=36,
        expected_benchmark_samples=540,pilot_seed=20260918,pilot_steps=1000,pilot_time_cap_s=600,
        training_is_acceptance_not_formal_convergence_study=True,gpu_budget=1,concurrency=1,compile=False))
    progress=[]
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    status=args.out/'status.txt'
    try:
        status.write_text('RUNNING\n')
        for name,command,timeout in cells:
            start=time.time()
            with (args.out/(name+'.log')).open('w') as f:
                try: code=subprocess.run([sys.executable,*command],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
                except subprocess.TimeoutExpired:code=124
            progress.append(dict(cell=name,exit_code=code,elapsed_s=time.time()-start,command=[sys.executable,*command]))
            save(args.out/'progress.json',progress);print(json.dumps(progress[-1]),flush=True)
            if code:raise RuntimeError(f'{name} failed: {code}')
        a=json.loads((args.out/'pilot_nested_jvp'/'batch_hashes.json').read_text())
        b=json.loads((args.out/'pilot_shared_jet_linear'/'batch_hashes.json').read_text())
        count=min(len(a),len(b))
        if a[:count]!=b[:count]:raise AssertionError('unmatched training batches')
        save(args.out/'acceptance.json',dict(execution='PASS',matched_training_steps=count,
             formal_speed_or_convergence_claim=False,pilots={m:json.loads((args.out/f'pilot_{m}'/'summary.json').read_text())
                 for m in ('nested_jvp','shared_jet_linear')}))
        status.write_text('COMPLETE\n')
    except Exception as exc:
        save(args.out/'failure.json',dict(error=repr(exc)));status.write_text('FAILED\n');raise
    finally:seal(args.out)


if __name__=='__main__':main()
