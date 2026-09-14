"""One GPU, sequential cells, complete recovery for the paired ten-minute run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from torch_pinn.records import save, seal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    methods = ('nested_jvp', 'shared_jet_linear')
    cells = [('tests', ['-m', 'pytest', '-q', str(here / 'tests')], 300)]
    for method in methods:
        name = 'smoke_' + method
        cells.append((name, [str(here / 'train_equal_time.py'), '--device', 'cuda', '--method', method,
                     '--out', str(args.out / name), '--seconds', '1', '--eval-seconds', '.5',
                     '--width', '4', '--depth', '2', '--batch', '3', '--constraints', '3',
                     '--val-nx', '3', '--val-nt', '3'], 120))
    for method in methods:
        cells.append((method, [str(here / 'train_equal_time.py'), '--device', 'cuda', '--method', method,
                      '--out', str(args.out / method), '--seconds', '600', '--eval-seconds', '10'], 780))
    save(args.out / 'manifest.json', dict(protocol='pinn_equal_training_wall_time_v1',
         source_commit=subprocess.check_output(['git', '-C', str(here), 'rev-parse', 'HEAD'], text=True).strip(),
         cells=[name for name, _, _ in cells], formal_methods=list(methods), seconds_per_method=600,
         seed=20260918, gpu_budget=1, concurrency=1, compile=False,
         primary_metric='independent fixed-point full loss vs actual training wall time'))
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    progress = []
    status = args.out / 'status.txt'
    try:
        status.write_text('RUNNING\n')
        for name, command, timeout in cells:
            print('START ' + name, flush=True)
            start = time.time()
            with (args.out / (name + '.log')).open('w') as stream:
                try:
                    code = subprocess.run([sys.executable, *command], env=env, stdout=stream,
                                          stderr=subprocess.STDOUT, timeout=timeout).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            progress.append(dict(cell=name, exit_code=code, elapsed_s=time.time()-start,
                                 command=[sys.executable, *command]))
            save(args.out / 'progress.json', progress)
            print(json.dumps(progress[-1]), flush=True)
            if code:
                raise RuntimeError(f'{name} failed: {code}')
        summaries = {m: json.loads((args.out / m / 'summary.json').read_text()) for m in methods}
        hashes = [json.loads((args.out / m / 'batch_hashes.json').read_text()) for m in methods]
        count = min(map(len, hashes))
        assert hashes[0][:count] == hashes[1][:count]
        assert len({s['initial_parameter_hash'] for s in summaries.values()}) == 1
        assert len({s['evaluation_point_hash'] for s in summaries.values()}) == 1
        assert all(s['training_wall_seconds'] >= 600 and s['termination'] == 'wall_time_budget' for s in summaries.values())
        save(args.out / 'acceptance.json', dict(execution='PASS', matched_training_prefix=count,
             summaries=summaries, statistical_multi_seed_claim=False))
        status.write_text('COMPLETE\n')
    except Exception as exc:
        save(args.out / 'failure.json', dict(error=repr(exc)))
        status.write_text('FAILED\n')
        raise
    finally:
        seal(args.out)


if __name__ == '__main__':
    main()
