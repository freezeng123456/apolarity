"""Run the paper's paired PDE matrix with resampled training constraints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

from torch_pinn.operators import METHODS
from torch_pinn.records import save, seal

HERE = Path(__file__).resolve().parent
CASES = ('kdv1d', 'kdv2d', 'ch2d')
SEEDS = tuple(range(20260919, 20260924))


def matrix_cells(cases=CASES, seeds=SEEDS):
    return [(case, seed, method) for case in cases for seed in seeds for method in METHODS]


def command(case, seed, method, out, device='cuda', smoke=False):
    args = [sys.executable, str(HERE/'train_fixed_wall.py'), '--out', str(out),
            '--case', case, '--method', method, '--seed', str(seed), '--device', device]
    if smoke:
        args += ['--width', '4', '--depth', '2', '--batch', '3', '--constraint-side', '2',
                 '--max-steps', '3', '--eval-points', '16', '--trajectory-seconds', '0']
    return args


def verify_pair(left, right):
    read = lambda root, name: json.loads((root/name).read_text())
    a, b = read(left, 'summary.json'), read(right, 'summary.json')
    assert a['initial_hash'] == b['initial_hash']
    assert a['constraint_sampling'] == b['constraint_sampling'] == 'resampled'
    count = min(a['steps'], b['steps'])
    for name in ('batch_hashes.json', 'constraint_batch_hashes.json'):
        x, y = read(left, name), read(right, name)
        assert len(x) == a['steps'] and len(y) == b['steps']
        assert x[:count] == y[:count]
        if name == 'constraint_batch_hashes.json':
            assert len(set(x)) == len(x) and len(set(y)) == len(y)
    with np.load(left/'evaluation_points.npz') as x, np.load(right/'evaluation_points.npz') as y:
        assert x.files == y.files
        for key in x.files: np.testing.assert_array_equal(x[key], y[key])
    return dict(case=a['case'], matched_updates=count, update_ratio=b['steps']/a['steps'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--cases', choices=CASES, nargs='+', default=list(CASES))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--smoke', action='store_true', help='Three updates on a small network; not a timing result')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without launching or writing results')
    args = parser.parse_args()
    if len(set(args.cases)) != len(args.cases) or len(set(args.seeds)) != len(args.seeds):
        parser.error('cases and seeds must be unique')
    cells = matrix_cells(args.cases, args.seeds)
    commands = [command(c,s,m,args.out/f'{c}_seed{s}_{m}',args.device,args.smoke) for c,s,m in cells]
    if args.dry_run:
        print(json.dumps(commands, indent=2)); return
    args.out.mkdir(parents=True, exist_ok=False)
    env = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'OMP_NUM_THREADS': '1'}
    save(args.out/'manifest.json',dict(protocol='resampled_paper_pde', smoke=args.smoke,
         cells=[dict(case=c,seed=s,method=m) for c,s,m in cells], commands=commands))
    try:
        for (c,s,m), cmd in zip(cells, commands):
            with (args.out/f'{c}_seed{s}_{m}.log').open('w') as log:
                subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        pairs = [verify_pair(args.out/f'{c}_seed{s}_nested_jvp',args.out/f'{c}_seed{s}_shared_jet_linear')
                 for c in args.cases for s in args.seeds]
        save(args.out/'paired_summary.json',pairs)
        (args.out/'status.txt').write_text('COMPLETE\n')
    except Exception:
        (args.out/'status.txt').write_text('FAILED\n'); raise
    finally:
        seal(args.out)


if __name__ == '__main__':
    main()
