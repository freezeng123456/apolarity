"""Exact-update PDE matrix and memory-only supplement; no timing rerun."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
UPDATES = {'kdv1d': 25000, 'kdv2d': 20000, 'ch2d': 1500}
INTERVALS = {'kdv1d': 100, 'kdv2d': 100, 'ch2d': 10}
SEEDS = tuple(range(20260919, 20260924))
METHODS = ('nested_jvp', 'shared_jet_linear')
TARGETS = ('111111', '111222', '112233', '123456', '11223344', '111222333')


def matrix(output, python=sys.executable):
    cells = []
    for case, updates in UPDATES.items():
        for seed in SEEDS:
            for method in METHODS:
                name = f'{case}_seed{seed}_{method}'
                cells.append(dict(kind='pde', name=name, case=case, seed=seed, method=method,
                    updates=updates, command=[python, str(HERE/'train_equal_updates.py'),
                    '--case', case, '--method', method, '--seed', str(seed),
                    '--updates', str(updates), '--eval-every', str(INTERVALS[case]),
                    '--out', str(output/'pde'/name)]))
    for target in TARGETS:
        for method in ('nested_jvp', 'waring_batched'):
            name = f'{target}_{method}'
            cells.append(dict(kind='memory', name=name, target=target, method=method,
                command=[python, str(HERE/'measure_partial_memory.py'), '--target', target,
                         '--method', method, '--seconds', '30', '--batch', '100',
                         '--out', str(output/'memory'/name)]))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'cell'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    cells = matrix(args.out)
    if args.action == 'plan':
        print(json.dumps(dict(protocol='apolarity_equal_updates_v1', cells=cells,
            checkpoints=False, section_4_1_timing_rerun=False), indent=2))
        return
    if args.index is None or not 0 <= args.index < len(cells):
        parser.error('valid cell index required')
    cell = cells[args.index]
    root = args.out/('pde' if cell['kind'] == 'pde' else 'memory')/cell['name']
    if root.exists():
        if (root/'status.txt').is_file() and (root/'status.txt').read_text().strip() == 'COMPLETE':
            print('ALREADY_COMPLETE', cell['name'])
            return
        raise FileExistsError(f'Existing incomplete cell must be inspected: {root}')
    subprocess.run(cell['command'], check=True)


if __name__ == '__main__':
    main()
