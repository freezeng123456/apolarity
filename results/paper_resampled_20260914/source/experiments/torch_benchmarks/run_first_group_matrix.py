"""Run the complete 22-cell fixed-mixed-partial PyTorch matrix sequentially."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


TARGETS = ('111111', '111222', '112233', '123456', '11223344', '111222333')
SEEDS = (20260908, 20260909, 20260910)


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha256_manifest(root: Path) -> None:
    entries = []
    for file in sorted(root.rglob('*')):
        if file.is_file() and file.name != 'SHA256SUMS':
            entries.append(f'{hashlib.sha256(file.read_bytes()).hexdigest()}  {file.relative_to(root)}')
    (root/'SHA256SUMS').write_text('\n'.join(entries)+'\n')


def matrix_cells():
    cells = [(target, method, 100) for target in TARGETS for method in ('nested_jvp', 'waring_batched')]
    cells += [(target, 'waring_serial', 100) for target in ('123456', '111222333')]
    cells += [
        (target, method, batch)
        for target in ('123456', '111222333')
        for batch in (400, 1600)
        for method in ('nested_jvp', 'waring_batched')
    ]
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    cells = matrix_cells()
    if len(cells) != 22:
        raise AssertionError(cells)
    args.out.mkdir(parents=True)
    manifest = {
        'protocol': 'torch_first_group_values_v1',
        'source_commit': args.source_commit,
        'expected_cells': len(cells),
        'expected_rows': len(cells)*len(SEEDS),
        'expected_timed_calls': len(cells)*len(SEEDS)*30,
        'seeds': list(SEEDS),
        'warmups': 10,
        'timed_repetitions_per_seed': 30,
        'baseline': 'nested_jvp',
        'automatic_baseline_selection': False,
        'execution': 'eager_no_outer_jit',
        'cells': [dict(target=target, method=method, batch=batch) for target, method, batch in cells],
        'python': sys.executable,
        'timeout_per_cell_seconds': args.timeout,
    }
    save(args.out/'manifest.json', manifest)
    (args.out/'status.txt').write_text('PREPARING\n')
    env = os.environ.copy()
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    source_dir = Path(__file__).resolve().parent
    verification = args.out/'verification'
    verification_cmd = [sys.executable, str(source_dir/'first_group_verify.py'), '--out', str(verification)]
    with (args.out/'verification.log').open('w') as log:
        check = subprocess.run(verification_cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if check != 0:
        (args.out/'status.txt').write_text('FAILED_VERIFICATION\n')
        sha256_manifest(args.out)
        raise SystemExit(check)
    (args.out/'status.txt').write_text('RUNNING\n')
    outcomes = []
    for target, method, batch in cells:
        name = f'{target}_{method}_b{batch}'
        command = [
            sys.executable, str(source_dir/'first_group.py'), '--target', target,
            '--method', method, '--batch', str(batch), '--out', str(args.out/name),
        ]
        started = time.time()
        with (args.out/f'{name}.log').open('w') as log:
            try:
                exit_code = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
                (args.out/f'{name}.timeout').write_text('cell exceeded declared wall-clock budget\n')
        outcomes.append({'cell': name, 'exit_code': exit_code, 'elapsed_seconds': time.time()-started})
        save(args.out/'progress.json', outcomes)
        print(json.dumps(outcomes[-1]), flush=True)
        if exit_code != 0:
            break
    complete = len(outcomes) == len(cells) and all(row['exit_code'] == 0 for row in outcomes)
    (args.out/'status.txt').write_text('COMPLETE\n' if complete else 'FINISHED_WITH_FAILURES\n')
    sha256_manifest(args.out)
    if not complete:
        raise SystemExit('matrix incomplete; raw artifacts preserved')


if __name__ == '__main__':
    main()
