"""Frozen single-H20 matrix; independent processes and fail-closed correctness."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

CASES = ('kdv2d', 'gkdv1d')
METHODS = ('nested_jvp', 'shared_jet', 'termwise_jet')
SEEDS = (20260908, 20260909, 20260910)


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def compare_group(root, cells, seeds):
    rows = []
    for case, method, batch in cells:
        reference = root / f'{case}_nested_jvp_b{batch}'
        current = root / f'{case}_{method}_b{batch}'
        arows = json.loads((current / 'results.json').read_text())
        brows = json.loads((reference / 'results.json').read_text())
        if [r['seed'] for r in arows] != list(seeds) or [r['seed'] for r in brows] != list(seeds):
            raise AssertionError('incomplete seed records')
        for seed, arow, brow in zip(seeds, arows, brows):
            for key in ('params_sha256', 'points_sha256'):
                if arow[key] != brow[key]:
                    raise AssertionError('unmatched parameters or inputs')
            with np.load(current / f'values_{seed}.npz') as a, np.load(reference / f'values_{seed}.npz') as b:
                if set(a.files) != set(b.files):
                    raise AssertionError('unmatched component keys')
                for key in b.files:
                    np.testing.assert_allclose(a[key], b[key], rtol=1e-3, atol=1e-6)
                    rows.append(dict(case=case, method=method, batch=batch, seed=seed, component=key,
                                     max_abs_error=float(np.max(np.abs(a[key]-b[key])))))
    save(root / 'pairwise_validation.json', rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=3600)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    formal = [(c, m, b) for c in CASES for b in (100, 400, 1600) for m in METHODS]
    diagnostic = [(c, m, 100) for c in CASES for m in ('nested_jvp', 'nested_grad')]
    env = os.environ.copy()
    env.update(JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', JAX_DEFAULT_MATMUL_PRECISION='highest')
    env.pop('LD_LIBRARY_PATH', None)
    save(args.out / 'manifest.json', dict(protocol='pde_shared_real_no_outer_jit_v1',
         formal_cells=formal, diagnostic_cells=diagnostic, expected_formal_cells=18,
         expected_formal_rows=54, expected_formal_timed_calls=1620, formal_seeds=SEEDS,
         baseline='nested_jvp', automatic_baseline_selection=False, python=sys.executable,
         timeout_per_process_s=args.timeout, width=128, depth=4, warmups=10, repeats=30,
         diagnostic_seed=20260918, diagnostic_warmups=3, diagnostic_repeats=7,
         source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
         gpu_budget=1, process_concurrency=1, canonical_root=str(args.out.resolve())))
    outcomes = []
    status = args.out / 'status.txt'
    runner = Path(__file__).with_name('pde_benchmark.py')

    def run(name, options):
        cmd = [sys.executable, str(runner), '--require-gpu', '--out', str(args.out / name), *options]
        logpath = args.out / (name.replace('/', '_') + '.log')
        started = time.time()
        with logpath.open('w') as log:
            try:
                code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = 124
        outcomes.append(dict(cell=name, exit_code=code, elapsed_s=time.time()-started, command=cmd))
        save(args.out / 'progress.json', outcomes)
        print(json.dumps(outcomes[-1]), flush=True)
        if code != 0:
            raise RuntimeError(f'cell failed: {name}, code={code}')

    try:
        status.write_text('VERIFYING\n')
        run('verification', ['--verify'])
        status.write_text('DIAGNOSTIC\n')
        for c, m, b in diagnostic:
            run(f'diagnostic/{c}_{m}_b{b}', ['--case', c, '--method', m, '--batch', str(b),
                '--seeds', '20260918', '--warmups', '3', '--repeats', '7'])
        compare_group(args.out / 'diagnostic', diagnostic, (20260918,))
        status.write_text('RUNNING\n')
        for c, m, b in formal:
            run(f'formal/{c}_{m}_b{b}', ['--case', c, '--method', m, '--batch', str(b)])
        compare_group(args.out / 'formal', formal, SEEDS)
        status.write_text('COMPLETE\n')
    except Exception as exc:
        save(args.out / 'failure.json', {'error': repr(exc)})
        status.write_text('FAILED\n')
        raise
    finally:
        hashes = [hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + str(f.relative_to(args.out))
                  for f in sorted(args.out.rglob('*')) if f.is_file() and f.name != 'SHA256SUMS']
        (args.out / 'SHA256SUMS').write_text('\n'.join(hashes) + '\n')


if __name__ == '__main__':
    main()
