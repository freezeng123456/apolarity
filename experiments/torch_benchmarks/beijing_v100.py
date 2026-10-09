"""Bounded V100 reproduction of the manuscript tables, with retained raw data."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGETS = ('111111', '111222', '112233', '123456', '11223344', '111222333')
PARTIAL_SEEDS = (20260908, 20260909, 20260910)
PDE_SEEDS = tuple(range(20260919, 20260924))
CASES = ('kdv1d', 'kdv2d', 'ch2d')
METHODS = ('nested_jvp', 'shared_jet_linear')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(path.read_text())


def source():
    paths = sorted(HERE.rglob('*.py'))
    paths += [HERE/'beijing_worker.sh', HERE/'requirements-v100.txt']
    paths += [ROOT/'docs/paper/verify_experiment_identities.py']
    paths += sorted((ROOT/'results/paper_resampled_20260914/source').rglob('*.py'))
    export = ROOT/'SOURCE_MANIFEST.json'
    if export.is_file():
        identity = read(export)
        for name, sha in identity['files'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha, name
        base_commit = identity['source_commit']
    else:
        base_commit = subprocess.check_output(
            ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    return dict(base_commit=base_commit,
        files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def partial_command(target, method, out):
    return [sys.executable, str(HERE/'first_group.py'), '--target', target,
            '--method', method, '--batch', '100', '--width', '128', '--depth', '4',
            '--seeds', *map(str, PARTIAL_SEEDS), '--warmups', '10', '--repeats', '30',
            '--out', str(out)]


def pde_command(case, seed, method, out, *, smoke=False):
    cmd = [sys.executable, str(HERE/'train_fixed_wall.py'), '--case', case,
           '--method', method, '--seed', str(seed), '--device', 'cuda',
           '--expected-device', 'V100', '--width', '128', '--depth', '5',
           '--batch', '400', '--seconds', '1200', '--constraint-sampling', 'resampled',
           '--constraint-side', '128' if case == 'kdv1d' else '16',
           '--lr', '0.0001' if case == 'ch2d' else '0.00001',
           '--eval-seed-base', '20261400', '--eval-points', '10000',
           '--checkpoint-seconds', '10', '--trajectory-seconds', '2', '--out', str(out)]
    if smoke:
        cmd += ['--width', '4', '--depth', '2', '--batch', '3', '--constraint-side', '2',
                '--max-steps', '3', '--eval-points', '16', '--trajectory-seconds', '0']
    return cmd


def plan(out):
    return dict(protocol='apolarity_paper_v100_20261001', source=source(),
        partial_cells=[dict(target=t, method=m, command=partial_command(t, m, out/'partials'/f'{t}_{m}'))
                       for t in TARGETS for m in ('nested_jvp', 'waring_batched')],
        pde_cells=[dict(case=c, seed=s, method=m,
                       command=pde_command(c, s, m, out/'pde'/f'{c}_seed{s}_{m}'))
                   for c in CASES for s in PDE_SEEDS for m in METHODS],
        pde_training_gpu_hours=10, partial_records=36, timed_partial_calls=1080,
        precision='float32; complex64 monomial directions where required',
        execution='eager; TF32 off; one experiment at a time on the allocated V100',
        retention='All checkpoints, predictions, timing samples and sampling hashes retained')


def gpu_environment():
    import numpy as np
    import torch
    if not torch.cuda.is_available() or 'V100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('This campaign requires an allocated V100')
    if torch.__version__ != '2.5.1+cu121' or np.__version__ != '1.26.4':
        raise RuntimeError('Use pinned torch==2.5.1+cu121 and numpy==1.26.4')
    return dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                capability=list(torch.cuda.get_device_capability(0)),
                scheduler={k: os.environ.get(k) for k in
                    ('SLURM_JOB_ID', 'SLURM_JOB_GPUS', 'CUDA_VISIBLE_DEVICES')},
                packages=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True),
                nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True))


def run(cmd, log, timeout):
    log.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(command=cmd, log=str(log))), flush=True)
    env = {**os.environ, 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONUNBUFFERED': '1', 'PYTHONDONTWRITEBYTECODE': '1'}
    with log.open('x') as stream:
        subprocess.run(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)


@contextmanager
def locked(out):
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out/'campaign.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stream.seek(0); stream.truncate(); stream.write(str(os.getpid()) + '\n'); stream.flush()
        yield


def resume_check(cell, device='cuda'):
    """Verify a saved small-run model, Adam state and RNG can make an update."""
    import numpy as np
    import torch
    from torch_pinn.constraint_sampling import ConstraintSampler, as_tensors
    from torch_pinn.model import MLP
    from torch_pinn.records import configure, digest
    from train_fixed_wall import PROBLEMS, full_loss, sample
    config = read(cell/'config.json')
    configure(device, 'V100')
    state = torch.load(cell/read(cell/'checkpoints.json')[-1]['file'], map_location=device, weights_only=False)
    model = MLP(2 if config['case'] == 'kdv1d' else 3, 4, 2, config['seed']+2, device=device)
    model.load_state_dict(state['model'])
    opt = torch.optim.Adam(model.parameters(), lr=config['lr'], foreach=False)
    opt.load_state_dict(state['optimizer'])
    rng = np.random.default_rng(); rng.bit_generator.state = state['sampler_state']
    sampler = ConstraintSampler(config['case'], 2, config['seed'])
    sampler.rng.bit_generator.state = state['constraint_sampler_state']
    points = torch.as_tensor(sample(rng, 3, config['case']), device=device)
    data = as_tensors(config['case'], sampler.sample_numpy(), device)
    initial = digest(model.parameters())
    assert initial == read(cell/'summary.json')['final_hash']
    value, _ = full_loss(PROBLEMS[config['case']], model, points, data, config['method'])
    value.backward()
    assert bool(torch.isfinite(value)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    opt.step()
    assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
    assert digest(model.parameters()) != initial
    assert all(int(s['step']) == state['step'] + 1 for s in opt.state.values())
    return dict(case=config['case'], method=config['method'], restored_step=state['step'],
                resumed_loss=float(value), changed_parameters=True)


def preflight(out):
    from run_fixed_wall_matrix import verify_pair
    gate = out/'preflight'
    gate.mkdir(exist_ok=False)
    save(gate/'environment.json', gpu_environment())
    run([sys.executable, '-m', 'pytest', '-q', 'experiments/torch_benchmarks/tests'], gate/'tests.log', 1800)
    run([sys.executable, str(HERE/'first_group_verify.py'), '--out', str(gate/'partials')],
        gate/'partials.log', 1800)
    resumed = []
    for case in CASES:
        for method in METHODS:
            cell = gate/f'{case}_{method}'
            run(pde_command(case, PDE_SEEDS[0], method, cell, smoke=True), gate/f'{cell.name}.log', 600)
            resumed.append(resume_check(cell))
        verify_pair(gate/f'{case}_nested_jvp', gate/f'{case}_shared_jet_linear')
    save(gate/'completion.json', dict(status='PASS', source=source(), resume_checks=resumed))


def validate_partials(out):
    import numpy as np
    rows = []
    for target in TARGETS:
        roots = [out/'partials'/f'{target}_{m}' for m in ('nested_jvp', 'waring_batched')]
        records = [read(p/'results.json') for p in roots]
        for p, rec in zip(roots, records):
            assert (p/'status.txt').read_text().strip() == 'COMPLETE'
            cfg = read(p/'config.json')
            assert 'V100' in cfg['device_kind'] and cfg['torch'] == '2.5.1+cu121'
            assert cfg['batch'] == 100 and cfg['width'] == 128 and cfg['depth'] == 4
            assert [r['seed'] for r in rec] == list(PARTIAL_SEEDS)
            assert all(len(r['times_ms']) == 30 for r in rec)
        errors = []
        for a, b in zip(*records):
            assert a['parameters_sha256'] == b['parameters_sha256']
            assert a['points_sha256'] == b['points_sha256']
            av, bv = [np.load(p/f"values_{a['seed']}.npy") for p in roots]
            assert np.isfinite(av).all() and np.isfinite(bv).all()
            rel = float(np.linalg.norm(av-bv)/max(np.linalg.norm(av), 1e-30))
            assert rel < 1e-3, (target, a['seed'], rel)
            errors.append(rel)
        medians = [[r['median_ms'] for r in rec] for rec in records]
        rows.append(dict(target=target, baseline_mean_ms=statistics.mean(medians[0]),
            baseline_sd_ms=statistics.stdev(medians[0]), wdd_mean_ms=statistics.mean(medians[1]),
            wdd_sd_ms=statistics.stdev(medians[1]),
            speedup=statistics.mean(medians[0])/statistics.mean(medians[1]), max_relative_l2=max(errors)))
    return rows


def audit(out):
    from run_fixed_wall_matrix import verify_pair
    partials = validate_partials(out)
    pairs = []
    for case in CASES:
        for seed in PDE_SEEDS:
            roots = [out/'pde'/f'{case}_seed{seed}_{m}' for m in METHODS]
            for cell in roots:
                assert (cell/'status.txt').read_text().strip() == 'COMPLETE'
                cfg, summary = read(cell/'config.json'), read(cell/'summary.json')
                assert cfg['expected_device'] == 'V100' and cfg['seconds'] == 1200
                assert cfg['max_steps'] == 0 and summary['terminal'] == 'wall_time_budget'
                assert summary['initial_hash'] != summary['final_hash']
                assert summary['wall_seconds'] >= 1200
            pairs.append({**verify_pair(*roots), 'seed':seed})
    save(out/'audit.json', dict(status='PASS', partials=partials, pde_pairs=pairs,
                              source=source(), completed_unix=time.time()))
    with (out/'partial_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(partials[0]))
        writer.writeheader(); writer.writerows(partials)


def seal_results(out):
    # Scheduler logs can still be open; seal experiment evidence only.
    lines = []
    for folder in ('preflight', 'partials', 'pde'):
        for path in sorted((out/folder).rglob('*')):
            if path.is_file():
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024*1024), b''):
                        digest.update(block)
                lines.append(f'{digest.hexdigest()}  {path.relative_to(out)}')
    (out/'EVIDENCE_SHA256SUMS').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'preflight', 'run', 'audit'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == 'plan':
        print(json.dumps(plan(out), indent=2)); return
    with locked(out):
        identity = source()
        if (out/'manifest.json').exists():
            assert read(out/'manifest.json')['source'] == identity, 'Campaign source changed'
        else:
            save(out/'manifest.json', plan(out))
        if args.stage == 'preflight':
            preflight(out); return
        assert read(out/'preflight/completion.json')['source'] == identity
        save(out/f'environment_{args.stage}_{int(time.time())}.json', gpu_environment())
        if args.stage == 'run':
            for group, cells in (('partials', plan(out)['partial_cells']), ('pde', plan(out)['pde_cells'])):
                for cell in cells:
                    cmd = cell['command']; destination = Path(cmd[cmd.index('--out')+1])
                    status = destination/'status.txt'
                    if status.exists() and status.read_text().strip() == 'COMPLETE':
                        print(f'Already complete: {destination}', flush=True); continue
                    if destination.exists():
                        raise RuntimeError(f'Incomplete artifacts retained; inspect before retry: {destination}')
                    run(cmd, out/'logs'/f'{group}_{destination.name}.log', 3600)
        audit(out)
        seal_results(out)
        (out/'status.txt').write_text('COMPLETE\n')


if __name__ == '__main__':
    main()
