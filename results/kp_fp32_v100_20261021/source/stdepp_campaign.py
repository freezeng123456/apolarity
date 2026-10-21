"""Matched V100 reproduction of STDE++ on every current manuscript case.

The existing equal-update trainer, sample streams, loss weights and timing
scope are reused. Only the interior derivative backend changes for the PDE
cells. No network or optimizer weights are written.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib
import json
from math import prod
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

import train_equal_updates as trainer
from equal_updates_campaign import INTERVALS, SEEDS, TARGETS, UPDATES
from first_group import make_model_and_points, parameters_sha256, parse_one_based_pattern, tensor_sha256
from torch_pinn.model import MLP
from torch_pinn.monomials import monomial, schedule
from torch_pinn.records import run_record, save, seal, sync

HERE = Path(__file__).resolve().parent
PROTOCOL = 'apolarity_stdepp_matched_v1'
PARTIAL_METHODS = ('nested_jvp', 'waring_batched', 'stdepp')
PARTIAL_SEEDS = (20260908, 20260909, 20260910)


def _stdepp():
    return importlib.import_module('torch_pinn.stdepp')


def source_fingerprint():
    files = sorted(HERE.rglob('*.py')) + [HERE/'stdepp_worker.sh']
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(HERE)).encode()+b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def matrix(output):
    output = Path(output)
    cells = []
    for target in TARGETS:
        for method in PARTIAL_METHODS:
            cells.append(dict(kind='partial', name=f'{target}_{method}', target=target,
                method=method, batch=100, width=128, depth=4,
                seeds=list(PARTIAL_SEEDS), warmups=10, repeats=30))
    for case, updates in UPDATES.items():
        for seed in SEEDS:
            cells.append(dict(kind='pde', name=f'{case}_seed{seed}_stdepp', case=case,
                method='stdepp', seed=seed, updates=updates, eval_every=INTERVALS[case],
                batch=400, width=128, depth=5,
                constraint_side=128 if case == 'kdv1d' else 16,
                eval_points=10000, eval_seed_base=20261400,
                lr=1e-4 if case == 'ch2d' else 1e-5))
    for index, cell in enumerate(cells):
        cell['index'] = index
        cell['out'] = str(output/('partials' if cell['kind'] == 'partial' else 'pde')/cell['name'])
    return cells


def plan(output):
    return dict(protocol=PROTOCOL, source_fingerprint=source_fingerprint(), cells=matrix(output),
        allocation=dict(partition='gpu', nodes=1, ntasks=1, gpus=1, cpus_per_task=10,
                        explicit_memory_request=False),
        execution=dict(gpu='Tesla V100-SXM2-32GB', torch='2.5.1+cu121', numpy='1.26.4',
                       python='3.11.7', cuda='12.1', threads=1, dtype='float32',
                       mode='eager', tf32=False, autocast=False),
        stdepp_variant='Practical deterministic multi-jet construction, Section 6.1; '
                       'not the theoretical prime-polynomial construction.',
        stdepp_pde_constructions=dict(kdv1d='Two order-seven jets and one order-two jet',
            kdv2d='One order-nine jet and two order-three jets',
            ch2d='Four real-polarization jets, adapted to the present CH objective'),
        partial_cells=18, pde_cells=15, checkpoints=False,
        pde_reference='Existing 30 V100 equal-update runs; same initialization and sample hashes.',
        partial_reference='Rerun Nested JVP and WDD together with STDE++ on V100; original table uses T4.')


@contextmanager
def stdepp_training_dispatch():
    previous = trainer.full_loss

    def dispatch(module, model, points, data, method, independent=False):
        if method != 'stdepp' or independent:
            return previous(module, model, points, data, method, independent=independent)
        case = next(key for key, value in trainer.PROBLEMS.items() if value is module)
        return _stdepp().pde_loss(case, model, points, data)

    trainer.full_loss = dispatch
    try:
        yield
    finally:
        trainer.full_loss = previous


def pde_args(cell, device='cuda'):
    # Parse through the original runner to preserve every original default.
    args = trainer.parse_args(['--case', cell['case'], '--method', 'nested_jvp',
        '--out', cell['out'], '--device', device, '--expected-device', 'V100',
        '--seed', str(cell['seed']), '--updates', str(cell['updates']),
        '--eval-every', str(cell['eval_every']), '--width', str(cell['width']),
        '--depth', str(cell['depth']), '--batch', str(cell['batch']),
        '--constraint-side', str(cell['constraint_side']),
        '--eval-points', str(cell['eval_points']), '--eval-seed-base', str(cell['eval_seed_base']),
        '--lr', str(cell['lr'])])
    args.method = cell['method']
    args.campaign_protocol = PROTOCOL
    if args.method == 'stdepp':
        metadata = getattr(_stdepp(), 'metadata', None)
        if metadata is not None:
            args.algorithm_metadata = metadata()
    return args


def run_pde(cell, device='cuda'):
    args = pde_args(cell, device)
    with stdepp_training_dispatch():
        run_record(args, trainer.train)


def partial_values(model, points, alpha, method):
    if method == 'stdepp':
        return _stdepp().partial(model, points, alpha)
    return monomial(model, points, alpha, method)


def _partial_body(args):
    alpha = parse_one_based_pattern(args.target)
    directions, _ = schedule(alpha, 16, dtype=torch.float32, device=args.device)
    card_path = args.out/'config.json'
    card = json.loads(card_path.read_text())
    card.update(protocol=PROTOCOL, input_dim=16, order=len(alpha),
        waring_minimum_complex_direction_count=len(directions),
        parameter_backward=False, outer_jit=False, torch_compile=False,
        input_distribution='normal(mean=0,std=0.35) in all 16 coordinates',
        initialization='NumPy PCG64 uniform[-1/sqrt(fan_in),+1/sqrt(fan_in)]',
        cuda_synchronized_timing=True)
    if args.method == 'stdepp':
        card['algorithm_metadata'] = _stdepp().metadata(alpha)
        card['direction_count'] = card['algorithm_metadata']['direction_count']
    elif args.method == 'waring_batched':
        card['direction_count'] = len(directions)
    records = []
    cold_model, cold_points = make_model_and_points(args.seeds[0], args.batch, args.width, args.depth, args.device)
    with torch.no_grad():
        sync(args.device)
        began = time.perf_counter()
        cold_value = partial_values(cold_model, cold_points, alpha, args.method)
        sync(args.device)
        card['cold_call_ms_excluded_from_timing'] = 1000*(time.perf_counter()-began)
        if cold_value.requires_grad or cold_value.shape != (args.batch,) or not bool(torch.isfinite(cold_value).all()):
            raise AssertionError('Invalid value-only cold call')
    save(card_path, card)
    del cold_model, cold_points, cold_value
    for seed in args.seeds:
        model, points = make_model_and_points(seed, args.batch, args.width, args.depth, args.device)
        parameter_hash, points_hash = parameters_sha256(model), tensor_sha256(points)
        with torch.no_grad():
            for _ in range(args.warmups):
                partial_values(model, points, alpha, args.method)
            sync(args.device)
            samples = []
            for _ in range(args.repeats):
                sync(args.device)
                began = time.perf_counter()
                value = partial_values(model, points, alpha, args.method)
                sync(args.device)
                samples.append(1000*(time.perf_counter()-began))
        if value.requires_grad or value.shape != (args.batch,) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise AssertionError('Invalid value-only timed result')
        array = value.detach().cpu().numpy()
        np.save(args.out/f'values_{seed}.npy', array)
        records.append(dict(target=args.target, method=args.method, seed=seed, batch=args.batch,
            order=len(alpha), parameters_sha256=parameter_hash, points_sha256=points_hash,
            output_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
            output_dtype=str(array.dtype), output_shape=list(array.shape),
            times_ms=samples, median_ms=statistics.median(samples)))
        save(args.out/'results.json', records)
        print(f'{args.target}/{args.method} seed={seed} median_ms={records[-1]["median_ms"]:.6f}', flush=True)


def run_partial(cell, device='cuda'):
    args = argparse.Namespace(**{key: cell[key] for key in
        ('target', 'method', 'batch', 'width', 'depth', 'seeds', 'warmups', 'repeats')},
        out=Path(cell['out']), device=device, expected_device='V100', campaign_protocol=PROTOCOL)
    run_record(args, _partial_body)


def _tanh_derivative(value, order):
    """Independent polynomial recurrence P_{n+1}(t)=(1-t^2)P'_n(t)."""
    coefficients = [0, 1]
    for _ in range(order):
        derivative = [(index+1)*entry for index, entry in enumerate(coefficients[1:])]
        next_coefficients = [0]*(len(derivative)+2)
        for index, entry in enumerate(derivative):
            next_coefficients[index] += entry
            next_coefficients[index+2] -= entry
        coefficients = next_coefficients
    result = torch.zeros_like(value)
    for entry in reversed(coefficients):
        result = result*value+entry
    return result


def correctness_checks(device='cpu'):
    rows = []
    for target in TARGETS:
        alpha = parse_one_based_pattern(target)
        model = MLP(16, 1, 2, 20260910, dtype=torch.float64, device=device)
        with torch.no_grad():
            model.layers[0].weight.copy_(torch.arange(16, dtype=torch.float64, device=device).reshape(1, -1)*.02+.5)
            model.layers[0].bias.fill_(.2)
            model.layers[1].weight.fill_(.7)
            model.layers[1].bias.zero_()
            points = torch.tensor([[-.03]*16, [0.]*16, [.03]*16], dtype=torch.float64, device=device)
            z = torch.tanh(model.layers[0](points)[:, 0])
            expected = .7*_tanh_derivative(z, len(alpha))*prod(model.layers[0].weight[0, axis] for axis in alpha)
            observed = partial_values(model, points, alpha, 'stdepp')
            torch.testing.assert_close(observed, expected, rtol=1e-7, atol=1e-9)
            # An independently initialized small nonlinear network also checks
            # arbitrary weights, biases and axis patterns against coordinate JVP.
            tiny = MLP(16, 2, 2, 20260908, dtype=torch.float64, device=device)
            nested = partial_values(tiny, points[:1], alpha, 'nested_jvp')
            actual = partial_values(tiny, points[:1], alpha, 'stdepp')
            torch.testing.assert_close(actual, nested, rtol=1e-7, atol=1e-10)
        rows.append(dict(kind='partial', target=target,
                         analytic_max_abs=float((observed-expected).abs().max()),
                         nested_max_abs=float((actual-nested).abs().max())))
    for case, module in trainer.PROBLEMS.items():
        dim = 2 if case == 'kdv1d' else 3
        reference = MLP(dim, 4, 2, 20260921, dtype=torch.float64, device=device)
        candidate = deepcopy(reference)
        points = torch.as_tensor(trainer.sample(np.random.default_rng(20260920), 3, case),
                                 dtype=torch.float64, device=device)
        data = module.constraints(2, dtype=torch.float64, device=device)
        baseline, baseline_parts = trainer.full_loss(module, reference, points, data, 'nested_jvp')
        observed, parts = _stdepp().pde_loss(case, candidate, points, data)
        torch.testing.assert_close(observed, baseline, rtol=1e-7, atol=1e-9)
        baseline.backward()
        observed.backward()
        max_grad = 0.
        for left, right in zip(reference.parameters(), candidate.parameters()):
            if left.grad is None or right.grad is None:
                raise AssertionError('Missing parameter gradient')
            torch.testing.assert_close(right.grad, left.grad, rtol=1e-6, atol=1e-9)
            max_grad = max(max_grad, float((right.grad-left.grad).abs().max()))
        # Validate that the derivative graph survives a real optimizer update.
        for model in (reference, candidate):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 if case == 'ch2d' else 1e-5,
                                         betas=(.9, .999), eps=1e-8, foreach=False)
            optimizer.step()
        for left, right in zip(reference.parameters(), candidate.parameters()):
            torch.testing.assert_close(right, left, rtol=1e-7, atol=1e-9)
        rows.append(dict(kind='pde', case=case, loss=float(observed.detach()),
                         reference_loss=float(baseline.detach()), max_abs_gradient_error=max_grad))
    return rows


def gpu_environment():
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != 'Tesla V100-SXM2-32GB':
        raise RuntimeError('Expected one allocated Tesla V100-SXM2-32GB')
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one visible allocated GPU')
    if torch.__version__ != '2.5.1+cu121' or np.__version__ != '1.26.4' or platform.python_version() != '3.11.7':
        raise RuntimeError('Expected the original pinned Python/Torch/NumPy environment')
    import os
    if os.environ.get('SLURM_CPUS_PER_TASK') != '10' or torch.get_num_threads() != 1:
        raise RuntimeError('Expected ten allocated CPUs and one active PyTorch CPU thread')
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), threads=torch.get_num_threads(),
        scheduler={name: os.environ.get(name) for name in
                   ('SLURM_JOB_ID', 'SLURM_CPUS_PER_TASK', 'SLURM_JOB_GPUS', 'CUDA_VISIBLE_DEVICES')},
        lscpu=subprocess.check_output(['lscpu'], text=True),
        nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True))


def preflight(output):
    gate = Path(output)/'preflight'
    gate.mkdir(parents=True, exist_ok=False)
    from torch_pinn.records import configure
    configure('cuda', 'V100')
    environment = gpu_environment()
    save(gate/'environment.json', environment)
    rows = correctness_checks('cuda')
    save(gate/'correctness.json', dict(status='PASS', checks=rows))
    estimates = []
    for cell in matrix(output):
        if cell['kind'] != 'pde' or cell['seed'] != SEEDS[0]:
            continue
        smoke = {**cell, 'updates': 3, 'eval_every': 3, 'eval_points': 16,
                 'out': str(gate/cell['case'])}
        run_pde(smoke)
        summary = json.loads((Path(smoke['out'])/'summary.json').read_text())
        if summary['steps'] != 3 or summary['initial_hash'] == summary['final_hash']:
            raise AssertionError('Full-size update smoke failed')
        estimates.append(dict(case=cell['case'], measured_updates=3,
            seconds_per_update=summary['wall_seconds']/3,
            estimated_training_seconds=summary['wall_seconds']/3*cell['updates']))
    # Exercise every formal partial backend and direction size once. This is
    # deliberately excluded from performance reporting and does not run the
    # expensive nested order-eight/order-nine formal workloads during the gate.
    formal_partial_checks = []
    for target in TARGETS:
        alpha = parse_one_based_pattern(target)
        model, points = make_model_and_points(PARTIAL_SEEDS[0], 100, 128, 4, 'cuda')
        with torch.no_grad():
            values = {}
            for method in ('waring_batched', 'stdepp'):
                value = partial_values(model, points, alpha, method)
                if value.shape != (100,) or not bool(torch.isfinite(value).all()):
                    raise AssertionError((target, method, 'formal shape or finiteness failure'))
                sync('cuda')
                values[method] = value
            torch.testing.assert_close(values['stdepp'], values['waring_batched'], rtol=1e-3, atol=1e-6)
            formal_partial_checks.append(dict(target=target,
                max_abs=float((values['stdepp']-values['waring_batched']).abs().max()),
                relative_l2=float(torch.linalg.vector_norm(values['stdepp']-values['waring_batched'])/
                                  torch.linalg.vector_norm(values['waring_batched']).clamp_min(1e-30))))
    forbidden = [path for path in gate.rglob('*') if path.suffix in ('.pt', '.pth', '.ckpt', '.npz')]
    if forbidden:
        raise AssertionError(f'Unexpected weight/archive files: {forbidden}')
    save(gate/'PASS.json', dict(status='PASS', protocol=PROTOCOL,
        source_fingerprint=source_fingerprint(), environment=environment,
        correctness_checks=len(rows), full_size_pde_smokes=3, full_size_partial_smokes=12,
        formal_partial_checks=formal_partial_checks, estimates=estimates, checkpoints=False))
    seal(gate)
    print(json.dumps(dict(status='PASS', estimates=estimates)), flush=True)


def check_gate(output):
    path = Path(output)/'preflight/PASS.json'
    if not path.is_file():
        raise RuntimeError('A successful V100 preflight is required before formal cells')
    gate = json.loads(path.read_text())
    if gate.get('status') != 'PASS' or gate.get('protocol') != PROTOCOL:
        raise RuntimeError('Invalid preflight receipt')
    if gate.get('source_fingerprint') != source_fingerprint():
        raise RuntimeError('Source changed after preflight')
    if gate.get('environment', {}).get('gpu') != 'Tesla V100-SXM2-32GB':
        raise RuntimeError('Preflight was not performed on the matched V100')
    return gate


def run_cell(output, index):
    check_gate(output)
    gpu_environment()
    cells = matrix(output)
    if not 0 <= index < len(cells):
        raise ValueError('Cell index outside the campaign matrix')
    cell = cells[index]
    root = Path(cell['out'])
    if root.exists():
        raise FileExistsError(f'Inspect existing cell before resubmission: {root}')
    (run_partial if cell['kind'] == 'partial' else run_pde)(cell)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'preflight', 'cell'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args(argv)
    if args.action == 'plan':
        print(json.dumps(plan(args.out), indent=2))
    elif args.action == 'preflight':
        preflight(args.out)
    else:
        if args.index is None:
            parser.error('--index is required for cell')
        run_cell(args.out, args.index)


if __name__ == '__main__':
    main()
