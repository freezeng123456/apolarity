"""Paired WDD/STDE++ values-only timing with schedules prepared before timing.

One cell contains both methods for one prescribed partial. The original model,
inputs, seeds, warm-ups and synchronized wall-clock timer are unchanged. Only
CPU schedule construction and its single bulk device transfer are excluded.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from equal_updates_campaign import TARGETS
from first_group import make_model_and_points, parameters_sha256, parse_one_based_pattern, tensor_sha256
from stdepp_campaign import PARTIAL_SEEDS, gpu_environment, source_fingerprint
from torch_pinn import stdepp
from torch_pinn.jets import mlp_jet
from torch_pinn.model import MLP, sample_normal
from torch_pinn.monomials import monomial, schedule as wdd_schedule
from torch_pinn.records import run_record, save, sync

PROTOCOL = 'apolarity_stdepp_prepared_partials_v1'
METHODS = ('waring_batched', 'stdepp')


@dataclass(frozen=True)
class Prepared:
    method: str
    order: int
    directions: torch.Tensor
    weights: torch.Tensor


def prepare(method, alpha, dim=16, *, dtype=torch.float32, device='cpu'):
    """Construct on CPU, then transfer each complete tensor once to the device."""
    if method not in METHODS:
        raise ValueError(method)
    constructor = wdd_schedule if method == 'waring_batched' else stdepp.schedule
    directions, weights = constructor(tuple(alpha), dim, dtype=dtype, device='cpu')
    return Prepared(method, len(alpha), directions.to(device), weights.to(device))


def evaluate_prepared(model, points, prepared):
    """Identical contraction path for both methods; no schedule API is called."""
    return (prepared.weights @ mlp_jet(model, points, prepared.directions, prepared.order)[prepared.order]).real


def storage_identity(prepared):
    return tuple((id(t), t.untyped_storage().data_ptr(), t._version,
                  tuple(t.shape), str(t.dtype), str(t.device))
                 for t in (prepared.directions, prepared.weights))


def prepared_metadata(prepared):
    return dict(method=prepared.method, order=prepared.order,
        direction_count=len(prepared.directions),
        direction_dtype=str(prepared.directions.dtype), weight_dtype=str(prepared.weights.dtype),
        directions_sha256=tensor_sha256(prepared.directions),
        weights_sha256=tensor_sha256(prepared.weights),
        construction_device='cpu', execution_device=str(prepared.directions.device),
        direction_execution='batched', direction_chunking=False)


def validate_value(value, batch):
    if value.requires_grad or value.shape != (batch,) or value.dtype != torch.float32:
        raise AssertionError('Expected one detached float32 value per point')
    if not bool(torch.isfinite(value).all()):
        raise AssertionError('Nonfinite derivative value')


def benchmark_pair(model, points, schedules, *, seed_index=0, warmups=10, repeats=30):
    """Run alternating AB/BA rounds, keeping setup outside each timed closure."""
    if warmups < 0 or repeats < 1 or set(schedules) != set(METHODS):
        raise ValueError('Expected both prepared methods and positive repeat count')
    identities = {method: storage_identity(schedules[method]) for method in METHODS}
    calls = {method: (lambda p=schedules[method]: evaluate_prepared(model, points, p))
             for method in METHODS}
    samples = {method: [] for method in METHODS}
    orders, values = [], {}
    with torch.no_grad():
        for index in range(warmups):
            for method in METHODS[::1 if (index+seed_index) % 2 == 0 else -1]:
                values[method] = calls[method]()
        sync(points.device)
        for index in range(repeats):
            order = METHODS[::1 if (index+seed_index) % 2 == 0 else -1]
            orders.append(list(order))
            for method in order:
                sync(points.device)
                began = time.perf_counter()
                values[method] = calls[method]()
                sync(points.device)
                samples[method].append(1000*(time.perf_counter()-began))
    for method in METHODS:
        validate_value(values[method], len(points))
        if storage_identity(schedules[method]) != identities[method]:
            raise AssertionError('Prepared schedule storage changed during timing')
    torch.testing.assert_close(values['stdepp'], values['waring_batched'], rtol=1e-3, atol=1e-6)
    medians = {method: statistics.median(samples[method]) for method in METHODS}
    difference = values['stdepp']-values['waring_batched']
    agreement = dict(max_abs=float(difference.abs().max()),
        relative_l2=float(torch.linalg.vector_norm(difference)/
                          torch.linalg.vector_norm(values['waring_batched']).clamp_min(1e-30)))
    return dict(times_ms=samples, median_ms=medians, round_orders=orders,
                stdepp_over_wdd=medians['stdepp']/medians['waring_batched'],
                prepared_storage_stable=True, agreement_vs_wdd=agreement), values


def matrix(output):
    return [dict(index=index, target=target, out=str(Path(output)/'partials'/target),
                 methods=list(METHODS), seeds=list(PARTIAL_SEEDS), input_dim=16,
                 batch=100, width=128, depth=4, warmups=10, repeats=30)
            for index, target in enumerate(TARGETS)]


def plan(output):
    return dict(protocol=PROTOCOL, source_fingerprint=source_fingerprint(), cells=matrix(output),
        allocation=dict(partition='gpu', nodes=1, ntasks=1, gpus=1, cpus_per_task=10),
        execution=dict(gpu='Tesla V100-SXM2-32GB', python='3.11.7', torch='2.5.1+cu121',
                       numpy='1.26.4', cuda='12.1', threads=1, mode='eager',
                       tf32=False, autocast=False, dtype='float32'),
        schedule_preparation='CPU construction and bulk device transfer before warm-up/timing',
        timing='30 paired rounds, alternating AB/BA; per-call CUDA-synchronized wall time',
        parameter_backward=False, checkpoints=False,
        stdepp_variant=stdepp.VARIANT,
        reference='Existing Nested timings unchanged; neither PDE training nor Nested is rerun.')


def _cell_body(args):
    environment = gpu_environment()
    alpha = parse_one_based_pattern(args.target)
    schedules = {method: prepare(method, alpha, device=args.device) for method in METHODS}
    metadata = {method: prepared_metadata(value) for method, value in schedules.items()}
    for method, value in schedules.items():
        # Numeric schedules are evidence, not network or optimizer checkpoints.
        np.save(args.out/f'directions_{method}.npy', value.directions.cpu().numpy())
        np.save(args.out/f'weights_{method}.npy', value.weights.cpu().numpy())
    card = json.loads((args.out/'config.json').read_text())
    card.update(protocol=PROTOCOL, environment=environment, prepared_schedules=metadata,
        order=len(alpha), alpha=list(alpha), parameter_backward=False,
        outer_jit=False, torch_compile=False, input_distribution='normal(mean=0,std=0.35)',
        initialization='NumPy PCG64; original model seed + 2, input seed + 1',
        schedule_preparation_excluded=True, cuda_synchronized_timing=True,
        algorithm_metadata=stdepp.metadata(alpha))
    save(args.out/'config.json', card)
    rows = []
    for seed_index, seed in enumerate(args.seeds):
        model, points = make_model_and_points(seed, args.batch, args.width, args.depth, args.device)
        parameter_hash, points_hash = parameters_sha256(model), tensor_sha256(points)
        report, values = benchmark_pair(model, points, schedules, seed_index=seed_index,
                                       warmups=args.warmups, repeats=args.repeats)
        outputs = {}
        for method, value in values.items():
            array = value.cpu().numpy()
            np.save(args.out/f'values_{seed}_{method}.npy', array)
            outputs[method] = dict(output_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                output_dtype=str(array.dtype), output_shape=list(array.shape))
            if prepared_metadata(schedules[method]) != metadata[method]:
                raise AssertionError('Prepared schedule contents changed')
        rows.append(dict(target=args.target, seed=seed, parameters_sha256=parameter_hash,
            points_sha256=points_hash, prepared_schedules=metadata, outputs=outputs, **report))
        save(args.out/'results.json', rows)
        print(json.dumps(dict(target=args.target, seed=seed, median_ms=report['median_ms'])), flush=True)
    summary = {method: dict(mean_ms=statistics.mean(row['median_ms'][method] for row in rows),
        std_ms=statistics.stdev(row['median_ms'][method] for row in rows)) for method in METHODS}
    save(args.out/'summary.json', dict(target=args.target, seeds=len(rows), methods=summary,
        paired_stdepp_over_wdd_mean=statistics.mean(row['stdepp_over_wdd'] for row in rows),
        checkpoints_saved=False))


def correctness_checks(device='cpu'):
    """Independent small float64 JVP reference for all six partial patterns."""
    rows = []
    for target in TARGETS:
        alpha = parse_one_based_pattern(target)
        model = MLP(16, 2, 2, 20260908, dtype=torch.float64, device=device)
        points = sample_normal(20260909, 1, 16, dtype=torch.float64, device=device)
        with torch.no_grad():
            reference = monomial(model, points, alpha, 'nested_jvp')
            for method in METHODS:
                prepared = prepare(method, alpha, dtype=torch.float64, device=device)
                observed = evaluate_prepared(model, points, prepared)
                torch.testing.assert_close(observed, reference, rtol=1e-7, atol=1e-10)
                if observed.requires_grad:
                    raise AssertionError('Value-only graph retained')
                rows.append(dict(target=target, method=method,
                    max_abs=float((observed-reference).abs().max())))
    return rows


def _preflight_body(args):
    environment = gpu_environment()
    checks = correctness_checks(args.device)
    formal = []
    for target in TARGETS:
        alpha = parse_one_based_pattern(target)
        model, points = make_model_and_points(PARTIAL_SEEDS[0], 100, 128, 4, args.device)
        with torch.no_grad():
            original = monomial(model, points, alpha, 'waring_batched')
            observed = {}
            for method in METHODS:
                prepared = prepare(method, alpha, device=args.device)
                observed[method] = evaluate_prepared(model, points, prepared)
                validate_value(observed[method], 100)
                torch.testing.assert_close(observed[method], original, rtol=1e-3, atol=1e-6)
                formal.append(dict(target=target, **prepared_metadata(prepared),
                    max_abs=float((observed[method]-original).abs().max()),
                    relative_l2=float(torch.linalg.vector_norm(observed[method]-original)/
                                      torch.linalg.vector_norm(original).clamp_min(1e-30))))
        sync(args.device)
    save(args.out/'PASS.json', dict(status='PASS', protocol=PROTOCOL, environment=environment,
        source_fingerprint=source_fingerprint(), independent_float64_checks=checks,
        full_size_float32_checks=formal, checkpoints=False))


def check_gate(output):
    path = Path(output)/'preflight/PASS.json'
    if not path.is_file():
        raise RuntimeError('A successful prepared-partial V100 preflight is required')
    gate = json.loads(path.read_text())
    if gate.get('status') != 'PASS' or gate.get('protocol') != PROTOCOL:
        raise RuntimeError('Invalid preflight receipt')
    if gate.get('source_fingerprint') != source_fingerprint():
        raise RuntimeError('Source changed after preflight')
    if gate.get('environment', {}).get('gpu') != 'Tesla V100-SXM2-32GB':
        raise RuntimeError('Preflight GPU differs from the matched V100')
    return gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'preflight', 'cell'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args(argv)
    if args.action == 'plan':
        print(json.dumps(plan(args.out), indent=2))
        return
    common = dict(device='cuda', expected_device='V100', campaign_protocol=PROTOCOL)
    if args.action == 'preflight':
        run_record(argparse.Namespace(action=args.action, out=args.out/'preflight', **common), _preflight_body)
    else:
        if args.index is None or not 0 <= args.index < len(TARGETS):
            parser.error('--index must be between 0 and 5 for cell')
        check_gate(args.out)
        cell = matrix(args.out)[args.index]
        cell['out'] = Path(cell['out'])
        run_record(argparse.Namespace(action=args.action, **cell, **common), _cell_body)


if __name__ == '__main__':
    main()
