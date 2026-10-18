"""Compare WDD, Section 6.1 and the actual Section 6.2.4 prime single jet.

All three timed methods use the same structural-zero-aware sparse Taylor
backend. Prime selection/certification, dtype conversion, conjugate pairing
and recurrence planning are performed before any timed call. No PDE training
or model/optimizer checkpoint is part of this campaign.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
from itertools import permutations
import json
from math import isqrt, prod
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from equal_updates_campaign import TARGETS
from first_group import make_model_and_points, parameters_sha256, parse_one_based_pattern, tensor_sha256
from stdepp_campaign import PARTIAL_SEEDS, _tanh_derivative, gpu_environment, source_fingerprint
from stdepp_prepared_partials import prepare, evaluate_prepared, validate_value
from torch_pinn.model import MLP
from torch_pinn.records import run_record, save, sync

PROTOCOL = 'apolarity_stdepp_prime_single_jet_v1'
METHODS = ('wdd', 'stde61', 'prime624')
ORDERS = tuple(permutations(METHODS))


def matrix(output):
    return [dict(index=index, target=target, out=str(Path(output)/'partials'/target),
        methods=list(METHODS), seeds=list(PARTIAL_SEEDS), input_dim=16,
        batch=100, width=128, depth=4, warmups=10, repeats=30)
        for index, target in enumerate(TARGETS)]


def plan(output):
    return dict(protocol=PROTOCOL, source_fingerprint=source_fingerprint(), cells=matrix(output),
        stages=['preflight', 'pilot: one measured call per method, first seed only', 'cell'],
        allocation=dict(partition='gpu', nodes=1, ntasks=1, gpus=1, cpus_per_task=10),
        execution=dict(gpu='Tesla V100-SXM2-32GB', python='3.11.7', torch='2.5.1+cu121',
            numpy='1.26.4', cuda='12.1', threads=1, dtype='float32', mode='eager',
            tf32=False, autocast=False, parameter_backward=False),
        methods=dict(wdd='Minimum complex directions with conjugate reuse and preconversion',
            stde61='STDE++ Section 6.1 real multi-jet with preconversion',
            prime624='STDE++ Section 6.2.4 one prime-order curved jet, independently certified'),
        timing_scope='All methods use the same structural-zero-aware sparse Taylor backend; '
                     'per-call synchronized wall time excludes preparation, prime precomputation and checks.',
        balancing='All six method permutations repeated five times; every method position occurs ten times.',
        prime_normalization='Ordinary input coefficients equal coordinate basis vectors at prime orders. '
            'The raw input tangent at order p is p! times that basis vector; q_L is the target '
            'mixed partial directly, with no floating-point L! or p! conversion.',
        prime_precomputation='Tabulated/frozen independent prime search and uniqueness checks are separate from per-call timing.',
        pde_training=False, checkpoints=False)


def round_orders(repeats=30, seed_index=0):
    if repeats < 1 or repeats % len(ORDERS):
        raise ValueError('Repeat count must be a positive multiple of six')
    return [list(ORDERS[(index+seed_index) % len(ORDERS)]) for index in range(repeats)]


def prime_certificate(orders):
    """Independent unbounded-coin DP counts every nonnegative multiplicity vector."""
    orders = tuple(orders)
    if (not orders or len(set(orders)) != len(orders) or
        any(not isinstance(p, int) or p < 2 or any(p % d == 0 for d in range(2, isqrt(p)+1)) for p in orders)):
        raise ValueError('Input orders must be distinct primes')
    total = sum(orders)
    ways = [0]*(total+1)
    ways[0] = 1
    for prime in orders:
        for value in range(prime, total+1):
            ways[value] = min(2, ways[value]+ways[value-prime])
    if ways[total] != 1:
        raise ValueError('Prime orders do not give a unique nonnegative multiplicity vector')
    return dict(primes=list(orders), output_order=total,
        nonnegative_solution_count_capped_at_two=ways[total],
        unique_solution_is_all_ones=True,
        proof='The all-ones vector is one solution and independent unbounded-coin DP finds exactly one.')


def build_evaluators(model, points, alpha):
    sparse = importlib.import_module('torch_pinn.sparse_jets')
    prime_module = importlib.import_module('torch_pinn.prime_single_jet')
    prepared_module = importlib.import_module('torch_pinn.prepared_partials')
    calls, metadata, seconds = {}, {}, {}
    for method in ('wdd', 'stde61'):
        began = time.perf_counter()
        schedule = prepare('waring_batched' if method == 'wdd' else 'stdepp', alpha,
                           points.shape[-1], dtype=points.dtype, device=points.device)
        common = prepared_module.prepare_evaluation(model, points, schedule.directions,
            schedule.weights, len(alpha), conjugate_reuse=method == 'wdd')
        evaluator = sparse.prepare_curve_evaluation(model, points, (1,),
            common.directions.unsqueeze(0), len(alpha), common.weights)
        sync(points.device)
        seconds[method] = time.perf_counter()-began
        calls[method] = evaluator
        metadata[method] = dict(method=method, common_backend='torch_pinn.sparse_jets',
            schedule=common.metadata(), backend=evaluator.metadata(),
            output_order=len(alpha), input_orders=[1], preparation_excluded=True)
    began = time.perf_counter()
    prime = prime_module.prime_schedule(alpha, points.shape[-1], dtype=points.dtype, device=points.device)
    certificate = prime_certificate(prime.orders)
    if prime.output_order != sum(prime.orders) or len(prime.orders) != len(alpha):
        raise AssertionError('Inconsistent prime schedule')
    # Ordinary coefficients are exactly one-hot e_alpha, not 1/p! tangents.
    expected = points.new_zeros((len(alpha), points.shape[-1]))
    for row, axis in enumerate(alpha):
        expected[row, axis] = 1
    torch.testing.assert_close(prime.coefficients, expected, rtol=0, atol=0)
    sync(points.device)
    prime_precompute_seconds = time.perf_counter()-began
    began = time.perf_counter()
    evaluator = sparse.prepare_curve_evaluation(model, points, tuple(prime.orders),
        prime.coefficients.unsqueeze(1), prime.output_order)
    sync(points.device)
    seconds['prime624'] = time.perf_counter()-began
    calls['prime624'] = evaluator
    prime_metadata = prime.metadata() if callable(prime.metadata) else prime.metadata
    metadata['prime624'] = dict(method='prime624', common_backend='torch_pinn.sparse_jets',
        schedule=prime_metadata, backend=evaluator.metadata(), input_orders=list(prime.orders),
        output_order=prime.output_order, unique_prime_certificate=certificate,
        normalization='ordinary coefficient q_L equals the target mixed partial; extraction multiplier is one',
        preparation_excluded=True, prime_precompute_excluded=True)
    return calls, metadata, dict(prime_schedule_and_certificate_seconds=prime_precompute_seconds,
                                 method_preparation_seconds=seconds)


def agreement(value, reference):
    difference = value-reference
    return dict(max_abs=float(difference.abs().max()),
        relative_l2=float(torch.linalg.vector_norm(difference)/
                          torch.linalg.vector_norm(reference).clamp_min(1e-30)))


def validate_values(values, batch):
    for value in values.values():
        validate_value(value, batch)
        torch.testing.assert_close(value, values['wdd'], rtol=1e-3, atol=1e-6)
    return {method: agreement(value, values['wdd']) for method, value in values.items()}


def benchmark(model, points, alpha, *, seed_index=0, warmups=10, repeats=30):
    if warmups < 0:
        raise ValueError('Warm-up count must be nonnegative')
    orders = round_orders(repeats, seed_index)
    calls, metadata, setup = build_evaluators(model, points, alpha)
    identities = {method: call.storage_identity() for method, call in calls.items()}
    samples, values = {method: [] for method in METHODS}, {}
    with torch.no_grad():
        for index in range(warmups):
            for method in orders[index % len(orders)]:
                values[method] = calls[method]()
        sync(points.device)
        for order in orders:
            for method in order:
                sync(points.device)
                began = time.perf_counter()
                values[method] = calls[method]()
                sync(points.device)
                samples[method].append(1000*(time.perf_counter()-began))
    checks = validate_values(values, len(points))
    if any(call.storage_identity() != identities[method] for method, call in calls.items()):
        raise AssertionError('Prepared storage changed during timing')
    medians = {method: statistics.median(value) for method, value in samples.items()}
    return dict(times_ms=samples, median_ms=medians, round_orders=orders, metadata=metadata,
        setup=setup, agreements_vs_wdd=checks, prepared_storage_stable=True,
        evaluation_counts={method: dict(warmup=warmups, timed=repeats) for method in METHODS},
        prime_over_wdd=medians['prime624']/medians['wdd'],
        prime_over_stde61=medians['prime624']/medians['stde61']), values


def receipt(output, stage):
    root = Path(output)/stage
    if not (root/'PASS.json').is_file() or not (root/'status.txt').is_file() or (root/'status.txt').read_text().strip() != 'COMPLETE':
        raise RuntimeError(f'A completed {stage} stage is required')
    result = json.loads((root/'PASS.json').read_text())
    if result.get('status') != 'PASS' or result.get('protocol') != PROTOCOL:
        raise RuntimeError(f'Invalid {stage} receipt')
    if result.get('source_fingerprint') != source_fingerprint():
        raise RuntimeError(f'Source changed after {stage}')
    if result.get('environment', {}).get('gpu') != 'Tesla V100-SXM2-32GB':
        raise RuntimeError(f'{stage} did not use the matched V100')
    return result


def independent_checks(device='cpu'):
    rows = []
    for target in TARGETS:
        alpha = parse_one_based_pattern(target)
        model = MLP(16, 1, 2, 20260908, dtype=torch.float64, device=device)
        with torch.no_grad():
            model.layers[0].weight.copy_(torch.arange(16, dtype=torch.float64, device=device).reshape(1, -1)*.02+.5)
            model.layers[0].bias.fill_(.2)
            model.layers[1].weight.fill_(.7)
            model.layers[1].bias.zero_()
            points = torch.tensor([[-.03]*16, [0.]*16, [.03]*16], dtype=torch.float64, device=device)
            expected = .7*_tanh_derivative(torch.tanh(model.layers[0](points)[:, 0]), len(alpha))*prod(
                model.layers[0].weight[0, axis] for axis in alpha)
            calls, metadata, setup = build_evaluators(model, points, alpha)
            for method, call in calls.items():
                observed = call()
                if observed.requires_grad or not bool(torch.isfinite(observed).all()):
                    raise AssertionError('Invalid independent analytic result')
                torch.testing.assert_close(observed, expected, rtol=1e-7, atol=1e-9)
                rows.append(dict(target=target, method=method, metadata=metadata[method],
                                 setup=setup, **agreement(observed, expected)))
        sync(device)
    return rows


def _preflight_body(args):
    environment = gpu_environment()
    checks = independent_checks(args.device)
    formal = []
    for target in TARGETS:
        alpha = parse_one_based_pattern(target)
        model, points = make_model_and_points(PARTIAL_SEEDS[0], 100, 128, 4, args.device)
        with torch.no_grad():
            reference = evaluate_prepared(model, points, prepare('waring_batched', alpha, device=args.device))
            calls, metadata, setup = build_evaluators(model, points, alpha)
            for method, call in calls.items():
                began = time.perf_counter()
                value = call()
                sync(args.device)
                duration = time.perf_counter()-began
                validate_value(value, 100)
                torch.testing.assert_close(value, reference, rtol=1e-3, atol=1e-6)
                formal.append(dict(target=target, method=method, metadata=metadata[method],
                    correctness_call_seconds=duration, setup=setup,
                    parameters_sha256=parameters_sha256(model), points_sha256=tensor_sha256(points),
                    **agreement(value, reference)))
    save(args.out/'PASS.json', dict(status='PASS', protocol=PROTOCOL, source_fingerprint=source_fingerprint(),
        environment=environment, independent_analytic_float64=checks,
        full_shape_float32_checks=formal, checkpoints=False))


def _pilot_body(args):
    receipt(args.campaign_root, 'preflight')
    environment = gpu_environment()
    rows = []
    for target in TARGETS:
        alpha = parse_one_based_pattern(target)
        model, points = make_model_and_points(PARTIAL_SEEDS[0], 100, 128, 4, args.device)
        calls, metadata, setup = build_evaluators(model, points, alpha)
        values, seconds = {}, {}
        with torch.no_grad():
            for method in METHODS:
                sync(args.device)
                began = time.perf_counter()
                values[method] = calls[method]()
                sync(args.device)
                seconds[method] = time.perf_counter()-began
        checks = validate_values(values, 100)
        rows.append(dict(target=target, seed=PARTIAL_SEEDS[0], metadata=metadata, setup=setup,
            measured_calls_per_method=1, excluded_warmups=0, call_seconds=seconds,
            agreements_vs_wdd=checks, parameters_sha256=parameters_sha256(model),
            points_sha256=tensor_sha256(points),
            rough_projected_formal_seconds=3*(10+30)*sum(seconds.values())))
        save(args.out/'measurements.json', rows)
        print(json.dumps(dict(target=target, pilot_call_seconds=seconds,
            rough_projected_formal_seconds=rows[-1]['rough_projected_formal_seconds'])), flush=True)
    save(args.out/'PASS.json', dict(status='PASS', protocol=PROTOCOL, source_fingerprint=source_fingerprint(),
        environment=environment, measurements=rows, measured_calls=18,
        projection='Single-call pilot extrapolation only; not a formal benchmark or scheduler timeout.',
        rough_projected_all_formal_seconds=sum(row['rough_projected_formal_seconds'] for row in rows),
        checkpoints=False))


def _cell_body(args):
    environment = gpu_environment()
    card = json.loads((args.out/'config.json').read_text())
    card.update(protocol=PROTOCOL, environment=environment, common_backend='torch_pinn.sparse_jets',
        parameter_backward=False, schedule_preparation_excluded=True,
        prime_precomputation_excluded=True, dtype_preconversion_excluded=True,
        input_distribution='normal(mean=0,std=0.35)',
        initialization='Original NumPy PCG64 model seed+2 and input seed+1')
    save(args.out/'config.json', card)
    alpha = parse_one_based_pattern(args.target)
    rows = []
    for seed_index, seed in enumerate(args.seeds):
        model, points = make_model_and_points(seed, args.batch, args.width, args.depth, args.device)
        before = parameters_sha256(model), tensor_sha256(points)
        report, values = benchmark(model, points, alpha, seed_index=seed_index,
                                   warmups=args.warmups, repeats=args.repeats)
        if before != (parameters_sha256(model), tensor_sha256(points)):
            raise AssertionError('Model parameters or input points changed')
        outputs = {}
        for method, value in values.items():
            array = value.cpu().numpy()
            name = f'values_{seed}_{method}.npy'
            np.save(args.out/name, array)
            outputs[method] = dict(file=name, sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                                   dtype=str(array.dtype), shape=list(array.shape))
        rows.append(dict(target=args.target, seed=seed, parameters_sha256=before[0],
                         points_sha256=before[1], outputs=outputs, **report))
        save(args.out/'results.json', rows)
        print(json.dumps(dict(target=args.target, seed=seed, median_ms=report['median_ms'])), flush=True)
    save(args.out/'summary.json', dict(target=args.target, seeds=len(rows),
        methods={method: dict(mean_ms=statistics.mean(row['median_ms'][method] for row in rows),
                              sd_ms=statistics.stdev(row['median_ms'][method] for row in rows)) for method in METHODS},
        mean_paired_prime_over_wdd=statistics.mean(row['prime_over_wdd'] for row in rows),
        mean_paired_prime_over_stde61=statistics.mean(row['prime_over_stde61'] for row in rows),
        common_backend='torch_pinn.sparse_jets', checkpoints_saved=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'preflight', 'pilot', 'cell'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args(argv)
    if args.action == 'plan':
        print(json.dumps(plan(args.out), indent=2))
        return
    common = dict(device='cuda', expected_device='V100', campaign_protocol=PROTOCOL,
                  campaign_root=str(args.out))
    if args.action == 'preflight':
        run_record(argparse.Namespace(action=args.action, out=args.out/'preflight', **common), _preflight_body)
    elif args.action == 'pilot':
        receipt(args.out, 'preflight')
        run_record(argparse.Namespace(action=args.action, out=args.out/'pilot', **common), _pilot_body)
    else:
        if args.index is None or not 0 <= args.index < len(TARGETS):
            parser.error('--index must be between zero and five')
        receipt(args.out, 'preflight')
        receipt(args.out, 'pilot')
        cell = matrix(args.out)[args.index]
        cell['out'] = Path(cell['out'])
        run_record(argparse.Namespace(action=args.action, **cell, **common), _cell_body)


if __name__ == '__main__':
    main()
