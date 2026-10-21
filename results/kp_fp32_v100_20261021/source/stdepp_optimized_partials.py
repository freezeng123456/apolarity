"""Profile, verify and compare prepared high-order partial implementations.

Profiling and formal timing are separate stages. The original prepared
baseline is profiled before the preflight and before any formal measurement.
Only six individual partials are covered; no PDE training is performed.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from first_group import make_model_and_points, parameters_sha256, parse_one_based_pattern, tensor_sha256
from stdepp_campaign import PARTIAL_SEEDS, gpu_environment, source_fingerprint
from stdepp_prepared_partials import METHODS, TARGETS, evaluate_prepared, prepare, prepared_metadata, validate_value
from torch_pinn.model import MLP, sample_normal
from torch_pinn.monomials import monomial
from torch_pinn.records import run_record, save, sync

PROTOCOL = 'apolarity_stdepp_optimized_partials_v1'
VARIANTS = ('baseline', 'precast', 'optimized')
VARIANT_ORDERS = tuple(VARIANTS[index:]+VARIANTS[:index] for index in range(len(VARIANTS)))
MAX_COMPRESSED_TRACE_BYTES = 2*1024*1024


def matrix(output):
    return [dict(index=index, target=target, out=str(Path(output)/'partials'/target),
        methods=list(METHODS), variants=list(VARIANTS), seeds=list(PARTIAL_SEEDS),
        input_dim=16, batch=100, width=128, depth=4, warmups=10, repeats=30)
        for index, target in enumerate(TARGETS)]


def plan(output):
    return dict(protocol=PROTOCOL, source_fingerprint=source_fingerprint(), cells=matrix(output),
        stages=['profile: original prepared baseline, CPU and CUDA activity',
                'preflight: independent float64 and full-size float32 agreement',
                'cell: balanced formal timings, then optimized CPU/CUDA profile'],
        allocation=dict(partition='gpu', nodes=1, ntasks=1, gpus=1, cpus_per_task=10),
        execution=dict(gpu='Tesla V100-SXM2-32GB', python='3.11.7', torch='2.5.1+cu121',
            numpy='1.26.4', cuda='12.1', threads=1, mode='eager', tf32=False,
            autocast=False, dtype='float32', parameter_backward=False),
        variants=dict(baseline='Original prepared schedule and original mlp_jet evaluator',
            precast='Common evaluator with dtype conversion before timing, without conjugate reduction',
            optimized='Common evaluator with dtype conversion and conjugate reduction before timing'),
        round_design='Three cyclic variant orders repeated ten times; AB/BA alternates each round. '
                     'Every variant position contains five AB and five BA rounds.',
        profile_timing='Instrumented; excluded from benchmark measurements and ratios',
        raw_trace_limit_bytes_each=MAX_COMPRESSED_TRACE_BYTES,
        pde_training=False, checkpoints=False)


def evaluation_key(variant, method):
    return f'{variant}/{method}'


def balanced_round_orders(repeats=30, seed_index=0):
    if repeats < 1 or repeats % (2*len(VARIANT_ORDERS)):
        raise ValueError('Repeat count must be a positive multiple of six')
    rounds = []
    for index in range(repeats):
        variants = VARIANT_ORDERS[(index+seed_index) % len(VARIANT_ORDERS)]
        methods = METHODS if (index+seed_index) % 2 == 0 else tuple(reversed(METHODS))
        rounds.append([evaluation_key(variant, method) for variant in variants for method in methods])
    return rounds


def make_evaluations(model, points, alpha):
    prepare_evaluation = importlib.import_module('torch_pinn.prepared_partials').prepare_evaluation
    calls, metadata, identities = {}, {}, {}
    for method in METHODS:
        old = prepare(method, alpha, points.shape[-1], dtype=points.dtype, device=points.device)
        key = evaluation_key('baseline', method)
        calls[key] = lambda old=old: evaluate_prepared(model, points, old)
        metadata[key] = dict(variant='baseline', **prepared_metadata(old))
        # The original schedule tensors are held by the closure for the entire run.
        identities[key] = lambda old=old: tuple((id(t), t.untyped_storage().data_ptr(), t._version)
                                               for t in (old.directions, old.weights))
        for variant in ('precast', 'optimized'):
            evaluator = prepare_evaluation(model, points, old.directions, old.weights, len(alpha),
                                          conjugate_reuse=variant == 'optimized')
            key = evaluation_key(variant, method)
            calls[key] = evaluator
            metadata[key] = dict(variant=variant, method=method, **evaluator.metadata())
            identities[key] = evaluator.storage_identity
    return calls, metadata, identities


def agreement(observed, reference):
    difference = observed-reference
    return dict(max_abs=float(difference.abs().max()),
        relative_l2=float(torch.linalg.vector_norm(difference)/
                          torch.linalg.vector_norm(reference).clamp_min(1e-30)))


def benchmark(model, points, alpha, *, seed_index=0, warmups=10, repeats=30):
    if warmups < 0:
        raise ValueError('Warm-ups must be nonnegative')
    orders = balanced_round_orders(repeats, seed_index)
    calls, metadata, identities = make_evaluations(model, points, alpha)
    identity_before = {key: identity() for key, identity in identities.items()}
    values, samples = {}, {key: [] for key in calls}
    with torch.no_grad():
        for index in range(warmups):
            for key in orders[index % len(orders)]:
                values[key] = calls[key]()
        sync(points.device)
        for order in orders:
            for key in order:
                sync(points.device)
                began = time.perf_counter()
                values[key] = calls[key]()
                sync(points.device)
                samples[key].append(1000*(time.perf_counter()-began))
    reference = values[evaluation_key('baseline', 'waring_batched')]
    checks = {}
    for key, value in values.items():
        validate_value(value, len(points))
        torch.testing.assert_close(value, reference, rtol=1e-3, atol=1e-6)
        checks[key] = agreement(value, reference)
        if identities[key]() != identity_before[key]:
            raise AssertionError(f'Prepared storage changed during timing: {key}')
    medians = {key: statistics.median(value) for key, value in samples.items()}
    return dict(times_ms=samples, median_ms=medians, round_orders=orders,
        evaluation_counts={key: dict(warmup=warmups, timed=repeats) for key in calls},
        metadata=metadata, agreements_vs_original_wdd=checks, prepared_storage_stable=True,
        timing='CUDA-synchronized per-call wall time; setup, profiler and correctness excluded',
        ratios={method: dict(baseline_over_precast=medians[evaluation_key('baseline', method)]/
                                medians[evaluation_key('precast', method)],
                            baseline_over_optimized=medians[evaluation_key('baseline', method)]/
                                medians[evaluation_key('optimized', method)]) for method in METHODS}), values


def _interval_stats(events):
    intervals = sorted((float(event['ts']), float(event['ts'])+float(event['dur'])) for event in events
                       if event.get('dur', 0) > 0)
    if not intervals:
        return dict(count=0, summed_duration_us=0., active_union_us=0., span_us=0., internal_gap_us=0.)
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    active = sum(end-start for start, end in merged)
    span = merged[-1][1]-merged[0][0]
    return dict(count=len(intervals), summed_duration_us=sum(end-start for start, end in intervals),
                active_union_us=active, span_us=span, internal_gap_us=span-active)


def summarize_trace(trace):
    events = [event for event in trace.get('traceEvents', []) if event.get('ph') == 'X']
    kernels = [event for event in events if event.get('cat') == 'kernel']
    copies = [event for event in events if str(event.get('cat', '')).startswith(('gpu_memcpy', 'gpu_memset'))]
    runtime = [event for event in events if event.get('cat') == 'cuda_runtime']
    enqueue = [event for event in runtime if 'launch' in str(event.get('name', '')).lower()]
    cpu_ops = [event for event in events if event.get('cat') == 'cpu_op']
    grouped = {}
    for event in kernels:
        name = event.get('name', '<unnamed>')
        row = grouped.setdefault(name, dict(name=name, calls=0, summed_duration_us=0.))
        row['calls'] += 1
        row['summed_duration_us'] += float(event.get('dur', 0))
    copy_groups = {}
    for event in copies:
        name = event.get('name', '<unnamed>')
        row = copy_groups.setdefault(name, dict(name=name, calls=0, summed_duration_us=0.))
        row['calls'] += 1
        row['summed_duration_us'] += float(event.get('dur', 0))
    return dict(compute_kernels=_interval_stats(kernels), gpu_copies=_interval_stats(copies),
        all_gpu_activity=_interval_stats(kernels+copies), cpu_cuda_runtime=_interval_stats(runtime),
        cpu_kernel_enqueue=_interval_stats(enqueue), cpu_ops=_interval_stats(cpu_ops),
        kernel_summaries=sorted(grouped.values(), key=lambda row: -row['summed_duration_us']),
        copy_summaries=sorted(copy_groups.values(), key=lambda row: -row['summed_duration_us']),
        interpretation='Duration sums can overlap and are not wall-time fractions. Union/gaps use '
                       'first-to-last traced event only; GPU gaps also include CPU launch delay and profiler overhead.')


def profile_evaluation(call, *, output, name, metadata, warmups=10):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for _ in range(warmups):
            call()
        sync('cuda')
        began = time.perf_counter()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False, profile_memory=False,
                                    with_stack=False) as profiler:
            with torch.profiler.record_function('derivative_evaluation'):
                value = call()
                sync('cuda')
        profiler_elapsed_ms = 1000*(time.perf_counter()-began)
    validate_value(value, 100)
    trace_path = output/f'{name}.trace.json'
    profiler.export_chrome_trace(str(trace_path))
    trace_bytes = trace_path.read_bytes()
    trace = json.loads(trace_bytes)
    timeline = summarize_trace(trace)
    if timeline['compute_kernels']['count'] < 1 or timeline['cpu_ops']['count'] < 1:
        raise AssertionError('Profiler did not capture expected CPU operations and CUDA kernels')
    operator_rows = []
    for event in profiler.key_averages():
        operator_rows.append(dict(name=event.key, calls=event.count,
            self_cpu_duration_us=float(event.self_cpu_time_total),
            cpu_duration_us=float(event.cpu_time_total),
            self_device_duration_us=float(getattr(event, 'self_device_time_total',
                                                getattr(event, 'self_cuda_time_total', 0.))),
            device_duration_us=float(getattr(event, 'device_time_total',
                                           getattr(event, 'cuda_time_total', 0.)))))
    compressed = gzip.compress(trace_bytes, compresslevel=6, mtime=0)
    trace_path.unlink()
    raw_trace = dict(uncompressed_bytes=len(trace_bytes), compressed_bytes=len(compressed),
                     sha256=hashlib.sha256(trace_bytes).hexdigest(), retained=False)
    if len(compressed) <= MAX_COMPRESSED_TRACE_BYTES:
        target = output/f'{name}.trace.json.gz'
        target.write_bytes(compressed)
        raw_trace.update(retained=True, file=target.name,
                         compressed_sha256=hashlib.sha256(compressed).hexdigest())
    report = dict(name=name, instrumented=True, benchmark_measurement=False,
        traced_evaluations=1, excluded_warmups=warmups,
        profiler_context_elapsed_ms=profiler_elapsed_ms, metadata=metadata,
        timeline=timeline, operators=sorted(operator_rows, key=lambda row: -row['self_cpu_duration_us']),
        raw_trace=raw_trace)
    save(output/f'{name}.json', report)
    return report


def receipt(output, stage):
    root = Path(output)/stage
    path = root/'PASS.json'
    if not path.is_file() or not (root/'status.txt').is_file() or (root/'status.txt').read_text().strip() != 'COMPLETE':
        raise RuntimeError(f'A completed successful {stage} stage is required')
    result = json.loads(path.read_text())
    if result.get('status') != 'PASS' or result.get('protocol') != PROTOCOL:
        raise RuntimeError(f'Invalid {stage} receipt')
    if result.get('source_fingerprint') != source_fingerprint():
        raise RuntimeError(f'Source changed after {stage}')
    if result.get('environment', {}).get('gpu') != 'Tesla V100-SXM2-32GB':
        raise RuntimeError(f'{stage} did not use the matched V100')
    return result


def _profile_body(args):
    environment = gpu_environment()
    rows = []
    for target in TARGETS:
        alpha = parse_one_based_pattern(target)
        model, points = make_model_and_points(PARTIAL_SEEDS[0], 100, 128, 4, args.device)
        identities = dict(parameters_sha256=parameters_sha256(model), points_sha256=tensor_sha256(points))
        for method in METHODS:
            prepared = prepare(method, alpha, device=args.device)
            call = lambda prepared=prepared: evaluate_prepared(model, points, prepared)
            report = profile_evaluation(call, output=args.out/'profiles', name=f'{target}_{method}',
                metadata=dict(target=target, variant='baseline', seed=PARTIAL_SEEDS[0],
                              **identities, **prepared_metadata(prepared)))
            rows.append(dict(target=target, method=method, compute_kernel_count=report['timeline']['compute_kernels']['count'],
                             report=f'profiles/{target}_{method}.json'))
    save(args.out/'PASS.json', dict(status='PASS', protocol=PROTOCOL, source_fingerprint=source_fingerprint(),
        environment=environment, profiled_cells=rows, expected_profiles=12,
        benchmark_measurements=False, baseline_only=True, checkpoints=False))


def correctness_checks(device='cpu', full_size=False):
    rows = []
    for target in TARGETS:
        alpha = parse_one_based_pattern(target)
        if full_size:
            model, points = make_model_and_points(PARTIAL_SEEDS[0], 100, 128, 4, device)
        else:
            model = MLP(16, 2, 2, 20260908, dtype=torch.float64, device=device)
            points = sample_normal(20260909, 1, 16, dtype=torch.float64, device=device)
        with torch.no_grad():
            reference = (evaluate_prepared(model, points, prepare('waring_batched', alpha, device=device))
                         if full_size else monomial(model, points, alpha, 'nested_jvp'))
            calls, metadata, _ = make_evaluations(model, points, alpha)
            for key, call in calls.items():
                observed = call()
                if full_size:
                    validate_value(observed, 100)
                elif observed.requires_grad or not bool(torch.isfinite(observed).all()):
                    raise AssertionError('Invalid independent-reference result')
                torch.testing.assert_close(observed, reference,
                    rtol=1e-3 if full_size else 1e-7, atol=1e-6 if full_size else 1e-10)
                rows.append(dict(target=target, evaluation=key, metadata=metadata[key],
                                 **agreement(observed, reference)))
        sync(device)
    return rows


def _preflight_body(args):
    profile = receipt(args.campaign_root, 'profile')
    environment = gpu_environment()
    checks = correctness_checks(args.device)
    formal_checks = correctness_checks(args.device, full_size=True)
    save(args.out/'PASS.json', dict(status='PASS', protocol=PROTOCOL, source_fingerprint=source_fingerprint(),
        environment=environment, prior_profile_count=len(profile['profiled_cells']),
        independent_float64_checks=checks, full_size_float32_checks=formal_checks, checkpoints=False))


def _cell_body(args):
    environment = gpu_environment()
    alpha = parse_one_based_pattern(args.target)
    card = json.loads((args.out/'config.json').read_text())
    card.update(protocol=PROTOCOL, environment=environment, parameter_backward=False,
        profile_measurements_excluded=True, variants=list(VARIANTS),
        dtype_preparation_excluded=True, schedule_preparation_excluded=True,
        input_distribution='normal(mean=0,std=0.35)',
        initialization='Original NumPy PCG64 model seed+2 and input seed+1')
    save(args.out/'config.json', card)
    rows = []
    for seed_index, seed in enumerate(args.seeds):
        model, points = make_model_and_points(seed, args.batch, args.width, args.depth, args.device)
        report, values = benchmark(model, points, alpha, seed_index=seed_index,
                                   warmups=args.warmups, repeats=args.repeats)
        outputs = {}
        for key, value in values.items():
            array = value.cpu().numpy()
            filename = f'values_{seed}_{key.replace("/", "_")}.npy'
            np.save(args.out/filename, array)
            outputs[key] = dict(file=filename, sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                                shape=list(array.shape), dtype=str(array.dtype))
        rows.append(dict(target=args.target, seed=seed, parameters_sha256=parameters_sha256(model),
            points_sha256=tensor_sha256(points), outputs=outputs, **report))
        save(args.out/'results.json', rows)
        print(json.dumps(dict(target=args.target, seed=seed, median_ms=report['median_ms'])), flush=True)
    # Post-optimization profiles run only after every seed's formal timing.
    model, points = make_model_and_points(PARTIAL_SEEDS[0], 100, 128, 4, args.device)
    calls, metadata, _ = make_evaluations(model, points, alpha)
    post_profiles = []
    for method in METHODS:
        key = evaluation_key('optimized', method)
        profile = profile_evaluation(calls[key], output=args.out/'profiles', name=f'optimized_{method}',
            metadata=dict(target=args.target, seed=PARTIAL_SEEDS[0],
                parameters_sha256=parameters_sha256(model), points_sha256=tensor_sha256(points), **metadata[key]))
        post_profiles.append(dict(method=method, report=f'profiles/optimized_{method}.json',
                                  compute_kernel_count=profile['timeline']['compute_kernels']['count']))
    methods = {key: dict(mean_ms=statistics.mean(row['median_ms'][key] for row in rows),
                         std_ms=statistics.stdev(row['median_ms'][key] for row in rows))
               for key in rows[0]['median_ms']}
    save(args.out/'summary.json', dict(target=args.target, seeds=len(rows), evaluations=methods,
        post_timing_profiles=post_profiles, profiler_measurements_excluded=True,
        warmup_calls_per_evaluation_per_seed=args.warmups,
        timed_calls_per_evaluation_per_seed=args.repeats, checkpoints_saved=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'profile', 'preflight', 'cell'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int)
    args = parser.parse_args(argv)
    if args.action == 'plan':
        print(json.dumps(plan(args.out), indent=2))
        return
    common = dict(device='cuda', expected_device='V100', campaign_protocol=PROTOCOL,
                  campaign_root=str(args.out))
    if args.action == 'profile':
        run_record(argparse.Namespace(action=args.action, out=args.out/'profile', **common), _profile_body)
    elif args.action == 'preflight':
        receipt(args.out, 'profile')
        run_record(argparse.Namespace(action=args.action, out=args.out/'preflight', **common), _preflight_body)
    else:
        if args.index is None or not 0 <= args.index < len(TARGETS):
            parser.error('--index must be between zero and five')
        receipt(args.out, 'profile')
        receipt(args.out, 'preflight')
        cell = matrix(args.out)[args.index]
        cell['out'] = Path(cell['out'])
        run_record(argparse.Namespace(action=args.action, **cell, **common), _cell_body)


if __name__ == '__main__':
    main()
