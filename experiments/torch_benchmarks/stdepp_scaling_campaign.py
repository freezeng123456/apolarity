"""Frozen values-only WDD versus STDE++ Section 6.1 scaling comparison.

Both methods use the same sparse Taylor kernel. Schedule construction, conjugate
pairing, dtype conversion and all accuracy gates are outside timed calls. Failed
numerical cells remain in the result inventory; no PDE or checkpoint is involved.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from math import prod
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch

from first_group import make_model_and_points, parameters_sha256, tensor_sha256
from stdepp_campaign import PARTIAL_SEEDS, _tanh_derivative, gpu_environment, source_fingerprint
from stdepp_prepared_partials import prepare
from torch_pinn import prepared_partials, sparse_jets
from torch_pinn.model import MLP
from torch_pinn.records import configure, provenance, save, seal, sync

PROTOCOL = 'apolarity_stdepp_section61_scaling_v1'
METHODS = ('wdd', 'stde61')
EXPONENTS = ((6,), (2, 2, 2), (2, 2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5))
SHAPES = ((100, 128), (1000, 128), (1000, 256))
FLOAT64_TOLERANCE = 1e-7
FLOAT32_TOLERANCE = 1e-3
REFERENCE_POINTS = 8


def alpha_from_exponents(exponents):
    if not exponents or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in exponents):
        raise ValueError('Exponents must be positive integers')
    return tuple(axis for axis, count in enumerate(exponents) for _ in range(count))


def target_name(exponents):
    return ''.join(str(axis+1) for axis in alpha_from_exponents(exponents))


def matrix(output):
    cells = []
    for exponents in EXPONENTS:
        for batch, width in SHAPES:
            target = target_name(exponents)
            name = f'{target}_B{batch}_W{width}'
            cells.append(dict(index=len(cells), name=name, target=target,
                exponents=list(exponents), order=sum(exponents),
                out=str(Path(output)/'partials'/name), methods=list(METHODS),
                seeds=list(PARTIAL_SEEDS), input_dim=16, batch=batch, width=width,
                depth=4, warmups=10, repeats=30))
    return cells


def plan(output):
    return dict(protocol=PROTOCOL, source_fingerprint=source_fingerprint(), cells=matrix(output),
        allocation=dict(partition='gpu', nodes=1, ntasks=1, gpus=1, cpus_per_task=10),
        execution=dict(gpu='Tesla V100-SXM2-32GB', cpu='Intel Xeon Platinum 8255C', python='3.11.7', torch='2.5.1+cu121',
            numpy='1.26.4', cuda='12.1', threads=1, dtype='float32', mode='eager',
            tf32=False, autocast=False, parameter_backward=False),
        methods=dict(wdd='Minimum complex directions with verified conjugate reuse and preconversion',
                     stde61='STDE++ Section 6.1 deterministic real multi-jet, grouped and preconverted'),
        common_backend='torch_pinn.sparse_jets',
        timing_scope='Both methods use the same sparse Taylor backend, input order 1 and output order k. '
            'Synchronized wall time excludes schedules, conjugate pairing, preconversion and all accuracy checks.',
        balancing='AB/BA alternate; each method is first in fifteen of thirty rounds per seed.',
        correctness=dict(reference='First eight FP32 input points and the same FP32 model converted to float64',
            reference_points=REFERENCE_POINTS, float64_pair_tolerance=FLOAT64_TOLERANCE,
            float32_tolerance=FLOAT32_TOLERANCE,
            metrics=['relative_l2', 'max_abs_over_reference_max_abs'], absolute_tolerance_floor=False,
            invalid_policy='Save INVALID_NUMERICS with outputs and errors; do not time failed seeds; continue cells.'),
        pde_training=False, checkpoints=False)


def round_orders(repeats=30, seed_index=0):
    if repeats < 1 or repeats % 2:
        raise ValueError('Repeat count must be a positive multiple of two')
    return [list(METHODS[::1 if (index+seed_index) % 2 == 0 else -1]) for index in range(repeats)]


def build_evaluators(model, points, alpha):
    calls, metadata, seconds = {}, {}, {}
    for method in METHODS:
        began = time.perf_counter()
        schedule = prepare('waring_batched' if method == 'wdd' else 'stdepp', alpha,
            points.shape[-1], dtype=points.dtype, device=points.device)
        common = prepared_partials.prepare_evaluation(model, points, schedule.directions,
            schedule.weights, len(alpha), conjugate_reuse=method == 'wdd')
        evaluator = sparse_jets.prepare_curve_evaluation(model, points, (1,),
            common.directions.unsqueeze(0), len(alpha), common.weights)
        sync(points.device)
        seconds[method] = time.perf_counter()-began
        calls[method] = evaluator
        metadata[method] = dict(method=method, common_backend='torch_pinn.sparse_jets',
            schedule=common.metadata(), backend=evaluator.metadata(), input_orders=[1],
            output_order=len(alpha), preparation_excluded=True)
    return calls, metadata, seconds


def value_validity(value, batch, dtype):
    return dict(shape=list(value.shape), dtype=str(value.dtype),
        shape_ok=value.shape == (batch,), dtype_ok=value.dtype == dtype,
        detached=not value.requires_grad, finite=bool(torch.isfinite(value).all()))


def validity_passes(report):
    return all(report[name] for name in ('shape_ok', 'dtype_ok', 'detached', 'finite'))


def agreement(value, reference, tolerance):
    """Scale errors by reference magnitude, never by an absolute tolerance floor."""
    if value.shape != reference.shape or not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(reference).all()):
        return dict(passed=False, reason='shape_mismatch_or_nonfinite', tolerance=tolerance)
    difference = value.detach().double()-reference.detach().double()
    reference = reference.detach().double()
    max_abs = float(difference.abs().max())
    reference_max = float(reference.abs().max())
    error_l2 = float(torch.linalg.vector_norm(difference))
    reference_l2 = float(torch.linalg.vector_norm(reference))
    # None represents an undefined nonzero/zero error ratio and fails the gate;
    # unlike infinity, it remains valid JSON in retained numerical-failure files.
    ratio = lambda error, scale: error/scale if scale > 0 else (0. if error == 0 else None)
    relative_l2 = ratio(error_l2, reference_l2)
    relative_max = ratio(max_abs, reference_max)
    return dict(max_abs=max_abs, reference_max_abs=reference_max,
        reference_l2=reference_l2, relative_l2=relative_l2,
        max_abs_over_reference_max_abs=relative_max, tolerance=tolerance,
        passed=all(value is not None and value <= tolerance for value in (relative_l2, relative_max)))


def reference_inputs(model, points):
    """Promote the actual quantized FP32 inputs; never initialize a new model."""
    return deepcopy(model).to(dtype=torch.float64), points[:REFERENCE_POINTS].detach().to(dtype=torch.float64).clone()


def numerical_gate(model, points, alpha, calls):
    with torch.no_grad():
        values = {method: call() for method, call in calls.items()}
    validity = {method: value_validity(value, len(points), torch.float32) for method, value in values.items()}
    report = dict(full_batch_validity=validity, passed=False)
    if not all(validity_passes(row) for row in validity.values()):
        report['reason'] = 'invalid_full_batch_output'
        return report, values, {}
    report['full_batch_pair'] = agreement(values['stde61'], values['wdd'], FLOAT32_TOLERANCE)
    reference_model, reference_points = reference_inputs(model, points)
    reference_calls, reference_metadata, reference_setup = build_evaluators(reference_model, reference_points, alpha)
    with torch.no_grad():
        references = {method: call() for method, call in reference_calls.items()}
    reference_validity = {method: value_validity(value, len(reference_points), torch.float64)
                          for method, value in references.items()}
    report.update(reference_metadata=reference_metadata, reference_setup_seconds=reference_setup,
        reference_validity=reference_validity, reference_points=len(reference_points),
        reference_source='same FP32 model and first input points promoted to FP64')
    if not all(validity_passes(row) for row in reference_validity.values()):
        report['reason'] = 'invalid_float64_reference_output'
        return report, values, references
    report['float64_pair'] = agreement(references['stde61'], references['wdd'], FLOAT64_TOLERANCE)
    report['float32_vs_reference'] = {
        method: agreement(value[:len(reference_points)], references['wdd'], FLOAT32_TOLERANCE)
        for method, value in values.items()}
    report['passed'] = (report['full_batch_pair']['passed'] and report['float64_pair']['passed']
        and all(row['passed'] for row in report['float32_vs_reference'].values()))
    if not report['passed']:
        report['reason'] = 'normalized_accuracy_tolerance_exceeded'
    return report, values, references


def time_evaluators(calls, device, *, seed_index=0, warmups=10, repeats=30):
    if warmups < 0:
        raise ValueError('Warm-up count must be nonnegative')
    orders = round_orders(repeats, seed_index)
    samples, values = {method: [] for method in METHODS}, {}
    with torch.no_grad():
        for index in range(warmups):
            for method in orders[index % len(orders)]:
                values[method] = calls[method]()
        sync(device)
        for order in orders:
            for method in order:
                sync(device)
                began = time.perf_counter()
                values[method] = calls[method]()
                sync(device)
                samples[method].append(1000*(time.perf_counter()-began))
    medians = {method: statistics.median(samples[method]) for method in METHODS}
    return dict(times_ms=samples, median_ms=medians, round_orders=orders,
        evaluation_counts={method: dict(warmup=warmups, timed=repeats) for method in METHODS},
        stde61_over_wdd=medians['stde61']/medians['wdd']), values


def benchmark(model, points, alpha, *, seed_index=0, warmups=10, repeats=30):
    round_orders(repeats, seed_index)
    calls, metadata, setup = build_evaluators(model, points, alpha)
    identities = {method: call.storage_identity() for method, call in calls.items()}
    checks, values, references = numerical_gate(model, points, alpha, calls)
    report = dict(status='INVALID_NUMERICS', metadata=metadata, preparation_seconds=setup,
        correctness=checks, timed=False, evaluation_counts={method: dict(warmup=0, timed=0) for method in METHODS})
    if checks['passed']:
        timings, timed_values = time_evaluators(calls, points.device, seed_index=seed_index,
            warmups=warmups, repeats=repeats)
        post_checks = {method: agreement(timed_values[method], values[method], 0.) for method in METHODS}
        report.update(timings, timed=True, post_timing_vs_gate_values=post_checks,
                      status='COMPLETE' if all(row['passed'] for row in post_checks.values()) else 'INVALID_NUMERICS')
        values = timed_values
    if any(call.storage_identity() != identities[method] for method, call in calls.items()):
        raise AssertionError('Prepared storage changed during evaluation')
    report['prepared_storage_stable'] = True
    return report, values, references


def environment(device):
    if torch.device(device).type == 'cuda':
        observed = gpu_environment()
        if '8255C' not in observed.get('lscpu', ''):
            raise RuntimeError('Expected the matched Intel Xeon Platinum 8255C CPU')
        return observed
    return dict(gpu='CPU_LOCAL_VALIDATION_ONLY', formal=False, python=platform.python_version(),
        torch=torch.__version__, numpy=np.__version__, threads=torch.get_num_threads())


def independent_checks(device='cpu', exponents_to_check=None):
    rows = []
    for exponents in EXPONENTS if exponents_to_check is None else exponents_to_check:
        alpha = alpha_from_exponents(exponents)
        model = MLP(16, 1, 2, 20260908, dtype=torch.float64, device=device)
        with torch.no_grad():
            model.layers[0].weight.copy_(torch.arange(16, dtype=torch.float64, device=device).reshape(1, -1)*.02+.5)
            model.layers[0].bias.fill_(.2)
            model.layers[1].weight.fill_(.7)
            model.layers[1].bias.zero_()
            points = torch.tensor([[-.03]*16, [0.]*16, [.03]*16], dtype=torch.float64, device=device)
            expected = .7*_tanh_derivative(torch.tanh(model.layers[0](points)[:, 0]), len(alpha))*prod(
                model.layers[0].weight[0, axis] for axis in alpha)
            calls, metadata, _ = build_evaluators(model, points, alpha)
            for method, call in calls.items():
                value = call()
                validity = value_validity(value, len(points), torch.float64)
                error = agreement(value, expected, FLOAT64_TOLERANCE)
                row = dict(target=target_name(exponents), exponents=list(exponents), method=method,
                    validity=validity, error=error, metadata=metadata[method],
                    passed=validity_passes(validity) and error['passed'])
                rows.append(row)
        sync(device)
    return rows


def receipt(output, device):
    root = Path(output)/'preflight'
    if not (root/'PASS.json').is_file() or not (root/'status.txt').is_file() or (root/'status.txt').read_text().strip() != 'COMPLETE':
        raise RuntimeError('A completed preflight stage is required')
    result = json.loads((root/'PASS.json').read_text())
    if result.get('status') != 'PASS' or result.get('protocol') != PROTOCOL:
        raise RuntimeError('Invalid preflight receipt')
    if result.get('source_fingerprint') != source_fingerprint():
        raise RuntimeError('Source changed after preflight')
    expected = 'Tesla V100-SXM2-32GB' if torch.device(device).type == 'cuda' else 'CPU_LOCAL_VALIDATION_ONLY'
    if result.get('environment', {}).get('gpu') != expected:
        raise RuntimeError('Preflight did not use the matched execution environment')
    return result


def _preflight_body(args):
    env = environment(args.device)
    rows = independent_checks(args.device)
    largest = largest_shape_probe(args.device)
    report = dict(status='PASS' if all(row['passed'] for row in rows) else 'INVALID_NUMERICS',
        protocol=PROTOCOL, source_fingerprint=source_fingerprint(), environment=env,
        independent_analytic_float64=rows, largest_shape_probe=largest, checkpoints=False)
    save(args.out/'PASS.json', report)
    return 'COMPLETE' if report['status'] == 'PASS' else 'INVALID_NUMERICS'


def largest_shape_probe(device, exponents=None, shape=None):
    """Check the largest forward fits; cold calls are not performance estimates."""
    exponents = EXPONENTS[-1] if exponents is None else tuple(exponents)
    batch, width = SHAPES[-1] if shape is None else shape
    model, points = make_model_and_points(PARTIAL_SEEDS[0], batch, width, 4, device)
    calls, metadata, setup = build_evaluators(model, points, alpha_from_exponents(exponents))
    rows, values = {}, {}
    with torch.no_grad():
        for method, call in calls.items():
            sync(device)
            cuda = torch.device(device).type == 'cuda'
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
                before = torch.cuda.memory_allocated(device)
            began = time.perf_counter()
            values[method] = call()
            sync(device)
            rows[method] = dict(cold_call_seconds=time.perf_counter()-began,
                validity=value_validity(values[method], batch, torch.float32),
                allocated_before_bytes=before if cuda else None,
                allocated_after_bytes=torch.cuda.memory_allocated(device) if cuda else None,
                allocated_peak_bytes=torch.cuda.max_memory_allocated(device) if cuda else None)
    pair = agreement(values['stde61'], values['wdd'], FLOAT32_TOLERANCE)
    return dict(target=target_name(exponents), batch=batch, width=width, depth=4,
        seed=PARTIAL_SEEDS[0], metadata=metadata, preparation_seconds=setup, methods=rows,
        pair=pair, numerical_pass=pair['passed'] and all(validity_passes(row['validity']) for row in rows.values()),
        purpose='One untimed-for-comparison cold call per method checks largest-shape memory feasibility; '
            'numerical exclusions are recorded by each formal cell; OOM/infrastructure errors propagate.')


def save_outputs(output, seed, values, references):
    result = {}
    for kind, collection in [('values', values), ('reference64', references)]:
        result[kind] = {}
        for method, value in collection.items():
            array = value.detach().cpu().numpy()
            name = f'{kind}_{seed}_{method}.npy'
            np.save(output/name, array)
            result[kind][method] = dict(file=name, sha256=hashlib.sha256(array.tobytes()).hexdigest(),
                dtype=str(array.dtype), shape=list(array.shape))
    return result


def _cell_body(args):
    env = environment(args.device)
    card = json.loads((args.out/'config.json').read_text())
    card.update(protocol=getattr(args, 'campaign_protocol', PROTOCOL), environment=env, source_fingerprint=source_fingerprint(),
        common_backend='torch_pinn.sparse_jets', parameter_backward=False,
        schedule_preparation_excluded=True, dtype_preconversion_excluded=True,
        input_distribution='normal(mean=0,std=0.35)',
        initialization='Original NumPy PCG64 FP32 model seed+2 and input seed+1')
    save(args.out/'config.json', card)
    rows = []
    for seed_index, seed in enumerate(args.seeds):
        model, points = make_model_and_points(seed, args.batch, args.width, args.depth, args.device)
        before = parameters_sha256(model), tensor_sha256(points)
        report, values, references = benchmark(model, points, alpha_from_exponents(args.exponents),
            seed_index=seed_index, warmups=args.warmups, repeats=args.repeats)
        if before != (parameters_sha256(model), tensor_sha256(points)):
            raise AssertionError('Model parameters or input points changed')
        row = dict(target=args.target, exponents=args.exponents, seed=seed,
            parameters_sha256=before[0], points_sha256=before[1],
            outputs=save_outputs(args.out, seed, values, references), **report)
        rows.append(row)
        save(args.out/'results.json', rows)
        print(json.dumps(dict(target=args.target, batch=args.batch, width=args.width, seed=seed,
            status=report['status'], median_ms=report.get('median_ms'))), flush=True)
    valid = [row for row in rows if row['status'] == 'COMPLETE']
    status = 'COMPLETE' if len(valid) == len(rows) else 'INVALID_NUMERICS'
    summary = dict(status=status, target=args.target, exponents=args.exponents, batch=args.batch,
        width=args.width, seeds=len(rows), valid_seeds=len(valid),
        seed_statuses={str(row['seed']): row['status'] for row in rows},
        common_backend='torch_pinn.sparse_jets', checkpoints_saved=False,
        aggregate_policy='Aggregate timing only when every prescribed seed passes numerical validation')
    if status == 'COMPLETE':
        summary.update(methods={method: dict(mean_ms=statistics.mean(row['median_ms'][method] for row in rows),
            sd_ms=statistics.stdev(row['median_ms'][method] for row in rows)) for method in METHODS},
            mean_paired_stde61_over_wdd=statistics.mean(row['stde61_over_wdd'] for row in rows))
    save(args.out/'summary.json', summary)
    return status


def run_stage(args, body):
    """Retain numerical exclusions without disguising infrastructure exceptions."""
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        configure(args.device, 'V100')
        save(args.out/'config.json', provenance(args))
        (args.out/'status.txt').write_text('RUNNING\n')
        result = body(args)
        if result not in ('COMPLETE', 'INVALID_NUMERICS'):
            raise AssertionError('Unexpected campaign terminal status')
        save(args.out/'completion.json', dict(status=result, elapsed_s=time.time()-started))
        (args.out/'status.txt').write_text(result+'\n')
    except Exception as exc:
        save(args.out/'failure.json', dict(error=repr(exc)))
        (args.out/'status.txt').write_text('FAILED\n')
        raise
    finally:
        seal(args.out)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'preflight', 'cell'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args(argv)
    if args.action == 'plan':
        print(json.dumps(plan(args.out), indent=2))
        return
    common = dict(device=args.device, expected_device='V100', campaign_protocol=PROTOCOL,
        campaign_root=str(args.out))
    if args.action == 'preflight':
        run_stage(argparse.Namespace(action=args.action, out=args.out/'preflight', **common), _preflight_body)
    else:
        cells = matrix(args.out)
        if args.index is None or not 0 <= args.index < len(cells):
            parser.error(f'--index must be between zero and {len(cells)-1}')
        receipt(args.out, args.device)
        cell = cells[args.index]
        cell['out'] = Path(cell['out'])
        run_stage(argparse.Namespace(action=args.action, **cell, **common), _cell_body)


if __name__ == '__main__':
    main()
