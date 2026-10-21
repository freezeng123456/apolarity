"""Five-seed CH repeat using the unchanged equal-update PDE implementations.

Each seed's three methods run sequentially on the same GPU, in separate
processes. Anomalies trigger one complete paired confirmation repeat; all
attempts remain available and no result is selected by its final error.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

import stdepp_campaign as campaign
import train_equal_updates as trainer
from equal_updates_campaign import SEEDS
from torch_pinn import problem_ch2d
from torch_pinn.model import MLP, sample_normal
from torch_pinn.records import configure, run_record, save, source_identity

PROTOCOL = 'apolarity_ch2d_paired_repeat_v1'
METHODS = ('nested_jvp', 'shared_jet_linear', 'stdepp')
HASH_FIELDS = ('initial_hash', 'interior_samples_sha256',
               'constraint_samples_sha256', 'spacetime_test_sha256',
               'terminal_test_sha256')
ORDERS = ((1, 2, 0), (2, 0, 1), (0, 1, 2), (0, 2, 1), (1, 0, 2))


def plan(output):
    cells, lanes = [], []
    for lane, (seed, order) in enumerate(zip(SEEDS, ORDERS)):
        indices = []
        for position, method_index in enumerate(order):
            method = METHODS[method_index]
            index = len(cells)
            indices.append(index)
            name = f'ch2d_seed{seed}_{method}'
            cells.append(dict(index=index, lane=lane, position=position,
                kind='pde', name=name, case='ch2d', method=method, seed=seed,
                updates=1500, eval_every=10, batch=400, width=128, depth=5,
                constraint_side=16, eval_points=10000, eval_seed_base=20261400,
                lr=1e-4, out=str(Path(output)/name)))
        lanes.append(dict(lane=lane, seed=seed, indices=indices,
                          methods=[METHODS[i] for i in order]))
    return dict(protocol=PROTOCOL, source_fingerprint=campaign.source_fingerprint(),
        cells=cells, lanes=lanes, seeds=list(SEEDS), checkpoints=False,
        case='ch2d', implementations_changed=False,
        allocation=dict(partition='gpu', gpus_per_lane=1, cpus_per_task=10,
                        lanes=5, same_gpu_within_seed=True),
        execution=dict(gpu='Tesla V100-SXM2-32GB', torch='2.5.1+cu121',
                       numpy='1.26.4', python='3.11.7', threads=1,
                       dtype='float32', tf32=False, autocast=False),
        anomaly_policy=dict(max_confirmation_repeats=1,
            same_seed_curve_rtol=1e-3, same_seed_curve_atol=1e-6,
            cross_seed_time_relative_deviation=0.10,
            paired_time_ratio_relative_deviation=0.10,
            interval_p95_to_median=1.25, interval_max_to_median=1.75,
            timing_diagnostic_ignored_initial_intervals=5,
            normal_common_error_oscillations_are_not_excluded=True,
            selection_by_final_error=False, preserve_all_attempts=True))


def check_objective(device, *, full_size):
    dtype = torch.float32 if full_size else torch.float64
    width, depth, batch, side = (128, 5, 400, 16) if full_size else (3, 2, 2, 2)
    points = sample_normal(92, batch, 3, dtype=dtype, device=device)
    data = problem_ch2d.constraints(side, dtype=dtype, device=device)
    rows, reference = [], None
    for method in METHODS:
        model = MLP(3, width, depth, 91, dtype=dtype, device=device)
        with campaign.stdepp_training_dispatch():
            loss, parts = trainer.full_loss(problem_ch2d, model, points, data, method)
        loss.backward()
        gradients = tuple(p.grad.detach().clone() for p in model.parameters())
        if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('nonfinite preflight objective or gradient')
        rtol, atol = (1e-3, 1e-5) if full_size else (1e-8, 1e-10)
        error = 0.0
        if reference is None:
            reference = (loss.detach().clone(), gradients,
                         {k: v.detach().clone() for k, v in parts.items()})
        else:
            torch.testing.assert_close(loss, reference[0], rtol=rtol, atol=atol)
            for g, expected in zip(gradients, reference[1]):
                torch.testing.assert_close(g, expected, rtol=rtol, atol=atol)
                error = max(error, float((g-expected).abs().max()))
            for key, value in parts.items():
                torch.testing.assert_close(value, reference[2][key], rtol=rtol, atol=atol)
        rows.append(dict(method=method, loss=float(loss.detach()),
                         max_abs_gradient_error=error, full_size=full_size))
    return rows


def preflight(output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    configure(device, 'V100')
    source_identity()
    rows = check_objective(device, full_size=False)
    if device == 'cuda':
        if torch.__version__ != '2.5.1+cu121' or np.__version__ != '1.26.4':
            raise RuntimeError('pinned numerical runtime mismatch')
        rows += check_objective(device, full_size=True)
    save(output/'PASS.json', dict(status='PASS', protocol=PROTOCOL,
        source_fingerprint=campaign.source_fingerprint(), checks=rows,
        device=torch.cuda.get_device_name() if device == 'cuda' else 'CPU'))
    print('PREFLIGHT_PASS', flush=True)


def run_cell(output, index, gate_root, *, device='cuda'):
    gate = json.loads((Path(gate_root)/'PASS.json').read_text())
    if gate['status'] != 'PASS' or gate['source_fingerprint'] != campaign.source_fingerprint():
        raise RuntimeError('preflight does not match the frozen source')
    cell = plan(output)['cells'][index]
    args = campaign.pde_args(cell, device)
    args.campaign_protocol = PROTOCOL
    args.paired_lane = cell['lane']
    args.method_position = cell['position']
    with campaign.stdepp_training_dispatch():
        run_record(args, trainer.train)


def inspect_group(output, lane, references):
    specification = plan(output)
    seed = specification['lanes'][lane]['seed']
    results, issues, curves = {}, [], {}
    policy = specification['anomaly_policy']
    for index in specification['lanes'][lane]['indices']:
        cell = specification['cells'][index]
        folder, method = Path(cell['out']), cell['method']
        if not (folder/'status.txt').is_file() or (folder/'status.txt').read_text().strip() != 'COMPLETE':
            issues.append(dict(reason='run_incomplete_or_failed', method=method))
            continue
        if not all((folder/name).is_file() for name in ('summary.json', 'curve.csv')):
            issues.append(dict(reason='missing_numeric_result', method=method))
            continue
        summary = json.loads((folder/'summary.json').read_text())
        rows = list(csv.DictReader((folder/'curve.csv').open()))
        if summary['steps'] != 1500 or [int(r['step']) for r in rows] != list(range(0, 1501, 10)):
            issues.append(dict(reason='wrong_update_count_or_curve_grid', method=method))
            continue
        matches = {k: summary[k] == references[f'ch2d_{seed}'][k] for k in HASH_FIELDS}
        if not all(matches.values()):
            issues.append(dict(reason='paired_hash_mismatch', method=method, matches=matches))
        errors = [float(r['spacetime_relative_l2']) for r in rows]
        times = [float(r['time_s']) for r in rows]
        if not all(math.isfinite(x) for x in errors+times):
            issues.append(dict(reason='nonfinite_curve', method=method))
            continue
        intervals = [b-a for a, b in zip(times, times[1:])][5:]
        med = statistics.median(intervals)
        p95 = sorted(intervals)[math.ceil(.95*len(intervals))-1]
        maximum = max(intervals)
        if any(x <= 0 for x in intervals) or p95/med > policy['interval_p95_to_median'] or maximum/med > policy['interval_max_to_median']:
            issues.append(dict(reason='within_run_timing_jitter', method=method,
                               interval_median_s=med, interval_p95_s=p95, interval_max_s=maximum))
        curves[method] = errors
        results[method] = dict(training_seconds=summary['wall_seconds'],
            re=summary['errors']['spacetime']['relative_l2'],
            interval_median_s=med, interval_p95_s=p95, interval_max_s=maximum)
    if len(curves) == len(METHODS):
        reference = curves['nested_jvp']
        for method in METHODS[1:]:
            if any(abs(a-b) > policy['same_seed_curve_atol']+policy['same_seed_curve_rtol']*abs(b)
                   for a, b in zip(curves[method], reference)):
                issues.append(dict(reason='same_seed_error_curve_mismatch', method=method))
    return dict(lane=lane, seed=seed, results=results, issues=issues)


def audit(output, references, *, cross_seed=True):
    groups = [inspect_group(output, lane, references) for lane in range(len(SEEDS))]
    return audit_groups(groups, cross_seed=cross_seed)


def audit_groups(groups, *, cross_seed=True):
    policy = plan('.')['anomaly_policy']
    if cross_seed:
        for method in METHODS:
            valid = [g for g in groups if method in g['results']]
            if len(valid) < 3:
                continue
            med = statistics.median(g['results'][method]['training_seconds'] for g in valid)
            for group in valid:
                ratio = group['results'][method]['training_seconds']/med
                if abs(ratio-1) > policy['cross_seed_time_relative_deviation']:
                    group['issues'].append(dict(reason='cross_seed_timing_outlier',
                        method=method, seconds_to_seed_median=ratio))
        valid = [g for g in groups if all(m in g['results'] for m in METHODS[1:])]
        if len(valid) >= 3:
            ratios = {g['lane']: g['results']['stdepp']['training_seconds']/
                      g['results']['shared_jet_linear']['training_seconds'] for g in valid}
            med = statistics.median(ratios.values())
            for group in valid:
                if abs(ratios[group['lane']]/med-1) > policy['paired_time_ratio_relative_deviation']:
                    group['issues'].append(dict(reason='paired_timing_ratio_outlier',
                        stdepp_to_wdd=ratios[group['lane']], ratio_to_seed_median=ratios[group['lane']]/med))
    return dict(protocol=PROTOCOL, groups=groups,
                flagged_lanes=[g['lane'] for g in groups if g['issues']],
                selection_by_final_error=False, preserve_all_attempts=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'preflight', 'cell'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--index', type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.stage == 'plan':
        print(json.dumps(plan(args.out), indent=2))
    elif args.stage == 'preflight':
        preflight(args.out, args.device)
    else:
        if args.index is None or args.gate is None:
            parser.error('cell requires --index and --gate')
        run_cell(args.out, args.index, args.gate, device=args.device)


if __name__ == '__main__':
    main()
