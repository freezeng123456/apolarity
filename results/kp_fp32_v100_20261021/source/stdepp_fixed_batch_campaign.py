"""Original six partials plus the three balanced candidates, fixed B=1000.

The order-nine (3,3,3) candidate already belongs to the original six, so the
deduplicated campaign contains eight targets. Arithmetic and accuracy/timing
gates are shared with the earlier scaling study.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

import stdepp_scaling_campaign as core
from equal_updates_campaign import TARGETS

PROTOCOL = 'apolarity_stdepp_section61_fixed_b1000_v1'
EXPONENTS = ((6,), (3,3), (2,2,2), (1,1,1,1,1,1), (2,2,2,2), (3,3,3), (4,4,4), (5,5,5))
BATCH, WIDTH = 1000, 128


def matrix(output):
    rows = []
    for exponents in EXPONENTS:
        target = core.target_name(exponents)
        name = f'{target}_B{BATCH}_W{WIDTH}'
        rows.append(dict(index=len(rows), name=name, target=target, exponents=list(exponents),
            order=sum(exponents), out=str(Path(output)/'partials'/name),
            methods=list(core.METHODS), seeds=list(core.PARTIAL_SEEDS), input_dim=16,
            batch=BATCH, width=WIDTH, depth=4, warmups=10, repeats=30))
    assert len(rows) == len({r['target'] for r in rows}) == 8
    assert set(TARGETS).issubset({r['target'] for r in rows})
    return rows


def plan(output):
    result = core.plan(output)
    result.update(protocol=PROTOCOL, cells=matrix(output), fixed_batch=BATCH, fixed_width=WIDTH,
        original_targets=list(TARGETS), added_candidate_exponents=[[3,3,3],[4,4,4],[5,5,5]],
        duplicate_candidate='(3,3,3) is the original 111222333 target; evaluated once per seed',
        fresh_rerun=True)
    return result


def preflight_body(args):
    env = core.environment(args.device)
    rows = core.independent_checks(args.device, EXPONENTS)
    largest = core.largest_shape_probe(args.device, EXPONENTS[-1], (BATCH, WIDTH))
    status = 'PASS' if all(r['passed'] for r in rows) else 'INVALID_NUMERICS'
    core.save(args.out/'PASS.json', dict(status=status, protocol=PROTOCOL,
        source_fingerprint=core.source_fingerprint(), environment=env,
        independent_analytic_float64=rows, largest_shape_probe=largest,
        batch=BATCH, width=WIDTH, checkpoints=False))
    return 'COMPLETE' if status == 'PASS' else 'INVALID_NUMERICS'


def receipt(output, device):
    folder = Path(output)/'preflight'
    if not (folder/'status.txt').is_file() or (folder/'status.txt').read_text().strip() != 'COMPLETE':
        raise RuntimeError('A completed fixed-batch preflight is required')
    record = json.loads((folder/'PASS.json').read_text())
    expected = 'Tesla V100-SXM2-32GB' if torch.device(device).type == 'cuda' else 'CPU_LOCAL_VALIDATION_ONLY'
    if (record['status'] != 'PASS' or record['protocol'] != PROTOCOL or
        record['source_fingerprint'] != core.source_fingerprint() or
        record['environment']['gpu'] != expected or record['batch'] != BATCH or record['width'] != WIDTH):
        raise RuntimeError('Fixed-batch preflight configuration or source mismatch')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan','preflight','cell'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--index', type=int)
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    args = parser.parse_args(argv)
    if args.action == 'plan':
        print(json.dumps(plan(args.out), indent=2))
        return
    common = dict(device=args.device, expected_device='V100', campaign_protocol=PROTOCOL,
        campaign_root=str(args.out))
    if args.action == 'preflight':
        core.run_stage(argparse.Namespace(action=args.action, out=args.out/'preflight', **common), preflight_body)
    else:
        cells = matrix(args.out)
        if args.index is None or not 0 <= args.index < len(cells):
            parser.error('--index must be between zero and seven')
        receipt(args.out, args.device)
        cell = cells[args.index]
        cell['out'] = Path(cell['out'])
        core.run_stage(argparse.Namespace(action=args.action, **cell, **common), core._cell_body)


if __name__ == '__main__':
    main()
