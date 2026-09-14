"""Pre-timing correctness gates for the eager-PyTorch first-group matrix."""
from __future__ import annotations

import argparse
from collections import Counter
from math import factorial, prod
from pathlib import Path
import json

import numpy as np
import torch

from first_group import (
    configure_eager_cuda,
    make_model_and_points,
    parse_one_based_pattern,
    source_commit,
    source_files_sha256,
)
from torch_pinn.monomials import monomial, schedule


TARGETS = ('111111', '111222', '112233', '123456', '11223344', '111222333')
EXPECTED_RANKS = dict(zip(TARGETS, (1, 4, 9, 32, 27, 16)))
EXPECTED_DTYPE = {
    '111111': 'float32', '111222': 'complex64', '112233': 'complex64',
    '123456': 'float32', '11223344': 'complex64', '111222333': 'complex64',
}


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def independent_polynomial_check(alpha: tuple[int, ...], directions, weights) -> tuple[float, float]:
    """Check the schedule without using any AD/Taylor implementation.

    For P(x)=prod_i x_i**alpha_i, its normalized order-n directional Taylor
    coefficient is prod_i v_i**alpha_i.  The target mixed partial is exactly
    prod_i alpha_i!, so this checks roots-of-unity weights and factorial
    normalization independently of the network evaluator.
    """
    counts = Counter(alpha)
    direction_array = directions.detach().cpu().numpy()
    weight_array = weights.detach().cpu().numpy()
    coefficient = np.prod([
        direction_array[:, axis]**exponent for axis, exponent in counts.items()
    ], axis=0)
    observed = np.sum(weight_array*coefficient)
    expected = prod(factorial(exponent) for exponent in counts.values())
    return float(np.real(observed)), float(expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    status = args.out/'status.txt'
    status.write_text('PREPARING\n')
    try:
        configure_eager_cuda()
        rows = []
        with torch.no_grad():
            for target in TARGETS:
                alpha = parse_one_based_pattern(target)
                # Float64 reduces comparison rounding; schedules retain their
                # required float32/complex64 semantics in the formal cells.
                directions, weights = schedule(alpha, 16, dtype=torch.float64, device='cuda')
                observed, expected = independent_polynomial_check(alpha, directions, weights)
                if not np.isclose(observed, expected, rtol=1e-12, atol=1e-12):
                    raise AssertionError((target, observed, expected))
                model, points = make_model_and_points(20260908, 3, 16, 2, 'cuda')
                model = model.to(dtype=torch.float64)
                points = points.to(dtype=torch.float64)
                nested = monomial(model, points, alpha, 'nested_jvp')
                batched = monomial(model, points, alpha, 'waring_batched')
                serial = monomial(model, points, alpha, 'waring_serial')
                torch.testing.assert_close(batched, nested, rtol=1e-7, atol=1e-9)
                torch.testing.assert_close(serial, batched, rtol=1e-7, atol=1e-9)
                if nested.requires_grad or batched.requires_grad or serial.requires_grad:
                    raise AssertionError('correctness gate retained a parameter reverse graph')
                formal_directions, _ = schedule(alpha, 16, dtype=torch.float32, device='cuda')
                rows.append({
                    'target': target,
                    'order': len(alpha),
                    'expected_rank': EXPECTED_RANKS[target],
                    'observed_rank': len(directions),
                    'formal_direction_dtype': str(formal_directions.dtype).removeprefix('torch.'),
                    'expected_formal_direction_dtype': EXPECTED_DTYPE[target],
                    'independent_polynomial_value': observed,
                    'independent_polynomial_expected': expected,
                    'max_abs_batched_vs_nested': float((batched-nested).abs().max().cpu()),
                    'max_abs_serial_vs_batched': float((serial-batched).abs().max().cpu()),
                })
                if len(directions) != EXPECTED_RANKS[target] or rows[-1]['formal_direction_dtype'] != EXPECTED_DTYPE[target]:
                    raise AssertionError(rows[-1])
        result = {
            'status': 'PASS',
            'protocol': 'torch_first_group_values_v1_preflight',
            'source_commit': source_commit(),
            'source_files_sha256': source_files_sha256(),
            'backend': 'gpu',
            'device_kind': torch.cuda.get_device_name(0),
            'torch': torch.__version__,
            'parameter_backward': False,
            'outer_jit': False,
            'torch_compile': False,
            'independent_reference': 'analytic monomial directional coefficient and exact mixed partial',
            'mlp_crosscheck': 'float64 official coordinate nested JVP versus batched/serial Waring Taylor values',
            'rows': rows,
        }
        save(args.out/'verification.json', result)
        status.write_text('PASS\n')
        print(json.dumps({'status': 'PASS', 'targets': len(rows)}))
    except Exception as exc:
        save(args.out/'failure.json', {'error': repr(exc)})
        status.write_text('FAILED\n')
        raise


if __name__ == '__main__':
    main()
