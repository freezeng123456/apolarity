"""One isolated eager-PyTorch cell of the fixed mixed-partial matrix.

The timed operation is value-only.  It deliberately performs no network
parameter backward pass and uses neither ``torch.compile`` nor a whole-function
JIT wrapper.  ``nested_jvp`` is the fixed coordinate-JVP baseline; the two
Waring methods use the existing Taylor engine with identical directions and
weights, differing only in serial versus batched direction execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from torch_pinn.model import MLP, sample_normal
from torch_pinn.monomials import monomial, schedule


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    Path('experiments/torch_benchmarks/first_group.py'),
    Path('experiments/torch_benchmarks/first_group_verify.py'),
    Path('experiments/torch_benchmarks/run_first_group_matrix.py'),
    Path('experiments/torch_benchmarks/audit_first_group.py'),
    Path('experiments/torch_benchmarks/launch_first_group_t4.py'),
    Path('experiments/torch_benchmarks/torch_pinn/model.py'),
    Path('experiments/torch_benchmarks/torch_pinn/monomials.py'),
    Path('experiments/torch_benchmarks/torch_pinn/jets.py'),
    Path('experiments/torch_benchmarks/torch_pinn/operators.py'),
    Path('experiments/torch_benchmarks/torch_pinn/coordinate_jvp.py'),
    Path('experiments/torch_benchmarks/torch_pinn/stde_sampling.py'),
)


def parse_one_based_pattern(pattern: str) -> tuple[int, ...]:
    text = pattern.strip().replace('u_', '').replace('_', '')
    if not text or not text.isdigit() or '0' in text:
        raise ValueError(f'invalid one-based pattern {pattern!r}')
    return tuple(int(char)-1 for char in text)


def source_files_sha256() -> dict[str, str]:
    return {
        str(path): hashlib.sha256((REPO_ROOT/path).read_bytes()).hexdigest()
        for path in SOURCE_FILES
    }


def source_commit() -> str:
    return subprocess.check_output(
        ['git', '-C', str(REPO_ROOT), 'rev-parse', 'HEAD'], text=True
    ).strip()


def tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    header = f'{cpu.dtype}|{tuple(cpu.shape)}|'.encode()
    return hashlib.sha256(header+cpu.numpy().tobytes()).hexdigest()


def parameters_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode()+b'\0'+tensor_sha256(parameter).encode()+b'\n')
    return digest.hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def configure_eager_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the formal first-group matrix')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('highest')


def make_model_and_points(seed: int, batch: int, width: int, depth: int, device: str):
    # This matches the historical JAX convention exactly: model seed + 2 and
    # normal-input seed + 1, with NumPy PCG64-generated float32 arrays.
    model = MLP(16, width, depth, seed+2, dtype=torch.float32, device=device)
    model.eval()
    points = sample_normal(seed+1, batch, 16, dtype=torch.float32, device=device)
    return model, points


def evaluate(model, points: torch.Tensor, alpha: tuple[int, ...], method: str):
    return monomial(model, points, alpha, method=method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--method', choices=('nested_jvp', 'waring_batched'), required=True)
    parser.add_argument('--batch', type=int, default=100)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=(20260908, 20260909, 20260910))
    parser.add_argument('--warmups', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.batch < 1 or args.warmups < 0 or args.repeats < 1:
        raise ValueError('batch must be positive, warmups nonnegative and repeats positive')
    args.out.mkdir(parents=True)
    status = args.out/'status.txt'
    status.write_text('PREPARING\n')
    try:
        configure_eager_cuda()
        alpha = parse_one_based_pattern(args.target)
        directions, weights = schedule(alpha, 16, dtype=torch.float32, device='cuda')
        cold_model, cold_points = make_model_and_points(args.seeds[0], args.batch, args.width, args.depth, 'cuda')
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            torch.cuda.synchronize()
            started = time.perf_counter()
            cold_value = evaluate(cold_model, cold_points, alpha, args.method)
            torch.cuda.synchronize()
            cold_call_ms = (time.perf_counter()-started)*1000
        if cold_value.requires_grad:
            raise AssertionError('value-only evaluator unexpectedly retained a parameter reverse graph')
        if cold_value.shape != (args.batch,) or cold_value.dtype != torch.float32 or not torch.isfinite(cold_value).all():
            raise ValueError('cold call produced nonfinite or incorrectly typed values')
        card = {
            'protocol': 'torch_first_group_values_v1',
            'mode': 'derivative_values_only',
            'parameter_backward': False,
            'outer_jit': False,
            'torch_compile': False,
            'third_party_differentiation_library': False,
            'target': args.target,
            'method': args.method,
            'batch': args.batch,
            'width': args.width,
            'depth': args.depth,
            'seeds': list(args.seeds),
            'warmups': args.warmups,
            'repeats': args.repeats,
            'input_dim': 16,
            'order': len(alpha),
            'rank': len(directions),
            'alpha': list(alpha),
            'dtype_real': 'float32',
            'dtype_direction': str(directions.dtype).removeprefix('torch.'),
            'dtype_weight': str(weights.dtype).removeprefix('torch.'),
            'directions_sha256': tensor_sha256(directions),
            'weights_sha256': tensor_sha256(weights),
            'direction_execution': 'batched',
            'direction_chunking': False,
            'input_distribution': 'normal(mean=0,std=0.35) in all 16 coordinates',
            'initialization': 'NumPy PCG64 uniform[-1/sqrt(fan_in),+1/sqrt(fan_in)]',
            'cold_call_ms_excluded_from_timing': cold_call_ms,
            'cuda_synchronized_timing': True,
            'matmul_precision': 'highest',
            'tf32': False,
            'python': sys.version,
            'torch': torch.__version__,
            'torch_cuda': torch.version.cuda,
            'numpy': np.__version__,
            'platform': platform.platform(),
            'backend': 'gpu',
            'device_kind': torch.cuda.get_device_name(0),
            'device_capability': list(torch.cuda.get_device_capability(0)),
            'source_commit': source_commit(),
            'source_files_sha256': source_files_sha256(),
        }
        save(args.out/'config.json', card)
        status.write_text('RUNNING\n')
        records = []
        for seed in args.seeds:
            model, points = make_model_and_points(seed, args.batch, args.width, args.depth, 'cuda')
            parameter_hash = parameters_sha256(model)
            points_hash = tensor_sha256(points)
            with torch.no_grad():
                for _ in range(args.warmups):
                    warm_value = evaluate(model, points, alpha, args.method)
                torch.cuda.synchronize()
                samples = []
                for _ in range(args.repeats):
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    value = evaluate(model, points, alpha, args.method)
                    torch.cuda.synchronize()
                    samples.append((time.perf_counter()-started)*1000)
            if value.requires_grad:
                raise AssertionError('timed value unexpectedly retained a parameter reverse graph')
            if value.shape != (args.batch,) or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError('timed call produced nonfinite or incorrectly typed values')
            array = value.detach().cpu().numpy()
            np.save(args.out/f'values_{seed}.npy', array)
            records.append({
                'target': args.target,
                'method': args.method,
                'batch': args.batch,
                'seed': seed,
                'order': len(alpha),
                'rank': len(directions),
                'parameters_sha256': parameter_hash,
                'points_sha256': points_hash,
                'output_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                'output_shape': list(array.shape),
                'output_dtype': str(array.dtype),
                'times_ms': samples,
                'median_ms': statistics.median(samples),
                'allocator_peak_bytes': torch.cuda.max_memory_allocated(),
                'status': 'ok',
            })
            save(args.out/'results.json', records)
        status.write_text('COMPLETE\n')
        print(json.dumps({'out': str(args.out), 'records': len(records), 'cold_call_ms': cold_call_ms}))
    except Exception as exc:
        save(args.out/'failure.json', {'error': repr(exc)})
        status.write_text('FAILED\n')
        raise


if __name__ == '__main__':
    main()
