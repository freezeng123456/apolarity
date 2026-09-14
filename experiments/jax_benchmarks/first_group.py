"""Value-only mixed-partial benchmark: official JAX AD versus Waring + jet.

Each process handles one target/method/batch; parameters and points are dynamic
arguments to both evaluators. No network-parameter gradients are taken.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from jax_apolarity_bench.derivatives import directional_taylor
from jax_apolarity_bench.directions import parse_one_based_pattern, rank_optimal_schedule
from jax_apolarity_bench.models import init_mlp, mlp_one, sample_normal


def make_evaluator(alpha, dim, method, *, serial_python=False):
    """All methods compute one identical partial, not its full derivative tensor."""
    if method == "nested_jvp":
        basis = jnp.eye(dim, dtype=jnp.float32)

        def evaluate(params, points):
            f = lambda x: mlp_one(params, x)
            for idx in alpha:
                previous = f
                f = lambda x, previous=previous, idx=idx: jax.jvp(previous, (x,), (basis[idx],))[1]
            return jax.vmap(f)(points)

        return evaluate
    schedule = rank_optimal_schedule(alpha, dim)

    def evaluate(params, points):
        def along(v):
            return jax.vmap(lambda x: directional_taylor(lambda z: mlp_one(params, z), x, v, len(alpha)))(points)

        if method == "waring_batched":
            coefficients = jax.vmap(along)(schedule.directions)
        elif method == "waring_serial":
            if serial_python:
                # No compiled scan body in the no-outer-JIT execution policy.
                coefficients = jnp.stack([along(v) for v in schedule.directions])
            else:
                def body(carry, v):
                    return carry, along(v)
                _, coefficients = jax.lax.scan(body, None, schedule.directions)
        else:
            raise ValueError(method)
        return jnp.real(jnp.sum(schedule.weights[:, None] * coefficients, axis=0))

    return evaluate


def source_hash():
    h = hashlib.sha256()
    root = Path(__file__).resolve().parent
    for p in sorted([Path(__file__).resolve(), *root.joinpath("src").rglob("*.py")]):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--method", choices=["nested_jvp", "waring_batched", "waring_serial"], required=True)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260908, 20260909, 20260910])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--execution", choices=["no_outer_jit", "whole_function_jit"], default="no_outer_jit")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    jax.config.update("jax_default_matmul_precision", "highest")
    jax.config.update("jax_enable_x64", False)
    alpha = parse_one_based_pattern(args.target)
    dim = 16
    schedule = rank_optimal_schedule(alpha, dim)
    params = init_mlp(args.seeds[0] + 2, dim, args.width, args.depth)
    points = sample_normal(args.seeds[0] + 1, args.batch, dim)
    jax.block_until_ready((params, points))
    device = jax.devices()[0]
    card = {
        **vars(args), "out": str(args.out), "protocol": "jax_first_group_values_v2",
        "mode": "derivative_values_only", "parameter_backward": False,
        "python": sys.version, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "platform": platform.platform(), "device": str(device), "device_kind": device.device_kind,
        "backend": jax.default_backend(), "source_sha256": source_hash(),
        "input_dim": dim, "order": len(alpha), "rank": schedule.theoretical_rank,
        "dtype_real": "float32", "dtype_direction": str(schedule.directions.dtype),
        "precision": "highest", "preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "parameters_are_dynamic": True, "inputs_are_dynamic": True,
        "whole_function_jit": args.execution == "whole_function_jit",
        "serial_direction_execution": "python_loop" if args.execution == "no_outer_jit" else "lax_scan",
        "note": "No outer JIT does not disable JAX primitive-level compilation.",
    }
    save(args.out / "config.json", card)
    status = args.out / "status.txt"
    status.write_text("PREPARING\n")
    try:
        start = time.perf_counter()
        evaluate = make_evaluator(alpha, dim, args.method, serial_python=args.execution == "no_outer_jit")
        if args.execution == "whole_function_jit":
            evaluate = jax.jit(evaluate).lower(params, points).compile()
            compile_s = time.perf_counter() - start
            analysis = evaluate.memory_analysis()
        else:
            compile_s, analysis = None, None
        memory = {name: int(getattr(analysis, name)) for name in (
            "argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes"
        )} if analysis else None
        start = time.perf_counter()
        evaluate(params, points).block_until_ready()
        first_call_s = time.perf_counter() - start
        status.write_text("RUNNING\n")
        rows = []
        for seed in args.seeds:
            params = init_mlp(seed + 2, dim, args.width, args.depth)
            points = sample_normal(seed + 1, args.batch, dim)
            jax.block_until_ready((params, points))
            for _ in range(args.warmups):
                evaluate(params, points).block_until_ready()
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                value = evaluate(params, points)
                value.block_until_ready()
                samples.append((time.perf_counter() - start) * 1000)
            array = np.asarray(value)
            if array.shape != (args.batch,) or not np.isfinite(array).all():
                raise ValueError("nonfinite or incorrect output shape")
            np.save(args.out / f"values_{seed}.npy", array)
            stats = device.memory_stats() or {}
            rows.append({
                "target": args.target, "method": args.method, "batch": args.batch, "seed": seed,
                "order": len(alpha), "rank": schedule.theoretical_rank,
                "execution": args.execution, "first_call_seconds": first_call_s,
                "median_ms": statistics.median(samples), "times_ms": samples,
                "compile_seconds": compile_s, "executable_memory": memory,
                "allocator_peak_bytes": stats.get("peak_bytes_in_use"),
                "allocator_bytes_in_use": stats.get("bytes_in_use"),
                "status": "ok",
            })
            save(args.out / "results.json", rows)
        status.write_text("COMPLETE\n")
        print(json.dumps({"out": str(args.out), "rows": len(rows), "compile_s": compile_s}), flush=True)
    except Exception as exc:
        save(args.out / "failure.json", {"error": repr(exc)})
        status.write_text("FAILED\n")
        raise


if __name__ == "__main__":
    main()
