"""Sequential single-GPU matrix with isolated cell processes and durable outputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--execution", choices=["no_outer_jit", "whole_function_jit"], default="no_outer_jit")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    targets = ["111111", "111222", "112233", "123456", "11223344", "111222333"]
    cells = [(t, m, 100) for t in targets for m in ("nested_jvp", "waring_batched")]
    cells += [(t, "waring_serial", 100) for t in ("123456", "111222333")]
    cells += [(t, m, b) for t in ("123456", "111222333") for b in (400, 1600)
              for m in ("nested_jvp", "waring_batched")]
    manifest = {"expected_cells": len(cells), "expected_rows": len(cells) * 3,
                "seeds": [20260908, 20260909, 20260910], "cells": cells,
                "python": sys.executable, "timeout_per_cell_s": args.timeout,
                "protocol": "jax_first_group_values_v2", "execution": args.execution}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    env = os.environ.copy()
    env.update(JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false",
               OMP_NUM_THREADS="1", JAX_DEFAULT_MATMUL_PRECISION="highest")
    env.pop("LD_LIBRARY_PATH", None)
    status = args.out / "status.txt"
    status.write_text("RUNNING\n")
    outcomes = []
    for target, method, batch in cells:
        name = f"{target}_{method}_b{batch}"
        cmd = [sys.executable, str(Path(__file__).with_name("first_group.py")),
               "--target", target, "--method", method, "--batch", str(batch),
               "--execution", args.execution, "--out", str(args.out / name)]
        with (args.out / f"{name}.log").open("w") as log:
            start = time.time()
            try:
                code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout).returncode
            except subprocess.TimeoutExpired:
                code = 124
                (args.out / f"{name}.timeout").write_text("cell exceeded declared wall-clock budget\n")
            outcomes.append({"cell": name, "exit_code": code, "elapsed_seconds": time.time() - start})
        (args.out / "progress.json").write_text(json.dumps(outcomes, indent=2))
        print(json.dumps(outcomes[-1]), flush=True)
    status.write_text("COMPLETE\n" if all(o["exit_code"] == 0 for o in outcomes) else "FINISHED_WITH_FAILURES\n")
    hashes = []
    for p in sorted(args.out.rglob("*")):
        if p.is_file():
            hashes.append(hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(args.out)))
    (args.out / "SHA256SUMS").write_text("\n".join(hashes) + "\n")


if __name__ == "__main__":
    main()
