#!/usr/bin/env python3
"""Audit the current common-Xavier float32/complex64 Poly o6 setup.

The diagnostic records initial loss-component gradient norms for WAR and real
autodiff. It is intentionally read-only with respect to formal result bundles.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
SRC = ROOT / "src"
for path in (str(COMMON), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from weight_search import (  # noqa: E402
    METHODS,
    TASKS,
    build_search_model,
    make_poly_loss,
)


def gradient_summary(
    value: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    retain_graph: bool,
) -> dict[str, float | int]:
    gradients = torch.autograd.grad(
        value,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    squared = 0.0
    maximum = 0.0
    finite = True
    element_count = 0
    nonzero_count = 0
    for gradient in gradients:
        if gradient is None:
            continue
        magnitude = gradient.detach().abs()
        finite = finite and bool(torch.isfinite(magnitude).all().item())
        squared += float(magnitude.square().sum().item())
        maximum = max(maximum, float(magnitude.max().item()))
        element_count += magnitude.numel()
        nonzero_count += int((magnitude > 0).sum().item())
    return {
        "l2": math.sqrt(squared),
        "max_abs": maximum,
        "finite": int(finite),
        "element_count": element_count,
        "nonzero_count": nonzero_count,
        "nonzero_fraction": (
            nonzero_count / element_count if element_count else 0.0
        ),
    }


def diagnose_method(
    method: str,
    seed: int,
    device: torch.device,
    n_int: int,
    n_bc: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    task = TASKS["poly_d2_o6"]
    model, dtype, backend = build_search_model(task, method, device)
    bundle = make_poly_loss(
        task,
        model,
        dtype,
        backend,
        (10.0, 1.0, 1.0),
        device,
        n_int=n_int,
        n_bc=n_bc,
        n_eval=max(1024, n_int),
        history_eval_n=min(1024, max(1024, n_int)),
        train_seed=seed,
        eval_seed=54321,
    )
    loss, components = bundle.loss_fn()
    parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    names = (
        "L_PDE",
        "weighted_L_bc_j0",
        "weighted_L_bc_j1",
        "weighted_L_bc_j2",
        "loss",
    )
    gradients: dict[str, dict[str, float | int]] = {}
    for index, name in enumerate(names):
        gradients[name] = gradient_summary(
            components[name],
            parameters,
            retain_graph=index < len(names) - 1,
        )
    return {
        "method": method,
        "seed": seed,
        "dtype": str(dtype),
        "backend": backend,
        "initial_rel_error": bundle.eval_fn(),
        "components": {
            key: float(value.detach().real.item())
            for key, value in components.items()
        },
        "gradient_norms": gradients,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-int", type=int, default=512)
    parser.add_argument("--n-bc", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seeds <= 0 or args.n_int <= 0 or args.n_bc <= 0:
        raise ValueError("seeds and sample counts must be positive")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    payload = {
        "protocol_id": "poly_d2_o6_common_xavier_gradient_audit_v1",
        "task_id": "poly_d2_o6",
        "weights": [10.0, 1.0, 1.0],
        "device": str(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "n_int": args.n_int,
        "n_bc": args.n_bc,
        "results": [
            diagnose_method(method, seed, device, args.n_int, args.n_bc)
            for seed in range(args.seeds)
            for method in METHODS
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded + "\n")
        temporary.replace(args.output)
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
