"""5-minute PINN backend comparison on the 4D Cahn--Hilliard-type 6th-order PDE.

PDE:    partial_{x1}^4 partial_{x2}^2 u(x) = f(x)  on  (-1,1)^4
        with manufactured  u_exact = sinh(x1) cos(x2) exp(-(x3^2+x4^2)/4),
        f = -u_exact.

Network: 4-layer Linear-sinh MLP, complex128 parameters.
Loss:   ||Re(d^{(4,2)} u) - f||^2 + bc_w ||Re(u) - u_exact||^2_{boundary}
        + im_w sum ||Im(W)||^2.

Each backend trains for a fixed wall-clock budget (default 60 s).  Final
metrics: steps/sec, peak GPU memory, final relative L2 error vs u_exact.

Run:
    bash scripts/cuda_env.sh
    PYTHONPATH=src python3.11 experiments/pinn_5min_compare.py
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from apolarity import single_monomial_partial


# ---------------------------------------------------------------------------
# Manufactured solution
# ---------------------------------------------------------------------------

def u_exact(x: Tensor) -> Tensor:
    x1, x2, x3, x4 = x.unbind(dim=-1)
    return torch.sinh(x1) * torch.cos(x2) * torch.exp(-(x3 ** 2 + x4 ** 2) / 4.0)


def source_f(x: Tensor) -> Tensor:
    # d^{(4,2)} u_exact = sinh(x1) * (-cos(x2)) * exp(...) = -u_exact
    return -u_exact(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SinhActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sinh(x)


def build_complex_sinh_mlp(d: int, hidden: int, depth: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_dim = d
    for _ in range(depth):
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(SinhActivation())
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers).to(dtype=torch.complex128)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_interior(B: int, d: int, *, device, dtype) -> Tensor:
    x = torch.empty(B, d, device=device, dtype=torch.float64).uniform_(-1.0, 1.0)
    return x.to(dtype=dtype)


def sample_boundary(B: int, d: int, *, device, dtype) -> Tensor:
    x = torch.empty(B, d, device=device, dtype=torch.float64).uniform_(-1.0, 1.0)
    face = torch.randint(0, d, (B,), device=device)
    sign = torch.where(
        torch.rand(B, device=device) < 0.5,
        torch.tensor(-1.0, dtype=torch.float64, device=device),
        torch.tensor(1.0, dtype=torch.float64, device=device),
    )
    rows = torch.arange(B, device=device)
    x[rows, face] = sign
    return x.to(dtype=dtype)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

ALPHA = (0, 0, 0, 0, 1, 1)  # u_{x1 x1 x1 x1 x2 x2}, expanded form
BC_W, IM_W = 100.0, 1.0e-6


def pinn_loss(model, x_int, x_bc, backend):
    deriv = single_monomial_partial(model, x_int, ALPHA, backend=backend)
    f = source_f(x_int.real).unsqueeze(-1)
    L_int = ((deriv.real - f) ** 2).mean()
    u_bc = model(x_bc)
    L_bc = ((u_bc.real - u_exact(x_bc.real).unsqueeze(-1)) ** 2).mean()
    L_im = sum((p.imag ** 2).mean() for p in model.parameters())
    return L_int + BC_W * L_bc + IM_W * L_im, L_int.item(), L_bc.item()


@torch.no_grad()
def l2_error(model, n_eval, d, *, device) -> float:
    x = torch.empty(n_eval, d, device=device, dtype=torch.float64).uniform_(-1.0, 1.0)
    pred = model(x.to(dtype=torch.complex128)).real.squeeze(-1)
    target = u_exact(x)
    return (((pred - target) ** 2).mean().sqrt() / (target ** 2).mean().sqrt()).item()


# ---------------------------------------------------------------------------
# Train one backend for a fixed wall-clock budget
# ---------------------------------------------------------------------------

def run_backend(
    backend: str,
    *,
    seconds: float,
    hidden: int,
    depth: int,
    n_int: int,
    n_bc: int,
    lr: float,
    seed: int,
    device,
    d: int = 4,
):
    torch.manual_seed(seed)
    dtype = torch.complex128
    model = build_complex_sinh_mlp(d, hidden, depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Warm-up: 3 steps to compile / build cudnn handles.
    for _ in range(3):
        x_int = sample_interior(n_int, d, device=device, dtype=dtype)
        x_bc = sample_boundary(n_bc, d, device=device, dtype=dtype)
        loss, _, _ = pinn_loss(model, x_int, x_bc, backend)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    n_steps = 0
    losses: list[float] = []
    while time.perf_counter() - t0 < seconds:
        x_int = sample_interior(n_int, d, device=device, dtype=dtype)
        x_bc = sample_boundary(n_bc, d, device=device, dtype=dtype)
        loss, L_int, _ = pinn_loss(model, x_int, x_bc, backend)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n_steps += 1
        losses.append(L_int)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    err = l2_error(model, n_eval=4096, d=d, device=device)
    peak = (
        torch.cuda.max_memory_allocated(device) / 2 ** 20
        if device.type == "cuda" else float("nan")
    )
    return {
        "backend": backend,
        "steps": n_steps,
        "elapsed_s": elapsed,
        "ms_per_step": 1000.0 * elapsed / max(1, n_steps),
        "peak_mb": peak,
        "L_int_first": losses[0] if losses else float("nan"),
        "L_int_last": sum(losses[-20:]) / max(1, min(20, len(losses))) if losses else float("nan"),
        "L2_err": err,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=60.0,
                   help="wall-clock seconds per backend")
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--n-int", type=int, default=128)
    p.add_argument("--n-bc", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backends", default="waring_complex_jet,polarization_jet,direct_autodiff")
    p.add_argument("--out", default="",
                   help="Optional CSV path; a JSON file with the same stem is also written")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}, dtype=complex128, hidden={args.hidden}, depth={args.depth}, "
          f"n_int={args.n_int}, n_bc={args.n_bc}, seconds/backend={args.seconds}")
    print(f"PDE: d^(4,2) u = f on (-1,1)^4,  alpha={ALPHA}\n")

    results = []
    for be in args.backends.split(","):
        be = be.strip()
        print(f"-- running backend = {be} ...")
        try:
            r = run_backend(
                be, seconds=args.seconds,
                hidden=args.hidden, depth=args.depth,
                n_int=args.n_int, n_bc=args.n_bc,
                lr=args.lr, seed=args.seed, device=device,
            )
            results.append(r)
            print(f"   steps={r['steps']:5d}  ms/step={r['ms_per_step']:7.2f}  "
                  f"peak={r['peak_mb']:7.1f} MB  "
                  f"L_int: {r['L_int_first']:.2e} -> {r['L_int_last']:.2e}  "
                  f"L2 err={r['L2_err']:.3e}")
        except Exception as e:
            print(f"   FAILED: {type(e).__name__}: {e}")
            results.append({"backend": be, "error": str(e)})

    print("\n=== summary ===")
    print(f"{'backend':<22s} {'steps':>6s} {'ms/step':>9s} {'peak (MB)':>10s} "
          f"{'L_int_init':>10s} {'L_int_end':>10s} {'L2 err':>10s}")
    for r in results:
        if "error" in r:
            print(f"{r['backend']:<22s}  FAILED: {r['error'][:60]}")
            continue
        print(f"{r['backend']:<22s} {r['steps']:>6d} {r['ms_per_step']:>9.2f} "
              f"{r['peak_mb']:>10.1f} {r['L_int_first']:>10.2e} "
              f"{r['L_int_last']:>10.2e} {r['L2_err']:>10.3e}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = []
        for row in results:
            for key in row:
                if key not in keys:
                    keys.append(key)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        with out.with_suffix(".json").open("w") as f:
            json.dump(results, f, indent=2)
        print(f"[ok] wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
