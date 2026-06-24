"""4D Cahn-Hilliard-type sixth-order PINN training (Section 5.4 of the JSC paper).

PDE
===
On Omega = (-1, 1)^4,
    partial_{x1}^4 partial_{x2}^2 u(x) = f(x),       u|_{boundary} given,
with manufactured exact solution
    u_exact(x) = sinh(x1) cos(x2) exp(-(x3^2 + x4^2) / 4)
and source f computed analytically below.

Active indices for the dominant operator are {1, 2} with active-exponent
pattern (4, 2). Theorem 3.4 of the paper gives R_C = 5, vs polarization
direction count 7.

Network
=======
Six-layer Linear -- sinh MLP, complex parameters (torch.complex128). The
PINN loss is built on Re(u). Real-input data is embedded as x + 0j.

Backends
========
We expose the three backends of Section 4.2 via a single command-line flag:
    --backend [direct | polarization | waring_complex]
plus an oracle reference computed via direct nested autograd.

Usage
=====
    python experiments/train_pinn_ch_sixth_order.py \
        --backend waring_complex --hidden 64 --depth 6 --steps 20000

The script reports per-step wall-clock, peak memory, and final L2 error on
a held-out grid against u_exact. Match against Table 3 of the paper.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from apolarity import single_monomial_partial


# ---------------------------------------------------------------------------
# Manufactured solution and source
# ---------------------------------------------------------------------------

def u_exact(x: Tensor) -> Tensor:
    """u_exact(x) = sinh(x1) cos(x2) exp(-(x3^2 + x4^2) / 4)."""
    x1, x2, x3, x4 = x.unbind(dim=-1)
    return torch.sinh(x1) * torch.cos(x2) * torch.exp(-(x3 ** 2 + x4 ** 2) / 4.0)


def source_f(x: Tensor) -> Tensor:
    """f(x) = partial_{x1}^4 partial_{x2}^2 u_exact(x).

    sinh^{(4)}(x1) = sinh(x1).
    cos^{(2)}(x2)  = -cos(x2).
    The Gaussian factor in x3, x4 is unaffected.
    Hence f = -sinh(x1) cos(x2) exp(-(x3^2 + x4^2)/4) = -u_exact.
    """
    return -u_exact(x)


# ---------------------------------------------------------------------------
# Complex-parameter MLP with sinh activation
# ---------------------------------------------------------------------------

class SinhActivation(nn.Module):
    """Pointwise sinh; named so that taylor_jet's _is_sinh_module recognises it."""

    def forward(self, x: Tensor) -> Tensor:
        return torch.sinh(x)


def build_complex_sinh_mlp(d: int, hidden: int, depth: int) -> nn.Sequential:
    """Linear -- sinh MLP with torch.complex128 parameters and 1-D output.

    The final layer outputs a complex scalar; the PINN loss takes Re(.)
    in train_step.
    """
    layers: list[nn.Module] = []
    in_dim = d
    for k in range(depth):
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(SinhActivation())
        in_dim = hidden
    layers.append(nn.Linear(in_dim, 1))
    net = nn.Sequential(*layers)
    return net.to(dtype=torch.complex128)


# ---------------------------------------------------------------------------
# Sampling and the PINN loss
# ---------------------------------------------------------------------------

def sample_interior(B: int, d: int, *, device, dtype) -> Tensor:
    """B uniform samples from (-1, 1)^d, embedded as complex with 0 imaginary part."""
    x = torch.empty(B, d, device=device, dtype=torch.float64).uniform_(-1.0, 1.0)
    return x.to(dtype=dtype)


def sample_boundary(B: int, d: int, *, device, dtype) -> Tensor:
    """B uniform samples on the boundary of (-1, 1)^d. Picks a random face per sample."""
    x = torch.empty(B, d, device=device, dtype=torch.float64).uniform_(-1.0, 1.0)
    face = torch.randint(0, d, (B,), device=device)
    sign = torch.where(torch.rand(B, device=device) < 0.5,
                       torch.tensor(-1.0, dtype=torch.float64, device=device),
                       torch.tensor(1.0, dtype=torch.float64, device=device))
    rows = torch.arange(B, device=device)
    x[rows, face] = sign
    return x.to(dtype=dtype)


@dataclass
class PINNLossConfig:
    backend: str = "auto"
    bc_weight: float = 100.0
    im_weight: float = 1.0e-6
    alpha: tuple[int, ...] = (0, 0, 0, 0, 1, 1)  # zero-based: u_{x1 x1 x1 x1 x2 x2}


def pinn_loss(
    model: nn.Sequential,
    x_int: Tensor,
    x_bc: Tensor,
    cfg: PINNLossConfig,
) -> tuple[Tensor, dict[str, float]]:
    """Residual + boundary + Tikhonov on imaginary parts of weights."""
    deriv = single_monomial_partial(model, x_int, cfg.alpha,
                                    backend=cfg.backend)  # (B, 1) complex
    f = source_f(x_int.real).unsqueeze(-1)
    res = deriv.real - f                                       # match Re(u)
    L_int = (res ** 2).mean()

    u_bc = model(x_bc)                                          # (B, 1) complex
    bc_target = u_exact(x_bc.real).unsqueeze(-1)
    L_bc = ((u_bc.real - bc_target) ** 2).mean()

    L_im = sum((p.imag ** 2).mean() for p in model.parameters())

    loss = L_int + cfg.bc_weight * L_bc + cfg.im_weight * L_im
    return loss, {"L_int": L_int.item(), "L_bc": L_bc.item(), "L_im": L_im.item()}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def l2_error(model: nn.Sequential, n_eval: int, d: int, *, device) -> float:
    x = torch.empty(n_eval, d, device=device, dtype=torch.float64).uniform_(-1.0, 1.0)
    xc = x.to(dtype=torch.complex128)
    pred = model(xc).real.squeeze(-1)
    target = u_exact(x)
    err = ((pred - target) ** 2).mean().sqrt() / (target ** 2).mean().sqrt()
    return err.item()


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------

def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.complex128
    d = 4

    torch.manual_seed(args.seed)

    model = build_complex_sinh_mlp(d=d, hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    cfg = PINNLossConfig(backend=args.backend)

    # Warm up
    for _ in range(10):
        x_int = sample_interior(args.n_int, d, device=device, dtype=dtype)
        x_bc = sample_boundary(args.n_bc, d, device=device, dtype=dtype)
        loss, _ = pinn_loss(model, x_int, x_bc, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    for step in range(args.steps):
        x_int = sample_interior(args.n_int, d, device=device, dtype=dtype)
        x_bc = sample_boundary(args.n_bc, d, device=device, dtype=dtype)
        loss, parts = pinn_loss(model, x_int, x_bc, cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if step % args.log_every == 0:
            err = l2_error(model, n_eval=4096, d=d, device=device)
            print(f"step {step:6d}  loss {loss.item():.3e}  "
                  f"L_int {parts['L_int']:.3e}  L_bc {parts['L_bc']:.3e}  "
                  f"L2 err {err:.3e}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_total = time.perf_counter() - t_start

    err_final = l2_error(model, n_eval=16384, d=d, device=device)
    s_per_step = t_total / args.steps
    peak = torch.cuda.max_memory_allocated(device) / 2 ** 30 if device.type == "cuda" else float("nan")
    print(f"\n=== {args.backend:>16s} | s/step {s_per_step:.4f} | "
          f"peak {peak:.2f} GB | final L2 err {err_final:.3e} ===")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["direct_autodiff", "polarization_jet",
                                          "waring_complex_jet", "auto"],
                   default="waring_complex_jet")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--n-int", type=int, default=512)
    p.add_argument("--n-bc", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
