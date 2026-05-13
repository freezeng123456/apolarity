"""Forward-mode Taylor jet propagation for Linear+Tanh MLPs (STDE-style).

Mathematical setting
====================
Let g(tau) = u(x + tau * Z), where u: R^d -> R is an MLP composed of
nn.Linear and nn.Tanh layers in nn.Sequential, and Z in R^d is a fixed
"direction" vector.  We want the directional Taylor coefficients

    T_k(Z) := g^{(k)}(0) / k!,    k = 1, ..., p.

A standard reverse-mode implementation does p nested ``torch.autograd.grad``
calls, building a graph of depth O(p * L), peak memory
O((p+1) * L * B * K * H) with a non-trivial PyTorch allocator constant.

This module replaces that with a forward-mode "Taylor jet" pass: each
intermediate activation in the MLP is augmented from a single tensor
``y_0`` to a tuple ``(y_0, y_1, ..., y_p)`` representing
``y_k = (1/k!) d^k y / dtau^k |_{tau=0}``.  The MLP is then evaluated once
with each primitive (Linear / Tanh / +) replaced by its jet rule.

Key jet rules (probabilist convention y_k = (1/k!) d^k y / dtau^k):

  Linear   y = W x + b:
    y_0 = W x_0 + b
    y_k = W x_k                      for k >= 1

  Affine offset (input to first jet):
    x(tau) = x + tau * Z  ->  jet = (x, Z, 0, 0, ..., 0)

  Pointwise tanh   y = tanh(x):
    z := 1 - y^2 = tanh'(x), itself a jet
    y_0 = tanh(x_0)
    z_0 = 1 - y_0 * y_0
    For k = 1, ..., p:
      y_k = (1/k) sum_{j=0}^{k-1} (k - j) * z_j * x_{k-j}
      z_k = -sum_{j=0}^{k} y_j * y_{k-j}

  Add (broadcast bias-like):  trivial elementwise add per order.

The recurrence for tanh comes from the standard Faa-di-Bruno expansion using
the ODE form  y' = z * x'  (chain rule outer factor) together with the
identity z = 1 - y^2.

Numerical correctness: verified against torch nested autograd to machine
precision (max rel err < 5e-16 in fp64) for p up to 6 -- see
tests/test_taylor_jet.py.

Memory and gradient backflow
============================
Each jet term ``y_k`` is a regular torch.Tensor with shape identical to the
ordinary forward activation at that layer.  Storing p+1 such tensors per
layer makes the per-step activation footprint ``(p+1) * L * B * K * H``
tensor elements -- same asymptotic order as reverse-mode nested grads, but
*without* the reverse-engine constant blow-up (no graph chaining).

Because every jet primitive is built from standard differentiable PyTorch
ops (matmul, add, mul, tanh), gradients with respect to model parameters
flow naturally through the entire jet propagation when the caller invokes
``loss.backward()``.  The output T_p tensor remains a leaf of the autograd
graph leading back to the model parameters.

torch.compile integration (opt-in)
==================================
Setting the env var ``TAYLOR_JET_COMPILE=1`` (or calling
``set_compile_mode(True)``) wraps ``jet_forward_sequential`` with
``torch.compile(mode="reduce-overhead")`` on first use.  The wrapper is
cached per ``(net_id, p, dtype)`` so successive calls with identical net
instance and jet order hit the fast path.

Risk profile (project-internal evaluation, 2026-05-11):
- Shape-stable: ``B*K*d`` and ``p`` are fixed across training steps (KdV
  and CH6 both use constant ``n_residual``, ``K``, and operator order).
  No recompile thrash.
- ``model(...)`` calls outside the jet path (forward eval, BC, IC,
  validation) are NOT affected -- compile only wraps jet_forward_sequential.
- ``reverse`` and ``fd`` backends are NOT compiled (they contain
  ``torch.autograd.grad(create_graph=True)`` which Dynamo cannot trace).
- ``loss.backward()`` through a compiled jet still flows to model parameters
  (verified by tests/test_taylor_jet.py::test_jet_backward_to_theta_matches_reverse
   when run with TAYLOR_JET_COMPILE=1).
- First compile on each net+p incurs ~1-30 s overhead (Inductor fusion +
  kernel selection); amortized over a 20-min training run.
"""
from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ============================================================================
# Compile mode (opt-in)
# ============================================================================

_COMPILE_ENABLED: bool = os.environ.get("TAYLOR_JET_COMPILE", "0") == "1"
_COMPILE_CACHE: Dict[Tuple[int, int, torch.dtype], Callable] = {}


def set_compile_mode(enabled: bool) -> None:
    """Enable/disable torch.compile for jet_forward_sequential at runtime.

    When enabled, the first call for each (model.id, p, dtype) triple
    triggers compilation (~1-30 s); subsequent calls hit the fast path.
    Disabling clears the per-net compile cache.
    """
    global _COMPILE_ENABLED
    _COMPILE_ENABLED = enabled
    if not enabled:
        _COMPILE_CACHE.clear()


def is_compile_enabled() -> bool:
    return _COMPILE_ENABLED


# ============================================================================
# Jet container
# ============================================================================

class TaylorJet:
    """A length-(p+1) sequence of tensors, where ``terms[k]`` represents
    the k-th Taylor coefficient ``(1/k!) d^k / dtau^k`` of the jet's value
    at ``tau = 0``.

    All terms share the same shape (broadcast-compatible).  We do NOT enforce
    any specific dtype/device; rules below propagate the input convention.
    """

    __slots__ = ("terms",)

    def __init__(self, terms: List[Tensor]) -> None:
        if len(terms) == 0:
            raise ValueError("TaylorJet must have at least one term (order 0)")
        self.terms = terms

    @property
    def order(self) -> int:
        """Highest order p stored (so length is p+1)."""
        return len(self.terms) - 1

    @classmethod
    def from_input(cls, x: Tensor, Z: Tensor, p: int) -> "TaylorJet":
        """Build the input jet for ``g(tau) = u(x + tau * Z)``:
            terms[0] = x,  terms[1] = Z,  terms[k>=2] = 0.
        Both ``x`` and ``Z`` have shape (N, d) with N = B*K samples.
        """
        if p < 1:
            raise ValueError(f"p must be >= 1; got {p}")
        zero = torch.zeros_like(x)
        return cls([x, Z] + [zero for _ in range(p - 1)])

    @classmethod
    def from_constant(cls, x: Tensor, p: int) -> "TaylorJet":
        """Build a jet with constant value ``x`` (used only in tests)."""
        if p < 0:
            raise ValueError(f"p must be >= 0; got {p}")
        zero = torch.zeros_like(x)
        return cls([x] + [zero for _ in range(p)])


# ============================================================================
# Jet rules for primitives appearing in our MLPs
# ============================================================================

def jet_linear(jet: TaylorJet, weight: Tensor, bias: Tensor | None) -> TaylorJet:
    """Apply  y = x @ W^T + b  to a jet.  Bias adds only to the order-0 term."""
    out: List[Tensor] = []
    for k, xk in enumerate(jet.terms):
        yk = xk @ weight.T
        if k == 0 and bias is not None:
            yk = yk + bias
        out.append(yk)
    return TaylorJet(out)


def jet_tanh(jet: TaylorJet) -> TaylorJet:
    """Pointwise tanh on a jet.  Implements the recurrence

        y_0 = tanh(x_0)
        z_0 = 1 - y_0 * y_0
        y_k = (1/k) sum_{j=0}^{k-1} (k - j) * z_j * x_{k-j}        for k >= 1
        z_k = -sum_{j=0}^{k} y_j * y_{k-j}

    where ``z(tau) = 1 - y(tau)^2 = tanh'(x(tau))``.  Cost is O(p^2) ops
    over hidden-dim tensors per call.
    """
    p = jet.order
    x = jet.terms

    y: List[Tensor] = [torch.tanh(x[0])]
    z: List[Tensor] = [1.0 - y[0] * y[0]]

    for k in range(1, p + 1):
        # y_k = (1/k) * sum_{j=0..k-1} (k - j) * z_j * x_{k-j}
        acc = (k - 0) * z[0] * x[k]
        for j in range(1, k):
            acc = acc + (k - j) * z[j] * x[k - j]
        y_k = acc / float(k)
        y.append(y_k)
        # z_k = -sum_{j=0..k} y_j * y_{k-j}
        zk = -y[0] * y[k]
        for j in range(1, k + 1):
            zk = zk - y[j] * y[k - j]
        z.append(zk)

    return TaylorJet(y)


# ============================================================================
# MLP-level driver
# ============================================================================

def _is_supported_module(m: nn.Module) -> bool:
    return isinstance(m, (nn.Linear, nn.Tanh))


# Tensor-only inner driver: takes a list of tensors (the input jet terms)
# and returns the output jet terms.  This signature is friendly to
# torch.compile (no custom containers, no Python-level branching beyond a
# fixed-length for-loop over Sequential layers / a static integer ``p``).
def _jet_forward_tensors(
    layers: List[nn.Module], terms: List[Tensor],
) -> List[Tensor]:
    """Run jet ``terms`` (list of length p+1) through ``layers`` (a flat
    list of nn.Linear / nn.Tanh modules in order)."""
    out = list(terms)
    p = len(out) - 1
    for layer in layers:
        if isinstance(layer, nn.Linear):
            W = layer.weight
            b = layer.bias
            new_out: List[Tensor] = []
            for k, xk in enumerate(out):
                yk = xk @ W.T
                if k == 0 and b is not None:
                    yk = yk + b
                new_out.append(yk)
            out = new_out
        elif isinstance(layer, nn.Tanh):
            x = out
            y: List[Tensor] = [torch.tanh(x[0])]
            z: List[Tensor] = [1.0 - y[0] * y[0]]
            for k in range(1, p + 1):
                acc = (k - 0) * z[0] * x[k]
                for j in range(1, k):
                    acc = acc + (k - j) * z[j] * x[k - j]
                y_k = acc / float(k)
                y.append(y_k)
                zk = -y[0] * y[k]
                for j in range(1, k + 1):
                    zk = zk - y[j] * y[k - j]
                z.append(zk)
            out = y
        else:
            raise NotImplementedError(
                f"taylor_jet does not support layer type {type(layer).__name__}"
            )
    return out


def jet_forward_sequential(net: nn.Sequential, jet: TaylorJet) -> TaylorJet:
    """Run ``jet`` through the layers of ``nn.Sequential`` net.  Only
    supports Linear + Tanh stacks (the architecture used by KdV / CH6 /
    HardBC models in this project).  Raises if any other layer is found.

    When ``TAYLOR_JET_COMPILE=1`` is set in the environment (or
    ``set_compile_mode(True)`` was called), the per-step compute is wrapped
    with ``torch.compile(mode="reduce-overhead")``.  The compiled callable
    is cached per (net id, jet order, dtype).
    """
    layers = list(net)
    if _COMPILE_ENABLED:
        cache_key = (id(net), jet.order, jet.terms[0].dtype)
        compiled = _COMPILE_CACHE.get(cache_key)
        if compiled is None:
            # Build a closure that fixes the layers list (so the compiled
            # graph specialises to this net's parameter tensors).
            def _runner(terms_list):
                return _jet_forward_tensors(layers, terms_list)
            compiled = torch.compile(_runner, mode="reduce-overhead",
                                     dynamic=False, fullgraph=False)
            _COMPILE_CACHE[cache_key] = compiled
        out_terms = compiled(jet.terms)
    else:
        out_terms = _jet_forward_tensors(layers, jet.terms)
    return TaylorJet(out_terms)


def _resolve_sequential_net(model: nn.Module) -> nn.Sequential:
    """Find the underlying nn.Sequential.  Project models expose this as
    ``model.net``; for plain nn.Sequential we accept it directly."""
    if isinstance(model, nn.Sequential):
        return model
    if hasattr(model, "net") and isinstance(model.net, nn.Sequential):
        return model.net
    raise NotImplementedError(
        "taylor_jet requires the model to be (or wrap) an nn.Sequential of "
        "Linear/Tanh layers; got " + type(model).__name__
    )


def tp_directional_via_jet(
    model: nn.Module, xyt_input: Tensor, Z: Tensor, p: int,
) -> Tensor:
    """Compute T_p(Z) = g^{(p)}(0) / p!  via Taylor jet propagation.

    Same semantics as ``kdv_hd.estimators._tp_via_reverse``: shape
    ``(B, K, 1)``.

    The MLP is evaluated *once* on a flat ``(B*K, d)`` batch, with each
    layer replaced by its jet rule.  Memory: ``(p+1) * L * B * K * H``
    activation tensors; no autograd graph chaining.

    Note on convention: by construction, ``out_jet.terms[k]`` already
    equals ``(1/k!) * d^k g / dtau^k |_{tau=0}`` (probabilist convention),
    so the value of T_p is read off directly from ``terms[p]`` without
    any further division by ``p!``.

    Caveat: this routine ASSUMES ``model`` (or ``model.net``) is a single
    ``nn.Sequential`` of ``nn.Linear`` and ``nn.Tanh`` layers.  Networks
    wrapped in HardBC factors (e.g. ``CH6HardBCModel``) need to be unwrapped
    by the caller first; see
    ``ch6_ball.estimators._SpatialNWrapper`` etc. for the existing pattern.
    """
    if p < 1:
        raise ValueError(f"p must be >= 1; got {p}")
    net = _resolve_sequential_net(model)

    B, K, d = Z.shape
    xyt_exp = xyt_input.unsqueeze(1).expand(B, K, d).reshape(B * K, d)
    Z_flat = Z.reshape(B * K, d)

    in_jet = TaylorJet.from_input(xyt_exp, Z_flat, p)
    out_jet = jet_forward_sequential(net, in_jet)

    # out_jet.terms[p] is already T_p = g^{(p)}(0) / p!  (probabilist convention)
    return out_jet.terms[p].reshape(B, K, 1)


def tp_directional_all_via_jet(
    model: nn.Module, xyt_input: Tensor, Z: Tensor, p_max: int,
) -> List[Tensor]:
    """Compute T_1, T_2, ..., T_{p_max} via a SINGLE jet pass of order
    ``p_max``.  Returns list of ``(B, K, 1)`` tensors.

    This is strictly cheaper than calling ``tp_directional_via_jet`` p_max
    times because the same forward jet pass already contains every order.
    """
    if p_max < 1:
        raise ValueError(f"p_max must be >= 1; got {p_max}")
    net = _resolve_sequential_net(model)

    B, K, d = Z.shape
    xyt_exp = xyt_input.unsqueeze(1).expand(B, K, d).reshape(B * K, d)
    Z_flat = Z.reshape(B * K, d)

    in_jet = TaylorJet.from_input(xyt_exp, Z_flat, p_max)
    out_jet = jet_forward_sequential(net, in_jet)

    out_list: List[Tensor] = []
    for k in range(1, p_max + 1):
        # terms[k] is already T_k under the probabilist convention.
        out_list.append(out_jet.terms[k].reshape(B, K, 1))
    return out_list


__all__ = [
    "TaylorJet",
    "jet_linear",
    "jet_tanh",
    "jet_forward_sequential",
    "tp_directional_via_jet",
    "tp_directional_all_via_jet",
    "set_compile_mode",
    "is_compile_enabled",
]
