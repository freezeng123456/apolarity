#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from jax_apolarity_bench.bench_utils import (
    load_config,
    max_abs,
    measure,
    provenance,
    rel_l2,
    write_results,
)
from jax_apolarity_bench.derivatives import monomial_partial
from jax_apolarity_bench.directions import (
    exponent_pattern,
    parse_one_based_pattern,
    theoretical_waring_rank,
)
from jax_apolarity_bench.models import (
    init_mlp,
    mlp_one,
    sample_normal,
    sample_upstream_unit_ball_space_time,
)
from jax_apolarity_bench.residuals import (
    GKDV1D_TERMS,
    KDV2D_TERMS,
    POLYHARMONIC_2D,
    gkdv1d_gradient_enhanced_loss,
    kdv2d_residual,
    polyharmonic_2d_operator,
)


def dtype_from(cfg):
    return jnp.float32 if cfg["dtype"] == "float32" else jnp.float64


def analytic_coeff(d, dtype):
    return jnp.linspace(0.35, 0.35 + 0.07 * (d - 1), d, dtype=dtype)


def sinh_fun(coeff, bias=0.13):
    return lambda x: jnp.sinh(
        jnp.dot(coeff.astype(x.dtype), x) + jnp.asarray(bias, dtype=x.dtype)
    )


def sin_fun(coeff, bias=0.13):
    return lambda x: jnp.sin(
        jnp.dot(coeff.astype(x.dtype), x) + jnp.asarray(bias, dtype=x.dtype)
    )


def analytic_sinh_partial(fun, x, coeff, alpha):
    z = jax.vmap(lambda p: jnp.dot(coeff, p) + 0.13)(x)
    factor = math.prod(float(coeff[i]) for i in alpha)
    return factor * (jnp.sinh(z) if len(alpha) % 2 == 0 else jnp.cosh(z))


def flat_tree(tree):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(tree)])


def validate_fixed_grad(cfg, alpha, dtype):
    # A parameterized analytic fixture keeps the order-9 CPU validation
    # tractable while still differentiating *through* both high-order
    # derivative implementations with respect to model parameters.
    d = cfg["fixed_derivatives"]["input_dim"]
    x = sample_normal(cfg["seed"] + 18, 1, d, dtype, std=0.2)
    a0 = analytic_coeff(d, dtype)

    def loss(method, a):
        f = lambda z: jnp.sinh(
            jnp.dot(a.astype(z.dtype), z) + jnp.asarray(0.13, dtype=z.dtype)
        )
        val, _ = monomial_partial(f, x, alpha, method=method)
        return jnp.mean(jnp.real(val) ** 2)

    ga = jax.grad(lambda a: loss("nested_ad", a))(a0)
    gb = jax.grad(lambda a: loss("rank_optimal", a))(a0)
    return rel_l2(gb, ga), max_abs(gb, ga)


def fixed_rows(cfg, method, mode, dtype):
    out = []
    d = cfg["fixed_derivatives"]["input_dim"]
    net = cfg["network"]
    x = sample_normal(
        cfg["seed"] + 1,
        cfg["batch_size"],
        d,
        dtype,
        std=cfg["fixed_derivatives"]["point_std"],
    )
    timing_fixture = cfg["fixed_derivatives"].get("timing_fixture", "mlp")
    params = (
        init_mlp(cfg["seed"] + 2, d, net["width"], net["depth"], dtype)
        if timing_fixture == "mlp"
        else analytic_coeff(d, dtype)
    )
    for target in cfg["fixed_derivatives"]["targets"]:
        alpha = parse_one_based_pattern(target)
        coeff = analytic_coeff(d, dtype)
        af = sinh_fun(coeff)
        ax = x[: min(x.shape[0], cfg["validation"]["analytic_batch_size"])]
        actual, meta = monomial_partial(
            af, ax, alpha, method=method, return_complex=True
        )
        expected = analytic_sinh_partial(af, ax, coeff, alpha)
        ar = rel_l2(jnp.real(actual), expected)
        aa = max_abs(jnp.real(actual), expected)
        imag = (
            float(np.max(np.abs(np.imag(np.asarray(actual)))))
            if np.iscomplexobj(np.asarray(actual))
            else 0.0
        )
        gr, ga = validate_fixed_grad(cfg, alpha, dtype)

        def values(p):
            f = (
                (lambda z: mlp_one(p, z))
                if timing_fixture == "mlp"
                else (
                    lambda z: jnp.sinh(
                        jnp.dot(p.astype(z.dtype), z) + jnp.asarray(0.13, dtype=z.dtype)
                    )
                )
            )
            v, _ = monomial_partial(f, x, alpha, method=method)
            return v

        value_jit = jax.jit(values)
        grad_jit = jax.jit(jax.grad(lambda p: jnp.mean(jnp.real(values(p)) ** 2)))
        timing = measure(
            (lambda: value_jit(params))
            if mode == "value"
            else (lambda: grad_jit(params)),
            cfg["warmups"],
            cfg["repeats"],
        )
        out.append(
            {
                "status": "ok",
                "backend": "jax",
                "benchmark": "fixed_mixed_partial",
                "case": f"u_{target}",
                "method": method,
                "mode": mode,
                "dtype_real": cfg["dtype"],
                "dtype_direction": meta.direction_dtype,
                "requires_complex": meta.requires_complex,
                "batch_size": cfg["batch_size"],
                "network_depth_linear_layers": net["depth"],
                "network_hidden_layers": net["depth"] - 1,
                "network_width": net["width"],
                "timing_fixture": timing_fixture,
                "multi_index_zero_based": list(alpha),
                "order": len(alpha),
                "exponent_pattern": list(exponent_pattern(alpha)),
                "theoretical_direction_count": theoretical_waring_rank(alpha),
                "evaluated_direction_count": meta.direction_count,
                "analytic_rel_l2_error": ar,
                "analytic_max_abs_error": aa,
                "imaginary_leak_max_abs": imag,
                "parameter_gradient_rel_l2_vs_nested": gr,
                "parameter_gradient_max_abs_vs_nested": ga,
                **timing,
            }
        )
    return out


def validate_pde_grad(cfg, bench, dtype):
    # Use a parameterized entire analytic fixture.  This checks differentiation
    # through the complete high-order residual with respect to parameters while
    # keeping the CPU smoke independent of expensive high-order MLP transforms.
    if bench == "kdv2d":
        dim = 3
        x = jnp.asarray([[0.12, -0.18, 0.27]], dtype=dtype)
        a0 = analytic_coeff(dim, dtype)

        def loss(method, a):
            f = lambda z: jnp.sinh(
                jnp.dot(a.astype(z.dtype), z) + jnp.asarray(0.13, dtype=z.dtype)
            )
            return jnp.mean(kdv2d_residual(f, x, method=method).value ** 2)
    elif bench == "gkdv1d":
        dim = 2
        x = jnp.asarray([[0.12, 0.27]], dtype=dtype)
        a0 = analytic_coeff(dim, dtype)

        def loss(method, a):
            f = lambda z: jnp.sinh(
                jnp.dot(a.astype(z.dtype), z) + jnp.asarray(0.13, dtype=z.dtype)
            )
            return jnp.real(gkdv1d_gradient_enhanced_loss(f, x, method=method).value)
    else:
        order = 4 if bench == "poly4" else 6
        dim = 2
        x = jnp.asarray([[0.12, -0.18]], dtype=dtype)
        a0 = analytic_coeff(dim, dtype)

        def loss(method, a):
            f = lambda z: jnp.sin(
                jnp.dot(a.astype(z.dtype), z) + jnp.asarray(0.13, dtype=z.dtype)
            )
            v = polyharmonic_2d_operator(f, x, order=order, method=method).value
            return jnp.mean(jnp.real(v) ** 2)

    ga = jax.grad(lambda a: loss("nested_ad", a))(a0)
    gb = jax.grad(lambda a: loss("rank_optimal", a))(a0)
    return rel_l2(gb, ga), max_abs(gb, ga)


def kdv_analytic(method, dtype):
    c = analytic_coeff(3, dtype)
    f = sinh_fun(c)
    x = jnp.asarray([[0.1, -0.2, 0.3], [0.25, 0.15, 0.4]], dtype=dtype)
    r = kdv2d_residual(f, x, method=method)
    ex = {n: analytic_sinh_partial(f, x, c, a) for n, a in KDV2D_TERMS.items()}
    e = (
        ex["u_ty"]
        + ex["u_xxxy"]
        + 3 * (ex["u_xy"] * ex["u_x"] + ex["u_y"] * ex["u_xx"])
        - ex["u_xx"]
        + 2 * ex["u_yy"]
    )
    return rel_l2(r.value, e), max_abs(r.value, e)


def gkdv_analytic(method, dtype):
    c = analytic_coeff(2, dtype)
    f = sinh_fun(c)
    x = jnp.asarray([[0.1, 0.3], [-0.25, 0.4]], dtype=dtype)
    r = gkdv1d_gradient_enhanced_loss(f, x, method=method)
    d = {n: analytic_sinh_partial(f, x, c, a) for n, a in GKDV1D_TERMS.items()}
    u = jax.vmap(f)(x)
    ff = d["u_t"] + u * d["u_x"] + 0.0025 * d["u_xxx"]
    fx = d["u_tx"] + d["u_x"] ** 2 + u * d["u_xx"] + 0.0025 * d["u_xxxx"]
    ft = d["u_tt"] + d["u_t"] * d["u_x"] + u * d["u_tx"] + 0.0025 * d["u_txxx"]
    loss = jnp.mean(ff**2) + 1e-3 * (jnp.mean(fx**2) + jnp.mean(ft**2))
    ce = {
        "f": rel_l2(r.components["f"], ff),
        "f_x": rel_l2(r.components["f_x"], fx),
        "f_t": rel_l2(r.components["f_t"], ft),
    }
    return (
        rel_l2(jnp.atleast_1d(r.value), jnp.atleast_1d(loss)),
        max_abs(jnp.atleast_1d(r.value), jnp.atleast_1d(loss)),
        ce,
    )


def poly_analytic(method, order, dtype):
    c = analytic_coeff(2, dtype)
    f = sin_fun(c)
    x = jnp.asarray([[0.1, -0.2], [0.25, 0.15]], dtype=dtype)
    r = polyharmonic_2d_operator(f, x, order=order, method=method)
    u = jax.vmap(f)(x)
    e = (-jnp.sum(c**2)) ** (order // 2) * u
    return rel_l2(r.value, e), max_abs(r.value, e)


def residual_row(cfg, bench, method, mode, dtype):
    net = cfg["network"]
    if bench == "kdv2d":
        pts = sample_upstream_unit_ball_space_time(
            cfg["seed"] + 41,
            cfg["batch_size"],
            2,
            dtype,
            cfg["pde"]["max_radius"],
            cfg["pde"]["T"],
        )
        dim = 3
        ar, aa = kdv_analytic(method, dtype)
        ce = None
        source = "supplied STDE upstream KdV2d_res; source sets case=4"
        max_order = 4
        timing_fixture = cfg["pde"].get("timing_fixture", "mlp")
        params = (
            init_mlp(cfg["seed"] + 42, dim, net["width"], net["depth"], dtype)
            if timing_fixture == "mlp"
            else analytic_coeff(dim, dtype)
        )

        def make_fun(p):
            return (
                (lambda z: mlp_one(p, z))
                if timing_fixture == "mlp"
                else (
                    lambda z: jnp.sinh(
                        jnp.dot(p.astype(z.dtype), z) + jnp.asarray(0.13, dtype=z.dtype)
                    )
                )
            )

        evalp = lambda p: kdv2d_residual(make_fun(p), pts, method=method)
        value = lambda p: evalp(p).value
        loss = lambda p: jnp.mean(jnp.real(value(p)) ** 2)
    elif bench == "gkdv1d":
        pts = sample_upstream_unit_ball_space_time(
            cfg["seed"] + 41,
            cfg["batch_size"],
            1,
            dtype,
            cfg["pde"]["max_radius"],
            cfg["pde"]["T"],
        )
        dim = 2
        ar, aa, ce = gkdv_analytic(method, dtype)
        source = "supplied STDE upstream highord1d_res case=4, eq=2 full gradient-enhanced objective"
        max_order = 4
        timing_fixture = cfg["pde"].get("timing_fixture", "mlp")
        params = (
            init_mlp(cfg["seed"] + 42, dim, net["width"], net["depth"], dtype)
            if timing_fixture == "mlp"
            else analytic_coeff(dim, dtype)
        )

        def make_fun(p):
            return (
                (lambda z: mlp_one(p, z))
                if timing_fixture == "mlp"
                else (
                    lambda z: jnp.sinh(
                        jnp.dot(p.astype(z.dtype), z) + jnp.asarray(0.13, dtype=z.dtype)
                    )
                )
            )

        evalp = lambda p: gkdv1d_gradient_enhanced_loss(
            make_fun(p),
            pts,
            method=method,
            lam1=cfg["gkdv1d"]["lambda1"],
            dispersion=cfg["gkdv1d"]["dispersion"],
        )
        value = lambda p: evalp(p).value
        loss = lambda p: jnp.real(value(p))
    else:
        order = 4 if bench == "poly4" else 6
        pts = sample_normal(
            cfg["seed"] + 41,
            cfg["batch_size"],
            2,
            dtype,
            std=cfg["polyharmonic"]["point_std"],
        )
        dim = 2
        ar, aa = poly_analytic(method, order, dtype)
        ce = None
        source = f"Delta^{order // 2} in 2D expanded with binomial coefficients"
        max_order = order
        timing_fixture = cfg["polyharmonic"].get("timing_fixture", "mlp")
        params = (
            init_mlp(cfg["seed"] + 42, dim, net["width"], net["depth"], dtype)
            if timing_fixture == "mlp"
            else analytic_coeff(dim, dtype)
        )

        def make_fun(p):
            return (
                (lambda z: mlp_one(p, z))
                if timing_fixture == "mlp"
                else (
                    lambda z: jnp.sin(
                        jnp.dot(p.astype(z.dtype), z) + jnp.asarray(0.13, dtype=z.dtype)
                    )
                )
            )

        evalp = lambda p: polyharmonic_2d_operator(
            make_fun(p), pts, order=order, method=method
        )
        value = lambda p: evalp(p).value
        loss = lambda p: jnp.mean(jnp.real(value(p)) ** 2)
    sample_eval = evalp(params)
    gr, ga = validate_pde_grad(cfg, bench, dtype)
    vjit = jax.jit(value)
    gjit = jax.jit(jax.grad(loss))
    timing = measure(
        (lambda: vjit(params)) if mode == "value" else (lambda: gjit(params)),
        cfg["warmups"],
        cfg["repeats"],
    )
    row = {
        "status": "ok",
        "backend": "jax",
        "benchmark": bench,
        "case": bench,
        "method": method,
        "mode": mode,
        "dtype_real": cfg["dtype"],
        "batch_size": cfg["batch_size"],
        "network_depth_linear_layers": net["depth"],
        "network_hidden_layers": net["depth"] - 1,
        "network_width": net["width"],
        "timing_fixture": timing_fixture,
        "max_derivative_order": max_order,
        "evaluated_direction_count": sample_eval.direction_evaluations,
        "analytic_rel_l2_error": ar,
        "analytic_max_abs_error": aa,
        "parameter_gradient_rel_l2_vs_nested": gr,
        "parameter_gradient_max_abs_vs_nested": ga,
        "definition_source": source,
        **timing,
    }
    if ce is not None:
        row["analytic_component_rel_l2"] = ce
        row["gkdv_lambda1"] = cfg["gkdv1d"]["lambda1"]
        row["gkdv_dispersion"] = cfg["gkdv1d"]["dispersion"]
    if bench.startswith("poly"):
        order = 4 if bench == "poly4" else 6
        row["operator_terms"] = [[c, list(a)] for c, a in POLYHARMONIC_2D[order]]
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "configs/main.yaml"))
    ap.add_argument(
        "--benchmark",
        choices=["all", "fixed", "kdv2d", "gkdv1d", "poly4", "poly6"],
        default="all",
    )
    ap.add_argument(
        "--method", choices=["both", "nested_ad", "rank_optimal"], default="both"
    )
    ap.add_argument(
        "--mode", choices=["both", "value", "parameter_grad"], default="both"
    )
    ap.add_argument("--out", default=str(HERE / "results/jax_benchmark"))
    ap.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail instead of falling back when no JAX GPU is visible",
    )
    args = ap.parse_args()
    cp = Path(args.config).resolve()
    cfg = load_config(cp)
    dtype = dtype_from(cfg)
    methods = ["nested_ad", "rank_optimal"] if args.method == "both" else [args.method]
    modes = ["value", "parameter_grad"] if args.mode == "both" else [args.mode]
    benches = (
        ["fixed", "kdv2d", "gkdv1d", "poly4", "poly6"]
        if args.benchmark == "all"
        else [args.benchmark]
    )
    rows = []
    devices = jax.devices()
    actual = jax.default_backend()
    gpu_visible = any(getattr(d, "platform", "") == "gpu" for d in devices)
    requested = str(cfg.get("device", "auto"))
    fallback_reason = None
    if args.require_gpu and not gpu_visible:
        raise RuntimeError("--require-gpu was set but no JAX GPU device is visible")
    if requested in ("cuda", "gpu") and not gpu_visible:
        raise RuntimeError(
            f"config requested {requested!r}, but no JAX GPU device is visible"
        )
    if requested == "cpu" and actual != "cpu":
        raise RuntimeError(
            f"config requested CPU, but JAX selected backend {actual!r}; set JAX_PLATFORMS=cpu"
        )
    if requested == "auto" and not gpu_visible:
        fallback_reason = "No JAX GPU device is visible; auto selected the CPU backend. This is recorded as an explicit fallback, not a GPU result."
    if fallback_reason:
        print(f"[jax-benchmark] {fallback_reason}", file=sys.stderr, flush=True)
    for b in benches:
        for m in methods:
            for mode in modes:
                try:
                    rows.extend(
                        fixed_rows(cfg, m, mode, dtype)
                    ) if b == "fixed" else rows.append(
                        residual_row(cfg, b, m, mode, dtype)
                    )
                except Exception as e:
                    rows.append(
                        {
                            "status": "error",
                            "backend": "jax",
                            "benchmark": b,
                            "method": m,
                            "mode": mode,
                            "error": repr(e),
                        }
                    )
    for r in rows:
        r.update(
            {
                "seed": cfg["seed"],
                "device": str(devices[0]),
                "device_requested": requested,
                "backend_actual": actual,
                "gpu_visible": gpu_visible,
                "platform_fallback_reason": fallback_reason,
                "protocol_kind": cfg.get("protocol_kind", "primary"),
            }
        )
    jp, cs = write_results(rows, args.out)
    pp = jp.with_name(jp.stem + "_provenance.json")
    pp.write_text(json.dumps(provenance(HERE, cp), indent=2), encoding="utf-8")
    fp = jp.with_name(jp.stem + "_frozen_config.yaml")
    fp.write_text(cp.read_text(encoding="utf-8"), encoding="utf-8")
    print(
        json.dumps(
            {
                "results_json": str(jp),
                "results_csv": str(cs),
                "provenance": str(pp),
                "rows": rows,
            },
            indent=2,
            default=str,
        )
    )
    raise SystemExit(2 if any(r.get("status") != "ok" for r in rows) else 0)


if __name__ == "__main__":
    if os.environ.get("JAX_BENCH_FORCE_EXIT", "0") == "1":
        try:
            main()
        except SystemExit as exc:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(int(exc.code or 0))
    else:
        main()
