#!/usr/bin/env python3
"""Build LaTeX tables exclusively from validated jsc_v2 bundles."""

from __future__ import annotations

from pathlib import Path
from statistics import fmean, stdev

import plot_width as results


OUTDIR = results.ROOT / "docs" / "paper" / "tables"
HEAD = {
    "complex_sinh": r"Complex $\sinh$",
    "siren": "SIREN",
    "fourier": "mFF-PINN",
    "mscale": "MscaleDNN-2-sin",
}


def fmt(mean: float, std: float, *, bold: bool = False) -> str:
    """Format a mean and sample standard deviation as compact LaTeX."""
    if mean == 0.0:
        body = rf"0.00\pm{std:.2f}"
        return rf"$\mathbf{{{body}}}$" if bold else rf"${body}$"
    exponent = int(f"{mean:.2e}".split("e")[1])
    scale = 10.0**exponent
    body = rf"{mean / scale:.2f}\pm{std / scale:.2f}"
    if bold:
        body = rf"\mathbf{{{body}}}"
    if exponent == 0:
        return rf"${body}$"
    return rf"${body}\times10^{{{exponent}}}$"


def build(key: str, items) -> Path:
    values: dict[str, dict[float, list[float]]] = {}
    for _, rows, _ in items:
        for row in rows:
            method = str(row["variant"])
            sweep = float(row["sweep"])
            values.setdefault(method, {}).setdefault(sweep, []).append(
                float(row["L2_err"])
            )
    methods = [method for method in results.METHODS if method in values]
    sweeps = sorted({sweep for method in methods for sweep in values[method]})
    lookup = {
        method: {
            sweep: (fmean(samples), stdev(samples))
            for sweep, samples in by_sweep.items()
        }
        for method, by_sweep in values.items()
    }
    first_rows = items[0][1]
    dofs = {
        method: int(next(row for row in first_rows if row["variant"] == method)["real_dof"])
        for method in methods
    }
    symbol = "2m" if key.startswith("poly_") else "a"
    title = {
        "poly_d2": r"Polyharmonic benchmark, $d=2$",
        "poly_d3": r"Polyharmonic benchmark, $d=3$",
        "chirp": "Radial-chirp benchmark",
        "maxwell": "Time-harmonic Maxwell benchmark",
    }[key]
    headers = [HEAD[method] for method in methods]
    label_key = key
    lines = [
        "% auto-generated from validated protocol_id=jsc_v2 bundles",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{title}: mean $\pm$ sample standard deviation of the "
        r"held-out relative $L^2$ error over five seeds after 1200\,s. "
        r"The lowest mean in each row is set in bold.}",
        rf"\label{{tab_jsc_v2_{label_key}}}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}l" + "c" * len(methods) + r"@{}}",
        r"\toprule",
        f"${symbol}$ & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for sweep in sweeps:
        row_values = {method: lookup[method].get(sweep) for method in methods}
        best = min(value[0] for value in row_values.values() if value is not None)
        cells = []
        for method in methods:
            value = row_values[method]
            text = "--" if value is None else fmt(
                value[0], value[1], bold=abs(value[0] - best) < 1e-15
            )
            cells.append(text)
        lines.append(f"{sweep:g} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\par\smallskip",
            r"\parbox{0.98\linewidth}{\scriptsize All methods use four hidden "
            r"layers of width $H=128$. Trainable real degrees of freedom: "
            + "; ".join(f"{HEAD[method]}={dofs[method]:,}" for method in methods)
            + r".}",
            r"\end{table}",
            "",
        ]
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"jsc_v2_{key}.tex"
    path.write_text("\n".join(lines))
    return path


def build_throughput() -> Path:
    """Build a representative within-task optimizer-throughput table."""
    selected = {
        "poly_d2_o4": r"Polyharmonic, $d=2$, $2m=4$",
        "chirp_a2": r"Radial chirp, $a=2$",
        "maxwell_a4": r"Maxwell, $a=4$",
    }
    header = "setting"
    bundles = {
        task_dir.name: rows
        for task_dir, rows, _ in results.validated_bundles()
        if task_dir.name in selected
    }
    missing = selected.keys() - bundles.keys()
    if missing:
        raise ValueError(f"missing representative validated tasks: {sorted(missing)}")

    lines = [
        "% auto-generated from validated protocol_id=jsc_v2 bundles",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Representative optimizer throughput: mean $\pm$ sample "
        r"standard deviation of milliseconds per step over five seeds. "
        r"Methods should be compared within a row; the stored software "
        r"versions differ between settings.}",
        r"\label{tab_jsc_v2_throughput}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{@{}lcccc@{}}",
        r"\toprule",
        f"{header} & "
        + " & ".join(HEAD[method] for method in results.METHODS)
        + r" \\",
        r"\midrule",
    ]
    for task_id, label in selected.items():
        by_method: dict[str, list[float]] = {}
        for row in bundles[task_id]:
            by_method.setdefault(str(row["variant"]), []).append(
                float(row["ms_per_step"])
            )
        cells = []
        for method in results.METHODS:
            samples = by_method[method]
            cells.append(rf"${fmean(samples):.1f}\pm{stdev(samples):.1f}$")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
        ]
    )
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "jsc_v2_throughput.tex"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    grouped = results.grouped_bundles()
    if not grouped:
        print(f"[skip] no validated {results.PROTOCOL_ID} bundles")
        return
    for key, items in sorted(grouped.items()):
        print(f"[ok] {build(key, items)}")
    print(f"[ok] {build_throughput()}")


if __name__ == "__main__":
    main()
