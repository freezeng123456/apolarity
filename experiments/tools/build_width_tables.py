#!/usr/bin/env python3
"""Build LaTeX tables exclusively from validated jsc_v2 bundles."""

from __future__ import annotations

from pathlib import Path

import plot_width as results


OUTDIR = results.ROOT / "docs" / "paper" / "tables"
HEAD = {
    "complex_sinh": r"Complex $\sinh$",
    "siren": "SIREN",
    "fourier": "mFF-PINN",
    "mscale": "MscaleDNN-2-sin",
}


def fmt(value: float) -> str:
    return f"{value:.2e}".replace("e-0", "e-").replace("e+0", "e")


def build(key: str, items) -> Path:
    means = results.means_for(items)
    methods = [method for method in results.METHODS if method in means]
    sweeps = sorted({point[0] for method in methods for point in means[method]})
    lookup = {
        method: {sweep: mean for sweep, mean, _, _ in means[method]}
        for method in methods
    }
    first_rows = items[0][1]
    widths = {
        method: int(next(row for row in first_rows if row["variant"] == method)["actual_width"])
        for method in methods
    }
    dofs = {
        method: int(next(row for row in first_rows if row["variant"] == method)["real_dof"])
        for method in methods
    }
    symbol = "2m" if key.startswith("poly_") else "a"
    title = {
        "poly_d2": "Polyharmonic, d=2",
        "poly_d3": "Polyharmonic, d=3",
        "chirp": "Radial chirp",
        "maxwell": "Time-harmonic Maxwell",
    }[key]
    headers = [
        rf"{HEAD[method]} ($H={widths[method]}$)"
        for method in methods
    ]
    lines = [
        "% auto-generated from validated protocol_id=jsc_v2 bundles",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{title}: mean relative $L^2$ error after 1200\,s over five "
        r"seeds. Methods are matched to the Complex Sinh $H=128$ real-DOF "
        r"budget; actual integer widths are shown in the headers.}}",
        rf"\label{{tab:jsc-v2-{key}}}",
        r"\begin{tabular}{@{}l" + "r" * len(methods) + r"@{}}",
        r"\toprule",
        f"${symbol}$ & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for sweep in sweeps:
        values = {method: lookup[method].get(sweep) for method in methods}
        best = min(value for value in values.values() if value is not None)
        cells = []
        for method in methods:
            value = values[method]
            text = "--" if value is None else fmt(value)
            if value is not None and abs(value - best) < 1e-15:
                text = rf"\textbf{{{text}}}"
            cells.append(text)
        lines.append(f"{sweep:g} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\midrule",
        r"\multicolumn{" + str(len(methods) + 1) + r"}{l}{\footnotesize Real DOF: "
        + ", ".join(f"{HEAD[method]}={dofs[method]}" for method in methods)
        + r".}\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"jsc_v2_{key}.tex"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    grouped = results.grouped_bundles()
    if not grouped:
        print(f"[skip] no validated {results.PROTOCOL_ID} bundles")
        return
    for key, items in sorted(grouped.items()):
        print(f"[ok] {build(key, items)}")


if __name__ == "__main__":
    main()
