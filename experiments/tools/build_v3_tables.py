#!/usr/bin/env python3
"""Build the derivative-backend comparison tables from validated jsc_v3 bundles.

The jsc_v3 protocol holds the surrogate, the loss, the budget, and the seeds
fixed and changes only the derivative backend, so the two lines it contains are
directly comparable: ``complex_sinh`` evaluates every monomial derivative with
the rank-optimal Taylor jet, and ``complex_sinh_autodiff`` evaluates the same
derivative by nested reverse-mode automatic differentiation.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import fmean, stdev

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "experiments" / "results" / "jsc_v3"
OUTDIR = ROOT / "docs" / "paper" / "tables"
PROTOCOL_ID = "jsc_v3"
JET = "complex_sinh"
AUTODIFF = "complex_sinh_autodiff"
#: Reporting order: the polyharmonic family sweeps the derivative order, the
#: other two families sit at order two and sweep a frequency parameter.
TASKS = (
    ("poly_d2_o2", r"Polyharmonic, $2m=2$"),
    ("poly_d2_o4", r"Polyharmonic, $2m=4$"),
    ("poly_d2_o6", r"Polyharmonic, $2m=6$"),
    ("chirp_a1", r"Radial chirp, $a=1$"),
    ("chirp_a2", r"Radial chirp, $a=2$"),
    ("chirp_a3", r"Radial chirp, $a=3$"),
    ("maxwell_a2", r"Maxwell, $a=2$"),
    ("maxwell_a4", r"Maxwell, $a=4$"),
    ("maxwell_a6", r"Maxwell, $a=6$"),
)


def load(task_id: str) -> dict[str, list[dict]]:
    """Return the validated rows of one task, grouped by derivative backend."""
    task_dir = RESULT_ROOT / task_id
    if not (task_dir / "VALIDATED").exists():
        raise ValueError(f"{task_id} carries no VALIDATED marker")
    rows = json.loads((task_dir / f"{task_id}.json").read_text())
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["protocol_id"] != PROTOCOL_ID:
            raise ValueError(f"{task_id} contains a non-{PROTOCOL_ID} row")
        grouped.setdefault(str(row["variant"]), []).append(row)
    if set(grouped) != {JET, AUTODIFF}:
        raise ValueError(f"{task_id} does not contain exactly the two formal lines")
    backends = {row["backend"] for row in grouped[JET]}, {row["backend"] for row in grouped[AUTODIFF]}
    if backends != ({"jet"}, {"autograd"}):
        raise ValueError(f"{task_id} backend labels are not jet/autograd: {backends}")
    return grouped


def mean_of(rows: list[dict], field: str) -> float:
    return fmean(float(row[field]) for row in rows)


def sci(value: float) -> str:
    """Format one positive quantity as a compact LaTeX power of ten."""
    exponent = int(f"{value:.2e}".split("e")[1])
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.2f}{{\times}}10^{{{exponent}}}"


def error_cell(rows: list[dict]) -> str:
    values = [float(row["L2_err"]) for row in rows]
    mean, deviation = fmean(values), stdev(values)
    exponent = int(f"{mean:.2e}".split("e")[1])
    scale = 10.0**exponent
    return rf"${mean / scale:.2f}\pm{deviation / scale:.2f}{{\times}}10^{{{exponent}}}$"


def build_cost() -> Path:
    lines = [
        f"% auto-generated from validated protocol_id={PROTOCOL_ID} bundles",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Cost of one optimizer step under the jsc\_v3 protocol, which "
        r"changes only the derivative backend.  Times are the three-seed mean of "
        r"milliseconds per step and memory is the peak allocation of a run; the "
        r"ratio columns give nested automatic differentiation divided by the "
        r"Taylor jet, so larger is better for the jet.}",
        r"\label{tab_v3_cost}",
        r"\begin{tabular}{@{}lrrrcrrc@{}}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{ms per step} & \multicolumn{3}{c}{peak memory (MiB)} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"setting & $p$ & jet & nested & ratio & jet & nested & ratio \\",
        r"\midrule",
    ]
    for task_id, label in TASKS:
        grouped = load(task_id)
        jet, nested = grouped[JET], grouped[AUTODIFF]
        order = int(jet[0]["order"])
        jet_ms, nested_ms = mean_of(jet, "ms_per_step"), mean_of(nested, "ms_per_step")
        jet_mb, nested_mb = mean_of(jet, "peak_mb"), mean_of(nested, "peak_mb")
        lines.append(
            f"{label} & {order} & "
            f"${jet_ms:.1f}$ & ${nested_ms:.1f}$ & $\\mathbf{{{nested_ms / jet_ms:.2f}}}$ & "
            f"${jet_mb:.0f}$ & ${nested_mb:.0f}$ & $\\mathbf{{{nested_mb / jet_mb:.2f}}}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "jsc_v3_cost.tex"
    path.write_text("\n".join(lines))
    return path


def build_outcome() -> Path:
    lines = [
        f"% auto-generated from validated protocol_id={PROTOCOL_ID} bundles",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\footnotesize",
        r"\caption{What the saved time buys under the same protocol: optimizer "
        r"steps completed inside the common 1000-second budget, as a three-seed "
        r"mean, and the held-out relative $L^2$ error reached, as a three-seed "
        r"mean $\pm$ sample standard deviation.  The lower error in each row is "
        r"set in bold.}",
        r"\label{tab_v3_outcome}",
        r"\begin{tabular}{@{}lrrcc@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{steps completed} & \multicolumn{2}{c}{relative $L^2$ error} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"setting & jet & nested & jet & nested \\",
        r"\midrule",
    ]
    for task_id, label in TASKS:
        grouped = load(task_id)
        jet, nested = grouped[JET], grouped[AUTODIFF]
        jet_cell, nested_cell = error_cell(jet), error_cell(nested)
        if mean_of(jet, "L2_err") <= mean_of(nested, "L2_err"):
            jet_cell = jet_cell.replace("$", r"$\mathbf{", 1)[:-1] + "}$"
        else:
            nested_cell = nested_cell.replace("$", r"$\mathbf{", 1)[:-1] + "}$"
        lines.append(
            f"{label} & ${mean_of(jet, 'steps'):.0f}$ & ${mean_of(nested, 'steps'):.0f}$ & "
            f"{jet_cell} & {nested_cell} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / "jsc_v3_outcome.tex"
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    if not RESULT_ROOT.exists():
        print(f"[skip] no {PROTOCOL_ID} results in {RESULT_ROOT}")
        return
    print(f"[ok] {build_cost()}")
    print(f"[ok] {build_outcome()}")


if __name__ == "__main__":
    main()
