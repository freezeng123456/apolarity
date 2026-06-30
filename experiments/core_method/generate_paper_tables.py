#!/usr/bin/env python3
"""Generate LaTeX table rows from saved benchmark CSV files.

The script is intentionally small: it formats existing, versioned CSV outputs
instead of re-running experiments. Use it after running `benchmark_single_monomial.py`
and/or `pinn_5min_compare.py` with `--out`.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def fmt_float(value: str, digits: int = 3) -> str:
    if value in {"", "None", "nan"}:
        return "--"
    return f"{float(value):.{digits}f}"


def fmt_sci(value: str, digits: int = 2) -> str:
    if value in {"", "None", "nan"}:
        return "--"
    return f"{float(value):.{digits}e}"


def micro_table(rows: list[dict[str, str]]) -> str:
    by_alpha: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        by_alpha.setdefault(row["alpha"], {})[row["method"]] = row

    lines = []
    for alpha, methods in by_alpha.items():
        ref = methods.get("direct_autodiff")
        pol = methods.get("polarization_jet")
        war = methods.get("waring_complex_jet")
        if not (ref and pol and war):
            continue
        pattern = ref["active_exponents"].replace(" ", "")
        line = (
            f"{alpha} / {pattern} & {ref['order']} & {ref['complex_rank']} & "
            f"{ref['polarization_dirs']} & {fmt_float(ref['median_ms'])} & "
            f"{fmt_float(pol['median_ms'])} & {fmt_float(war['median_ms'])} & "
            f"{fmt_sci(max(float(pol['rel_err']), float(war['rel_err'])), 1)} \\\\"
        )
        lines.append(line)
    return "\n".join(lines)


def pinn_table(rows: list[dict[str, str]]) -> str:
    lines = []
    for row in rows:
        if row.get("error"):
            continue
        lines.append(
            f"{row['backend']} & {row['steps']} & {fmt_float(row['ms_per_step'], 2)} & "
            f"{fmt_float(row['peak_mb'], 1)} & {fmt_sci(row['L_int_last'], 2)} & "
            f"{fmt_sci(row['L2_err'], 3)} \\\\"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--micro-csv", default="")
    parser.add_argument("--pinn-csv", default="")
    parser.add_argument("--out-dir", default="results/paper_jsc_revision/table_snippets")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.micro_csv:
        text = micro_table(read_rows(args.micro_csv))
        path = out_dir / "micro_rows.tex"
        path.write_text(text + "\n")
        print(f"[ok] wrote {path}")

    if args.pinn_csv:
        text = pinn_table(read_rows(args.pinn_csv))
        path = out_dir / "pinn_rows.tex"
        path.write_text(text + "\n")
        print(f"[ok] wrote {path}")


if __name__ == "__main__":
    main()
