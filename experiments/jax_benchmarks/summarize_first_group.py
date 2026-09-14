"""Validate recovered cells and summarize matched, per-seed timings."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.root / "manifest.json").read_text())
    execution = manifest.get("execution", "whole_function_jit")
    rows, missing = [], []
    for target, method, batch in manifest["cells"]:
        name = f"{target}_{method}_b{batch}"
        path = args.root / name
        if not (path / "status.txt").exists() or (path / "status.txt").read_text().strip() != "COMPLETE":
            missing.append(name)
            continue
        records = json.loads((path / "results.json").read_text())
        config = json.loads((path / "config.json").read_text())
        if config.get("execution", "whole_function_jit") != execution:
            raise ValueError(f"mixed execution policies: {name}")
        if sorted(r["seed"] for r in records) != sorted(manifest["seeds"]):
            raise ValueError(f"incomplete or duplicate seeds: {name}")
        for r in records:
            r["cell"] = name
            rows.append(r)
    lookup = {(r["target"], r["method"], r["batch"], r["seed"]): r for r in rows}
    comparisons = []
    for r in rows:
        if r["method"] not in ("waring_batched", "waring_serial"):
            continue
        ref_method = "nested_jvp" if r["method"] == "waring_batched" else "waring_batched"
        ref = lookup.get((r["target"], ref_method, r["batch"], r["seed"]))
        if ref is None:
            continue
        actual = np.load(args.root / r["cell"] / f"values_{r['seed']}.npy")
        expected = np.load(args.root / ref["cell"] / f"values_{r['seed']}.npy")
        relative = float(np.linalg.norm(actual - expected) / max(float(np.linalg.norm(expected)), 1e-30))
        comparisons.append({"target": r["target"], "method": r["method"], "batch": r["batch"],
                            "seed": r["seed"], "reference": ref_method, "relative_l2": relative,
                            "max_absolute": float(np.max(np.abs(actual - expected))),
                            "consistent": bool(np.allclose(actual, expected, rtol=1e-3, atol=1e-7))})
    groups = sorted(set((r["target"], r["batch"]) for r in rows), key=lambda x: (x[1], len(x[0]), x[0]))
    summary = []
    for target, batch in groups:
        paired = [(lookup.get((target, "nested_jvp", batch, seed)), lookup.get((target, "waring_batched", batch, seed)))
                  for seed in manifest["seeds"]]
        if not all(a and b for a, b in paired):
            continue
        a = [x["median_ms"] for x, _ in paired]
        b = [y["median_ms"] for _, y in paired]
        speedups = [x / y for x, y in zip(a, b)]
        summary.append({"target": target, "batch": batch, "order": len(target),
                        "rank": paired[0][1]["rank"], "nested_mean_ms": statistics.mean(a),
                        "nested_std_ms": statistics.stdev(a), "waring_mean_ms": statistics.mean(b),
                        "waring_std_ms": statistics.stdev(b), "paired_speedup_mean": statistics.mean(speedups)})
    with (args.out / "summary.csv").open("w", newline="") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    checks = {"expected_rows": manifest["expected_rows"], "observed_rows": len(rows),
              "execution": execution, "missing_cells": missing,
              "all_outputs_consistent": bool(comparisons) and all(c["consistent"] for c in comparisons),
              "comparisons": comparisons}
    (args.out / "validation.json").write_text(json.dumps(checks, indent=2))
    lines = ["# JAX first-group results", "", f"Recovered {len(rows)}/{manifest['expected_rows']} measurements.",
             "", f"Execution policy: {execution} (both methods).",
             "Value-only; no parameter backward. Times include device synchronization; setup and cold call excluded.",
             "Mean ± sample standard deviation of three per-seed medians (30 timed calls per seed).",
             "", "| Target | Batch | Directions | Nested JVP (ms) | Waring (ms) | Paired speedup |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['target']} | {r['batch']} | {r['rank']} | {r['nested_mean_ms']:.4f} ± {r['nested_std_ms']:.4f} | {r['waring_mean_ms']:.4f} ± {r['waring_std_ms']:.4f} | {r['paired_speedup_mean']:.2f}x |")
    serial_kind = "Python direction loop" if execution == "no_outer_jit" else "Serial scan"
    lines += ["", "## Direction-parallelism ablation", "", f"{serial_kind} / batched runtime; same directions and weights.", ""]
    for target in ("123456", "111222333"):
        pairs = [(lookup.get((target, "waring_serial", 100, seed)), lookup.get((target, "waring_batched", 100, seed))) for seed in manifest["seeds"]]
        if all(a and b for a, b in pairs):
            ratio = statistics.mean(a["median_ms"] / b["median_ms"] for a, b in pairs)
            lines.append(f"- {target}: {ratio:.2f}x")
    lines += ["", "## Memory and compilation", "", "Compiler working-buffer estimates and allocator process-lifetime peaks are distinct; neither is labelled as measured runtime-only GPU peak.", "",
              "| Target | Method | Batch | Compile (s) | Compiler buffers (MiB) | Allocator lifetime peak (MiB) |", "|---|---|---:|---:|---:|---:|"]
    for r in rows:
        if r["seed"] != manifest["seeds"][0] or r["batch"] != 100:
            continue
        m = r["executable_memory"]
        buffer = ((m.get("argument_size_in_bytes", 0) + m.get("output_size_in_bytes", 0) + m.get("temp_size_in_bytes", 0) - m.get("alias_size_in_bytes", 0)) / 2**20) if m else None
        peak = r["allocator_peak_bytes"]
        peak_text = f"{peak / 2**20:.2f}" if peak is not None else "unavailable"
        compile_text = f"{r['compile_seconds']:.2f}" if r['compile_seconds'] is not None else "N/A"
        buffer_text = f"{buffer:.2f}" if buffer is not None else "N/A"
        lines.append(f"| {r['target']} | {r['method']} | {r['batch']} | {compile_text} | {buffer_text} | {peak_text} |")
    if missing:
        lines += ["", "Missing cells: " + ", ".join(missing)]
    lines += ["", "Output consistency: " + str(checks["all_outputs_consistent"]), ""]
    (args.out / "REPORT.md").write_text("\n".join(lines))
    # Standalone table fragments for author review; not inserted in the paper.
    for table_name, selected in (
        ("main_table.tex", [r for r in summary if r["batch"] == 100]),
        ("batch_table.tex", [r for r in summary if r["target"] in ("123456", "111222333")]),
    ):
        second_header = "Order" if table_name == "main_table.tex" else "Points"
        tex = [f"% Execution: {execution}; author review required before manuscript insertion.",
               r"\begin{tabular}{lrrrrr}", r"\toprule",
               "Derivative & " + second_header + r" & Directions & Nested JVP (ms) & Waring (ms) & Speedup \\", r"\midrule"]
        for r in selected:
            second_value = r['order'] if table_name == "main_table.tex" else r['batch']
            tex.append(f"$u_{{{r['target']}}}$ & {second_value} & {r['rank']} & "
                       f"${r['nested_mean_ms']:.2f} \\pm {r['nested_std_ms']:.2f}$ & "
                       f"${r['waring_mean_ms']:.2f} \\pm {r['waring_std_ms']:.2f}$ & "
                       f"${r['paired_speedup_mean']:.2f}\\times$ " + r"\\")
        tex += [r"\bottomrule", r"\end{tabular}", ""]
        (args.out / table_name).write_text("\n".join(tex))
    print(json.dumps({k: v for k, v in checks.items() if k != "comparisons"}))
    if missing or len(rows) != manifest["expected_rows"] or not checks["all_outputs_consistent"]:
        raise SystemExit("Incomplete or inconsistent results; report preserved for diagnosis")


if __name__ == "__main__":
    main()
