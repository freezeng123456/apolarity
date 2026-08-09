#!/usr/bin/env python3
"""Validate and merge one atomic jsc_v3 result bundle."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
sys.path.insert(0, str(COMMON))

from protocol import (  # noqa: E402
    BOUNDARY_PROFILE_ID,
    BOUNDARY_WEIGHTS,
    BUDGET_SECONDS,
    COLLOCATION_PROTOCOL,
    DEPTH,
    FORMAL_METHODS,
    FORMAL_WIDTH,
    N_BOUNDARY,
    N_INTERIOR,
    PROTOCOL_ID,
    SEEDS,
)


REQUIRED_FIELDS = {
    "protocol_id",
    "boundary_profile_id",
    "boundary_weights",
    "git_sha",
    "git_dirty",
    "task_id",
    "family",
    "dimension",
    "order",
    "sweep",
    "variant",
    "seed",
    "actual_width",
    "real_dof",
    "reference_real_dof",
    "relative_dof_difference",
    "representation",
    "collocation",
    "evaluation_protocol",
    "frequency_initialization",
    "complex_sinh_omega0",
    "siren_first_omega0",
    "siren_hidden_omega0",
    "fourier_branch_sigmas",
    "fourier_input_mean",
    "fourier_input_std",
    "mscale_scales",
    "budget_seconds",
    "n_int",
    "n_bc",
    "depth",
    "lr",
    "lr_schedule",
    "hardware",
    "steps",
    "ms_per_step",
    "loss_last",
    "L_int_last",
    "L2_err",
    "rel_error",
    "nan",
}
FINITE_METRICS = ("ms_per_step", "loss_last", "L_int_last", "L2_err", "rel_error")


def _read_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"{path} must contain a JSON list of result rows")
        rows.extend(payload)
    return rows


def _validate_rows(rows: list[dict], *, smoke: bool) -> tuple[str, list[dict]]:
    expected_seeds = {0} if smoke else set(SEEDS)
    expected_count = len(FORMAL_METHODS) * len(expected_seeds)
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows, found {len(rows)}")
    task_ids = {str(row.get("task_id")) for row in rows}
    if len(task_ids) != 1:
        raise ValueError(f"expected one task_id, found {sorted(task_ids)}")
    task_id = next(iter(task_ids))

    seen: set[tuple[str, int]] = set()
    by_method: dict[str, set[int]] = {method: set() for method in FORMAL_METHODS}
    reference_dofs: set[int] = set()
    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(f"row {index} missing metadata: {sorted(missing)}")
        if row["protocol_id"] != PROTOCOL_ID:
            raise ValueError(f"row {index} has protocol_id={row['protocol_id']!r}")
        if row["boundary_profile_id"] != BOUNDARY_PROFILE_ID:
            raise ValueError(
                f"row {index} has boundary_profile_id={row['boundary_profile_id']!r}"
            )
        try:
            weights = tuple(float(value) for value in json.loads(row["boundary_weights"]))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"row {index} has malformed boundary_weights") from exc
        expected_weights = BOUNDARY_WEIGHTS.get(str(row["task_id"]))
        if expected_weights is None or len(weights) != len(expected_weights):
            raise ValueError(f"row {index} has wrong boundary weight length")
        if any(not math.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError(f"row {index} has non-positive boundary weight")
        if any(not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)
               for value, expected in zip(weights, expected_weights)):
            raise ValueError(f"row {index} has boundary weights inconsistent with {BOUNDARY_PROFILE_ID}")
        method = str(row["variant"])
        seed = int(row["seed"])
        if method not in FORMAL_METHODS:
            raise ValueError(f"row {index} has non-formal method {method!r}")
        key = (method, seed)
        if key in seen:
            raise ValueError(f"duplicate method/seed key {key}")
        seen.add(key)
        by_method[method].add(seed)
        if int(row["actual_width"]) != FORMAL_WIDTH:
            raise ValueError(
                f"{PROTOCOL_ID} requires literal H={FORMAL_WIDTH}; "
                f"row {index} has H={row['actual_width']}"
            )
        if int(row["depth"]) != DEPTH:
            raise ValueError(f"row {index} depth does not match {PROTOCOL_ID}")
        if not smoke:
            if float(row["budget_seconds"]) != BUDGET_SECONDS:
                raise ValueError(f"row {index} budget does not match {PROTOCOL_ID}")
            if int(row["n_int"]) != N_INTERIOR or int(row["n_bc"]) != N_BOUNDARY:
                raise ValueError(
                    f"row {index} collocation sizes do not match {PROTOCOL_ID}"
                )
        if row["collocation"] != COLLOCATION_PROTOCOL:
            raise ValueError(f"row {index} has wrong collocation protocol")
        real_dof = int(row["real_dof"])
        reference_dof = int(row["reference_real_dof"])
        reference_dofs.add(reference_dof)
        difference = abs(real_dof - reference_dof) / reference_dof
        if abs(float(row["relative_dof_difference"]) - difference) > 1e-12:
            raise ValueError(f"row {index} has inconsistent real-DOF difference")
        if bool(row["nan"]):
            raise ValueError(f"row {index} reports a NaN training failure")
        if int(row["steps"]) <= 0:
            raise ValueError(f"row {index} completed no optimizer steps")
        for metric in FINITE_METRICS:
            if not math.isfinite(float(row[metric])):
                raise ValueError(f"row {index} has non-finite {metric}")
        if abs(float(row["rel_error"]) - float(row["L2_err"])) > 1e-12:
            raise ValueError(f"row {index} has inconsistent rel_error/L2_err")
        if not str(row["git_sha"]) or not str(row["hardware"]):
            raise ValueError(f"row {index} lacks provenance strings")

    if len(reference_dofs) != 1:
        raise ValueError("all rows must use the same Complex Sinh DOF reference")
    for method, seeds in by_method.items():
        if seeds != expected_seeds:
            raise ValueError(
                f"{method} seeds are {sorted(seeds)}, expected {sorted(expected_seeds)}"
            )
    return task_id, sorted(rows, key=lambda row: (FORMAL_METHODS.index(row["variant"]), int(row["seed"])))


def _validate_histories(paths: list[Path], rows: list[dict]) -> list[dict]:
    histories = _read_rows(paths)
    expected = {(row["variant"], int(row["seed"])) for row in rows}
    task_id = str(rows[0]["task_id"])
    expected_weights = BOUNDARY_WEIGHTS[task_id]
    seen: set[tuple[str, int]] = set()
    for index, item in enumerate(histories):
        if item.get("protocol_id") != PROTOCOL_ID:
            raise ValueError(f"history {index} has wrong protocol_id")
        if item.get("boundary_profile_id") != BOUNDARY_PROFILE_ID:
            raise ValueError(f"history {index} has wrong boundary_profile_id")
        try:
            history_weights = tuple(float(value) for value in item["boundary_weights"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"history {index} has malformed boundary_weights") from exc
        if history_weights != expected_weights:
            raise ValueError(f"history {index} has inconsistent boundary_weights")
        key = (str(item.get("variant")), int(item.get("seed", -1)))
        if key in seen:
            raise ValueError(f"duplicate history key {key}")
        seen.add(key)
        history = item.get("history")
        if not isinstance(history, list) or not history:
            raise ValueError(f"history {index} is empty")
        previous_time = -math.inf
        for point in history:
            if not isinstance(point, list) or len(point) != 4:
                raise ValueError(f"history {index} has malformed point")
            values = [float(value) for value in point]
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"history {index} has non-finite values")
            if values[0] < previous_time:
                raise ValueError(f"history {index} time is not monotone")
            previous_time = values[0]
    if seen != expected:
        raise ValueError(
            f"history keys do not match rows: missing={sorted(expected-seen)}, "
            f"extra={sorted(seen-expected)}"
        )
    return sorted(
        histories,
        key=lambda item: (FORMAL_METHODS.index(item["variant"]), int(item["seed"])),
    )


def _write_outputs(
    rows: list[dict],
    histories: list[dict],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    keys = sorted({key for row in rows for key in row})
    with output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    output.with_name(output.name + "_history").with_suffix(".json").write_text(
        json.dumps(histories, indent=2) + "\n"
    )


def validate_task_directory(
    task_dir: Path,
    *,
    output: Path | None = None,
    smoke: bool = False,
) -> Path:
    row_paths = sorted(task_dir.glob("*_part.json"))
    history_paths = sorted(task_dir.glob("*_part_history.json"))
    if not row_paths:
        raise ValueError(f"no *_part.json files found in {task_dir}")
    rows = _read_rows(row_paths)
    task_id, rows = _validate_rows(rows, smoke=smoke)
    histories = _validate_histories(history_paths, rows)
    canonical = output or task_dir / task_id
    _write_outputs(rows, histories, canonical)
    (task_dir / "VALIDATED").write_text(
        f"protocol_id={PROTOCOL_ID}\ntask_id={task_id}\nrows={len(rows)}\n"
    )
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    canonical = validate_task_directory(
        args.task_dir,
        output=args.output,
        smoke=args.smoke,
    )
    print(f"[validated] {canonical}")


if __name__ == "__main__":
    main()
