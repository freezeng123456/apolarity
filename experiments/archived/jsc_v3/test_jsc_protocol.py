from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(COMMON))
sys.path.insert(0, str(SCRIPTS))

from protocol import (  # noqa: E402
    BOUNDARY_PROFILE_ID,
    BUDGET_SECONDS,
    COLLOCATION_PROTOCOL,
    DEPTH,
    EVALUATION_PROTOCOL,
    FORMAL_METHODS,
    FORMAL_WIDTH,
    N_BOUNDARY,
    N_INTERIOR,
    PROTOCOL_ID,
    SEEDS,
    all_tasks,
    get_task,
    validate_architecture_table,
)
from validate_jsc_results import validate_task_directory  # noqa: E402


def test_preregistered_grid_has_nine_atomic_tasks_and_requested_poly_cases():
    tasks = all_tasks()
    assert len(tasks) == 9
    assert len({task.task_id for task in tasks}) == 9
    assert sum(task.family == "poly" for task in tasks) == 3
    assert sum(task.family == "chirp" for task in tasks) == 3
    assert sum(task.family == "maxwell" for task in tasks) == 3
    assert {task.task_id for task in tasks if task.family == "poly"} == {
        "poly_d2_o2", "poly_d2_o4", "poly_d2_o6"
    }


def test_v3_uses_two_backends_three_seeds_and_1000_seconds():
    assert FORMAL_METHODS == ("complex_sinh", "complex_sinh_autodiff")
    assert SEEDS == (0, 1, 2)
    assert BUDGET_SECONDS == 1000.0


@pytest.mark.parametrize(
    ("family", "kwargs"),
    [
        ("poly", {"dimension": 2, "order": 8}),
        ("poly", {"dimension": 3, "order": 2}),
        ("poly", {"dimension": 2, "order": 4, "sweep": 4}),
        ("chirp", {"sweep": 4}),
        ("maxwell", {"sweep": 3}),
    ],
)
def test_protocol_rejects_non_preregistered_settings(family, kwargs):
    with pytest.raises(ValueError):
        get_task(family, **kwargs)


@pytest.mark.parametrize(
    ("family", "kwargs"),
    [
        ("poly", {"dimension": 2, "order": 6}),
        ("poly", {"dimension": 3, "order": 6}),
        ("maxwell", {"sweep": 6}),
    ],
)
def test_protocol_architecture_tables_use_literal_width_128(family, kwargs):
    specs = validate_architecture_table(get_task(family, **kwargs))
    assert set(specs) == set(FORMAL_METHODS)
    assert all(spec.width == FORMAL_WIDTH for spec in specs.values())


def _row(method: str, seed: int, width: int, real_dof: int, reference: int) -> dict:
    return {
        "protocol_id": PROTOCOL_ID,
        "boundary_profile_id": BOUNDARY_PROFILE_ID,
        "boundary_weights": "[0.1]",
        "git_sha": "a" * 40,
        "git_dirty": False,
        "task_id": "poly_d2_o2",
        "family": "poly",
        "dimension": 2,
        "order": 2,
        "sweep": 2.0,
        "problem": "polyharm2d_o2",
        "variant": method,
        "seed": seed,
        "actual_width": width,
        "hidden": width,
        "params": real_dof,
        "real_dof": real_dof,
        "reference_real_dof": reference,
        "relative_dof_difference": abs(real_dof - reference) / reference,
        "representation": "native_complex" if method.startswith("complex_sinh") else "real",
        "collocation": COLLOCATION_PROTOCOL,
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "frequency_initialization": "{}",
        "complex_sinh_omega0": 6.28,
        "siren_first_omega0": 30.0,
        "siren_hidden_omega0": 30.0,
        "fourier_branch_sigmas": "[1.0, 3.14]",
        "fourier_input_mean": "[0.0, 0.0]",
        "fourier_input_std": "[0.577, 0.577]",
        "mscale_scales": "[1.0, 2.0, 4.0]",
        "budget_seconds": BUDGET_SECONDS,
        "n_int": N_INTERIOR,
        "n_bc": N_BOUNDARY,
        "depth": DEPTH,
        "lr": 1e-3,
        "lr_schedule": "cosine",
        "hardware": "test",
        "steps": 10,
        "ms_per_step": 1.0,
        "loss_last": 0.3,
        "L_int_last": 0.1,
        "L2_err": 0.2,
        "rel_error": 0.2,
        "nan": False,
    }


def _write_fake_bundle(task_dir: Path, *, bad_width: bool = False) -> None:
    target = 100_000
    for method in FORMAL_METHODS:
        width = 64 if bad_width and method == "siren" else FORMAL_WIDTH
        rows = [_row(method, seed, width, target, target) for seed in SEEDS]
        (task_dir / f"{method}_part.json").write_text(json.dumps(rows))
        histories = [
            {
                "protocol_id": PROTOCOL_ID,
                "boundary_profile_id": BOUNDARY_PROFILE_ID,
                "boundary_weights": [0.1],
                "task_id": "poly_d2_o2",
                "variant": method,
                "seed": seed,
                "history": [[0.0, 1.0, 0.8, 1.0], [1000.0, 0.2, 0.3, 0.1]],
            }
            for seed in SEEDS
        ]
        (task_dir / f"{method}_part_history.json").write_text(json.dumps(histories))


def test_validator_merges_only_complete_five_seed_bundle(tmp_path: Path):
    _write_fake_bundle(tmp_path)
    output = validate_task_directory(tmp_path)
    rows = json.loads(output.with_suffix(".json").read_text())
    assert len(rows) == 10
    assert (tmp_path / "VALIDATED").exists()
    assert {row["variant"] for row in rows} == set(FORMAL_METHODS)


def test_validator_rejects_any_non_128_width(tmp_path: Path):
    _write_fake_bundle(tmp_path, bad_width=True)
    with pytest.raises(ValueError, match="requires literal H=128"):
        validate_task_directory(tmp_path)
