from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "run_dynamic_plate_weight_search.py"
SOURCE = PATH.read_text()
MODULE = ast.parse(SOURCE)


def _function(name: str) -> ast.FunctionDef:
    return next(
        node
        for node in MODULE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_worker_subprocess_reenters_task_specific_wrapper() -> None:
    function = _function("_worker_command")
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Subscript) for target in node.targets)
        for node in function.body
    )
    assert "runner.worker_command = _worker_command" in SOURCE


def test_runner_preserves_failed_attempts_and_strict_resume_fields() -> None:
    assert "MAX_RETRIES = 1" in SOURCE
    assert "preexisting_incomplete" in SOURCE
    assert "failed_attempt_" in SOURCE
    function_source = ast.get_source_segment(SOURCE, _function("_cell_complete"))
    assert function_source is not None
    for required in (
        "train_seed",
        "eval_seed",
        "budget_seconds",
        "weights",
        "SAMPLE_COUNTS",
    ):
        assert required in function_source


def test_candidate_method_order_alternates_and_run_pid_is_not_hashed() -> None:
    orchestrator = ast.get_source_segment(SOURCE, _function("_run_orchestrator"))
    checksums = ast.get_source_segment(SOURCE, _function("_write_checksums"))
    assert orchestrator is not None and "index % 2 == 0" in orchestrator
    assert checksums is not None and '"run.pid"' in checksums


def test_terminal_record_precedes_final_checksums() -> None:
    function = _function("_run_orchestrator")
    terminal_index = next(
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "print"
        and "ORCHESTRATOR_FINAL" in ast.get_source_segment(SOURCE, node)
    )
    checksum_index = next(
        index
        for index, node in enumerate(function.body)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_write_checksums"
    )
    assert terminal_index < checksum_index
