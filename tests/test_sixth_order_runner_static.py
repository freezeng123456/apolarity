from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_sixth_order_wrapper_reenters_itself_for_workers() -> None:
    path = ROOT / "scripts" / "run_sixth_order_plate.py"
    source = path.read_text()
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_sixth_order_worker_command"
    )
    assignments = [
        node for node in function.body if isinstance(node, ast.Assign)
    ]
    assert any(
        isinstance(target, ast.Subscript)
        for node in assignments
        for target in node.targets
    )
    assert "runner.worker_command = _sixth_order_worker_command" in source
