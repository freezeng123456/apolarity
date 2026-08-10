from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_orchestrator_hashes_after_terminal_log_record() -> None:
    source = (ROOT / "scripts" / "run_high_order_candidate_screen.py").read_text()
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_orchestrate"
    )
    terminal_print, checksum_call, return_node = function.body[-3:]
    assert isinstance(terminal_print, ast.Expr)
    assert isinstance(terminal_print.value, ast.Call)
    assert isinstance(terminal_print.value.func, ast.Name)
    assert terminal_print.value.func.id == "print"
    assert isinstance(checksum_call, ast.Expr)
    assert isinstance(checksum_call.value, ast.Call)
    assert isinstance(checksum_call.value.func, ast.Name)
    assert checksum_call.value.func.id == "write_checksums"
    assert isinstance(return_node, ast.Return)
