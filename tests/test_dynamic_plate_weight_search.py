from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "experiments" / "common"
SRC = ROOT / "src"
for path in (str(ROOT), str(COMMON), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

from experiments.dynamic_plate_weight_search import problem  # noqa: E402


def test_frozen_search_table_has_two_tasks_and_49_candidates_each() -> None:
    assert problem.TASK_ORDER == (
        "dynamic_plate_2d_o4",
        "strain_gradient_plate_2d_o6",
    )
    assert problem.METHODS == ("war", "real_tanh_autodiff")
    assert problem.GRID_VALUES == (
        1e-3,
        1e-2,
        1e-1,
        1.0,
        1e1,
        1e2,
        1e3,
    )
    assert len(tuple(itertools.product(problem.GRID_VALUES, repeat=2))) == 49
    for task in problem.TASKS.values():
        assert task.weight_names == ("lambda_ic", "lambda_bc")
        assert task.weight_count == 2
        assert task.center_weights == (10.0, 10.0)
        assert task.spatial_dim == 2
        assert task.order in (4, 6)


def test_weight_replacement_is_immutable_and_validated() -> None:
    original = problem.TASKS["dynamic_plate_2d_o4"]
    changed = problem.with_weights(original, (1e-2, 1e2))
    assert original.weights == (10.0, 10.0)
    assert changed.weights == (1e-2, 1e2)
    assert changed.task_id == original.task_id
    for invalid in ((1.0,), (0.0, 1.0), (-1.0, 1.0)):
        try:
            problem.with_weights(original, invalid)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"invalid weights accepted: {invalid}")


def test_both_tasks_apply_requested_weights_and_have_finite_gradients() -> None:
    for task in problem.TASKS.values():
        weighted = problem.with_weights(task, (1.0, 10.0))
        for method in problem.METHODS:
            torch.manual_seed(7)
            model, dtype, backend = problem.build_model(
                weighted,
                method,
                torch.device("cpu"),
                hidden=4,
                depth=1,
            )
            bundle = problem.make_loss_bundle(
                weighted,
                model,
                dtype,
                backend,
                torch.device("cpu"),
                n_int=2,
                n_ic=2,
                n_bc=8,
                n_eval=4,
                history_eval_n=2,
                train_seed=7,
                eval_seed=11,
            )
            loss, components = bundle.loss_fn()
            torch.testing.assert_close(
                components["weighted_L_IC"], components["L_IC"]
            )
            torch.testing.assert_close(
                components["weighted_L_BC"], 10.0 * components["L_BC"]
            )
            assert bundle.metadata["weights"] == {
                "lambda_ic": 1.0,
                "lambda_bc": 10.0,
            }
            assert bundle.metadata["weight_search_protocol_id"] == problem.PROTOCOL_ID
            assert torch.isfinite(loss)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            assert gradients and all(value is not None for value in gradients)
            assert all(
                torch.isfinite(value).all()
                for value in gradients
                if value is not None
            )


def test_frozen_sample_counts_and_problem_protocols() -> None:
    assert problem.SAMPLE_COUNTS == {
        "n_int": 2048,
        "n_ic": 512,
        "n_bc": 1024,
        "n_eval": 16384,
        "history_eval_n": 2048,
    }
    assert problem.TRAIN_SEED == 42
    assert problem.EVAL_SEED == 68421
    assert math.isclose(problem.GRAD_CLIP, 10.0)
    assert problem.problem_protocol_id(
        problem.TASKS["dynamic_plate_2d_o4"]
    ).startswith("high_order_candidate")
    assert problem.problem_protocol_id(
        problem.TASKS["strain_gradient_plate_2d_o6"]
    ).startswith("strain_gradient_plate")
