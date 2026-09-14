"""Checks target the current first-group path, not the old nested-grad runner."""
import inspect
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import first_group
from jax_apolarity_bench.models import init_mlp, sample_normal
from jax_apolarity_bench.directions import parse_one_based_pattern

@pytest.mark.parametrize('target', ['111', '112', '123'])
def test_current_evaluators_agree_without_jit(target, monkeypatch):
    # A hidden explicit JIT/scan in the evaluated wrapper must fail this test.
    def forbidden(*args, **kwargs):
        raise AssertionError('outer compilation is not allowed in this test')
    params = init_mlp(14, 3, 4, 2)
    points = sample_normal(15, 2, 3)
    alpha = parse_one_based_pattern(target)
    fs = [first_group.make_evaluator(alpha,3,m,serial_python=True)
          for m in ('nested_jvp','waring_batched','waring_serial')]
    monkeypatch.setattr(jax, 'jit', forbidden)
    monkeypatch.setattr(jax.lax, 'scan', forbidden)
    vals = [np.asarray(f(params,points)) for f in fs]
    np.testing.assert_allclose(vals[1],vals[0],rtol=1e-3,atol=1e-7)
    np.testing.assert_allclose(vals[2],vals[1],rtol=1e-3,atol=1e-7)
    assert not np.allclose(np.asarray(fs[0](params, points + 0.2)), vals[0], rtol=1e-5, atol=1e-10)

def test_main_execution_default_and_null_compilation_metadata():
    source = inspect.getsource(first_group.main)
    assert 'default="no_outer_jit"' in source
    assert 'compile_s, analysis = None, None' in source
