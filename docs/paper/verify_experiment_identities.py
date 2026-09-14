"""Check the paper identities against the frozen experiment reconstruction."""
from pathlib import Path
from fractions import Fraction
from math import factorial
import json
import sys
import importlib.util
import sympy as sp
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/torch_benchmarks'
FROZEN = ROOT / 'results/paper_resampled_20260914/source/experiments/torch_benchmarks/torch_pinn'
sys.path.insert(0, str(SOURCE))
from torch_pinn.operators import GKDV_DIRS, KDV_DIRS, GKDV_TERMS, KDV_TERMS, reconstruct


def main():
    spec = importlib.util.spec_from_file_location('_identity_frozen', FROZEN/'__init__.py',
                                                 submodule_search_locations=[str(FROZEN)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    from _identity_frozen.operators import reconstruct as frozen_reconstruct
    x, y, t = sp.symbols('z_x z_y z_t')
    checks = []

    def check(name, left, right):
        assert sp.expand(left - right) == 0, name
        checks.append(name)

    quartic = lambda a, b: ((a+b)**4-(a-b)**4)/6 - ((a+2*b)**4-(a-2*b)**4)/48
    check('KdV1D quartic identity', x**3*t, quartic(x, t))
    check('KdV2D quartic identity', x**3*y, quartic(x, y))
    check('KdV2D time identity', y*t, ((y+t)**2-(y-t)**2)/4)
    for case, directions, terms, z in [
            ('gkdv1d', GKDV_DIRS, GKDV_TERMS, (x, t)),
            ('kdv2d', KDV_DIRS, KDV_TERMS, (x, y, t))]:
        n = len(directions)
        forms = [sum(a*b for a, b in zip(v, z)) for v in directions]
        for name, axes in terms.items():
            order = len(axes)
            q = [torch.zeros((n, n), dtype=torch.float64) for _ in range(5)]
            q[order] = torch.eye(n, dtype=torch.float64)
            weights = reconstruct(case, q)[name].tolist()
            assert weights == frozen_reconstruct(case, q, linear=True)[name].tolist()
            weights = [sp.Rational(Fraction(w).limit_denominator(1000)) for w in weights]
            rhs = sum(w * form**order for w, form in zip(weights, forms))/factorial(order)
            lhs = sp.prod(z[axis] for axis in axes)
            check(case + ' current/frozen weights: ' + name, lhs, rhs)
    forms = (x, (x+sp.sqrt(3)*y)/2, (-x+sp.sqrt(3)*y)/2)
    check('CH biharmonic identity', (x*x+y*y)**2, sp.Rational(8, 9)*sum(v**4 for v in forms))
    check('CH Laplacian identity', x*x+y*y, sp.Rational(2, 3)*sum(v*v for v in forms))
    check('CH fourth Taylor weights', (x*x+y*y)**2,
          sp.Rational(64, 3)*sum(v**4 for v in forms)/factorial(4))
    check('CH second Taylor weights', x*x+y*y,
          sp.Rational(4, 3)*sum(v*v for v in forms)/factorial(2))
    a, b, weight = sp.symbols('a b weight', positive=True)
    candidate = weight*(x**4+(a*x+b*y)**4+(-a*x+b*y)**4)
    check('CH symmetric coefficient matching', (x*x+y*y)**2,
          candidate.subs({a: sp.Rational(1, 2), b: sp.sqrt(3)/2, weight: sp.Rational(8, 9)}))
    for order in (1, 2, 3):
        matrix = sp.Matrix([[v[1]**j for v in GKDV_DIRS] for j in range(order+1)])
        assert matrix.rank() == order+1
        checks.append(f'KdV1D lower-order span: {order}')
    for order in (1, 2):
        matrix = sp.Matrix([[v[1]**j for v in KDV_DIRS[:4]] for j in range(order+1)])
        assert matrix.rank() == order+1
        checks.append(f'KdV2D spatial span: {order}')
    result = {'status': 'PASS', 'checks': checks,
              'source_commit': 'b94fd9ce20f1f6c79bd6195ab6d79097f74c8044',
              'method': 'exact polynomial expansion and equality of current/frozen reconstruction weights'}
    output = Path(__file__).resolve().parent / 'data/resampled/active_direction_identity_checks.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
