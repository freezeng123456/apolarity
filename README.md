# apolarity

`apolarity` provides a deterministic, exact backend for one fixed
high-order mixed partial derivative of a neural-network surrogate, based
on identifying the directional schedule problem with the Waring
decomposition of a monomial.

Scope:

- one fixed multi-index at a time (no Laplacian powers, no trace
  contractions, no operator sums);
- no stochastic estimation;
- value and parameter-gradient computation through Taylor-mode
  automatic differentiation.

## Method in one sentence

For a fixed multi-index of order `p`, the Carlini--Catalisano--Geramita
(2012) rank theorem and its roots-of-unity construction give an
explicit complex directional schedule of length

```
R_C(z^alpha) = prod_{j>=1} (a_j + 1)
```

(where `a_0 <= ... <= a_n` are the active exponents). The directional
Taylor coefficients are evaluated by a forward-mode jet pass through a
`sinh`-activated MLP with complex parameters; because `sinh` is entire
on the complex plane, the jet rules are well-defined globally.

## Repository layout

```text
src/apolarity/
  waring.py          # complex monomial Waring directions
  polarization.py    # antipodally merged real polarization directions
  taylor_jet.py      # Taylor-mode AD for Linear / sinh / tanh MLPs
  operators.py       # single_monomial_partial entry point
experiments/
  common/            # shared harness (osc_common.py)
  tools/             # historical jsc_v2 figure and LaTeX-table builders
  polyharmonic/      # active Polyharmonic family
  chirp/             # active radial-chirp family
  maxwell/           # active Maxwell family
  results/jsc_v2/    # historical validated formal bundles
  archived/          # other families, auxiliary results, and historical runners
docs/
  beamer/  apolarity_report_zh.tex
  paper/   jsc_paper_main.tex
scripts/
  run_jsc_main3.sh                    # launch exactly one jsc_v3 setting
  validate_jsc_results.py             # validate one atomic result bundle
tests/
```

See `experiments/README.md` for the frozen experiment protocol.

## Experiment status

All `experiments/*/data/` directories have been cleared. The active experiment
scope is limited to Polyharmonic, Chirp, and Maxwell. Their historical validated
bundles live under `experiments/results/jsc_v2/`; current v3 output is under
`experiments/results/jsc_v3/`. All other experiment families,
auxiliary result bundles, and historical runners are under
`experiments/archived/` and are not part of the active paper inventory.

The compact v3 formal methods are:

- `complex_sinh` (Complex Sinh, the proposed method);
- `complex_sinh_autodiff` (the same network with direct nested coordinate autodiff).

The current formal protocol is `jsc_v3`. Both lines use the same literal hidden
width \(H=128\), initialization, loss weights, 1000-second budget, and three
seeds; only the derivative backend changes. Trainable real degrees of freedom
are recorded for transparency and \(H=64\) is not used.

The preregistered settings are:

- Poly: \(d=2\), operator order \(2,4,6\);
- Chirp: \(a\in\{1,2,3\}\);
- Maxwell: \(a\in\{2,4,6\}\).

Launch exactly one setting through `scripts/run_jsc_main3.sh`:

```bash
bash scripts/run_jsc_main3.sh poly --dim 2 --order 6
bash scripts/run_jsc_main3.sh chirp --sweep 2
bash scripts/run_jsc_main3.sh maxwell --sweep 4
```

Each command above is a separate example; do not combine settings into one
launch. A formal output bundle is admissible only after
`validate_jsc_results.py` succeeds, for example:

```bash
python scripts/validate_jsc_results.py \
  experiments/results/jsc_v3/poly_d2_o6
```

Historical or archived runners, every family-local `run.sh`, and all material
under `experiments/archived/` are retained only for implementation diagnosis.
Their outputs are not part of the active paper evidence.

## API

```python
from apolarity import single_monomial_partial

deriv = single_monomial_partial(model, x, alpha, backend="auto")
```

`alpha` is the expanded multi-index in zero-based indexing
(for instance `(0, 0, 1)` for the third-order partial that differentiates
twice in coordinate 0 and once in coordinate 1). For complex-parameter
networks, pass an `nn.Sequential` whose linear layers carry
`torch.complex128` weights; the jet rules dispatch on tensor dtype. Inputs must
have shape `(batch, d)` and models must return one scalar per input with shape
`(batch, 1)`; invalid coordinate indices and multi-output models are rejected.

## Documents

- Slide deck:

  ```bash
  cd docs/beamer && xelatex -interaction=nonstopmode apolarity_report_zh.tex
  ```

- Paper draft:

  ```bash
  cd docs/paper && xelatex -interaction=nonstopmode jsc_paper_main.tex
  ```
