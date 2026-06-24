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
  benchmark_single_monomial.py        # micro-benchmark
  train_pinn_ch_sixth_order.py        # 4D Cahn-Hilliard PINN
  pinn_5min_compare.py                # one-shot 3-backend comparison
  generate_paper_tables.py            # CSV-to-LaTeX table snippets
docs/
  beamer/  apolarity_report_zh.tex    # Chinese slide deck (17 pages)
  paper/   jsc_paper_main.tex         # JSC paper draft (12 pages)
scripts/
  cuda_env.sh                         # CUDA loader for the T4 host
  run_quick_compare.sh                # 5-minute backend sweep
tests/
```

## Quick benchmark

```bash
source scripts/cuda_env.sh
PYTHONPATH=src python3.11 experiments/benchmark_single_monomial.py \
  --device auto --dtype float64 --d 8 --batch 8 \
  --hidden 64 --depth 4 --activation sinh --warmup 5 --repeats 60 \
  --methods direct_autodiff,polarization_jet,waring_complex_jet \
  --alphas '111;1111;1122;111111;111122;112233;123456' \
  --out results/paper_jsc_revision/single_monomial_value.csv
```

## PINN comparison (3 backends, 60 s each)

```bash
source scripts/cuda_env.sh
PYTHONPATH=src python3.11 experiments/pinn_5min_compare.py \
  --seconds 60 --hidden 32 --depth 4 --n-int 128 --n-bc 64 \
  --backends waring_complex_jet,polarization_jet,direct_autodiff \
  --out results/paper_jsc_revision/pinn_60s.csv
```

The three backends compared are `direct_autodiff`, `polarization_jet`,
and `waring_complex_jet`. Results for the four-dimensional manufactured
sixth-order PDE are reported in the paper, Section 5.

## Paper table snippets

```bash
PYTHONPATH=src python3.11 experiments/generate_paper_tables.py \
  --micro-csv results/paper_jsc_revision/single_monomial_value.csv \
  --pinn-csv results/paper_jsc_revision/pinn_60s.csv \
  --out-dir results/paper_jsc_revision/table_snippets
```

## API

```python
from apolarity import single_monomial_partial

deriv = single_monomial_partial(model, x, alpha, backend="auto")
```

`alpha` is the expanded multi-index in zero-based indexing
(for instance `(0, 0, 1)` for the third-order partial that differentiates
twice in coordinate 0 and once in coordinate 1). For complex-parameter
networks, pass an `nn.Sequential` whose linear layers carry
`torch.complex128` weights; the jet rules dispatch on tensor dtype.

## Environment (Tesla T4 host)

The host carries `python3.11` and `torch 2.5.1+cu121` at
`/usr/local/lib/python3.11/site-packages/`, but the NVIDIA wheel
shared libraries are not on the default `LD_LIBRARY_PATH`. The
canonical loader is committed:

```bash
source scripts/cuda_env.sh
python3.11 -c "import torch; print(torch.cuda.get_device_name(0))"
# -> Tesla T4
```

## Documents

- Slide deck (Chinese, 17 pages):

  ```bash
  cd docs/beamer && xelatex -interaction=nonstopmode apolarity_report_zh.tex
  ```

- Paper draft (12 pages):

  ```bash
  cd docs/paper && xelatex -interaction=nonstopmode jsc_paper_main.tex
  ```
