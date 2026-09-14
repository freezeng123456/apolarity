# Current paper experiments

Use [torch_benchmarks/](torch_benchmarks/README.md) for both parts of the paper:

- Single partials: batched roots-of-unity Waring formula versus nested JVP.
- PDE training: the WDD method with prescribed real directions versus nested JVP,
  with interior and constraint points resampled at every update.

`jax_benchmarks/` is historical and excluded from package installation and the
current paper protocol. The older automatic-selection and fixed-constraint
training entry points are retained as source archives under `../archive/`.
