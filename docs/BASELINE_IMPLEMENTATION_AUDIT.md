# Baseline implementation audit

Status: implementation freeze candidate, 2026-07-11.

This document is the traceability record for the four formal methods. The three
external baselines were checked against source files pinned to immutable GitHub
commits. Public GitHub API/raw pages were used because the installed `gh` binary
requires an authenticated token on this host. The immutable URLs below are the
evidence used by the executable tests.

## Decision summary

| formal name | upstream contract | local implementation |
|---|---|---|
| Complex Sinh | method of this project | complex128 sinh MLP with frequency-rich complex initialization |
| SIREN | `vsitzmann/siren` | explicit `sin(omega * Linear(x))`, upstream weight initialization |
| mFF-PINN | `PredictiveIntelligenceLab/MultiscalePINNs` | two frozen Fourier branches, shared tanh trunk, concatenated linear output |
| MscaleDNN-2-sin | `xuzhiqin1990/mscalednn` | independent explicitly scaled sine subnets with summed output |

Only these names are admitted to the formal runner. `tanh`, Cauchy, and
`complex_sinh_noinit` remain auxiliary code. The superseded single-scale RFF
network and the initialization-folded Mscale ensemble are not formal methods.

## SIREN

Pinned source:

- repository: <https://github.com/vsitzmann/siren>
- commit: [`4df34baee3f0f9c8f351630992c1fe1f69114b5f`](https://github.com/vsitzmann/siren/commit/4df34baee3f0f9c8f351630992c1fe1f69114b5f)
- compact implementation:
  [`explore_siren.ipynb`](https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/explore_siren.ipynb)
- experiment implementation:
  [`modules.py`, `Sine`, `FCBlock`, and `SingleBVPNet`](https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py)
- initialization functions:
  [`sine_init` and `first_layer_sine_init`](https://github.com/vsitzmann/siren/blob/4df34baee3f0f9c8f351630992c1fe1f69114b5f/modules.py#L599-L611)

Upstream contract:

1. The first layer is
   `sin(first_omega0 * Linear(d, H)(x))`, with
   `weight ~ Uniform(-1/d, 1/d)`.
2. Every later sine layer is
   `sin(hidden_omega0 * Linear(H, H)(h))`, with
   `weight ~ Uniform(-sqrt(6/H)/hidden_omega0,
   sqrt(6/H)/hidden_omega0)`.
3. The output is linear. Its weight uses the same hidden-layer bound.
4. The default is `first_omega0=hidden_omega0=30`.
5. The custom initializer changes weights only; biases retain the PyTorch
   `nn.Linear` default.
6. Upstream `hidden_layers=N` means one first sine layer plus `N` additional
   sine layers. Local `depth` means the total number of sine hidden layers, so
   `local depth = upstream hidden_layers + 1`.

Old local difference: omega was folded into initialized weights, every bias was
reinitialized with the weight bound, and the output used a hard-coded `0.1`
factor. Initial forward phases were only partly equivalent and parameter
gradients were not.

Frozen local adaptation:

- `ScaledSin` keeps omega explicit in the graph.
- SIREN always uses the upstream default omegas; the problem-specific
  `omega0` argument belongs only to Complex Sinh.
- float64 tensor output replaces the upstream MetaModule wrapper. This is a
  numerical/interface adaptation, not an architecture change.
- the Taylor-jet sine rule multiplies every input jet coefficient by omega
  before applying the sine recurrence.

Executable evidence:
`tests/test_architecture_fidelity.py::test_siren_matches_upstream_layer_and_initialization_contract`
and the scaled-SIREN jet/gradient tests.

## Fourier-feature PINN

Pinned PINN source:

- repository: <https://github.com/PredictiveIntelligenceLab/MultiscalePINNs>
- commit: [`ba7d6bb8af6cabe348def80bed72110f5f0e3621`](https://github.com/PredictiveIntelligenceLab/MultiscalePINNs/commit/ba7d6bb8af6cabe348def80bed72110f5f0e3621)
- Poisson `NN_FF` / `NN_mFF`:
  [`Poisson1D/models_tf.py`](https://github.com/PredictiveIntelligenceLab/MultiscalePINNs/blob/ba7d6bb8af6cabe348def80bed72110f5f0e3621/Poisson1D/models_tf.py)
- heat space-time variants:
  [`heat1D/models_tf.py`](https://github.com/PredictiveIntelligenceLab/MultiscalePINNs/blob/ba7d6bb8af6cabe348def80bed72110f5f0e3621/heat1D/models_tf.py)
- wave mFF variants:
  [`wave1D/wave_models_tf.py`](https://github.com/PredictiveIntelligenceLab/MultiscalePINNs/blob/ba7d6bb8af6cabe348def80bed72110f5f0e3621/wave1D/wave_models_tf.py)

The plan's `bmild/fourfeat` URL is not a live repository. The Fourier Features
authors' code is:

- repository: <https://github.com/tancik/fourier-feature-networks>
- commit: [`9c110c31ce3794222fff408ac27bbf74d8fe8993`](https://github.com/tancik/fourier-feature-networks/commit/9c110c31ce3794222fff408ac27bbf74d8fe8993)
- mapping: [`Demo.ipynb`](https://github.com/tancik/fourier-feature-networks/blob/9c110c31ce3794222fff408ac27bbf74d8fe8993/Demo.ipynb)

These sources must not be conflated. The general coordinate-regression code
uses `[sin(2*pi*B*x), cos(2*pi*B*x)]` followed by a ReLU RGB network. The PINN
repository uses `[sin(B*xbar), cos(B*xbar)]`, tanh hidden layers, and normalized
coordinates.

Formal mFF-PINN contract:

1. Two frozen maps use angular-frequency scales `(1, sigma)`.
2. Each branch has `H/2` frequencies, hence an `H`-dimensional sin/cos map.
3. The branches share all trainable tanh trunk parameters.
4. `depth` counts trainable tanh layers; the frozen mapping is not hidden depth.
5. The final branch states are concatenated and passed to a linear output.
6. Trainable weights use Xavier-normal. Upstream normal biases are retained.
7. Sigma has no extra `2*pi`; it is defined after deterministic input
   standardization.

Old local difference: one frozen branch, `H` rather than `H/2` frequencies,
three trainable tanh layers when `depth=4`, no coordinate standardization, and
PyTorch default initialization. It was a single-scale RFF ablation, not the
MultiscalePINNs mFF architecture.

The local composite model exposes a Taylor-jet hook. It applies the same frozen
maps and shared trunk to each jet, concatenates corresponding coefficients, and
then applies the output layer. Tests compare both input derivatives and
trainable-parameter gradients with direct automatic differentiation.

## MscaleDNN

Pinned original source:

- repository: <https://github.com/xuzhiqin1990/mscalednn>
- commit: [`1c6c6f69e9ad586ccaea90a8e8fa0d07313460b2`](https://github.com/xuzhiqin1990/mscalednn/commit/1c6c6f69e9ad586ccaea90a8e8fa0d07313460b2)
- independent-subnet implementation:
  [`code/ritzM.py::neural_net`](https://github.com/xuzhiqin1990/mscalednn/blob/1c6c6f69e9ad586ccaea90a8e8fa0d07313460b2/code/ritzM.py)
- activations:
  [`code/my_act.py`](https://github.com/xuzhiqin1990/mscalednn/blob/1c6c6f69e9ad586ccaea90a8e8fa0d07313460b2/code/my_act.py)
- formula notes:
  [`notes/notes.tex`](https://github.com/xuzhiqin1990/mscalednn/blob/1c6c6f69e9ad586ccaea90a8e8fa0d07313460b2/notes/notes.tex)

Pinned PyTorch cross-check:

- repository: <https://github.com/Blue-Giant/MscaleDNN_torch>
- commit: [`b63796dd42a2020a0c2b241b1c824cc0405fad91`](https://github.com/Blue-Giant/MscaleDNN_torch/commit/b63796dd42a2020a0c2b241b1c824cc0405fad91)
- layers:
  [`Network/DNN_base.py`](https://github.com/Blue-Giant/MscaleDNN_torch/blob/b63796dd42a2020a0c2b241b1c824cc0405fad91/Network/DNN_base.py)

MscaleDNN-1 divides the first hidden layer of one wide network into scale
groups; later layers mix the groups. MscaleDNN-2 instead uses independent
subnets

`u(x) = sum_k F_k(a_k*x; theta_k)`.

The formal high-order-PINN adaptation is named **MscaleDNN-2-sin**:

1. scales are fixed to `(1, 2, 4)`, a set explicitly used in the paper;
2. scale multiplication is explicit in every forward pass and is not folded
   into a trainable first-layer weight;
3. the three subnets have independent parameters and their outputs are summed;
4. every subnet has `depth` sine hidden layers and a linear output;
5. weights and biases use the original Gaussian standard deviation
   `2/sqrt(fan_in+fan_out)`, correcting the upstream output-layer loop-variable
   bug;
6. no SIREN omega or SIREN initialization is used.

The original paper prefers compactly supported activations, but its runnable
`ritzM.py` includes a sine configuration. The compact sReLU/B-spline variants
are only C0/C1 and therefore cannot supply classical fourth- and sixth-order
PINN derivatives. The `-sin` suffix makes this necessary smooth adaptation
explicit.

Old local difference: scale was folded once into the first weight
initialization. That preserves an initial forward map under a parameter
relabeling, but does not preserve the optimization dynamics; the scale was no
longer fixed in the computational graph. It also used default Kaiming-uniform
initialization and gave every subnet the full baseline width.

The local jet hook multiplies all input jet coefficients by each fixed scale,
which gives the required factor `a_k**order` in derivatives. Tests verify the
forward decomposition, mixed derivatives, and parameter gradients.

## Complex Sinh

Complex Sinh is the paper method and has no external baseline repository. The
contract is:

1. complex128 trainable parameters;
2. four `Linear -> sinh` hidden layers and a linear scalar output;
3. first-layer real weights uniform in `[-1/d, 1/d]`;
4. first-layer imaginary weights uniform in
   `[-omega0/d, omega0/d]`;
5. first-layer imaginary bias uniform in `[-pi, pi]`;
6. exact Taylor-jet derivatives through the entire activation.

Its evidence chain is the paper formula, `complex_freq_init_`, backend
equivalence tests, complex parameter-gradient tests, and the architecture
fidelity test.

## Parameter-budget rule

The sole capacity reference is one native-complex Complex Sinh model with
`H=128`. A complex trainable scalar counts as two real degrees of freedom.
Frozen Fourier matrices do not count. For a complex target, each real baseline
uses two independent scalar networks and both count.

`formal_architecture_budgets` searches integer widths and rejects a table if a
method differs from the reference by more than 5%. Every result row records the
actual width, real DOF, representation, and completed optimizer steps. No
formal H=64 output is accepted.

## Training details not copied from upstream

The goal is architecture fidelity under one controlled PDE protocol, not a
mixture of each repository's optimizer and data pipeline. Upstream batching,
float32 choices, TensorFlow/JAX wrappers, dataset-specific losses, and stopping
rules are not copied. All four methods use the common jsc_v2 optimizer,
collocation, wall-clock, precision, and evaluation rules. These shared choices
are experimental controls and are not presented as upstream defaults.
