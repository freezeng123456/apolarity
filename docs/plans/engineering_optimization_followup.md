# Engineering optimization follow-up

Date: 2026-05-14

## What was completed after the first optimization round

### 1. Merged Linear-layer GEMMs

Each Linear layer now concatenates the Taylor-order dimension into the batch dimension and uses one GEMM per layer instead of one GEMM per Taylor coefficient.

Effect:

- `aten::mm` calls dropped from 83 to 11 for the representative order-8, rank-27 case.
- `MmBackward0` calls dropped from 28 to 4.
- Backward benchmark improved substantially for both real polarization and complex Waring.

### 2. Tested custom Linear VJP

A custom Linear-jet VJP was implemented experimentally but did **not** improve runtime beyond the merged-GEMM native PyTorch version.  It was reverted.

Conclusion: PyTorch's native autograd for the merged large GEMM is already efficient enough.  Further Linear-level optimization should focus on lower-level fused kernels only if needed.

### 3. Mode-aware `auto`

The backend selection is now conservative:

- value mode: use complex Waring when it gives a direction-count advantage;
- backward/PINN mode: prefer real polarization for now, because complex autograd overhead remains significant even after custom activation VJP and merged GEMM.

## Current bottleneck

After custom activation VJP and merged Linear GEMM, the remaining bottleneck is mostly:

1. complex GEMM backward;
2. custom activation VJP elementwise convolutions;
3. complex tensor memory traffic.

Direction generation and weighted summation are negligible.

## Next plausible engineering directions

1. fused activation VJP kernels with Triton/CUDA;
2. real Waring formulas for repeated exponent patterns;
3. better cost model for mode-aware backend selection;
4. optional `torch.compile` experiments for value-mode fixed shapes.
