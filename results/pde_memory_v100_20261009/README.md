# V100 mean GPU memory for the three PDE examples

Each of the six cells uses the paper's network (four tanh hidden layers of
width 128), 400 resampled interior points, and 128 constraint points per
face in 1D or 256 in 2D. Paired methods share the initial parameters and
the samples in the two warm-up Adam updates. After warm-up, each process
repeats complete loss/backward/Adam updates for a 30-second measurement
window. These short runs supplement memory only; they do not replace the
completed equal-update accuracy/time runs.

The metric is the arithmetic mean of nvidia-smi process used_gpu_memory
samples, approximately every 0.1 seconds. It includes CUDA context and
cached allocator memory. Each cell has 256–260 valid samples. The cached
process memory is constant within each observed window, so its sample
standard deviation is zero. This is an absolute process-memory measure,
not a peak-tensor-memory measure or a five-seed training statistic.

| Case | Nested JVP (MiB) | WDD (MiB) |
| --- | ---: | ---: |
| 1D KdV | 578 | 618 |
| 2D KdV | 564 | 622 |
| Cahn–Hilliard | 752 | 666 |

Reproduce one cell on an allocated V100 from the repository root:

```sh
python experiments/torch_benchmarks/measure_pde_memory.py --case kdv1d \
  --method shared_jet_linear --seconds 30 --out /absolute/new/memory-kdv1d
```

Use all three cases and both methods for the six-cell comparison.
summary.csv and raw/*.csv retain the numeric measurements; provenance.json
records source hashes, settings, paired warm-up hashes and successful runs.
All six Slurm array tasks completed with exit 0:0. The transferred archive
was 25,167 bytes and matched its independently recorded remote SHA-256.
No model or optimizer weights were saved.
