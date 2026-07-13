#!/usr/bin/env bash
# Source this file to make python3.11 + torch (cu121 wheel) find NVIDIA shared libs
# AND select the CUDA visible device on the active experiment host.
#
# Usage:
#   source scripts/cuda_env.sh
#   python3.11 experiments/benchmark_single_monomial.py ...
#
# Path layout verified on the H20 host (2026-07-14):
#   - python3.11 binary:    /root/miniconda3/envs/apolarity/bin/python
#   - torch + nvidia wheels: /usr/local/lib/python3.11/site-packages/nvidia/<cmp>/lib
#   - GPU:                  NVIDIA H20 (CUDA 12.x driver, 12.1 cu wheel)
#
# This script is committed so any future tool / experiment in apolarity/ can
# rely on a single canonical entry point for the experiment environment.

_NV_BASE=/usr/local/lib/python3.11/site-packages/nvidia

export LD_LIBRARY_PATH="\
${_NV_BASE}/cudnn/lib:\
${_NV_BASE}/cuda_runtime/lib:\
${_NV_BASE}/cuda_nvrtc/lib:\
${_NV_BASE}/cuda_cupti/lib:\
${_NV_BASE}/cublas/lib:\
${_NV_BASE}/nccl/lib:\
${_NV_BASE}/cusparse/lib:\
${_NV_BASE}/cufft/lib:\
${_NV_BASE}/curand/lib:\
${_NV_BASE}/cusolver/lib:\
${_NV_BASE}/nvjitlink/lib:\
${_NV_BASE}/nvtx/lib:\
${LD_LIBRARY_PATH:-}"

unset _NV_BASE

# Default to the first visible CUDA device.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Convenience alias: prefer python3.11 as the project interpreter.
export APOLARITY_PYTHON="${APOLARITY_PYTHON:-/root/miniconda3/envs/apolarity/bin/python}"

# Make `python` resolve to python3.11 inside scripts that invoke `python`.
if ! command -v python >/dev/null 2>&1; then
  alias python="$APOLARITY_PYTHON"
fi
